# TIGER 框架系统性诊断与修复方案

> 基于多维度深度分析: 代码追踪 + 指标数学性质 + 架构逻辑 + 科研创新定位

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [问题 1: 指标异常 (WAPE/MRR@10)](#2-问题-1-指标异常)
3. [问题 2: cond_mode 逻辑不一致](#3-问题-2-cond_mode-逻辑不一致)
4. [根因深度剖析](#4-根因深度剖析)
5. [架构重设计方案](#5-架构重设计方案)
6. [科研创新重新定位](#6-科研创新重新定位)
7. [实施路线图](#7-实施路线图)
8. [验证实验设计](#8-验证实验设计)

---

## 1. 执行摘要

### 1.1 核心发现

TIGER 框架存在 **两个相互关联的根本性问题**:

| 问题 | 表面现象 | 根因 |
|------|----------|------|
| 指标异常 | WAPE/MRR@10 一开始就是 SOTA, 训练中几乎不变 | **条件泄露 (Conditioning Leakage)**: 训练和评估时, 模型通过条件路径直接访问目标图像 |
| 逻辑不一致 | image_only 和 text+image 模式在评估时使用了真实图像 | **任务定义错误**: 这些模式不是"生成"而是"重建", 违反了推理时只有文本的现实约束 |

**一句话总结**: 当前框架本质上是一个 **条件去噪自编码器 (Conditional Denoising Autoencoder)**, 而非真正的文本引导生成模型。模型通过条件编码器"看到"了目标图像, 因此不需要真正学习从文本到时间序列的映射。

### 1.2 信息泄露路径

```
训练时:
  真实图像 ──→ Image Encoder ──→ 条件嵌入 ──→ DiT (每一步去噪都能看到目标)
      │
      └──→ 加噪 ──→ DiT ──→ 预测噪声 ──→ MSE Loss

评估时:
  真实图像 ──→ Image Encoder ──→ 条件嵌入 ──→ DiT (迭代去噪)
      │
      └── (作为参考, 模型通过条件路径获取目标信息)
  随机噪声 ──→ DiT ──→ "生成"图像 (实际是条件重建)
```

### 1.3 设计文档 vs 实际实现的差距

设计文档 (`docs/02_TS_Image_Text_Generation_Design.md`) 正确定义了:

```
训练: Text + TS图像 → Image Diffusion → 预测噪声 (训练时用图像做条件是合理的)
推理: Text → Image Diffusion → 生成图像 → TS (推理时只能用文本)
```

但实际代码在推理时 **仍然传入真实图像**:

```python
# generator.py:201-202 — generate() 的签名
def generate(self, images, texts, n_samples=1, ...):
    # images 参数在 text+image 和 image_only 模式下被用作条件输入!
```

```python
# train.py:541 — 评估时传入真实图像
gen_imgs = self.model.generate(images, texts, n_samples=1)
#                              ^^^^^^ 这就是真实验证集图像!
```

---

## 2. 问题 1: 指标异常

### 2.1 现象描述

| 指标 | 训练初期 | 训练后期 | 变化趋势 |
|------|----------|----------|----------|
| MSE | 较低 | 更低 | 持续下降 |
| MAPE | 中等 | 略低 | 缓慢下降 |
| WAPE | **已经很低** | 几乎不变 | **平坦** |
| MRR@10 | **已经很高** | 几乎不变 | **平坦** |

### 2.2 根因: 条件泄露的数据流追踪

#### 数据集层: 图像与时间序列同源

```python
# dataset.py:128-146 — _precompute_images()
def _precompute_images(self):
    ts_tensor = torch.tensor(self.ts_data[:self.n_samples], dtype=torch.float32)
    # ...
    ts_norm = (ts_tensor - ts_min) / ts_range
    self.images, self.norm_params = self.ts_to_image.encode(ts_norm)
    self.ts_norm = ts_norm   # ← 归一化后的时间序列
    self.ts_min = ts_min
    self.ts_max = ts_max
```

```python
# dataset.py:151-168 — __getitem__()
def __getitem__(self, idx):
    sample = {
        "image": self.images[idx],    # ← 从 ts_data[idx] 生成的图像
        "ts": self.ts_norm[idx],      # ← 同一个 ts_data[idx] 的归一化版本
        # ...
    }
```

**关键**: `"image"` 和 `"ts"` 来自 **同一个时间序列**。图像是时间序列的编码表示。

#### 训练层: 条件 = 目标

```python
# generator.py:172-189
def forward(self, batch, is_train=True):
    images = batch["image"]      # ← 真实目标图像 (B, 3, 64, 64)
    texts = batch.get("cap")

    if is_train:
        t = torch.randint(0, self.num_steps, [B])
        attr_emb = self.compute_condition(images, texts, t)  # ← 用真实图像做条件
        loss = self._noise_estimation_loss(images, attr_emb, t)  # ← 目标也是同一张图像
```

**关键问题**: `compute_condition` 的输入 `images` 和 `_noise_estimation_loss` 的目标 `x` 是 **同一张图像**。

#### compute_condition 中的信息流

```python
# generator.py:137-161
def compute_condition(self, images, texts, diffusion_step):
    cond_mode = self.config["condition"].get("cond_mode", "text+image")

    if cond_mode == "text+image":
        image_emb = self.encode_image(images)     # ← 图像编码器看到真实图像
        text_emb = self.encode_text(texts)
        attr_emb = self.cond_projector(text_emb, image_emb, diffusion_step)
        #                                    ↑
        #                    image_emb 携带了目标图像的完整信息
```

#### 评估层: 真实图像作为生成条件

```python
# train.py:534-541 — _compute_gen_metrics
for batch in self.val_loader:
    images = batch["image"].to(self.device).float()   # ← 真实验证集图像
    texts = batch.get("cap", None)

    gen_imgs = self.model.generate(images, texts, n_samples=1)
    #                              ^^^^^^
    #  在 text+image 和 image_only 模式下, 这些真实图像被用作条件!
```

```python
# evaluator.py:119-121 — 同样的问题
gen_images = self.model.generate(
    images, texts, n_samples=self.n_samples, sampler=self.sampler
)
```

### 2.3 各指标的数学解释

#### 为什么 WAPE 一开始就很低?

**WAPE 的数学性质**:
```
WAPE = Σ|real - gen| / Σ|real|
```

- WAPE 是 **相对误差**, 分母是真实值的绝对值之和
- 一旦模型通过条件信号获得了正确的幅度范围和大致形状, WAPE 就已经很低
- 条件编码器通过 `image_emb` 直接传递了目标图像的幅度信息
- 即使生成的图像略有偏差, 经过 GASF 解码和反归一化后, 输出的幅度分布已接近真实值

#### 为什么 MRR@10 一开始就很低?

**MRR@10 的数学性质**:
```
cosine_sim(real, gen) = (real · gen) / (||real|| × ||gen||)
```

- 余弦相似度 **完全忽略幅度, 只衡量形状/方向**
- 条件信号已经传递了时间序列的形状/趋势
- 即使绝对值有偏差, 形状相似度 (余弦相似度) 已经很高
- `threshold = 0.5` 是一个很低的阈值, 粗略重建就能超过

#### 为什么 MSE 持续下降?

**MSE 的数学性质**:
```
MSE = (1/BT) × Σ(real - gen)²
```

- MSE 是 **像素级指标**, 对幅度误差敏感
- 模型在逐步优化条件编码和噪声估计的精度
- 类比: 一张模糊但大致正确的照片, 随训练进行逐渐变清晰 -- MSE 反映"清晰度"

#### 为什么 MAPE 缓慢下降?

```
MAPE = (1/BT) × Σ|real_i - gen_i| / max(|real_i|, eps)
```

- 逐元素相对误差, eps = 1e-3 防止除零
- 介于 MSE 和 WAPE 之间
- 小值区域的相对误差在逐步改善

### 2.4 指标分裂总结

| 指标 | 衡量什么 | 为什么一开始就低 | 为什么还在下降 |
|------|----------|-----------------|---------------|
| MSE | 像素精度 | 条件信号提供大致正确值 | 像素级精细化 |
| WAPE | 幅度相对误差 | 条件信号传递幅度范围 | 幅度已饱和 |
| MRR@10 | 形状相似度 | 条件信号传递形状结构 | 形状已饱和 |
| MAPE | 逐元素相对误差 | 介于两者之间 | 部分元素在改善 |

---

## 3. 问题 2: cond_mode 逻辑不一致

### 3.1 三种模式的任务定义

| cond_mode | 训练时条件 | 推理时条件 | 实际任务 | 是否合理? |
|-----------|-----------|-----------|----------|----------|
| `image_only` | 真实图像 | **真实图像** | 图像→图像 (自编码) | **否**: 推理时没有目标图像 |
| `text_only` | 文本 | 文本 | 文本→图像→TS | **是**: 符合生成任务定义 |
| `text+image` | 文本+真实图像 | **真实图像** | 图像修复/翻译 | **部分**: 训练合理, 推理不合理 |

### 3.2 image_only 模式的问题

```python
# generator.py:155-157
elif cond_mode == "image_only":
    image_emb = self.encode_image(images)  # ← 推理时需要真实图像!
    attr_emb = self.cond_projector(image_emb, diffusion_step)
```

**问题**: 推理时没有目标图像, 这个模式完全无法用于生成任务。它本质上是一个 **图像自编码器** (给定图像, 生成类似的图像)。

**代码追踪**:
```
训练: compute_condition(images, texts, t) → encode_image(images) → ImageOnlyProjector
推理: generate(images, texts) → compute_condition(images, texts, t) → encode_image(images)
                                       ↑
                              推理时需要真实图像, 但此时不应该有!
```

### 3.3 text+image 模式的问题

```python
# generator.py:141-148
if cond_mode == "text+image":
    image_emb = self.encode_image(images)  # ← 推理时需要真实图像!
    text_emb = self.encode_text(texts)
    attr_emb = self.cond_projector(text_emb, image_emb, diffusion_step)
```

**问题**: 虽然设计上是"文本+图像"双条件, 但推理时只能用文本。当前实现:

1. 训练时: 模型学习利用图像条件 (因为图像条件信息量最大)
2. 推理时: 如果去掉图像条件, 模型性能会大幅下降 (因为模型依赖图像条件)

**门控融合的问题**:

```python
# cond_projector.py:207-210
gate = torch.sigmoid(self.gate_linear(torch.cat([text_proj, image_proj], dim=-1)))
out = gate * text_proj + (1.0 - gate) * image_proj
```

门控机制会自动学习给图像分支更高的权重 (因为图像信息量更大), 导致文本分支的学习被抑制。这是一个 **条件依赖陷阱**。

### 3.4 text_only 模式: 唯一正确的选择

```python
# generator.py:149-154
elif cond_mode == "text_only":
    if texts is not None:
        text_emb = self.encode_text(texts)
    else:
        text_emb = self.encode_text([""] * (images.shape[0] if images is not None else 1))
    attr_emb = self.cond_projector(text_emb, diffusion_step)
```

**这是唯一符合生成任务定义的模式**: 推理时只有文本可用。

**详细追踪 (text_only 模式)**:
```
训练:
  compute_condition(images, texts, t)
    → encode_text(texts)           # 只用文本
    → TextOnlyProjector(text_emb, diffusion_step)
    → attr_emb

推理:
  generate(images, texts)
    → compute_condition(images, texts, t)
        → encode_text(texts)       # 只用文本 (images 不参与条件计算)
        → TextOnlyProjector(text_emb, diffusion_step)
        → attr_emb

images 参数的唯一用途:
  B = images.shape[0]              # 批大小 (可替代)
  sample_shape = images.shape      # 输出形状 (可替代)
```

**结论**: text_only 模式没有信息泄露。但 images 参数仍被要求传入 (用于确定 batch size), 设计上不够干净。

### 3.5 代码级泄露点汇总

| 泄露点 | 文件 | 行号 | 模式 | 严重程度 |
|--------|------|------|------|----------|
| 训练 forward | `generator.py` | 187 | text+image, image_only | 高 (训练时条件=目标) |
| 评估 gen_metrics | `train.py` | 541 | text+image, image_only | **致命** (推理时用真实图像) |
| 评估 MRR | `train.py` | 576 | text+image, image_only | **致命** |
| 评估器 | `evaluator.py` | 119 | text+image, image_only | **致命** |

---

## 4. 根因深度剖析

### 4.1 信息泄露的两条路径

**路径 1: 直接泄露 (image_only, text+image 模式)**
```
真实图像 → Image Encoder → image_emb → cond_projector → attr_emb → DiT
                                                               ↑
                                            每一步去噪都能看到这个信息
```

**路径 2: 间接泄露 (text+image 模式, 通过门控融合)**
```
text_proj ←── text_emb ←── encode_text(texts)
    ↓
gate * text_proj + (1-gate) * image_proj ←── image_emb ←── encode_image(真实图像)
    ↓                                                        ↑
attr_emb → DiT                                    门控自动给图像更高权重
```

### 4.2 为什么训练 loss 在下降但指标不改善?

训练 loss 是 **噪声预测误差**:
```
L = ||ε - ε_θ(x_t, t, c)||²
```

这个 loss 衡量的是"模型能否预测加噪图像中的噪声", 而不是"模型能否从文本生成好的图像"。

即使条件信号已经提供了足够的信息, 模型仍然需要学习精确的噪声预测, 所以 loss 在下降。但生成质量 (WAPE/MRR) 已经被条件信号"锁死"在高位。

### 4.3 与 T2S/VerbalTS 的关键对比

| 框架 | 训练条件 | 推理条件 | 是否有泄露? |
|------|----------|----------|------------|
| T2S | 文本 embedding | 文本 embedding | **无** (训练和推理条件一致) |
| VerbalTS | 文本 (LongCLIP) | 文本 (LongCLIP) | **无** |
| TIGER (当前) | 文本 + **真实图像** | 文本 + **真实图像** | **有** (推理时用真实图像) |
| TIGER (修复后) | 文本 + 真实图像 (训练) | 仅文本 (推理) | 需要架构调整 |

T2S 和 VerbalTS 的共同特点: **训练和推理使用相同的条件输入 (只有文本)**。TIGER 的问题在于训练和推理的条件输入不一致。

### 4.4 CSV 数据集的额外问题

```python
# dataset.py:109-113
# Train/test split: 99% train, 1% test (same as T2S)
n = len(ts_data)
rng = np.random.RandomState(123)       # T2S 用 seed=2023
perm = rng.permutation(n)
n_train = int(np.ceil(n * 0.99))
```

问题:
1. **没有验证集**: 只有 train/test, 没有 valid split 用于超参调优
2. **随机种子不同**: T2S 用 seed=2023, TIGER 用 seed=123, 数据划分不同, 指标不可直接比较
3. **99/1 划分**: 测试集只有 1%, 样本量太小, 指标波动大

---

## 5. 架构重设计方案

### 5.1 核心原则

**推理时只能用文本, 训练时需要设计合理的条件机制。**

```
┌─────────────────────────────────────────────────────────────────┐
│                     重设计后的架构                                 │
│                                                                  │
│  训练阶段:                                                        │
│    Text ──→ CLIP ──→ text_emb ──→ TextProjector ──→ attr_emb    │
│    Image ──→ 加噪 ──→ DiT(attr_emb) ──→ 预测噪声 ──→ Loss       │
│                                                                  │
│  推理阶段:                                                        │
│    Text ──→ CLIP ──→ text_emb ──→ TextProjector ──→ attr_emb    │
│    Noise ──→ DiT(attr_emb) ──→ 生成图像 ──→ GASF⁻¹/STFT⁻¹ ──→ TS│
│                                                                  │
│  关键: 推理时 attr_emb 只从文本计算, 不需要图像!                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 具体修改方案

#### 修改 1: 统一使用 text_only 模式

**文件**: `mmldm/tiger/train.py` — `get_default_config()`

```python
# 当前 (line 83):
"cond_mode": "text+image"

# 修改为:
"cond_mode": "text_only"
```

#### 修改 2: 重构 generate() 方法

**文件**: `mmldm/tiger/generator.py` — `generate()` (line 201)

```python
@torch.no_grad()
def generate(self, texts, n_samples=1, image_shape=None,
             sampler="ddim", guidance_scale: float = 1.0):
    """Generate images conditioned on text only.

    Args:
        texts: list of text descriptions (or None for unconditional)
        n_samples: number of samples per text
        image_shape: (B, C, H, W) tuple for shape inference
        sampler: "ddim" or "ddpm"
        guidance_scale: CFG guidance weight
    """
    if image_shape is None:
        image_shape = (1, 3, 64, 64)  # default

    B = image_shape[0]
    sample_shape = image_shape

    samples = []
    for i in range(n_samples):
        x = torch.randn(sample_shape, device=self.device)
        for step in range(self.num_steps - 1, -1, -1):
            noise = torch.randn_like(x)
            t = torch.full((B,), step, device=self.device, dtype=torch.long)

            # 只用文本条件
            attr_emb = self._compute_text_condition(texts, t)
            pred_noise = self.dit(x, t, attr_emb)

            use_cfg = guidance_scale > 1.0 and texts is not None
            if use_cfg:
                attr_uncond = self._compute_text_condition(None, t)
                pred_uncond = self.dit(x, t, attr_uncond)
                pred_noise = pred_uncond + guidance_scale * (pred_noise - pred_uncond)

            if sampler == "ddpm":
                x = self.ddpm.reverse(x, pred_noise, t, noise)
            else:
                x = self.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
        samples.append(x)

    return torch.stack(samples)
```

#### 修改 3: 添加 _compute_text_condition 方法

**文件**: `mmldm/tiger/generator.py`

```python
def _compute_text_condition(self, texts, diffusion_step):
    """Compute condition from text only (no image needed)."""
    B = 1 if texts is None else len(texts)
    if texts is not None:
        text_emb = self.encode_text(texts)
    else:
        text_emb = self.encode_text([""] * B)

    cond_mode = self.config["condition"].get("cond_mode", "text_only")
    if cond_mode == "text+image":
        # 需要图像但没有 — 回退到纯文本路径
        if hasattr(self.cond_projector, 'text_projector'):
            attr_emb = self.cond_projector.text_projector(text_emb, diffusion_step)
            return attr_emb.permute(0, 3, 1, 2)
    return self.cond_projector(text_emb, diffusion_step)
```

#### 修改 4: 修复评估流程

**文件**: `mmldm/tiger/train.py` — `_compute_gen_metrics` (line 527)

```python
@torch.no_grad()
def _compute_gen_metrics(self, epoch: int, do_mrr: bool = False):
    self.model.eval()
    dc = self.config["data"]
    ts_len = dc.get("time_interval", 24)

    all_real, all_gen = [], []
    for batch in self.val_loader:
        texts = batch.get("cap", None)
        ts_real = batch["ts"].to(self.device).float()
        ts_min = _squeeze_trailing_singletons(batch["ts_min"].to(self.device).float())
        ts_max = _squeeze_trailing_singletons(batch["ts_max"].to(self.device).float())

        # 关键修改: 不传 images, 只用 texts
        image_shape = (len(texts), 3, dc["image_size"], dc["image_size"])
        gen_imgs = self.model.generate(texts=texts, n_samples=1, image_shape=image_shape)
        gen_img = gen_imgs[0]
        norm_params = NormParams(min_val=ts_min, max_val=ts_max, n_vars=1, original_length=ts_len)
        gen_ts = self.decoder.decode(gen_img, ts_len, norm_params)
        real_ts = denormalize_ts_batch(ts_real, ts_min, ts_max)
        # ... 后续逻辑不变
```

### 5.3 训练时的条件机制选择

#### 选项 A: 纯文本条件 (最简单, 最公平) — 推荐

```
训练: Text → CLIP → text_emb → TextProjector → attr_emb → DiT
推理: Text → CLIP → text_emb → TextProjector → attr_emb → DiT
```

优点:
- 训练和推理完全一致, 无泄露
- 与 T2S/VerbalTS 公平对比
- 实现简单

缺点:
- 文本信息量有限, 生成质量可能不如图像条件

#### 选项 B: 训练时用图像+文本, 推理时用文本 (需要小心设计)

```
训练: Text + Image → ImageTextProjector → attr_emb → DiT (CFG dropout 0.5-0.7)
推理: Text only → TextProjector → attr_emb → DiT
```

需要:
1. 训练时使用高 CFG dropout (0.5-0.7, 当前 0.3 太低)
2. 推理时切换到 text_only 路径
3. 确保 TextProjector 和 ImageTextProjector 的文本路径共享权重

**风险**: 模型可能过度依赖图像条件, 文本路径学习不足。

#### 选项 C: 渐进式条件训练 (推荐用于最佳效果)

```
Phase 1 (预训练): 纯文本条件 (text_only) — 建立文本→图像的基础映射
Phase 2 (微调):   文本+图像条件 (text+image) + 高 CFG dropout (0.5-0.7)
推理: 纯文本条件
```

这确保模型首先学会从文本生成, 然后可以利用图像条件进一步提升。

### 5.4 评估指标的修正

#### MRR@10 阈值调整

当前 `threshold = 0.5` 过低, 建议提高到 0.7:

```python
# train.py:259
def calc_mrr(real, gen_samples, k=10, threshold=0.7):  # 从 0.5 改为 0.7
```

#### 数据划分修正

```python
# 使用与 T2S 一致的划分
rng = np.random.RandomState(2023)  # T2S 的 seed
n_train = int(np.ceil(n * 0.99))
n_valid = int(np.ceil(n * 0.005))  # 添加验证集
n_test = n - n_train - n_valid
```

---

## 6. 科研创新重新定位

### 6.1 当前定位的问题

当前论文将 TIGER 定位为 "Text+Image Guided Generation", 但:
1. 推理时只能用文本 (图像是不可用的)
2. image_only 和 text+image 模式在推理时不合理
3. 如果评估时用真实图像, 那就不是"生成"而是"重建"

### 6.2 推荐定位: 文本引导的图像空间时间序列生成

**核心创新**:
- 将时间序列编码为三通道图像 (GASF + STFT + RP)
- 在图像空间进行文本引导的扩散生成
- 利用 GASF 双射性实现无损逆变换
- **视觉思维链**: Text → 可见的中间图像 → TS (可解释)

**与 T2S/VerbalTS 的差异化**:

| 方法 | 生成路径 | 中间表示 | 可解释性 |
|------|----------|----------|----------|
| T2S | Text → latent → TS | 黑盒 latent | 低 |
| VerbalTS | Text → DDPM → TS | 无中间表示 | 低 |
| **TIGER** | Text → **图像** → TS | **可见的 GASF/STFT/RP 图像** | **高** |

**论文标题建议**:
> "TIGER: Text-guided Image-space Generation for Explainable Time Series Synthesis"

### 6.3 消融实验设计

| 消融项 | 变体 | 目的 |
|--------|------|------|
| 条件模式 | text_only vs image_only vs text+image | 验证文本条件的有效性 |
| 图像编码 | GASF-only vs STFT-only vs GASF+STFT+RP | 验证三通道设计 |
| 解码模式 | gasf vs fused | 验证融合策略 |
| CFG scale | 1.0, 2.0, 5.0, 7.0 | 验证 CFG 的效果 |
| CFG dropout | 0.0, 0.3, 0.5, 0.7 | 验证 dropout 对泛化的影响 |

### 6.4 论文写作建议

1. **不要回避条件泄露问题** — 审稿人很可能会发现。建议在论文中明确讨论:
   - "训练时使用图像条件加速学习"
   - "推理时仅使用文本条件"
   - "通过 CFG dropout 确保文本条件的有效性"

2. **强调视觉思维链的可解释性**:
   - 展示中间生成的图像 (GASF + STFT + RP)
   - 与 T2S/VerbalTS 的黑盒生成对比
   - 中间图像可以帮助理解模型的生成过程

3. **诚实报告指标**:
   - 使用 text_only 模式进行公平对比
   - 不要用 image 条件的指标作为主结果
   - 可以将 image 条件作为 "oracle" 上界参考

---

## 7. 实施路线图

### Phase 1: 紧急修复 (1-2 天)

| 任务 | 文件 | 修改内容 |
|------|------|----------|
| 1.1 | `train.py` | `_compute_gen_metrics` 中不传 images |
| 1.2 | `generator.py` | `generate()` 支持 text-only 推理 |
| 1.3 | `train.py` | 默认 `cond_mode` 改为 `"text_only"` |
| 1.4 | `train.py` | MRR@10 阈值从 0.5 提高到 0.7 |

### Phase 2: 架构优化 (3-5 天)

| 任务 | 文件 | 修改内容 |
|------|------|----------|
| 2.1 | `generator.py` | 添加 `_compute_text_condition` 方法 |
| 2.2 | `generator.py` | 重构 `generate()` 签名, images 可选 |
| 2.3 | `evaluator.py` | 评估流程不使用真实图像 |
| 2.4 | `dataset.py` | 添加验证集, 使用 T2S 的 seed |
| 2.5 | `cond_projector.py` | 确保 TextProjector 独立可用 |

### Phase 3: 实验验证 (5-7 天)

| 任务 | 内容 |
|------|------|
| 3.1 | 用 text_only 模式重新训练 |
| 3.2 | 对比 text_only vs text+image 的指标 |
| 3.3 | 调优 CFG scale 和 dropout |
| 3.4 | 在 T2S 数据集上对比 |
| 3.5 | 在 VerbalTS Weather 数据集上对比 |

### Phase 4: 论文完善 (持续)

| 任务 | 内容 |
|------|------|
| 4.1 | 更新实验结果 (text_only 模式) |
| 4.2 | 添加消融实验 |
| 4.3 | 讨论条件机制的设计选择 |
| 4.4 | 展示中间图像的可解释性 |

---

## 8. 验证实验设计

### 实验 1: 条件泄露验证

**目的**: 证实条件泄露是指标异常的根因

**方法**:
```python
# 在 forward() 中, 将条件图像替换为随机图像
shuffled_idx = torch.randperm(B)
attr_emb = self.compute_condition(images[shuffled_idx], texts, t)
loss = self._noise_estimation_loss(images, attr_emb, t)
```

**预期结果**: 如果条件泄露是根因, 随机化后 WAPE 和 MRR@10 将从一开始就很差, 并随训练逐步改善。

### 实验 2: text_only 模式训练

**目的**: 验证纯文本条件的可行性

**方法**:
```python
config["condition"]["cond_mode"] = "text_only"
```

**预期结果**:
- 所有指标都从较差的起点开始
- 随训练逐步改善
- 四个指标的变化趋势更加一致

### 实验 3: CFG scale 影响

**目的**: 找到最优的 CFG scale

**方法**: 在 text_only 模式下, 测试不同的 guidance_scale: [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

**预期结果**: 存在一个最优的 CFG scale, 平衡生成质量和多样性。

### 实验 4: MRR@10 阈值敏感性

**目的**: 验证阈值选择对 MRR 的影响

**方法**: 测试不同的 threshold: [0.3, 0.5, 0.7, 0.8, 0.9]

**预期结果**: 高阈值下 MRR 更能区分模型质量。

### 实验 5: 与 T2S 公平对比

**目的**: 在相同条件下对比 T2S

**方法**:
- 使用相同的数据集 (ETTh1, 99/1 split, seed=2023)
- 使用相同的指标 (MSE, WAPE, MRR@10)
- 使用 text_only 模式

**预期结果**: 在公平条件下, TIGER 应该能展示出图像空间表示的优势。

---

## 附录 A: 关键代码位置索引

| 问题 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 条件泄露 (训练) | `generator.py` | 187 | `compute_condition(images, texts, t)` |
| 条件泄露 (评估) | `train.py` | 541 | `generate(images, texts, ...)` |
| 条件泄露 (评估器) | `evaluator.py` | 119 | `generate(images, texts, ...)` |
| 图像条件使用 | `generator.py` | 142 | `encode_image(images)` |
| 门控融合 | `cond_projector.py` | 207 | `gate * text_proj + (1-gate) * image_proj` |
| MRR 阈值 | `train.py` | 259 | `threshold=0.5` |
| 数据划分 | `dataset.py` | 113 | `n_train = int(np.ceil(n * 0.99))` |
| 数据 seed | `dataset.py` | 111 | `np.random.RandomState(123)` |

## 附录 B: 设计文档与实现的差距

| 设计文档描述 | 实际实现 | 差距 |
|-------------|----------|------|
| 推理时只用文本 | `generate()` 需要 images 参数 | images 在 image/text+image 模式下被用作条件 |
| 文本+图像双条件 | 门控融合, 图像权重更高 | 文本路径学习被抑制 |
| 三通道图像编码 | 已实现 (GASF+STFT+RP) | 无差距 |
| GASF 双射性 | 已实现 (对角线提取) | 无差距 |
| CFG dropout | 已实现 (0.3) | 可能需要调高到 0.5-0.7 |

## 附录 C: 与已有分析文档的关系

本文档是对 `METRIC_ANALYSIS.md` 的系统性扩展:

| 维度 | METRIC_ANALYSIS.md | 本文档 |
|------|-------------------|--------|
| 指标分析 | 有 (数学性质) | 有 (扩展 + 数据流追踪) |
| 条件泄露 | 有 (基础诊断) | 有 (深度剖析 + 代码定位) |
| cond_mode 分析 | 无 | **新增** (三种模式的逻辑分析) |
| 架构修复 | 无 | **新增** (具体代码修改方案) |
| 科研定位 | 有 (基础建议) | **扩展** (重新定位 + 消融设计) |
| 实施路线 | 无 | **新增** (4 阶段路线图) |
| 验证实验 | 有 (基础思路) | **扩展** (5 个具体实验) |
