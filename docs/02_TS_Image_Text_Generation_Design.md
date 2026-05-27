# TS→Image + Text → Image → TS 创新设计方案

> 核心思想: 将时间序列转化为图像, 结合文本指导, 在图像空间生成新图像, 再转回时间序列

---

## 1. 创新动机与定位

### 1.1 现有方法的局限

| 方法 | 局限 |
|------|------|
| T2S (IJCAI 2025) | 文本→TS 直接生成, 无中间视觉表示, 文本信息被压缩 |
| VerbalTS (ICML 2025) | 文本→TS, 多焦点对齐但仍在原始时域操作 |
| LDM4TS (arXiv 2025) | TS→多视图图像→潜空间扩散→TS, 但无文本条件 |
| ImagenTime (arXiv 2024) | TS→STFT图像→扩散→TS, 无文本条件 |

### 1.2 我们的创新点

**Text+Image 双条件图像空间时间序列生成**:
1. TS → 多视图图像 (GAF + Spectrogram + RP)
2. Text + Image → Diffusion → 生成新图像
3. 生成图像 → 逆变换 → TS

**与 T2S/VerbalTS 的差异化**:
- T2S: Text → latent → TS (无视觉中间表示)
- VerbalTS: Text → DDPM → TS (无图像)
- **Ours**: Text + TS图像 → Image Diffusion → 新图像 → TS (视觉思维链)

---

## 2. 时间序列→图像编码方法

### 2.1 格拉姆角场 (Gramian Angular Field, GAF)

**原理**: 将时间序列 X = {x_1, ..., x_n} 映射到极坐标:
- 半径: r_t = t/n (归一化时间戳)
- 角度: phi_t = arccos(x_t) (值映射到角度)

然后计算格拉姆矩阵:
- **GASF** (Summation): G_ij = cos(phi_i + phi_j)
- **GADF** (Difference): G_ij = sin(phi_i - phi_j)

**双射性分析**:

| 变体 | 双射? | 逆变换 | 条件 |
|------|-------|--------|------|
| **GASF (值域 [0,1])** | **是** | x_t = sqrt((G_tt + 1) / 2) | 主对角线直接恢复 |
| GASF (值域 [-1,1]) | 近似 | 需要额外信息 | cos 歧义 |
| GADF | **否** | 不可精确逆变换 | 不同序列可产生相同图像 |

**关键发现**: 当时间序列归一化到 [0,1] 时, GASF 是双射的, 可以从主对角线精确恢复原始值。

**优点**: 保留时序关系, 捕获长程相关性 (off-diagonal entries)
**缺点**: O(T²) 复杂度, 长序列产生巨大图像 (T×T)
**适用**: 短-中等长度序列 (T ≤ 256)

### 2.2 频谱图 (Spectrogram / STFT)

**原理**: 短时傅里叶变换, 产生 时间×频率×能量 的 2D 表示

**双射性**: **是** (通过 iSTFT + Griffin-Lim / HiFi-GAN 可逆)
- 相位信息需要重建 (Griffin-Lim 迭代或神经网络 vocoder)
- 现代 vocoder (HiFi-GAN) 几乎无损

**优点**: O(T log T) 高效, 音频领域验证成熟, 多尺度频率信息
**缺点**: 时频分辨率 trade-off (Heisenberg 不确定性)
**适用**: 任意长度序列

### 2.3 连续小波变换 (CWT Scalogram)

**原理**: 多尺度小波变换, 产生 时间×尺度(频率) 的 2D 表示

**双射性**: **是** (通过 ICWT 逆变换)

**优点**: 多分辨率分析, 低频高时间分辨率 + 高频高频率分辨率
**缺点**: 计算开销略高于 STFT
**适用**: 任意长度, 特别适合非平稳信号

### 2.4 递归图 (Recurrence Plot, RP)

**原理**: RP_ij = Theta(epsilon - ||x_i - x_j||), 二值 T×T 矩阵

**双射性**: **否** (丢失幅度信息)

**用途**: 作为辅助通道, 编码周期性和状态转移信息

### 2.5 综合对比

| 方法 | 双射? | 复杂度 | 信息保留 | 生成适用性 |
|------|-------|--------|----------|-----------|
| **GASF [0,1]** | **是** | O(T²) | 高 | 好 (短序列) |
| **STFT** | **是** | O(T log T) | 高 | **优秀** |
| **CWT** | **是** | O(T log T) | 高 | **优秀** |
| RP | 否 | O(T²) | 结构 | 辅助通道 |
| MTF | 否 | O(T²) | 转移 | 差 |

---

## 3. 推荐方案: 三通道图像编码

### 3.1 编码设计

将时间序列编码为 **3 通道 RGB 图像**, 每个通道捕获不同特征:

```
时间序列 X (length T)
    ├── Channel R: GASF [0,1]  → 结构信息 (双射, 可精确恢复)
    ├── Channel G: STFT/Mel    → 频率信息 (双射, iSTFT 可逆)
    └── Channel B: RP (unthresholded) → 递归结构 (辅助, 不要求可逆)
    ↓
3-channel Image I ∈ R^{T×T×3}  (或 resized to fixed size)
```

**为什么用 3 通道?**
- GASF 保留精确的时序结构 (双射)
- STFT 保留频率信息 (双射)
- RP 保留递归/周期结构 (辅助)
- 三者互补, 类似 LDM4TS 的多视图策略
- 可以用预训练的图像扩散模型 (Stable Diffusion) 处理

### 3.2 逆变换设计

```
生成图像 I' ∈ R^{T×T×3}
    ├── R channel → GASF 逆变换: x_t = sqrt((G_tt + 1) / 2)
    ├── G channel → iSTFT 逆变换: 频率→时域
    └── 融合策略:
        - 主通道: R (GASF) 提供精确值
        - 辅助通道: G (STFT) 提供频率约束
        - 融合: x_final = alpha * x_gaf + (1-alpha) * x_stft
    ↓
时间序列 X' (length T)
```

**融合公式**:
```python
# GASF 恢复 (精确)
x_gaf = sqrt((generated_image[:, :, 0].diag() + 1) / 2)

# STFT 恢复
x_stft = istft(generated_image[:, :, 1])  # Griffin-Lim or HiFi-GAN

# 自适应融合 (可学习)
alpha = sigmoid(learnable_weight)
x_final = alpha * x_gaf + (1 - alpha) * x_stft
```

---

## 4. 整体架构: TIGER (Text+Image Guided Encoding for Recomposition)

### 4.1 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Phase                             │
│                                                               │
│  输入: (Text, Time Series)                                    │
│                                                               │
│  Step 1: TS → 3-channel Image                                │
│    TS → GASF (R) + STFT (G) + RP (B) → Image I_gt           │
│                                                               │
│  Step 2: Train Image Diffusion Model                         │
│    Text ──→ Text Encoder (T5/CLIP) ──→ c_text               │
│    I_gt ──→ Add noise → I_t                                  │
│    DiT(I_t, c_text) → predict noise → Loss                   │
│                                                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    Inference Phase                             │
│                                                               │
│  输入: Text                                                   │
│                                                               │
│  Step 1: Text → Image Diffusion → Generated Image I'         │
│    Text ──→ Text Encoder ──→ c_text                           │
│    DiT(noise, c_text) → I' (3-channel image)                 │
│                                                               │
│  Step 2: Image → Time Series                                 │
│    I' → GASF逆 (R) + STFT逆 (G) → 融合 → TS                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 详细架构设计

#### 模块 1: TS→Image Encoder

```python
class TSToImageEncoder:
    """将时间序列编码为3通道图像"""

    def __init__(self, image_size=64, use_gaf=True, use_stft=True, use_rp=True):
        self.image_size = image_size

    def encode(self, ts: torch.Tensor) -> torch.Tensor:
        """
        ts: (B, T) 或 (B, T, C)
        返回: (B, 3, H, W) 3通道图像
        """
        channels = []

        # Channel R: GASF [0,1] — 双射
        ts_norm = minmax_normalize(ts, to_range=(0, 1))
        phi = torch.arccos(ts_norm)
        gaf = torch.cos(phi.unsqueeze(-1) + phi.unsqueeze(-2))
        channels.append(resize(gaf, self.image_size))

        # Channel G: STFT — 双射
        stft = torch.stft(ts, n_fft=64, hop_length=8, return_complex=True)
        spec = torch.abs(stft)
        channels.append(resize(spec, self.image_size))

        # Channel B: Recurrence Plot — 辅助
        dist = torch.cdist(ts.unsqueeze(1), ts.unsqueeze(1))
        epsilon = dist.quantile(0.1)
        rp = (dist < epsilon).float()
        channels.append(resize(rp, self.image_size))

        return torch.stack(channels, dim=1)  # (B, 3, H, W)
```

#### 模块 2: Image Diffusion with Text Conditioning

```python
class TextImageDiffusion(nn.Module):
    """文本+图像条件扩散模型 (在图像空间操作)"""

    def __init__(self, image_size=64, text_dim=768, dit_dim=256):
        # 使用 SD VAE 将图像编码到潜空间
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae")
        # DiT denoiser (4 channels from VAE latent)
        self.dit = DiT(in_channels=4, dim=dit_dim)
        # Text encoder
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")

    def forward(self, image, text):
        # Encode image to latent
        z_image = self.vae.encode(image).latent_dist.sample()  # (B, 4, H/8, W/8)
        # Encode text
        c_text = self.text_encoder(text).last_hidden_state  # (B, seq_len, 768)
        # Flow Matching
        t = torch.rand(B)
        noise = torch.randn_like(z_image)
        z_t = (1 - t) * z_image + t * noise
        target = noise - z_image
        pred = self.dit(z_t, t, c_text)
        return F.mse_loss(pred, target)
```

#### 模块 3: Image→TS Decoder

```python
class ImageToTSDecoder:
    """将生成的图像逆变换回时间序列"""

    def decode(self, image: torch.Tensor, ts_length: int) -> torch.Tensor:
        """
        image: (B, 3, H, W) 生成的3通道图像
        返回: (B, T) 时间序列
        """
        # GASF 逆变换 (精确, 主通道)
        gaf_channel = image[:, 0]
        ts_gaf = torch.sqrt((gaf_channel.diagonal(dim1=-2, dim2=-1) + 1) / 2)

        # STFT 逆变换 (辅助)
        spec_channel = image[:, 1]
        ts_stft = torch.istft(spec_channel, n_fft=64, hop_length=8,
                              length=ts_length)

        # 自适应融合
        alpha = 0.7  # GASF 主导
        ts_final = alpha * resample(ts_gaf, ts_length) + (1 - alpha) * ts_stft

        return minmax_denormalize(ts_final, original_range)
```

### 4.3 与现有 MMLDM 的集成方案

**方案 A: 替代 Stage 2 (完全图像空间生成)**

```
当前: Text → VAE(z0) → DiT(FM) → z0' → VAE Decode → TS
新:   Text → Image Diffusion → 3-ch Image → GASF⁻¹/STFT⁻¹ → TS
```

**方案 B: 图像作为辅助条件 (IP-Adapter 风格)**

```
Text ──→ SBERT ──→ Text Emb ──┐
                                ├──→ DiT ──→ z0 → VAE Decode → TS
TS_img ──→ CLIP ──→ Img Emb ──┘
```

**方案 C: 两阶段生成 (Visual Blueprint) — 推荐**

```
Stage A: Text → Image Diffusion → 3-ch Image (GAF+STFT+RP)
Stage B: Text + Image → MMLDM DiT → z0 → VAE Decode → TS
```

---

## 5. 关键技术问题与解决方案

### 5.1 GAF 的 O(T²) 问题

| T | 图像大小 | 可行性 |
|---|----------|--------|
| 24 | 24×24 | 完全可行 |
| 96 | 96×96 | 可行 |
| 256 | 256×256 | 勉强可行 |
| 512 | 512×512 | 需要 Patch-GAF |

**解决方案**: Patch-GAF — 将长序列分段, 每段独立 GAF, 拼接为多帧

### 5.2 STFT 相位重建

| 方法 | 质量 | 是否需要训练 |
|------|------|-------------|
| Griffin-Lim | 中等 | 否 |
| HiFi-GAN | 高 | 是 (预训练) |
| 直接生成复数 STFT | 高 | 否 (但增加生成难度) |

### 5.3 多变量时间序列 (Weather 21 vars)

**推荐**: Heatmap 编码 — 将 21 个变量堆叠为 21×T 热力图

```
21 variables × T timesteps → (21, T) heatmap
每个变量独立 GASF → (21, T, T) → 拼接为 7×3 grid of T×T patches
```

### 5.4 与 T2S/VerbalTS 公平对比

| 设置 | 输入 | 对标 |
|------|------|------|
| **Fair** | Text only → Image → TS | T2S, VerbalTS |
| **Full** | Text + Ref_TS → Image → TS | Oracle (信息更多) |

---

## 6. 实验计划

### 6.1 数据集与指标

| 实验 | 数据集 | 指标 | 对标 |
|------|--------|------|------|
| 主实验 | TSFragment-600K (ETTh1) | MSE, WAPE, C-FID | T2S |
| 主实验 | Weather | CTTP, FID, JFTSD | VerbalTS |
| 消融 | ETTh1 | 各图像编码方法对比 | — |

### 6.2 消融实验

| 消融项 | 变体 |
|--------|------|
| 图像编码 | GAF-only vs STFT-only vs GAF+STFT+RP |
| 逆变换 | GASF-inverse vs STFT-inverse vs 融合 |
| 条件方式 | Text-only vs Image-only vs Text+Image |
| 融合策略 | alpha=0.7 (固定) vs learnable alpha |
| 图像尺寸 | 32×32 vs 64×64 vs 128×128 |

### 6.3 新增评估指标

| 指标 | 说明 |
|------|------|
| Image FID | 生成图像 vs 真实图像的质量 |
| PSNR / SSIM | 生成图像 vs 真实图像的像素级质量 |
| Round-trip Error | TS→Image→TS 的往返误差 (验证双射性) |

---

## 7. 相关工作

| 论文 | 会议 | 关键贡献 | 与我们的关系 |
|------|------|----------|-------------|
| **LDM4TS** | arXiv 2025 | TS→GAF+RP+SEG→Latent Diffusion→TS | 最接近, 但无文本条件 |
| **ImagenTime** | arXiv 2024 | TS→STFT→EDM→TS | STFT 方案验证 |
| **WaveletDiff** | arXiv 2025 | CWT→Diffusion→TS | CWT 方案验证 |
| **TimeOmni-VL** | arXiv 2026 | Bi-TSI 双向 TS↔Image | 双射转换验证 |
| **XIRP** | 2022 | 可逆 2D 表示→GAN→TS | 可逆表示验证 |
| **IP-Adapter** | arXiv 2023 | Decoupled cross-attention | 融合架构参考 |
| **GAF (Wang&Oates)** | 2015 | 格拉姆角场定义 | 基础编码方法 |

---

## 8. 总结

### 核心创新

1. **三通道图像编码**: GASF (双射结构) + STFT (双射频率) + RP (递归辅助)
2. **文本+图像双条件扩散**: 在图像空间同时接受文本和图像条件
3. **可逆图像→TS**: 利用 GASF 双射性和 STFT 逆变换, 保证信息无损
4. **视觉思维链**: Text → Image (规划) → TS (执行), 可解释性强

### 与 T2S/VerbalTS 的差异化

- T2S: Text → latent → TS (黑盒)
- VerbalTS: Text → DDPM → TS (黑盒)
- **TIGER**: Text → **可见的中间图像** → TS (可解释)

### 预期优势

1. 图像空间提供更丰富的结构先验 (2D 空间关系)
2. 可利用预训练图像扩散模型 (SD, DALL-E) 的知识
3. 中间图像可解释, 便于调试和论文展示
4. GASF 双射性保证信息无损转换
