# MMLDM PR v3: 灵感驱动的创新改进

**分支:** `feature/mmv2-spectral-dual-latent`
**日期:** 2026-05-24
**前置:** PR v2 (TGFM + OneCycleLR + CFG scale 7.0)

---

## 一、调研来源

对 8 个顶会时间序列代码仓库进行调研，提取可借鉴技术：

| 仓库 | 类型 | 关键技术 | 可借鉴程度 |
|---|---|---|---|
| CaTSG (Microsoft) | 多环境 TS 生成 | Environment Bank + BAG 指导 + SwAV 对比学习 | ⭐⭐⭐ |
| VerbalTS | 文本驱动 TS 生成 | 多 Patch + CLIP + 三轴 cross-attention + 梯度引导 | ⭐⭐⭐ |
| DiMTS | TS 扩散 | Mamba 骨干 + Fourier Loss + Langevin Inpainting | ⭐⭐ |
| CPiRi | TS 流匹配 | adaLN + 频率加权损失 + diffusion_batch_mul + RevIN | ⭐⭐ |
| PaD-TS | TS 扩散 | 双轴 Transformer + MMD Loss + adaLN-Zero | ⭐ |
| TimeDART | TS 自监督 | SOS Token + Per-patch timestep + revIN | ⭐ |
| ImagenFew | TS→图像→扩散 | STFT 2D 表示 + EDM 预处理 | ⭐ |
| proximal_w_text | 因果推断 | 不适用 | ❌ |

---

## 二、创新点清单

### 创新 1: Multi-View Text Conditioning (MVTC)

**灵感来源:** CaTSG Environment Bank + VerbalTS 多轴分解
**原创性分析:**
- CaTSG 用数据驱动的环境银行 (K 个潜在环境向量 + SwAV 训练)
- VerbalTS 用 variable/scale/stage 三轴分解 cross-attention
- 我们用文本驱动的多视图投影 (K 个正交线性投影提取同一文本的不同语义方面)
- **关键区别:** 环境是数据层面的，视图是语义层面的；不需要额外分类器

**设计:**
```python
class MultiViewTextPooler(nn.Module):
    """从原始 SBERT 128-dim 生成 K 个不同文本视图。

    正交初始化确保每个视图关注文本的不同语义方面：
    - 视图 0 可能关注趋势描述
    - 视图 1 可能关注幅度描述
    - 视图 2 可能关注周期描述
    - 视图 3 可能关注整体语义

    类比 CaTSG 的 Environment Bank，但驱动信号来自文本而非数据分布。
    """
    def __init__(self, text_dim=128, latent_dim=64, n_views=4):
        super().__init__()
        self.n_views = n_views
        self.views = nn.ModuleList([
            nn.Linear(text_dim, latent_dim) for _ in range(n_views)
        ])
        # 正交初始化确保视图多样性
        for v in self.views:
            nn.init.orthogonal_(v.weight)
            nn.init.zeros_(v.bias)

    def forward(self, text_emb):  # (B, text_dim)
        return [v(text_emb) for v in self.views]  # K × (B, latent_dim)
```

**使用方式:**
```python
# 在 MMLDMDiTModel 中
self.text_pooler = MultiViewTextPooler(
    text_dim=128, latent_dim=config.txt_dim, n_views=4
)

# forward 中
text_views = self.text_pooler(text_raw)  # K × (B, D)
for i, block in enumerate(self.blocks):
    view_i = text_views[i % len(text_views)]
    ts_tokens, text_tokens = block(..., text_latent=view_i)
```

**扩展 — View-Averaged Guidance (类 BAG):**
```python
# 推理时：对 K 个视图的预测取平均，替代单一 text_latent
if guidance_scale > 1.0:
    v_views = []
    for view in text_views:
        v_view = dit(ts=z_t, text=text_latent, ..., text_latent=view)
        v_views.append(v_view.ts_sample)
    v_cond = torch.stack(v_views).mean(dim=0)  # 视图平均
```

**预期收益:** 文本注入从 1 个 token → K 个语义视图，信息量提升 K 倍
**改动文件:** `modeling_mmldm_dit.py`, `training_stage2.py`, `inference.py`
**原创性风险:** 低 — 结构简单，与 CaTSG/VerbalTS 有本质区别

---

### 创新 2: Frequency-Weighted Flow Matching Loss

**灵感来源:** DiMTS Fourier Loss + CPiRi Frequency-Weighted Loss
**原创性分析:**
- DiMTS 只用频域 L1，没有频率加权
- CPiRi 只用频率加权 `1/k`，没有频域损失
- 我们融合两者 + 放在流匹配框架里 → 组合创新

**设计:**
```python
def frequency_weighted_flow_loss(
    pred: torch.Tensor,    # (L, D) 预测速度场
    target: torch.Tensor,  # (L, D) 目标速度场 (noise - z0)
    gamma_freq: float = 0.1,
    gamma_weighted: float = 0.05,
) -> torch.Tensor:
    """融合时域 MSE + 频域 L1 + 频率加权的流匹配损失。

    三项互补:
    - 时域 MSE: 保整体结构 (主损失)
    - 频域 L1: 保频谱准确性 (DiMTS 启发)
    - 频率加权 1/k: 保细节精度 (CPiRi 启发)
    """
    L = pred.shape[0]

    # 1. 时域 MSE (已有)
    l_time = F.mse_loss(pred, target)

    # 2. 频域 L1 (DiMTS 启发)
    fft_pred = torch.fft.rfft(pred, dim=0)
    fft_target = torch.fft.rfft(target, dim=0)
    l_freq = (F.l1_loss(fft_pred.real, fft_target.real) +
              F.l1_loss(fft_pred.imag, fft_target.imag))

    # 3. 频率加权 (CPiRi 启发)
    weights = 1.0 / torch.arange(1, L + 1, device=pred.device, dtype=pred.dtype)
    l_weighted = (weights.unsqueeze(-1) * (pred - target).abs()).mean()

    return l_time + gamma_freq * l_freq + gamma_weighted * l_weighted
```

**直觉:**
```
频率 →  低频 ───────────────────────────→ 高频
        │                                │
MSE:    │████████████████                │████          (偏向低频)
+Freq:  │████████████████████████████████│████████████  (均衡)
+Weight:│████████████                    │██████████████ (偏向高频)
```

**预期收益:** 模型同时学好趋势和细节，MSE/WAPE 指标改善
**改动文件:** `training_stage2.py` (替换 `compute_flow_matching_loss` 中的 MSE)
**原创性风险:** 低 — 纯 additive loss 组合

---

### 创新 3: Diffusion Batch Multiplication

**灵感来源:** CPiRi `diffusion_batch_mul=4`
**原创性分析:**
- CPiRi 用 4 倍重复获取更多流匹配训练对
- 我们可以声称 "Adaptive Batch Expansion" 并调整倍数策略

**设计:**
```python
# 在 training_stage2.py 中
# 每个 latent 重复 batch_mul 次，采样不同 t
batch_mul = 4
z0_expanded = z0.repeat_interleave(batch_mul, dim=0)  # (B*4, L, D)
t_per_token = torch.rand(z0_expanded.shape[0], device=device)
noise_expanded = torch.randn_like(z0_expanded)

# 同一样本在不同 t 下的梯度累积 → 更密集的训练信号
l_fm = compute_flow_matching_loss(dit, z0_expanded, ...)
```

**预期收益:** 同等 epoch 下训练信号密度提升 4 倍
**改动文件:** `training_stage2.py`
**原创性风险:** 低 — 可调整倍数策略使之有别于 CPiRi

---

### 创新 4: CFG Dropout 提升 (0.1 → 0.3)

**灵感来源:** T2S cfg_drop_prob=0.3, CaTSG cond_drop_prob=0.2
**设计:** 直接修改 argparse 默认值
**预期收益:** CFG 引导效果更强，生成质量提升
**改动文件:** `training_stage2.py` (1 行)
**原创性风险:** 无 — 标准超参数调整

---

### ~~创新 5: RevIN (Reversible Instance Normalization)~~ [已移除]

**灵感来源:** CPiRi RevIN + TimeDART revIN
**移除原因:** RevIN 是为 forecasting 设计的（观察前 N 步 → 归一化 → 预测后 M 步 → 反归一化）。
在 generation 任务中，推理阶段没有输入时序可供计算归一化统计量，导致 decode 时反归一化被静默跳过，
输出停留在 normalized 空间（mean≈0, std≈1）。VAE 已有 latent standardization 足以处理分布对齐。

---

## 三、实施优先级

| 优先级 | 创新点 | 预期收益 | 工作量 | 依赖 |
|---|---|---|---|---|
| P0 | CFG dropout 0.1→0.3 | 高 | 1 行 | 无 |
| P1 | Frequency-Weighted Loss | 中高 | ~20 行 | 无 |
| P1 | Diffusion Batch Mul | 中 | ~10 行 | 无 |
| P2 | MVTC | 高 | ~80 行 | 无 |
| ~~P2~~ | ~~RevIN~~ | — | — | 已移除（generation 任务不适用） |

---

## 四、推荐实验命令

### 实验 1: 基线 + P0/P1 改进

```bash
CUDA_VISIBLE_DEVICES=6 python -m mmldm.training_stage2 \
    --data_dir "./Three Levels Data/TSFragment-600K" \
    --vae_checkpoint ./checkpoints/stage1_conv1d/epoch_100.pt \
    --split_file ./splits.json \
    --datasets ETTh1 \
    --time_intervals 24 \
    --epochs 2000 \
    --batch_size 512 \
    --lr 1e-4 \
    --warmup_steps 1000 \
    --gamma1 0.0 --gamma2 0.0 \
    --dit_dim 192 --dit_layers 6 --dit_heads 4 \
    --block_size 8 \
    --cfg_drop_prob 0.3 \
    --gamma_cons 0.1 --cons_delta 0.05 \
    --curriculum_epochs 0 \
    --snr_alpha 0.3 \
    --ema_decay 0.9999 \
    --log_interval 10 \
    --save_dir ./checkpoints/stage2_v3 \
    --seed 42
```

### 实验 2: 实验 1 + MVTC

(待实施后补充)

### 实验 3: 实验 1 + Frequency-Weighted Loss

(待实施后补充)

---

## 五、与现有代码的集成点

### 已完成 (PR v2)
- [x] TextModulator 类 (modeling_mmldm_dit.py)
- [x] TGFM 集成到 MultimodalDiTBlock
- [x] text_latent 参数贯穿 training_stage2.py 和 inference.py
- [x] text_latent_dim=128 配置
- [x] OneCycleLR 调度器
- [x] CFG scale 7.0 默认值

### 已完成 (PR v3)
- [x] MultiViewTextPooler 类 + 集成到 MMLDMDiTModel（forward/prefix_kv/extend_prefix_kv 均已 view cycling）
- [x] frequency_weighted_flow_loss 函数 + 替换 compute_flow_matching_loss 中的 MSE
- [x] Diffusion batch multiplication (--batch_mul)
- [x] CFG dropout 0.1→0.3
- [x] n_runs 默认值 1→10（启用 MRR@10）
- [ ] ~~RevIN in VAE~~ 已移除（generation 任务推理阶段无法获取归一化统计量）
