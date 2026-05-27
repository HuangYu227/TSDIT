# PR: MMLDM v2 — Spectral Dual-Latent VAE + 工程改进 + PR v3

**分支:** `feature/mmv2-spectral-dual-latent`
**基准分支:** `main`
**日期:** 2026-05-24（更新）

---

## 摘要

MMLDM v2 是一次重大升级，引入 **Spectral Dual-Latent VAE**（通过 FFT 将时序分解为 trend 低频 + residual 高频，用双 Conv1d 编码器分别编码）以及两阶段的全面工程改进。

### 当前最佳结果

| 指标 | V1 (main, recon=0) | V2 Stage1 (训练中) | 目标 |
|------|-------------------|-------------------|------|
| Stage1 Recon | 0.0000 | 训练中 | < 0.001 |
| Stage2 WAPE | — | 待测 | < 0.183 (T2S SOTA) |

### 当前状态

- **V1 Stage 1** (main 分支): recon 已收敛到 0，VAE 可用
- **V2 Stage 1** (feature 分支): 训练中，20 组消融实验设计完成
- **V1 Stage 2**: 5 组消融实验设计完成，待 Stage 1 完成后执行
- **PR v3 创新**: TGFM/MVTC/FreqLoss/BatchMul/CFG 全部实装

---

## 一、创新架构

### 1. Spectral Dual-Latent VAE（创新 A） — `modeling_mmldm_vae.py`

**重写** VAE 编码器架构：

```
输入时序 x ∈ R^{L×C}
    ↓ FFT
x_low (低频趋势) ──→ trend_encoder (Conv1d) ──→ trend_proj ──→ z_trend
x_high (高频残差) ──→ residual_encoder (Conv1d) ──→ residual_proj ──→ z_residual
                                                                    ↓
                                                            z = [z_trend; z_residual]
                                                                    ↓
                                                            decoder ──→ x_recon
```

- **FFT 分解**: `fft_decompose(cutoff_ratio)` 将时序分解为低频趋势和高频残差
- **双 Conv1d 编码器**: `trend_encoder` 和 `residual_encoder` 共享架构，独立参数
- **双潜变量投影**: 分别产生 trend 和 residual 的 Gaussian 分布
- **合并潜变量**: `latent_dim = 64`, `trend_dim = 32`, `residual_dim = 32`
- **可配置 FFT 分割点**: `--fft_cutoff_ratio`（默认 0.3，实验范围 0.15~0.50）

**移除**: 旧版 Transformer-based 编码器（JointAttention, JointEncoderBlock 等，已不复存在）

**保留**: `encode_text_condition()` 用于 Stage 2 文本条件编码（独立 TextTransformerBlock）

### 2. Temporal Contrastive Latent Regularization — TCLR（创新 C）

- 向量化实现，正确梯度流
- 正样本对: 相邻时间步 (z_i, z_{i+1})
- 负样本对: k 步间隔 (k = L//4)
- Hinge loss, margin=1.0
- 逐样本独立计算，跨 batch 取均值
- 权重: `--gamma_tclr`（默认 0.1）

### 3. Text-Guided Feature Modulation — TGFM（PR v2 创新）

双通路条件注入，时间步控制 adaLN，文本独立调制特征：

```python
# 通路 1: 时间步条件（已有，不变）
x = x + gamma_attn * attn(adaLN(x, t_emb))

# 通路 2: 文本调制（新增，独立于时间步）
x = x + gamma_text * (scale * FFN(x) + shift)
#                     ↑
#              TextModulator(text_latent) → (scale, shift)
```

- 保持 MMDiT joint attention 不变
- TGFM 是加法增强
- 每层独立 TextModulator + 可学习 gate（初始化为 0.1，避免死梯度）

### 4. Multi-View Text Conditioning — MVTC（PR v3 创新）

```python
class MultiViewTextPooler:
    """从 SBERT 128-dim 生成 K=4 个正交文本视图"""
    def __init__(self):
        self.views = [Linear(128, 128) for _ in range(4)]  # 正交初始化
    
    def forward(self, text_emb):  # (B, 128)
        return [v(text_emb) for v in self.views]  # 4 × (B, 128)
```

- 每个 DiT block 循环使用不同的文本视图
- 正交初始化确保语义多样性
- 类比 CaTSG Environment Bank，但驱动信号来自文本而非数据分布

### 5. Frequency-Weighted Flow Matching Loss（PR v3 创新）

三项互补：
```
L = L_time (MSE) + γ_freq · L_freq (FFT L1) + γ_weighted · (1/k) · |pred - target|
```
- 时域 MSE: 保整体结构
- 频域 L1: 保频谱准确性（DiMTS 启发）
- 频率加权 1/k: 保高频细节（CPiRi 启发）
- 逐样本计算 FFT，避免跨边界伪影

### 6. Diffusion Batch Multiplication（PR v3 创新）

```python
z0_expanded = z0.repeat_interleave(batch_mul, dim=0)  # (B*4, L, D)
```
- `--batch_mul`（默认 1，推荐 4）
- 同一样本在不同 t 下的梯度累积 → 更密集的训练信号
- ts_shape 同步扩展，非 bug（经过分析确认）

### 7. Classifier-Free Guidance 改进（PR v3）

- CFG dropout: 0.1 → 0.3
- 同时覆盖 `text_latent` 和 `text_emb`（修复了只 drop text_latent 的漏洞）
- 推理 guidance_scale 默认 7.0

---

## 二、工程改进

| 改进 | 文件 | 描述 |
|------|------|------|
| **频谱重建损失** | `modeling_mmldm_vae.py` | 逐样本 FFT 实部/虚部 L1 损失，避免跨边界伪影 |
| **KL 退火** | `training_stage1.py` | `kl_anneal_start` → `kl_anneal_end` 线性插值 |
| **潜变量标准化** | `modeling_mmldm_vae.py` | epoch 0 后计算 dataset-level mean/std，存于 buffer |
| **OneCycleLR** | `training_stage2.py` | 替换 Cosine LambdaLR，更快的收敛 |
| **EMA** | `training_stage2.py` | decay=0.9999，`@torch.no_grad()` 应用/恢复 |
| **SiLU + GroupNorm** | `modeling_mmldm_vae.py` | ConvResidual 使用 SiLU + GroupNorm(1, dim) |
| **`--fft_cutoff_ratio`** | `training_stage1.py` | CLI 参数控制 FFT 分割点 |

---

## 三、Bug 修复清单

| # | Bug | 严重性 | 修复 |
|---|-----|--------|------|
| 1 | **TGFM 死梯度** — TextModulator 零初始化 + gate 零初始化 = 双死 | P0 | gate 初始化为 0.1 |
| 2 | **encode_text_condition 跨样本注意力** — (1, B, dim) Transformer 让样本互相可见 | P0 | 添加 diagonal mask |
| 3 | **推理双重标准化** — DiT 在标准化空间生成，vae.standardize_latent 再标准化一次 | P1 | 移除多余调用 |
| 4 | **CFG dropout 不完整** — text_latent 被 drop 但 text_emb 泄漏到 TGFM | P1 | `text_emb = text_emb * drop_mask` |
| 5 | **频率损失跨样本 FFT** — rfft 对 flat tensor 做，跨越样本边界 | P1 | 逐样本 FFT，传入 ts_shape |
| 6 | **snr_alpha train/val 不匹配** — 训练用 adaptive timestep，验证用 uniform | P1 | 验证 loop 添加 text_adaptive_timestep |
| 7 | **Stage2 潜变量统计覆盖** — Stage2 重算 stats 可能与 Stage1 不同 | P1 | Stage1 checkpoint 已有 stats 时跳过 |
| 8 | **RevIN 推理静默失败** — generation 任务无法在推理时算归一化统计量 | P2 | 完全移除 RevIN |

---

## 四、已知问题（已解决）

| 问题 | 根因 | 状态 |
|------|------|------|
| Val FM 不下降 | batch-level latent standardization 被移除前造成 train/val 分布不匹配 | ✅ 已修复 — 移除 batch-level fallback |
| V2 Stage1 初始 loss 巨大（2.18B） | V2 双编码器冷启动比 V1 单编码器差，正常现象 | ✅ 快速下降（10 step -27%） |
| KL 崩溃 | KL annealing 配置不当 | ✅ 修复 — `kl_anneal_end=1e-5` |

---

## 五、文件变更

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `mmldm/modeling_mmldm_vae.py` | 重写 | FFT 分解、双 Conv1d 编码器、TCLR、latent 标准化、文本编码器 diagonal mask |
| `mmldm/modeling_mmldm_dit.py` | 大幅修改 | TGFM、MVTC、TextModulator、gate 初始化修复 |
| `mmldm/training_stage2.py` | 大幅修改 | FreqLoss、BatchMul、OneCycleLR、CFG 双路径 dropout、snr_alpha 验证修复 |
| `mmldm/training_stage1.py` | 中等修改 | `--fft_cutoff_ratio`、latent stats 计算 |
| `mmldm/inference.py` | 中等修改 | 移除双重标准化、guidance_scale 默认 7.0、n_runs 默认 10 |
| `mmldm/configuration_mmldm.py` | 微调 | `n_text_views=4` |

---

## 六、训练命令

### Stage 1: V2 VAE Baseline（推荐 batch_size=1024）

```bash
CUDA_VISIBLE_DEVICES=6 python -m mmldm.training_stage1 \
    --data_dir "./Three Levels Data/TSFragment-600K" \
    --datasets ETTh1 --time_intervals 24 \
    --epochs 100 --batch_size 1024 --lr 1e-4 \
    --warmup_steps 200 --kl_anneal_epochs 10 \
    --kl_anneal_start 0.0 --kl_anneal_end 1e-5 \
    --gamma_spectral 0.1 --gamma_tclr 0.1 \
    --fft_cutoff_ratio 0.3 \
    --dim 256 --latent_dim 64 --num_heads 4 \
    --num_conv_layers 4 --encoder_blocks 6 --decoder_blocks 6 \
    --block_size 8 \
    --save_dir ./checkpoints/stage1_conv1d_v2 --seed 42
```

### Stage 2: DiT（V2 VAE checkpoint 就绪后使用）

```bash
CUDA_VISIBLE_DEVICES=6 python -m mmldm.training_stage2 \
    --data_dir "./Three Levels Data/TSFragment-600K" \
    --vae_checkpoint ./checkpoints/stage1_conv1d_v2/epoch_100.pt \
    --split_file ./splits.json \
    --datasets ETTh1 --time_intervals 24 \
    --epochs 2000 --batch_size 512 --lr 1e-4 \
    --warmup_steps 1000 \
    --gamma1 0.0 --gamma2 0.0 \
    --dit_dim 192 --dit_layers 6 --dit_heads 4 \
    --block_size 8 --cfg_drop_prob 0.3 \
    --gamma_cons 0.1 --cons_delta 0.05 \
    --curriculum_epochs 0 --snr_alpha 0.3 \
    --ema_decay 0.9999 \
    --log_interval 10 \
    --save_dir ./checkpoints/stage2_v2_tgfm --seed 42
```

### 推理

```bash
CUDA_VISIBLE_DEVICES=6 python -m mmldm.inference \
    --vae_checkpoint ./checkpoints/stage1_conv1d_v2/epoch_100.pt \
    --dit_checkpoint ./checkpoints/stage2_v2_tgfm/best.pt \
    --eval_data_dir "./Three Levels Data/TSFragment-600K" \
    --split_file ./splits.json \
    --eval_datasets ETTh1 --eval_time_intervals 24 \
    --output_len 96 --block_size 8 \
    --timestep_num 20 --guidance_scale 7.0 \
    --n_runs 10 --metrics "MSE,WAPE,MRR" \
    --eval_seed 42
```

---

## 七、消融实验设计

### Stage 1: 20 组实验

| 维度 | 组数 | 关键参数 |
|------|------|---------|
| **Loss 组合** | 5 | γ_spectral ∈ {0, 0.1, 0.5}, γ_tclr ∈ {0, 0.1, 0.5} |
| **KL 正则化** | 4 | kl_anneal_end ∈ {1e-6, 1e-5, 1e-4, 1e-3}, anneal_epochs ∈ {10, 30} |
| **Latent 维度** | 3 | latent_dim ∈ {32, 64, 128, 256} |
| **模型容量** | 3 | dim ∈ {128, 256, 384}, conv_layers ∈ {2, 4, 6} |
| **学习率** | 3 | lr ∈ {5e-5, 1e-4, 3e-4}, warmup ∈ {200, 500, 1000} |
| **FFT cutoff** | 2 | fft_cutoff_ratio ∈ {0.15, 0.30, 0.50} |

### Stage 2: 5 组实验（main 分支 V1 VAE）

| 实验 | 关键参数 |
|------|---------|
| Baseline | cfg_drop=0.0, γ1=0, γ2=0, dit_dim=256, dit_layers=8 |
| +CFG | cfg_drop=0.3 |
| +DCD | γ1=1.0, γ2=0.5 |
| +大 DiT | dit_dim=384, dit_layers=12, epochs=2000 |
| +大 Batch | batch_size=1024, dit_layers=12, epochs=2000, snr_alpha=0.3 |

---

## 八、待办清单

- [x] TGFM 双通路条件注入
- [x] MVTC K=4 正交文本视图
- [x] Frequency-Weighted Flow Matching Loss
- [x] Diffusion Batch Multiplication
- [x] CFG dropout 0.3（双路径）
- [x] TGFM gate 初始化修复（0 → 0.1）
- [x] encode_text_condition diagonal mask
- [x] 推理双重标准化修复
- [x] 频率损失跨样本 FFT 修复
- [x] snr_alpha train/val 一致性
- [x] Stage2 latent stats 跳过逻辑
- [x] RevIN 移除
- [x] n_runs 默认 10 (MRR@10)
- [x] OneCycleLR 调度器
- [x] `--fft_cutoff_ratio` CLI 参数
- [ ] V2 Stage 1 训练完成（100 epochs）
- [ ] V2 Stage 1 消融实验完成（20 组）
- [ ] V1 Stage 2 消融实验完成（5 组）
- [ ] V2 Stage 2 完整训练
- [ ] 评估 vs T2S baseline (WAPE < 0.183)
- [ ] View-Averaged Guidance 推理增强
- [ ] Optimal Transport Flow Matching 路径
- [ ] Langevin 修正步集成

---

## 九、相关讨论

- V1 vs V2 VAE: V1 单编码器收敛快但容量有限；V2 双编码器初始 loss 大但理论上限更高
- Batch size: 推荐 1024（类比 T2S batch=9216），显存充足时越大越好
- DCD: Stage 2 训练 loss，非纯推理技术。γ1 控制 mix loss，γ2 控制 aux loss
- RevIN: 不适合 generation 任务（无法在推理时计算归一化统计量），已完全移除
- Latent 标准化: 在 Stage 1 epoch 0 后计算 dataset-level mean/std，存入 model buffer，Stage 2 直接复用
