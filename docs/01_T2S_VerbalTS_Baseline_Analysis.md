# T2S & VerbalTS Baseline 详细对比分析

> 为后续 Text+Image → TS 创新实验提供 baseline 参照

---

## 1. T2S (Text-to-Series, IJCAI 2025)

**论文**: High-resolution Time Series Generation with Text-to-Series Diffusion Models
**代码**: https://github.com/WinfredGe/T2S

### 1.1 评估指标

| 指标 | 方向 | 计算方式 | 说明 |
|------|------|----------|------|
| **MSE** | ↓ | `mean((real - gen)^2)` per-sample, averaged | 归一化后的均方误差 |
| **WAPE** | ↓ | `sum(\|real - gen\|) / sum(\|real\|)` per-sample | 加权绝对百分比误差 |
| **C-FID** | ↓ | FID in TS2Vec embedding space | 条件 FID |
| **MRR** | ↑ | Mean Reciprocal Rank, cosine sim threshold=0.5 | 多样本指标 (10 runs) |
| **CRPS** | ↓ | Continuous Ranked Probability Score | 概率校准指标 |

默认 `--method_list` = `MSE,WAPE,MRR`

Feature-based measures (代码中存在但未默认启用): MDD, ACD, SD, KD

### 1.2 数据集格式 (TSFragment-600K)

**文件格式**: CSV, 列 `Text`, `OT`, `TextEmbedding`

```
embedding_cleaned_ETTh1_24.csv
embedding_cleaned_ETTh1_48.csv
embedding_cleaned_ETTh1_96.csv
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `Text` | str | 自然语言描述 |
| `OT` | str (list) | 时间序列, `ast.literal_eval` 解析, 长度 24/48/96 |
| `TextEmbedding` | str (list) | 预计算 OpenAI embedding |

**归一化**: MinMaxScaler, 加载时应用

**数据层级**: TSFragment-600K (6 domains), SUSHI (单条长序列), MMD (8 domains)

### 1.3 训练管线

```
Stage 1: LA-VAE (VQ-VAE)
  - compression_factor=4, embedding_dim=64, num_embeddings=128
  - Loss: VQ-VAE reconstruction + commitment loss
  - 支持 24/48/96 混合训练

Stage 2: DiT Flow Matching
  - Backbone: RectifiedFlow (默认)
  - Flow: x_t = t * x_1 + (1-t) * x_0, target: v = x_1 - x_0
  - Loss: MSE(pred_velocity, noise_gt)
  - CFG: 30% 概率丢弃 text embedding
  - Optimizer: AdamW, lr=1e-4, OneCycleLR
  - Default: 20000 epochs, batch_size=9216
```

### 1.4 推理

- Euler ODE solver, 100 steps
- CFG scale: 5.0-13.0 (per dataset)
- 生成 10 runs 用于多样本指标

### 1.5 数据划分

- **比例**: 99% train / 1% test
- **Seed**: 2023
- **方式**: 随机排列 (fragment-level, 非时序分割)

---

## 2. VerbalTS (ICML 2025)

**论文**: VerbalTS: Generating Time Series from Texts
**代码**: https://github.com/seqml/VerbalTS

### 2.1 评估指标

基于预训练 **CTTP** (Contrastive Time-series Text Pretraining) 代理模型:

| 指标 | 方向 | 计算方式 | 说明 |
|------|------|----------|------|
| **CTTP Score** | ↑ | `trace(ts_emb @ cap_emb.T) / N` | 语义对齐, 类似 CLIP Score |
| **FID** | ↓ | Frechet Distance in TS embedding space | 生成分布 vs 训练集分布 |
| **JFTSD** | ↓ | Frechet Distance in [ts_emb, text_emb] joint space | 联合分布距离 |

**CTTP 模型**: PatchTST (TS encoder) + LongCLIP (text encoder, frozen), margin-based contrastive loss

**评估**: DDIM sampling, n_samples=10, 取 median, 3 runs with seeds [1, 7, 42]

### 2.2 数据集格式 (Weather)

**文件格式**: `.npy`

```
Weather/
├── train_ts.npy        # [n_samples, n_steps]
├── train_attrs_idx.npy # [n_samples, n_attrs]
├── train_text_caps.npy # [n_samples, ...] (allow_pickle=True)
├── valid_*.npy
├── test_*.npy
└── meta.json           # attr_list, attr_n_ops
```

每个样本: `{"ts": [n_steps, 1], "ts_len": int, "attrs": [n_attrs], "cap": str, "tp": [n_steps]}`

Weather: n_var=21 变量

### 2.3 训练管线

```
Stage 1: Unconditional Pretrain (DDPM)
  - 700 epochs, batch_size=512, 50 diffusion steps

Stage 2: Conditional Finetune
  - Multi-focal text alignment:
    - LongCLIP (frozen) → learned projection MLP
    - 3 sets of learnable anchors with cross-attention:
      - Variable anchors (per-var)
      - Scale anchors (per-patch-scale)
      - Diffusion-step anchors (per-stage)
  - Optimizer: Adam, lr=1e-3, MultiStepLR
  - Condition type: adaLN (Weather)
```

### 2.4 Baseline 对比

| Baseline | Weather FID | Weather JFTSD | Weather CTTP |
|----------|-------------|---------------|--------------|
| TEdit (Jing 2024) | 14.86 | 18.33 | 27.56 |
| **VerbalTS** | **6.13** | **8.56** | **30.62** |

---

## 3. T2S vs VerbalTS 关键对比

| 维度 | T2S | VerbalTS |
|------|-----|----------|
| 生成范式 | VAE latent → Flow Matching | DDPM noise → Denoise |
| 文本编码 | OpenAI embedding (预计算) | LongCLIP (frozen, 在线) |
| 文本对齐 | concat → cross-attention | Multi-focal anchors |
| VAE | LA-VAE (VQ-VAE, comp=4) | 无 VAE |
| 默认指标 | MSE, WAPE, MRR | CTTP, FID, JFTSD |
| 评估代理 | TS2Vec | CTTP (PatchTST+LongCLIP) |
| 数据格式 | CSV | .npy |
| 划分 | 99/1, seed=2023 | 自带 split |
| 多变量 | 单变量 (per-channel) | 多变量 (21 vars) |

---

## 4. 对 MMLDM 实验的启示

### 需要实现的评估指标

| 指标 | 对标 | 实现方式 |
|------|------|----------|
| MSE + WAPE | T2S | MinMax 归一化后计算 |
| CTTP + FID + JFTSD | VerbalTS | 实现代理模型 (PatchTST+LongCLIP) |
| C-FID | T2S | 预训练 TS2Vec |
| MRR + CRPS | T2S | 生成 10 runs |

### 数据集对齐

| 实验 | 数据集 | 划分 | 归一化 |
|------|--------|------|--------|
| 对比 T2S | TSFragment-600K | 99/1, seed=2023 | MinMax |
| 对比 VerbalTS | Weather (.npy) | 自带 split | 原始值 |
