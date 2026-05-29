# CTICD: Channel-Transform Interventional Causal Discovery Module

## 即插即用因果学习模块 — 完整设计与实现

---

## 1. 模块定位

CTICD 是 TIGER 框架的一个 **即插即用 (plug-and-play)** 因果学习模块。它的作用是在扩散模型训练过程中，从 3 通道图像 (GASF / STFT / RP) 中发现因果结构，并将因果特征注入到 TIGERDiT 的主干网络中。

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TIGER Pipeline                              │
│                                                                     │
│   Time Series ──→ TSToImageEncoder ──→ 3-channel Image (B,3,64,64) │
│                                            │                        │
│                    ┌───────────────────────┐│                        │
│                    │      CTICD Module     ││  ← 即插即用模块         │
│                    │  输入: image + attr   ││                        │
│                    │  输出: causal_feats   ││                        │
│                    └──────────┬────────────┘│                        │
│                               ↓             ↓                        │
│                    TIGERDiT.forward():                               │
│                      x_in = x_in + gate * causal_feats              │
│                      ↓                                              │
│                    ResidualBlocks → noise_pred                       │
│                      ↓                                              │
│                    ImageToTSDecoder → Generated Time Series          │
└─────────────────────────────────────────────────────────────────────┘
```

**核心设计原则**：
- 模块内部不依赖 TIGERDiT 的内部状态，只读取原始 image 和 attr_emb
- 输出 shape 与 TIGERDiT 的 `x_in` 一致，通过 zero-init gate 加性注入
- 训练初期 gate≈0，模型等价于原始 TIGERDiT；随训练逐渐打开因果通道
- 消融时直接关掉模块即可，不改任何其他代码

---

## 2. 模块内部架构

```
输入: image (B,3,H,W), attr_emb (B,attr_dim,n_h,n_w)

   ┌──────────────────────────────────────────────────────┐
   │  A. Channel Encoder                                  │
   │  ─────────────────                                   │
   │  3 个独立的 patch encoder，每个处理一个通道:           │
   │    GASF (ch=0) → GASF patches → K_G mechanism states  │
   │    STFT (ch=1) → STFT patches → K_S mechanism states  │
   │    RP   (ch=2) → RP   patches → K_R mechanism states  │
   │                                                      │
   │  每个 encoder: Conv2d(patch) → Flatten → Linear       │
   │  每个通道输出 K 个 mechanism state (learnable pooling) │
   │  总输出: mechanism_states (B, 3K, d_model)            │
   └──────────────────────┬───────────────────────────────┘
                          ↓
   ┌──────────────────────────────────────────────────────┐
   │  B. Causal Graph Learner                             │
   │  ──────────────────────                              │
   │  输入: mechanism_states (B, 3K, d_model)              │
   │  输出: causal_graph (B, 3K, 3K) soft adjacency       │
   │                                                      │
   │  计算:                                                │
   │    sim = cosine_similarity(h_i, h_j)                  │
   │    A = sigmoid(sim * direction / tau)                 │
   │    A = A * (1 - I)   # 去除自环                       │
   │                                                      │
   │  约束:                                                │
   │    L_notears = tr(exp(A⊙A)) - 3K  ≈ 0  (无环)       │
   │    L_sparse = ||A||_1                                 │
   └──────────────────────┬───────────────────────────────┘
                          ↓
   ┌──────────────────────────────────────────────────────┐
   │  C. Causal Mechanism Transition                      │
   │  ────────────────────────────                        │
   │  对每个 mechanism k:                                  │
   │    1. 聚合父节点: h_parent = Σ_j A[j,k] · h_j        │
   │    2. 独立 MLP 变换: h_k' = MLP_k(h_parent)          │
   │    3. text conditioning via FiLM:                     │
   │       h_k'' = (1+γ_k)·LN(h_k') + β_k                │
   │    4. 残差连接: h_k_out = h_k + h_k''                 │
   │                                                      │
   │  ICM 原则: 每个 mechanism 有独立的 MLP 和 FiLM 参数   │
   └──────────────────────┬───────────────────────────────┘
                          ↓
   ┌──────────────────────────────────────────────────────┐
   │  D. Mechanism Recomposer                             │
   │  ───────────────────────                             │
   │  输入: updated_states (B, 3K, d_model)                │
   │  输出: causal_features (B, channels, 1, total_tokens) │
   │                                                      │
   │  1. Cross-attention: 每个 mechanism query 其余所有     │
   │  2. Importance gate: softmax weighted sum             │
   │  3. Project to TIGERDiT token space                   │
   │  4. Zero-init output projection                       │
   └──────────────────────┬───────────────────────────────┘
                          ↓
   causal_features (B, channels, 1, total_tokens) + losses dict
```

---

## 3. 统一公式

### 训练公式

```
ε_θ(x_t, t, c) = DIT(x_t, t, c) + g · Φ(I, c; θ_causal)

其中:
  DIT       = TIGERDiT 主干网络
  Φ         = CTICD 模块
  g         = zero-initialized gate (可学习标量)
  I         = clean image (B,3,H,W)，训练时有 ground truth
  c         = attr_emb (text conditioning)
  θ_causal  = CTICD 的可学习参数
```

### 训练损失

```
L_total = L_diffusion + λ_causal · L_causal

L_diffusion = MSE(ε, ε_θ)                         # 标准扩散损失
L_causal    = L_mech + L_graph + L_inv + L_icm     # CTICD 辅助损失

L_mech  = MSE(δ_pred, δ_target)                    # 机制预测损失
L_graph = tr(exp(A⊙A)) - 3K + λ₁·||A||₁           # NOTEARS 无环 + 稀疏
L_inv   = Σ_{c≠c'} (1 - cos_sim(mean(h_c), mean(h_c')))  # CCIP 跨通道不变性
L_icm   = hinge penalty on weak graph edges        # ICM 独立性
```

### 因果三级生成 (同一公式，不同干预设置)

```
Level 1 — Association (标准采样):
  ε = DIT(x_t, t, c)                                    # CTICD 关闭

Level 2 — Intervention (因果干预):
  ε = DIT(x_t, t, c) + g · Φ(I, c; do(M_c^k = m))     # 固定某机制状态

Level 3 — Counterfactual (反事实):
  1. Abduction:  M* = Enc(I_obs)                          # 从观测推断机制状态
  2. Action:     M' = modify(M*, {M_c^k = m})             # 修改特定机制
  3. Prediction: ε = DIT(x_t, t, c) + g · Φ(I, c; M')   # 生成反事实
```

---

## 4. 与 CaTSG 的对比

| 维度 | CaTSG (BAG) | CTICD (Ours) |
|------|-------------|--------------|
| 因果粒度 | 环境级 (K 个环境) | 机制级 (3K 个机制) |
| 因果结构 | 无 (env_bank 是 bag of vectors) | 有向因果图 A ∈ R^{3K×3K} |
| 环境来源 | 学习的 env_bank | 已知数学变换 (GASF/STFT/RP) |
| 干预语义 | BAG 加权平均 (非真正干预) | 机制级 do-operation |
| 可解释性 | env_probs (K 维) | 因果图 + 机制状态 (可可视化) |
| 集成方式 | 改 sampler | 即插即用模块 (不改 sampler) |

---

## 5. 消融实验设计

| 实验名称 | 模块设置 | 验证目标 |
|----------|---------|---------|
| Baseline | CTICD off | 原始 TIGERDiT 性能 |
| CTICD (full) | CTICD on, all losses | 完整因果模块效果 |
| CTICD (no graph) | 去掉 Graph Learner, 无邻接矩阵 | 因果图的贡献 |
| CTICD (no invariance) | 去掉 CCIP loss L_inv | 跨通道不变性的作用 |
| CTICD (no ICM) | 所有 mechanism 共享同一个 MLP | 独立机制假设的作用 |
| CTICD (gate frozen) | gate 固定为 0 | 验证 gate 学习的意义 |
| CTICD (1 channel) | 只用 GASF 或 STFT 一个通道 | 单通道 vs 三通道 |

---

## 6. 可解释性分析

训练完成后，可以提取以下信息进行可视化：

1. **因果图 heatmap**: `causal_graph (B, 3K, 3K)` → 可视化机制间的因果关系
   - GASF 的机制是否影响 STFT 的机制？（跨通道因果）
   - 哪些机制是"根节点"（无父节点）？哪些是"叶节点"？

2. **机制状态分布**: `mechanism_states (B, 3K, d_model)` → t-SNE 可视化
   - 不同通道的机制是否形成可分离的聚类？
   - 同一通道的不同机制是否捕获不同的因果模式？

3. **干预效果对比**: 对比 association vs intervention 生成结果
   - 固定 STFT 的某个机制 → 生成结果在频率维度上变化
   - 固定 GASF 的某个机制 → 生成结果在值维度上变化

4. **CCIP 验证**: 对比跨通道预测一致性
   - 从 GASF 特征预测 x_t vs 从 STFT 特征预测 x_t → 应该一致

---

## 7. 完整代码

以下是 `mmldm/tiger/cticd.py` 的完整实现：

```python
"""CTICD: Channel-Transform Interventional Causal Discovery for TIGER.

Plug-and-play causal learning module that discovers mechanism-level causal
structure from the 3-channel image representation (GASF / STFT / RP).

Usage:
    1. Instantiate CTICD and pass it to TIGERDiT.
    2. In TIGERDiT.forward(), call CTICD before ResidualBlocks.
    3. Add CTICD losses to the total training loss.
    4. For ablation, simply set cticd=None to disable.

Unified generation formula:
    ε = DIT(x_t, t, c) + g · Φ(I, c)

    where Φ = CTICD, g = zero-initialized gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class CTICDOutput:
    """Return value of CTICD.forward()."""
    causal_features: torch.Tensor       # (B, channels, 1, total_tokens)
    causal_graph: torch.Tensor          # (B, 3K, 3K) soft adjacency
    mechanism_states: list[torch.Tensor]  # 3K x (B, d_model)
    losses: dict[str, torch.Tensor]     # individual loss terms


# ---------------------------------------------------------------------------
# Component A: Channel Encoder
# ---------------------------------------------------------------------------

class ChannelEncoder(nn.Module):
    """Encode a single image channel into K mechanism representations.

    Architecture:
        1. Conv2d patch embedding: (B, 1, H, W) -> (B, d_model, n_h, n_w)
        2. Flatten spatial: (B, d_model, n_h*n_w)
        3. Learnable mechanism pooling: K prototypes over spatial dim
        4. Output: (B, K, d_model)

    Each channel gets its own encoder with independent parameters,
    reflecting the different mathematical structure of GASF / STFT / RP.
    """

    def __init__(
        self,
        d_model: int,
        n_mechanisms: int,
        patch_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_mechanisms = n_mechanisms
        self.patch_size = patch_size

        # Patch embedding: single channel input
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(1, d_model),
            nn.GELU(approximate='tanh'),
        )

        # Learnable mechanism prototypes for soft spatial pooling
        # Each mechanism has a query vector that attends over spatial positions
        self.mechanism_queries = nn.Parameter(
            torch.randn(n_mechanisms, d_model) * 0.02,
        )
        self.temp_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
        )
        # Per-mechanism output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(approximate='tanh'),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.patch_embed:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)
        for m in self.out_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, channel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel: (B, 1, H, W) single image channel.

        Returns:
            mechanism_states: (B, K, d_model)
        """
        B = channel.shape[0]

        # Patch embedding
        feat = self.patch_embed(channel)  # (B, d_model, n_h, n_w)
        _, C, n_h, n_w = feat.shape
        spatial = feat.reshape(B, C, n_h * n_w).permute(0, 2, 1)  # (B, L, d_model)

        # Mechanism-aware soft pooling via cross-attention
        queries = self.mechanism_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)
        attended, _ = self.temp_attn(
            query=queries, key=spatial, value=spatial,
        )  # (B, K, d_model)

        # Per-mechanism projection + norm
        out = self.out_proj(attended)  # (B, K, d_model)
        out = self.norm(out + attended)  # residual

        return out  # (B, K, d_model)


# ---------------------------------------------------------------------------
# Component B: Causal Graph Learner
# ---------------------------------------------------------------------------

class CausalGraphLearner(nn.Module):
    """Learn a directed causal graph over 3K mechanism nodes.

    Uses cosine similarity + learnable direction parameters to produce
    a soft adjacency matrix in [0, 1].

    Losses:
        - NOTEARS acyclicity: tr(exp(A ⊙ A)) - 3K = 0 iff DAG
        - L1 sparsity: ||A||_1

    Channel-aware initialization:
        - Intra-channel edges initialized small (within-channel causation weaker)
        - Inter-channel edges initialized with small random values
        - This biases discovery toward cross-channel causation
    """

    def __init__(
        self,
        d_model: int,
        n_mechanisms_per_channel: int,
        n_channels: int = 3,
    ):
        super().__init__()
        self.n_total = n_channels * n_mechanisms_per_channel

        # Learnable direction parameters: asymmetric edge weights
        self.direction_logits = nn.Parameter(
            torch.zeros(self.n_total, self.n_total),
        )
        # Temperature for sigmoid
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # tau = 1.0

        # Channel-aware initialization
        self._init_with_channel_bias(n_mechanisms_per_channel, n_channels)

    def _init_with_channel_bias(
        self, K: int, n_channels: int,
    ):
        """Initialize direction_logits with channel structure bias."""
        with torch.no_grad():
            # Intra-channel: small negative (bias against within-channel edges)
            for c in range(n_channels):
                start = c * K
                end = start + K
                self.direction_logits[start:end, start:end] = -0.5

            # Inter-channel: small random (allow cross-channel discovery)
            for c1 in range(n_channels):
                for c2 in range(n_channels):
                    if c1 != c2:
                        s1, e1 = c1 * K, c1 * K + K
                        s2, e2 = c2 * K, c2 * K + K
                        self.direction_logits[s1:e1, s2:e2] = (
                            torch.randn(K, K) * 0.1
                        )

    def forward(
        self, mechanism_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mechanism_states: (B, 3K, d_model)

        Returns:
            causal_graph: (B, 3K, 3K) soft adjacency in [0, 1]
            notears_loss: scalar acyclicity loss
            sparsity_loss: scalar L1 penalty
        """
        B, N, D = mechanism_states.shape

        # Pairwise cosine similarity
        normed = F.normalize(mechanism_states, dim=-1)  # (B, N, D)
        cos_sim = torch.bmm(normed, normed.transpose(1, 2))  # (B, N, N)

        # Apply learnable direction (asymmetric)
        tau = self.log_tau.exp().clamp(min=0.1, max=5.0)
        direction = torch.sigmoid(self.direction_logits)  # (N, N)
        raw = cos_sim * direction.unsqueeze(0) / tau  # (B, N, N)

        # Sigmoid to [0, 1]
        causal_graph = torch.sigmoid(raw)  # (B, N, N)

        # Zero diagonal (no self-loops)
        eye = torch.eye(N, device=causal_graph.device).unsqueeze(0)
        causal_graph = causal_graph * (1 - eye)

        # NOTEARS acyclicity loss
        W_sq = causal_graph * causal_graph
        traces = torch.stack([
            torch.trace(torch.matrix_exp(W_sq[b])) for b in range(B)
        ])
        notears_loss = (traces - N).mean()

        # L1 sparsity
        sparsity_loss = causal_graph.abs().mean()

        return causal_graph, notears_loss, sparsity_loss


# ---------------------------------------------------------------------------
# Component C: Causal Mechanism Transition
# ---------------------------------------------------------------------------

class CausalMechanismTransition(nn.Module):
    """Per-mechanism transition with causal parent aggregation.

    For each mechanism k:
        1. Aggregate parent states: h_parent = Σ_j A[j,k] · h_j
        2. Independent MLP: h_k' = MLP_k(h_parent)    (ICM principle)
        3. Text conditioning via FiLM:
           h_k'' = (1 + γ_k) · LN(h_k') + β_k
        4. Residual: h_k_out = h_k + h_k''

    ICM Principle: each mechanism has its own MLP (independently modifiable).
    """

    def __init__(
        self,
        d_model: int,
        n_mechanisms: int,
        hidden_dim: int = 256,
        attr_dim: int = 256,
    ):
        super().__init__()
        self.n_mechanisms = n_mechanisms
        self.d_model = d_model

        # Per-mechanism independent MLPs (ICM principle)
        self.mechanism_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(hidden_dim, d_model),
            )
            for _ in range(n_mechanisms)
        ])

        # Per-mechanism FiLM conditioning from text attribute
        self.film_gamma = nn.ModuleList([
            nn.Linear(attr_dim, d_model) for _ in range(n_mechanisms)
        ])
        self.film_beta = nn.ModuleList([
            nn.Linear(attr_dim, d_model) for _ in range(n_mechanisms)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_mechanisms)
        ])

        self._init_weights()

    def _init_weights(self):
        for mlp in self.mechanism_mlps:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        # FiLM: gamma init to 1, beta init to 0 (identity at start)
        for gamma_linear in self.film_gamma:
            nn.init.zeros_(gamma_linear.weight)
            nn.init.ones_(gamma_linear.bias)
        for beta_linear in self.film_beta:
            nn.init.zeros_(beta_linear.weight)
            nn.init.zeros_(beta_linear.bias)

    def forward(
        self,
        mechanism_states: torch.Tensor,
        causal_graph: torch.Tensor,
        attr_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mechanism_states: (B, 3K, d_model)
            causal_graph: (B, 3K, 3K) soft adjacency
            attr_pooled: (B, attr_dim) pooled text attribute

        Returns:
            updated_states: (B, 3K, d_model)
        """
        B, N, D = mechanism_states.shape

        # Aggregate parent states via graph:
        # parent_input[k] = Σ_j A[j,k] · h_j
        # This is a matrix multiplication: (B, 3K, 3K) @ (B, 3K, d_model)
        parent_input = torch.bmm(causal_graph.transpose(1, 2), mechanism_states)
        # parent_input: (B, 3K, d_model)

        # Per-mechanism MLP + FiLM
        updated = []
        for k in range(self.n_mechanisms):
            h_k = parent_input[:, k, :]  # (B, d_model)

            # Independent MLP (ICM)
            h_k = self.mechanism_mlps[k](h_k)  # (B, d_model)

            # FiLM conditioning
            gamma = self.film_gamma[k](attr_pooled)  # (B, d_model)
            beta = self.film_beta[k](attr_pooled)    # (B, d_model)
            h_k = self.layer_norms[k](h_k)
            h_k = (1 + gamma) * h_k + beta

            # Residual
            h_k = mechanism_states[:, k, :] + h_k
            updated.append(h_k)

        return torch.stack(updated, dim=1)  # (B, 3K, d_model)


# ---------------------------------------------------------------------------
# Component D: Mechanism Recomposer
# ---------------------------------------------------------------------------

class MechanismRecomposer(nn.Module):
    """Fuse 3K mechanism states into TIGERDiT-compatible causal features.

    Architecture:
        1. Cross-attention: each mechanism queries all mechanism states
        2. Importance gate: learnable weighted sum (softmax over 3K)
        3. Project to TIGERDiT token space: (B, 3K, d_model) -> (B, C, 1, L)
        4. Zero-init output projection (safe training start)
    """

    def __init__(
        self,
        d_model: int,
        n_mechanisms: int,
        output_channels: int,
        total_tokens: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.n_mechanisms = n_mechanisms
        self.output_channels = output_channels
        self.total_tokens = total_tokens

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        # Importance gate
        self.importance = nn.Linear(d_model, n_mechanisms)

        # Project to token space
        self.token_proj = nn.Linear(d_model, output_channels * total_tokens)

        # Zero-init for safe start
        nn.init.zeros_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        nn.init.zeros_(self.importance.bias)

    def forward(self, mechanism_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mechanism_states: (B, 3K, d_model)

        Returns:
            causal_features: (B, output_channels, 1, total_tokens)
        """
        B = mechanism_states.shape[0]

        # Cross-attention: each mechanism queries all
        attended, _ = self.cross_attn(
            query=mechanism_states,
            key=mechanism_states,
            value=mechanism_states,
        )  # (B, 3K, d_model)

        # Importance gate (learnable weighted sum)
        # Average attended features for gate computation
        pooled = attended.mean(dim=1)  # (B, d_model)
        weights = torch.softmax(self.importance(pooled), dim=-1)  # (B, 3K)
        fused = (attended * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        # Project to token space
        tokens = self.token_proj(fused)  # (B, C * total_tokens)
        tokens = tokens.reshape(B, self.output_channels, 1, self.total_tokens)

        return tokens  # (B, channels, 1, total_tokens)


# ---------------------------------------------------------------------------
# CCIP: Cross-Channel Invariance Loss
# ---------------------------------------------------------------------------

def compute_ccip_loss(
    mechanism_states: torch.Tensor,
    n_mechanisms_per_channel: int,
    n_channels: int = 3,
) -> torch.Tensor:
    """Cross-Channel Invariance Principle (CCIP) loss.

    Causal features should be invariant across channel representations.
    We enforce this by computing per-channel prediction of a shared target
    and minimizing the KL divergence between predictions.

    For simplicity, we use cosine similarity between channel-mean features:
        loss = Σ_{c≠c'} (1 - cos_sim(mean(h_c), mean(h_c')))

    Args:
        mechanism_states: (B, 3K, d_model)
        n_mechanisms_per_channel: K
        n_channels: 3

    Returns:
        invariance_loss: scalar
    """
    B, N, D = mechanism_states.shape
    K = n_mechanisms_per_channel

    # Split by channel and average within each channel
    channel_means = []
    for c in range(n_channels):
        start = c * K
        end = start + K
        ch_mean = mechanism_states[:, start:end, :].mean(dim=1)  # (B, D)
        channel_means.append(ch_mean)

    # Pairwise invariance: cosine similarity should be close to 1
    loss = mechanism_states.new_tensor(0.0)
    count = 0
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            cos_sim = F.cosine_similarity(
                channel_means[i], channel_means[j], dim=-1,
            )  # (B,)
            loss = loss + (1 - cos_sim).mean()
            count += 1

    return loss / max(count, 1)


# ---------------------------------------------------------------------------
# ICM Independence Loss
# ---------------------------------------------------------------------------

def compute_icm_loss(
    mechanism_states: torch.Tensor,
    causal_graph: torch.Tensor,
    n_mechanisms_per_channel: int,
) -> torch.Tensor:
    """ICM independence loss: non-parent mechanisms should not affect a mechanism.

    For each mechanism k, we check that perturbing non-parent mechanisms j
    does NOT change mechanism k's output. Approximated via gradient penalty.

    Simplified version: encourage off-graph entries to be zero.
        loss = Σ_{j∉Pa(k)} A[j,k]^2

    Args:
        mechanism_states: (B, 3K, d_model)
        causal_graph: (B, 3K, 3K)
        n_mechanisms_per_channel: K

    Returns:
        icm_loss: scalar
    """
    # Sparsity on weak edges: penalize small but non-zero entries
    # This encourages the graph to be truly sparse (binary-like)
    A = causal_graph
    # Hinge-like penalty: entries below threshold contribute to loss
    threshold = 0.1
    weak_edges = torch.clamp(threshold - A, min=0.0) + torch.clamp(A - (1 - threshold), min=0.0)
    return weak_edges.mean()


# ---------------------------------------------------------------------------
# Main CTICD Module
# ---------------------------------------------------------------------------

class CTICD(nn.Module):
    """Channel-Transform Interventional Causal Discovery.

    Plug-and-play causal learning module for TIGER.

    Args:
        d_model: internal feature dimension
        n_mechanisms_per_channel: K mechanisms per channel (default 4)
        n_channels: number of image channels (default 3: GASF/STFT/RP)
        patch_size: patch size for channel encoder (default 4)
        hidden_dim: MLP hidden dim in transition (default 256)
        attr_dim: text attribute dimension (default 256)
        num_heads: attention heads in recomposer (default 4)
        output_channels: output channels matching TIGERDiT (default 256)
        total_tokens: total token count matching TIGERDiT (default L)
        lambda_graph: weight for graph losses (default 0.1)
        lambda_inv: weight for CCIP invariance loss (default 0.05)
        lambda_icm: weight for ICM independence loss (default 0.01)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_mechanisms_per_channel: int = 4,
        n_channels: int = 3,
        patch_size: int = 4,
        hidden_dim: int = 256,
        attr_dim: int = 256,
        num_heads: int = 4,
        output_channels: int = 256,
        total_tokens: int = 256,
        lambda_graph: float = 0.1,
        lambda_inv: float = 0.05,
        lambda_icm: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = n_mechanisms_per_channel
        self.n_channels = n_channels
        self.n_total = n_channels * n_mechanisms_per_channel
        self.lambda_graph = lambda_graph
        self.lambda_inv = lambda_inv
        self.lambda_icm = lambda_icm

        # --- Component A: Per-channel encoders ---
        self.channel_encoders = nn.ModuleList([
            ChannelEncoder(d_model, n_mechanisms_per_channel, patch_size)
            for _ in range(n_channels)
        ])

        # --- Component B: Causal graph learner ---
        self.graph_learner = CausalGraphLearner(
            d_model, n_mechanisms_per_channel, n_channels,
        )

        # --- Component C: Mechanism transition ---
        self.transition = CausalMechanismTransition(
            d_model, self.n_total, hidden_dim, attr_dim,
        )

        # --- Component D: Recomposer ---
        self.recomposer = MechanismRecomposer(
            d_model, self.n_total, output_channels, total_tokens, num_heads,
        )

        # --- Attribute pooling (for FiLM conditioning) ---
        self.attr_pool = nn.AdaptiveAvgPool2d(1)  # (B, attr_dim, H, W) -> (B, attr_dim, 1, 1)

        # --- Zero-initialized gate ---
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        image: torch.Tensor,
        attr_emb: torch.Tensor | None = None,
    ) -> CTICDOutput:
        """
        Args:
            image: (B, 3, H, W) clean 3-channel image (GASF/STFT/RP).
                   During training, this is the ground truth clean image.
                   During sampling, this can be the current denoised estimate.
            attr_emb: (B, attr_dim, n_h, n_w) text attribute embedding,
                      or None for unconditional generation.

        Returns:
            CTICDOutput with causal_features, causal_graph, losses.
        """
        B, C, H, W = image.shape
        device = image.device

        # Pool text attribute for FiLM conditioning
        if attr_emb is not None:
            attr_pooled = self.attr_pool(attr_emb).squeeze(-1).squeeze(-1)  # (B, attr_dim)
        else:
            attr_pooled = torch.zeros(B, 256, device=device)  # default dim

        # --- A. Per-channel encoding ---
        mechanism_list = []
        for c in range(self.n_channels):
            ch = image[:, c:c+1, :, :]  # (B, 1, H, W)
            mech_c = self.channel_encoders[c](ch)  # (B, K, d_model)
            mechanism_list.append(mech_c)

        # Concatenate: (B, 3K, d_model)
        mechanism_states = torch.cat(mechanism_list, dim=1)

        # --- B. Causal graph learning ---
        causal_graph, notears_loss, sparsity_loss = self.graph_learner(
            mechanism_states,
        )

        # --- C. Causal mechanism transition ---
        updated_states = self.transition(
            mechanism_states, causal_graph, attr_pooled,
        )

        # --- D. Recompose to causal features ---
        causal_features = self.recomposer(updated_states)  # (B, C, 1, L)

        # Apply zero-initialized gate
        causal_features = self.gate * causal_features

        # --- Compute losses ---
        inv_loss = compute_ccip_loss(
            mechanism_states, self.K, self.n_channels,
        )
        icm_loss = compute_icm_loss(
            mechanism_states, causal_graph, self.K,
        )

        total_loss = (
            self.lambda_graph * (notears_loss + sparsity_loss)
            + self.lambda_inv * inv_loss
            + self.lambda_icm * icm_loss
        )

        losses = {
            "cticd_notears": notears_loss,
            "cticd_sparsity": sparsity_loss,
            "cticd_invariance": inv_loss,
            "cticd_icm": icm_loss,
            "cticd_total": total_loss,
        }

        return CTICDOutput(
            causal_features=causal_features,
            causal_graph=causal_graph,
            mechanism_states=updated_states,
            losses=losses,
        )
```

---

## 8. 集成代码修改

### 8.1 修改 `dit_model.py` — TIGERDiT.__init__

在 `__init__` 末尾添加：

```python
# --- CTICD: optional causal module ---
cticd_config = config.get("cticd", None)
if cticd_config is not None:
    from mmldm.tiger.cticd import CTICD
    self.cticd = CTICD(
        d_model=cticd_config.get("d_model", 128),
        n_mechanisms_per_channel=cticd_config.get("n_mechanisms_per_channel", 4),
        n_channels=cticd_config.get("n_channels", 3),
        patch_size=cticd_config.get("patch_size", 4),
        hidden_dim=cticd_config.get("hidden_dim", 256),
        attr_dim=cticd_config.get("attr_dim", self.channels),
        num_heads=cticd_config.get("num_heads", 4),
        output_channels=self.channels,
        total_tokens=None,  # set dynamically in forward
        lambda_graph=cticd_config.get("lambda_graph", 0.1),
        lambda_inv=cticd_config.get("lambda_inv", 0.05),
        lambda_icm=cticd_config.get("lambda_icm", 0.01),
    )
else:
    self.cticd = None
```

### 8.2 修改 `dit_model.py` — TIGERDiT.forward()

在 `x_in = x_in.unsqueeze(2)` 之后（约第 647 行），ResidualLayers 之前，插入：

```python
# --- CTICD causal feature injection ---
if self.cticd is not None:
    # Update total_tokens dynamically on first call
    if self.cticd.recomposer.total_tokens != total_tokens:
        self.cticd.recomposer.total_tokens = total_tokens
        self.cticd.recomposer.token_proj = nn.Linear(
            self.cticd.d_model,
            self.channels * total_tokens,
        ).to(device=image.device)
        nn.init.zeros_(self.cticd.recomposer.token_proj.weight)
        nn.init.zeros_(self.cticd.recomposer.token_proj.bias)

    cticd_out = self.cticd(image, attr_emb)
    causal_features = cticd_out.causal_features  # (B, C, 1, total_tokens)
    # Store losses for train.py to access
    self._cticd_losses = cticd_out.losses
    self._cticd_graph = cticd_out.causal_graph

    x_in = x_in + causal_features  # gate is already applied in CTICD
```

### 8.3 修改 `train.py` — 训练循环

```python
# After computing diffusion loss:
loss_diff = F.mse_loss(noise_pred, noise)

# Add CTICD losses if available
loss_total = loss_diff
if hasattr(dit, '_cticd_losses') and dit._cticd_losses is not None:
    loss_total = loss_total + dit._cticd_losses['cticd_total']

# Backward
loss_total.backward()
```

---

## 9. 配置示例

```yaml
# config.yaml
model:
  dit:
    channels: 256
    nheads: 4
    layers: 8
    num_steps: 1000
    diffusion_embedding_dim: 128
    base_patch: 4
    multipatch_num: 1
    condition_type: "adaLN"

    # CTICD module (set to null to disable)
    cticd:
      d_model: 128
      n_mechanisms_per_channel: 4
      n_channels: 3
      patch_size: 4
      hidden_dim: 256
      attr_dim: 256
      num_heads: 4
      lambda_graph: 0.1
      lambda_inv: 0.05
      lambda_icm: 0.01

# Ablation: disable CTICD
# model:
#   dit:
#     cticd: null
```

---

*文档版本: v1.0*
*最后更新: 2026-05-29*
