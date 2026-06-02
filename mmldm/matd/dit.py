"""Patch Latent Diffusion DiT backbone for time series generation.

Implements a Diffusion Transformer (DiT) denoiser that operates on patch-level
latent representations produced by the MATD planner.  The architecture uses
relative temporal self-attention with learnable bucketized biases and
adaLN-Zero conditioning on both timestep and pooled text embeddings.

Classes:
    RelativeTemporalSelfAttention -- Multi-head self-attention with relative
        temporal position bias and optional density bias.
    TextTemporalDiTBlock -- Pre-norm transformer block with adaLN-Zero
        conditioning, self-attention, cross-attention to text, and MLP.
    T2PDenoiser -- Full DiT denoiser supporting epsilon-prediction and
        v-prediction modes.

Reference: MATD framework design doc.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------


def _get_sinusoidal_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (diffusers / Transformer convention).

    Args:
        timesteps: (B,) float or int timestep values.
        embedding_dim: output embedding dimension.

    Returns:
        (B, embedding_dim) sinusoidal positional encoding.
    """
    assert timesteps.ndim == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device,
    )
    exponent = exponent / half_dim
    freq = torch.exp(exponent)
    emb = timesteps.float()[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Adaptive LayerNorm modulation: x * (1 + scale) + shift.

    Args:
        x: (B, K, D) input features.
        shift: (B, D) shift parameter (broadcast over K).
        scale: (B, D) scale parameter (broadcast over K).

    Returns:
        (B, K, D) modulated features.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# -----------------------------------------------------------------------
#  Relative Temporal Self-Attention
# -----------------------------------------------------------------------


class RelativeTemporalSelfAttention(nn.Module):
    """Multi-head self-attention with relative temporal bias.

    Extends standard scaled dot-product attention with two additive bias
    terms derived from patch metadata:

    1. Relative temporal bias: pairwise center distances are bucketized
       into num_buckets signed bins, and a learned per-head embedding
       provides the bias for each (query, key) pair.
    2. Density bias (optional): per-query bias derived from
       log(density) via a learned linear projection, encouraging
       high-density patches to attract more attention.

    The final attention logits are::

        attn = Q K^T / sqrt(d) + rel_time_bias + density_bias
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        num_buckets: int = 32,
        max_rel_dist: float = 1.0,
        use_density_bias: bool = True,
        dropout: float = 0.0,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        """
        Args:
            dim: input / output feature dimension.
            n_heads: number of attention heads.
            num_buckets: number of signed buckets for relative position.
            max_rel_dist: maximum relative distance to bucketize.
            use_density_bias: whether to add a density-based query bias.
            dropout: attention dropout probability.
            qk_norm: whether to apply LayerNorm to Q and K before attention.
            norm_eps: epsilon for QK LayerNorm.
        """
        super().__init__()
        assert dim % n_heads == 0, (
            f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        )
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.num_buckets = num_buckets
        self.max_rel_dist = max_rel_dist
        self.use_density_bias = use_density_bias
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.to_qkv = nn.Linear(dim, dim * 3)

        # Output projection
        self.to_out = nn.Linear(dim, dim)

        # Relative temporal bias: signed bucketized embedding
        self.rel_time_bias = nn.Embedding(num_buckets, n_heads)

        # Optional density bias
        if use_density_bias:
            self.density_proj = nn.Linear(1, n_heads)

        # Optional QK normalization
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)

        self.attn_dropout = nn.Dropout(dropout)

    def _bucketize_centers(self, centers: torch.Tensor) -> torch.Tensor:
        """Compute bucketized relative temporal bias.

        Args:
            centers: (B, K) patch center positions in [0, 1].

        Returns:
            (B, n_heads, K, K) relative temporal bias logits.
        """
        # Pairwise center distances: (B, K, K)
        delta = centers[:, :, None] - centers[:, None, :]
        # Map to bucket indices in [0, num_buckets)
        bucket_size = 2.0 * self.max_rel_dist / self.num_buckets
        bucket_idx = (
            (delta + self.max_rel_dist) / bucket_size
        ).long().clamp(0, self.num_buckets - 1)
        # Lookup: (B, K, K, n_heads) -> (B, n_heads, K, K)
        bias = self.rel_time_bias(bucket_idx)
        return bias.permute(0, 3, 1, 2)

    def _density_bias(self, log_density: torch.Tensor) -> torch.Tensor:
        """Compute density-based query bias.

        Args:
            log_density: (B, K) log-density per patch.

        Returns:
            (B, n_heads, K, 1) density bias (broadcasts over key dim).
        """
        # (B, K, 1) -> Linear -> (B, K, n_heads) -> (B, n_heads, K, 1)
        bias = self.density_proj(log_density.unsqueeze(-1))
        return bias.permute(0, 2, 1).unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        centers: torch.Tensor,
        log_density: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, K, D) input features.
            centers: (B, K) patch center positions for relative bias.
            log_density: (B, K) optional log-density per patch.

        Returns:
            (B, K, D) attention output.
        """
        B, K, _D = x.shape

        # QKV projection and reshape to (B, n_heads, K, head_dim)
        qkv = self.to_qkv(x).reshape(B, K, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)  # (B, H, K, d)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Optional QK normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product: (B, H, K, K)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative temporal bias: (B, H, K, K)
        attn = attn + self._bucketize_centers(centers)

        # Add density bias: (B, H, K, 1) broadcasts over key dim
        if self.use_density_bias and log_density is not None:
            attn = attn + self._density_bias(log_density)

        # Softmax + dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum: (B, H, K, d) -> (B, K, D)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, K, self.dim)
        return self.to_out(out)


# -----------------------------------------------------------------------
#  Cross-Attention (query attends to text tokens)
# -----------------------------------------------------------------------


class TextCrossAttention(nn.Module):
    """Standard multi-head cross-attention from time-series to text tokens.

    Queries come from the TS hidden state; keys and values come from the
    projected text token sequence.
    """

    def __init__(self, dim: int, text_dim: int, n_heads: int = 8,
        dropout: float = 0.0, qk_norm: bool = False, norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(text_dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """Attend from TS queries to text key/value pairs.

        Args:
            x: (B, K, D) query features (TS hidden state).
            text_tokens: (B, M, text_dim) text key/value features.

        Returns:
            (B, K, D) cross-attention output.
        """
        B, K, _D = x.shape
        M = text_tokens.shape[1]
        q = self.to_q(x).reshape(B, K, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        kv = self.to_kv(text_tokens).reshape(B, M, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, K, self.dim)
        return self.to_out(out)


# -----------------------------------------------------------------------
#  TextTemporalDiTBlock
# -----------------------------------------------------------------------


class TextTemporalDiTBlock(nn.Module):
    """Pre-norm DiT block with adaLN-Zero conditioning.

    Each block applies three sub-layers with adaLN-Zero modulation from
    the global conditioning signal:

    1. Self-attention path:
       modulate(norm1(x), shift1, scale1)
       -> RelativeTemporalSelfAttention
       -> x + gate1 * attn_out
    2. Cross-attention path:
       norm2(x) -> TextCrossAttention -> x + ca_out
    3. MLP path:
       modulate(norm3(x), shift2, scale2) -> MLP -> x + gate2 * mlp_out

    The adaLN projection produces six modulation parameters from the
    conditioning vector::
        SiLU -> Linear(dim, 6 * dim) -> [shift1, scale1, gate1,
                                           shift2, scale2, gate2]

    The last linear layer is zero-initialized so that all modulation
    parameters start at zero (identity self-attention, zero gate),
    following the adaLN-Zero recipe from the DiT paper.
    """

    def __init__(self, dim: int, text_dim: int, n_heads: int = 8,
        mlp_expand: int = 4, num_buckets: int = 32, max_rel_dist: float = 1.0,
        use_density_bias: bool = True, dropout: float = 0.0,
        qk_norm: bool = False, norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim = dim

        # adaLN-Zero projection: produces 6 modulation params
        self.ada_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada_proj[1].weight)
        nn.init.zeros_(self.ada_proj[1].bias)

        # Norms (no learnable affine; adaLN provides modulation)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)

        self.self_attn = RelativeTemporalSelfAttention(
            dim=dim, n_heads=n_heads, num_buckets=num_buckets,
            max_rel_dist=max_rel_dist, use_density_bias=use_density_bias,
            dropout=dropout, qk_norm=qk_norm, norm_eps=norm_eps)

        self.cross_attn = TextCrossAttention(
            dim=dim, text_dim=text_dim, n_heads=n_heads,
            dropout=dropout, qk_norm=qk_norm, norm_eps=norm_eps)

        mlp_hidden = dim * mlp_expand
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim))

    def forward(self, x: torch.Tensor, text_tokens: torch.Tensor,
        cond: torch.Tensor, centers: torch.Tensor,
        log_density: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one adaLN-Zero DiT block: self-attn + cross-attn + MLP.

        Args:
            x: (B, K, D) TS hidden features.
            text_tokens: (B, M, text_dim) text token features.
            cond: (B, D) global conditioning vector (from adaLN MLP).
            centers: (B, K) patch center positions in [0, 1].
            log_density: (B, K) optional log-density per patch.

        Returns:
            (B, K, D) updated TS features.
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.ada_proj(cond).chunk(6, dim=-1))

        # 1. Self-attention path
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.self_attn(h, centers=centers, log_density=log_density)
        x = x + gate1.unsqueeze(1) * h

        # 2. Cross-attention path
        h = self.norm2(x)
        h = self.cross_attn(h, text_tokens)
        x = x + h

        # 3. MLP path
        h = modulate(self.norm3(x), shift2, scale2)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x


# -----------------------------------------------------------------------
#  T2PDenoiser  (full DiT denoiser)
# -----------------------------------------------------------------------


class T2PDenoiser(nn.Module):
    """Patch Latent Diffusion DiT denoiser for time series generation.

    Predicts noise (epsilon-prediction) or velocity (v-prediction) from
    noisy patch latents, conditioned on diffusion timestep, text tokens,
    and patch metadata from the TextToPatchPlanner.

    Architecture::
        t (int) -> sinusoidal_embed -> MLP -> timestep_emb (B, D)
        cat[timestep_emb, pooled_text] -> global_cond_mlp -> cond (B, D)
        z_t -> input_proj -> (B, K, D)
        for each block:
            x = TextTemporalDiTBlock(x, text_tokens, cond, centers, log_density)
        x -> final_norm -> output_proj -> prediction

    Both prediction_type='epsilon' and prediction_type='v' are
    supported.  The v-prediction parameterization uses::
        v_t = alpha_t * epsilon - sigma_t * x_0
    so that at inference time::
        x_0 = alpha_t * z_t - sigma_t * v_pred
    """

    def __init__(self, input_dim: int, output_dim: int, text_dim: int,
        hidden_dim: int = 256, n_heads: int = 8, n_layers: int = 8,
        mlp_expand: int = 4, num_buckets: int = 32, max_rel_dist: float = 1.0,
        use_density_bias: bool = True, dropout: float = 0.0,
        qk_norm: bool = False, norm_eps: float = 1e-5,
        sinusoidal_dim: int = 256, prediction_type: str = "epsilon",
    ) -> None:
        """
        Args:
            input_dim: dimension of input noisy patch latents.
            output_dim: dimension of output prediction (noise or velocity).
            text_dim: dimension of text token features.
            hidden_dim: hidden dimension of the DiT trunk.
            n_heads: number of attention heads per block.
            n_layers: number of TextTemporalDiTBlock layers.
            mlp_expand: MLP expansion ratio within each block.
            num_buckets: number of relative position buckets.
            max_rel_dist: maximum relative distance for bucketization.
            use_density_bias: whether to use density bias in self-attention.
            dropout: dropout probability.
            qk_norm: whether to apply QK normalization.
            norm_eps: epsilon for LayerNorm.
            sinusoidal_dim: dimension of sinusoidal timestep embedding.
            prediction_type: 'epsilon' or 'v'.
        """
        super().__init__()
        assert prediction_type in ("epsilon", "v")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.prediction_type = prediction_type

        # Timestep embedding: sinusoidal + MLP
        self.sinusoidal_dim = sinusoidal_dim
        self.timestep_mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

        # Input projection (if dimensions differ)
        if input_dim != hidden_dim:
            self.input_proj: nn.Module = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Global conditioning MLP
        self.global_cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim + text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

        # DiT blocks
        self.blocks = nn.ModuleList([
            TextTemporalDiTBlock(dim=hidden_dim, text_dim=text_dim,
                n_heads=n_heads, mlp_expand=mlp_expand, num_buckets=num_buckets,
                max_rel_dist=max_rel_dist, use_density_bias=use_density_bias,
                dropout=dropout, qk_norm=qk_norm, norm_eps=norm_eps)
            for _ in range(n_layers)])

        # Output head
        self.final_norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _timestep_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Convert integer timesteps to conditioning embeddings via sinusoidal + MLP."""
        emb = _get_sinusoidal_embedding(t.float(), self.sinusoidal_dim)
        return self.timestep_mlp(emb)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
        text_tokens: torch.Tensor, meta: torch.Tensor,
        pooled_text: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise or velocity from noisy patch latents.

        Args:
            z_t: (B, K, input_dim) noisy patch latent tokens.
            t: (B,) integer diffusion timesteps.
            text_tokens: (B, M, text_dim) encoded text token sequence.
            meta: (B, K, 9) patch metadata from the planner with
                channels [start, end, center, length, log_length,
                mass, density, log_density, order].
            pooled_text: (B, text_dim) pooled text embedding.

        Returns:
            (B, K, output_dim) predicted noise or velocity.
        """
        # Extract patch metadata channels
        centers = meta[..., 2]       # (B, K) center positions
        log_density = meta[..., 7]   # (B, K) log-density

        # Timestep embedding
        timestep_emb = self._timestep_embed(t)

        # Global conditioning
        cond = self.global_cond_mlp(
            torch.cat([timestep_emb, pooled_text], dim=-1))

        # Input projection
        x = self.input_proj(z_t)

        # DiT blocks
        for block in self.blocks:
            x = block(x=x, text_tokens=text_tokens, cond=cond,
                centers=centers, log_density=log_density)

        # Output
        x = self.final_norm(x)
        return self.output_proj(x)

