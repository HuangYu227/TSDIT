"""Text-to-Patch Planner for MATD.

Predicts a patch-level meta-layout from text tokens.  Each of the *K*
learnable patch queries attends over the text sequence and produces a
9-dimensional metadata vector::

    meta_pred (B, K, 9) = [start, end, center, length, log_length,
                            mass, density, log_density, order]

The layout is soft: lengths and masses sum to 1 across patches so that
downstream modules can treat them as probability-like weights.

Reference: MATD framework design doc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Core Planner
# ---------------------------------------------------------------------------


@dataclass
class PlanOutput:
    """Latent temporal plan plus the layout metadata required downstream."""

    plan_tokens: torch.Tensor
    plan_global: torch.Tensor
    meta: torch.Tensor
    plan_mu: Optional[torch.Tensor] = None
    plan_logvar: Optional[torch.Tensor] = None
    plan_kl: Optional[torch.Tensor] = None


def build_canonical_meta(
    batch_size: int,
    n_patches: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a deterministic uniform patch layout with MATD's 9 meta channels."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_patches <= 0:
        raise ValueError(f"n_patches must be positive, got {n_patches}")

    length = torch.full((batch_size, n_patches), 1.0 / n_patches, device=device, dtype=dtype)
    end = torch.cumsum(length, dim=-1)
    start = end - length
    center = 0.5 * (start + end)
    log_length = torch.log(length.clamp_min(1e-8))
    mass = length.clone()
    density = mass / length.clamp_min(1e-8)
    log_density = torch.log(density.clamp_min(1e-8))
    order = torch.linspace(0.0, 1.0, n_patches, device=device, dtype=dtype)
    order = order.unsqueeze(0).expand(batch_size, -1)
    return torch.stack(
        [start, end, center, length, log_length, mass, density, log_density, order],
        dim=-1,
    )


class TextLatentPlanGenerator(nn.Module):
    """PlanFormer: text-conditioned stochastic latent temporal planning.

    The module deliberately avoids regressing patch geometry from text.  Plan
    queries repeatedly cross-attend to text, self-attend with temporal relative
    bias, and are modulated by pooled text through AdaLN.  The final plan is a
    reparameterized latent distribution, not a shallow deterministic MLP.
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_patches: int = 16,
        dropout: float = 0.1,
        depth: int = 3,
        mlp_ratio: float = 4.0,
        num_rel_buckets: int = 32,
        stochastic: bool = True,
    ) -> None:
        super().__init__()
        self._n_patches = n_patches
        self.hidden_dim = hidden_dim
        self.stochastic = bool(stochastic)
        self.patch_queries = nn.Parameter(torch.randn(n_patches, hidden_dim) * 0.02)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_proj = nn.Linear(16, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _PlanFormerBlock(
                    dim=hidden_dim,
                    text_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    num_rel_buckets=num_rel_buckets,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.to_mu = nn.Linear(hidden_dim, hidden_dim)
        self.to_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        modules = [self.text_proj, self.time_proj, self.to_mu, self.to_logvar]
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -4.0)

    @staticmethod
    def _time_features(meta: torch.Tensor) -> torch.Tensor:
        centers = meta[..., 2]
        lengths = meta[..., 3]
        freqs = torch.arange(1, 5, device=meta.device, dtype=meta.dtype)
        angles = centers.unsqueeze(-1) * freqs.view(1, 1, -1) * math.pi
        return torch.cat(
            [
                meta[..., 0:1],
                meta[..., 1:2],
                centers.unsqueeze(-1),
                lengths.unsqueeze(-1),
                torch.sin(angles),
                torch.cos(angles),
                meta[..., 5:6],
                meta[..., 8:9],
                meta[..., 6:7],
                meta[..., 7:8],
            ],
            dim=-1,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        pooled_text: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
        n_patches: int | None = None,
    ) -> PlanOutput:
        B = text_tokens.size(0)
        K = n_patches if n_patches is not None else self._n_patches
        if K > self._n_patches:
            raise ValueError(
                f"Requested n_patches={K} exceeds planner capacity "
                f"{self._n_patches}. Increase max_tokens in config."
            )

        meta = build_canonical_meta(B, K, text_tokens.device, text_tokens.dtype)
        text_hidden = self.text_proj(text_tokens)
        if pooled_text is None:
            if text_padding_mask is None:
                pooled_text = text_tokens.mean(dim=1)
            else:
                valid = (~text_padding_mask).to(text_tokens.dtype).unsqueeze(-1)
                pooled_text = (text_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        cond = self.cond_proj(pooled_text)
        queries = self.patch_queries[:K].unsqueeze(0).expand(B, -1, -1)
        hidden = queries + self.time_proj(self._time_features(meta))
        for block in self.blocks:
            hidden = block(hidden, text_hidden, cond, meta, text_padding_mask)
        hidden = self.final_norm(hidden)
        plan_mu = self.to_mu(hidden)
        plan_logvar = self.to_logvar(hidden).clamp(min=-8.0, max=4.0)
        if self.stochastic and self.training:
            eps = torch.randn_like(plan_mu)
            plan_tokens = plan_mu + eps * torch.exp(0.5 * plan_logvar)
        else:
            plan_tokens = plan_mu
        plan_kl = 0.5 * (plan_mu.pow(2) + plan_logvar.exp() - 1.0 - plan_logvar).mean()
        if pooled_text is None:
            plan_global = plan_tokens.mean(dim=1)
        else:
            plan_global = plan_tokens.mean(dim=1) + cond
        plan_global = self.global_norm(plan_global)
        return PlanOutput(
            plan_tokens=plan_tokens,
            plan_global=plan_global,
            meta=meta,
            plan_mu=plan_mu,
            plan_logvar=plan_logvar,
            plan_kl=plan_kl,
        )


class _AdaLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.cond(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _RelativePlanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        num_rel_buckets: int = 32,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.num_rel_buckets = num_rel_buckets
        self.qkv = nn.Linear(dim, dim * 3)
        self.rel_bias = nn.Embedding(num_rel_buckets, n_heads)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _relative_bucket(self, meta: torch.Tensor) -> torch.Tensor:
        centers = meta[..., 2].clamp(0.0, 1.0)
        delta = centers[:, :, None] - centers[:, None, :]
        bucket = ((delta + 1.0) * 0.5 * (self.num_rel_buckets - 1)).round().long()
        return bucket.clamp(0, self.num_rel_buckets - 1)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        B, K, _ = x.shape
        qkv = self.qkv(x).view(B, K, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        bias = self.rel_bias(self._relative_bucket(meta)).permute(0, 3, 1, 2)
        attn = torch.softmax(logits + bias, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, K, self.dim)
        return self.out(out)


class _SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Linear(dim, hidden * 2)
        self.down = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(self.drop(value * F.silu(gate)))


class _PlanFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        text_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        num_rel_buckets: int = 32,
    ) -> None:
        super().__init__()
        self.cross_norm = _AdaLayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            kdim=text_dim,
            vdim=text_dim,
        )
        self.self_norm = _AdaLayerNorm(dim)
        self.self_attn = _RelativePlanSelfAttention(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            num_rel_buckets=num_rel_buckets,
        )
        self.ffn_norm = _AdaLayerNorm(dim)
        self.ffn = _SwiGLUFFN(dim, int(dim * mlp_ratio), dropout=dropout)
        self.res_scale = nn.Parameter(torch.ones(3) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        cond: torch.Tensor,
        meta: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_cross = self.cross_norm(x, cond)
        cross, _ = self.cross_attn(
            x_cross,
            text,
            text,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )
        x = x + self.res_scale[0] * cross
        x = x + self.res_scale[1] * self.self_attn(self.self_norm(x, cond), meta)
        x = x + self.res_scale[2] * self.ffn(self.ffn_norm(x, cond))
        return x


class TextToPatchPlanner(nn.Module):
    """Predict patch layout metadata from text tokens.

    Architecture::

        text_tokens --[text_proj]--> text_hidden
        patch_queries (learnable) --+
                                    |-- MHA --> LayerNorm
                                              |
                                   len_head --> length (sum=1)
                                   info_head --> mass (sum=1), density (>0)

    The 9 output channels per patch are assembled as:

        end      = cumsum(length)
        start    = end - length
        center   = 0.5 * (start + end)
        log_len  = log(length)
        mass     = softplus(raw_mass), normalised to sum=1
        density  = softplus(raw_density) + eps
        log_dens = log(density)
        order    = linspace(0, 1, K)  (constant, broadcast)
    """

    def __init__(
        self,
        text_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_patches: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._n_patches = n_patches
        self.hidden_dim = hidden_dim

        # Learnable patch queries
        self.patch_queries = nn.Parameter(
            torch.randn(n_patches, hidden_dim) * 0.02
        )

        # Project text tokens to hidden dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-attention: queries attend over text keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Length head: hidden -> 1 (length per patch)
        self.len_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Info head: hidden -> 2 (raw mass, raw density)
        self.info_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in [self.text_proj, *self.len_head, *self.info_head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        n_patches: int | None = None,
    ) -> torch.Tensor:
        """Predict patch metadata from text.

        Args:
            text_tokens: (B, M, text_dim) encoded text sequence.
            text_padding_mask: (B, M) True where padded (optional).
            n_patches: Override number of patches (K).  When *None* the
                value from ``__init__`` is used.

        Returns:
            meta_pred: (B, K, 9) with channels
                ``[start, end, center, length, log_length,
                   mass, density, log_density, order]``.
        """
        B = text_tokens.size(0)
        K = n_patches if n_patches is not None else self._n_patches
        if K > self._n_patches:
            raise ValueError(
                f"Requested n_patches={K} exceeds planner capacity "
                f"{self._n_patches}. Increase max_tokens in config."
            )
        device = text_tokens.device

        # Project text to hidden dim
        text_hidden = self.text_proj(text_tokens)  # (B, M, H)

        # key_padding_mask for MHA (True = ignore)
        key_padding_mask = text_padding_mask  # may be None

        # Expand learnable queries for batch, slice to K
        queries = self.patch_queries[:K, :].unsqueeze(0).expand(B, -1, -1)  # (B, K, H)

        # Cross-attention + residual + norm
        attn_out, _ = self.cross_attn(
            query=queries,
            key=text_hidden,
            value=text_hidden,
            key_padding_mask=key_padding_mask,
        )  # (B, K, H)
        hidden = self.norm(queries + attn_out)  # (B, K, H)

        # --- Length prediction ---
        raw_len = self.len_head(hidden).squeeze(-1)  # (B, K)
        length = F.softplus(raw_len)  # positive
        length = length / (length.sum(dim=-1, keepdim=True) + 1e-8)  # sum=1

        log_length = torch.log(length + 1e-8)  # (B, K)

        # Cumulative sums for start/end/center
        end = torch.cumsum(length, dim=-1)  # (B, K)
        start = end - length  # (B, K)
        center = 0.5 * (start + end)  # (B, K)

        # --- Mass & density prediction ---
        info_raw = self.info_head(hidden)  # (B, K, 2)
        raw_mass, raw_density_hint = info_raw.unbind(dim=-1)

        # Keep planner metadata semantically consistent with the oracle
        # tokenizer, where density is defined as mass / length.  The second
        # info head output is retained as a small learned hint for mass so old
        # checkpoints remain shape-compatible while all metadata channels stay
        # self-consistent.
        mass = F.softplus(raw_mass + 0.1 * torch.tanh(raw_density_hint))  # positive
        mass = mass / (mass.sum(dim=-1, keepdim=True) + 1e-8)  # sum=1

        density = mass / (length + 1e-6)
        log_density = torch.log(density + 1e-8)

        # --- Order: constant ramp ---
        order = torch.linspace(0.0, 1.0, K, device=device)  # (K,)
        order = order.unsqueeze(0).expand(B, -1)  # (B, K)

        # Assemble (B, K, 9)
        meta_pred = torch.stack(
            [start, end, center, length, log_length,
             mass, density, log_density, order],
            dim=-1,
        )  # (B, K, 9)
        return meta_pred


# ---------------------------------------------------------------------------
#  Planner Loss
# ---------------------------------------------------------------------------


# Channel indices
_START, _END, _CENTER, _LENGTH, _LOG_LEN = range(5)
_MASS, _DENSITY, _LOG_DENS, _ORDER = 5, 6, 7, 8


class PlannerLoss(nn.Module):
    """Loss for the Text-to-Patch Planner.

    Components:

    - **Smooth L1** on center, length, mass, log_density against GT.
    - **CDF boundary loss** on end positions: penalises deviation of the
      predicted cumulative end positions from the ground-truth CDF.

    All losses are returned as a dict of scalar tensors so the caller can
    weight them freely.
    """

    def __init__(self, beta: float = 1.0) -> None:
        """
        Args:
            beta: SmoothL1 beta parameter (transition point between
                  L2 and L1 regions).
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        meta_pred: torch.Tensor,
        meta_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute planner losses.

        Args:
            meta_pred:   (B, K, 9) predicted metadata.
            meta_target: (B, K, 9) ground-truth metadata.

        Returns:
            Dict with keys loss_center, loss_length, loss_mass,
            loss_log_density, loss_cdf, and loss_planner (sum).
        """
        # Individual Smooth L1 terms
        loss_center = F.smooth_l1_loss(
            meta_pred[..., _CENTER], meta_target[..., _CENTER], beta=self.beta
        )
        loss_length = F.smooth_l1_loss(
            meta_pred[..., _LENGTH], meta_target[..., _LENGTH], beta=self.beta
        )
        loss_mass = F.smooth_l1_loss(
            meta_pred[..., _MASS], meta_target[..., _MASS], beta=self.beta
        )
        loss_log_dens = F.smooth_l1_loss(
            meta_pred[..., _LOG_DENS], meta_target[..., _LOG_DENS],
            beta=self.beta
        )

        # CDF boundary loss on end positions
        end_pred = meta_pred[..., _END]      # (B, K)
        end_target = meta_target[..., _END]  # (B, K)
        loss_cdf = F.smooth_l1_loss(
            end_pred, end_target, beta=self.beta
        )

        # Aggregate
        loss_planner = (
            loss_center + loss_length + loss_mass + loss_log_dens + loss_cdf
        )

        return {
            "loss_center": loss_center,
            "loss_length": loss_length,
            "loss_mass": loss_mass,
            "loss_log_density": loss_log_dens,
            "loss_cdf": loss_cdf,
            "loss_planner": loss_planner,
        }
