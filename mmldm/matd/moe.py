"""Adaptive Semantic-Causal Temporal Mixture-of-Experts for MATD.

This file is a drop-in replacement for ``moe.py``.  It keeps the existing
MATD call signature:

    z_out, router_p, prior_p = moe(z, meta, text_cond, t_emb, causal_feat)

while upgrading the previous hand-written MLP expert bank into a sparse,
noisy top-k MoE layer with task-aware temporal, semantic, causal and diffusion
routing priors.

Design goals
------------
1. Standard MoE baseline mechanics:
   - noisy top-k routing;
   - capacity factor and token dropping for overloaded experts;
   - load-balance / router z-loss / prior-KL auxiliary losses;
   - sparse expert dispatch via per-expert gather + index_add.

2. MATD-specific inductive bias:
   - routing uses adaptive patch metadata: length, density, center/order;
   - routing uses SCCI text condition ``u``;
   - routing uses patch-aligned causal features;
   - routing is timestep-aware for diffusion training;
   - experts are specialized for time-series generation roles.

3. Backward compatibility:
   - returns ``(z_out, router_p, prior_p)`` as before;
   - stores richer differentiable auxiliary losses in
     ``self.last_aux_losses`` for optional use in MATDModel / trainer.

The returned ``router_p`` is the dense router distribution after adding the
structured prior.  Actual expert computation is sparse and uses only top-k
experts per patch token.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _cv_squared(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Squared coefficient of variation used by classic sparse MoE losses."""
    if x.numel() <= 1:
        return x.new_tensor(0.0)
    mean = x.float().mean()
    var = x.float().var(unbiased=False)
    return var / (mean.pow(2) + eps)


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.clamp_min(eps).log()


class SwiGLUBlock(nn.Module):
    """LayerNorm -> SwiGLU -> projection block.

    It is still lightweight, but stronger than a plain Linear-GELU-Linear FFN.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        zero_out: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.up = nn.Linear(in_dim, hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, out_dim)
        if zero_out:
            _zero_module(self.down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(self.drop(F.silu(a) * b))


class TemporalPatchContextEncoder(nn.Module):
    """Shared local context encoder over the adaptive patch axis.

    Experts are dispatched sparsely per token.  To still let selected tokens
    know about their local temporal neighborhood, we compute a cheap shared
    depthwise-separable convolutional context before routing.
    """

    def __init__(
        self,
        dim: int,
        meta_dim: int = 9,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length padding")
        self.dim = dim
        self.meta_proj = nn.Linear(meta_dim, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.causal_proj = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.pwconv = nn.Conv1d(dim, dim, kernel_size=1)
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        meta: torch.Tensor,
        text_cond: torch.Tensor,
        t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            z
            + self.meta_proj(meta)
            + self.text_proj(text_cond)
            + self.causal_proj(causal_feat)
            + self.time_proj(t_emb)
        )
        h = self.norm(x).transpose(1, 2)  # (B, D, K)
        h = self.pwconv(self.dwconv(h)).transpose(1, 2)
        return self.out(h)


# ---------------------------------------------------------------------------
# MATD-specialized experts
# ---------------------------------------------------------------------------


class TrendExpert(nn.Module):
    """Long-span, low-density patch expert.

    Intended for slow monotonic trends / level shifts.  It receives shared
    local context and explicitly gates its update using patch length.
    """

    def __init__(self, dim: int, meta_dim: int = 9, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.meta_proj = nn.Linear(meta_dim, dim)
        self.block = SwiGLUBlock(dim * 2 + meta_dim, hidden, dim, dropout)
        self.length_gate = nn.Sequential(nn.Linear(1, dim), nn.Sigmoid())

    def forward(self, z, meta, text_cond, t_emb, causal_feat, context):
        length = meta[:, 3:4].clamp(0.0, 1.0)
        gate = self.length_gate(length)
        inp = torch.cat([z + self.meta_proj(meta), context, meta], dim=-1)
        return gate * self.block(inp)


class PeriodicExpert(nn.Module):
    """Seasonality / oscillation expert with Fourier patch-position features."""

    def __init__(
        self,
        dim: int,
        meta_dim: int = 9,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        n_freqs: int = 4,
    ) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        fourier_dim = n_freqs * 4  # sin/cos for center and order
        hidden = dim * hidden_mult
        self.fourier_proj = nn.Linear(fourier_dim, dim)
        self.text_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Sigmoid())
        self.block = SwiGLUBlock(dim * 3 + meta_dim, hidden, dim, dropout)

    def _fourier(self, meta: torch.Tensor) -> torch.Tensor:
        center = meta[:, 2:3].clamp(0.0, 1.0)
        order = meta[:, 8:9].clamp(0.0, 1.0)
        freqs = torch.arange(1, self.n_freqs + 1, device=meta.device, dtype=meta.dtype)
        c = 2.0 * math.pi * center * freqs
        o = 2.0 * math.pi * order * freqs
        return torch.cat([torch.sin(c), torch.cos(c), torch.sin(o), torch.cos(o)], dim=-1)

    def forward(self, z, meta, text_cond, t_emb, causal_feat, context):
        f = self.fourier_proj(self._fourier(meta))
        gate = self.text_gate(text_cond)
        inp = torch.cat([z + f, text_cond, context, meta], dim=-1)
        return gate * self.block(inp)


class EventExpert(nn.Module):
    """High-density / short-patch expert for spikes and abrupt changes."""

    def __init__(self, dim: int, meta_dim: int = 9, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.event_score = nn.Sequential(
            nn.Linear(meta_dim + dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.block = SwiGLUBlock(dim * 4 + meta_dim, hidden, dim, dropout)

    def forward(self, z, meta, text_cond, t_emb, causal_feat, context):
        score_in = torch.cat([meta, causal_feat], dim=-1)
        gate = self.event_score(score_in)
        inp = torch.cat([z, causal_feat, t_emb, context, meta], dim=-1)
        return gate * self.block(inp)


class SemanticGroundingExpert(nn.Module):
    """Text-grounding expert using FiLM modulation from SCCI text condition."""

    def __init__(self, dim: int, meta_dim: int = 9, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.film = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2))
        self.block = SwiGLUBlock(dim * 2 + meta_dim, hidden, dim, dropout)

    def forward(self, z, meta, text_cond, t_emb, causal_feat, context):
        gamma, beta = self.film(text_cond).chunk(2, dim=-1)
        h = z * (1.0 + torch.tanh(gamma)) + beta
        inp = torch.cat([h, context, meta], dim=-1)
        return self.block(inp)


class CausalDynamicsExpert(nn.Module):
    """Causal-mechanism expert guided by patch-aligned causal features."""

    def __init__(self, dim: int, meta_dim: int = 9, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.causal_gate = nn.Sequential(
            nn.LayerNorm(dim * 2 + meta_dim),
            nn.Linear(dim * 2 + meta_dim, dim),
            nn.Sigmoid(),
        )
        self.block = SwiGLUBlock(dim * 3 + meta_dim, hidden, dim, dropout)

    def forward(self, z, meta, text_cond, t_emb, causal_feat, context):
        gate = self.causal_gate(torch.cat([z, causal_feat, meta], dim=-1))
        inp = torch.cat([z * (1.0 + gate), causal_feat, context, meta], dim=-1)
        return gate * self.block(inp)


class DetailDenoisingExpert(nn.Module):
    """Timestep-aware detail expert for local numerical refinement."""

    def __init__(self, dim: int, meta_dim: int = 9, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.time_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Sigmoid())
        self.block = SwiGLUBlock(dim * 4 + meta_dim, hidden, dim, dropout)

    def forward(self, z, meta, text_cond, t_emb, causal_feat, context):
        gate = self.time_gate(t_emb)
        inp = torch.cat([z, text_cond, causal_feat, t_emb, meta], dim=-1)
        return gate * self.block(inp)


class GenericTemporalExpert(nn.Module):
    """Fallback expert when n_experts > the six named experts."""

    def __init__(self, dim: int, meta_dim: int = 9, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.block = SwiGLUBlock(dim * 5 + meta_dim, hidden, dim, dropout)

    def forward(self, z, meta, text_cond, t_emb, causal_feat, context):
        return self.block(torch.cat([z, text_cond, causal_feat, t_emb, context, meta], dim=-1))


# ---------------------------------------------------------------------------
# Sparse semantic-causal temporal MoE
# ---------------------------------------------------------------------------


class SemanticCausalTemporalMoE(nn.Module):
    """Sparse top-k MoE for MATD patch latents.

    Args:
        dim: latent dimension D.
        meta_dim: metadata dimension, normally 9.
        n_experts: number of experts.  Six named experts are provided by
            default; extra experts use a generic temporal expert.
        top_k: number of experts selected per token.
        hidden_mult: expert hidden width multiplier.
        router_hidden_mult: router hidden width multiplier.
        noisy_gating: add learned Gaussian noise to router logits in training.
        capacity_factor_train/eval: per-expert capacity multiplier.  A value
            >= 1.0 follows standard top-k MoE practice; set to None to disable
            dropping entirely.
        prior_scale: multiplier for the structured MATD prior logits.
        balance_weight, z_loss_weight, prior_kl_weight, smooth_weight,
            capacity_weight: weights for ``self.last_aux_losses['loss_moe']``.
        residual_scale: scales sparse expert delta before adding it to z.
    """

    def __init__(
        self,
        dim: int,
        meta_dim: int,
        n_experts: int = 6,
        top_k: int = 2,
        hidden_mult: int = 4,
        router_hidden_mult: int = 2,
        dropout: float = 0.0,
        noisy_gating: bool = True,
        capacity_factor_train: Optional[float] = 1.25,
        capacity_factor_eval: Optional[float] = 2.0,
        prior_scale: float = 1.0,
        balance_weight: float = 0.01,
        z_loss_weight: float = 1e-3,
        prior_kl_weight: float = 0.05,
        smooth_weight: float = 0.01,
        capacity_weight: float = 0.1,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if meta_dim <= 0:
            raise ValueError(f"meta_dim must be positive, got {meta_dim}")
        if n_experts <= 0:
            raise ValueError(f"n_experts must be positive, got {n_experts}")
        if top_k <= 0 or top_k > n_experts:
            raise ValueError(f"top_k must be in [1, n_experts], got top_k={top_k}, n_experts={n_experts}")
        if capacity_factor_train is not None and capacity_factor_train <= 0:
            raise ValueError("capacity_factor_train must be positive or None")
        if capacity_factor_eval is not None and capacity_factor_eval <= 0:
            raise ValueError("capacity_factor_eval must be positive or None")

        self.dim = dim
        self.meta_dim = meta_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.balance_weight = float(balance_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.prior_kl_weight = float(prior_kl_weight)
        self.smooth_weight = float(smooth_weight)
        self.capacity_weight = float(capacity_weight)
        self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale)))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        self.context = TemporalPatchContextEncoder(dim, meta_dim, dropout=dropout)

        route_in = dim * 5 + meta_dim  # z, context, text, t, causal, meta
        route_hidden = dim * router_hidden_mult
        self.router = nn.Sequential(
            nn.LayerNorm(route_in),
            nn.Linear(route_in, route_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden, n_experts),
        )
        self.noise_proj = nn.Sequential(
            nn.LayerNorm(route_in),
            nn.Linear(route_in, n_experts),
        )
        self.learned_prior = nn.Sequential(
            nn.LayerNorm(route_in),
            nn.Linear(route_in, route_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden, n_experts),
        )

        self.expert_prototypes = nn.Parameter(torch.randn(n_experts, dim) * 0.02)
        self.detail_time_proj = nn.Linear(dim, 1)
        self.heuristic_scale = nn.Parameter(torch.tensor(1.0))

        expert_bank = [
            TrendExpert(dim, meta_dim, hidden_mult, dropout),
            PeriodicExpert(dim, meta_dim, hidden_mult, dropout),
            EventExpert(dim, meta_dim, hidden_mult, dropout),
            SemanticGroundingExpert(dim, meta_dim, hidden_mult, dropout),
            CausalDynamicsExpert(dim, meta_dim, hidden_mult, dropout),
            DetailDenoisingExpert(dim, meta_dim, hidden_mult, dropout),
        ]
        while len(expert_bank) < n_experts:
            expert_bank.append(GenericTemporalExpert(dim, meta_dim, hidden_mult, dropout))
        self.experts = nn.ModuleList(expert_bank[:n_experts])

        self.last_aux_losses: Dict[str, torch.Tensor] = {}
        self.last_router_stats: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Input checks and routing priors
    # ------------------------------------------------------------------

    def _check_inputs(
        self,
        z: torch.Tensor,
        meta: torch.Tensor,
        text_cond: torch.Tensor,
        t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> Tuple[int, int, int, torch.Tensor]:
        if z.dim() != 3:
            raise ValueError(f"z must be (B,K,D), got {tuple(z.shape)}")
        B, K, D = z.shape
        if D != self.dim:
            raise ValueError(f"z dim={D} does not match self.dim={self.dim}")

        if meta.dim() == 2:
            meta = meta.unsqueeze(1).expand(-1, K, -1)
        if meta.dim() != 3 or meta.shape[0] != B or meta.shape[1] != K:
            raise ValueError(f"meta must be (B,K,C) or (B,C), got {tuple(meta.shape)}")
        if meta.shape[-1] < self.meta_dim:
            raise ValueError(f"meta has {meta.shape[-1]} channels, expected at least {self.meta_dim}")
        meta = meta[..., :self.meta_dim].to(device=z.device, dtype=z.dtype)

        for name, tensor in [
            ("text_cond", text_cond),
            ("t_emb", t_emb),
            ("causal_feat", causal_feat),
        ]:
            if tensor.shape != (B, K, D):
                raise ValueError(f"{name} must be {(B, K, D)}, got {tuple(tensor.shape)}")
        return B, K, D, meta

    def _structured_prior_logits(
        self,
        route_in: torch.Tensor,
        meta: torch.Tensor,
        text_cond: torch.Tensor,
        t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Learned + interpretable routing prior.

        Expert index convention for the first six experts:
          0 trend, 1 period, 2 event, 3 semantic, 4 causal, 5 detail.
        Extra experts receive only the learned prior/prototype affinity.
        """
        B, K, _ = text_cond.shape
        learned = self.learned_prior(route_in)

        # Prototype affinity lets experts specialize without hard-coded names.
        proto = F.normalize(self.expert_prototypes, dim=-1)
        text_n = F.normalize(text_cond.float(), dim=-1).to(text_cond.dtype)
        causal_n = F.normalize(causal_feat.float(), dim=-1).to(causal_feat.dtype)
        affinity = 0.5 * (text_n @ proto.T) + 0.5 * (causal_n @ proto.T)

        heuristic = learned.new_zeros(B, K, self.n_experts)
        if self.meta_dim >= 9:
            length = meta[..., 3].clamp(0.0, 1.0)
            density = meta[..., 6].clamp_min(1e-6)
            log_density = meta[..., 7]
        else:
            length = meta.new_full((B, K), 1.0 / max(K, 1))
            density = meta.new_ones(B, K)
            log_density = meta.new_zeros(B, K)

        causal_strength = causal_feat.float().norm(dim=-1).to(meta.dtype) / math.sqrt(self.dim)
        text_strength = text_cond.float().norm(dim=-1).to(meta.dtype) / math.sqrt(self.dim)
        detail_phase = torch.sigmoid(self.detail_time_proj(t_emb).squeeze(-1))

        # Robust per-batch normalization for score-like features.
        dens_z = (log_density - log_density.mean(dim=1, keepdim=True)) / (log_density.std(dim=1, keepdim=True) + 1e-5)
        cau_z = (causal_strength - causal_strength.mean(dim=1, keepdim=True)) / (causal_strength.std(dim=1, keepdim=True) + 1e-5)
        txt_z = (text_strength - text_strength.mean(dim=1, keepdim=True)) / (text_strength.std(dim=1, keepdim=True) + 1e-5)

        if self.n_experts >= 1:  # trend: long, lower-density patches
            heuristic[..., 0] = 2.0 * length - 0.25 * dens_z
        if self.n_experts >= 2:  # periodic: text/prototype affinity + non-event density
            heuristic[..., 1] = affinity[..., 1] - 0.15 * dens_z
        if self.n_experts >= 3:  # event: short, dense, causal-active patches
            heuristic[..., 2] = -2.0 * length + 0.5 * dens_z + 0.25 * cau_z
        if self.n_experts >= 4:  # semantic grounding
            heuristic[..., 3] = txt_z + affinity[..., 3]
        if self.n_experts >= 5:  # causal dynamics
            heuristic[..., 4] = cau_z + affinity[..., 4]
        if self.n_experts >= 6:  # fine detail / denoising stage
            heuristic[..., 5] = detail_phase + 0.10 * dens_z

        return learned + affinity + F.softplus(self.heuristic_scale).to(learned.dtype) * heuristic

    def _capacity(self, num_tokens: int, dtype: torch.dtype) -> Optional[int]:
        cf = self.capacity_factor_train if self.training else self.capacity_factor_eval
        if cf is None:
            return None
        return max(1, int(math.ceil(float(cf) * num_tokens * self.top_k / self.n_experts)))

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def _router_smoothness_loss(self, router_p: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Density-aware temporal router smoothness.

        Long / low-density neighboring patches should not route wildly
        differently.  Dense/event-like patches are allowed to switch experts.
        """
        if router_p.shape[1] <= 1:
            return router_p.new_tensor(0.0)
        diff = (router_p[:, 1:] - router_p[:, :-1]).pow(2).sum(dim=-1)
        if meta.shape[-1] >= 8:
            log_density = meta[..., 7]
            density_weight = torch.sigmoid(-0.5 * (log_density[:, 1:] + log_density[:, :-1]))
            length_weight = 0.5 * (meta[:, 1:, 3] + meta[:, :-1, 3]).clamp(0.0, 1.0)
            w = (density_weight + length_weight).detach()
            return (diff * w).mean()
        return diff.mean()

    def _set_aux_losses(
        self,
        router_p: torch.Tensor,
        prior_p: torch.Tensor,
        combined_logits: torch.Tensor,
        dispatch_mask: torch.Tensor,
        dropped: torch.Tensor,
        meta: torch.Tensor,
    ) -> None:
        # Flatten token dimensions for classic MoE balancing terms.
        B, K, E = router_p.shape
        N = B * K
        probs_f = router_p.reshape(N, E)
        prior_f = prior_p.reshape(N, E)

        importance = probs_f.sum(dim=0)                # soft expected usage
        load = dispatch_mask.float().sum(dim=0)        # hard top-k usage
        prob_frac = importance / max(N, 1)
        load_frac = load / load.sum().clamp_min(1.0)

        switch_aux = E * (prob_frac * load_frac).sum()
        cv_balance = _cv_squared(importance) + _cv_squared(load)
        load_balance = switch_aux + cv_balance

        router_z = torch.logsumexp(combined_logits.float(), dim=-1).pow(2).mean()
        prior_kl = (probs_f * (_safe_log(probs_f) - _safe_log(prior_f))).sum(dim=-1).mean()
        entropy = -(probs_f * _safe_log(probs_f)).sum(dim=-1).mean()
        smooth = self._router_smoothness_loss(router_p, meta)
        drop_rate = dropped.to(router_p.dtype) / max(float(N * self.top_k), 1.0)

        loss_moe = (
            self.balance_weight * load_balance
            + self.z_loss_weight * router_z
            + self.prior_kl_weight * prior_kl
            + self.smooth_weight * smooth
            + self.capacity_weight * drop_rate
        )

        self.last_aux_losses = {
            "loss_moe": loss_moe,
            "loss_moe_balance": load_balance,
            "loss_moe_switch_aux": switch_aux,
            "loss_moe_cv_balance": cv_balance,
            "loss_moe_router_z": router_z,
            "loss_moe_prior_kl": prior_kl,
            "loss_moe_entropy": entropy,
            "loss_moe_router_smooth": smooth,
            "loss_moe_capacity_drop": drop_rate,
        }
        self.last_router_stats = {
            "importance": importance.detach(),
            "load": load.detach(),
            "prob_frac": prob_frac.detach(),
            "load_frac": load_frac.detach(),
            "capacity_drop_rate": drop_rate.detach(),
            "router_entropy": entropy.detach(),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        meta: torch.Tensor,
        text_cond: torch.Tensor,
        t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sparse temporal-semantic-causal MoE.

        Args:
            z:           (B, K, D) conditioned patch tokens from SCCI.
            meta:        (B, 9) pooled metadata or (B, K, 9) per-patch metadata.
                         For best routing, pass per-patch metadata when possible.
            text_cond:   (B, K, D) SCCI text-conditioned patch features ``u``.
            t_emb:       (B, K, D) timestep embedding expanded over patches.
            causal_feat: (B, K, D) patch-aligned causal feature.

        Returns:
            z_out:    (B, K, D), residual MoE-refined tokens.
            router_p: (B, K, E), dense router probabilities after prior fusion.
            prior_p:  (B, K, E), structured prior probabilities.
        """
        B, K, D, meta = self._check_inputs(z, meta, text_cond, t_emb, causal_feat)
        N = B * K
        E = self.n_experts

        context = self.context(z, meta, text_cond, t_emb, causal_feat)
        route_in = torch.cat([z, context, text_cond, t_emb, causal_feat, meta], dim=-1)

        router_logits = self.router(route_in)
        if self.noisy_gating and self.training:
            noise_std = F.softplus(self.noise_proj(route_in)) + 1e-3
            router_logits = router_logits + torch.randn_like(router_logits) * noise_std

        prior_logits = self._structured_prior_logits(route_in, meta, text_cond, t_emb, causal_feat)
        combined_logits = router_logits + F.softplus(self.prior_scale).to(router_logits.dtype) * prior_logits

        router_p = F.softmax(combined_logits.float(), dim=-1).to(z.dtype)
        prior_p = F.softmax(prior_logits.float(), dim=-1).to(z.dtype)

        topk_logits, topk_idx = combined_logits.topk(self.top_k, dim=-1)
        topk_gates = F.softmax(topk_logits.float(), dim=-1).to(z.dtype)

        flat_z = z.reshape(N, D)
        flat_meta = meta.reshape(N, self.meta_dim)
        flat_text = text_cond.reshape(N, D)
        flat_t = t_emb.reshape(N, D)
        flat_causal = causal_feat.reshape(N, D)
        flat_context = context.reshape(N, D)
        flat_idx = topk_idx.reshape(N, self.top_k)
        flat_gate = topk_gates.reshape(N, self.top_k)

        delta = z.new_zeros(N, D)
        dispatch_mask = z.new_zeros(N, E)
        capacity = self._capacity(N, z.dtype)
        dropped = z.new_tensor(0.0)

        for e_idx, expert in enumerate(self.experts):
            pos = (flat_idx == e_idx).nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue
            token_ids = pos[:, 0]
            kth_ids = pos[:, 1]
            gates = flat_gate[token_ids, kth_ids]

            if capacity is not None and token_ids.numel() > capacity:
                keep_val, keep_order = gates.topk(capacity, dim=0)
                dropped = dropped + (token_ids.numel() - capacity)
                token_ids = token_ids[keep_order]
                gates = keep_val

            expert_out = expert(
                flat_z[token_ids],
                flat_meta[token_ids],
                flat_text[token_ids],
                flat_t[token_ids],
                flat_causal[token_ids],
                flat_context[token_ids],
            )
            weighted = expert_out.to(delta.dtype) * gates.unsqueeze(-1)
            delta.index_add_(0, token_ids, weighted)
            dispatch_mask[token_ids, e_idx] = 1.0

        self._set_aux_losses(router_p, prior_p, combined_logits, dispatch_mask, dropped, meta)

        z_out = z + self.residual_scale.to(z.dtype) * delta.reshape(B, K, D)
        return z_out, router_p, prior_p


# ---------------------------------------------------------------------------
# Standalone auxiliary loss helper
# ---------------------------------------------------------------------------


def moe_losses(
    router_p: torch.Tensor,
    prior_p: torch.Tensor,
    *,
    balance_weight: float = 0.01,
    kl_weight: float = 0.05,
    entropy_target: Optional[float] = None,
    entropy_weight: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Backward-compatible MoE losses for code that cannot access the module.

    Prefer using ``module.last_aux_losses`` from ``SemanticCausalTemporalMoE``
    because it also includes hard-dispatch load, z-loss and capacity drop rate.
    """
    if router_p.shape != prior_p.shape:
        raise ValueError(f"router_p and prior_p must have same shape, got {router_p.shape} vs {prior_p.shape}")
    E = router_p.shape[-1]
    probs = router_p.reshape(-1, E)
    prior = prior_p.reshape(-1, E)
    usage = probs.mean(dim=0)
    ideal = torch.full_like(usage, 1.0 / E)
    balance = (usage - ideal).pow(2).sum()
    kl = (probs * (_safe_log(probs) - _safe_log(prior))).sum(dim=-1).mean()
    ent = -(probs * _safe_log(probs)).sum(dim=-1).mean()
    if entropy_target is None:
        entropy_target = math.log(max(E, 1)) * 0.5
    ent_loss = (ent - float(entropy_target)).pow(2)
    loss = balance_weight * balance + kl_weight * kl + entropy_weight * ent_loss
    return {
        "loss_moe": loss,
        "loss_moe_balance": balance,
        "loss_moe_prior_kl": kl,
        "loss_moe_entropy_target": ent_loss,
        "moe_entropy": ent,
    }
