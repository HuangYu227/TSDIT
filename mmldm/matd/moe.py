"""Semantic-Causal Temporal Mixture-of-Experts (SCT-MoE).

Six specialised experts with a semantic router that combines data-driven
routing with a causal prior.  All expert dispatch is vectorised via boolean
masks -- no Python for-loop over batch/token pairs.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrendExpert(nn.Module):
    """Captures low-frequency trend components.
    
    Architecture: LayerNorm -> Linear(dim, 4*dim) -> GELU -> Linear(4*dim, dim).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim))

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor,
        text_cond: torch.Tensor, t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(self.norm(z))


class PeriodExpert(nn.Module):
    """Models periodic / seasonal patterns via frequency projection.
    
    Architecture: freq_proj MLP over (z, text_cond) with residual.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.freq_proj = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor,
        text_cond: torch.Tensor, t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.freq_proj(torch.cat([z, text_cond], dim=-1))
        return z + h


class EventExpert(nn.Module):
    """Detects and models discrete events via gated activation.
    
    Architecture: gate + up(2*dim) + down with GELU, sigmoid-gated.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim * 2)
        self.up = nn.Linear(dim * 2, dim * 2)
        self.down = nn.Linear(dim * 2, dim)

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor,
        text_cond: torch.Tensor, t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.cat([z, t_emb], dim=-1)
        return self.down(torch.sigmoid(self.gate(inp)) * F.gelu(self.up(inp)))


class TextGroundingExpert(nn.Module):
    """Injects text semantics into the diffusion latent.
    
    Architecture: LayerNorm -> MLP over (z, text_cond) with residual.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor,
        text_cond: torch.Tensor, t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.mlp(torch.cat([self.norm(z), text_cond], dim=-1))
        return z + h


class CausalDynamicsExpert(nn.Module):
    """Models causal dynamics by conditioning on causal features.
    
    Architecture: LayerNorm -> MLP over (z, causal_feat) with residual.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor,
        text_cond: torch.Tensor, t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.mlp(torch.cat([self.norm(z), causal_feat], dim=-1))
        return z + h


class DetailExpert(nn.Module):
    """Sharpens fine-grained details via sigmoid-gated activation.
    
    Architecture: sigmoid(norm(z)) * silu(up(norm(z))) -> down.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim * 4)
        self.up = nn.Linear(dim, dim * 4)
        self.down = nn.Linear(dim * 4, dim)

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor,
        text_cond: torch.Tensor, t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm(z)
        return self.down(torch.sigmoid(self.gate(h)) * F.silu(self.up(h)))


# ---------------------------------------------------------------------------
# Semantic-Causal Temporal MoE
# ---------------------------------------------------------------------------


class SemanticCausalTemporalMoE(nn.Module):
    """Mixture-of-Experts layer with semantic routing and causal prior.

    The router combines a data-driven component (from pooled token statistics)
    with a prior derived from meta-data and text.  Expert dispatch is fully
    vectorised via boolean masks.

    Returns ``(z + delta_z, router_p, prior_p)`` where
    ``router_p`` and ``prior_p``
    are the routing and prior probability distributions over experts.
    """

    def __init__(
        self, dim: int, meta_dim: int,
        n_experts: int = 6, top_k: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.meta_dim = meta_dim
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Sequential(
            nn.Linear(dim * 4 + meta_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, n_experts))

        self.prior_proj = nn.Sequential(
            nn.Linear(meta_dim + dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, n_experts))

        self.experts = nn.ModuleList([
            TrendExpert(dim),
            PeriodExpert(dim),
            EventExpert(dim),
            TextGroundingExpert(dim),
            CausalDynamicsExpert(dim),
            DetailExpert(dim)])

    def forward(
        self, z: torch.Tensor, meta: torch.Tensor,
        text_cond: torch.Tensor, t_emb: torch.Tensor,
        causal_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z:           (B, K, dim)  diffusion latent tokens.
            meta:        (B, meta_dim) metadata embedding.
            text_cond:   (B, K, dim)  text conditioning per token.
            t_emb:       (B, K, dim)  diffusion timestep embedding per token.
            causal_feat: (B, K, dim)  causal features per token.
        Returns:
            z_out, router_p, prior_p
        """
        B, K, D = z.shape

        meta_b = meta.unsqueeze(1).expand(-1, K, -1)
        router_in = torch.cat([z, meta_b, text_cond, t_emb, causal_feat], dim=-1)
        mean_z = z.mean(dim=1)
        mean_text = text_cond.mean(dim=1)
        prior_in = torch.cat([meta, mean_z, mean_text], dim=-1)

        router_logits: torch.Tensor = self.router(router_in)
        prior_logits: torch.Tensor = self.prior_proj(prior_in)
        prior_logits = prior_logits.unsqueeze(1).expand(-1, K, -1)

        router_p = F.softmax(router_logits, dim=-1)
        prior_p = F.softmax(prior_logits, dim=-1)
        combined_logits = router_logits + prior_logits
        combined_p = F.softmax(combined_logits, dim=-1)
        _, topk_idx = combined_p.topk(self.top_k, dim=-1)

        flat_z = z.reshape(B * K, D)
        flat_meta = meta_b.reshape(B * K, -1)
        flat_text = text_cond.reshape(B * K, D)
        flat_t = t_emb.reshape(B * K, D)
        flat_causal = causal_feat.reshape(B * K, D)
        delta_z = torch.zeros_like(flat_z)

        for e_idx in range(self.n_experts):
            mask = (topk_idx == e_idx).any(dim=-1).reshape(-1)
            if not mask.any():
                continue
            expert_out = self.experts[e_idx](
                flat_z[mask], flat_meta[mask],
                flat_text[mask], flat_t[mask],
                flat_causal[mask])
            delta_z = delta_z.masked_scatter_(
                mask.unsqueeze(-1).expand(-1, D), expert_out.to(delta_z.dtype))

        z_out = z + delta_z.reshape(B, K, D)
        return z_out, router_p, prior_p


# ---------------------------------------------------------------------------
# MoE regularisation losses
# ---------------------------------------------------------------------------


def moe_losses(
    router_p: torch.Tensor, prior_p: torch.Tensor,
    *, balance_weight: float = 0.01, kl_weight: float = 0.1,
    entropy_target: float = 1.5, entropy_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Compute MoE regularisation losses.
    Args:
        router_p:        (B, K, E) router probability distribution.
        prior_p:         (B, K, E) prior probability distribution.
        balance_weight:  Weight for the load-balance loss.
        kl_weight:       Weight for the prior KL divergence.
        entropy_target:  Target entropy for the router distribution.
        entropy_weight:  Weight for the entropy-target loss.
    Returns:
        Dictionary of named scalar losses.
    """
    n_experts = router_p.shape[-1]

    expert_usage = router_p.mean(dim=(0, 1))
    ideal = torch.ones_like(expert_usage) / n_experts
    balance_loss = (expert_usage - ideal).pow(2).sum()

    log_router = router_p.clamp(min=1e-8).log()
    log_prior = prior_p.clamp(min=1e-8).log()
    kl = (router_p * (log_router - log_prior)).sum(dim=-1)
    prior_kl = kl.mean()

    entropy = -(router_p * log_router).sum(dim=-1)
    entropy_loss = (entropy.mean() - entropy_target).pow(2)

    return {
        "moe_balance": balance_weight * balance_loss,
        "moe_prior_kl": kl_weight * prior_kl,
        "moe_entropy": entropy_weight * entropy_loss,
    }