"""High-standard Semantic-Causal Conditional Injector for MATD.

This module replaces the earlier SCCI implementation while preserving the public
interfaces expected by MATD:

    slots = TextSemanticSlotExtractor(...)(text_hidden, text_padding_mask)
    z_out, attn_weights, text_cond = SemanticCausalConditionInjector(...)(
        z, meta, slots, t_emb, causal_feat
    )

Key upgrades:
- iterative semantic slot refinement instead of one-shot slot extraction;
- multi-head patch-to-slot alignment with metadata-derived attention bias;
- separate semantic and causal residual injection paths;
- adaLN/FiLM-style zero-initialized residual branches for stable diffusion
  training;
- optional auxiliary losses exposed through ``last_aux_losses``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize all parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class FeedForward(nn.Module):
    """LayerNorm -> SwiGLU -> Linear residual block."""

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(self.drop(F.silu(a) * b))


class TextSemanticSlotExtractor(nn.Module):
    """Iterative semantic slot extractor over text tokens.

    Args:
        text_dim: dimension of input text hidden states.
        dim: output slot dimension used by MATD.
        n_slots: number of semantic slots.
        n_heads: attention heads.
        n_iters: number of iterative slot-refinement steps.
        dropout: dropout in attention and FFN.
        slot_mlp_mult: FFN expansion ratio for slot refinement.

    Returns:
        slots: (B, n_slots, dim)
    """

    def __init__(
        self,
        text_dim: int,
        dim: int,
        n_slots: int = 6,
        n_heads: int = 4,
        n_iters: int = 2,
        dropout: float = 0.0,
        slot_mlp_mult: int = 4,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        if n_iters <= 0:
            raise ValueError("n_iters must be positive")
        self.n_slots = n_slots
        self.dim = dim
        self.n_iters = n_iters

        self.text_proj = nn.Linear(text_dim, dim)
        self.slot_queries = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_mult=slot_mlp_mult, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        text_hidden: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_hidden.dim() != 3:
            raise ValueError(f"text_hidden must be (B,N,D), got {tuple(text_hidden.shape)}")
        B = text_hidden.shape[0]
        kv = self.text_norm(self.text_proj(text_hidden))
        slots = self.slot_queries.expand(B, -1, -1)

        # Iterative refinement encourages slot specialization while preserving
        # the same fixed-size output interface.
        for _ in range(self.n_iters):
            q = self.slot_norm(slots)
            attn_out, _ = self.cross_attn(
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=text_padding_mask,
                need_weights=False,
            )
            slots = slots + attn_out
            slots = slots + self.ffn(slots)

        return self.final_norm(slots)


class MultiHeadPatchSlotAlignment(nn.Module):
    """Multi-head patch-to-semantic-slot attention with metadata bias."""

    def __init__(
        self,
        dim: int,
        meta_dim: int = 9,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.meta_dim = meta_dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.bias_mlp = nn.Sequential(
            nn.Linear(meta_dim + dim, dim),
            nn.SiLU(),
            nn.Linear(dim, n_heads),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        meta: torch.Tensor,
        slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, D = z.shape
        M = slots.shape[1]
        q = self.q_proj(z).reshape(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(slots).reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(slots).reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,K,M)

        meta_expand = meta.unsqueeze(2).expand(B, K, M, -1)
        slot_expand = slots.unsqueeze(1).expand(B, K, M, -1)
        bias = self.bias_mlp(torch.cat([meta_expand, slot_expand], dim=-1))
        bias = bias.permute(0, 3, 1, 2)  # (B,H,K,M)
        score = score + bias

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,H,K,d)
        out = out.transpose(1, 2).reshape(B, K, D)
        # Mean attention over heads is useful for visualization and backward compatibility.
        return self.out_proj(out), attn.mean(dim=1)


class GatedAdaLNBranch(nn.Module):
    """Zero-initialized gated residual update branch.

    The branch computes a FiLM-style modulation from conditioning features and
    applies it to normalized ``z``.  Final modulation is zero-initialized, so
    the whole injector starts close to identity.
    """

    def __init__(
        self,
        z_dim: int,
        cond_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or z_dim * 4
        self.z_norm = nn.LayerNorm(z_dim)
        self.gate = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
            nn.Sigmoid(),
        )
        self.mod = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3 * z_dim),
        )
        _zero_module(self.mod[-1])

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([z, cond], dim=-1))
        gamma, beta, alpha = self.mod(cond).chunk(3, dim=-1)
        h = self.z_norm(z)
        update = h * (1.0 + gamma) + beta
        return torch.tanh(alpha) * gate * update


class SemanticCausalConditionInjector(nn.Module):
    """Inject semantic and causal conditioning into patch latents.

    Args:
        dim: latent feature dimension.
        meta_dim: metadata dimension, usually 9.
        n_heads: patch-slot attention heads.
        dropout: dropout probability.
        hidden_mult: hidden width multiplier for update branches.
        return_aux_losses: if True, stores lightweight auxiliary losses in
            ``last_aux_losses``.

    Returns:
        out: (B,K,D) conditioned patch latents.
        attn_weights: (B,K,n_slots) patch-to-slot alignment.
        u: (B,K,D) text-conditioned patch representation.
    """

    def __init__(
        self,
        dim: int,
        meta_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        hidden_mult: int = 4,
        return_aux_losses: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.meta_dim = meta_dim
        self.return_aux_losses = return_aux_losses
        self.last_aux_losses: Dict[str, torch.Tensor] = {}

        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        _zero_module(self.meta_proj[-1])

        self.align = MultiHeadPatchSlotAlignment(dim, meta_dim=meta_dim, n_heads=n_heads, dropout=dropout)
        cond_sem_dim = dim + dim + meta_dim  # text patch u, timestep, meta
        cond_cau_dim = dim + dim + meta_dim  # causal, timestep, meta
        hidden = dim * hidden_mult
        self.semantic_branch = GatedAdaLNBranch(dim, cond_sem_dim, hidden_dim=hidden, dropout=dropout)
        self.causal_branch = GatedAdaLNBranch(dim, cond_cau_dim, hidden_dim=hidden, dropout=dropout)
        self.mix_gate = nn.Sequential(
            nn.Linear(dim * 3 + meta_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 2),
        )
        _zero_module(self.mix_gate[-1])
        self.out_norm = nn.LayerNorm(dim)

    def _slot_diversity_loss(self, slots_like: torch.Tensor) -> torch.Tensor:
        # slots_like can be (B,K,D) text-conditioned patch features; use covariance over tokens.
        if slots_like.shape[1] <= 1:
            return slots_like.new_tensor(0.0)
        x = F.normalize(slots_like, dim=-1)
        gram = torch.matmul(x, x.transpose(1, 2)).abs()
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
        return (gram * (1.0 - eye)).mean()

    def forward(
        self,
        z: torch.Tensor,
        meta: torch.Tensor,
        slots: torch.Tensor,
        t_emb: torch.Tensor,
        causal_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.dim() != 3:
            raise ValueError(f"z must be (B,K,D), got {tuple(z.shape)}")
        if meta.dim() != 3:
            raise ValueError(f"meta must be (B,K,meta_dim), got {tuple(meta.shape)}")
        B, K, D = z.shape
        if causal_feat is None:
            causal_feat = torch.zeros_like(z)
        if t_emb.dim() == 2:
            t_expand = t_emb.unsqueeze(1).expand(-1, K, -1)
        elif t_emb.dim() == 3:
            t_expand = t_emb
        else:
            raise ValueError("t_emb must be (B,D) or (B,K,D)")

        z_base = z + self.meta_proj(meta)
        u, attn_weights = self.align(z_base, meta, slots)

        sem_cond = torch.cat([u, t_expand, meta], dim=-1)
        cau_cond = torch.cat([causal_feat, t_expand, meta], dim=-1)
        sem_update = self.semantic_branch(z_base, sem_cond)
        cau_update = self.causal_branch(z_base, cau_cond)

        mix_logits = self.mix_gate(torch.cat([z_base, u, causal_feat, meta], dim=-1))
        mix = F.softmax(mix_logits, dim=-1)
        update = mix[..., 0:1] * sem_update + mix[..., 1:2] * cau_update
        out = self.out_norm(z_base + update)

        if self.return_aux_losses:
            # Small diagnostic losses; callers may ignore them.
            ent = -(attn_weights.clamp_min(1e-8) * attn_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
            self.last_aux_losses = {
                "loss_scci_attn_entropy": ent,
                "loss_scci_slot_diversity": self._slot_diversity_loss(u),
            }
        else:
            self.last_aux_losses = {}
        return out, attn_weights, u
