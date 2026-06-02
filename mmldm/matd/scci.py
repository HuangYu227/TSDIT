"""Semantic-Causal Conditional Injector (SCCI) for MATD framework.

Extracts semantic slots from text via cross-attention, then injects
text and causal information into diffusion patch tokens through a
timestep-aware gated mechanism.

Two-stage pipeline:
    1. TextSemanticSlotExtractor -- compress text tokens into M semantic slots.
    2. SemanticCausalConditionInjector -- fuse slots, causal features,
       timestep, and metadata into the diffusion latent z.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Stage 1: Text Semantic Slot Extractor
# ---------------------------------------------------------------------------


class TextSemanticSlotExtractor(nn.Module):
    """Compress a variable-length text sequence into fixed semantic slots.

    Architecture::

        text_tokens --[text_proj]--> text_hidden
        slot_queries (learnable) --+
                                   |-- cross_attn --> residual + LayerNorm
                                             |
                                     slots (B, n_slots, dim)

    Args:
        text_dim:  Dimension of input text hidden states.
        dim:       Internal / output slot dimension.
        n_slots:   Number of semantic slots (default 6).
        n_heads:   Number of attention heads (default 4).
        dropout:   Dropout rate for multi-head attention.
    """

    def __init__(
        self,
        text_dim: int,
        dim: int,
        n_slots: int = 6,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim

        self.text_proj = nn.Linear(text_dim, dim)
        self.slot_queries = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        text_hidden: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract semantic slots from text hidden states.

        Args:
            text_hidden:       (B, N_txt, text_dim) encoded text tokens.
            text_padding_mask: (B, N_txt) True where padded (optional).

        Returns:
            slots: (B, n_slots, dim) semantic slot representations.
        """
        B = text_hidden.size(0)

        # Project text tokens to slot dimension
        kv = self.text_proj(text_hidden)  # (B, N_txt, dim)

        # Expand learnable queries for the batch
        queries = self.slot_queries.expand(B, -1, -1)  # (B, n_slots, dim)

        # Cross-attend: slots query over text
        attn_out, _ = self.cross_attn(
            query=queries,
            key=kv,
            value=kv,
            key_padding_mask=text_padding_mask,
        )  # (B, n_slots, dim)

        # Residual + norm
        slots = self.norm(queries + attn_out)
        return slots


# ---------------------------------------------------------------------------
#  Stage 2: Semantic-Causal Condition Injector
# ---------------------------------------------------------------------------


class SemanticCausalConditionInjector(nn.Module):
    """Fuse text semantics, causal features, and metadata into diffusion latents.

    The injection is *timestep-aware*: a gate and modulation network,
    conditioned on the diffusion timestep embedding, control how much
    semantic-causal information is mixed into each patch token.

    Architecture (per forward call)::

        a) meta_proj: z <- z + MLP(meta)
        b) Text-patch alignment:
             score = (patch_q @ slot_k^T) / sqrt(d) + bias(meta, slot)
             u     = softmax(score) @ slot_v
        c) Timestep-aware gated injection:
             gate  = sigmoid(MLP([z, u, causal, t, meta]))
             gamma, beta, alpha = MLP([u, causal, t, meta]).chunk(3)
             h     = layer_norm(z)
             out   = z + tanh(alpha) * gate * (h * (1 + gamma) + beta)

    Args:
        dim:      Feature dimension (shared across all inputs).
        meta_dim: Dimension of the metadata vector.
        n_heads:  Number of attention heads for slot alignment.
        dropout:  Dropout rate (reserved for future use).
    """

    def __init__(
        self,
        dim: int,
        meta_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.meta_dim = meta_dim

        # -- a) Metadata projection --
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # -- b) Text-patch alignment --
        self.patch_q = nn.Linear(dim, dim)
        self.slot_k = nn.Linear(dim, dim)
        self.slot_v = nn.Linear(dim, dim)
        self.bias_mlp = nn.Sequential(
            nn.Linear(meta_dim + dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

        # -- c) Timestep-aware gated injection --
        self.gate = nn.Sequential(
            nn.Linear(dim * 4 + meta_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.mod = nn.Sequential(
            nn.Linear(dim * 3 + meta_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 3 * dim),
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        z: torch.Tensor,
        meta: torch.Tensor,
        slots: torch.Tensor,
        t_emb: torch.Tensor,
        causal_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inject semantic-causal conditioning into diffusion patch tokens.

        Args:
            z:           (B, K, dim) diffusion patch tokens.
            meta:        (B, K, meta_dim) per-patch metadata.
            slots:       (B, M, dim) semantic slots from text encoder.
            t_emb:       (B, dim) diffusion timestep embedding.
            causal_feat: (B, K, dim) causal features (optional;
                         defaults to zeros if not provided).

        Returns:
            out:         (B, K, dim) conditioned patch tokens.
            attn_weights:(B, K, M) attention weights over semantic slots.
            u:           (B, K, dim) text-conditioned representation per patch.
        """
        B, K, D = z.shape
        M = slots.size(1)

        # Default causal features
        if causal_feat is None:
            causal_feat = torch.zeros_like(z)

        # Expand timestep embedding: (B, dim) -> (B, K, dim)
        t_expand = t_emb.unsqueeze(1).expand(-1, K, -1)

        # ---- a) Metadata projection ----
        z = z + self.meta_proj(meta)  # (B, K, dim)

        # ---- b) Text-patch alignment ----
        q = self.patch_q(z)            # (B, K, dim)
        k = self.slot_k(slots)         # (B, M, dim)
        v = self.slot_v(slots)         # (B, M, dim)

        # Scaled dot-product scores
        scale = D ** 0.5
        score = torch.bmm(q, k.transpose(1, 2)) / scale  # (B, K, M)

        # Attention bias from metadata and slot content
        meta_expand = meta.unsqueeze(2).expand(-1, -1, M, -1)    # (B, K, M, meta_dim)
        slot_expand = slots.unsqueeze(1).expand(-1, K, -1, -1)   # (B, K, M, dim)
        bias_in = torch.cat([meta_expand, slot_expand], dim=-1)  # (B, K, M, meta_dim+dim)
        bias = self.bias_mlp(bias_in).squeeze(-1)                # (B, K, M)

        attn_weights = F.softmax(score + bias, dim=-1)  # (B, K, M)
        u = torch.bmm(attn_weights, v)                  # (B, K, dim)

        # ---- c) Timestep-aware gated injection ----
        gate_in = torch.cat([z, u, causal_feat, t_expand, meta], dim=-1)
        gate = self.gate(gate_in)  # (B, K, dim)

        mod_in = torch.cat([u, causal_feat, t_expand, meta], dim=-1)
        mod_out = self.mod(mod_in)                       # (B, K, 3*dim)
        gamma, beta, alpha = mod_out.chunk(3, dim=-1)    # each (B, K, dim)

        h = self.layer_norm(z)
        injected = h * (1.0 + gamma) + beta
        out = z + torch.tanh(alpha) * gate * injected

        return out, attn_weights, u
