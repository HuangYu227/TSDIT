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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Core Planner
# ---------------------------------------------------------------------------


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
        raw_mass, raw_density = info_raw.unbind(dim=-1)

        mass = F.softplus(raw_mass)  # positive
        mass = mass / (mass.sum(dim=-1, keepdim=True) + 1e-8)  # sum=1

        density = F.softplus(raw_density) + 1e-6  # positive + eps
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
