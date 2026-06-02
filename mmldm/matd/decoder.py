"""Variable Patch Decoders for MATD framework.

Reconstruct time series from patch latents produced by the diffusion prior.
Two implementations:

1. **VariablePatchDecoder** -- sincos positional encoding + MLP, produces
   variable-length output per patch that respects the predicted patch layout.
2. **LinearPatchDecoder** -- fast linear prototype, projects each latent
   directly to its maximum patch length.

Both decoders handle variable-length patches, concatenate them, and
crop/pad to the requested target length.

Reference: MATD framework design doc.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
#  Positional Encoding Helpers
# ==========================================================================


def sincos_positional_encoding(
    positions: torch.Tensor,
    dim: int = 16,
) -> torch.Tensor:
    """Generate sinusoidal positional encodings for positions in [0, 1].

    Uses ``dim // 2`` frequency bands with log-spaced wavelengths from
    2pi to 2pi * 2^(dim//2 - 1), following the standard transformer
    positional encoding scheme scaled to the unit interval.

    Args:
        positions: (N,) tensor of positions in [0, 1].
        dim:       Output feature dimension (must be even).

    Returns:
        (N, dim) tensor of sin/cos features.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.arange(half, device=positions.device, dtype=positions.dtype)
    freqs = 2.0 * math.pi * (2.0 ** freqs)  # (half,)
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (N, dim)


# ==========================================================================
#  1. Variable Patch Decoder (sincos + MLP)
# ==========================================================================


class VariablePatchDecoder(nn.Module):
    """Variable-length patch decoder with sincos positional encoding.

    For each patch *k* the decoder:

    1. Reads the target length ``L_k = round(length_k * T)`` from metadata.
    2. Generates local positions ``u_j = j / max(L_k - 1, 1)`` in [0, 1].
    3. Concatenates ``[z_k, meta_k, sincos(u_j)]`` per position.
    4. Passes through an MLP to produce scalar values.

    All patches are concatenated and cropped / padded to ``target_length``.

    Architecture::

        [z_k (D) | meta_k (9) | sincos(u_j) (pos_dim)]  -->  MLP  -->  value

        MLP: Linear(D + 9 + pos_dim, hidden) -> GELU
              -> Linear(hidden, hidden) -> GELU
              -> Linear(hidden, 1)
    """

    def __init__(
        self,
        latent_dim: int,
        meta_dim: int = 9,
        hidden_dim: int = 256,
        pos_dim: int = 16,
    ) -> None:
        """
        Args:
            latent_dim: Dimension D of patch latents.
            meta_dim:   Dimension of per-patch metadata (default 9).
            hidden_dim: Hidden layer width in the MLP.
            pos_dim:    Dimension of sincos positional encoding (must be even).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.meta_dim = meta_dim
        self.pos_dim = pos_dim

        input_dim = latent_dim + meta_dim + pos_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        meta: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Decode patch latents into a time series.

        Args:
            z:             (B, K, D) patch latents.
            meta:          (B, K, 9) patch metadata with channel layout
                           ``[start, end, center, length, log_length,
                              mass, density, log_density, order]``.
            target_length: Desired output length T.

        Returns:
            x_hat: (B, T, 1) reconstructed time series.
        """
        B, K, D = z.shape
        device = z.device
        dtype = z.dtype

        # meta channel 3 = normalised length
        lengths = meta[:, :, 3]  # (B, K) -- each sums to ~1 across K

        decoded_patches: list[torch.Tensor] = []

        for b in range(B):
            patch_list: list[torch.Tensor] = []
            for k in range(K):
                z_k = z[b, k]        # (D,)
                m_k = meta[b, k]     # (9,)

                # Target number of time-steps for this patch
                L_k = max(1, round(float(lengths[b, k].item()) * target_length))

                # Local positions u_j in [0, 1]
                denom = max(L_k - 1, 1)
                u = torch.arange(L_k, device=device, dtype=dtype) / denom  # (L_k,)
                pos_enc = sincos_positional_encoding(u, dim=self.pos_dim)

                # Broadcast z_k and m_k across positions
                z_expand = z_k.unsqueeze(0).expand(L_k, -1)     # (L_k, D)
                m_expand = m_k.unsqueeze(0).expand(L_k, -1)     # (L_k, 9)

                inp = torch.cat([z_expand, m_expand, pos_enc], dim=-1)
                vals = self.net(inp).squeeze(-1)  # (L_k,)
                patch_list.append(vals)

            # Concatenate all patches for this sample
            decoded = torch.cat(patch_list, dim=0)  # (sum_L,)

            # Crop or pad to target_length
            total_len = decoded.shape[0]
            if total_len >= target_length:
                decoded = decoded[:target_length]
            else:
                pad = torch.zeros(
                    target_length - total_len, device=device, dtype=dtype,
                )
                decoded = torch.cat([decoded, pad], dim=0)

            decoded_patches.append(decoded)

        x_hat = torch.stack(decoded_patches, dim=0)  # (B, T)
        return x_hat.unsqueeze(-1)  # (B, T, 1)


# ==========================================================================
#  2. Linear Patch Decoder (fast prototype)
# ==========================================================================


class LinearPatchDecoder(nn.Module):
    """Fast linear patch decoder.

    Each patch latent is projected through a single linear layer to
    produce ``max_patch_len`` values.  The actual number of values kept
    per patch is determined by the metadata length channel, and the
    remainder is discarded.

    Suitable for rapid prototyping where decoder fidelity is secondary
    to iteration speed.

    Architecture::

        z_k (D) --[Linear(D, max_patch_len)]--> values  --> crop(L_k)
    """

    def __init__(
        self,
        latent_dim: int,
        max_patch_len: int = 128,
    ) -> None:
        """
        Args:
            latent_dim:    Dimension D of patch latents.
            max_patch_len: Maximum decoded length per patch.
                           Values beyond the actual patch length are discarded.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.max_patch_len = max_patch_len

        self.proj = nn.Linear(latent_dim, max_patch_len)

    def forward(
        self,
        z: torch.Tensor,
        meta: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Decode patch latents into a time series.

        Args:
            z:             (B, K, D) patch latents.
            meta:          (B, K, 9) patch metadata.
            target_length: Desired output length T.

        Returns:
            x_hat: (B, T, 1) reconstructed time series.
        """
        B, K, D = z.shape
        device = z.device
        dtype = z.dtype

        # Project all patches at once: (B, K, max_patch_len)
        all_values = self.proj(z)

        # meta channel 3 = normalised length
        lengths = meta[:, :, 3]  # (B, K)

        decoded_patches: list[torch.Tensor] = []

        for b in range(B):
            patch_list: list[torch.Tensor] = []
            for k in range(K):
                L_k = max(1, round(float(lengths[b, k].item()) * target_length))
                L_k = min(L_k, self.max_patch_len)
                vals = all_values[b, k, :L_k]  # (L_k,)
                patch_list.append(vals)

            decoded = torch.cat(patch_list, dim=0)  # (sum_L,)

            # Crop or pad to target_length
            total_len = decoded.shape[0]
            if total_len >= target_length:
                decoded = decoded[:target_length]
            else:
                pad = torch.zeros(
                    target_length - total_len, device=device, dtype=dtype,
                )
                decoded = torch.cat([decoded, pad], dim=0)

            decoded_patches.append(decoded)

        x_hat = torch.stack(decoded_patches, dim=0)  # (B, T)
        return x_hat.unsqueeze(-1)  # (B, T, 1)

