"""MATD decoder1: Partition-of-Unity Neural Field Decoder.

Drop-in replacement for ``decoder.py``.

The public class ``VariablePatchDecoder`` preserves the current interface:

    x_hat = decoder(z, meta, target_length)

where ``z`` is ``(B, K, D)``, ``meta`` is ``(B, K, 9)``, and output is
``(B, target_length, 1)``.

Compared with the earlier decoder, this version does not round each patch
length and concatenate hard segments.  Instead, it evaluates a continuous
per-patch neural field at every target time coordinate and blends patch fields
with a differentiable partition-of-unity weight derived from metadata.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fourier_features(x: torch.Tensor, num_bands: int) -> torch.Tensor:
    """Fourier features for scalar coordinates.

    Args:
        x: Tensor of arbitrary shape.
        num_bands: Number of frequency bands.
    Returns:
        Tensor with shape ``x.shape + (2 * num_bands,)``.
    """
    if num_bands <= 0:
        return x.new_zeros(*x.shape, 0)
    freqs = 2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype)
    angles = 2.0 * math.pi * x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class DecoderTemporalAttention(nn.Module):
    """Lightweight temporal context block over patch latents before decoding."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.rel_bias = nn.Sequential(nn.Linear(3, num_heads), nn.Tanh())
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        B, K, D = z.shape
        h = self.norm(z)
        qkv = self.qkv(h).reshape(B, K, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        center = meta[..., 2]
        length = meta[..., 3]
        density = meta[..., 7]
        rel = torch.stack(
            [
                center[:, :, None] - center[:, None, :],
                length[:, :, None] - length[:, None, :],
                density[:, :, None] - density[:, None, :],
            ],
            dim=-1,
        )
        bias = self.rel_bias(rel).permute(0, 3, 1, 2)
        attn = attn + bias
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, K, D)
        z = z + self.out(out)
        z = z + self.ffn(z)
        return z


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class PartitionOfUnityPatchDecoder(nn.Module):
    """Continuous neural-field decoder for adaptive temporal patches.

    Each patch latent defines a local neural field over normalized local time.
    At every global output coordinate, all patch fields are blended by a soft
    partition of unity computed from patch center, length, and density.  This
    avoids non-differentiable length rounding and reduces boundary artifacts.
    """

    def __init__(
        self,
        latent_dim: int,
        meta_dim: int = 9,
        hidden_dim: int = 256,
        local_bands: int = 8,
        global_bands: int = 6,
        context_depth: int = 1,
        context_heads: int = 8,
        dropout: float = 0.0,
        sharpness: float = 6.0,
        density_weight: float = 0.15,
        chunk_size: int = 16,
        detail_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if meta_dim < 9:
            raise ValueError("meta_dim must be at least 9")
        self.latent_dim = latent_dim
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.local_bands = local_bands
        self.global_bands = global_bands
        self.sharpness = float(sharpness)
        self.density_weight = float(density_weight)
        self.chunk_size = int(chunk_size)
        self.detail_scale = float(detail_scale)

        self.z_norm = nn.LayerNorm(latent_dim)
        self.meta_to_z = nn.Sequential(
            nn.Linear(9, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.context_blocks = nn.ModuleList(
            [DecoderTemporalAttention(latent_dim, num_heads=context_heads, dropout=dropout) for _ in range(context_depth)]
        )

        coord_dim = 2 * local_bands + 2 * global_bands + 4  # local/global raw coords too
        in_dim = latent_dim + 9 + coord_dim

        self.field_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.trend_head = nn.Sequential(
            nn.LayerNorm(latent_dim + 9),
            nn.Linear(latent_dim + 9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

        calib_hidden = max(4, hidden_dim // 4)
        self.output_calib = nn.Sequential(
            nn.Linear(1, calib_hidden),
            nn.SiLU(),
            nn.Linear(calib_hidden, 1),
        )
        # Residual calibration starts near identity.
        nn.init.zeros_(self.output_calib[-1].weight)
        nn.init.zeros_(self.output_calib[-1].bias)

    def _check_inputs(self, z: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"z must have shape (B,K,D), got {tuple(z.shape)}")
        if meta.dim() != 3 or meta.shape[0] != z.shape[0] or meta.shape[1] != z.shape[1]:
            raise ValueError(f"meta must have shape (B,K,C) aligned to z; got z={tuple(z.shape)}, meta={tuple(meta.shape)}")
        if meta.shape[-1] < 9:
            raise ValueError(f"meta must have at least 9 channels, got {meta.shape[-1]}")
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"z latent dim={z.shape[-1]} does not match latent_dim={self.latent_dim}")
        return meta[..., :9].to(device=z.device, dtype=z.dtype)

    def _contextualize(self, z: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        h = self.z_norm(z) + self.meta_to_z(meta)
        for block in self.context_blocks:
            h = block(h, meta)
        return h

    def _partition_weights(self, pos: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Compute soft patch weights.

        Args:
            pos: (C,) global coordinates in [0, 1].
            meta: (B, K, 9)
        Returns:
            weights: (B, C, K)
        """
        center = meta[..., 2].clamp(0.0, 1.0)
        length = meta[..., 3].clamp_min(1e-4)
        log_density = meta[..., 7].clamp(-10.0, 10.0)

        dist = (pos[None, :, None] - center[:, None, :]) / (0.5 * length[:, None, :] + 1e-4)
        score = -self.sharpness * dist.pow(2) + self.density_weight * log_density[:, None, :]

        # Encourage exact interval coverage without hard masking.
        start = meta[..., 0]
        end = meta[..., 1]
        edge_temp = 80.0
        inside_left = torch.sigmoid((pos[None, :, None] - start[:, None, :]) * edge_temp)
        inside_right = torch.sigmoid((end[:, None, :] - pos[None, :, None]) * edge_temp)
        inside_bonus = torch.log((inside_left * inside_right).clamp_min(1e-6))
        score = score + 0.5 * inside_bonus
        return F.softmax(score, dim=-1)

    def _field_values(self, z_ctx: torch.Tensor, meta: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Evaluate per-patch fields at global coordinates.

        Args:
            z_ctx: (B, K, D)
            meta:  (B, K, 9)
            pos:   (C,)
        Returns:
            values: (B, C, K, 1)
        """
        B, K, D = z_ctx.shape
        C = pos.numel()

        start = meta[..., 0]
        length = meta[..., 3].clamp_min(1e-4)
        center = meta[..., 2]
        local = (pos[None, :, None] - start[:, None, :]) / length[:, None, :]
        centered_local = local * 2.0 - 1.0
        global_rel = pos[None, :, None] - center[:, None, :]

        local_ff = fourier_features(local, self.local_bands)
        global_ff = fourier_features(pos, self.global_bands)
        global_ff = global_ff[None, :, None, :].expand(B, C, K, -1)

        raw_coords = torch.stack(
            [
                local,
                centered_local,
                global_rel,
                pos[None, :, None].expand(B, C, K),
            ],
            dim=-1,
        )
        coord = torch.cat([raw_coords, local_ff, global_ff], dim=-1)

        z_e = z_ctx[:, None, :, :].expand(B, C, K, D)
        m_e = meta[:, None, :, :].expand(B, C, K, 9)
        inp = torch.cat([z_e, m_e, coord], dim=-1)
        detail = self.field_net(inp)

        coeff = self.trend_head(torch.cat([z_ctx, meta], dim=-1))
        u = centered_local.unsqueeze(-1)
        trend = (
            coeff[:, None, :, 0:1]
            + coeff[:, None, :, 1:2] * u
            + coeff[:, None, :, 2:3] * u.pow(2)
            + coeff[:, None, :, 3:4] * u.pow(3)
        )
        return trend + self.detail_scale * detail

    def forward(self, z: torch.Tensor, meta: torch.Tensor, target_length: int) -> torch.Tensor:
        if target_length <= 0:
            raise ValueError(f"target_length must be positive, got {target_length}")
        meta9 = self._check_inputs(z, meta)
        z_ctx = self._contextualize(z, meta9)

        device = z.device
        dtype = z.dtype
        T = int(target_length)
        coords = (torch.arange(T, device=device, dtype=dtype) + 0.5) / T

        outputs: list[torch.Tensor] = []
        chunk = max(1, self.chunk_size)
        for start in range(0, T, chunk):
            pos = coords[start:start + chunk]
            weights = self._partition_weights(pos, meta9)
            values = self._field_values(z_ctx, meta9, pos)
            y = (weights.unsqueeze(-1) * values).sum(dim=2)
            outputs.append(y)

        out = torch.cat(outputs, dim=1)
        return out + self.output_calib(out)

    @torch.no_grad()
    def partition_weights(self, meta: torch.Tensor, target_length: int) -> torch.Tensor:
        """Expose soft patch assignment weights for visualization/debugging."""
        if meta.dim() != 3 or meta.shape[-1] < 9:
            raise ValueError("meta must have shape (B,K,>=9)")
        meta9 = meta[..., :9]
        T = int(target_length)
        coords = (torch.arange(T, device=meta.device, dtype=meta.dtype) + 0.5) / T
        return self._partition_weights(coords, meta9)


# Backward-compatible names expected by MATDModel.
VariablePatchDecoder = PartitionOfUnityPatchDecoder


class LinearPatchDecoder(nn.Module):
    """Compatibility wrapper: linear prototype retained for ablations."""

    def __init__(self, latent_dim: int, max_patch_len: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.max_patch_len = max_patch_len
        self.proj = nn.Linear(latent_dim, max_patch_len)

    def forward(self, z: torch.Tensor, meta: torch.Tensor, target_length: int) -> torch.Tensor:
        B, K, D = z.shape
        vals = self.proj(z)
        lengths = meta[..., 3]
        decoded = []
        for b in range(B):
            parts = []
            for k in range(K):
                L = max(1, round(float(lengths[b, k].item()) * target_length))
                parts.append(vals[b, k, : min(L, self.max_patch_len)])
            y = torch.cat(parts, dim=0)
            if y.numel() < target_length:
                y = F.pad(y, (0, target_length - y.numel()))
            else:
                y = y[:target_length]
            decoded.append(y)
        return torch.stack(decoded, dim=0).unsqueeze(-1)
