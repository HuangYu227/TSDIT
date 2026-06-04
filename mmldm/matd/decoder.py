"""MATD decoder2: Hybrid Residual Partition-of-Unity Decoder.

Drop-in replacement for ``decoder.py``.

The public class ``VariablePatchDecoder`` preserves the current interface:

    x_hat = decoder(z, meta, target_length)

where ``z`` is ``(B, K, D)``, ``meta`` is ``(B, K, 9)``, and output is
``(B, target_length, 1)``.

Compared with the earlier decoder, this version does not round each patch
length and concatenate hard segments.  Instead, it evaluates a continuous
per-patch neural field at every target time coordinate and blends patch fields
with a differentiable partition-of-unity weight derived from metadata.

The local field is a hybrid residual implicit network: a normalized SwiGLU
branch for stable low/mid-frequency structure plus a small coordinate-only
SIREN branch for peak/valley detail.  Both detail heads start at zero so the
existing polynomial trend path remains the safe initial signal.
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
# Local field network
# ---------------------------------------------------------------------------


class SwiGLUResidualBlock(nn.Module):
    """Pre-norm residual SwiGLU block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, b = self.up(h).chunk(2, dim=-1)
        h = F.silu(a) * b
        return x + self.down(self.drop(h))


class SineLayer(nn.Module):
    """SIREN-style sine layer with stable initialization."""

    def __init__(self, in_dim: int, out_dim: int, omega_0: float = 30.0, is_first: bool = False) -> None:
        super().__init__()
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters(is_first=is_first)

    def reset_parameters(self, is_first: bool = False) -> None:
        with torch.no_grad():
            if is_first:
                bound = 1.0 / max(1, self.linear.in_features)
            else:
                bound = math.sqrt(6.0 / max(1, self.linear.in_features)) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class HybridResidualFieldNet(nn.Module):
    """Patch-conditioned residual implicit field."""

    def __init__(
        self,
        latent_dim: int,
        meta_dim: int,
        coord_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        n_blocks: int = 3,
        siren_width: int | None = None,
        siren_omega: float = 18.0,
        siren_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.siren_scale = float(siren_scale)

        in_dim = latent_dim + meta_dim + coord_dim
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [SwiGLUResidualBlock(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(n_blocks)]
        )

        self.film = nn.Sequential(
            nn.LayerNorm(latent_dim + meta_dim),
            nn.Linear(latent_dim + meta_dim, hidden_dim * 2),
        )

        siren_width = siren_width or max(32, hidden_dim // 2)
        self.siren = nn.Sequential(
            SineLayer(coord_dim, siren_width, omega_0=siren_omega, is_first=True),
            SineLayer(siren_width, siren_width, omega_0=siren_omega, is_first=False),
            nn.Linear(siren_width, hidden_dim),
        )

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.siren_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        nn.init.zeros_(self.siren_out[-1].weight)
        nn.init.zeros_(self.siren_out[-1].bias)

    def forward(self, z_ctx: torch.Tensor, meta: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        patch_cond = torch.cat([z_ctx, meta], dim=-1)
        inp = torch.cat([z_ctx, meta, coord], dim=-1)
        h = self.in_proj(self.in_norm(inp))

        gamma, beta = self.film(patch_cond).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        for block in self.blocks:
            h = block(h)
        h = h * (1.0 + gamma) + beta

        siren_h = self.siren(coord)
        return self.out(h) + self.siren_scale * self.siren_out(siren_h)


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
        field_blocks: int = 3,
        siren_omega: float = 18.0,
        siren_scale: float = 0.1,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        if meta_dim < 9:
            raise ValueError("meta_dim must be at least 9")
        if field_blocks < 1:
            raise ValueError("field_blocks must be >= 1")
        if siren_omega <= 0:
            raise ValueError("siren_omega must be positive")
        if siren_scale < 0:
            raise ValueError("siren_scale must be non-negative")
        if output_activation not in ("none", "sigmoid", "clamp"):
            raise ValueError("output_activation must be 'none', 'sigmoid' or 'clamp'")
        self.latent_dim = latent_dim
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.local_bands = local_bands
        self.global_bands = global_bands
        self.sharpness = float(sharpness)
        self.density_weight = float(density_weight)
        self.chunk_size = int(chunk_size)
        self.detail_scale = float(detail_scale)
        self.output_activation = output_activation

        self.z_norm = nn.LayerNorm(latent_dim)
        self.meta_to_z = nn.Sequential(
            nn.Linear(9, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.context_blocks = nn.ModuleList(
            [DecoderTemporalAttention(latent_dim, num_heads=context_heads, dropout=dropout) for _ in range(context_depth)]
        )

        coord_dim = 2 * local_bands + 2 * global_bands + 4
        self.field_net = HybridResidualFieldNet(
            latent_dim=latent_dim,
            meta_dim=9,
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_blocks=field_blocks,
            siren_omega=siren_omega,
            siren_scale=siren_scale,
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
        detail = self.field_net(z_e, m_e, coord)

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
        out = out + self.output_calib(out)
        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.output_activation == "clamp":
            out = out.clamp(0.0, 1.0)
        return out

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
