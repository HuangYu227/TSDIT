"""MATD encoder1: Multi-Resolution Density-Aware Temporal Tokenizer.

Drop-in replacement for the current ``tokenizer.py`` encoder.

The public class ``DensityAwareAdaptivePatch`` preserves the existing MATD
interface:

    z0, meta = encoder(x0)

where ``x0`` is ``(B, T)`` and outputs are ``z0: (B, K, D)`` and
``meta: (B, K, 9)``.  The 9 metadata channels are unchanged:
[start, end, center, length, log_length, mass, density, log_density, order].

Compared with the earlier DA-ATP implementation, this version keeps the stable
length-aware density partitioning but strengthens the patch representation with
multi-resolution shape features, derivative/spectral statistics, and optional
meta-aware temporal context attention over patch tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def _sincos_features(x: torch.Tensor, num_bands: int) -> torch.Tensor:
    """Sin/cos features for values in roughly [0, 1]."""
    if num_bands <= 0:
        return x.new_zeros(*x.shape, 0)
    freqs = (2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype))
    angles = 2.0 * math.pi * x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _interp_1d(segment: torch.Tensor, out_len: int) -> torch.Tensor:
    """Interpolate a 1D tensor to ``out_len``."""
    return F.interpolate(
        segment.view(1, 1, -1),
        size=out_len,
        mode="linear",
        align_corners=False,
    ).view(out_len)


# ---------------------------------------------------------------------------
# Meta-aware temporal context transformer
# ---------------------------------------------------------------------------


class MetaRelativeTemporalAttention(nn.Module):
    """Self-attention over patch tokens with metadata-derived temporal bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_buckets: int = 33,
        dropout: float = 0.0,
        use_density_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_buckets = num_buckets
        self.use_density_bias = use_density_bias

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rel_time_bias = nn.Embedding(num_buckets, num_heads)
        if use_density_bias:
            self.density_bias = nn.Linear(1, num_heads)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        B, K, D = x.shape
        qkv = self.qkv(x).reshape(B, K, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, K, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        centers = meta[..., 2].clamp(0.0, 1.0)
        delta = centers[:, :, None] - centers[:, None, :]
        bucket = ((delta + 1.0) * 0.5 * (self.num_buckets - 1)).long()
        bucket = bucket.clamp(0, self.num_buckets - 1)
        bias = self.rel_time_bias(bucket).permute(0, 3, 1, 2)
        attn = attn + bias

        if self.use_density_bias:
            log_density = meta[..., 7].unsqueeze(-1)
            den_bias = self.density_bias(log_density).permute(0, 2, 1).unsqueeze(-1)
            attn = attn + den_bias

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, K, D)
        return self.proj_drop(self.proj(out))


class TemporalContextBlock(nn.Module):
    """Pre-norm transformer block for contextualizing adaptive patch tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MetaRelativeTemporalAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), meta)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class DensityAwareAdaptivePatchV2(nn.Module):
    """Multi-resolution density-aware adaptive patch tokenizer.

    This module is intended as the training-time oracle encoder/tokenizer for
    MATD.  It keeps hard adaptive boundaries so planner supervision remains
    stable, but produces stronger continuous patch latents for diffusion.
    """

    _META_DIM = 9

    def __init__(
        self,
        embed_dim: int = 256,
        target_tokens: Optional[int] = None,
        ref_len: int = 16,
        min_len: int = 4,
        max_len: int = 64,
        rfft_win: int = 32,
        tau: float = 0.5,
        base_score: float = 0.05,
        target_seg_len: int = 8,
        min_tokens: int = 6,
        max_tokens: int = 128,
        spectral_bins: int = 8,
        meta_fourier_bands: int = 4,
        context_depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if ref_len < 4:
            raise ValueError("ref_len must be >= 4")
        if min_len <= 0 or max_len < min_len:
            raise ValueError("Require 0 < min_len <= max_len")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self._target_tokens = target_tokens
        self.ref_len = ref_len
        self.min_len = min_len
        self.max_len = max_len
        self.rfft_win = rfft_win
        self.tau = tau
        self.base_score = base_score
        self.target_seg_len = target_seg_len
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.spectral_bins = spectral_bins
        self.meta_fourier_bands = meta_fourier_bands

        self.stat_dim = 16
        shape_dim = ref_len * 2  # value trace + derivative trace
        meta_in_dim = self._META_DIM + self._META_DIM * meta_fourier_bands * 2

        self.shape_proj = nn.Sequential(
            nn.Linear(shape_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(embed_dim, embed_dim),
        )
        self.stat_proj = nn.Sequential(
            nn.Linear(self.stat_dim + spectral_bins, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(embed_dim, embed_dim),
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(embed_dim, embed_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(embed_dim, embed_dim),
        )

        self.context_blocks = nn.ModuleList(
            [
                TemporalContextBlock(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(context_depth)
            ]
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    # Boundary scoring
    # ------------------------------------------------------------------

    def _choose_k(self, T: int) -> int:
        """Length-aware token number selection."""
        if self._target_tokens is None:
            K = math.ceil(T / self.target_seg_len)
            K = max(K, self.min_tokens)
            K = min(K, self.max_tokens)
        else:
            K = int(self._target_tokens)

        max_allowed_k = max(1, T // self.min_len)
        min_required_k = math.ceil(T / self.max_len)
        K = min(K, max_allowed_k)
        K = max(K, min_required_k)
        return int(K)

    @staticmethod
    def _safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        return x / (mean + eps)

    def compute_rfft_score(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        W = min(self.rfft_win, T)
        if W < 4:
            return torch.zeros_like(x)

        windows = x.unfold(1, W, 1)
        N = windows.shape[1]
        hann = torch.hann_window(W, device=x.device, dtype=x.dtype)
        windows = windows * hann.view(1, 1, W)
        X = torch.fft.rfft(windows.float(), dim=-1)
        P = X.abs().pow(2).to(x.dtype).clamp_min(1e-6)
        P_sum = P.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        Pn = P / P_sum
        entropy = -(Pn * Pn.clamp_min(1e-6).log()).sum(dim=-1) / math.log(P.shape[-1])
        hf_start = max(1, int(P.shape[-1] * 0.5))
        P_total = P.sum(dim=-1).clamp_min(1e-6)
        hf_ratio = P[..., hf_start:].sum(dim=-1) / P_total
        win_score = entropy + 0.5 * hf_ratio

        score = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        count = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        half = W // 2
        for i in range(N):
            c = min(i + half, T - 1)
            score[:, c] += win_score[:, i]
            count[:, c] += 1
        score = score / count.clamp_min(1.0)
        if half < T:
            score[:, :half] = score[:, half:half + 1]
        last = min(T - 1, N - 1 + half)
        score[:, last:] = score[:, last:last + 1]
        return score

    def compute_raw_score(self, x: torch.Tensor) -> torch.Tensor:
        d1 = torch.zeros_like(x)
        d1[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
        d2 = torch.zeros_like(x)
        d2[:, 1:-1] = (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]).abs()

        d1n = self._safe_norm(d1)
        d2n = self._safe_norm(d2)
        freq = self._safe_norm(self.compute_rfft_score(x))

        local_energy = F.avg_pool1d((x - x.mean(dim=-1, keepdim=True)).abs().unsqueeze(1), 5, stride=1, padding=2).squeeze(1)
        local_energy = self._safe_norm(local_energy)

        raw = d1n + 0.5 * d2n + freq + 0.25 * local_energy + 1e-4
        raw = F.avg_pool1d(raw.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        return raw.clamp_min(1e-4)

    def make_boundaries(self, raw_score: torch.Tensor) -> list[int]:
        T = raw_score.shape[0]
        K = self._choose_k(T)
        if K * self.min_len > T:
            raise ValueError(f"K*min_len={K * self.min_len} > T={T}")
        if K * self.max_len < T:
            raise ValueError(f"K*max_len={K * self.max_len} < T={T}")

        alloc = raw_score.pow(self.tau) + self.base_score
        cdf = torch.cumsum(alloc, dim=0)
        total = cdf[-1].clamp_min(1e-6)

        boundaries = [0]
        prev = 0
        for k in range(1, K):
            idx = torch.searchsorted(cdf, total * (k / K)).item()
            lower = max(prev + self.min_len, T - (K - k) * self.max_len)
            upper = min(prev + self.max_len, T - (K - k) * self.min_len)
            if lower > upper:
                raise ValueError(
                    f"No feasible boundary: T={T}, K={K}, k={k}, "
                    f"lower={lower}, upper={upper}"
                )
            idx = min(max(idx, lower), upper)
            boundaries.append(idx)
            prev = idx
        boundaries.append(T)
        return boundaries

    # ------------------------------------------------------------------
    # Segment encoding
    # ------------------------------------------------------------------

    def _spectral_features(self, segment: torch.Tensor) -> torch.Tensor:
        if self.spectral_bins <= 0:
            return segment.new_zeros(0)
        seg = _interp_1d(segment, self.ref_len)
        seg = seg - seg.mean()
        spec = torch.fft.rfft(seg.float(), dim=0).abs().to(segment.dtype)
        spec = spec[1:]  # remove DC
        if spec.numel() < self.spectral_bins:
            spec = F.pad(spec, (0, self.spectral_bins - spec.numel()))
        else:
            spec = spec[: self.spectral_bins]
        return spec / (spec.mean().clamp_min(1e-6))

    def _segment_stats(
        self,
        segment: torch.Tensor,
        score_seg: torch.Tensor,
        total_score: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L = segment.shape[0]
        dtype = segment.dtype
        device = segment.device
        length = torch.tensor(L / T, device=device, dtype=dtype)
        mass = score_seg.sum() / total_score.clamp_min(1e-6)
        density = mass / (length + 1e-6)

        if L > 1:
            diff = segment[1:] - segment[:-1]
            abs_diff_mean = diff.abs().mean()
            diff_std = diff.std(unbiased=False)
            slope = (segment[-1] - segment[0]) / max(L - 1, 1)
        else:
            diff = segment.new_zeros(1)
            abs_diff_mean = segment.new_tensor(0.0)
            diff_std = segment.new_tensor(0.0)
            slope = segment.new_tensor(0.0)
        if L > 2:
            curv = (segment[2:] - 2.0 * segment[1:-1] + segment[:-2]).abs().mean()
        else:
            curv = segment.new_tensor(0.0)

        centered = segment - segment.mean()
        energy = centered.pow(2).mean()
        stats = torch.stack(
            [
                segment.mean(),
                segment.std(unbiased=False),
                segment.min(),
                segment.max(),
                segment[0],
                segment[-1],
                segment[-1] - segment[0],
                abs_diff_mean,
                diff_std,
                curv,
                slope,
                energy,
                mass,
                density,
                length,
                _safe_log(density + 1e-6),
            ]
        )
        return stats, mass, density, length

    def encode_segment(
        self,
        segment: torch.Tensor,
        score_seg: torch.Tensor,
        total_score: torch.Tensor,
        T: int,
        start_idx: int,
        end_idx: int,
        order_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L = segment.shape[0]
        values = _interp_1d(segment, self.ref_len)
        if L > 1:
            d = segment[1:] - segment[:-1]
        else:
            d = segment.new_zeros(1)
        d_values = _interp_1d(d, self.ref_len)
        shape_feat = torch.cat([values, d_values], dim=0)

        stats, mass, density, length = self._segment_stats(segment, score_seg, total_score, T)
        spec = self._spectral_features(segment)
        stat_feat = torch.cat([stats, spec], dim=0)

        start = torch.tensor(start_idx / T, device=segment.device, dtype=segment.dtype)
        end = torch.tensor(end_idx / T, device=segment.device, dtype=segment.dtype)
        center = torch.tensor(((start_idx + end_idx - 1) / 2) / max(T - 1, 1), device=segment.device, dtype=segment.dtype)
        meta = torch.stack(
            [
                start,
                end,
                center,
                length,
                _safe_log(length),
                mass,
                density,
                _safe_log(density),
                order_value,
            ]
        )

        meta_feat = torch.cat([meta, _sincos_features(meta, self.meta_fourier_bands).flatten()], dim=0)
        token = self.shape_proj(shape_feat) + self.stat_proj(stat_feat) + self.meta_proj(meta_feat)
        token = token + self.fuse(token)
        return token, meta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (B,T) or (B,T,1), got {tuple(x.shape)}")

        B, T = x.shape
        K = self._choose_k(T)
        raw_score = self.compute_raw_score(x)

        all_tokens: list[torch.Tensor] = []
        all_meta: list[torch.Tensor] = []
        for b in range(B):
            score_b = raw_score[b]
            boundaries = self.make_boundaries(score_b)
            total = score_b.sum().clamp_min(1e-6)
            tokens_b: list[torch.Tensor] = []
            meta_b: list[torch.Tensor] = []
            for k in range(K):
                s, e = boundaries[k], boundaries[k + 1]
                order = torch.tensor(k / max(K - 1, 1), device=x.device, dtype=x.dtype)
                token, meta = self.encode_segment(x[b, s:e], score_b[s:e], total, T, s, e, order)
                tokens_b.append(token)
                meta_b.append(meta)
            all_tokens.append(torch.stack(tokens_b, dim=0))
            all_meta.append(torch.stack(meta_b, dim=0))

        tokens = torch.stack(all_tokens, dim=0)
        meta = torch.stack(all_meta, dim=0)

        for block in self.context_blocks:
            tokens = block(tokens, meta)
        tokens = self.out_norm(tokens)
        return tokens, meta


# Backward-compatible class name expected by MATDModel._import_tokenizer().
DensityAwareAdaptivePatch = DensityAwareAdaptivePatchV2


class AdaptiveTemporalEncoder(nn.Module):
    """Optional encoder wrapper kept for compatibility with older experiments."""

    def __init__(
        self,
        embed_dim: int = 256,
        out_dim: int = 128,
        target_tokens: Optional[int] = None,
        depth: int = 2,
        num_heads: int = 8,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = DensityAwareAdaptivePatchV2(
            embed_dim=embed_dim,
            target_tokens=target_tokens,
            context_depth=depth,
            num_heads=num_heads,
            dropout=drop,
        )
        self.proj = nn.Linear(embed_dim, out_dim) if out_dim != embed_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, meta = self.patch_embed(x)
        return self.proj(tokens), meta
