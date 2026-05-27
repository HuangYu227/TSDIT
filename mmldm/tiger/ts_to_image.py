"""TS-to-Image Encoder for the TIGER module.

Converts time series into 3-channel images suitable for diffusion models:
  - Channel R: Gramian Angular Summation Field (GASF)  [bijective via diagonal]
  - Channel G: STFT magnitude spectrogram             [bijective via iSTFT]
  - Channel B: Recurrence Plot                         [auxiliary]

All outputs are resized to a fixed spatial resolution (default 64x64).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalization parameter container
# ---------------------------------------------------------------------------

@dataclass
class NormParams:
    """Stores per-channel min/max so the decoder can denormalize."""

    min_val: torch.Tensor   # (B,) or (B, C)
    max_val: torch.Tensor   # (B,) or (B, C)
    n_vars: int = 1         # number of variates (1 for univariate)
    original_length: int = 0

    def to(self, device: torch.device) -> "NormParams":
        return NormParams(
            min_val=self.min_val.to(device),
            max_val=self.max_val.to(device),
            n_vars=self.n_vars,
            original_length=self.original_length,
        )


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def _safe_min_max(
    x: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute min/max with a guard for constant sequences.

    Args:
        x: (..., T) tensor.
    Returns:
        min_val, max_val with same leading shape as x minus the last dim.
    """
    min_val = x.amin(dim=-1)
    max_val = x.amax(dim=-1)
    # If max == min the sequence is constant; expand range slightly so
    # normalization doesn't produce NaN.
    same = (max_val - min_val).abs() < eps
    max_val = torch.where(same, min_val + eps, max_val)
    return min_val, max_val


def _normalize_01(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize to [0, 1].  Returns (x_norm, min_val, max_val)."""
    min_val, max_val = _safe_min_max(x)
    # Broadcast shape: (..., 1)
    x_norm = (x - min_val.unsqueeze(-1)) / (max_val - min_val).unsqueeze(-1)
    # Clamp for numerical safety after arccos.
    x_norm = x_norm.clamp(0.0, 1.0)
    return x_norm, min_val, max_val


# ---------------------------------------------------------------------------
# GASF  (Gramian Angular Summation Field)
# ---------------------------------------------------------------------------

def ts_to_gasf(ts: torch.Tensor) -> torch.Tensor:
    """Compute the Gramian Angular Summation Field.

    Args:
        ts: (..., T) values in [0, 1].
    Returns:
        (..., T, T) GASF matrix with values in [-1, 1].
    """
    # phi = arccos(x),  x in [0,1] -> phi in [0, pi]
    phi = torch.acos(ts.clamp(0.0, 1.0))
    # G_ij = cos(phi_i + phi_j)
    phi_i = phi.unsqueeze(-1)   # (..., T, 1)
    phi_j = phi.unsqueeze(-2)   # (..., 1, T)
    gasf = torch.cos(phi_i + phi_j)
    return gasf


# ---------------------------------------------------------------------------
# STFT magnitude
# ---------------------------------------------------------------------------

def ts_to_stft_mag(
    ts: torch.Tensor,
    n_fft: int = 64,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> torch.Tensor:
    """Compute the STFT magnitude spectrogram.

    Args:
        ts: (..., T) real-valued signal.
        n_fft: FFT size.
        hop_length: Hop size (default n_fft // 4).
        win_length: Window length (default n_fft).
    Returns:
        (..., F, frames) magnitude spectrogram (non-negative).
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # torch.stft expects (..., T).
    # return_complex=True gives (..., F, frames) complex output.
    window = torch.hann_window(win_length, device=ts.device, dtype=ts.dtype)
    spec = torch.stft(
        ts,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return spec.abs()


# ---------------------------------------------------------------------------
# Recurrence Plot
# ---------------------------------------------------------------------------

def ts_to_recurrence(
    ts: torch.Tensor, epsilon_quantile: float = 0.1
) -> torch.Tensor:
    """Compute a thresholded recurrence plot.

    Args:
        ts: (..., T) time series.
        epsilon_quantile: quantile of pairwise distances used as threshold.
    Returns:
        (..., T, T) binary recurrence matrix (float, 0/1).
    """
    # Pairwise L2 distance: (..., T, T)
    # d_ij = |x_i - x_j|
    diff = ts.unsqueeze(-1) - ts.unsqueeze(-2)
    dist = diff.abs()

    # Threshold at the given quantile.
    # Flatten the spatial dims to compute quantile.
    flat = dist.reshape(*dist.shape[:-2], -1)
    k = max(1, int(math.ceil(flat.shape[-1] * epsilon_quantile)))
    # topk on the *smallest* values
    threshold = flat.topk(k, dim=-1, largest=False).values[..., -1:]
    # Broadcast threshold back to (..., 1, 1)
    threshold = threshold.unsqueeze(-1)
    rp = (dist <= threshold).float()
    return rp


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class TSToImageEncoder:
    """Encodes time series into 3-channel images for TIGER diffusion.

    Channels:
        R = GASF  (Gramian Angular Summation Field)
        G = STFT  magnitude spectrogram
        B = RP    (Recurrence Plot)

    Usage:
        encoder = TSToImageEncoder(image_size=64)
        image, norm_params = encoder.encode(ts)         # (B, 3, 64, 64)
        ch, norm_params = encoder.encode_single_channel(ts, "gasf")
    """

    ChannelName = Literal["gasf", "stft", "rp", "all"]

    def __init__(
        self,
        image_size: int = 64,
        n_fft: int = 64,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        epsilon_quantile: float = 0.1,
    ) -> None:
        self.image_size = image_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.epsilon_quantile = epsilon_quantile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_2d(ts: torch.Tensor) -> Tuple[torch.Tensor, bool, int]:
        """Ensure input is (B, T).  Returns (ts_2d, was_multivariate, n_vars)."""
        if ts.dim() == 1:
            # (T,) -> (1, T)
            return ts.unsqueeze(0), False, 1
        if ts.dim() == 2:
            # (B, T)
            return ts, False, 1
        if ts.dim() == 3:
            # (B, T, C) -> flatten variates into batch
            B, T, C = ts.shape
            return ts.permute(0, 2, 1).reshape(B * C, T), True, C
        raise ValueError(f"Expected 1D/2D/3D input, got {ts.dim()}D")

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize (..., H, W) spatial dims to (image_size, image_size) via bicubic."""
        # F.interpolate works on (N, C, H, W).  We may have extra leading dims.
        shape = x.shape
        # Flatten everything except the last two spatial dims into batch.
        flat = x.reshape(-1, 1, shape[-2], shape[-1])
        resized = F.interpolate(
            flat, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
        )
        return resized.reshape(*shape[:-2], self.image_size, self.image_size)

    def _pad_short(self, ts: torch.Tensor, min_len: int) -> Tuple[torch.Tensor, bool]:
        """Zero-pad sequences shorter than *min_len*.  Returns (ts_padded, did_pad)."""
        T = ts.shape[-1]
        if T >= min_len:
            return ts, False
        pad = min_len - T
        # Pad on the right.
        ts = F.pad(ts, (0, pad))
        return ts, True

    def _compute_gasf(self, ts_norm: torch.Tensor) -> torch.Tensor:
        """GASF channel.  Input (B, T) in [0,1], output (B, 1, H, W)."""
        gasf = ts_to_gasf(ts_norm)                       # (B, T, T)
        gasf = self._resize(gasf)                         # (B, img, img)
        # Map from [-1, 1] to [0, 1] for image range.
        gasf = (gasf + 1.0) / 2.0
        return gasf.unsqueeze(1)                          # (B, 1, img, img)

    def _compute_stft(self, ts: torch.Tensor) -> torch.Tensor:
        """STFT magnitude channel.  Input (B, T), output (B, 1, H, W)."""
        mag = ts_to_stft_mag(
            ts, n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )                                                  # (B, F, frames)
        # Normalize per-sample to [0, 1].
        flat = mag.reshape(mag.shape[0], -1)
        mag_min = flat.amin(dim=-1, keepdim=True).unsqueeze(-1)
        mag_max = flat.amax(dim=-1, keepdim=True).unsqueeze(-1)
        mag = (mag - mag_min) / (mag_max - mag_min + 1e-8)
        mag = self._resize(mag)                            # (B, img, img)
        return mag.unsqueeze(1)                            # (B, 1, img, img)

    def _compute_rp(self, ts: torch.Tensor) -> torch.Tensor:
        """Recurrence plot channel.  Input (B, T), output (B, 1, H, W)."""
        rp = ts_to_recurrence(ts, epsilon_quantile=self.epsilon_quantile)  # (B, T, T)
        rp = self._resize(rp)                              # (B, img, img)
        return rp.unsqueeze(1)                             # (B, 1, img, img)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self, ts: torch.Tensor
    ) -> Tuple[torch.Tensor, NormParams]:
        """Encode a batch of time series into 3-channel images.

        Args:
            ts: (B, T) univariate or (B, T, C) multivariate.
        Returns:
            image: (B, 3, H, W) with H=W=image_size.  Values in [0, 1].
            norm_params: stored for the inverse decoder.
        """
        ts_2d, is_multi, n_vars = self._to_2d(ts)
        B_flat, T = ts_2d.shape

        # Handle very short sequences (STFT needs at least n_fft samples).
        min_len = max(self.n_fft, 4)
        ts_2d, _ = self._pad_short(ts_2d, min_len)

        # Normalize to [0, 1].
        ts_norm, min_val, max_val = _normalize_01(ts_2d)

        # Replace NaN with 0 after normalization (rare but possible).
        ts_norm = torch.nan_to_num(ts_norm, nan=0.0, posinf=1.0, neginf=0.0)

        # Compute the three channels.
        r = self._compute_gasf(ts_norm)                    # (B_flat, 1, img, img)
        g = self._compute_stft(ts_2d)                      # (B_flat, 1, img, img)
        b = self._compute_rp(ts_norm)                      # (B_flat, 1, img, img)

        image = torch.cat([r, g, b], dim=1)                # (B_flat, 3, img, img)

        # If multivariate, reshape back: (B_flat, ...) -> (B, C, ...)
        if is_multi:
            B = ts.shape[0]
            image = image.reshape(B, n_vars, 3, self.image_size, self.image_size)
            # Merge variate dim into batch for diffusion: (B*C, 3, img, img)
            image = image.reshape(B * n_vars, 3, self.image_size, self.image_size)
            min_val = min_val.reshape(B, n_vars)
            max_val = max_val.reshape(B, n_vars)
        else:
            min_val = min_val.squeeze(-1)  # (B,)
            max_val = max_val.squeeze(-1)  # (B,)

        norm_params = NormParams(
            min_val=min_val,
            max_val=max_val,
            n_vars=n_vars,
            original_length=T,
        )
        return image, norm_params

    def encode_single_channel(
        self, ts: torch.Tensor, channel: str
    ) -> Tuple[torch.Tensor, NormParams]:
        """Encode to a single-channel image.

        Args:
            ts: (B, T) or (B, T, C).
            channel: one of "gasf", "stft", "rp".
        Returns:
            image: (B, 1, H, W).
            norm_params.
        """
        ts_2d, is_multi, n_vars = self._to_2d(ts)
        B_flat, T = ts_2d.shape

        min_len = max(self.n_fft, 4)
        ts_2d, _ = self._pad_short(ts_2d, min_len)

        ts_norm, min_val, max_val = _normalize_01(ts_2d)
        ts_norm = torch.nan_to_num(ts_norm, nan=0.0, posinf=1.0, neginf=0.0)

        if channel == "gasf":
            ch = self._compute_gasf(ts_norm)
        elif channel == "stft":
            ch = self._compute_stft(ts_2d)
        elif channel == "rp":
            ch = self._compute_rp(ts_norm)
        else:
            raise ValueError(f"Unknown channel '{channel}'. Expected gasf/stft/rp.")

        if is_multi:
            B = ts.shape[0]
            ch = ch.reshape(B * n_vars, 1, self.image_size, self.image_size)
            min_val = min_val.reshape(B, n_vars)
            max_val = max_val.reshape(B, n_vars)
        else:
            min_val = min_val.squeeze(-1)
            max_val = max_val.squeeze(-1)

        norm_params = NormParams(
            min_val=min_val, max_val=max_val,
            n_vars=n_vars, original_length=T,
        )
        return ch, norm_params
