"""Image-to-TS Decoder for the TIGER module.

Reconstructs time series from generated 3-channel images:
  - Channel R (GASF): diagonal extraction -> arccos -> inverse normalise
  - Channel G (STFT): Griffin-Lim phase reconstruction -> inverse normalise
  - Fusion: adaptive weighted average of the two estimates

The recurrence plot (Channel B) is auxiliary and not used for reconstruction.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ts_to_image import NormParams


# ---------------------------------------------------------------------------
# GASF inverse
# ---------------------------------------------------------------------------

def gasf_to_ts(gasf: torch.Tensor, ts_length: int) -> torch.Tensor:
    """Recover a time series from the GASF matrix diagonal.

    The GASF is defined as G_ij = cos(arccos(x_i) + arccos(x_j)).
    The diagonal satisfies G_tt = cos(2 * arccos(x_t)), so
        x_t = sqrt((G_tt + 1) / 2).

    Args:
        gasf: (..., H, W) GASF image with values in [0, 1] (mapped from [-1,1]).
        ts_length: desired output length T.
    Returns:
        (..., T) time-series values in [0, 1].
    """
    # Map back from [0,1] to [-1,1].
    gasf = gasf * 2.0 - 1.0

    # Take the diagonal.  For non-square or resized images we interpolate
    # the diagonal to ts_length.
    H, W = gasf.shape[-2:]
    # Extract diagonal (works for square images).
    n = min(H, W)
    diag = gasf[..., torch.arange(n), torch.arange(n)]  # (..., n)

    # Interpolate diagonal to target ts_length if needed.
    if n != ts_length:
        diag = F.interpolate(
            diag.unsqueeze(1), size=ts_length, mode="linear", align_corners=True
        ).squeeze(1)

    # Recover x from diag = cos(2*arccos(x)) = 2*x^2 - 1
    # => x = sqrt((diag + 1) / 2)
    diag = diag.clamp(-1.0, 1.0)
    x = torch.sqrt((diag + 1.0) / 2.0)
    return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# STFT inverse via Griffin-Lim
# ---------------------------------------------------------------------------

def griffin_lim(
    magnitude: torch.Tensor,
    n_fft: int = 64,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_iter: int = 32,
) -> torch.Tensor:
    """Reconstruct a signal from its magnitude spectrogram using Griffin-Lim.

    Args:
        magnitude: (..., F, frames) non-negative magnitude.
        n_fft: FFT size used during the forward STFT.
        hop_length: hop size (default n_fft // 4).
        win_length: window length (default n_fft).
        n_iter: number of Griffin-Lim iterations.
    Returns:
        (..., T) reconstructed time-domain signal.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    window = torch.hann_window(win_length, device=magnitude.device, dtype=magnitude.dtype)

    # Initialise with random phase uniformly distributed in [0, 2*pi].
    shape = magnitude.shape
    angles = torch.rand(shape, device=magnitude.device, dtype=magnitude.dtype) * 2.0 * math.pi

    complex_spec = magnitude * torch.exp(1j * angles)

    for _ in range(n_iter):
        # Inverse STFT.
        signal = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=None,
        )
        # Re-compute STFT to get updated phase.
        complex_spec = torch.stft(
            signal,
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
        # Replace magnitude with the target, keep the estimated phase.
        complex_spec = magnitude * torch.exp(1j * complex_spec.angle())

    # Final inverse STFT.
    signal = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        length=None,
    )
    return signal


def stft_image_to_ts(
    stft_img: torch.Tensor,
    ts_length: int,
    n_fft: int = 64,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_iter: int = 32,
) -> torch.Tensor:
    """Recover a time series from the STFT magnitude image.

    Args:
        stft_img: (..., H, W) image with values in [0, 1].
        ts_length: desired output time-series length.
        n_fft, hop_length, win_length, n_iter: Griffin-Lim parameters.
    Returns:
        (..., T) time-series values (unnormalised).
    """
    # The image was normalised to [0,1] per-sample.  We don't have the
    # original scale, but Griffin-Lim only needs relative magnitudes so
    # this is acceptable (the decoder's denormalisation handles the rest).

    # Resize back to (F, frames) expected by Griffin-Lim.
    # We use F = n_fft//2 + 1; frames is approximate.
    F_bins = n_fft // 2 + 1
    # Estimate number of frames for the target ts_length.
    frames = max(1, math.ceil(ts_length / (hop_length or (n_fft // 4))))

    # Interpolate to the required spectrogram shape.
    mag = F.interpolate(
        stft_img, size=(F_bins, frames), mode="bicubic", align_corners=False
    )
    mag = mag.clamp(min=0.0)

    # Flatten batch dims for griffin_lim.
    leading = mag.shape[:-2]
    mag_flat = mag.reshape(-1, F_bins, frames)

    signal = griffin_lim(mag_flat, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, n_iter=n_iter)

    # Trim or pad to ts_length.
    T = signal.shape[-1]
    if T > ts_length:
        signal = signal[..., :ts_length]
    elif T < ts_length:
        signal = F.pad(signal, (0, ts_length - T))

    return signal.reshape(*leading, ts_length)


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------

class ImageToTSDecoder:
    """Decodes generated 3-channel images back into time series.

    Two modes:
        - ``"gasf"``: use only the GASF diagonal (fast, exact if no noise).
        - ``"fused"``: weighted blend of GASF and STFT estimates (default).

    Usage:
        decoder = ImageToTSDecoder()
        ts = decoder.decode(image, ts_length=96, norm_params=norm_params)
    """

    def __init__(
        self,
        mode: Literal["gasf", "fused"] = "fused",
        alpha: float = 0.5,
        learnable_alpha: bool = False,
        n_fft: int = 64,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        griffin_lim_iters: int = 32,
    ) -> None:
        """
        Args:
            mode: "gasf" for GASF-only, "fused" for GASF+STFT blend.
            alpha: initial blending weight for GASF (0 = pure STFT, 1 = pure GASF).
            learnable_alpha: if True, alpha is a learnable parameter.
            n_fft: FFT size (must match the encoder).
            hop_length: hop size (must match the encoder).
            win_length: window length (must match the encoder).
            griffin_lim_iters: number of Griffin-Lim iterations.
        """
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.griffin_lim_iters = griffin_lim_iters

        if learnable_alpha:
            self._logit_alpha = nn.Parameter(torch.tensor(math.log(alpha / (1 - alpha + 1e-8))))
            self._fixed_alpha = None
        else:
            self._logit_alpha = None
            self._fixed_alpha = alpha

    @property
    def alpha(self) -> float:
        """Current blending weight (scalar Python float for non-learnable case)."""
        if self._fixed_alpha is not None:
            return self._fixed_alpha
        return torch.sigmoid(self._logit_alpha).item()

    def _get_alpha(self, device: torch.device) -> torch.Tensor:
        """Return alpha as a scalar tensor on the correct device."""
        if self._fixed_alpha is not None:
            return torch.tensor(self._fixed_alpha, device=device)
        return torch.sigmoid(self._logit_alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(
        self,
        image: torch.Tensor,
        ts_length: int,
        norm_params: NormParams,
    ) -> torch.Tensor:
        """Decode a batch of 3-channel images into time series.

        Args:
            image: (B, 3, H, W) with values in [0, 1].
            ts_length: target output length T.
            norm_params: stored by ``TSToImageEncoder.encode()``.
        Returns:
            (B, T) time series in the original scale.
        """
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W), got {tuple(image.shape)}")

        r = image[:, 0]  # GASF
        g = image[:, 1]  # STFT

        # GASF path: always computed.
        x_gasf = gasf_to_ts(r, ts_length)   # (B, T) in [0, 1]

        if self.mode == "gasf":
            x_norm = x_gasf
        elif self.mode == "fused":
            x_stft = stft_image_to_ts(
                g, ts_length,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_iter=self.griffin_lim_iters,
            )
            # STFT output is unnormalised; bring it to [0, 1] range for blending.
            x_stft_norm = self._normalise_01(x_stft)

            alpha = self._get_alpha(image.device)
            x_norm = alpha * x_gasf + (1.0 - alpha) * x_stft_norm
            x_norm = x_norm.clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'gasf' or 'fused'.")

        # Denormalise to original scale.
        x = self._denormalize(x_norm, norm_params)
        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_01(x: torch.Tensor) -> torch.Tensor:
        """Per-sample min-max normalisation to [0, 1]."""
        mn = x.amin(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        return (x - mn) / (mx - mn + 1e-8)

    @staticmethod
    def _denormalize(x_norm: torch.Tensor, norm_params: NormParams) -> torch.Tensor:
        """Map from [0, 1] back to the original scale stored in norm_params.

        Args:
            x_norm: (B, T) in [0, 1].
            norm_params: from the encoder.
        Returns:
            (B, T) in original scale.
        """
        min_val = norm_params.min_val
        max_val = norm_params.max_val

        if min_val.dim() == 1:
            # Univariate: (B,) -> (B, 1)
            min_val = min_val.unsqueeze(-1)
            max_val = max_val.unsqueeze(-1)

        # If multivariate, norm_params has shape (B, C) but x_norm is (B*C, T).
        # The caller is responsible for reshaping before and after if needed.
        # For the common univariate case this just broadcasts correctly.
        x = x_norm * (max_val - min_val) + min_val
        return x
