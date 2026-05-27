"""Evaluation metrics for TIGER.

Includes:
  - T2S-style metrics: MSE, WAPE, MAE, MRE (on raw time series)
  - Image-level metrics: PSNR, SSIM (on generated vs real images)
  - Round-trip error: TS -> Image -> TS reconstruction quality
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# T2S-style time-series metrics
# ---------------------------------------------------------------------------

def compute_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Squared Error (lower is better)."""
    return float(np.mean((pred - true) ** 2))


def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error (lower is better)."""
    return float(np.mean(np.abs(pred - true)))


def compute_wape(pred: np.ndarray, true: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (lower is better).

    WAPE = sum(|pred - true|) / sum(|true|)
    """
    return float(np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + 1e-8))


def compute_mre(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Relative Error (lower is better).

    MRE = mean(|pred - true| / (|true| + eps))
    """
    return float(np.mean(np.abs(pred - true) / (np.abs(true) + 1e-8)))


# ---------------------------------------------------------------------------
# Image-level metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: np.ndarray, true: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (higher is better).

    Args:
        pred, true: (B, C, H, W) or (C, H, W) or (H, W) arrays in [0, max_val].
    """
    mse = np.mean((pred - true) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(pred: np.ndarray, true: np.ndarray, max_val: float = 1.0) -> float:
    """Structural Similarity Index (higher is better, max 1.0).

    Simplified SSIM computed per-sample then averaged.
    """
    from skimage.metrics import structural_similarity as ssim_fn

    if pred.ndim == 4:
        vals = []
        for i in range(pred.shape[0]):
            p = np.transpose(pred[i], (1, 2, 0))
            t = np.transpose(true[i], (1, 2, 0))
            c_axis = -1 if p.shape[-1] > 1 else None
            vals.append(ssim_fn(t, p, data_range=max_val, channel_axis=c_axis))
        return float(np.mean(vals))
    elif pred.ndim == 3:
        p = np.transpose(pred, (1, 2, 0))
        t = np.transpose(true, (1, 2, 0))
        c_axis = -1 if p.shape[-1] > 1 else None
        return float(ssim_fn(t, p, data_range=max_val, channel_axis=c_axis))
    else:
        return float(ssim_fn(true, pred, data_range=max_val))


# ---------------------------------------------------------------------------
# Round-trip error: TS -> Image -> TS
# ---------------------------------------------------------------------------

def compute_roundtrip_error(
    ts_original: torch.Tensor,
    ts_reconstructed: torch.Tensor,
) -> dict:
    """Compute round-trip TS -> Image -> TS reconstruction error.

    Args:
        ts_original: (B, T) original time series.
        ts_reconstructed: (B, T) reconstructed time series.
    Returns:
        dict with mse, mae, mre keys.
    """
    orig = ts_original.detach().cpu().numpy()
    recon = ts_reconstructed.detach().cpu().numpy()
    return {
        "roundtrip_mse": compute_mse(recon, orig),
        "roundtrip_mae": compute_mae(recon, orig),
        "roundtrip_mre": compute_mre(recon, orig),
    }


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def compute_all_ts_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute all time-series metrics at once.

    Args:
        pred: (B, T) or (B, n_samples, T) predicted time series.
        true: (B, T) ground truth.
    Returns:
        dict of metric name -> value.
    """
    if pred.ndim == 3:
        pred = np.median(pred, axis=1)

    return {
        "mse": compute_mse(pred, true),
        "mae": compute_mae(pred, true),
        "wape": compute_wape(pred, true),
        "mre": compute_mre(pred, true),
    }
