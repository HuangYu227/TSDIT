"""Small diagnostic helpers for MATD evaluation outputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def denorm_roundtrip_diagnostics(
    x_norm: np.ndarray,
    ts_min: np.ndarray,
    ts_max: np.ndarray,
) -> dict[str, float]:
    """Check per-sample min-max denormalization and renormalization drift.

    Args:
        x_norm: Normalized series shaped ``(N,T)`` or ``(N,T,D)``.
        ts_min: Per-sample raw minima shaped ``(N,)``.
        ts_max: Per-sample raw maxima shaped ``(N,)``.

    Returns:
        Flat diagnostics for logging.
    """
    x = np.asarray(x_norm, dtype=np.float64)
    if x.ndim == 2:
        x_work = x[:, :, np.newaxis]
    elif x.ndim == 3:
        x_work = x
    else:
        raise ValueError(f"expected (N,T) or (N,T,D), got shape {x.shape}")

    mins = np.asarray(ts_min, dtype=np.float64).reshape(-1)
    maxs = np.asarray(ts_max, dtype=np.float64).reshape(-1)
    if mins.shape[0] != x_work.shape[0] or maxs.shape[0] != x_work.shape[0]:
        raise ValueError(
            f"bounds length mismatch: x has N={x_work.shape[0]}, "
            f"ts_min={mins.shape[0]}, ts_max={maxs.shape[0]}"
        )

    scale = np.clip(maxs - mins, a_min=1e-8, a_max=None)
    raw = x_work * scale[:, None, None] + mins[:, None, None]
    roundtrip = (raw - mins[:, None, None]) / scale[:, None, None]
    err = np.abs(roundtrip - x_work)
    return {
        "denorm_roundtrip_max_abs": float(np.nanmax(err)) if err.size else 0.0,
        "denorm_roundtrip_mean_abs": float(np.nanmean(err)) if err.size else 0.0,
        "norm_finite_ratio": float(np.isfinite(x_work).mean()) if x_work.size else 1.0,
        "raw_finite_ratio": float(np.isfinite(raw).mean()) if raw.size else 1.0,
        "range_tiny_ratio": float(np.mean((maxs - mins) <= 1e-8)) if mins.size else 0.0,
        "norm_below_0_rate": float(np.nanmean((x_work < 0.0).astype(np.float64))) if x_work.size else 0.0,
        "norm_above_1_rate": float(np.nanmean((x_work > 1.0).astype(np.float64))) if x_work.size else 0.0,
    }


def _get_float(results: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(results.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def summarize_metric_pathologies(results: dict[str, Any]) -> dict[str, float]:
    """Flag common metric failure modes without changing metric values.

    Args:
        results: Flat evaluator result dictionary.

    Returns:
        ``1.0``/``0.0`` flags and supporting scalar summaries.
    """
    mse = _get_float(results, "MSE")
    wape = _get_float(results, "WAPE")
    mrr = _get_float(results, "MRR", _get_float(results, "retrieval/mrr_at_k"))
    tiny_den = _get_float(results, "diagnostics/target_tiny_denominator_ratio")
    jftsd_vals = [
        _get_float(results, "J-FTSD_text"),
        _get_float(results, "J-FTSD_planner"),
        _get_float(results, "J-FTSD_slots"),
    ]
    finite_jftsd = [v for v in jftsd_vals if np.isfinite(v)]
    max_jftsd = max(finite_jftsd) if finite_jftsd else float("nan")

    low_mse_high_wape = np.isfinite(mse) and np.isfinite(wape) and mse < 1e-2 and wape > 1.0
    high_mse_low_wape = np.isfinite(mse) and np.isfinite(wape) and mse > 100.0 and wape < 1.0
    mrr_saturated = np.isfinite(mrr) and mrr >= 0.999
    jftsd_extreme = np.isfinite(max_jftsd) and max_jftsd > 1_000.0
    denominator_risky = np.isfinite(tiny_den) and tiny_den > 0.01

    return {
        "pathology_low_mse_high_wape": float(low_mse_high_wape),
        "pathology_high_mse_low_wape": float(high_mse_low_wape),
        "pathology_mrr_saturated": float(mrr_saturated),
        "pathology_jftsd_extreme": float(jftsd_extreme),
        "pathology_tiny_denominator": float(denominator_risky),
        "pathology_max_jftsd": float(max_jftsd) if np.isfinite(max_jftsd) else float("nan"),
    }
