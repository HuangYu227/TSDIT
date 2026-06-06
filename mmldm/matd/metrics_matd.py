"""MATD-specific evaluation metrics and shape utilities.

The functions in this module are intentionally dependency-light.  They operate
on in-memory NumPy arrays so they can be reused from the evaluator, offline
analysis scripts, and small CPU tests without requiring the training stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ensure_btd(x: np.ndarray) -> np.ndarray:
    """Return a time-series array with shape ``(N, T, D)``.

    Args:
        x: Array shaped ``(N, T)`` or ``(N, T, D)``.  A light heuristic also
            accepts ``(N, D, T)`` when the variable axis is clearly smaller
            than the temporal axis.

    Returns:
        Float64 array with shape ``(N, T, D)``.

    Raises:
        ValueError: If the input rank is unsupported.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    if arr.ndim == 3:
        n, a, b = arr.shape
        if a <= 32 and b > a:
            return arr.transpose(0, 2, 1).reshape(n, b, a)
        return arr
    raise ValueError(f"expected (N,T) or (N,T,D), got shape {arr.shape}")


def ensure_nktd(x: np.ndarray) -> np.ndarray:
    """Return multi-sample generations with shape ``(N, K, T, D)``.

    Supports the MATD evaluator's current ``(N, T, D, K)`` layout and the
    canonical ``(N, K, T, D)`` layout.

    Args:
        x: Generated samples.

    Returns:
        Float64 array with shape ``(N, K, T, D)``.

    Raises:
        ValueError: If the input rank is unsupported.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 3:
        # Interpret as (N, K, T) when used for multi-sample generation.
        return arr[:, :, :, np.newaxis]
    if arr.ndim != 4:
        raise ValueError(f"expected (N,K,T,D) or (N,T,D,K), got shape {arr.shape}")

    n, a, b, c = arr.shape
    # Evaluator legacy: (N, T, D, K).  The variable axis D is usually small
    # and the sample axis K is larger than D.
    if b <= 32 and c > b:
        return arr.transpose(0, 3, 1, 2).reshape(n, c, a, b)
    # Canonical: (N, K, T, D).  Typical time-series D is small.
    if c <= 32:
        return arr
    # Fall back to canonical to avoid surprising transposes for long K sweeps.
    return arr


def _finite_values(x: np.ndarray) -> np.ndarray:
    vals = np.asarray(x, dtype=np.float64).reshape(-1)
    return vals[np.isfinite(vals)]


def _safe_float(value: Any) -> float:
    value = float(value)
    if not np.isfinite(value):
        return float("nan")
    return value


def _mse_per_sample(real: np.ndarray, gen: np.ndarray) -> np.ndarray:
    return np.nanmean((gen - real) ** 2, axis=(1, 2))


def _wape_per_sample(real: np.ndarray, gen: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = np.nansum(np.abs(gen - real), axis=(1, 2))
    den = np.nansum(np.abs(real), axis=(1, 2))
    return np.where(den > eps, num / den, np.nan)


def _macro_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(~np.isfinite(values)):
        return float("nan")
    return _safe_float(np.nanmean(values))


def _hist_prob(
    values: np.ndarray,
    bins: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    vals = _finite_values(values)
    if vals.size == 0:
        prob = np.full(len(bins) - 1, 1.0 / max(len(bins) - 1, 1), dtype=np.float64)
        return prob
    counts, _ = np.histogram(vals, bins=bins, density=False)
    prob = counts.astype(np.float64) + eps
    return prob / np.sum(prob)


def _combined_bins(real: np.ndarray, gen: np.ndarray, n_bins: int) -> np.ndarray:
    vals = np.concatenate([_finite_values(real), _finite_values(gen)])
    if vals.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.linspace(0.0, 1.0, n_bins + 1)
    if hi <= lo:
        pad = max(abs(lo), 1.0) * 1e-6
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, n_bins + 1)


def _js_divergence_prob(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return _safe_float(0.5 * (kl_pm + kl_qm))


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)


def _mmd_rbf_median(
    real_flat: np.ndarray,
    gen_flat: np.ndarray,
    max_samples: int = 1024,
) -> float:
    n = min(real_flat.shape[0], gen_flat.shape[0])
    if n == 0:
        return float("nan")
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=np.int64)
        real_flat = real_flat[idx]
        gen_flat = gen_flat[idx]
        n = max_samples
    combined = np.concatenate([real_flat, gen_flat], axis=0)
    d2_combined = _pairwise_sq_dists(combined, combined)
    positive = d2_combined[d2_combined > 1e-12]
    if positive.size == 0:
        return 0.0
    median_sq = float(np.median(positive))
    gamma = 1.0 / max(2.0 * median_sq, 1e-12)
    k_xx = np.exp(-gamma * _pairwise_sq_dists(real_flat, real_flat))
    k_yy = np.exp(-gamma * _pairwise_sq_dists(gen_flat, gen_flat))
    k_xy = np.exp(-gamma * _pairwise_sq_dists(real_flat, gen_flat))
    return _safe_float(max(float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()), 0.0))


def compute_robust_distribution_metrics(
    real_raw: np.ndarray,
    gen_raw: np.ndarray,
    n_bins: int = 50,
    mmd_max_samples: int = 1024,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute distribution diagnostics with stricter distance properties.

    These metrics are intentionally separate from the CaTSG-compatible
    histogram metrics.  They use shared probability-mass histograms, finite
    smoothing, and a median-heuristic RBF kernel so identical arrays return
    zero up to numerical precision and out-of-range generations do not create
    NaNs.
    """
    real = ensure_btd(real_raw)
    gen = ensure_btd(gen_raw)
    if real.shape != gen.shape:
        raise ValueError(f"shape mismatch: real {real.shape} vs gen {gen.shape}")

    n_bins = max(int(n_bins), 2)
    mdd_terms = []
    for d in range(real.shape[2]):
        for t in range(real.shape[1]):
            r = real[:, t, d]
            g = gen[:, t, d]
            bins = _combined_bins(r, g, n_bins)
            p = _hist_prob(r, bins, eps=eps)
            q = _hist_prob(g, bins, eps=eps)
            mdd_terms.append(np.mean(np.abs(p - q)))

    flat_bins = _combined_bins(real, gen, n_bins)
    p_flat = _hist_prob(real, flat_bins, eps=eps)
    q_flat = _hist_prob(gen, flat_bins, eps=eps)
    js_flat = _js_divergence_prob(p_flat, q_flat, eps=eps)

    real_flat_values = _finite_values(real)
    gen_flat_values = _finite_values(gen)
    if real_flat_values.size and gen_flat_values.size:
        n = min(real_flat_values.size, gen_flat_values.size)
        wasserstein1 = np.mean(
            np.abs(np.sort(real_flat_values)[:n] - np.sort(gen_flat_values)[:n])
        )
    else:
        wasserstein1 = np.nan

    real_seq = np.nan_to_num(real.reshape(real.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
    gen_seq = np.nan_to_num(gen.reshape(gen.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)
    mmd_median = _mmd_rbf_median(real_seq, gen_seq, max_samples=mmd_max_samples)

    real_min = np.nanmin(real) if np.isfinite(real).any() else np.nan
    real_max = np.nanmax(real) if np.isfinite(real).any() else np.nan
    if np.isfinite(real_min) and np.isfinite(real_max):
        outside = (gen < real_min) | (gen > real_max)
        outside_rate = np.nanmean(outside.astype(np.float64))
    else:
        outside_rate = np.nan

    return {
        "mdd_prob": _safe_float(np.nanmean(mdd_terms)),
        "js_flat": _safe_float(js_flat),
        "mmd_median": _safe_float(mmd_median),
        "wasserstein1_flat": _safe_float(wasserstein1),
        "outside_real_range_rate": _safe_float(outside_rate),
    }


def compute_scale_diagnostics(
    real_raw: np.ndarray,
    gen_raw: np.ndarray | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Summarize raw-scale pathologies that distort MSE/WAPE/J-FTSD.

    Args:
        real_raw: Ground-truth series shaped ``(N,T)`` or ``(N,T,D)``.
        gen_raw: Optional generated series with the same single-sample shape.
        eps: Denominator threshold used to identify tiny targets.

    Returns:
        Flat diagnostic dictionary suitable for metric logging.
    """
    real = ensure_btd(real_raw)
    real_finite = _finite_values(real)
    abs_sum = np.nansum(np.abs(real), axis=(1, 2))
    q = np.nanquantile(abs_sum, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])

    out = {
        "target_abs_sum_min": _safe_float(q[0]),
        "target_abs_sum_q01": _safe_float(q[1]),
        "target_abs_sum_q05": _safe_float(q[2]),
        "target_abs_sum_q25": _safe_float(q[3]),
        "target_abs_sum_q50": _safe_float(q[4]),
        "target_abs_sum_q75": _safe_float(q[5]),
        "target_abs_sum_q95": _safe_float(q[6]),
        "target_abs_sum_q99": _safe_float(q[7]),
        "target_abs_sum_max": _safe_float(q[8]),
        "target_tiny_denominator_ratio": _safe_float(np.mean(abs_sum <= eps)),
        "real_finite_ratio": _safe_float(np.isfinite(real).mean()),
        "real_raw_min": _safe_float(np.min(real_finite)) if real_finite.size else float("nan"),
        "real_raw_max": _safe_float(np.max(real_finite)) if real_finite.size else float("nan"),
        "real_raw_mean_abs": _safe_float(np.mean(np.abs(real_finite))) if real_finite.size else float("nan"),
        "real_raw_std": _safe_float(np.std(real_finite)) if real_finite.size else float("nan"),
    }

    if gen_raw is not None:
        gen = ensure_btd(gen_raw)
        if gen.shape != real.shape:
            raise ValueError(f"shape mismatch: real {real.shape} vs gen {gen.shape}")
        gen_finite = _finite_values(gen)
        real_min = out["real_raw_min"]
        real_max = out["real_raw_max"]
        if gen_finite.size and np.isfinite(real_min) and np.isfinite(real_max):
            outside = (gen < real_min) | (gen > real_max)
            outside_rate = np.nanmean(outside.astype(np.float64))
        else:
            outside_rate = np.nan
        out.update(
            {
                "gen_finite_ratio": _safe_float(np.isfinite(gen).mean()),
                "gen_raw_min": _safe_float(np.min(gen_finite)) if gen_finite.size else float("nan"),
                "gen_raw_max": _safe_float(np.max(gen_finite)) if gen_finite.size else float("nan"),
                "gen_raw_mean_abs": _safe_float(np.mean(np.abs(gen_finite))) if gen_finite.size else float("nan"),
                "gen_raw_std": _safe_float(np.std(gen_finite)) if gen_finite.size else float("nan"),
                "gen_outside_real_range_rate": _safe_float(outside_rate),
            }
        )
    return out


def _acf_features(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = ensure_btd(x)
    n, t, d = x.shape
    feats = np.zeros((n, max_lag, d), dtype=np.float64)
    centered = x - np.nanmean(x, axis=1, keepdims=True)
    var = np.nanmean(centered ** 2, axis=1)
    for lag in range(1, max_lag + 1):
        if lag >= t:
            break
        cov = np.nanmean(centered[:, :-lag, :] * centered[:, lag:, :], axis=1)
        feats[:, lag - 1, :] = np.where(var > 1e-12, cov / np.clip(var, 1e-12, None), 0.0)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _normalized_psd(x: np.ndarray) -> np.ndarray:
    x = ensure_btd(x)
    centered = x - np.nanmean(x, axis=1, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    psd = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    denom = np.sum(psd, axis=1, keepdims=True)
    return psd / np.clip(denom, 1e-12, None)


def compute_temporal_structure_metrics(
    real_raw: np.ndarray,
    gen_raw: np.ndarray,
) -> dict[str, float]:
    """Compute structure-sensitive single-sample time-series metrics."""
    real = ensure_btd(real_raw)
    gen = ensure_btd(gen_raw)
    if real.shape != gen.shape:
        raise ValueError(f"shape mismatch: real {real.shape} vs gen {gen.shape}")

    max_lag = max(1, min(10, real.shape[1] - 1))
    real_acf = _acf_features(real, max_lag)
    gen_acf = _acf_features(gen, max_lag)
    real_psd = _normalized_psd(real)
    gen_psd = _normalized_psd(gen)

    real_delta = np.diff(real, axis=1)
    gen_delta = np.diff(gen, axis=1)
    delta_mae = np.nanmean(np.abs(real_delta - gen_delta)) if real_delta.size else 0.0

    return {
        "acf_mae": _safe_float(np.nanmean(np.abs(real_acf - gen_acf))),
        "psd_l1": _safe_float(np.nanmean(np.abs(real_psd - gen_psd))),
        "psd_l2": _safe_float(np.sqrt(np.nanmean((real_psd - gen_psd) ** 2))),
        "delta_mae": _safe_float(delta_mae),
    }


def compute_multisample_metrics(
    real_raw: np.ndarray,
    gen_multi: np.ndarray,
) -> dict[str, float]:
    """Compute metrics that use all generated samples per condition."""
    real = ensure_btd(real_raw)
    gen = ensure_nktd(gen_multi)
    if gen.shape[0] != real.shape[0] or gen.shape[2:] != real.shape[1:]:
        raise ValueError(f"shape mismatch: real {real.shape} vs gen {gen.shape}")

    first = gen[:, 0]
    mean = np.nanmean(gen, axis=1)
    median = np.nanmedian(gen, axis=1)
    real_exp = real[:, np.newaxis, :, :]

    mse_k = np.nanmean((gen - real_exp) ** 2, axis=(2, 3))
    wape_k = np.stack([_wape_per_sample(real, gen[:, k]) for k in range(gen.shape[1])], axis=1)
    best_idx = np.nanargmin(mse_k, axis=1)
    best = gen[np.arange(gen.shape[0]), best_idx]

    err = np.abs(gen - real_exp)
    target_scale = np.nanmean(np.abs(real), axis=(1, 2), keepdims=True)
    tol = np.clip(0.1 * target_scale, 1e-8, None)[:, :, np.newaxis, :]
    covered = np.any(err <= tol, axis=1)

    # Energy score: E||X-y|| - 0.5 E||X-X'||.
    gen_flat = gen.reshape(gen.shape[0], gen.shape[1], -1)
    real_flat = real.reshape(real.shape[0], -1)
    dist_xy = np.linalg.norm(gen_flat - real_flat[:, np.newaxis, :], axis=-1).mean(axis=1)
    pairwise = gen_flat[:, :, np.newaxis, :] - gen_flat[:, np.newaxis, :, :]
    dist_xx = np.linalg.norm(pairwise, axis=-1).mean(axis=(1, 2))
    energy = dist_xy - 0.5 * dist_xx

    return {
        "first_mse": _macro_mean(_mse_per_sample(real, first)),
        "mean_mse": _macro_mean(_mse_per_sample(real, mean)),
        "median_mse": _macro_mean(_mse_per_sample(real, median)),
        "best_mse": _macro_mean(np.nanmin(mse_k, axis=1)),
        "first_wape": _macro_mean(_wape_per_sample(real, first)),
        "mean_wape": _macro_mean(_wape_per_sample(real, mean)),
        "median_wape": _macro_mean(_wape_per_sample(real, median)),
        "best_wape": _macro_mean(np.nanmin(wape_k, axis=1)),
        "coverage_10pct_mean_abs": _safe_float(np.nanmean(covered.astype(np.float64))),
        "sample_variance": _safe_float(np.nanmean(np.nanvar(gen, axis=1))),
        "energy_score": _macro_mean(energy),
        "best_sample_index_mean": _safe_float(np.nanmean(best_idx)),
        "best_temporal_delta_mae": compute_temporal_structure_metrics(real, best)["delta_mae"],
    }


def compute_retrieval_diagnostics(
    real_raw: np.ndarray,
    gen_multi: np.ndarray,
    top_k: int = 10,
) -> dict[str, float]:
    """Rank all generated samples for each real sample and report retrieval stats."""
    real = ensure_btd(real_raw)
    gen = ensure_nktd(gen_multi)
    if gen.shape[0] != real.shape[0] or gen.shape[2:] != real.shape[1:]:
        raise ValueError(f"shape mismatch: real {real.shape} vs gen {gen.shape}")

    n, k, _, _ = gen.shape
    real_flat = real.reshape(n, -1)
    gen_flat = gen.reshape(n * k, -1)
    real_norm = np.linalg.norm(real_flat, axis=1, keepdims=True)
    gen_norm = np.linalg.norm(gen_flat, axis=1, keepdims=True)
    real_normed = real_flat / np.clip(real_norm, 1e-8, None)
    gen_normed = gen_flat / np.clip(gen_norm, 1e-8, None)
    sims = real_normed @ gen_normed.T

    cutoff = min(int(top_k), n * k)
    ranks = np.full(n, np.inf, dtype=np.float64)
    top1_self = np.zeros(n, dtype=np.float64)
    relevant_in_top = np.zeros(n, dtype=np.float64)
    for i in range(n):
        order = np.argsort(-sims[i])
        relevant = set(range(i * k, i * k + k))
        top = order[:cutoff]
        top1_self[i] = 1.0 if order[0] in relevant else 0.0
        relevant_in_top[i] = 1.0 if any(idx in relevant for idx in top) else 0.0
        for pos, idx in enumerate(order, start=1):
            if idx in relevant:
                ranks[i] = float(pos)
                break

    reciprocal = np.where(np.isfinite(ranks) & (ranks <= cutoff), 1.0 / ranks, 0.0)
    return {
        "mrr_at_k": _safe_float(np.mean(reciprocal)),
        "mean_rank": _safe_float(np.mean(ranks[np.isfinite(ranks)])) if np.isfinite(ranks).any() else float("nan"),
        "median_rank": _safe_float(np.median(ranks[np.isfinite(ranks)])) if np.isfinite(ranks).any() else float("nan"),
        "top1_self_rate": _safe_float(np.mean(top1_self)),
        "relevant_in_topk_rate": _safe_float(np.mean(relevant_in_top)),
        "candidate_count": float(n * k),
        "samples_per_text": float(k),
        "top_k": float(cutoff),
    }
