"""Unified evaluation metrics for TIGER paper.

All metrics are computed on (B, T, D) shaped arrays.
Raw-scale metrics are the primary metrics for paper tables.
Normalized metrics get _01 suffix for diagnostics only.

Metric sources:
- MSE, WAPE: T2S evaluation.py (per-sample macro averaging)
- MDD: CaTSG feature_distance_eval.py (HistoLoss, n_bins=20)
- KL: CaTSG feature_distance_eval.py (flat histogram, 50 bins)
- MMD: CaTSG feature_distance_eval.py (RBF kernel)
- C-FID: T2S evaluation.py (TS2Vec embeddings)
- J-FTSD: CaTSG feature_distance_eval.py / Time Weaver (contrastive + Frechet)
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy
from sklearn.metrics.pairwise import rbf_kernel


# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------

def ensure_btd(x: np.ndarray) -> np.ndarray:
    """Ensure array has shape (B, T, D).

    Handles:
    - (B, T) -> (B, T, 1)
    - (B, D, T) -> (B, T, D) if D < T (heuristic)
    - (B, T, D) -> unchanged
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    elif x.ndim == 3:
        B, dim1, dim2 = x.shape
        # If first dim after B is smaller, likely (B, D, T) -> transpose
        if dim1 < dim2 and dim1 <= 32:
            x = x.transpose(0, 2, 1)
    return x


def global_normalize(
    real_raw: np.ndarray,
    gen_raw: np.ndarray,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize both arrays to [0, 1] using global min/max.

    If global_min/global_max not provided, uses real_raw's global statistics.
    """
    if global_min is None:
        global_min = float(np.nanmin(real_raw))
    if global_max is None:
        global_max = float(np.nanmax(real_raw))
    g_range = max(global_max - global_min, 1e-8)
    real_01 = (real_raw - global_min) / g_range
    gen_01 = (gen_raw - global_min) / g_range
    return real_01, gen_01


# ---------------------------------------------------------------------------
# 1. MSE (T2S-compatible)
# ---------------------------------------------------------------------------

def calculate_mse_baseline(real: np.ndarray, gen: np.ndarray) -> float:
    """MSE: per-sample per-dimension average, then macro average.

    Matches T2S evaluation.py calculate_mse exactly.
    Input: (B, T, D)
    """
    real = ensure_btd(real)
    gen = ensure_btd(gen)
    assert real.shape == gen.shape, f"Shape mismatch: {real.shape} vs {gen.shape}"

    B, T, D = real.shape
    values = []
    for i in range(B):
        total = 0.0
        for j in range(D):
            total += np.mean((real[i, :, j] - gen[i, :, j]) ** 2)
        values.append(total / D)
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# 2. WAPE (T2S-compatible, macro)
# ---------------------------------------------------------------------------

def calculate_wape_baseline(real: np.ndarray, gen: np.ndarray) -> float:
    """WAPE: per-sample macro averaging.

    Matches T2S evaluation.py calculate_wape exactly.
    Input: (B, T, D)
    """
    real = ensure_btd(real)
    gen = ensure_btd(gen)
    assert real.shape == gen.shape, f"Shape mismatch: {real.shape} vs {gen.shape}"

    B, T, D = real.shape
    values = []
    for i in range(B):
        numerator = 0.0
        denominator = 0.0
        for j in range(D):
            numerator += np.sum(np.abs(real[i, :, j] - gen[i, :, j]))
            denominator += np.sum(np.abs(real[i, :, j]))
        if denominator != 0:
            values.append(numerator / denominator)
        else:
            values.append(np.nan)
    return float(np.nanmean(values))


# ---------------------------------------------------------------------------
# 3. MDD (CaTSG-compatible, n_bins=20)
# ---------------------------------------------------------------------------

def calculate_mdd_baseline(
    real: np.ndarray, gen: np.ndarray, n_bins: int = 20
) -> float:
    """MDD: Marginal Distribution Distance.

    Matches CaTSG HistoLoss / feature_distance_eval.py get_mdd_eval.
    Input: (B, T, D)
    """
    real = ensure_btd(real)
    gen = ensure_btd(gen)
    assert real.shape == gen.shape, f"Shape mismatch: {real.shape} vs {gen.shape}"

    B, T, D = real.shape
    losses = []

    for d in range(D):
        for t in range(T):
            real_col = real[:, t, d]
            gen_col = gen[:, t, d]

            # Build histogram from real
            a, b = real_col.min(), real_col.max()
            if b == a:
                b = a + 1e-2
            bins = np.linspace(a, b, n_bins + 1)
            delta = bins[1] - bins[0]
            loc = 0.5 * (bins[1:] + bins[:-1])

            # Real density
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)

            # Estimate density for gen at same bin centers
            dist = np.abs(gen_col[:, np.newaxis] - loc[np.newaxis, :])
            left_counter = ((delta / 2.0 - (loc[np.newaxis, :] - gen_col[:, np.newaxis])) == 0).astype(float)
            counter = (np.maximum(delta / 2.0 - dist, 0) > 0).astype(float) + left_counter
            density = counter.mean(axis=0) / delta

            abs_metric = np.abs(density - real_hist)
            losses.append(float(np.mean(abs_metric)))

    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# 4. KL divergence (flat, CaTSG-compatible)
# ---------------------------------------------------------------------------

def calculate_flat_kl_baseline(
    real: np.ndarray, gen: np.ndarray, n_bins: int = 50
) -> tuple[float, float]:
    """KL divergence on flattened distributions.

    Matches CaTSG feature_distance_eval.py cal_distances / get_flat_distance.
    Input: (B, T, D)
    Returns: (kl_value, out_of_range_rate)
    """
    real = ensure_btd(real)
    gen = ensure_btd(gen)

    real_flat = real.flatten()
    gen_flat = gen.flatten()
    real_flat = real_flat[~np.isnan(real_flat)]
    gen_flat = gen_flat[~np.isnan(gen_flat)]

    hist_real, edge_real = np.histogram(real_flat, density=True, bins=n_bins)
    hist_gen, _ = np.histogram(gen_flat, density=True, bins=edge_real)

    kl = float(entropy(hist_real, hist_gen + 1e-9))

    # Out-of-range rate
    out_of_range = np.sum((gen_flat < edge_real[0]) | (gen_flat > edge_real[-1]))
    out_of_range_rate = float(out_of_range / len(gen_flat)) if len(gen_flat) > 0 else 0.0

    return kl, out_of_range_rate


# ---------------------------------------------------------------------------
# 5. MMD (RBF kernel, CaTSG-compatible)
# ---------------------------------------------------------------------------

def calculate_mmd_rbf_baseline(real: np.ndarray, gen: np.ndarray) -> float:
    """MMD with RBF kernel.

    Matches CaTSG feature_distance_eval.py mmd_metric / calculate_mmd.
    Input: (B, T, D)
    """
    real = ensure_btd(real)
    gen = ensure_btd(gen)
    assert real.shape == gen.shape, f"Shape mismatch: {real.shape} vs {gen.shape}"

    B = real.shape[0]
    real_flat = real.reshape(B, -1).astype(np.float64)
    gen_flat = gen.reshape(B, -1).astype(np.float64)

    xx = rbf_kernel(real_flat, real_flat)
    yy = rbf_kernel(gen_flat, gen_flat)
    xy = rbf_kernel(real_flat, gen_flat)

    mmd = float(xx.mean() + yy.mean() - 2 * xy.mean())
    return max(mmd, 0.0)


# ---------------------------------------------------------------------------
# 6. C-FID (TS2Vec)
# ---------------------------------------------------------------------------

def calculate_cfid_ts2vec_baseline(
    real: np.ndarray, gen: np.ndarray, device: str = "cuda", seed: int = 42
) -> tuple[Optional[float], str, str]:
    """C-FID using TS2Vec embeddings.

    Matches T2S evaluation.py calculate_fid with TS2Vec.
    Input: (B, T, D)
    Returns: (fid_value, status, reason)
    """
    try:
        from ts2vec import TS2Vec
    except ImportError:
        return None, "failed", "ts2vec package not installed"

    real = ensure_btd(real)
    gen = ensure_btd(gen)
    assert real.shape == gen.shape, f"Shape mismatch: {real.shape} vs {gen.shape}"

    B, T, D = real.shape

    try:
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)

        model = TS2Vec(
            input_dims=D,
            device=device,
            output_dims=64,
        )
        model.fit(real, verbose=False)

        real_repr = model.encode(real, encoding_window="full_series")
        gen_repr = model.encode(gen, encoding_window="full_series")

        if real_repr.ndim == 3:
            real_repr = real_repr.reshape(B, -1)
        if gen_repr.ndim == 3:
            gen_repr = gen_repr.reshape(B, -1)

        mu_real = np.mean(real_repr, axis=0)
        mu_gen = np.mean(gen_repr, axis=0)
        sigma_real = np.cov(real_repr.T)
        sigma_gen = np.cov(gen_repr.T)

        diff = mu_real - mu_gen
        covmean = sqrtm(sigma_real @ sigma_gen)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(diff @ diff + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean))
        return max(fid, 0.0), "ok", ""

    except Exception as e:
        return None, "failed", str(e)


# ---------------------------------------------------------------------------
# 7. J-FTSD
# ---------------------------------------------------------------------------

def calculate_jftsd_baseline(
    real: np.ndarray,
    gen: np.ndarray,
    condition: np.ndarray,
    device: str = "cuda",
    emb_dim: int = 64,
    train_steps: int = 200,
    seed: int = 42,
) -> tuple[Optional[float], str, str]:
    """J-FTSD: Joint Frechet Time Series Distance.

    Matches CaTSG feature_distance_eval.py get_jftsd / Time Weaver.
    Input: real (B,T,D), gen (B,T,D), condition (B,L,C) or (B,C)
    Returns: (jftsd_value, status, reason)
    """
    if condition is None:
        return None, "failed", "missing condition input"

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F_torch
    except ImportError:
        return None, "failed", "PyTorch not available"

    real = ensure_btd(real)
    gen = ensure_btd(gen)
    assert real.shape == gen.shape, f"Shape mismatch: {real.shape} vs {gen.shape}"

    B, L, D_x = real.shape

    cond = np.asarray(condition, dtype=np.float32)
    if cond.ndim == 2:
        cond = cond[:, np.newaxis, :]
    D_c = cond.shape[-1]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    real_t = torch.tensor(real, dtype=torch.float32, device=device)
    gen_t = torch.tensor(gen, dtype=torch.float32, device=device)
    cond_t = torch.tensor(cond, dtype=torch.float32, device=device)

    class XEncoder(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
            )
        def forward(self, x):
            return self.encoder(x)

    class CEncoder(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim))
        def forward(self, c_data):
            B_c, L_c, D_c_inner = c_data.shape
            c_flat = c_data.reshape(-1, D_c_inner)
            c_encoded = self.encoder(c_flat)
            c_encoded = c_encoded.reshape(B_c, L_c, -1)
            return c_encoded.mean(dim=1)

    x_enc = XEncoder(L * D_x, emb_dim).to(device)
    c_enc = CEncoder(D_c, emb_dim).to(device)
    opt = torch.optim.Adam(list(x_enc.parameters()) + list(c_enc.parameters()), lr=1e-3)

    try:
        for _ in range(train_steps):
            idx = torch.randperm(B)
            z_t = F_torch.normalize(x_enc(real_t[idx]), dim=-1)
            z_m = F_torch.normalize(c_enc(cond_t[idx]), dim=-1)
            logits = (z_t @ z_m.T) / np.sqrt(emb_dim)
            labels = torch.arange(B, device=device)
            loss = (F_torch.cross_entropy(logits, labels) + F_torch.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            x_real_rep = x_enc(real_t)
            c_real_rep = c_enc(cond_t)
            x_gen_rep = x_enc(gen_t)

            z_real = torch.cat([x_real_rep, c_real_rep], dim=-1).cpu().numpy()
            z_gen = torch.cat([x_gen_rep, c_real_rep], dim=-1).cpu().numpy()

        mu_real = np.mean(z_real, axis=0)
        mu_gen = np.mean(z_gen, axis=0)
        sigma_real = np.cov(z_real.T)
        sigma_gen = np.cov(z_gen.T)

        eps = 1e-6
        sigma_real += np.eye(sigma_real.shape[0]) * eps
        sigma_gen += np.eye(sigma_gen.shape[0]) * eps

        diff = mu_real - mu_gen
        covmean = sqrtm(sigma_real @ sigma_gen)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        jftsd = float(diff @ diff + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean))
        return max(jftsd, 0.0), "ok", ""

    except Exception as e:
        return None, "failed", str(e)


# ---------------------------------------------------------------------------
# 8. Compute all metrics
# ---------------------------------------------------------------------------

def compute_all_unified_metrics(
    real_raw: np.ndarray,
    gen_raw: np.ndarray,
    condition: Optional[np.ndarray] = None,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
    device: str = "cuda",
    compute_01: bool = True,
    compute_cfid: bool = False,
    compute_jftsd: bool = True,
) -> dict:
    """Compute all unified metrics.

    Args:
        real_raw: (B, T, D) raw-scale real data
        gen_raw: (B, T, D) raw-scale generated data
        condition: optional (B, L, C) or (B, C) condition
        global_min: for [0,1] normalization (uses real_raw min if None)
        global_max: for [0,1] normalization (uses real_raw max if None)
        device: "cuda" or "cpu"
        compute_01: whether to compute [0,1] normalized metrics
        compute_cfid: whether to compute C-FID (requires ts2vec)
        compute_jftsd: whether to compute J-FTSD (requires condition)

    Returns:
        dict with all metrics
    """
    real_raw = ensure_btd(real_raw)
    gen_raw = ensure_btd(gen_raw)
    assert real_raw.shape == gen_raw.shape, f"Shape mismatch: {real_raw.shape} vs {gen_raw.shape}"

    B, T, D = real_raw.shape
    results = {
        "num_samples": B,
        "seq_len": T,
        "num_dims": D,
    }

    # --- Raw-scale metrics ---
    results["MSE_raw"] = calculate_mse_baseline(real_raw, gen_raw)
    results["WAPE_raw_macro"] = calculate_wape_baseline(real_raw, gen_raw)
    results["MDD_raw_20"] = calculate_mdd_baseline(real_raw, gen_raw, n_bins=20)

    kl_raw, oor_raw = calculate_flat_kl_baseline(real_raw, gen_raw, n_bins=50)
    results["KL_raw_flat"] = kl_raw
    results["out_of_range_rate_raw"] = oor_raw

    results["MMD_raw_rbf"] = calculate_mmd_rbf_baseline(real_raw, gen_raw)

    # --- C-FID ---
    if compute_cfid:
        cfid, cfid_status, cfid_reason = calculate_cfid_ts2vec_baseline(
            real_raw, gen_raw, device=device
        )
        results["C_FID_TS2Vec"] = cfid
        results["C_FID_TS2Vec_status"] = cfid_status
        results["C_FID_TS2Vec_reason"] = cfid_reason
    else:
        results["C_FID_TS2Vec"] = None
        results["C_FID_TS2Vec_status"] = "skipped"
        results["C_FID_TS2Vec_reason"] = "compute_cfid=False"

    # --- J-FTSD ---
    if compute_jftsd and condition is not None:
        jftsd, jftsd_status, jftsd_reason = calculate_jftsd_baseline(
            real_raw, gen_raw, condition, device=device
        )
        results["J_FTSD"] = jftsd
        results["J_FTSD_status"] = jftsd_status
        results["J_FTSD_reason"] = jftsd_reason
    else:
        results["J_FTSD"] = None
        results["J_FTSD_status"] = "skipped"
        results["J_FTSD_reason"] = "missing condition input" if condition is None else "compute_jftsd=False"

    # --- Normalized [0,1] metrics ---
    if compute_01:
        real_01, gen_01 = global_normalize(real_raw, gen_raw, global_min, global_max)
        results["MSE_01"] = calculate_mse_baseline(real_01, gen_01)
        results["WAPE_01_macro"] = calculate_wape_baseline(real_01, gen_01)
        results["MDD_01_20"] = calculate_mdd_baseline(real_01, gen_01, n_bins=20)

        kl_01, oor_01 = calculate_flat_kl_baseline(real_01, gen_01, n_bins=50)
        results["KL_01_flat"] = kl_01
        results["out_of_range_rate_01"] = oor_01

        results["MMD_01_rbf"] = calculate_mmd_rbf_baseline(real_01, gen_01)

    return results


def save_metrics(metrics: dict, path: str):
    """Save metrics dict to JSON file."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")
