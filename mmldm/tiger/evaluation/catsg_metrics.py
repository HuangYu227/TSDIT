"""CaTSG-compatible evaluation metrics.

Implements the same metrics as CaTSG/utils/metrics/:
  - MDD (Marginal Distribution Distance)
  - KL (Kullback-Leibler Divergence)
  - MMD (Maximum Mean Discrepancy with RBF kernel)
  - J-FTSD (Joint Frechet Time Series Distance)

All functions accept numpy arrays of shape (N, T) or (N, T, 1).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b + 1e-2 if b == a else b
    bins = torch.linspace(a, b, n_bins + 1)
    delta = bins[1] - bins[0]
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


def compute_mdd(real: np.ndarray, gen: np.ndarray, n_bins: int = 20) -> float:
    """Marginal Distribution Distance (lower is better).

    Matches CaTSG HistoLoss: iterates over both variable dimension and time
    step.  Accepts (N, T) for univariate or (N, T, D) for multivariate.
    """
    real_t = torch.as_tensor(real, dtype=torch.float64)
    gen_t = torch.as_tensor(gen, dtype=torch.float64)

    # Ensure 3D: (N, T, D)
    if real_t.ndim == 2:
        real_t = real_t.unsqueeze(-1)
    if gen_t.ndim == 2:
        gen_t = gen_t.unsqueeze(-1)

    losses = []
    N, T, D = real_t.shape

    for i in range(D):
        for t in range(T):
            # Build histogram from real data (same as HistoLoss.__init__)
            real_ti = real_t[:, t, i].reshape(-1, 1)
            d_r, b = histogram_torch(real_ti, n_bins, density=True)
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1]).view(1, -1)

            # Compute density from generated data (same as HistoLoss.compute())
            x_ti = gen_t[:, t, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_ti - loc)
            # Use a tolerance-based boundary check instead of exact equality,
            # which is virtually never true for floating-point values.
            left_counter = ((delta / 2. - (loc - x_ti)).abs() < 1e-8).float()
            counter = (torch.relu(delta / 2. - dist) > 0.).float() + left_counter
            density = counter.mean(0) / delta
            abs_metric = torch.abs(density - d_r)
            losses.append(torch.mean(abs_metric))

    return float(torch.stack(losses).mean().item())


def compute_kl(real: np.ndarray, gen: np.ndarray, n_bins: int = 50) -> float:
    """KL divergence on flattened distributions (lower is better)."""
    real_flat = real.flatten()
    gen_flat = gen.flatten()
    # Joint NaN filtering: remove positions where EITHER array has NaN
    valid_mask = ~(np.isnan(real_flat) | np.isnan(gen_flat))
    real_flat = real_flat[valid_mask]
    gen_flat = gen_flat[valid_mask]

    n_dropped = np.sum(~valid_mask)
    if n_dropped > 0:
        print(f"WARNING: KL computation dropped {n_dropped} NaN values")

    hist_real, edge_real = np.histogram(real_flat, density=True, bins=n_bins)
    hist_gen, _ = np.histogram(gen_flat, density=True, bins=edge_real)
    return float(entropy(hist_real, hist_gen + 1e-9))


def compute_mmd(real: np.ndarray, gen: np.ndarray) -> float:
    """MMD with RBF kernel (lower is better)."""
    real_flat = real.reshape(real.shape[0], -1).astype(np.float64)
    gen_flat = gen.reshape(gen.shape[0], -1).astype(np.float64)

    xx = rbf_kernel(real_flat, real_flat)
    yy = rbf_kernel(gen_flat, gen_flat)
    xy = rbf_kernel(real_flat, gen_flat)
    return float(max(xx.mean() + yy.mean() - 2 * xy.mean(), 0.0))


# ---------------------------------------------------------------------------
# J-FTSD
# ---------------------------------------------------------------------------

class _XEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class _CEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, c_data):
        B, L, D_c = c_data.shape
        c_encoded = self.encoder(c_data.reshape(-1, D_c)).reshape(B, L, -1)
        return c_encoded.mean(dim=1)


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = mu1.detach().cpu().numpy(), mu2.detach().cpu().numpy()
    sigma1, sigma2 = sigma1.detach().cpu().numpy(), sigma2.detach().cpu().numpy()
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_jftsd(real, gen, cond, emb_dim=64, train_steps=200, device="cpu"):
    """Joint Frechet Time Series Distance (lower is better)."""
    real = torch.tensor(real, dtype=torch.float32, device=device, requires_grad=False)
    gen = torch.tensor(gen, dtype=torch.float32, device=device, requires_grad=False)
    cond = torch.tensor(cond, dtype=torch.float32, device=device, requires_grad=False)

    if real.dim() == 2:
        real = real.unsqueeze(-1)
    if gen.dim() == 2:
        gen = gen.unsqueeze(-1)
    if cond.dim() == 2:
        cond = cond.unsqueeze(1)  # (B, D) -> (B, 1, D) for CEncoder

    B, L, D_x = real.shape
    D = cond.shape[-1]

    x_enc = _XEncoder(L * D_x, emb_dim).to(device)
    c_enc = _CEncoder(D, emb_dim).to(device)
    opt = torch.optim.Adam(list(x_enc.parameters()) + list(c_enc.parameters()), lr=1e-3)

    for _ in range(train_steps):
        idx = torch.randperm(B)
        z_t = F.normalize(x_enc(real[idx]), dim=-1, eps=1e-8)
        z_m = F.normalize(c_enc(cond[idx]), dim=-1, eps=1e-8)
        logits = (z_t @ z_m.T) / np.sqrt(emb_dim)
        labels = torch.arange(B, device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        if not loss.requires_grad:
            raise RuntimeError(
                "J-FTSD training loss has no gradient — encoder parameters "
                "may be frozen. Ensure the call is not inside torch.no_grad()."
            )
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        z_real = torch.cat([x_enc(real), c_enc(cond)], dim=-1)
        z_gen = torch.cat([x_enc(gen), c_enc(cond)], dim=-1)
        return _frechet_distance(z_real.mean(0), torch.cov(z_real),
                                  z_gen.mean(0), torch.cov(z_gen))


def compute_all_catsg_metrics(real, gen, cond=None, device="cpu", include_jftsd=False):
    """Compute all CaTSG metrics. Returns dict with MDD, KL, MMD, [J-FTSD].

    Args:
        include_jftsd: If True, compute J-FTSD (slow, ~200 training steps).
                       Default False for training-time metrics; set True for
                       final evaluation only.
    """
    metrics = {
        "MDD": compute_mdd(real, gen),
        "KL": compute_kl(real, gen),
        "MMD": compute_mmd(real, gen),
    }
    if include_jftsd and cond is not None:
        metrics["J-FTSD"] = compute_jftsd(real, gen, cond, device=device)
    return metrics
