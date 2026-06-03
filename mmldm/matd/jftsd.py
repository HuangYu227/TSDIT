# J-FTSD (Joint Fréchet Time Series Distance)
# Reference: Narasimhan et al., "Time Weaver: A Conditional Time Series
#            Generation Model", 2024.  https://arxiv.org/abs/2403.02682
#
# Copied verbatim from CaTSG (catsg_eval.py) to ensure metric fairness.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm


class XEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
    def forward(self, x): return self.encoder(x)


class CEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim))

    def forward(self, c_data):  # c_data: (B, L, D_c) or (B, D_c)
        if c_data.dim() == 2:
            c_data = c_data.unsqueeze(1)  # (B, D_c) -> (B, 1, D_c)
        elif c_data.dim() != 3:
            raise ValueError(f"c_data must be (B,D) or (B,L,D), got {tuple(c_data.shape)}")
        B, L, D_c = c_data.shape
        c_flat = c_data.reshape(-1, D_c)            # (B * L, D_c)
        c_encoded = self.encoder(c_flat)         # (B * L, emb_dim)
        c_encoded = c_encoded.reshape(B, L, -1)     # (B, L, emb_dim)
        return c_encoded.mean(dim=1)             # (B, emb_dim)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Frechet Distance between two multivariate Gaussians
    """
    # Convert to CPU numpy arrays for scipy
    mu1 = mu1.detach().cpu().numpy()
    mu2 = mu2.detach().cpu().numpy()
    sigma1 = sigma1.detach().cpu().numpy()
    sigma2 = sigma2.detach().cpu().numpy()

    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2

    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    # Final FTSD
    ftsd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(ftsd)


def get_jftsd(x_real, c_real, x_gen, emb_dim=64, train_steps=200, device="cpu"):
    """
    J-FTSD (proposed in Time Weaver paper: https://arxiv.org/abs/2403.02682)
    compares the joint distributions of real vs. generated samples in a learned embedding space.
    J-FTSD is the Frechet distance: ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2})
    (Eq. 6 in Time Weaver paper)

    References
    ----------
    Narasimhan et al., "Time Weaver: A Conditional Time Series Generation Model", 2024.
    """

    if not isinstance(x_real, torch.Tensor):
        x_real = torch.tensor(x_real, dtype=torch.float32).to(device)
    else:
        x_real = x_real.clone().detach().to(device).float()

    if not isinstance(x_gen, torch.Tensor):
        x_gen = torch.tensor(x_gen, dtype=torch.float32).to(device)
    else:
        x_gen = x_gen.clone().detach().to(device).float()

    if not isinstance(c_real, torch.Tensor):
        c_real = torch.tensor(c_real, dtype=torch.float32).to(device)
    else:
        c_real = c_real.clone().detach().to(device).float()

    B, L, D_x = x_real.shape
    D = c_real.shape[-1]
    x_evl_encoder = XEncoder(in_dim=L * D_x, out_dim=emb_dim).to(device)
    c_evl_encoder = CEncoder(in_dim=D, out_dim=emb_dim).to(device)
    optimizer = torch.optim.Adam(list(x_evl_encoder.parameters()) + list(c_evl_encoder.parameters()), lr=1e-3)

    for _ in range(train_steps):
        idx = torch.randperm(B)
        x = x_real[idx]
        c = c_real[idx]

        z_t = x_evl_encoder(x)
        z_m = c_evl_encoder(c)

        z_t = F.normalize(z_t, dim=-1)
        z_m = F.normalize(z_m, dim=-1)
        logits = (z_t @ z_m.T) / np.sqrt(emb_dim)

        labels = torch.arange(B).to(device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # get embeddings
    with torch.no_grad():
        # get embeddings for real part
        x_real_rep = x_evl_encoder(x_real)
        c_real_rep = c_evl_encoder(c_real)

        z_real = torch.cat([x_real_rep, c_real_rep], dim=-1)
        mu_real = z_real.mean(0)
        sigma_real = torch.cov(z_real.T)

        # get embeddings for generated part
        x_gen_rep = x_evl_encoder(x_gen)

        z_gen_xc = torch.cat([x_gen_rep, c_real_rep], dim=-1)
        mu_gen = z_gen_xc.mean(0)
        sigma_gen = torch.cov(z_gen_xc.T)

        j_ftsd = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        return j_ftsd
