#!/usr/bin/env python
"""Evaluate MMLDM-generated time series using VerbalTS's CTTP metrics.

Computes CTTP score, FID, and JFTSD on externally generated time series.
Must be run from the VerbalTS repo root (depends on models.cttp module).

Usage (in VerbalTS conda env):
    python eval_verbalts.py \
        --weather_data_dir ./datasets/Weather \
        --cttp_checkpoint ./save/Weather_cttp/clip_model_best.pth \
        --cttp_config ./save/Weather_cttp/model_configs.yaml \
        --generated_ts /path/to/weather_generated.npy \
        --cache_dir ./cache/Weather_mmldm_eval \
        --batch_size 128
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from scipy import linalg
from tqdm import tqdm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def load_cttp(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load pre-trained CTTP model."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.cttp.cttp_model import CTTP

    configs = yaml.safe_load(open(config_path))
    configs["device"] = device
    configs["text"]["device"] = device
    configs["ts"]["device"] = device

    model = CTTP(configs)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle both full checkpoint and state_dict-only
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def compute_or_load_stats(
    cttp,
    weather_data_dir: str,
    cache_dir: str,
    batch_size: int,
    device: str,
):
    """Load cached FID statistics or compute from training set."""
    os.makedirs(cache_dir, exist_ok=True)
    fid_mean_path = os.path.join(cache_dir, "fid_mean.npy")
    fid_cov_path = os.path.join(cache_dir, "fid_cov.npy")
    jftsd_mean_path = os.path.join(cache_dir, "jftsd_mean.npy")
    jftsd_cov_path = os.path.join(cache_dir, "jftsd_cov.npy")

    if all(os.path.exists(p) for p in [fid_mean_path, fid_cov_path, jftsd_mean_path, jftsd_cov_path]):
        ts_mean = np.load(fid_mean_path)
        ts_cov = np.load(fid_cov_path)
        joint_mean = np.load(jftsd_mean_path)
        joint_cov = np.load(jftsd_cov_path)
        print(f"Loaded cached FID stats from {cache_dir}")
        return ts_mean, ts_cov, joint_mean, joint_cov

    print("Computing FID statistics from training set...")
    train_ts = np.load(os.path.join(weather_data_dir, "train_ts.npy"))  # (N, L, C)
    train_caps = np.load(os.path.join(weather_data_dir, "train_text_caps.npy"), allow_pickle=True)

    if train_ts.ndim == 2:
        train_ts = train_ts[:, :, np.newaxis]

    all_ts_emb = []
    all_joint_emb = []

    with torch.no_grad():
        for start in tqdm(range(0, len(train_ts), batch_size), desc="Train stats"):
            end = min(start + batch_size, len(train_ts))
            ts = torch.from_numpy(train_ts[start:end]).float().to(device)
            ts_len = torch.full((ts.shape[0],), ts.shape[1], dtype=torch.long).to(device)

            # Get first caption per sample (consistent with WeatherDataset)
            caps = [str(c[0]) for c in train_caps[start:end]]

            ts_emb = cttp.get_ts_coemb(ts, ts_len)
            cap_emb = cttp.get_text_coemb(caps, None)

            all_ts_emb.append(ts_emb.cpu())
            all_joint_emb.append(torch.cat([ts_emb, cap_emb], dim=-1).cpu())

    all_ts_emb = torch.cat(all_ts_emb, dim=0).numpy()
    ts_mean = np.mean(all_ts_emb, axis=0)
    ts_cov = np.cov(all_ts_emb, rowvar=False)

    all_joint_emb = torch.cat(all_joint_emb, dim=0).numpy()
    joint_mean = np.mean(all_joint_emb, axis=0)
    joint_cov = np.cov(all_joint_emb, rowvar=False)

    np.save(fid_mean_path, ts_mean)
    np.save(fid_cov_path, ts_cov)
    np.save(jftsd_mean_path, joint_mean)
    np.save(jftsd_cov_path, joint_cov)
    print(f"Saved FID stats to {cache_dir}")
    return ts_mean, ts_cov, joint_mean, joint_cov


def main():
    parser = argparse.ArgumentParser(description="Evaluate TS using VerbalTS CTTP metrics")
    parser.add_argument("--weather_data_dir", type=str, required=True,
                        help="Path to VerbalTS Weather dataset (.npy files)")
    parser.add_argument("--cttp_checkpoint", type=str, required=True,
                        help="Path to CTTP clip_model_best.pth")
    parser.add_argument("--cttp_config", type=str, required=True,
                        help="Path to CTTP model_configs.yaml")
    parser.add_argument("--generated_ts", type=str, required=True,
                        help="Path to generated TS .npy file (N, L, C)")
    parser.add_argument("--cache_dir", type=str, default="./cache/Weather_mmldm_eval",
                        help="Cache directory for FID statistics")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load CTTP model
    print("Loading CTTP model...")
    cttp = load_cttp(args.cttp_checkpoint, args.cttp_config, str(device))

    # 2. Load/compute FID statistics
    ts_mean, ts_cov, joint_mean, joint_cov = compute_or_load_stats(
        cttp, args.weather_data_dir, args.cache_dir, args.batch_size, str(device),
    )

    # 3. Load generated TS
    gen_ts = np.load(args.generated_ts)  # (N, L, C)
    if gen_ts.ndim == 2:
        gen_ts = gen_ts[:, :, np.newaxis]
    print(f"Loaded generated TS: {gen_ts.shape}")

    # 4. Load test captions
    test_caps = np.load(
        os.path.join(args.weather_data_dir, "test_text_caps.npy"), allow_pickle=True,
    )
    print(f"Loaded test captions: {test_caps.shape}")

    # 5. Embed generated TS and text
    all_gen_emb = []
    all_joint_emb = []
    cttp_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(gen_ts), args.batch_size), desc="Gen stats"):
            end = min(start + args.batch_size, len(gen_ts))
            ts = torch.from_numpy(gen_ts[start:end]).float().to(device)
            ts_len = torch.full((ts.shape[0],), ts.shape[1], dtype=torch.long).to(device)
            caps = [str(c[0]) for c in test_caps[start:end]]

            gen_emb = cttp.get_ts_coemb(ts, ts_len)
            cap_emb = cttp.get_text_coemb(caps, None)

            all_gen_emb.append(gen_emb.cpu())
            all_joint_emb.append(torch.cat([gen_emb, cap_emb], dim=-1).cpu())
            cttp_sum += torch.mm(gen_emb, cap_emb.permute(1, 0)).trace().item()
            n_samples += gen_emb.shape[0]

    # 6. Compute metrics
    cttp_score = cttp_sum / n_samples

    all_gen_emb = torch.cat(all_gen_emb, dim=0).numpy()
    gen_mean = np.mean(all_gen_emb, axis=0)
    gen_cov = np.cov(all_gen_emb, rowvar=False)
    fid = calculate_frechet_distance(ts_mean, ts_cov, gen_mean, gen_cov)

    all_joint_emb = torch.cat(all_joint_emb, dim=0).numpy()
    joint_gen_mean = np.mean(all_joint_emb, axis=0)
    joint_gen_cov = np.cov(all_joint_emb, rowvar=False)
    jftsd = calculate_frechet_distance(joint_mean, joint_cov, joint_gen_mean, joint_gen_cov)

    # 7. Print results
    print(f"\n{'='*50}")
    print(f"VerbalTS Metrics (Weather)")
    print(f"{'='*50}")
    print(f"  CTTP:  {cttp_score:.6f}")
    print(f"  FID:   {fid:.6f}")
    print(f"  JFTSD: {jftsd:.6f}")
    print(f"{'='*50}")

    # Save to file
    import json
    results = {"CTTP": cttp_score, "FID": fid, "JFTSD": jftsd}
    result_path = os.path.join(os.path.dirname(args.generated_ts), "verbalts_metrics.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
