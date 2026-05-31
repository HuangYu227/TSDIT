"""Offline unified evaluation script for TIGER paper.

Usage:
    python -m mmldm.tiger.evaluation.run_unified_eval \\
        --real_raw path/to/real_raw.npy \\
        --gen_raw path/to/gen_raw.npy \\
        --condition path/to/condition.npy \\
        --out path/to/metrics.json \\
        --device cuda
"""

import argparse
import os
import sys

import numpy as np

from .unified_metrics import compute_all_unified_metrics, save_metrics


def main():
    parser = argparse.ArgumentParser(description="TIGER Unified Evaluation")
    parser.add_argument("--real_raw", type=str, required=True, help="Path to real_raw.npy (B, T, D)")
    parser.add_argument("--gen_raw", type=str, required=True, help="Path to gen_raw.npy (B, T, D)")
    parser.add_argument("--condition", type=str, default=None, help="Path to condition.npy (B, L, C) or (B, C)")
    parser.add_argument("--global_min", type=float, default=None, help="Global min for [0,1] normalization")
    parser.add_argument("--global_max", type=float, default=None, help="Global max for [0,1] normalization")
    parser.add_argument("--out", type=str, default="metrics.json", help="Output path for metrics.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for neural metrics")
    parser.add_argument("--no_01", action="store_true", help="Skip [0,1] normalized metrics")
    parser.add_argument("--cfid", action="store_true", help="Compute C-FID (requires ts2vec)")
    parser.add_argument("--no_jftsd", action="store_true", help="Skip J-FTSD computation")
    args = parser.parse_args()

    # Load data
    print(f"Loading real data from {args.real_raw}")
    real_raw = np.load(args.real_raw)
    print(f"  Shape: {real_raw.shape}")

    print(f"Loading gen data from {args.gen_raw}")
    gen_raw = np.load(args.gen_raw)
    print(f"  Shape: {gen_raw.shape}")

    condition = None
    if args.condition:
        print(f"Loading condition from {args.condition}")
        condition = np.load(args.condition)
        print(f"  Shape: {condition.shape}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_unified_metrics(
        real_raw=real_raw,
        gen_raw=gen_raw,
        condition=condition,
        global_min=args.global_min,
        global_max=args.global_max,
        device=args.device,
        compute_01=not args.no_01,
        compute_cfid=args.cfid,
        compute_jftsd=not args.no_jftsd and condition is not None,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples: {metrics['num_samples']}")
    print(f"  Seq len: {metrics['seq_len']}")
    print(f"  Dims:    {metrics['num_dims']}")
    print()
    print("  Raw-scale metrics (paper main table):")
    print(f"    MSE_raw:         {metrics['MSE_raw']:.6f}")
    print(f"    WAPE_raw_macro:  {metrics['WAPE_raw_macro']:.4f}")
    print(f"    MDD_raw_20:      {metrics['MDD_raw_20']:.4f}")
    print(f"    KL_raw_flat:     {metrics['KL_raw_flat']:.4f}")
    print(f"    out_of_range:    {metrics['out_of_range_rate_raw']:.4f}")
    print(f"    MMD_raw_rbf:     {metrics['MMD_raw_rbf']:.6f}")
    print()
    print(f"    C_FID_TS2Vec:    {metrics['C_FID_TS2Vec']} ({metrics['C_FID_TS2Vec_status']})")
    print(f"    J_FTSD:          {metrics['J_FTSD']} ({metrics['J_FTSD_status']})")

    if not args.no_01:
        print()
        print("  [0,1] normalized metrics (diagnostics):")
        print(f"    MSE_01:          {metrics['MSE_01']:.6f}")
        print(f"    WAPE_01_macro:   {metrics['WAPE_01_macro']:.4f}")
        print(f"    MDD_01_20:       {metrics['MDD_01_20']:.4f}")
        print(f"    KL_01_flat:      {metrics['KL_01_flat']:.4f}")
        print(f"    out_of_range_01: {metrics['out_of_range_rate_01']:.4f}")
        print(f"    MMD_01_rbf:      {metrics['MMD_01_rbf']:.6f}")

    # Save
    save_metrics(metrics, args.out)
    print(f"\nMetrics saved to {args.out}")


if __name__ == "__main__":
    main()
