"""Generate train/val/test splits for TSFragment-600K at SampleID level.

Splits are saved to a JSON file for reproducibility. All fragments from
the same SampleID go to the same split to prevent data leakage (especially
for ETTh1 which has 5 sliding-window fragments per series).

Usage:
    python -m mmldm.data.split_dataset \\
        --data_dir "E:/Research/TSG/myTSG_V0/Three Levels Data/TSFragment-600K" \\
        --output splits.json \\
        --ratio 0.8 0.1 0.1 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = [
    "ETTh1", "ETTh1s", "ETTm1",
    "airquality", "electricity", "exchangerate", "traffic",
]
INTERVALS = [24, 48, 96]


def extract_sample_ids(data_dir: Path, dataset: str) -> set[int]:
    """Extract the intersection of SampleIDs across all intervals for a dataset."""
    sets = []
    for interval in INTERVALS:
        fname = f"embedding_cleaned_{dataset}_{interval}.csv"
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        sids = set()
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                sids.add(int(row[0]))
        sets.append(sids)
    if not sets:
        return set()
    # Intersection across all intervals
    return set.intersection(*sets)


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test splits")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="splits.json")
    parser.add_argument("--ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert abs(sum(args.ratio) - 1.0) < 1e-6, f"Ratios must sum to 1, got {sum(args.ratio)}"
    r_train, r_val, r_test = args.ratio

    import random

    rng = random.Random(args.seed)

    splits = {
        "seed": args.seed,
        "ratio": args.ratio,
        "datasets": {},
    }

    for ds in DATASETS:
        sids = sorted(extract_sample_ids(data_dir, ds))
        if not sids:
            print(f"  {ds}: no data found, skipping")
            continue

        rng.shuffle(sids)
        n = len(sids)
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        # test gets the remainder
        train_ids = sids[:n_train]
        val_ids = sids[n_train : n_train + n_val]
        test_ids = sids[n_train + n_val :]

        splits["datasets"][ds] = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        print(
            f"  {ds}: {n} samples -> "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)
    print(f"\nSplits saved to {output_path}")


if __name__ == "__main__":
    main()
