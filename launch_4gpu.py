#!/usr/bin/env python
"""Launch TIGER paper experiments across 4 GPUs in parallel.

Each GPU runs independently via subprocess with ``CUDA_VISIBLE_DEVICES``.
Edit ``DATA_DIR`` below to point to your dataset before running.

Usage:
    python launch_4gpu.py

Or override the data directory on the command line:
    python launch_4gpu.py --data_dir /path/to/Weather
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# EDIT THIS to point to your dataset directory on the server
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/path/to/your/dataset"
# ──────────────────────────────────────────────────────────────────────

# Output root for all runs
OUT_DIR = "./runs/tiger_4gpu"

# Common training flags
EPOCHS = 500
BATCH_SIZE = 64
LR = 3e-4

# GPU → list of (variant_name, extra_flags) assigned to that GPU
# Each GPU can run multiple experiments sequentially, separated by --next
GPU_ASSIGNMENTS = {
    0: [
        ("text_image",  "--cond_mode text_image --cticd --csa_moe"),
    ],
    1: [
        ("text_only",   "--cond_mode text_only --cticd --no_csa_moe"),
        ("no_cticd",    "--cond_mode text_image --no_cticd --csa_moe"),
    ],
    2: [
        ("image_only",  "--cond_mode text_only --no_cticd --no_csa_moe --drop_text 1.0"),
        ("no_csa_moe",  "--cond_mode text_image --cticd --no_csa_moe"),
    ],
    3: [
        ("global_scale","--decode_scale global"),
        ("oracle_scale","--decode_scale oracle"),
    ],
}


def build_command(gpu_id: int, variant: str, extra_flags: str, data_dir: str,
                  out_dir: str) -> list[str]:
    """Build a single-experiment command line."""
    log_dir = os.path.join(out_dir, variant, "logs")
    save_dir = os.path.join(out_dir, variant, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Save variant config for reproducibility
    cfg = {
        "variant": variant,
        "gpu": gpu_id,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "extra_flags": extra_flags,
        "data_dir": data_dir,
    }
    with open(os.path.join(out_dir, variant, "variant.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    return [
        sys.executable, "-u", "-m", "mmldm.tiger.train",
        "--data_dir", data_dir,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--save_dir", save_dir,
        "--log_dir", log_dir,
    ] + extra_flags.split()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch 4-GPU TIGER experiments")
    parser.add_argument("--data_dir", default=DATA_DIR,
                        help="Path to dataset directory")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="Root output directory for all runs")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data_dir '{args.data_dir}' does not exist.")
        print("Edit DATA_DIR in launch_4gpu.py or pass --data_dir <path>")
        sys.exit(1)

    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # Build per-GPU command lists (multi-line for readability)
    gpu_processes: dict[int, list[str]] = {}
    for gpu_id, variants in GPU_ASSIGNMENTS.items():
        cmds_parts: list[str] = []
        for variant, extra_flags in variants:
            cmd = build_command(gpu_id, variant, extra_flags,
                                args.data_dir, args.out_dir)
            if cmds_parts:
                cmds_parts.append("&&")
            cmds_parts.append(" ".join(cmd))
        gpu_processes[gpu_id] = ["bash", "-c", " ".join(cmds_parts)]

    # Print plan
    print("=" * 70)
    print("TIGER 4-GPU Experiment Launcher")
    print("=" * 70)
    print(f"Data:      {args.data_dir}")
    print(f"Output:    {args.out_dir}")
    print(f"Epochs:    {EPOCHS}")
    print(f"Batch:     {BATCH_SIZE}")
    print()

    for gpu_id, variants in GPU_ASSIGNMENTS.items():
        names = ", ".join(v[0] for v in variants)
        print(f"  GPU {gpu_id}: {names}")
    print()

    if args.dry_run:
        print("[DRY RUN] Commands that would execute:")
        for gpu_id, proc in gpu_processes.items():
            print(f"\n  GPU {gpu_id}:")
            print(f"    {proc[-1]}")
        return

    # Launch all GPUs in parallel
    procs: dict[int, subprocess.Popen] = {}
    for gpu_id, cmd in gpu_processes.items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Launching GPU {gpu_id}  (CUDA_VISIBLE_DEVICES={gpu_id})")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs[gpu_id] = proc

    print(f"\nAll {len(procs)} GPUs launched.  Waiting for completion...\n")

    # Monitor and stream output
    while procs:
        for gpu_id, proc in list(procs.items()):
            if proc.poll() is not None:
                print(f"[GPU {gpu_id}] FINISHED with exit code {proc.returncode}")
                # Drain remaining output
                remaining = proc.stdout.read() if proc.stdout else ""
                if remaining:
                    for line in remaining.strip().split("\n"):
                        print(f"[GPU {gpu_id}] {line}")
                del procs[gpu_id]
            else:
                # Non-blocking read one line
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    print(f"[GPU {gpu_id}] {line.rstrip()}")
        time.sleep(0.5)

    print("\nAll GPUs finished.  Summary saved under:", args.out_dir)


if __name__ == "__main__":
    main()
