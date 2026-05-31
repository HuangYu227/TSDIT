"""Run paper ablations for TIGER.

Example:
    python -m TIGER_paper_ready.experiments.paper_ablation_runner \
        --data_dir ./Three\ Levels\ Data/Weather \
        --base_config configs/tiger_weather.json \
        --epochs 200 --variants text_only image_only text_image shuffled_text

This script creates independent run directories and checkpoint directories for
common ablations needed in the paper.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from ..train import TIGERTrainer, apply_cli_overrides, get_default_config, load_config


def configure_variant(cfg: dict, variant: str) -> dict:
    cfg = copy.deepcopy(cfg)
    ccfg = cfg.setdefault("condition", {})
    variant = variant.lower()
    if variant == "text_only":
        ccfg["cond_mode"] = "text_only"
        cfg["cticd"]["enabled"] = True
    elif variant == "image_only":
        ccfg["cond_mode"] = "text_image"
        ccfg["drop_text_prob"] = 1.0
        ccfg["drop_image_prob"] = 0.0
        ccfg["drop_both_prob"] = 0.0
    elif variant == "text_image":
        ccfg["cond_mode"] = "text_image"
        ccfg["drop_text_prob"] = 0.10
        ccfg["drop_image_prob"] = 0.10
        ccfg["drop_both_prob"] = 0.10
    elif variant == "no_cticd":
        ccfg["cond_mode"] = "text_image"
        cfg.setdefault("cticd", {})["enabled"] = False
    elif variant == "no_csa_moe":
        ccfg["cond_mode"] = "text_image"
        cfg.setdefault("csa_moe", {})["enabled"] = False
    elif variant == "oracle_scale":
        ccfg["decode_scale_mode"] = "per_sample_oracle"
    elif variant == "global_scale":
        ccfg["decode_scale_mode"] = "global"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--base_config", default="")
    parser.add_argument("--out_dir", default="./runs/tiger_paper_ablation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["text_only", "image_only", "text_image", "no_cticd", "no_csa_moe", "global_scale"],
    )
    args = parser.parse_args()

    base = load_config(args.base_config) if args.base_config else get_default_config()
    base["data_dir"] = args.data_dir
    if args.epochs is not None:
        base["epochs"] = args.epochs
    if args.batch_size is not None:
        base["batch_size"] = args.batch_size
    if args.lr is not None:
        base["lr"] = args.lr
    if args.seed is not None:
        base["seed"] = args.seed

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for variant in args.variants:
        cfg = configure_variant(base, variant)
        cfg["log_dir"] = str(out / variant / "logs")
        cfg["save_dir"] = str(out / variant / "checkpoints")
        os.makedirs(cfg["log_dir"], exist_ok=True)
        os.makedirs(cfg["save_dir"], exist_ok=True)
        with open(out / variant / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        trainer = TIGERTrainer(cfg)
        trainer.train()
        summary.append({"variant": variant, "best_val_loss": trainer.best_val_loss})
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
