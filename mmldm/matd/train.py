"""MATD training script for CSV / weather_npy datasets.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -u -m mmldm.matd.train \\
        --data_dir data/weather \\
        --datasets traffic \\
        --time_interval 96 \\
        --dataset_type csv \\
        --batch_size 1024 \\
        --lr 5e-5 \\
        --warmup_steps 1000 \\
        --total_steps 100000 \\
        --save_dir results/matd_traffic96 \\
        --log_dir logs/matd_traffic96
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from mmldm.matd import MATDModel, MATDConfig, MATDTrainer, MATDDataModule, MATDEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="MATD Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Dataset names (e.g. traffic)")
    parser.add_argument("--time_interval", type=int, default=96, help="Sequence length (24, 48, 96)")
    parser.add_argument("--dataset_type", type=str, default="csv", choices=["csv", "weather_npy"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--save_dir", type=str, default="results/matd")
    parser.add_argument("--log_dir", type=str, default="logs/matd")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--dit_depth", type=int, default=8)
    parser.add_argument("--dit_heads", type=int, default=8)
    parser.add_argument("--stage", type=str, default="all",
                        choices=["0", "1", "2", "3", "4", "all"],
                        help="Training stage: 0=joint, 1/2/3/4=single stage, all=sequential 1->4")
    parser.add_argument("--joint_epochs", type=int, default=200,
                        help="Epochs for --stage 0 joint training (default: 200)")
    parser.add_argument("--no_causal_guidance", action="store_true",
                        help="Disable causal guidance during inference/sampling")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    cfg = MATDConfig(
        embed_dim=args.embed_dim,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
        n_segments=8,
        min_tokens=8,
        timesteps=1000,
        ddim_steps=50,
        lr=args.lr,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        log_interval=14,
        use_causal_guidance_in_sampling=not args.no_causal_guidance,
    )

    dm = MATDDataModule(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        datasets=args.datasets,
        time_interval=args.time_interval,
        batch_size=cfg.batch_size,
    )

    model = MATDModel(cfg).cuda()
    trainer = MATDTrainer(model, cfg.__dict__, device="cuda")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # 4-stage training: autoencoder -> planner -> diffusion -> joint finetune
    epochs_per_stage = {1: 50, 2: 50, 3: 300, 4: 100}
    train_loaders = {s: train_loader for s in (1, 2, 3, 4)}
    val_loaders = {s: val_loader for s in (1, 2, 3, 4)}

    # Create evaluator for stage-4 metric tracking
    evaluator = MATDEvaluator(
        model=model,
        data_module=dm,
        device='cuda',
        n_samples_per_text=10,
        cfg_scale=cfg.cfg_scale,
        ddim_steps=cfg.ddim_steps,
    )

    if args.stage == "0":
        # Joint training mode: all components, single pass
        joint_epochs = args.joint_epochs
        trainer.train_stage(
            train_loaders[1], joint_epochs, stage=0,
            val_dataloader=val_loaders.get(1), evaluator=evaluator,
            save_dir=args.save_dir,
        )
        if args.save_dir:
            trainer.save_checkpoint(os.path.join(args.save_dir, "joint.pt"))
    elif args.stage in ("1", "2", "3", "4"):
        # Single stage training
        s = int(args.stage)
        loader = train_loaders.get(s, train_loaders[1])
        val_loader = val_loaders.get(s)
        trainer.train_stage(
            loader, epochs_per_stage.get(s, 10), stage=s,
            val_dataloader=val_loader, evaluator=evaluator if s == 4 else None,
            save_dir=args.save_dir,
        )
    else:
        # Original multi-stage sequential training (default "all")
        trainer.train_all_stages(
            train_loaders=train_loaders,
            epochs_per_stage=epochs_per_stage,
            val_loaders=val_loaders,
            save_dir=args.save_dir,
            evaluator=evaluator,
        )


if __name__ == "__main__":
    main()
