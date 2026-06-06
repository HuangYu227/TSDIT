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
    parser.add_argument("--normalization", type=str, default="per_sample",
                        choices=["per_sample", "global_minmax"],
                        help="Time-series normalization protocol used by MATDDataModule")
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
    parser.add_argument("--architecture", type=str, default=None,
                        choices=["core_v2", "full"],
                        help="Model architecture path: core_v2=PlanFormer-DiT core, full=legacy full MATD")
    parser.add_argument("--joint_epochs", type=int, default=200,
                        help="Epochs for --stage 0 joint training (default: 200)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochs for single-stage training (overrides default per-stage epochs)")
    parser.add_argument("--no_causal_guidance", action="store_true",
                        help="Disable causal guidance during inference/sampling")
    parser.add_argument("--causal_guidance", action="store_true",
                        help="Enable sampling-time causal guidance")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="Override classifier-free guidance scale (default: 5.0)")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="Override DDIM sampling steps used by evaluation (default: 50)")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Run evaluator every N epochs for stage 0/4 training")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta used by evaluation sampling (default: 0.0)")
    parser.add_argument("--p_drop_text", type=float, default=None,
                        help="Override CFG text dropout probability during denoiser training")
    parser.add_argument("--use_oracle_meta_prob", type=float, default=None,
                        help="Probability of using oracle patch metadata during training")
    parser.add_argument("--lambda_diffusion", type=float, default=None,
                        help="Override diffusion bridge group weight (default: 1.0)")
    parser.add_argument("--lambda_x0", type=float, default=None,
                        help="Override patch field group weight (default: 0.2)")
    parser.add_argument("--lambda_plan", type=float, default=None,
                        help="Override text layout group weight (default: 0.5)")
    parser.add_argument("--planner_mode", type=str, default=None, choices=["latent", "layout"],
                        help="Planner mode: latent temporal plan or legacy layout regression")
    parser.add_argument("--planner_depth", type=int, default=None,
                        help="Number of PlanFormer blocks in latent planner mode")
    parser.add_argument("--layout_source", type=str, default=None, choices=["canonical", "oracle", "predicted"],
                        help="Layout metadata source for latent planner mode")
    parser.add_argument("--lambda_plan_stats", type=float, default=None)
    parser.add_argument("--lambda_plan_contrast", type=float, default=None)
    parser.add_argument("--lambda_plan_kl", type=float, default=None)
    parser.add_argument("--lambda_plan_layout", type=float, default=None)
    parser.add_argument("--plan_residual_scale", type=float, default=None)
    parser.add_argument("--plan_context_scale", type=float, default=None)
    # Sub-loss weights
    parser.add_argument("--lambda_latent_x0", type=float, default=None)
    parser.add_argument("--lambda_latent_cos", type=float, default=None)
    parser.add_argument("--lambda_endpoint_recon", type=float, default=None)
    parser.add_argument("--lambda_endpoint_moment", type=float, default=None)
    parser.add_argument("--lambda_endpoint_delta", type=float, default=None)
    parser.add_argument("--lambda_endpoint_acf", type=float, default=None)
    parser.add_argument("--endpoint_acf_lags", type=int, default=None)
    parser.add_argument("--endpoint_latent_clip", type=float, default=None)
    parser.add_argument("--lambda_delta", type=float, default=None)
    parser.add_argument("--lambda_curvature", type=float, default=None)
    parser.add_argument("--lambda_fft", type=float, default=None)
    parser.add_argument("--lambda_range", type=float, default=None)
    parser.add_argument("--lambda_causal", type=float, default=None)
    parser.add_argument("--lambda_moe", type=float, default=None)
    parser.add_argument("--lambda_scci", type=float, default=None)
    parser.add_argument("--min_snr_gamma", type=float, default=None,
                        help="Override Min-SNR gamma (default: 5.0)")
    parser.add_argument("--decoder_field_blocks", type=int, default=None,
                        help="Override hybrid decoder residual field blocks (default: 3)")
    parser.add_argument("--decoder_siren_omega", type=float, default=None,
                        help="Override hybrid decoder SIREN omega (default: 18.0)")
    parser.add_argument("--decoder_siren_scale", type=float, default=None,
                        help="Override hybrid decoder SIREN residual scale (default: 0.1)")
    parser.add_argument("--decoder_output_activation", type=str, default=None,
                        choices=["none", "sigmoid", "clamp"],
                        help="Optional decoder output activation/range diagnostic")
    parser.add_argument("--decoder_latent_noise_std", type=float, default=None,
                        help="Gaussian latent noise added before clean-path decoder reconstruction")
    parser.add_argument("--loss_balance_enabled", action="store_true",
                        help="Enable observe-only gradient conflict diagnostics")
    parser.add_argument("--loss_balance_interval", type=int, default=50,
                        help="Step interval for observe-only gradient diagnostics")
    parser.add_argument("--loss_balance_probe_components", type=str, nargs="*", default=None,
                        help="Optional parameter-name substrings to probe for gradient diagnostics")
    parser.add_argument("--module_grad_interval", type=int, default=None,
                        help="Log per-module gradient norms every N optimizer steps (0 disables)")
    parser.add_argument("--ema", dest="ema_enabled", action="store_true",
                        help="Enable EMA weights for evaluation/checkpointing")
    parser.add_argument("--no_ema", dest="ema_enabled", action="store_false",
                        help="Disable EMA weights for evaluation/checkpointing")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="EMA decay if EMA is enabled")
    parser.add_argument("--condition_sensitivity_eval", dest="condition_sensitivity_eval", action="store_true",
                        help="Enable shuffled-text condition sensitivity diagnostics during evaluation")
    parser.add_argument("--no_condition_sensitivity_eval", dest="condition_sensitivity_eval", action="store_false",
                        help="Disable shuffled-text condition sensitivity diagnostics during evaluation")
    parser.set_defaults(condition_sensitivity_eval=None)
    parser.set_defaults(ema_enabled=None)
    args = parser.parse_args()
    if args.no_causal_guidance and args.causal_guidance:
        parser.error("--no_causal_guidance and --causal_guidance are mutually exclusive")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    causal_guidance_override = None
    if args.causal_guidance:
        causal_guidance_override = True
    elif args.no_causal_guidance:
        causal_guidance_override = False

    cfg_overrides = dict(
        embed_dim=args.embed_dim,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
        n_segments=8,
        min_tokens=8,
        timesteps=1000,
        ddim_steps=50 if args.ddim_steps is None else args.ddim_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        log_interval=14,
    )
    optional_overrides = {
        "use_causal_guidance_in_sampling": causal_guidance_override,
        "architecture": args.architecture,
        "condition_sensitivity_eval": args.condition_sensitivity_eval,
        "module_grad_interval": args.module_grad_interval,
        "ema_enabled": args.ema_enabled,
        "ema_decay": args.ema_decay,
        "cfg_scale": args.cfg_scale,
        "p_drop_text": args.p_drop_text,
        "use_oracle_meta_prob": args.use_oracle_meta_prob,
        "lambda_diffusion": args.lambda_diffusion,
        "lambda_x0": args.lambda_x0,
        "lambda_plan": args.lambda_plan,
        "planner_mode": args.planner_mode,
        "planner_depth": args.planner_depth,
        "layout_source": args.layout_source,
        "lambda_plan_stats": args.lambda_plan_stats,
        "lambda_plan_contrast": args.lambda_plan_contrast,
        "lambda_plan_kl": args.lambda_plan_kl,
        "lambda_plan_layout": args.lambda_plan_layout,
        "plan_residual_scale": args.plan_residual_scale,
        "plan_context_scale": args.plan_context_scale,
        "min_snr_gamma": args.min_snr_gamma,
        # Sub-loss weights
        "lambda_latent_x0": args.lambda_latent_x0,
        "lambda_latent_cos": args.lambda_latent_cos,
        "lambda_endpoint_recon": args.lambda_endpoint_recon,
        "lambda_endpoint_moment": args.lambda_endpoint_moment,
        "lambda_endpoint_delta": args.lambda_endpoint_delta,
        "lambda_endpoint_acf": args.lambda_endpoint_acf,
        "endpoint_acf_lags": args.endpoint_acf_lags,
        "endpoint_latent_clip": args.endpoint_latent_clip,
        "lambda_delta": args.lambda_delta,
        "lambda_curvature": args.lambda_curvature,
        "lambda_fft": args.lambda_fft,
        "lambda_range": args.lambda_range,
        "lambda_causal": args.lambda_causal,
        "lambda_moe": args.lambda_moe,
        "lambda_scci": args.lambda_scci,
        # Decoder
        "decoder_field_blocks": args.decoder_field_blocks,
        "decoder_siren_omega": args.decoder_siren_omega,
        "decoder_siren_scale": args.decoder_siren_scale,
        "decoder_output_activation": args.decoder_output_activation,
        "decoder_latent_noise_std": args.decoder_latent_noise_std,
    }
    cfg_overrides.update({k: v for k, v in optional_overrides.items() if v is not None})
    cfg = MATDConfig(**cfg_overrides)
    trainer_config = dict(cfg.__dict__)
    trainer_config["loss_balance"] = {
        "enabled": args.loss_balance_enabled,
        "interval": args.loss_balance_interval,
        "probe_components": args.loss_balance_probe_components,
    }

    dm = MATDDataModule(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        datasets=args.datasets,
        time_interval=args.time_interval,
        batch_size=cfg.batch_size,
        normalization=args.normalization,
    )
    logging.getLogger(__name__).info(
        "MATD data protocol: dataset_type=%s datasets=%s time_interval=%d normalization=%s",
        args.dataset_type, ",".join(args.datasets), args.time_interval, args.normalization,
    )

    model = MATDModel(cfg).cuda()
    trainer = MATDTrainer(model, trainer_config, device="cuda")

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
        eta=args.eta,
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
        n_epochs = args.epochs if args.epochs is not None else epochs_per_stage.get(s, 10)
        trainer.train_stage(
            loader, n_epochs, stage=s,
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
