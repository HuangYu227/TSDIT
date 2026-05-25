# Copyright 2026 MMLDM Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage 1 training: Spectral Dual-Latent VAE.

Loss = L_recon + beta * KL + gamma_spectral * L_spectral + gamma_tclr * L_TCLR

Innovations:
- A: Spectral Dual-Latent (trend + residual)
- C: Temporal Contrastive Latent Regularization (TCLR)
- Engineering: Spectral reconstruction loss, KL annealing

Usage:
    python -m mmldm.training_stage1 --data_dir ./data --epochs 100
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .configuration_mmldm import MMLDMVAEConfig
from .data.tsfragment_dataset import CollateFn, TSFragmentDataset
from .data.weather_dataset import WeatherCollateFn, WeatherDataset
from .modeling_mmldm_vae import MMLDMVAEModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_kl_beta(epoch: int, kl_anneal_epochs: int, kl_anneal_start: float, kl_anneal_end: float) -> float:
    if kl_anneal_epochs <= 0:
        return kl_anneal_end
    if epoch >= kl_anneal_epochs:
        return kl_anneal_end
    progress = epoch / kl_anneal_epochs
    return kl_anneal_start + (kl_anneal_end - kl_anneal_start) * progress


def compute_kl_loss(latent_dists: tuple[list, list]) -> torch.Tensor:
    """KL for dual distributions (trend + residual)."""
    trend_dists, residual_dists = latent_dists
    kl = 0.0
    for d in trend_dists:
        kl = kl + d.kl()
    for d in residual_dists:
        kl = kl + d.kl()
    return kl / (len(trend_dists) + len(residual_dists))


def train_step(
    model: MMLDMVAEModel,
    batch: dict,
    beta: float = 1e-6,
    gamma_spectral: float = 0.1,
    gamma_tclr: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> dict:
    ot = batch["ot"].to(device)
    ot_lengths = batch["ot_lengths"]
    ot_list = [ot[i, :ot_lengths[i]] for i in range(ot.shape[0])]

    output = model(ot_list, tclr_weight=gamma_tclr)

    target = torch.cat(ot_list, dim=0).unsqueeze(0)
    recon = output["recon"][:, : target.shape[1], :]

    l_recon = F.mse_loss(recon, target)
    l_kl = compute_kl_loss(output["latent_dists"])
    l_spectral = output.get("spectral_loss", torch.tensor(0.0, device=device))
    l_tclr = output.get("tclr_loss", torch.tensor(0.0, device=device))

    total = l_recon + beta * l_kl + gamma_spectral * l_spectral + gamma_tclr * l_tclr

    return {"total": total, "recon": l_recon, "kl": l_kl, "spectral": l_spectral, "tclr": l_tclr}


def main():
    parser = argparse.ArgumentParser(description="MMLDM Stage 1: Spectral Dual-Latent VAE")
    parser.add_argument("--dataset_type", type=str, default="csv",
                        choices=["csv", "weather_npy"],
                        help="Dataset format: csv (TSFragment-600K) or weather_npy (VerbalTS Weather)")
    parser.add_argument("--weather_data_dir", type=str, default=None,
                        help="Path to Weather .npy data (required when --dataset_type weather_npy)")
    parser.add_argument("--ts_channels", type=int, default=1,
                        help="Number of TS channels/variables (1 for univariate, 21 for Weather)")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ETTh1"])
    parser.add_argument("--time_intervals", type=int, nargs="+", default=[24])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--kl_anneal_epochs", type=int, default=10)
    parser.add_argument("--kl_anneal_start", type=float, default=0.0)
    parser.add_argument("--kl_anneal_end", type=float, default=1e-5)
    parser.add_argument("--gamma_spectral", type=float, default=0.1, help="Spectral loss weight")
    parser.add_argument("--gamma_tclr", type=float, default=0.1, help="TCLR loss weight")
    parser.add_argument("--fft_cutoff_ratio", type=float, default=0.3, help="FFT cutoff ratio for trend/residual split")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_conv_layers", type=int, default=4)
    parser.add_argument("--encoder_blocks", type=int, default=6)
    parser.add_argument("--decoder_blocks", type=int, default=6)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/stage1_v2")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume finetuning from checkpoint")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    config = MMLDMVAEConfig(
        ts_channels=args.ts_channels, dim=args.dim, ffn_dim=args.dim * 4,
        latent_dim=args.latent_dim, num_heads=args.num_heads,
        head_dim=args.dim // args.num_heads, num_conv_layers=args.num_conv_layers,
        encoder_num_blocks=args.encoder_blocks, decoder_num_blocks=args.decoder_blocks,
        block_size=args.block_size, kl_anneal_start=args.kl_anneal_start,
        kl_anneal_end=args.kl_anneal_end, kl_anneal_epochs=args.kl_anneal_epochs,
        fft_cutoff_ratio=args.fft_cutoff_ratio,
    )

    if args.dataset_type == "weather_npy":
        if args.weather_data_dir is None:
            raise ValueError("--weather_data_dir is required when --dataset_type weather_npy")
        train_ds = WeatherDataset(weather_data_dir=args.weather_data_dir, split="train",
                                  max_samples=args.max_samples)
        val_ds = WeatherDataset(weather_data_dir=args.weather_data_dir, split="valid",
                                max_samples=args.max_samples)
        collate = WeatherCollateFn()
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)} (Weather .npy)")
    elif args.split_file is not None:
        if not args.data_dir:
            raise ValueError("--data_dir is required when --dataset_type csv")
        train_ds = TSFragmentDataset(data_dir=args.data_dir, datasets=args.datasets,
                                     time_intervals=args.time_intervals, max_samples=args.max_samples,
                                     split="train", split_file=args.split_file)
        val_ds = TSFragmentDataset(data_dir=args.data_dir, datasets=args.datasets,
                                   time_intervals=args.time_intervals, split="val", split_file=args.split_file)
        collate = CollateFn()
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    else:
        dataset = TSFragmentDataset(data_dir=args.data_dir, datasets=args.datasets,
                                    time_intervals=args.time_intervals, max_samples=args.max_samples)
        val_size = min(len(dataset) // 10, 1000)
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
        collate = CollateFn()
        print(f"Loaded {len(dataset)} samples")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model = MMLDMVAEModel(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {args.resume_checkpoint} (epoch {ckpt.get('epoch', '?')})")
    else:
        model = MMLDMVAEModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay_params if ("norm" in name or "bias" in name) else decay_params).append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(len(train_loader) * args.epochs - args.warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"total": 0.0, "recon": 0.0, "kl": 0.0, "spectral": 0.0, "tclr": 0.0}
        t0 = time.time()
        beta = get_kl_beta(epoch, args.kl_anneal_epochs, args.kl_anneal_start, args.kl_anneal_end)
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            losses = train_step(model, batch, beta=beta, gamma_spectral=args.gamma_spectral,
                                gamma_tclr=args.gamma_tclr, device=device)
            (losses["total"] / args.grad_accum_steps).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            if (step + 1) % args.log_interval == 0:
                avg = {k: v / (step + 1) for k, v in epoch_losses.items()}
                print(f"  Ep {epoch+1} Step {step+1}/{len(train_loader)}: "
                      f"loss={avg['total']:.4f} recon={avg['recon']:.4f} "
                      f"kl={avg['kl']:.4f} spectral={avg['spectral']:.4f} "
                      f"tclr={avg['tclr']:.4f} beta={beta:.2e}")

        elapsed = time.time() - t0
        avg = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s): "
              f"loss={avg['total']:.4f} recon={avg['recon']:.4f} "
              f"kl={avg['kl']:.4f} spectral={avg['spectral']:.4f} tclr={avg['tclr']:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                losses = train_step(model, batch, beta=beta, gamma_spectral=args.gamma_spectral,
                                    gamma_tclr=args.gamma_tclr, device=device)
                val_loss += losses["total"].item()
        val_loss /= max(len(val_loader), 1)
        print(f"  Val loss: {val_loss:.4f}")

        # Compute latent stats after first epoch (for standardization in later epochs)
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                all_latents = []
                for batch in train_loader:
                    ot = batch["ot"].to(device)
                    ot_lengths = batch["ot_lengths"]
                    ot_list = [ot[i, :ot_lengths[i]] for i in range(ot.shape[0])]
                    enc_output = model.encode(ot_list)
                    trend_dists, residual_dists = enc_output.latent_dists
                    for td, rd in zip(trend_dists, residual_dists):
                        all_latents.append(torch.cat([td.mean, rd.mean], dim=-1))
                model.compute_latent_stats(all_latents)
            print("  Latent stats computed for standardization.")

        ckpt_path = save_dir / f"epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1, "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config.to_dict(), "train_args": vars(args), "val_loss": val_loss,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    print("Stage 1 training complete.")


if __name__ == "__main__":
    main()
