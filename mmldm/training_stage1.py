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

"""Stage 1 training: Multimodal VAE pretraining.

Trains the MMLDM VAE with the objective::

    L_VAE = L_recon + beta * KL + lambda_mask * L_mask

where:
- L_recon: reconstruction loss (MSE for time series)
- KL: KL divergence of the posterior against N(0, I)
- L_mask: latent consistency loss (masked encoding ≈ full encoding)

Usage:
    python -m mmldm.training_stage1 --data_dir ./data --epochs 10
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
from .modeling_mmldm_vae import MMLDMVAEModel


# ---------------------------------------------------------------------------
# Seed / reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def compute_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss."""
    return F.mse_loss(recon, target)


def compute_kl_loss(latent_dists: list) -> torch.Tensor:
    """KL divergence against N(0, I)."""
    kl = 0.0
    for dist in latent_dists:
        kl = kl + dist.kl()
    return kl / len(latent_dists)


def compute_mask_loss(
    model: MMLDMVAEModel,
    ot_list: list[torch.Tensor],
    text_embs: torch.Tensor,
    mask_ratio: float = 0.3,
) -> torch.Tensor:
    """Masked reconstruction loss: reconstruct masked time-domain positions.

    Randomly masks a portion of the time series, encodes the masked
    version, decodes, and penalizes only at masked positions.
    """
    masked_ot_list = []
    masks_list = []
    for ot in ot_list:
        keep = torch.rand(ot.shape[0], device=ot.device) > mask_ratio  # True = keep
        masked = ot.clone()
        masked[~keep] = 0.0
        masked_ot_list.append(masked)
        masks_list.append(~keep)  # True = masked (need to predict)

    enc_out = model.encode(masked_ot_list, text_embs)
    z_sampled = torch.cat([d.sample() for d in enc_out.latent_dists], dim=0)

    txt_shape = torch.tensor(
        [[z.shape[0]] for z in masked_ot_list], dtype=torch.long, device=z_sampled.device,
    )
    recon = model.decode(z_sampled, txt_shape, txt_shape, attn_mask=None)

    target_flat = torch.cat(ot_list, dim=0)
    recon_flat = recon.squeeze(0)[: target_flat.shape[0]]
    global_mask = torch.cat(masks_list, dim=0).unsqueeze(-1)  # (L_total, 1)

    if global_mask.sum() == 0:
        return torch.tensor(0.0, device=z_sampled.device, requires_grad=True)
    return F.mse_loss(recon_flat[global_mask], target_flat[global_mask])


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------


def train_step(
    model: MMLDMVAEModel,
    batch: dict,
    beta: float = 1e-3,
    lambda_mask: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Single training step.

    Returns dict of losses: ``total``, ``recon``, ``kl``, ``mask``.
    """
    ot = batch["ot"].to(device)
    text_emb = batch["text_embedding"].to(device)
    ot_lengths = batch["ot_lengths"]

    ot_list = [ot[i, :ot_lengths[i]] for i in range(ot.shape[0])]

    output = model(ot_list, text_emb)

    target = torch.cat(ot_list, dim=0).unsqueeze(0)  # (1, L_total, 1)
    recon = output["recon"][:, : target.shape[1], :]

    l_recon = compute_recon_loss(recon, target)
    l_kl = compute_kl_loss(output["latent_dists"])
    l_mask = compute_mask_loss(model, ot_list, text_emb)

    total = l_recon + beta * l_kl + lambda_mask * l_mask

    return {
        "total": total,
        "recon": l_recon,
        "kl": l_kl,
        "mask": l_mask,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MMLDM Stage 1: VAE Pretraining")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TSFragment-600K directory")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ETTh1"], help="Datasets to use")
    parser.add_argument("--time_intervals", type=int, nargs="+", default=[24], help="Time intervals")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps")
    parser.add_argument("--beta", type=float, default=1e-3, help="KL weight")
    parser.add_argument("--lambda_mask", type=float, default=0.1, help="Mask loss weight")
    parser.add_argument("--dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--encoder_blocks", type=int, default=4, help="Encoder blocks")
    parser.add_argument("--decoder_blocks", type=int, default=4, help="Decoder blocks")
    parser.add_argument("--joint_blocks", type=int, default=2, help="Joint encoder blocks")
    parser.add_argument("--block_size", type=int, default=4, help="Block-causal block size")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (debug)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/stage1")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Build config
    config = MMLDMVAEConfig(
        ts_channels=1,
        text_dim=128,
        dim=args.dim,
        ffn_dim=args.dim * 4,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        head_dim=args.dim // args.num_heads,
        encoder_num_blocks=args.encoder_blocks,
        decoder_num_blocks=args.decoder_blocks,
        joint_num_blocks=args.joint_blocks,
        block_size=args.block_size,
    )

    # Build dataset
    dataset = TSFragmentDataset(
        data_dir=args.data_dir,
        datasets=args.datasets,
        time_intervals=args.time_intervals,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(dataset)} samples")

    # Split train/val
    val_size = min(len(dataset) // 10, 1000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    collate = CollateFn()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # Build model
    model = MMLDMVAEModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer with param groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    # Warmup + cosine scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(
            len(train_loader) * args.epochs - args.warmup_steps, 1
        )
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"total": 0.0, "recon": 0.0, "kl": 0.0, "mask": 0.0}
        t0 = time.time()

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            losses = train_step(model, batch, beta=args.beta, lambda_mask=args.lambda_mask, device=device)

            total = losses["total"] / args.grad_accum_steps
            total.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm = torch.tensor(0.0)

            global_step += 1

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            if (step + 1) % args.log_interval == 0:
                avg = {k: v / (step + 1) for k, v in epoch_losses.items()}
                print(
                    f"  Epoch {epoch+1} Step {step+1}/{len(train_loader)}: "
                    f"loss={avg['total']:.4f} recon={avg['recon']:.4f} "
                    f"kl={avg['kl']:.6f} mask={avg['mask']:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        elapsed = time.time() - t0
        avg = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        print(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s): "
            f"loss={avg['total']:.4f} recon={avg['recon']:.4f} "
            f"kl={avg['kl']:.6f} mask={avg['mask']:.4f}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                losses = train_step(model, batch, beta=args.beta, lambda_mask=args.lambda_mask, device=device)
                val_loss += losses["total"].item()
        val_loss /= max(len(val_loader), 1)
        print(f"  Val loss: {val_loss:.4f}")

        # Save checkpoint
        ckpt_path = save_dir / f"epoch_{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config.to_dict(),
                "train_args": vars(args),
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint: {ckpt_path}")

    print("Stage 1 training complete.")


if __name__ == "__main__":
    main()
