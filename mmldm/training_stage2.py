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

"""Stage 2 training: Multimodal DiT prior + DCD.

Trains the DiT prior with Flow Matching and DCD dual-condition denoising::

    L = L_FM + gamma1 * L_DCD_mix + gamma2 * L_DCD_aux

DCD: Dual-Condition Denoising for mixed latent samples.

Usage:
    python -m mmldm.training_stage2 --data_dir ./data --vae_checkpoint ./checkpoints/stage1/epoch_10.pt
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .configuration_mmldm import MMLDMDiTConfig, MMLDMVAEConfig
from .data.tsfragment_dataset import CollateFn, TSFragmentDataset
from .modeling_mmldm_dit import MMLDMDiTModel
from .modeling_mmldm_vae import MMLDMVAEModel
from .attention_utils import create_dit_readonly_text_mask
from .semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# Seed / reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Per-token timestep expansion
# ---------------------------------------------------------------------------


def expand_timesteps_per_token(
    t: torch.Tensor,
    ts_shape: torch.LongTensor,
) -> torch.Tensor:
    """Expand per-sample timesteps to per-token timesteps.

    Args:
        t: ``(B,)`` per-sample timestep values.
        ts_shape: ``(B, 1)`` or ``(B,)`` per-sample token counts.

    Returns:
        ``(L_total,)`` per-token timesteps.
    """
    lengths = ts_shape.flatten().tolist()
    return torch.repeat_interleave(t, torch.tensor(lengths, device=t.device))


# ---------------------------------------------------------------------------
# Flow Matching helpers
# ---------------------------------------------------------------------------


def sample_lambda(batch_size: int, device: torch.device, alpha: float = 0.2) -> torch.Tensor:
    """Sample mixing coefficient from Beta distribution."""
    dist = torch.distributions.Beta(alpha, alpha)
    return dist.sample((batch_size,)).to(device)


def q_sample_flow(
    z0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Flow Matching forward: z_t = (1-t) * z0 + t * noise.

    Args:
        z0: clean latent ``(L_total, D)``.
        t: per-token timestep ``(L_total,)``.
        noise: random noise ``(L_total, D)``.
    """
    t = t.unsqueeze(-1) if t.ndim == 1 else t  # (L_total, 1)
    return (1 - t) * z0 + t * noise


def compute_flow_matching_loss(
    model: MMLDMDiTModel,
    z0: torch.Tensor,
    text: torch.Tensor,
    ts_shape: torch.LongTensor,
    text_shape: torch.LongTensor,
    t_per_token: torch.Tensor,
    noise: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Flow Matching loss L_FM.

    Args:
        z0: clean latent ``(L_total, latent_dim)``.
        text: text latent ``(L_text_total, latent_dim)``.
        ts_shape: ``(B, 1)`` per-sample TS lengths.
        text_shape: ``(B, 1)`` per-sample text lengths.
        t_per_token: per-token timestep ``(L_total,)``.
        noise: random noise ``(L_total, latent_dim)``.
        attn_mask: optional precomputed attention mask.
    """
    z_t = q_sample_flow(z0, t_per_token, noise)
    u_t = noise - z0  # target velocity

    output = model(
        ts=z_t, text=text,
        ts_shape=ts_shape, text_shape=text_shape,
        timestep=t_per_token,
        attn_mask=attn_mask,
    )

    ts_pred = output.ts_sample
    loss = 0.5 * F.mse_loss(ts_pred, u_t)
    return loss


def compute_dcd_losses(
    model: MMLDMDiTModel,
    z0_a: torch.Tensor,
    z0_b: torch.Tensor,
    text_a: torch.Tensor,
    text_b: torch.Tensor,
    ts_shape_a: torch.LongTensor,
    ts_shape_b: torch.LongTensor,
    text_shape_a: torch.LongTensor,
    text_shape_b: torch.LongTensor,
    lam: torch.Tensor,
    t_per_token_a: torch.Tensor,
    noise: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DCD losses L_DCD_mix and L_DCD_aux.

    Per-sample latent mixing to handle variable-length NA layout:
    split -> per-sample mix with lam[i] -> re-concatenate.
    Uses vectorized lam_per_token for the weighted combination.

    Args:
        z0_a, z0_b: clean latents ``(L_a, D)``, ``(L_b, D)``.
        text_a, text_b: text latents for A/B.
        ts_shape_a, ts_shape_b: per-sample TS lengths for A/B.
        text_shape_a, text_shape_b: per-sample text lengths for A/B.
        lam: mixing coefficient ``(half,)``.
        t_per_token_a: per-token timestep ``(L_a,)``.
        noise: noise ``(L_a, D)``.
        attn_mask: attention mask for the mixed sequence.

    Returns:
        (L_DCD_mix, L_DCD_aux) both scalar.
    """
    sizes_a = ts_shape_a.flatten().tolist()
    sizes_b = ts_shape_b.flatten().tolist()
    z0_a_split = z0_a.split(sizes_a)
    z0_b_split = z0_b.split(sizes_b)

    # Per-sample mix with truncation to min length
    z0_mix_list = []
    mix_sizes = []
    for i, (za, zb) in enumerate(zip(z0_a_split, z0_b_split)):
        min_len = min(za.shape[0], zb.shape[0])
        z0_mix_i = lam[i] * za[:min_len] + (1 - lam[i]) * zb[:min_len]
        z0_mix_list.append(z0_mix_i)
        mix_sizes.append(min_len)

    z0_mix = torch.cat(z0_mix_list, dim=0)  # (L_mix, D)

    L_mix = z0_mix.shape[0]
    t_mix = t_per_token_a[:L_mix]
    noise_mix = noise[:L_mix]

    z_t_mix = q_sample_flow(z0_mix, t_mix, noise_mix)
    u_t_mix = noise_mix - z0_mix

    # Build ts_shape_mix from actual mixed per-sample lengths
    ts_shape_mix = torch.tensor(
        [[l] for l in mix_sizes], dtype=ts_shape_a.dtype, device=ts_shape_a.device,
    )

    # Dual-condition prediction — ts_shape_mix matches z_t_mix (L_mix tokens)
    output_a = model(
        ts=z_t_mix, text=text_a,
        ts_shape=ts_shape_mix, text_shape=text_shape_a,
        timestep=t_mix,
        attn_mask=attn_mask,
    )
    v_a = output_a.ts_sample

    output_b = model(
        ts=z_t_mix, text=text_b,
        ts_shape=ts_shape_mix, text_shape=text_shape_b,
        timestep=t_mix,
        attn_mask=attn_mask,
    )
    v_b = output_b.ts_sample

    # Vectorized per-token lam expansion: (half,) -> (L_mix, 1)
    lam_per_token = torch.repeat_interleave(
        lam, torch.tensor(mix_sizes, device=lam.device),
    ).unsqueeze(-1)

    # Weighted combination (vectorized)
    v_mix = lam_per_token * v_a + (1 - lam_per_token) * v_b
    l_mix = 0.5 * F.mse_loss(v_mix, u_t_mix)

    # Per-sample auxiliary losses
    v_a_split = v_a.split(mix_sizes)
    v_b_split = v_b.split(mix_sizes)
    u_split = u_t_mix.split(mix_sizes)

    l_aux_a = torch.stack([F.mse_loss(va_i, u_i) for va_i, u_i in zip(v_a_split, u_split)])
    l_aux_b = torch.stack([F.mse_loss(vb_i, u_i) for vb_i, u_i in zip(v_b_split, u_split)])
    l_aux = 0.5 * (lam * l_aux_a + (1 - lam) * l_aux_b).mean()

    return l_mix, l_aux


# ---------------------------------------------------------------------------
# Adaptive mask construction from router
# ---------------------------------------------------------------------------


def build_adaptive_mask_for_batch(
    router: SemanticRouter,
    text_emb: torch.Tensor,
    ts_shape: torch.LongTensor,
    text_shape: torch.LongTensor,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build adaptive multimodal joint mask for a batch using the router.

    Args:
        router: SemanticRouter instance.
        text_emb: ``(B, text_dim)`` raw text embeddings.
        ts_shape: ``(B, 1)`` per-sample latent lengths.
        text_shape: ``(B, 1)`` per-sample text lengths.
        block_size: default block size (fallback for router).
        device: compute device.
        dtype: mask dtype.

    Returns:
        ``(1, 1, L_total, L_total)`` additive mask.
    """
    lengths = ts_shape.flatten().tolist()
    n_blocks_per_sample = [max(1, round(l / block_size)) for l in lengths]

    nested_block_sizes = []
    for i, (n_lat, n_blk) in enumerate(zip(lengths, n_blocks_per_sample)):
        sample_text = text_emb[i : i + 1]
        block_sizes_i = router(sample_text, n_latent=n_lat, n_blocks=n_blk)[0]
        total = sum(block_sizes_i)
        if total != n_lat:
            block_sizes_i[-1] += n_lat - total
        nested_block_sizes.append(block_sizes_i)

    return create_dit_readonly_text_mask(
        ts_shape=ts_shape,
        text_shape=text_shape,
        block_sizes=nested_block_sizes,
        dtype=dtype,
        device=device,
    )


def _shape_tensor(lens: list[int], device: torch.device) -> torch.LongTensor:
    return torch.tensor([[int(l)] for l in lens], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MMLDM Stage 2: DiT + DCD Training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="Stage 1 VAE checkpoint")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ETTh1"])
    parser.add_argument("--time_intervals", type=int, nargs="+", default=[24])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps")
    parser.add_argument("--gamma1", type=float, default=1.0, help="DCD mix loss weight")
    parser.add_argument("--gamma2", type=float, default=0.0, help="DCD aux loss weight")
    parser.add_argument("--dit_dim", type=int, default=256, help="DiT hidden dimension")
    parser.add_argument("--dit_layers", type=int, default=12, help="DiT layers")
    parser.add_argument("--dit_heads", type=int, default=4, help="DiT heads")
    parser.add_argument("--block_size", type=int, default=4, help="Default block size")
    parser.add_argument("--use_adaptive_routing", action="store_true", help="Use semantic router")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1, help="CFG condition dropout prob")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split_file", type=str, default=None, help="Path to splits.json for train/val split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/stage2")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Load VAE
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae_config = MMLDMVAEConfig(**vae_ckpt["config"])
    vae = MMLDMVAEModel(vae_config).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"Loaded VAE from {args.vae_checkpoint}")

    # Build DiT
    dit_config = MMLDMDiTConfig(
        ts_in_channels=vae_config.latent_dim,
        ts_out_channels=vae_config.latent_dim,
        text_in_channels=vae_config.latent_dim,
        text_out_channels=vae_config.latent_dim,
        txt_dim=args.dit_dim,
        emb_dim=args.dit_dim,
        heads=args.dit_heads,
        head_dim=args.dit_dim // args.dit_heads,
        num_layers=args.dit_layers,
        block_size=args.block_size,
    )
    dit = MMLDMDiTModel(dit_config).to(device)
    dit_param_count = sum(p.numel() for p in dit.parameters())
    print(f"DiT parameters: {dit_param_count:,}")

    # Optional semantic router
    router = None
    if args.use_adaptive_routing:
        router = SemanticRouter(
            text_dim=vae_config.text_dim,
            n_latent=96,
        ).to(device)
        print("Semantic router enabled")

    # Build dataset with SampleID-level split
    collate = CollateFn()

    if args.split_file is not None:
        train_ds = TSFragmentDataset(
            data_dir=args.data_dir, datasets=args.datasets,
            time_intervals=args.time_intervals, max_samples=args.max_samples,
            split="train", split_file=args.split_file,
        )
        val_ds = TSFragmentDataset(
            data_dir=args.data_dir, datasets=args.datasets,
            time_intervals=args.time_intervals,
            split="val", split_file=args.split_file,
        )
        print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples (from {args.split_file})")
    else:
        # Fallback: random row-level split (may leak for ETTh1 sliding windows)
        dataset = TSFragmentDataset(
            data_dir=args.data_dir, datasets=args.datasets,
            time_intervals=args.time_intervals, max_samples=args.max_samples,
        )
        val_size = min(len(dataset) // 10, 1000)
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
        print(f"Loaded {len(dataset)} samples (random split, no split_file)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # Optimizer with param groups (no weight decay for norms/biases)
    # Only DiT is trained. Text latent comes from frozen VAE text encoder.
    trainable_models = [dit]

    decay_params = []
    no_decay_params = []
    for model in trainable_models:
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

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        dit.train()
        # Router stays in eval mode — it's a frozen heuristic module
        epoch_losses = {"total": 0.0, "fm": 0.0, "dcd_mix": 0.0, "dcd_aux": 0.0}
        t0 = time.time()

        optimizer.zero_grad()
        last_grad_norm = 0.0

        for step, batch in enumerate(train_loader):
            ot = batch["ot"].to(device)
            text_emb = batch["text_embedding"].to(device)
            ot_lengths = batch["ot_lengths"]
            B = ot.shape[0]

            # Convert to list
            ot_list = [ot[i, :ot_lengths[i]] for i in range(B)]

            # Encode with VAE
            with torch.no_grad():
                enc_output = vae.encode(ot_list, text_emb)
                z_list = [d.sample() for d in enc_output.latent_dists]

            z0 = torch.cat(z_list, dim=0)  # (L_total, latent_dim)
            ts_shape = torch.tensor(
                [[z.shape[0]] for z in z_list], dtype=torch.long, device=device
            )

            # Text latent: use clean text-only encoder (no TS leakage through joint blocks)
            text_latent = vae.encode_text_condition(text_emb)  # (B, latent_dim)

            # CFG training: randomly drop text condition per sample
            if args.cfg_drop_prob > 0:
                drop_mask = (
                    torch.rand(B, 1, device=device) > args.cfg_drop_prob
                ).to(text_latent.dtype)
                text_latent = text_latent * drop_mask

            # Per-sample text: each sample gets 1 text token
            text_shape = torch.tensor([[1]] * B, dtype=torch.long, device=device)

            # Sample per-sample timestep, then expand to per-token
            t_per_sample = torch.rand(B, device=device)
            t_per_token = expand_timesteps_per_token(t_per_sample, ts_shape)

            noise = torch.randn_like(z0)

            # Build attention mask (adaptive from router, or fixed default)
            attn_mask = None
            if router is not None:
                with torch.no_grad():
                    attn_mask = build_adaptive_mask_for_batch(
                        router, text_emb, ts_shape, text_shape,
                        args.block_size, device=device, dtype=z0.dtype,
                    )

            # Flow Matching loss
            l_fm = compute_flow_matching_loss(
                dit, z0, text_latent, ts_shape, text_shape, t_per_token, noise,
                attn_mask=attn_mask,
            )

            # DCD loss
            l_dcd_mix = torch.tensor(0.0, device=device)
            l_dcd_aux = torch.tensor(0.0, device=device)

            if args.gamma1 > 0 and B >= 2:
                half = B // 2
                # Symmetric split: both halves have exactly `half` samples.
                # When B is odd, the last sample is silently dropped.
                cut_a = ts_shape[:half].sum().item()
                cut_b = ts_shape[:half * 2].sum().item()
                z0_a = z0[:cut_a]
                z0_b = z0[cut_a:cut_b]
                ts_shape_a = ts_shape[:half]
                ts_shape_b = ts_shape[half:half * 2]

                lam = sample_lambda(half, device)

                t_a = t_per_token[:cut_a]
                noise_a = noise[:cut_a]

                # ts_shape_mix: per-sample min(len_a, len_b) after DCD mixing
                ts_shape_mix = torch.tensor(
                    [[min(a, b)] for a, b in zip(
                        ts_shape_a.flatten().tolist(),
                        ts_shape_b.flatten().tolist(),
                    )],
                    dtype=torch.long, device=device,
                )

                # Build DCD mask using the MIXED shape (not ts_shape_a)
                dcd_mask = None
                if router is not None:
                    with torch.no_grad():
                        dcd_mask = build_adaptive_mask_for_batch(
                            router, text_emb[:half], ts_shape_mix,
                            text_shape[:half], args.block_size,
                            device=device, dtype=z0.dtype,
                        )

                l_dcd_mix, l_dcd_aux = compute_dcd_losses(
                    model=dit,
                    z0_a=z0_a, z0_b=z0_b,
                    text_a=text_latent[:half],
                    text_b=text_latent[half:half * 2],
                    ts_shape_a=ts_shape_a, ts_shape_b=ts_shape_b,
                    text_shape_a=text_shape[:half],
                    text_shape_b=text_shape[half:half * 2],
                    lam=lam, t_per_token_a=t_a, noise=noise_a,
                    attn_mask=dcd_mask,
                )

            total = (l_fm + args.gamma1 * l_dcd_mix + args.gamma2 * l_dcd_aux)
            total = total / args.grad_accum_steps
            total.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                last_grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            epoch_losses["total"] += (total.item() * args.grad_accum_steps)
            epoch_losses["fm"] += l_fm.item()
            epoch_losses["dcd_mix"] += l_dcd_mix.item()
            epoch_losses["dcd_aux"] += l_dcd_aux.item()

            if (step + 1) % args.log_interval == 0:
                avg = {k: v / (step + 1) for k, v in epoch_losses.items()}
                print(
                    f"  Epoch {epoch+1} Step {step+1}/{len(train_loader)}: "
                    f"loss={avg['total']:.4f} fm={avg['fm']:.4f} "
                    f"dcd_mix={avg['dcd_mix']:.4f} dcd_aux={avg['dcd_aux']:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} grad_norm={last_grad_norm:.2f}"
                )

        elapsed = time.time() - t0
        avg = {k: v / max(len(train_loader), 1) for k, v in epoch_losses.items()}
        print(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s): "
            f"loss={avg['total']:.4f} fm={avg['fm']:.4f} "
            f"dcd_mix={avg['dcd_mix']:.4f} dcd_aux={avg['dcd_aux']:.4f}"
        )

        # Validation
        dit.eval()
        val_loss = 0.0
        val_fm = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                ot = val_batch["ot"].to(device)
                text_emb = val_batch["text_embedding"].to(device)
                ot_lengths = val_batch["ot_lengths"]
                B_v = ot.shape[0]
                ot_list = [ot[i, :ot_lengths[i]] for i in range(B_v)]

                enc_output_v = vae.encode(ot_list, text_emb)
                z_list = [d.sample() for d in enc_output_v.latent_dists]
                z0_v = torch.cat(z_list, dim=0)
                ts_shape_v = torch.tensor(
                    [[z.shape[0]] for z in z_list], dtype=torch.long, device=device,
                )
                text_latent_v = vae.encode_text_condition(text_emb)
                text_shape_v = torch.tensor([[1]] * B_v, dtype=torch.long, device=device)
                t_v = torch.rand(B_v, device=device)
                t_pt_v = expand_timesteps_per_token(t_v, ts_shape_v)
                noise_v = torch.randn_like(z0_v)

                l_fm_v = compute_flow_matching_loss(
                    dit, z0_v, text_latent_v, ts_shape_v, text_shape_v,
                    t_pt_v, noise_v,
                )
                val_loss += l_fm_v.item()
                val_fm += l_fm_v.item()

        val_loss /= max(len(val_loader), 1)
        val_fm /= max(len(val_loader), 1)
        print(f"  Val loss: {val_loss:.4f} (fm={val_fm:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = save_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "dit_state_dict": dit.state_dict(),
                    "config": dit_config.to_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            print(f"  New best model saved: {best_ckpt_path}")

        # Save checkpoint
        ckpt_path = save_dir / f"epoch_{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "dit_state_dict": dit.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": dit_config.to_dict(),
                "train_args": vars(args),
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"  Saved checkpoint: {ckpt_path}")

    print("Stage 2 training complete.")


if __name__ == "__main__":
    main()
