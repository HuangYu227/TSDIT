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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .configuration_mmldm import MMLDMDiTConfig, MMLDMVAEConfig
from .data.tsfragment_dataset import CollateFn, TSFragmentDataset
from .data.weather_dataset import WeatherCollateFn, WeatherDataset
from .modeling_mmldm_dit import MMLDMDiTModel
from .modeling_mmldm_vae import MMLDMVAEModel
from .attention_utils import create_dit_readonly_text_mask
from .semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) — Engineering improvement
# ---------------------------------------------------------------------------


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        """Swap model params with EMA params (call before eval)."""
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore original params after eval."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# Innovation E: Text-Adaptive SNR Scheduling
# ---------------------------------------------------------------------------


def text_adaptive_timestep(
    text_latent: torch.Tensor,
    base_t: torch.Tensor,
    snr_alpha: float = 0.3,
) -> torch.Tensor:
    """Adjust timesteps based on text complexity.

    Simple texts (low-norm embeddings) get lower timesteps (easier denoising).
    Complex texts (high-norm embeddings) get higher timesteps (harder denoising).

    Args:
        text_latent: (B, D) text latent embeddings.
        base_t: (B,) uniformly sampled timesteps in [0, 1].
        snr_alpha: strength of text-adaptive adjustment (0 = uniform).

    Returns:
        (B,) adjusted timesteps clamped to [0.01, 0.99].
    """
    # Text complexity proxy: L2 norm of text latent, normalized to [0, 1]
    norms = text_latent.norm(dim=-1)  # (B,)
    # Normalize to zero-mean unit-variance across batch
    if norms.std() > 1e-6:
        complexity = (norms - norms.mean()) / norms.std()
    else:
        complexity = torch.zeros_like(norms)
    # Shift and scale: complex texts get higher t
    adjusted = base_t + snr_alpha * complexity * base_t * (1 - base_t)
    return adjusted.clamp(0.01, 0.99)


# ---------------------------------------------------------------------------
# Innovation F: Cross-Block Consistency Distillation
# ---------------------------------------------------------------------------


def compute_consistency_loss(
    model: MMLDMDiTModel,
    z0: torch.Tensor,
    text_latent: torch.Tensor,
    ts_shape: torch.LongTensor,
    text_shape: torch.LongTensor,
    noise: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    delta: float = 0.05,
    text_raw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-Block Consistency Distillation loss.

    L_cons = ||DiT(z_t1, t1, c) - DiT(z_t2, t2, c)||²
    where t1, t2 are nearby timesteps (|t1-t2| < delta).

    This encourages the DiT vector field to be smooth, enabling fewer-step inference.

    Args:
        model: DiT model.
        z0: clean latent (L_total, D).
        text_latent: text condition (B, D).
        ts_shape: per-sample TS lengths.
        text_shape: per-sample text lengths.
        noise: random noise (L_total, D).
        attn_mask: optional attention mask.
        delta: max timestep difference for consistency pairs.
    """
    L = z0.shape[0]
    # Sample two nearby timesteps
    t1 = torch.rand(1, device=z0.device).expand(L)
    t2 = (t1 + delta * (torch.rand_like(t1) - 0.5)).clamp(0.01, 0.99)

    z_t1 = q_sample_flow(z0, t1, noise)
    z_t2 = q_sample_flow(z0, t2, noise)

    out1 = model(ts=z_t1, text=text_latent, ts_shape=ts_shape,
                 text_shape=text_shape, timestep=t1, attn_mask=attn_mask,
                 text_latent=text_raw)
    out2 = model(ts=z_t2, text=text_latent, ts_shape=ts_shape,
                 text_shape=text_shape, timestep=t2, attn_mask=attn_mask,
                 text_latent=text_raw)

    return F.mse_loss(out1.ts_sample, out2.ts_sample)


# ---------------------------------------------------------------------------
# Innovation D: Semantic Curriculum — complexity-scored batch ordering
# ---------------------------------------------------------------------------


class CurriculumSampler:
    """Sampler that orders by text complexity for early epochs, then random.

    Uses text embedding L2 norm as complexity proxy. During curriculum
    epochs, batches are ordered simple→complex. After curriculum, standard
    random shuffle is used.
    """

    def __init__(self, dataset, num_epochs: int, curriculum_epochs: int, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.curriculum_epochs = curriculum_epochs
        self.batch_size = batch_size
        self.seed = seed
        # Pre-compute complexity scores from text embeddings
        self._scores = self._compute_scores()

    def _compute_scores(self) -> np.ndarray:
        scores = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            emb = sample["text_embedding"]
            scores.append(float(torch.norm(emb, p=2).item()))
        return np.array(scores)

    def get_indices(self, epoch: int) -> list[int]:
        n = len(self.dataset)
        rng = np.random.RandomState(self.seed + epoch)
        if epoch < self.curriculum_epochs:
            # Progressive curriculum: fraction of easy samples increases
            progress = (epoch + 1) / self.curriculum_epochs
            n_include = max(self.batch_size * 2, int(n * progress))
            # Select n_include samples with lowest complexity
            sorted_indices = np.argsort(self._scores)
            indices = sorted_indices[:n_include].tolist()
            rng.shuffle(indices)
        else:
            indices = list(range(n))
            rng.shuffle(indices)
        return indices


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


def sample_lambda(batch_size: int, device: torch.device, alpha: float = 0.4) -> torch.Tensor:
    """Sample mixing coefficient from Beta distribution (Engineering: Beta(0.4,0.4) for stronger mixing)."""
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


def frequency_weighted_flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ts_shape: torch.LongTensor,
    gamma_freq: float = 0.1,
    gamma_weighted: float = 0.05,
) -> torch.Tensor:
    """Fused time-domain MSE + frequency-domain L1 + frequency-weighted loss.

    Computes per-sample to avoid cross-boundary FFT artifacts.
    """
    # Time-domain MSE (flat — no boundary issue)
    l_time = F.mse_loss(pred, target)

    # Frequency and weighted losses: per-sample to avoid cross-boundary FFT
    lengths = ts_shape.flatten().tolist()
    pred_list = pred.split(lengths, dim=0)
    target_list = target.split(lengths, dim=0)

    l_freq_total = torch.tensor(0.0, device=pred.device)
    l_weighted_total = torch.tensor(0.0, device=pred.device)

    for p_i, t_i in zip(pred_list, target_list):
        L_i = p_i.shape[0]
        # Frequency L1 (DiMTS-inspired)
        fft_p = torch.fft.rfft(p_i, dim=0)
        fft_t = torch.fft.rfft(t_i, dim=0)
        l_freq_total = l_freq_total + (F.l1_loss(fft_p.real, fft_t.real) +
                                        F.l1_loss(fft_p.imag, fft_t.imag))
        # Frequency-weighted (CPiRi-inspired)
        weights = 1.0 / torch.arange(1, L_i + 1, device=pred.device, dtype=pred.dtype)
        l_weighted_total = l_weighted_total + (weights.unsqueeze(-1) * (p_i - t_i).abs()).mean()

    n = len(lengths)
    return l_time + gamma_freq * (l_freq_total / n) + gamma_weighted * (l_weighted_total / n)


def compute_flow_matching_loss(
    model: MMLDMDiTModel,
    z0: torch.Tensor,
    text: torch.Tensor,
    ts_shape: torch.LongTensor,
    text_shape: torch.LongTensor,
    t_per_token: torch.Tensor,
    noise: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    text_raw: Optional[torch.Tensor] = None,
    gamma_freq: float = 0.0,
    gamma_weighted: float = 0.0,
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
        text_raw: ``(B, text_raw_dim)`` raw text embedding for TGFM.
        gamma_freq: frequency-domain L1 loss weight.
        gamma_weighted: frequency-weighted loss weight.
    """
    z_t = q_sample_flow(z0, t_per_token, noise)
    u_t = noise - z0  # target velocity

    output = model(
        ts=z_t, text=text,
        ts_shape=ts_shape, text_shape=text_shape,
        timestep=t_per_token,
        attn_mask=attn_mask,
        text_latent=text_raw,
    )

    ts_pred = output.ts_sample
    if gamma_freq > 0 or gamma_weighted > 0:
        loss = 0.5 * frequency_weighted_flow_loss(ts_pred, u_t, ts_shape, gamma_freq, gamma_weighted)
    else:
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
    text_raw_a: Optional[torch.Tensor] = None,
    text_raw_b: Optional[torch.Tensor] = None,
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
        text_latent=text_raw_a,
    )
    v_a = output_a.ts_sample

    output_b = model(
        ts=z_t_mix, text=text_b,
        ts_shape=ts_shape_mix, text_shape=text_shape_b,
        timestep=t_mix,
        attn_mask=attn_mask,
        text_latent=text_raw_b,
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
    parser.add_argument("--dataset_type", type=str, default="csv",
                        choices=["csv", "weather_npy"],
                        help="Dataset format: csv (TSFragment-600K) or weather_npy (VerbalTS Weather)")
    parser.add_argument("--weather_data_dir", type=str, default=None,
                        help="Path to Weather .npy data (required when --dataset_type weather_npy)")
    parser.add_argument("--data_dir", type=str, default="")
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
    parser.add_argument("--block_size", type=int, default=8, help="Default block size")
    parser.add_argument("--use_adaptive_routing", action="store_true", help="Use semantic router")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.3, help="CFG condition dropout prob")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    # Innovation: Frequency-Weighted Flow Loss
    parser.add_argument("--gamma_freq", type=float, default=0.1, help="Frequency-domain L1 loss weight")
    parser.add_argument("--gamma_weighted", type=float, default=0.05, help="Frequency-weighted 1/k loss weight")
    # Innovation: Diffusion Batch Multiplication
    parser.add_argument("--batch_mul", type=int, default=1, help="Repeat each sample N times with different t (1=disabled)")
    # Innovation E: Text-Adaptive SNR
    parser.add_argument("--snr_alpha", type=float, default=0.3, help="Text-adaptive SNR strength (0=uniform)")
    # Innovation F: Consistency Distillation
    parser.add_argument("--gamma_cons", type=float, default=0.1, help="Consistency distillation loss weight")
    parser.add_argument("--cons_delta", type=float, default=0.05, help="Timestep delta for consistency pairs")
    parser.add_argument("--cons_warmup_epochs", type=int, default=10, help="Epochs before enabling cons loss")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    # Innovation D: Curriculum
    parser.add_argument("--curriculum_epochs", type=int, default=3, help="Epochs of curriculum (simple->complex)")
    # Engineering: EMA
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate (0=disabled)")
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
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    vae_config = MMLDMVAEConfig(**vae_ckpt["config"])
    vae = MMLDMVAEModel(vae_config).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    # Freeze TS encoder/decoder — keep frozen
    for p in vae.parameters():
        p.requires_grad = False
    # Unfreeze text encoder — trained by FM loss for cross-modal alignment
    for p in vae.text_proj.parameters():
        p.requires_grad = True
    for block in vae.text_encoder_blocks:
        for p in block.parameters():
            p.requires_grad = True
    for p in vae.text_final_norm.parameters():
        p.requires_grad = True
    for p in vae.text_final_layer.parameters():
        p.requires_grad = True
    text_enc_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Loaded VAE from {args.vae_checkpoint}")
    print(f"  TS encoder/decoder: frozen | Text encoder: {text_enc_params:,} params trainable via FM")

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
        text_latent_dim=128,  # TGFM: raw SBERT embedding dim
        n_text_views=4,       # MVTC: K orthogonal text views
    )
    dit = MMLDMDiTModel(dit_config).to(device)
    dit_param_count = sum(p.numel() for p in dit.parameters())
    print(f"DiT parameters: {dit_param_count:,}")

    # Engineering: EMA
    ema = EMA(dit, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        print(f"EMA enabled with decay={args.ema_decay}")

    # Optional semantic router
    router = None
    if args.use_adaptive_routing:
        router = SemanticRouter(
            text_dim=vae_config.text_dim,
            n_latent=96,
        ).to(device)
        print("Semantic router enabled")

    # Build dataset
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
        collate = CollateFn()
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

    # Compute dataset-level latent statistics for standardization
    # Only if not already loaded from Stage1 checkpoint
    if vae._latent_stats_computed.item():
        print(f"Using latent stats from Stage1 checkpoint: mean={vae.latent_mean.mean():.4f}, std={vae.latent_std.mean():.4f}")
    else:
        print("Computing dataset-level latent statistics...")
        _stat_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate, num_workers=0)
        _all_latents = []
        with torch.no_grad():
            for _batch in _stat_loader:
                _ot = _batch["ot"].to(device)
                _ot_lengths = _batch["ot_lengths"]
                _ot_list = [_ot[i, :_ot_lengths[i]] for i in range(_ot.shape[0])]
                _enc = vae.encode(_ot_list)
                _trend_d, _res_d = _enc.latent_dists
                _z = [torch.cat([td.mean, rd.mean], dim=-1)
                      for td, rd in zip(_trend_d, _res_d)]
                _all_latents.extend(_z)
        vae.compute_latent_stats(_all_latents)
        print(f"  Latent stats: mean={vae.latent_mean.mean():.4f}, std={vae.latent_std.mean():.4f}")
        del _all_latents, _stat_loader

    # Save VAE with latent stats for inference
    vae_with_stats_path = Path(args.save_dir) / "vae_with_stats.pt"
    vae_with_stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": vae.state_dict(),
        "config": vae_config.to_dict(),
    }, vae_with_stats_path)
    print(f"  Saved VAE with latent stats: {vae_with_stats_path}")

    # Innovation D: Curriculum sampler
    curriculum = None
    if args.curriculum_epochs > 0 and len(train_ds) > 0:
        curriculum = CurriculumSampler(
            train_ds, num_epochs=args.epochs,
            curriculum_epochs=args.curriculum_epochs,
            batch_size=args.batch_size, seed=args.seed,
        )
        print(f"Curriculum learning enabled for first {args.curriculum_epochs} epochs")
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    # DataLoader — shuffle handled by curriculum or default
    _sampler = None
    _shuffle = True
    if curriculum is not None:
        from torch.utils.data import SubsetRandomSampler
        _sampler = SubsetRandomSampler(curriculum.get_indices(0))
        _shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=_shuffle,
        sampler=_sampler, collate_fn=collate, num_workers=0, pin_memory=True,
    )

    # Optimizer with param groups (no weight decay for norms/biases)
    # DiT + VAE text encoder jointly trained.
    # TS encoder/decoder frozen; text encoder receives FM gradient for cross-modal alignment.
    trainable_models = [dit, vae]

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

    # OneCycleLR: single-cycle schedule matching T2S's approach
    # Ramps lr up then decays over the entire training run
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.05,       # 5% warmup (short, since we have many epochs)
        anneal_strategy="cos",
        div_factor=25,        # initial lr = max_lr / 25
        final_div_factor=1e4, # final lr = max_lr / 10000
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Innovation D: Refresh curriculum indices each epoch
        if curriculum is not None:
            from torch.utils.data import SubsetRandomSampler
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size,
                sampler=SubsetRandomSampler(curriculum.get_indices(epoch)),
                collate_fn=collate, num_workers=0, pin_memory=True,
            )

        dit.train()
        # Router stays in eval mode — it's a frozen heuristic module
        epoch_losses = {"total": 0.0, "fm": 0.0, "dcd_mix": 0.0, "dcd_aux": 0.0, "cons": 0.0}
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

            # Encode with VAE (use mean for stable targets)
            with torch.no_grad():
                enc_output = vae.encode(ot_list)
                trend_dists, residual_dists = enc_output.latent_dists
                z_list = [torch.cat([td.mean, rd.mean], dim=-1)
                          for td, rd in zip(trend_dists, residual_dists)]

            z0 = torch.cat(z_list, dim=0)  # (L_total, latent_dim)
            # Standardize latent using dataset-level stats
            z0 = vae.standardize_latent(z0)
            ts_shape = torch.tensor(
                [[z.shape[0]] for z in z_list], dtype=torch.long, device=device
            )

            # Text latent: use clean text-only encoder (no TS leakage through joint blocks)
            text_latent = vae.encode_text_condition(text_emb)  # (B, latent_dim)

            # CFG training: randomly drop text condition per sample
            # Must drop BOTH text_latent AND text_emb (text_raw) to ensure
            # the unconditional path truly sees no text information.
            if args.cfg_drop_prob > 0:
                drop_mask = (
                    torch.rand(B, 1, device=device) > args.cfg_drop_prob
                ).to(text_latent.dtype)
                text_latent = text_latent * drop_mask
                text_emb = text_emb * drop_mask

            # Per-sample text: each sample gets 1 text token
            text_shape = torch.tensor([[1]] * B, dtype=torch.long, device=device)

            # Innovation E: Text-Adaptive SNR — bias timesteps by text complexity
            t_per_sample = torch.rand(B, device=device)
            if args.snr_alpha > 0:
                t_per_sample = text_adaptive_timestep(text_latent.detach(), t_per_sample, args.snr_alpha)
            t_per_token = expand_timesteps_per_token(t_per_sample, ts_shape)

            noise = torch.randn_like(z0)

            # Diffusion Batch Multiplication: repeat each sample with different t
            # Use .repeat() (not .repeat_interleave()) to preserve sample
            # boundaries in flat token layout. repeat_interleave duplicates
            # individual tokens, corrupting per-sample groupings.
            if args.batch_mul > 1:
                z0 = z0.repeat(args.batch_mul, 1)
                text_latent = text_latent.repeat(args.batch_mul, 1)
                ts_shape = ts_shape.repeat(args.batch_mul, 1)
                text_shape = text_shape.repeat(args.batch_mul, 1)
                text_emb = text_emb.repeat(args.batch_mul, 1)
                t_per_sample = torch.rand(B * args.batch_mul, device=device)
                if args.snr_alpha > 0:
                    t_per_sample = text_adaptive_timestep(text_latent.detach(), t_per_sample, args.snr_alpha)
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
                attn_mask=attn_mask, text_raw=text_emb,
                gamma_freq=args.gamma_freq, gamma_weighted=args.gamma_weighted,
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
                    text_raw_a=text_emb[:half],
                    text_raw_b=text_emb[half:half * 2],
                )

            # Innovation F: Consistency Distillation loss (delayed warmup)
            l_cons = torch.tensor(0.0, device=device)
            if args.gamma_cons > 0 and epoch >= args.cons_warmup_epochs:
                l_cons = compute_consistency_loss(
                    dit, z0, text_latent, ts_shape, text_shape, noise,
                    attn_mask=attn_mask, delta=args.cons_delta,
                    text_raw=text_emb,
                )

            total = (l_fm + args.gamma1 * l_dcd_mix + args.gamma2 * l_dcd_aux
                     + args.gamma_cons * l_cons)
            total = total / args.grad_accum_steps
            total.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                trainable_params = [p for group in optimizer.param_groups for p in group["params"]]
                last_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # Engineering: EMA update
                if ema:
                    ema.update(dit)

            global_step += 1

            epoch_losses["total"] += (total.item() * args.grad_accum_steps)
            epoch_losses["fm"] += l_fm.item()
            epoch_losses["dcd_mix"] += l_dcd_mix.item()
            epoch_losses["dcd_aux"] += l_dcd_aux.item()
            epoch_losses["cons"] += l_cons.item()

            if (step + 1) % args.log_interval == 0:
                avg = {k: v / (step + 1) for k, v in epoch_losses.items()}
                print(
                    f"  Epoch {epoch+1} Step {step+1}/{len(train_loader)}: "
                    f"loss={avg['total']:.4f} fm={avg['fm']:.4f} "
                    f"dcd_mix={avg['dcd_mix']:.4f} dcd_aux={avg['dcd_aux']:.4f} "
                    f"cons={avg.get('cons', 0.0):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} grad_norm={last_grad_norm:.2f}"
                )

        elapsed = time.time() - t0
        avg = {k: v / max(len(train_loader), 1) for k, v in epoch_losses.items()}
        print(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s): "
            f"loss={avg['total']:.4f} fm={avg['fm']:.4f} "
            f"dcd_mix={avg['dcd_mix']:.4f} dcd_aux={avg['dcd_aux']:.4f} "
            f"cons={avg.get('cons', 0.0):.4f}"
        )

        # Validation (use EMA weights if available)
        dit.eval()
        if ema:
            ema.apply(dit)
        val_loss = 0.0
        val_fm = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                ot = val_batch["ot"].to(device)
                text_emb = val_batch["text_embedding"].to(device)
                ot_lengths = val_batch["ot_lengths"]
                B_v = ot.shape[0]
                ot_list = [ot[i, :ot_lengths[i]] for i in range(B_v)]

                enc_output_v = vae.encode(ot_list)
                trend_dists_v, residual_dists_v = enc_output_v.latent_dists
                z_list = [torch.cat([td.mean, rd.mean], dim=-1)
                          for td, rd in zip(trend_dists_v, residual_dists_v)]
                z0_v = torch.cat(z_list, dim=0)
                # Standardize latent using dataset-level stats
                z0_v = vae.standardize_latent(z0_v)
                ts_shape_v = torch.tensor(
                    [[z.shape[0]] for z in z_list], dtype=torch.long, device=device,
                )
                text_latent_v = vae.encode_text_condition(text_emb)
                text_shape_v = torch.tensor([[1]] * B_v, dtype=torch.long, device=device)
                t_v = torch.rand(B_v, device=device)
                if args.snr_alpha > 0:
                    t_v = text_adaptive_timestep(text_latent_v, t_v, args.snr_alpha)
                t_pt_v = expand_timesteps_per_token(t_v, ts_shape_v)
                noise_v = torch.randn_like(z0_v)

                l_fm_v = compute_flow_matching_loss(
                    dit, z0_v, text_latent_v, ts_shape_v, text_shape_v,
                    t_pt_v, noise_v, text_raw=text_emb,
                    gamma_freq=args.gamma_freq, gamma_weighted=args.gamma_weighted,
                )
                val_loss += l_fm_v.item()
                val_fm += l_fm_v.item()

        val_loss /= max(len(val_loader), 1)
        val_fm /= max(len(val_loader), 1)
        if ema:
            ema.restore(dit)
        print(f"  Val loss: {val_loss:.4f} (fm={val_fm:.4f})")

        # Extract VAE text encoder weights (trained during Stage 2)
        vae_text_enc_state = {k: v for k, v in vae.state_dict().items()
                              if k.startswith(("text_proj", "text_encoder_blocks",
                                               "text_final_norm", "text_final_layer"))}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = save_dir / "best.pt"
            # Save EMA weights for best checkpoint (better inference quality)
            if ema:
                ema.apply(dit)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "dit_state_dict": dit.state_dict(),
                    "vae_text_encoder_state_dict": vae_text_enc_state,
                    "config": dit_config.to_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            print(f"  New best model saved: {best_ckpt_path}")
            if ema:
                ema.restore(dit)

        # Save checkpoint
        ckpt_path = save_dir / f"epoch_{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "dit_state_dict": dit.state_dict(),
                "vae_text_encoder_state_dict": vae_text_enc_state,
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
