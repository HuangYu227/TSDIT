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

"""MMLDM inference — block-by-block Euler integration + VAE decode.

Implements the three-step inference algorithm for time series generation:

1. **Text condition encode**: text description -> text embedding -> text
   projector -> text latent ``c``.

2. **Block-wise latent prior transport**: For each generation block
   ``b = 1, 2, ..., B``, draw ``eps^(b) ~ N(0, I)`` and integrate the
   DiT vector field ``v_psi`` from ``t = T`` to ``t = 0`` under the
   block-causal visible set ``V_b = {z_0^{(<b)}, z_t^(b), c}``.

3. **Conditional decode**: ``x_hat ~ p_theta(x | z_0^{(1:B)})`` via the
   VAE decoder, producing time series values.

Usage::

    python -m mmldm.inference \\
        --dit_checkpoint ./checkpoints/stage2/epoch_20.pt \\
        --vae_checkpoint ./checkpoints/stage1/epoch_10.pt \\
        --text "The time series shows a sharp increase in temperature..." \\
        --output_len 96
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attention_utils import create_multimodal_joint_mask
from .configuration_mmldm import MMLDMDiTConfig, MMLDMVAEConfig
from .data.tsfragment_dataset import CollateFn, TSFragmentDataset
from .evaluation import evaluate_multi, evaluate_single, save_results
from .modeling_mmldm_dit import MMLDMDiTModel
from .modeling_mmldm_vae import MMLDMVAEModel
from .semantic_router import SemanticRouter


# ---------------------------------------------------------------------------
# NA helpers
# ---------------------------------------------------------------------------


def _shape_tensor(lens: list[int], device: torch.device) -> torch.LongTensor:
    """Build ``(B, 1)`` shape tensor from per-sample lengths."""
    return torch.tensor([[int(l)] for l in lens], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Euler ODE integrator
# ---------------------------------------------------------------------------


@torch.no_grad()
def euler_ode_step(
    dit: MMLDMDiTModel,
    z_t: torch.Tensor,
    text_latent: torch.Tensor,
    ts_shape: torch.LongTensor,
    text_shape: torch.LongTensor,
    t_curr: float,
    t_next: float,
    T: float,
    attn_mask: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Single Euler ODE step with optional CFG.

    Args:
        z_t: current noisy latent ``(L_total, latent_dim)``.
        text_latent: text conditioning latent ``(L_text, latent_dim)``.
        ts_shape: ``(B, 1)`` per-sample TS lengths.
        text_shape: ``(B, 1)`` per-sample text lengths.
        t_curr: current timestep.
        t_next: next timestep.
        T: total diffusion time.
        attn_mask: multimodal joint attention mask.
        guidance_scale: CFG scale (1.0 = no guidance).

    Returns:
        Updated latent ``z_{t-dt}``.
    """
    device = z_t.device
    dtype = z_t.dtype
    L_total = z_t.shape[0]

    t_batch = torch.full((L_total,), t_curr, device=device, dtype=dtype)

    # Conditional prediction
    output_cond = dit(
        ts=z_t, text=text_latent,
        ts_shape=ts_shape, text_shape=text_shape,
        timestep=t_batch, attn_mask=attn_mask,
    )
    v_cond = output_cond.ts_sample

    # Unconditional prediction (empty text)
    if guidance_scale > 1.0:
        empty_text = torch.zeros_like(text_latent)
        empty_text_shape = text_shape.clone()
        output_uncond = dit(
            ts=z_t, text=empty_text,
            ts_shape=ts_shape, text_shape=empty_text_shape,
            timestep=t_batch, attn_mask=attn_mask,
        )
        v_uncond = output_uncond.ts_sample
        v = guidance_scale * (v_cond - v_uncond) + v_uncond
    else:
        v = v_cond

    dt = (t_curr - t_next) / max(T, 1.0)
    z_next = z_t - dt * v

    return z_next


# ---------------------------------------------------------------------------
# Block-wise generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_latent_blocks(
    dit: MMLDMDiTModel,
    text_latent: torch.Tensor,
    text_shape: torch.LongTensor,
    n_blocks: int,
    block_size: int,
    latent_dim: int,
    device: torch.device,
    T: float = 1.0,
    timestep_num: int = 20,
    guidance_scale: float = 2.0,
    dtype: torch.dtype = torch.float32,
    use_adaptive_mask: bool = False,
    router: Optional[SemanticRouter] = None,
    text_tokens_for_router: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate latent tokens block-by-block with Euler ODE integration.

    For each block ``b``:
    1. Sample ``eps^(b) ~ N(0, I)``
    2. Build block-causal mask covering all previous + current blocks
    3. Euler-integrate from ``t=T`` to ``t=0``
    4. Pin cleaned prefix, keep only the new block

    Args:
        dit: pretrained DiT model.
        text_latent: text conditioning ``(L_text, latent_dim)``.
        text_shape: ``(B, 1)`` text lengths (B=1 for inference).
        n_blocks: number of generation blocks.
        block_size: default tokens per block.
        latent_dim: latent dimension.
        device: compute device.
        T: diffusion time horizon.
        timestep_num: number of ODE steps.
        guidance_scale: CFG scale.
        dtype: computation dtype.
        use_adaptive_mask: use SemanticRouter for variable block sizes.
        router: optional SemanticRouter instance.
        text_tokens_for_router: raw text tokens for the router.

    Returns:
        Generated latent ``(sum(block_sizes), latent_dim)``.
    """
    dit.eval()
    timesteps = torch.linspace(T, 0, timestep_num + 1, device=device, dtype=torch.float32)
    total_latent_tokens = n_blocks * block_size

    # Determine block sizes
    if use_adaptive_mask and router is not None and text_tokens_for_router is not None:
        block_sizes = router(
            text_tokens_for_router, n_latent=total_latent_tokens, n_blocks=n_blocks,
        )[0]
        print(f"  Adaptive block sizes: {block_sizes}")
    else:
        block_sizes = [block_size] * n_blocks

    generated_blocks: list[torch.Tensor] = []

    for b, curr_block_len in enumerate(block_sizes):
        eps = torch.randn(curr_block_len, latent_dim, device=device, dtype=dtype)
        z_block = eps.clone()

        # K-length: prefix + current block
        prefix_len = sum(x.shape[0] for x in generated_blocks)
        total_len_k = prefix_len + curr_block_len

        ts_shape_k = _shape_tensor([total_len_k], device)

        # Block sizes up to and including current block
        full_block_sizes = block_sizes[: b + 1]

        # Multimodal joint mask: [ts(=total_len_k) ; text(=L_text)]
        attn_mask = create_multimodal_joint_mask(
            ts_shape=ts_shape_k,
            text_shape=text_shape,
            block_sizes=[full_block_sizes],
            dtype=dtype,
            device=device,
        )

        # Concatenate prefix + current block
        z_full = torch.cat(generated_blocks + [z_block], dim=0) if generated_blocks else z_block

        # Euler integration from t=T to t=0
        for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
            z_full = euler_ode_step(
                dit=dit,
                z_t=z_full,
                text_latent=text_latent,
                ts_shape=ts_shape_k,
                text_shape=text_shape,
                t_curr=t_curr.item(),
                t_next=t_next.item(),
                T=T,
                attn_mask=attn_mask,
                guidance_scale=guidance_scale,
            )
            # Repaint: pin prefix blocks to their cleaned values
            if generated_blocks:
                prefix = torch.cat(generated_blocks, dim=0)
                z_full = torch.cat([prefix, z_full[prefix_len:]], dim=0)

        # Extract the just-denoised block
        z_clean = z_full[-curr_block_len:]
        generated_blocks.append(z_clean)

    return torch.cat(generated_blocks, dim=0)


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_timeseries(
    dit: MMLDMDiTModel,
    vae: MMLDMVAEModel,
    text_embedding: torch.Tensor,
    output_len: int,
    block_size: int,
    device: torch.device,
    T: float = 1.0,
    timestep_num: int = 20,
    guidance_scale: float = 2.0,
    use_adaptive_routing: bool = False,
    router: Optional[SemanticRouter] = None,
    text_str: Optional[str] = None,
) -> torch.Tensor:
    """End-to-end time series generation from text condition.

    Pipeline:
    1. Encode text -> text latent via VAE text encoder
    2. Generate latent blocks via DiT with Euler ODE
    3. Decode latent -> time series via VAE decoder

    Args:
        dit: pretrained DiT prior model.
        vae: pretrained VAE model (frozen).
        text_embedding: ``(1, text_dim)`` raw text embedding.
        output_len: desired output time series length.
        block_size: block size for generation.
        device: compute device.
        T: diffusion time horizon.
        timestep_num: number of ODE integration steps.
        guidance_scale: CFG guidance scale.
        use_adaptive_routing: use SemanticRouter for adaptive block sizes.
        router: optional SemanticRouter instance.
        text_str: optional text string for the router.

    Returns:
        Generated time series ``(1, output_len, ts_channels)``.
    """
    dit.eval()
    vae.eval()

    latent_dim = vae.config.latent_dim
    patch_size = vae.config.patch_size

    n_latent_tokens = output_len // patch_size
    n_blocks = max(1, (n_latent_tokens + block_size - 1) // block_size)

    # Step 1: Encode text -> text latent via frozen VAE text encoder
    text_embs = text_embedding.to(device)  # (1, text_dim)

    with torch.no_grad():
        # Use a dummy TS to trigger VAE's text encoder path
        dummy_ts = [torch.zeros(1, 1, device=device)]  # (1 token, 1 channel)
        enc_output = vae.encode(dummy_ts, text_embs)
        text_latent = enc_output.text_latents  # (1, latent_dim)

    text_shape = _shape_tensor([1], device)

    # Optional: prepare router input
    router_tokens = None
    if use_adaptive_routing and router is not None:
        router_tokens = text_embs.unsqueeze(1)  # (1, 1, text_dim)

    # Step 2: Generate latent blocks via DiT
    z_generated = generate_latent_blocks(
        dit=dit,
        text_latent=text_latent,
        text_shape=text_shape,
        n_blocks=n_blocks,
        block_size=block_size,
        latent_dim=latent_dim,
        device=device,
        T=T,
        timestep_num=timestep_num,
        guidance_scale=guidance_scale,
        use_adaptive_mask=use_adaptive_routing,
        router=router,
        text_tokens_for_router=router_tokens,
    )

    # Step 3: Decode latent -> time series
    if z_generated.shape[0] > n_latent_tokens:
        z = z_generated[:n_latent_tokens]
    elif z_generated.shape[0] < n_latent_tokens:
        pad = torch.zeros(
            n_latent_tokens - z_generated.shape[0], latent_dim,
            device=device, dtype=z_generated.dtype,
        )
        z = torch.cat([z_generated, pad], dim=0)
    else:
        z = z_generated

    z_shape = _shape_tensor([z.shape[0]], device)

    # Build decoder mask (TS-only, no text)
    text_shape_zero = _shape_tensor([0], device)
    n_full = z.shape[0] // block_size
    dec_block_sizes = [block_size] * n_full
    remainder = z.shape[0] - n_full * block_size
    if remainder > 0:
        dec_block_sizes.append(remainder)
    if not dec_block_sizes:
        dec_block_sizes = [z.shape[0]]

    attn_mask = create_multimodal_joint_mask(
        ts_shape=z_shape,
        text_shape=text_shape_zero,
        block_sizes=[dec_block_sizes],
        dtype=z.dtype,
        device=device,
    )

    recon = vae.decode(z, z_shape, z_shape, attn_mask=attn_mask)
    ts_out = recon[:, :output_len, :]

    return ts_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_models(args, device):
    """Load VAE, DiT, and optional router from checkpoints."""
    # VAE
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae_config = MMLDMVAEConfig(**vae_ckpt["config"])
    vae = MMLDMVAEModel(vae_config).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"Loaded VAE from {args.vae_checkpoint}")

    # DiT
    dit_ckpt = torch.load(args.dit_checkpoint, map_location=device)
    dit_config = MMLDMDiTConfig(**dit_ckpt["config"])
    dit = MMLDMDiTModel(dit_config).to(device)
    dit.load_state_dict(dit_ckpt["dit_state_dict"])
    dit.eval()
    for p in dit.parameters():
        p.requires_grad = False
    print(f"Loaded DiT from {args.dit_checkpoint}")

    # Optional router
    router = None
    if args.use_adaptive_routing:
        router = SemanticRouter(
            text_dim=vae_config.text_dim,
            n_latent=args.output_len // vae_config.patch_size,
        ).to(device)
        print("Adaptive routing enabled")

    return vae, vae_config, dit, dit_config, router


def _encode_text_sbert(text_str: str, text_dim: int, device: torch.device) -> torch.Tensor:
    """Encode text via Sentence-BERT, with fallback to random."""
    try:
        from transformers import AutoModel, AutoTokenizer

        sbert_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(sbert_name)
        sbert = AutoModel.from_pretrained(sbert_name).to(device)
        sbert.eval()
        inputs = tokenizer(text_str, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb = sbert(**inputs).last_hidden_state.mean(dim=1)  # (1, 384)
        if emb.shape[-1] != text_dim:
            emb = torch.nn.Linear(emb.shape[-1], text_dim, device=device)(emb)
        del sbert, tokenizer
        return emb
    except Exception:
        return torch.randn(1, text_dim, device=device)


def main():
    parser = argparse.ArgumentParser(description="MMLDM Inference & Evaluation")
    # Shared
    parser.add_argument("--dit_checkpoint", type=str, required=True)
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--timestep_num", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--use_adaptive_routing", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Single-sample mode
    parser.add_argument("--text", type=str, default=None, help="Text for single generation")
    parser.add_argument("--save_path", type=str, default=None)
    # Evaluation mode
    parser.add_argument("--eval_data_dir", type=str, default=None, help="TSFragment-600K dir for eval")
    parser.add_argument("--eval_datasets", type=str, nargs="+", default=["ETTh1"])
    parser.add_argument("--eval_time_intervals", type=int, nargs="+", default=[24])
    parser.add_argument("--eval_seed", type=int, default=42, help="Must match training split seed")
    parser.add_argument("--split_file", type=str, default=None, help="Path to splits.json for test split")
    parser.add_argument("--n_runs", type=int, default=1, help="Generation runs per sample (for MRR)")
    parser.add_argument("--metrics", type=str, default="MSE,WAPE,MRR")
    parser.add_argument("--eval_output", type=str, default=None, help="JSON path for eval results")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    vae, vae_config, dit, dit_config, router = _load_models(args, device)

    # ---- Evaluation mode ----
    if args.eval_data_dir is not None:
        metrics = [m.strip() for m in args.metrics.split(",")]
        needs_single = any(m in metrics for m in ("MSE", "WAPE"))
        needs_multi = "MRR" in metrics

        # Load test split (SampleID-level, no data leakage)
        if args.split_file is not None:
            test_ds = TSFragmentDataset(
                data_dir=args.eval_data_dir,
                datasets=args.eval_datasets,
                time_intervals=args.eval_time_intervals,
                max_samples=args.max_eval_samples,
                split="test", split_file=args.split_file,
            )
        else:
            # Fallback: random split (may leak for ETTh1 sliding windows)
            from torch.utils.data import random_split
            dataset = TSFragmentDataset(
                data_dir=args.eval_data_dir,
                datasets=args.eval_datasets,
                time_intervals=args.eval_time_intervals,
                max_samples=args.max_eval_samples,
            )
            val_size = min(len(dataset) // 10, 1000)
            _, test_ds = random_split(dataset, [len(dataset) - val_size, val_size],
                                      generator=torch.Generator().manual_seed(args.eval_seed))
        print(f"Eval dataset: {len(test_ds)} test samples")

        # Collect ground truth and generations
        all_ori = []       # list of (T, 1) arrays
        all_gen_runs = []  # list of list-of-(T, 1) arrays (one per run)

        for idx in range(len(test_ds)):
            sample = test_ds[idx]
            text_emb = sample["text_embedding"].unsqueeze(0).to(device)  # (1, 128)
            gt_ot = sample["ot"].numpy()  # (L, 1)
            L = gt_ot.shape[0]

            all_ori.append(gt_ot)  # keep original length
            run_gens = []

            for run in range(args.n_runs):
                torch.manual_seed(run)
                ts_out = generate_timeseries(
                    dit=dit, vae=vae,
                    text_embedding=text_emb,
                    output_len=L,
                    block_size=args.block_size,
                    device=device,
                    T=args.T,
                    timestep_num=args.timestep_num,
                    guidance_scale=args.guidance_scale,
                    use_adaptive_routing=args.use_adaptive_routing,
                    router=router,
                )
                gen_np = ts_out.squeeze(0).cpu().numpy()  # (L, 1)
                # Trim to min length in case of mismatch
                min_len = min(gt_ot.shape[0], gen_np.shape[0])
                run_gens.append(gen_np[:min_len])

            all_gen_runs.append(run_gens)

            if (idx + 1) % 50 == 0:
                print(f"  Generated {idx + 1}/{len(test_ds)} samples")

        # Align all to common min length
        min_L = min(o.shape[0] for o in all_ori)
        ori_arr = np.array([o[:min_L] for o in all_ori])  # (N, T, 1)

        results = {}

        if needs_single:
            # Average across runs for point-estimate metrics
            gen_avg = np.mean(
                [np.array([g[:min_L] for g in runs]) for runs in all_gen_runs],
                axis=1,
            )  # (N, T, 1)
            results.update(evaluate_single(ori_arr, gen_avg, metrics))

        if needs_multi and args.n_runs > 1:
            # Stack runs: (N, T, 1, K)
            gen_multi = np.stack(
                [np.array([g[:min_L] for g in runs]) for runs in all_gen_runs],
                axis=-1,
            )  # (N, T, 1, K) — but we need (N, T, 1, K), check shape
            # all_gen_runs[i] is list of K arrays each (T, 1)
            # np.array gives (K, T, 1), then stacking gives (N, K, T, 1)
            # Need (N, T, 1, K)
            gen_multi = np.array([
                np.stack([g[:min_L] for g in runs], axis=-1)  # (T, 1, K)
                for runs in all_gen_runs
            ])
            results.update(evaluate_multi(ori_arr, gen_multi, metrics, k=args.n_runs))

        print(f"\n{'='*50}")
        print(f"Evaluation results ({args.eval_datasets}, {args.eval_time_intervals}):")
        for k, v in results.items():
            print(f"  {k}: {v:.6f}")
        print(f"{'='*50}")

        if args.eval_output:
            save_results(results, args.eval_output)
        return

    # ---- Single-sample generation mode ----
    if args.text is None:
        parser.error("Provide --text for single generation, or --eval_data_dir for evaluation.")

    text_embedding = _encode_text_sbert(args.text, vae_config.text_dim, device)
    print(f"Text condition: '{args.text[:80]}'")

    print(f"Generating time series of length {args.output_len}...")
    ts_output = generate_timeseries(
        dit=dit, vae=vae,
        text_embedding=text_embedding,
        output_len=args.output_len,
        block_size=args.block_size,
        device=device,
        T=args.T,
        timestep_num=args.timestep_num,
        guidance_scale=args.guidance_scale,
        use_adaptive_routing=args.use_adaptive_routing,
        router=router,
        text_str=args.text,
    )

    print(f"Generated shape: {ts_output.shape}")

    if args.save_path:
        torch.save(ts_output.cpu(), args.save_path)
        print(f"Saved to {args.save_path}")
    else:
        ts_flat = ts_output.squeeze(0).squeeze(-1).cpu().numpy()
        print(f"First 10 values: {ts_flat[:10]}")
        print(f"Mean: {ts_flat.mean():.4f}, Std: {ts_flat.std():.4f}")


if __name__ == "__main__":
    main()
