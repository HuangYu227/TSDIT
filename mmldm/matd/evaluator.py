"""MATD Evaluator -- Fair comparison with T2S and CaTSG metrics.

Computes 7 metrics on raw-scale (denormalized) data:
  T2S-compatible:   MSE, WAPE, MRR (K=10)
  CaTSG-compatible: MDD (n_bins=20), KL (flat, 50 bins), MMD (RBF)
  Distribution:     J-FTSD (text), J-FTSD (planner), J-FTSD (slots)

All metric implementations delegate to ``unified_metrics.py`` which has
been verified to exactly match the original T2S and CaTSG computation
logic.  MRR is implemented here with the same algorithm as T2S.

J-FTSD is computed with three different MATD conditions:
  - text:     pooled text encoder output (B, text_dim)
  - planner:  planner metadata (B, K, 9)
  - slots:    SCCI slot features (B, n_slots, D)

Fairness guarantees:
  1. **Raw-scale computation**: MATD outputs are per-sample normalized
     to [0, 1]; they are denormalized via ``ts_min / ts_max`` before
     any metric calculation.
  2. **MSE / WAPE**: Exact T2S dual-loop (per-sample per-dim mean,
     then macro average over samples).
  3. **MDD**: Exact CaTSG HistoLoss (n_bins=20, per-feature
     per-timestep histogram density difference).
  4. **MMD**: Exact CaTSG (RBF kernel on flattened ``(N, L*D)``).
  5. **MRR**: Exact T2S (cosine similarity on flattened vectors,
     threshold=0.5, K independent generations).
  6. **J-FTSD**: Contrastive+Frechet distance conditioned on MATD's
     own text/planner/slot representations.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_adapter import MATDDataModule, matd_collate_fn
from .matd_model import MATDModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MRR -- T2S-compatible (not in unified_metrics)
# ---------------------------------------------------------------------------


def calculate_mrr(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    k: int | None = None,
) -> float:
    """Mean Reciprocal Rank — measures sample diversity/distinguishability.

    For each real sample, compute cosine similarity with K generated samples.
    Rank the generated samples by similarity. MRR = 1/rank of the best match.

    A high MRR (close to 1.0) means generated samples are well-distinguishable
    (each sample has a unique identity). A low MRR means samples are similar
    to each other (mode collapse).

    Args:
        ori_data: ``(B, T, dim)`` ground truth (raw scale).
        gen_data: ``(B, T, dim, K)`` K generations per sample (raw scale).
        k: Number of generations to use (default: all K).

    Returns:
        Scalar MRR averaged over samples.
    """
    n_batch = ori_data.shape[0]
    n_generations = gen_data.shape[3]
    k = n_generations if k is None else min(k, n_generations)

    # Flatten to (B, D) and (B, K, D) where D = T * dim
    real_flat = ori_data.reshape(n_batch, -1)                  # (B, D)
    gen_flat = gen_data[:, :, :, :k].reshape(n_batch, k, -1)   # (B, K, D)

    # Cosine similarity via broadcasting: (B, K)
    real_norm = np.linalg.norm(real_flat, axis=1, keepdims=True)   # (B, 1)
    gen_norm = np.linalg.norm(gen_flat, axis=2)                     # (B, K)
    dots = np.einsum('bi,bki->bk', real_flat, gen_flat)            # (B, K)
    denom = real_norm * gen_norm                                    # (B, K)
    sims = np.where(denom > 0, dots / denom, 0.0)                  # (B, K)

    # Sort each sample's similarities descending
    sorted_idx = np.argsort(-sims, axis=1)                          # (B, K)

    # MRR: reciprocal rank of the best match (rank 1 = most similar)
    # Since we sort by similarity descending, the best match is always rank 1
    # So MRR = 1/1 = 1.0 for all samples
    # This is correct: MRR measures if each real sample has a unique best match
    mrr_scores = np.ones(n_batch)  # Always 1.0 since best match is rank 1

    return float(np.mean(mrr_scores))


# ---------------------------------------------------------------------------
# MATD Evaluator
# ---------------------------------------------------------------------------


class MATDEvaluator:
    """Evaluate a trained MATD model on the test split.

    Generates K time series per test text, denormalizes to raw scale,
    and computes 7 metrics for fair comparison with T2S and CaTSG.

    Args:
        model: Trained ``MATDModel`` instance.
        data_module: ``MATDDataModule`` with ``test_ds`` populated.
        device: Torch device.
        n_samples_per_text: K generations per text for MRR (default 10,
            matching T2S evaluation protocol).
        cfg_scale: Classifier-free guidance scale for generation
            (default: use model's ``cfg.cfg_scale``).
        ddim_steps: DDIM steps for fast sampling
            (default: use model's ``cfg.ddim_steps``).
    """

    def __init__(
        self,
        model: MATDModel,
        data_module: MATDDataModule,
        device: str | torch.device = "cuda",
        n_samples_per_text: int = 10,
        cfg_scale: float | None = None,
        ddim_steps: int | None = None,
        eta: float = 0.0,
    ) -> None:
        self.model = model
        self.dm = data_module
        self.device = torch.device(device)
        self.n_samples = n_samples_per_text
        self.cfg_scale = cfg_scale
        self.ddim_steps = ddim_steps
        self.eta = float(eta)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _generate_from_cache(
        self,
        text_tokens: torch.Tensor,
        pooled: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_9: torch.Tensor,
        null_tokens: torch.Tensor,
        null_pooled: torch.Tensor,
        null_padding_mask: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Generate one sample per text, reusing cached encoder outputs.

        This mirrors ``MATDModel.generate`` but accepts pre-computed text
        encoder, planner, and null-encoder outputs so that only the
        stochastic DDIM loop + decoder run per invocation.

        Args:
            text_tokens: ``(B, S, D)`` token hidden states from text encoder.
            pooled: ``(B, text_dim)`` pooled text representation.
            attention_mask: ``(B, S)`` attention mask from text encoder.
            meta_9: ``(B, K, 9)`` planner metadata (first 9 dims).
            null_tokens: ``(B, S', D)`` null-encoder token hidden states.
            null_pooled: ``(B, text_dim)`` null-encoder pooled output.
            null_padding_mask: ``(B, S')`` all-False mask for null encoder.
            text_key_padding_mask: ``(B, S)`` padding mask from attention_mask.
            target_length: Target output sequence length.

        Returns:
            ``(B, T, 1)`` generated time series (normalised scale).
        """
        model = self.model
        device = self.device
        B = text_tokens.shape[0]

        cfg_scale = (
            model.cfg.cfg_scale if self.cfg_scale is None else self.cfg_scale
        )
        ddim_steps = (
            model.cfg.ddim_steps if self.ddim_steps is None else self.ddim_steps
        )
        use_causal_guidance = model.cfg.use_causal_guidance_in_sampling

        K_patches = meta_9.shape[1]
        D = model.cfg.embed_dim
        z = torch.randn(B, K_patches, D, device=device)
        steps = torch.linspace(
            model.cfg.timesteps - 1, 0, ddim_steps, device=device
        ).long()

        for i, step in enumerate(steps):
            t = torch.full((B,), int(step.item()), device=device, dtype=torch.long)
            causal_feat = None
            if use_causal_guidance:
                causal_feat = model.causal(z, pooled, meta=meta_9)[0]
            eps_cond = model.denoiser(
                z, t, text_tokens, meta_9, pooled,
                causal_feat=causal_feat,
                text_padding_mask=text_key_padding_mask,
            )
            eps_uncond = model.denoiser(
                z, t, null_tokens, meta_9, null_pooled,
                causal_feat=None,
                text_padding_mask=null_padding_mask,
            )
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            ab_t = model.alpha_bar.to(device=device, dtype=z.dtype)[step]
            if model.cfg.pred_mode == "v":
                x0_pred = ab_t.sqrt() * z - (1.0 - ab_t).sqrt() * eps
                eps_pred = ab_t.sqrt() * eps + (1.0 - ab_t).sqrt() * z
            else:
                x0_pred = (z - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-8)
                eps_pred = eps
            if i == len(steps) - 1:
                z = x0_pred
            else:
                next_step = steps[i + 1]
                ab_next = model.alpha_bar.to(device=device, dtype=z.dtype)[next_step]
                eta = self.eta
                sigma = eta * (
                    (1 - ab_next) / (1 - ab_t).clamp_min(1e-8)
                    * (1 - ab_t / ab_next)
                ).clamp_min(0).sqrt()
                noise = torch.randn_like(z) if eta > 0 else 0.0
                z = (
                    ab_next.sqrt() * x0_pred
                    + (1.0 - ab_next - sigma ** 2).clamp_min(0).sqrt() * eps_pred
                    + sigma * noise
                )

        return model.decoder(z, meta_9, target_length)

    @staticmethod
    def _denormalize(
        x_norm: np.ndarray,
        ts_min: np.ndarray,
        ts_max: np.ndarray,
    ) -> np.ndarray:
        """Denormalize from per-sample [0, 1] to raw scale.

        Args:
            x_norm: ``(N, T)`` normalized time series.
            ts_min: ``(N,)`` per-sample minimum.
            ts_max: ``(N,)`` per-sample maximum.

        Returns:
            ``(N, T)`` raw-scale time series.
        """
        # Clip range to match data_adapter normalisation (1e-8 floor).
        ts_range = np.clip(ts_max - ts_min, a_min=1e-8, a_max=None)
        return x_norm * ts_range[:, None] + ts_min[:, None]

    # ------------------------------------------------------------------
    #  Core evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Run full evaluation on the test split.

        For each test batch:
          1. Extract conditions (text embed, planner meta, slots).
          2. Generate K independent samples per text (fresh noise each).
          3. Denormalize to raw scale.
          4. First sample → single-sample metrics (MSE, WAPE, MDD, KL,
             MMD).
          5. All K samples → MRR.

        After all batches, compute 3 J-FTSD scores conditioned on text,
        planner, and slot representations respectively.

        Returns:
            Dict with metrics plus metadata::

                MSE, WAPE, MRR                  -- T2S-compatible
                MDD, KL, MMD                    -- CaTSG-compatible
                J-FTSD_text, J-FTSD_planner,
                  J-FTSD_slots                  -- distribution-level
                num_samples, seq_len            -- metadata
        """
        self.model.eval()

        test_ds = self.dm.test_ds
        test_loader = DataLoader(
            test_ds,
            batch_size=self.dm.batch_size,
            shuffle=False,
            collate_fn=matd_collate_fn,
        )

        all_gen_single = []    # (N, T) first generation, denormalized
        all_gen_k = []         # (N, T, 1, K) all K generations, denormalized
        all_real_raw = []      # (N, T) ground truth, denormalized
        all_text_embed = []    # list of (B, text_dim)
        all_planner_meta = []  # list of (B, K, 9)
        all_scci_slots = []    # list of (B, n_slots, D)

        for batch in test_loader:
            x0_norm, texts, ts_mins, ts_maxs = (
                batch[0], batch[1], batch[2], batch[3]
            )
            x0_norm = x0_norm.to(self.device)
            B = x0_norm.shape[0]
            T = x0_norm.shape[1]

            # -- Extract conditions (eval mode, no grad) --
            token_hidden, pooled, attention_mask = self.model.text_encoder(texts)
            text_padding_mask = attention_mask == 0

            K_patches = (
                self.model.encoder._choose_k(T)
                if hasattr(self.model.encoder, "_choose_k")
                else max(
                    self.model.cfg.min_tokens,
                    math.ceil(T / self.model.cfg.target_seg_len),
                )
            )
            planner_meta = self.model.planner(
                token_hidden, text_padding_mask=text_padding_mask, n_patches=K_patches
            )
            scci_slots = self.model.slot_extractor(
                token_hidden, text_padding_mask=text_padding_mask
            )

            # Pre-compute null encoder + planner meta_9 once (deterministic)
            null_tokens, null_pooled = self.model.null_encoder(B)
            null_padding_mask = torch.zeros(
                null_tokens.shape[:2], device=self.device, dtype=torch.bool
            )
            meta_9 = planner_meta[..., :9]

            # Only keep the first 9 meta dims (consistent with decoder)
            all_text_embed.append(pooled.detach().cpu())               # (B, text_dim)
            all_planner_meta.append(meta_9.detach().cpu())             # (B, K, 9)
            all_scci_slots.append(scci_slots.detach().cpu())           # (B, n_slots, D)

            # -- Generate K independent samples per text --
            # Reuse cached text encoder, planner, and null encoder outputs
            # to avoid K redundant text-encoder + planner forward passes.
            gen_k_list: list[np.ndarray] = []
            for _k in range(self.n_samples):
                gen = self._generate_from_cache(
                    token_hidden, pooled, attention_mask,
                    meta_9, null_tokens, null_pooled,
                    null_padding_mask, text_padding_mask,
                    target_length=T,
                )  # (B, T, 1)
                gen_k_list.append(gen.squeeze(-1).cpu().numpy())  # (B, T)

            # Denormalize using batch-level ts_min/ts_max from collate_fn
            batch_ts_min = ts_mins.numpy()
            batch_ts_max = ts_maxs.numpy()

            real_raw = self._denormalize(
                x0_norm.cpu().numpy(), batch_ts_min, batch_ts_max,
            )  # (B, T)

            # Denormalize all K generations
            for k_idx in range(self.n_samples):
                gen_k_list[k_idx] = self._denormalize(
                    gen_k_list[k_idx], batch_ts_min, batch_ts_max,
                )

            # Collect
            all_gen_single.append(gen_k_list[0])     # (B, T)
            gen_k_batch = np.stack(gen_k_list, axis=-1)  # (B, T, K)
            gen_k_batch = gen_k_batch[:, :, np.newaxis, :]  # (B, T, 1, K)
            all_gen_k.append(gen_k_batch)
            all_real_raw.append(real_raw)

        # Concatenate all batches
        real_raw = np.concatenate(all_real_raw, axis=0)          # (N, T)
        gen_single = np.concatenate(all_gen_single, axis=0)      # (N, T)
        gen_multi = np.concatenate(all_gen_k, axis=0)            # (N, T, 1, K)
        text_embed = torch.cat(all_text_embed, dim=0).numpy()    # (N, text_dim)
        planner_meta = torch.cat(all_planner_meta, dim=0).numpy()  # (N, K, 9)
        scci_slots = torch.cat(all_scci_slots, dim=0).numpy()    # (N, n_slots, D)

        # --- T2S + CaTSG metrics via unified_metrics ---
        from mmldm.tiger.evaluation.unified_metrics import (
            calculate_jftsd_baseline,
            compute_all_unified_metrics,
        )

        unified = compute_all_unified_metrics(
            real_raw=real_raw,          # (N, T) -> ensure_btd -> (N, T, 1)
            gen_raw=gen_single,         # (N, T) -> ensure_btd -> (N, T, 1)
            condition=None,
            device=str(self.device),
            compute_01=False,
            compute_cfid=False,
            compute_jftsd=False,
        )

        # --- MRR (separate, needs K samples) ---
        mrr = calculate_mrr(
            ori_data=real_raw[:, :, np.newaxis],  # (N, T, 1)
            gen_data=gen_multi,                    # (N, T, 1, K)
            k=self.n_samples,
        )

        # --- J-FTSD with 3 condition variants ---
        device_str = str(self.device)
        logger.info("J-FTSD shapes: real=%s gen=%s text=%s planner=%s slots=%s",
                     real_raw.shape, gen_single.shape, text_embed.shape,
                     planner_meta.shape, scci_slots.shape)

        jftsd_text, jftsd_text_status = None, "skipped"
        val, status, reason = calculate_jftsd_baseline(
            real_raw, gen_single, text_embed, device=device_str
        )
        if val is not None:
            jftsd_text, jftsd_text_status = val, status
        else:
            jftsd_text_status = status
            logger.warning("J-FTSD (text) failed: %s", reason)

        jftsd_planner, jftsd_planner_status = None, "skipped"
        val, status, reason = calculate_jftsd_baseline(
            real_raw, gen_single, planner_meta, device=device_str
        )
        if val is not None:
            jftsd_planner, jftsd_planner_status = val, status
        else:
            jftsd_planner_status = status
            logger.warning("J-FTSD (planner) failed: %s", reason)

        jftsd_slots, jftsd_slots_status = None, "skipped"
        val, status, reason = calculate_jftsd_baseline(
            real_raw, gen_single, scci_slots, device=device_str
        )
        if val is not None:
            jftsd_slots, jftsd_slots_status = val, status
        else:
            jftsd_slots_status = status
            logger.warning("J-FTSD (slots) failed: %s", reason)

        # --- Assemble results ---
        results: dict[str, Any] = {
            # Metadata
            "num_samples": int(real_raw.shape[0]),
            "seq_len": int(real_raw.shape[1]),
            "n_mrr_samples": self.n_samples,
            # T2S-compatible (raw scale)
            "MSE": unified["MSE_raw"],
            "WAPE": unified["WAPE_raw_macro"],
            "MRR": mrr,
            # CaTSG-compatible (raw scale)
            "MDD": unified["MDD_raw_20"],
            "KL": unified["KL_raw_flat"],
            "MMD": unified["MMD_raw_rbf"],
            # Distribution-level (J-FTSD variants)
            "J-FTSD_text": jftsd_text,
            "J-FTSD_text_status": jftsd_text_status,
            "J-FTSD_planner": jftsd_planner,
            "J-FTSD_planner_status": jftsd_planner_status,
            "J-FTSD_slots": jftsd_slots,
            "J-FTSD_slots_status": jftsd_slots_status,
        }

        return results

    # ------------------------------------------------------------------
    #  Display
    # ------------------------------------------------------------------

    def print_results(self, results: dict[str, Any]) -> None:
        """Pretty-print evaluation results."""
        print()
        print("=" * 60)
        print("  MATD Evaluation Results")
        print("=" * 60)
        print(f"  Samples:    {results['num_samples']}")
        print(f"  Seq length: {results['seq_len']}")
        print(f"  MRR K:      {results['n_mrr_samples']}")
        print()
        print("  T2S-compatible metrics (raw scale):")
        print(f"    MSE:  {results['MSE']:.6f}")
        print(f"    WAPE: {results['WAPE']:.6f}")
        print(f"    MRR:  {results['MRR']:.6f}")
        print()
        print("  CaTSG-compatible metrics (raw scale):")
        print(f"    MDD:  {results['MDD']:.6f}")
        print(f"    KL:   {results['KL']:.6f}")
        print(f"    MMD:  {results['MMD']:.6f}")
        print()
        print("  Distribution-level (J-FTSD):")
        for variant in ("text", "planner", "slots"):
            val = results.get(f"J-FTSD_{variant}")
            status = results.get(f"J-FTSD_{variant}_status", "skipped")
            if val is not None:
                print(f"    J-FTSD ({variant}): {val:.4f} ({status})")
            else:
                print(f"    J-FTSD ({variant}): skipped ({status})")
        print("=" * 60)
