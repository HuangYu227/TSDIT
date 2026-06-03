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
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_adapter import MATDDataModule, matd_collate_fn
from .matd_model import MATDModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MRR -- T2S-compatible (not in unified_metrics)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (flattened).

    Matches T2S evaluation.py cosine_similarity: both inputs are
    ``.ravel()``-ed before computing dot / (||a|| * ||b||).
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    dot = np.sum(a * b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def calculate_mrr(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    k: int | None = None,
    threshold: float = 0.5,
) -> float:
    """Mean Reciprocal Rank (T2S-compatible).

    Exact reimplementation of ``T2S/evaluation.py calculate_mrr``.

    Args:
        ori_data: ``(B, T, dim)`` ground truth (raw scale).
        gen_data: ``(B, T, dim, K)`` K generations per sample (raw scale).
        k: Number of generations to use (default: all K).
        threshold: Cosine similarity threshold for relevance
            (default 0.5, matching T2S global variable ``therehold``).

    Returns:
        Scalar MRR averaged over samples.
    """
    n_batch = ori_data.shape[0]
    n_generations = gen_data.shape[3]
    k = n_generations if k is None else min(k, n_generations)

    mrr_scores = np.zeros(n_batch)
    for b in range(n_batch):
        real = ori_data[b].flatten()
        sims = []
        for g in range(k):
            gen = gen_data[b, :, :, g].flatten()
            sims.append(_cosine_similarity(real, gen))

        sorted_idx = np.argsort(sims)[::-1]
        rank = None
        for position, idx in enumerate(sorted_idx):
            if sims[idx] > threshold:
                rank = position + 1  # 1-indexed rank
                break
        mrr_scores[b] = 1.0 / rank if rank is not None else 0.0

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
    ) -> None:
        self.model = model
        self.dm = data_module
        self.device = torch.device(device)
        self.n_samples = n_samples_per_text
        self.cfg_scale = cfg_scale
        self.ddim_steps = ddim_steps

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

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
        return x_norm * (ts_max - ts_min)[:, None] + ts_min[:, None]

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
            batch_size=64,
            shuffle=False,
            collate_fn=matd_collate_fn,
        )

        all_gen_single = []    # (N, T) first generation, denormalized
        all_gen_k = []         # (N, T, 1, K) all K generations, denormalized
        all_real_raw = []      # (N, T) ground truth, denormalized
        all_text_embed = []    # list of (B, text_dim)
        all_planner_meta = []  # list of (B, K, 9)
        all_scci_slots = []    # list of (B, n_slots, D)
        sample_offset = 0

        for batch in test_loader:
            x0_norm, texts = batch[0], batch[1]
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

            # Only keep the first 9 meta dims (consistent with decoder)
            all_text_embed.append(pooled.detach().cpu())               # (B, text_dim)
            all_planner_meta.append(planner_meta[..., :9].detach().cpu())  # (B, K, 9)
            all_scci_slots.append(scci_slots.detach().cpu())           # (B, n_slots, D)

            # -- Generate K independent samples per text --
            gen_k_list: list[np.ndarray] = []
            for _k in range(self.n_samples):
                gen = self.model.generate(
                    texts,
                    target_length=T,
                    cfg_scale=self.cfg_scale,
                    ddim_steps=self.ddim_steps,
                )  # (B, T, 1)
                gen_k_list.append(gen.squeeze(-1).cpu().numpy())  # (B, T)

            # Denormalize ground truth
            idx_s = sample_offset
            idx_e = sample_offset + B
            real_raw = self._denormalize(
                x0_norm.cpu().numpy(),
                test_ds.ts_min[idx_s:idx_e],
                test_ds.ts_max[idx_s:idx_e],
            )  # (B, T)

            # Denormalize all K generations
            for k_idx in range(self.n_samples):
                gen_k_list[k_idx] = self._denormalize(
                    gen_k_list[k_idx],
                    test_ds.ts_min[idx_s:idx_e],
                    test_ds.ts_max[idx_s:idx_e],
                )

            # Collect
            all_gen_single.append(gen_k_list[0])     # (B, T)
            gen_k_batch = np.stack(gen_k_list, axis=-1)  # (B, T, K)
            gen_k_batch = gen_k_batch[:, :, np.newaxis, :]  # (B, T, 1, K)
            all_gen_k.append(gen_k_batch)
            all_real_raw.append(real_raw)

            sample_offset += B

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
            threshold=0.5,
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
