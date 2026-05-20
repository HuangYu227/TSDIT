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

"""Evaluation metrics for MMLDM — compatible with T2S conventions.

All functions accept **numpy** arrays; no torch dependency.
Shapes follow T2S ``evaluation.py``:

* ``calculate_mse`` / ``calculate_wape``: input ``(B, dim, T)``.
* ``calculate_mrr``: ``ori (B, T, dim)``, ``gen (B, T, dim, K)``.

No inverse-normalization is applied — metrics operate on decoded
(original-scale) values, same as T2S.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# MSE — per-sample, per-dim mean, then global mean
# ---------------------------------------------------------------------------


def calculate_mse(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Mean Squared Error (T2S-compatible).

    Args:
        ori_data: ``(B, dim, T)`` ground truth.
        gen_data: ``(B, dim, T)`` predictions.

    Returns:
        Scalar MSE averaged over samples and dimensions.
    """
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[1]
    mse_values = []
    for i in range(n_samples):
        total = 0.0
        for j in range(n_series):
            total += np.mean((ori_data[i, j] - gen_data[i, j]) ** 2)
        mse_values.append(total / n_series)
    return float(np.mean(mse_values))


# ---------------------------------------------------------------------------
# WAPE — per-sample sum(|err|)/sum(|gt|), nanmean
# ---------------------------------------------------------------------------


def calculate_wape(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (T2S-compatible).

    Args:
        ori_data: ``(B, dim, T)`` ground truth.
        gen_data: ``(B, dim, T)`` predictions.

    Returns:
        Scalar WAPE (nanmean over samples).
    """
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[1]
    wape_values = []
    for i in range(n_samples):
        abs_err = 0.0
        abs_gt = 0.0
        for j in range(n_series):
            abs_err += np.sum(np.abs(ori_data[i, j] - gen_data[i, j]))
            abs_gt += np.sum(np.abs(ori_data[i, j]))
        wape_values.append(abs_err / abs_gt if abs_gt != 0 else np.nan)
    return float(np.nanmean(wape_values))


# ---------------------------------------------------------------------------
# MRR — cosine similarity ranking over K generations
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def calculate_mrr(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    k: Optional[int] = None,
    threshold: float = 0.5,
) -> float:
    """Mean Reciprocal Rank (T2S-compatible).

    Args:
        ori_data: ``(B, T, dim)`` ground truth.
        gen_data: ``(B, T, dim, K)`` multiple generations per sample.
        k: number of generations to use (default: all).
        threshold: cosine-similarity threshold for relevance.

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
# High-level evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_single(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    metrics: list[str],
) -> dict[str, float]:
    """Evaluate MSE and/or WAPE on single-generation results.

    Args:
        ori_data: ``(B, T, dim)`` ground truth.
        gen_data: ``(B, T, dim)`` predictions.
        metrics: list of metric names (``"MSE"``, ``"WAPE"``).

    Returns:
        Dict of ``{metric_name: value}``.
    """
    # T2S convention: MSE/WAPE expect (B, dim, T)
    ori = np.transpose(ori_data, (0, 2, 1))
    gen = np.transpose(gen_data, (0, 2, 1))

    result = {}
    if "MSE" in metrics:
        result["MSE"] = calculate_mse(ori, gen)
    if "WAPE" in metrics:
        result["WAPE"] = calculate_wape(ori, gen)
    return result


def evaluate_multi(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    metrics: list[str],
    k: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate MRR on multi-generation results.

    Args:
        ori_data: ``(B, T, dim)`` ground truth.
        gen_data: ``(B, T, dim, K)`` K generations per sample.
        metrics: list of metric names (``"MRR"``).
        k: number of generations to use.

    Returns:
        Dict of ``{metric_name: value}``.
    """
    result = {}
    if "MRR" in metrics:
        result["MRR"] = calculate_mrr(ori_data, gen_data, k=k)
    return result


def save_results(results: dict, path: str) -> None:
    """Save evaluation results to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy floats to Python floats for JSON serialization
    clean = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in results.items()}
    with open(p, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Results saved to {p}")
