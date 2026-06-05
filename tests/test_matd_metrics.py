import math

import numpy as np

from mmldm.matd.metrics_matd import (
    compute_multisample_metrics,
    compute_retrieval_diagnostics,
    compute_scale_diagnostics,
    ensure_btd,
    ensure_nktd,
)


def test_shape_utilities_cover_single_and_multi_sample_layouts():
    nt = np.zeros((2, 5))
    assert ensure_btd(nt).shape == (2, 5, 1)

    ntd = np.zeros((2, 5, 3))
    assert ensure_btd(ntd).shape == (2, 5, 3)

    legacy = np.zeros((2, 5, 1, 3))  # (N, T, D, K)
    assert ensure_nktd(legacy).shape == (2, 3, 5, 1)

    canonical = np.zeros((2, 3, 5, 1))  # (N, K, T, D)
    assert ensure_nktd(canonical).shape == (2, 3, 5, 1)


def test_scale_diagnostics_flags_tiny_denominator_and_nonfinite_values():
    real = np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 3.0]])
    gen = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 3.0]])

    diag = compute_scale_diagnostics(real, gen)

    assert diag["target_tiny_denominator_ratio"] == 0.5
    assert diag["real_finite_ratio"] < 1.0
    assert diag["gen_finite_ratio"] == 1.0


def test_multisample_best_mse_is_no_worse_than_first_sample():
    real = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    gen = np.array(
        [
            [[10.0, 10.0, 10.0], [1.0, 2.0, 3.0]],
            [[5.0, 5.0, 5.0], [0.0, 1.0, 0.0]],
        ]
    )[:, :, :, np.newaxis]  # (N, K, T, D)

    metrics = compute_multisample_metrics(real, gen)

    assert metrics["best_mse"] <= metrics["first_mse"]
    assert metrics["best_mse"] == 0.0
    assert math.isfinite(metrics["energy_score"])


def test_retrieval_diagnostics_reports_saturation_and_failures():
    real = np.array([[1.0, 0.0], [0.0, 1.0]])
    perfect = real[:, np.newaxis, :, np.newaxis]  # (N, K=1, T, D)
    perfect_diag = compute_retrieval_diagnostics(real, perfect, top_k=2)

    assert perfect_diag["mrr_at_k"] == 1.0
    assert perfect_diag["top1_self_rate"] == 1.0

    swapped = np.array(
        [
            [[0.0, 1.0], [0.0, 0.5]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )[:, :, :, np.newaxis]
    swapped_diag = compute_retrieval_diagnostics(real, swapped, top_k=4)

    assert swapped_diag["mrr_at_k"] < 1.0
    assert swapped_diag["top1_self_rate"] < 1.0
