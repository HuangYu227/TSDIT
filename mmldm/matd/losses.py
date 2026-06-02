"""MATD loss functions.

All loss modules follow a uniform protocol:

- Inherit from nn.Module.
- forward(...) -> dict[str, Tensor] returning named scalar losses.
  Every dict contains a top-level aggregate key so the caller can always
  do sum(loss_dict.values()) or pick individual terms.

Reference: MATD framework design doc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
#  1. Diffusion Loss  (MSE noise prediction)
# ===========================================================================


class DiffusionLoss(nn.Module):
    """Standard MSE denoising loss for diffusion / flow-matching.

    Computes ||eps_pred - eps_true||^2 averaged over all elements.
    Optionally applies per-element weighting (e.g. SNR-based).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """MSE between predicted and true noise.

        Args:
            eps_pred:  (B, ...) predicted noise / velocity.
            eps_true:  (B, ...) ground-truth noise / velocity.
            weight:    (B,) or broadcastable per-sample weight.  If None
                       all samples are weighted equally.

        Returns:
            "{"loss_diffusion": scalar}".
        """
        sq_err = (eps_pred - eps_true).pow(2)  # element-wise
        if weight is not None:
            # Reshape weight to broadcast over non-batch dims
            while weight.dim() < sq_err.dim():
                weight = weight.unsqueeze(-1)
            sq_err = sq_err * weight
        loss = sq_err.mean()
        return {"loss_diffusion": loss}


# ===========================================================================
#  2. Reconstruction Loss  (L1 + lambda * MSE)
# ===========================================================================


class ReconstructionLoss(nn.Module):
    """Combined L1 and MSE reconstruction loss.

    L = L1(x, y) + lambda_mse * MSE(x, y)"""

    def __init__(self, lambda_mse: float = 0.5) -> None:
        super().__init__()
        self.lambda_mse = lambda_mse

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute reconstruction loss."""
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        total = l1 + self.lambda_mse * mse
        return {"loss_recon_l1": l1, "loss_recon_mse": mse, "loss_recon": total}


# ===========================================================================
#  3. Delta Loss  (first-order finite differences)
# ===========================================================================


class DeltaLoss(nn.Module):
    """L1 loss on first-order temporal differences.

    Encourages the predicted sequence to match the temporal dynamics of
    the target by penalising |(x_t - x_{t-1}) - (y_t - y_{t-1})|."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute delta (temporal-derivative) loss."""
        delta_pred = pred[:, 1:] - pred[:, :-1]
        delta_target = target[:, 1:] - target[:, :-1]
        loss = F.l1_loss(delta_pred, delta_target)
        return {"loss_delta": loss}


# ===========================================================================
#  4. FFT Loss  (spectral magnitude)
# ===========================================================================


class FFTLoss(nn.Module):
    """L1 loss on RFFT magnitude spectra.

    Encourages the predicted signal to match the frequency content of the
    target by comparing |RFFT(x)| vs |RFFT(y)|."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute FFT magnitude loss."""
        spec_pred = torch.fft.rfft(pred.float(), dim=1)
        spec_target = torch.fft.rfft(target.float(), dim=1)
        mag_pred = spec_pred.abs()
        mag_target = spec_target.abs()
        loss = F.l1_loss(mag_pred, mag_target)
        return {"loss_fft": loss}


# ===========================================================================
#  5. Density-Weighted Loss
# ===========================================================================


class DensityWeightedLoss(nn.Module):
    """MSE weighted element-wise by a density / raw score map.

    Useful when certain time-steps or patches carry more information and
    should contribute proportionally more to the gradient."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, raw_score: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute density-weighted MSE."""
        sq_err = (pred - target).pow(2)
        weighted = sq_err * raw_score
        loss = weighted.mean()
        return {"loss_density_weighted": loss}


# ===========================================================================
#  6. Alignment Loss  (symmetric InfoNCE)
# ===========================================================================


class AlignmentLoss(nn.Module):
    """Symmetric InfoNCE between time-series and text embeddings.

    Encourages paired (ts, text) representations to be closer than
    unpaired ones, following the CLIP-style contrastive objective::

        L = 0.5 * (CE(logits_ts2txt, labels) + CE(logits_txt2ts, labels))"""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, ts_embed: torch.Tensor, text_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute symmetric InfoNCE loss."""
        ts_embed = F.normalize(ts_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        logits = ts_embed @ text_embed.T / self.temperature  # (B, B)
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_ts2txt = F.cross_entropy(logits, labels)
        loss_txt2ts = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_ts2txt + loss_txt2ts)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
        return {"loss_align": loss, "align_acc": acc}


# ===========================================================================
#  7. MoE Losses  (load balancing + router prior KL + entropy)
# ===========================================================================


class MoELosses(nn.Module):
    """Auxiliary losses for Mixture-of-Experts routing.

    Three terms:

    1. Load balancing: encourages uniform expert utilisation.
    2. Router prior KL: regularises toward a uniform prior via KL(router || prior).
    3. Entropy target: penalises deviation from a target entropy value."""

    def __init__(self, n_experts: int, lb_weight: float = 0.01, prior_weight: float = 0.01,
        entropy_weight: float = 0.01, target_entropy: Optional[float] = None) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.lb_weight = lb_weight
        self.prior_weight = prior_weight
        self.entropy_weight = entropy_weight
        self.target_entropy = target_entropy if target_entropy is not None else math.log(n_experts)

    def forward(self, router_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute MoE auxiliary losses.  router_probs: (..., E)"""
        E = router_probs.shape[-1]
        assert E == self.n_experts
        router_probs = router_probs.reshape(-1, E)  # flatten to (N, E)
        # Load balancing
        assignments = router_probs.argmax(dim=-1)
        one_hot = F.one_hot(assignments, E).float()
        f = one_hot.mean(dim=0)
        p = router_probs.mean(dim=0)
        loss_lb = E * (f * p).sum()
        # Router prior KL
        uniform = torch.full_like(router_probs, 1.0 / E)
        rp_clamp = router_probs.clamp(min=1e-8)
        kl = (rp_clamp * (rp_clamp.log() - uniform.log())).sum(dim=-1)
        loss_kl = kl.mean()
        # Entropy target
        ent = -(rp_clamp * rp_clamp.log()).sum(dim=-1)
        mean_ent = ent.mean()
        loss_ent = (mean_ent - self.target_entropy).pow(2)
        loss_moe = self.lb_weight * loss_lb + self.prior_weight * loss_kl + self.entropy_weight * loss_ent
        return {"loss_load_balance": loss_lb, "loss_router_kl": loss_kl,
            "loss_entropy_target": loss_ent, "loss_moe": loss_moe}


# ===========================================================================
#  8. Causal Losses  (mechanism + NOTEARS DAG + sparsity)
# ===========================================================================


class CausalLosses(nn.Module):
    """Losses for the causal discovery module (SCMON / C-SCMON).

    Three terms:

    1. Mechanism prediction loss: MSE between predicted and target mechanism states.
    2. NOTEARS DAG penalty: tr(e^{A o A}) - d, zero iff DAG (Zheng et al., 2018).
    3. Sparsity loss: L1 on adjacency matrix."""

    def __init__(self, dag_weight: float = 1.0, sparsity_weight: float = 0.1,
        mechanism_weight: float = 1.0) -> None:
        super().__init__()
        self.dag_weight = dag_weight
        self.sparsity_weight = sparsity_weight
        self.mechanism_weight = mechanism_weight

    @staticmethod
    def _notears_penalty(adj: torch.Tensor) -> torch.Tensor:
        """NOTEARS acyclicity constraint: tr(e^{A o A}) - d."""
        d = adj.size(0)
        M = adj * adj
        eye = torch.eye(d, device=adj.device, dtype=adj.dtype)
        exp_m = eye + M
        power = M.clone()
        for k in range(2, 6):
            power = power @ M / k
            exp_m = exp_m + power
        return exp_m.trace() - d

    def forward(self, mechanism_pred: torch.Tensor, mechanism_target: torch.Tensor,
        causal_graph: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute causal discovery losses."""
        loss_mech = F.mse_loss(mechanism_pred, mechanism_target)
        loss_dag = self._notears_penalty(causal_graph)
        loss_sparsity = causal_graph.abs().mean()
        loss_causal = self.mechanism_weight * loss_mech + self.dag_weight * loss_dag + self.sparsity_weight * loss_sparsity
        return {"loss_mechanism": loss_mech, "loss_dag": loss_dag,
            "loss_sparsity": loss_sparsity, "loss_causal": loss_causal}


# ===========================================================================
#  9. Total Loss Combiner
# ===========================================================================


@dataclass
class LossWeights:
    """Configurable weights for each loss component.

    All weights default to 0.0 so that only explicitly enabled losses
    contribute to the total."""

    diffusion: float = 1.0
    reconstruction: float = 0.0
    delta: float = 0.0
    fft: float = 0.0
    density_weighted: float = 0.0
    alignment: float = 0.0
    moe: float = 0.0
    causal: float = 0.0
    planner: float = 0.0


def compute_total_loss(
    loss_dicts: dict[str, dict[str, torch.Tensor]],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine multiple loss dicts into a single scalar.

    Args:
        loss_dicts: Mapping from loss-group name to its loss dict.
        weights: LossWeights dataclass with per-group scalar weights.

    Returns:
        (total_loss, all_terms) where total_loss is the weighted
        scalar sum and all_terms is a flat dict of every individual
        loss term."""
    total = torch.tensor(0.0)
    all_terms: dict[str, torch.Tensor] = {}
    group_spec: list[tuple[str, float, str]] = [
        ("diffusion",        weights.diffusion,        "loss_diffusion"),
        ("reconstruction",   weights.reconstruction,   "loss_recon"),
        ("delta",            weights.delta,             "loss_delta"),
        ("fft",              weights.fft,               "loss_fft"),
        ("density_weighted", weights.density_weighted,  "loss_density_weighted"),
        ("alignment",        weights.alignment,         "loss_align"),
        ("moe",              weights.moe,               "loss_moe"),
        ("causal",           weights.causal,            "loss_causal"),
        ("planner",          weights.planner,           "loss_planner"),
    ]
    for group_name, w, agg_key in group_spec:
        if w == 0.0:
            continue
        sub = loss_dicts.get(group_name)
        if sub is None:
            continue
        for k, v in sub.items():
            all_terms[k] = v
        if agg_key in sub:
            total = total + w * sub[agg_key]
    all_terms["loss_total"] = total
    return total, all_terms
