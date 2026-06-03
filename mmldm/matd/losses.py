"""Losses for high-standard MATD training.

This file keeps the same public loss class names as the current framework while
adding production-grade details needed by the upgraded causal/SCCI/MoE/DiT
pipeline:
- optional Min-SNR diffusion weighting;
- robust reconstruction/delta/FFT losses;
- alignment loss with safe batch-size handling;
- generic loss combiner compatible with both flat and nested configs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Diffusion denoising loss with optional Min-SNR weighting.

    Args:
        min_snr_gamma: if not None, use Min-SNR-gamma style weights for a
            provided alpha_bar schedule and timestep tensor.
        prediction_type: 'epsilon' or 'v'.  This only affects diagnostics; the
            caller still passes the target tensor.
    """

    def __init__(self, min_snr_gamma: Optional[float] = None, prediction_type: str = "epsilon") -> None:
        super().__init__()
        self.min_snr_gamma = min_snr_gamma
        self.prediction_type = prediction_type

    @staticmethod
    def _min_snr_weight(t: torch.Tensor, alpha_bar: torch.Tensor, gamma: float, x: torch.Tensor) -> torch.Tensor:
        ab = alpha_bar.to(device=t.device, dtype=x.dtype).gather(0, t)
        snr = ab / (1.0 - ab).clamp_min(1e-8)
        weight = torch.minimum(snr, torch.full_like(snr, float(gamma))) / max(gamma, 1e-8)
        return weight

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        *,
        t: Optional[torch.Tensor] = None,
        alpha_bar: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        sq_err = (eps_pred - eps_true).pow(2)
        if weight is None and self.min_snr_gamma is not None and t is not None and alpha_bar is not None:
            weight = self._min_snr_weight(t, alpha_bar, self.min_snr_gamma, eps_pred)
        if weight is not None:
            while weight.dim() < sq_err.dim():
                weight = weight.unsqueeze(-1)
            sq_err = sq_err * weight
        loss = sq_err.mean()
        return {"loss_diffusion": loss}


class ReconstructionLoss(nn.Module):
    """L1 + lambda*MSE reconstruction loss."""

    def __init__(self, lambda_mse: float = 0.5) -> None:
        super().__init__()
        self.lambda_mse = lambda_mse

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = pred.float()
        target = target.float()
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        return {"loss_recon_l1": l1, "loss_recon_mse": mse, "loss_recon": l1 + self.lambda_mse * mse}


class DeltaLoss(nn.Module):
    """Temporal derivative loss up to a configurable finite-difference order."""

    def __init__(self, order: int = 1) -> None:
        super().__init__()
        self.order = order

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = pred.float()
        target = target.float()
        if pred.shape[1] <= self.order:
            z = pred.new_tensor(0.0)
            return {"loss_delta": z}
        dp, dt = pred, target
        for _ in range(self.order):
            dp = dp[:, 1:] - dp[:, :-1]
            dt = dt[:, 1:] - dt[:, :-1]
        return {"loss_delta": F.l1_loss(dp, dt)}


class FFTLoss(nn.Module):
    """Spectral magnitude loss on RFFT."""

    def __init__(self, log_magnitude: bool = True) -> None:
        super().__init__()
        self.log_magnitude = log_magnitude

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = pred.float()
        target = target.float()
        sp = torch.fft.rfft(pred, dim=1).abs()
        st = torch.fft.rfft(target, dim=1).abs()
        if self.log_magnitude:
            sp = torch.log1p(sp)
            st = torch.log1p(st)
        return {"loss_fft": F.l1_loss(sp, st)}


class DensityWeightedLoss(nn.Module):
    """MSE weighted by a density/raw-score map."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, raw_score: torch.Tensor) -> dict[str, torch.Tensor]:
        weight = raw_score.float()
        while weight.dim() < pred.dim():
            weight = weight.unsqueeze(-1)
        loss = ((pred.float() - target.float()).pow(2) * weight).mean()
        return {"loss_density_weighted": loss}


class AlignmentLoss(nn.Module):
    """Symmetric InfoNCE between paired time-series and text embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = max(temperature, 1e-6)

    def forward(self, ts_embed: torch.Tensor, text_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        B = ts_embed.shape[0]
        if B <= 1:
            z = ts_embed.new_tensor(0.0)
            return {"loss_align": z, "align_acc": z}
        ts = F.normalize(ts_embed.float(), dim=-1)
        txt = F.normalize(text_embed.float(), dim=-1)
        logits = ts @ txt.T / self.temperature
        labels = torch.arange(B, device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        acc = (logits.argmax(dim=-1) == labels).float().mean().detach()
        return {"loss_align": loss, "align_acc": acc}


class MoELosses(nn.Module):
    """Fallback MoE auxiliary losses for dense router probabilities.

    Prefer ``moe.last_aux_losses`` when using the upgraded sparse MoE.
    """

    def __init__(self, n_experts: int, lb_weight: float = 0.01, prior_weight: float = 0.01, entropy_weight: float = 0.01, target_entropy: Optional[float] = None) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.lb_weight = lb_weight
        self.prior_weight = prior_weight
        self.entropy_weight = entropy_weight
        self.target_entropy = math.log(n_experts) if target_entropy is None else target_entropy

    def forward(self, router_probs: torch.Tensor, prior_probs: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        E = router_probs.shape[-1]
        p = router_probs.reshape(-1, E).float().clamp_min(1e-8)
        usage = p.mean(dim=0)
        ideal = torch.ones_like(usage) / E
        loss_lb = (usage - ideal).pow(2).sum()
        if prior_probs is None:
            prior = torch.full_like(p, 1.0 / E)
        else:
            prior = prior_probs.reshape(-1, E).float().clamp_min(1e-8)
        loss_kl = (p * (p.log() - prior.log())).sum(dim=-1).mean()
        ent = -(p * p.log()).sum(dim=-1).mean()
        loss_ent = (ent - self.target_entropy).pow(2)
        loss_moe = self.lb_weight * loss_lb + self.prior_weight * loss_kl + self.entropy_weight * loss_ent
        return {"loss_load_balance": loss_lb, "loss_router_kl": loss_kl, "loss_entropy_target": loss_ent, "loss_moe": loss_moe}


class CausalLosses(nn.Module):
    """Fallback causal loss module for external causal graphs."""

    def __init__(self, dag_weight: float = 1.0, sparsity_weight: float = 0.1, mechanism_weight: float = 1.0) -> None:
        super().__init__()
        self.dag_weight = dag_weight
        self.sparsity_weight = sparsity_weight
        self.mechanism_weight = mechanism_weight

    @staticmethod
    def _notears_penalty(adj: torch.Tensor) -> torch.Tensor:
        d = adj.shape[-1]
        m = adj * adj
        return torch.linalg.matrix_exp(m).diagonal(dim1=-2, dim2=-1).sum(-1).sub(d).pow(2).mean()

    def forward(self, mechanism_pred: torch.Tensor, mechanism_target: torch.Tensor, causal_graph: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_mech = F.mse_loss(mechanism_pred, mechanism_target)
        loss_dag = self._notears_penalty(causal_graph)
        loss_sparsity = causal_graph.abs().mean()
        loss_causal = self.mechanism_weight * loss_mech + self.dag_weight * loss_dag + self.sparsity_weight * loss_sparsity
        return {"loss_mechanism": loss_mech, "loss_dag": loss_dag, "loss_sparsity": loss_sparsity, "loss_causal": loss_causal}


@dataclass
class LossWeights:
    diffusion: float = 1.0
    reconstruction: float = 0.0
    delta: float = 0.0
    fft: float = 0.0
    density_weighted: float = 0.0
    alignment: float = 0.0
    moe: float = 0.0
    causal: float = 0.0
    planner: float = 0.0
    scci: float = 0.0


def weights_from_config(config: dict) -> LossWeights:
    """Build LossWeights from either nested or flat MATD config dictionaries."""
    lw = config.get("loss_weights") if isinstance(config, dict) else None
    if isinstance(lw, dict):
        return LossWeights(**{k: lw.get(k, getattr(LossWeights(), k)) for k in LossWeights.__dataclass_fields__})
    return LossWeights(
        diffusion=config.get("lambda_diffusion", 1.0),
        reconstruction=config.get("lambda_x0", config.get("lambda_recon", 0.2)),
        delta=config.get("lambda_delta", 0.1),
        fft=config.get("lambda_fft", 0.05),
        alignment=config.get("lambda_align", 0.05),
        moe=config.get("lambda_moe", 0.01),
        causal=config.get("lambda_causal", 0.01),
        planner=config.get("lambda_plan", 0.5),
        scci=config.get("lambda_scci", 0.0),
    )


def compute_total_loss(loss_dicts: dict[str, dict[str, torch.Tensor]], weights: LossWeights) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine grouped losses into a weighted total and flat logging dict."""
    device = None
    for sub in loss_dicts.values():
        for v in sub.values():
            if torch.is_tensor(v):
                device = v.device
                break
        if device is not None:
            break
    total = torch.tensor(0.0, device=device)
    all_terms: dict[str, torch.Tensor] = {}
    group_spec = [
        ("diffusion", weights.diffusion, "loss_diffusion"),
        ("reconstruction", weights.reconstruction, "loss_recon"),
        ("delta", weights.delta, "loss_delta"),
        ("fft", weights.fft, "loss_fft"),
        ("density_weighted", weights.density_weighted, "loss_density_weighted"),
        ("alignment", weights.alignment, "loss_align"),
        ("moe", weights.moe, "loss_moe"),
        ("causal", weights.causal, "loss_causal"),
        ("planner", weights.planner, "loss_planner"),
        ("scci", weights.scci, "loss_scci"),
    ]
    for group_name, w, agg_key in group_spec:
        sub = loss_dicts.get(group_name)
        if sub is None:
            continue
        for k, v in sub.items():
            all_terms[k] = v
        if w != 0.0 and agg_key in sub:
            total = total + float(w) * sub[agg_key]
    all_terms["loss_total"] = total
    return total, all_terms
