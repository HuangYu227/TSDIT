"""MATD v2 losses: framework-aligned loss design.

Four loss groups:
1. LatentDiffusionBridgeLoss   -- DiT latent denoising + pred-x0 bridge.
2. AdaptivePatchFieldLoss      -- decoder field reconstruction with shape/frequency terms.
3. TextLayoutLoss              -- text-to-adaptive-patch layout supervision.
4. CausalSemanticRouterLoss    -- causal learner + SCCI + MoE auxiliary mechanism loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _as_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _zero_like_reference(ref: Optional[torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    if ref is not None and torch.is_tensor(ref):
        return ref.new_tensor(0.0)
    return torch.tensor(0.0, device=device)


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.clamp_min(eps).log()


def _extract_alpha_bar(alpha_bar: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ab = alpha_bar.to(device=t.device, dtype=x.dtype).gather(0, t)
    return ab.view(x.shape[0], *([1] * (x.dim() - 1)))


def _finite_difference(x: torch.Tensor, order: int = 1) -> torch.Tensor:
    y = x
    for _ in range(order):
        if y.shape[1] <= 1:
            return y.new_zeros(y.shape[0], 0, *y.shape[2:])
        y = y[:, 1:] - y[:, :-1]
    return y


def _symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean()


def _js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-6,
    log_ratio_clip: float = 8.0,
) -> torch.Tensor:
    """Bounded Jensen-Shannon divergence — more stable than symmetric KL."""
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    log_p_m = (p.log() - m.log()).clamp(-log_ratio_clip, log_ratio_clip)
    log_q_m = (q.log() - m.log()).clamp(-log_ratio_clip, log_ratio_clip)
    js = 0.5 * (p * log_p_m).sum(dim=-1) + 0.5 * (q * log_q_m).sum(dim=-1)
    return js.mean()


# -----------------------------------------------------------------------------
# 1) Latent diffusion bridge
# -----------------------------------------------------------------------------


class LatentDiffusionBridgeLoss(nn.Module):
    """Diffusion denoising loss plus a clean-latent bridge.

    The denoiser is trained on the usual epsilon/v target, but the same network
    output is also converted to an implied clean latent ``pred_x0`` and aligned
    with the clean MATD latent ``z_clean``.  This replaces the old practice of
    decoding arbitrary noisy latent ``z_t`` directly into a time series.

    Args:
        min_snr_gamma: Optional Min-SNR-gamma weighting.  If ``None``, uses
            unweighted MSE on the denoising target.
        latent_x0_weight: Weight of SmoothL1(pred_x0, z_clean).
        latent_cos_weight: Weight of cosine alignment between pred_x0 and z_clean.
    """

    def __init__(
        self,
        min_snr_gamma: Optional[float] = 5.0,
        latent_x0_weight: float = 0.25,
        latent_cos_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.min_snr_gamma = min_snr_gamma
        self.latent_x0_weight = float(latent_x0_weight)
        self.latent_cos_weight = float(latent_cos_weight)

    @staticmethod
    def min_snr_weight(
        t: torch.Tensor,
        alpha_bar: torch.Tensor,
        gamma: float,
        ref: torch.Tensor,
        prediction_type: str = "eps",
    ) -> torch.Tensor:
        ab = alpha_bar.to(device=t.device, dtype=torch.float32).gather(0, t)
        snr = ab / (1.0 - ab).clamp_min(1e-8)
        clipped = torch.minimum(snr, torch.full_like(snr, float(gamma)))
        pred = prediction_type.lower()
        if pred in {"v", "velocity", "v_prediction"}:
            weight = clipped / (snr + 1.0).clamp_min(1e-8)
        else:
            weight = clipped / snr.clamp_min(1e-8)
        weight = torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=0.0)
        while weight.dim() < ref.dim():
            weight = weight.unsqueeze(-1)
        return weight.to(dtype=ref.dtype)

    @staticmethod
    def predict_x0(
        z_t: torch.Tensor,
        model_out: torch.Tensor,
        t: torch.Tensor,
        alpha_bar: torch.Tensor,
        prediction_type: str = "eps",
    ) -> torch.Tensor:
        ab = _extract_alpha_bar(alpha_bar, t, z_t)
        pred = prediction_type.lower()
        if pred in {"v", "velocity", "v_prediction"}:
            return ab.sqrt() * z_t - (1.0 - ab).sqrt() * model_out
        return (z_t - (1.0 - ab).sqrt() * model_out) / ab.sqrt().clamp_min(1e-8)

    def forward(
        self,
        model_out: torch.Tensor,
        target: torch.Tensor,
        *,
        z_t: torch.Tensor,
        z_clean: torch.Tensor,
        t: torch.Tensor,
        alpha_bar: torch.Tensor,
        prediction_type: str = "eps",
    ) -> dict[str, torch.Tensor]:
        model_out_f = _as_float(model_out)
        target_f = _as_float(target)
        sq_err = (model_out_f - target_f).pow(2)
        if self.min_snr_gamma is not None:
            weight = self.min_snr_weight(t, alpha_bar, self.min_snr_gamma, sq_err, prediction_type)
            sq_err = sq_err * weight
        loss_denoise = sq_err.mean()

        pred_x0 = self.predict_x0(_as_float(z_t), model_out_f, t, alpha_bar, prediction_type)
        pred_x0 = pred_x0.clamp(-50.0, 50.0)  # Prevent numerical explosion at high noise
        z_clean_f = _as_float(z_clean).detach()
        loss_latent_x0 = F.smooth_l1_loss(pred_x0, z_clean_f)

        pred_flat = F.normalize(pred_x0.reshape(pred_x0.shape[0], -1), dim=-1)
        clean_flat = F.normalize(z_clean_f.reshape(z_clean_f.shape[0], -1), dim=-1)
        loss_latent_cos = (1.0 - (pred_flat * clean_flat).sum(dim=-1)).mean()

        total = (
            loss_denoise
            + self.latent_x0_weight * loss_latent_x0
            + self.latent_cos_weight * loss_latent_cos
        )
        return {
            "loss_diffusion_denoise": loss_denoise,
            "loss_latent_x0": loss_latent_x0,
            "loss_latent_cos": loss_latent_cos,
            "loss_diffusion_bridge": total,
        }


# -----------------------------------------------------------------------------
# 2) Adaptive patch neural-field loss
# -----------------------------------------------------------------------------


class AdaptivePatchFieldLoss(nn.Module):
    """Shape-aware reconstruction for the adaptive neural-field decoder.

    This loss is intentionally time-series specific: value, local slope,
    curvature, and spectral magnitude are optimized together.  It is not a
    direct proxy for WAPE/KL/MDD, but it attacks the same failure mode that often
    creates good MSE with poor amplitude/distribution statistics: over-smoothed
    mean-like reconstructions.
    """

    def __init__(
        self,
        mse_weight: float = 0.5,
        delta_weight: float = 0.10,
        curvature_weight: float = 0.05,
        fft_weight: float = 0.05,
        range_weight: float = 0.02,
        use_log_fft: bool = True,
        valid_min: float = 0.0,
        valid_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.delta_weight = float(delta_weight)
        self.curvature_weight = float(curvature_weight)
        self.fft_weight = float(fft_weight)
        self.range_weight = float(range_weight)
        self.use_log_fft = bool(use_log_fft)
        self.valid_min = float(valid_min)
        self.valid_max = float(valid_max)

    def _fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] <= 1:
            return pred.new_tensor(0.0)
        sp = torch.fft.rfft(pred, dim=1).abs()
        st = torch.fft.rfft(target, dim=1).abs()
        if self.use_log_fft:
            sp = torch.log1p(sp + 1e-8)
            st = torch.log1p(st + 1e-8)
        return F.l1_loss(sp, st)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = _as_float(pred)
        target = _as_float(target)
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        if pred.dim() == 2:
            pred = pred.unsqueeze(-1)

        loss_l1 = F.l1_loss(pred, target)
        loss_mse = F.mse_loss(pred, target)

        d1_pred = _finite_difference(pred, order=1)
        d1_tgt = _finite_difference(target, order=1)
        loss_delta = F.l1_loss(d1_pred, d1_tgt) if d1_pred.numel() > 0 else pred.new_tensor(0.0)

        d2_pred = _finite_difference(pred, order=2)
        d2_tgt = _finite_difference(target, order=2)
        loss_curv = F.l1_loss(d2_pred, d2_tgt) if d2_pred.numel() > 0 else pred.new_tensor(0.0)

        loss_fft = self._fft_loss(pred, target)
        lower = F.relu(self.valid_min - pred).pow(2)
        upper = F.relu(pred - self.valid_max).pow(2)
        loss_range = (lower + upper).mean()

        total = (
            loss_l1
            + self.mse_weight * loss_mse
            + self.delta_weight * loss_delta
            + self.curvature_weight * loss_curv
            + self.fft_weight * loss_fft
            + self.range_weight * loss_range
        )
        return {
            "loss_field_l1": loss_l1,
            "loss_field_mse": loss_mse,
            "loss_field_delta": loss_delta,
            "loss_field_curvature": loss_curv,
            "loss_field_fft": loss_fft,
            "loss_field_range": loss_range,
            "loss_patch_field": total,
        }


class EndpointSeriesLoss(nn.Module):
    """Sequence-level loss on the denoiser-implied clean endpoint.

    The regular bridge objective aligns ``pred_x0`` with the clean latent.  This
    loss closes the train/eval gap by decoding that endpoint and optimizing the
    actual time-series statistics used by evaluation.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        moment_weight: float = 0.2,
        delta_weight: float = 0.2,
        acf_weight: float = 0.1,
        acf_lags: int = 8,
    ) -> None:
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.moment_weight = float(moment_weight)
        self.delta_weight = float(delta_weight)
        self.acf_weight = float(acf_weight)
        self.acf_lags = int(acf_lags)

    @staticmethod
    def _ensure_btc(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1)
        return x

    @staticmethod
    def _moment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=1)
        tgt_mean = target.mean(dim=1)
        pred_std = pred.std(dim=1, unbiased=False)
        tgt_std = target.std(dim=1, unbiased=False)
        pred_range = pred.amax(dim=1) - pred.amin(dim=1)
        tgt_range = target.amax(dim=1) - target.amin(dim=1)
        return (
            F.l1_loss(pred_mean, tgt_mean)
            + F.l1_loss(pred_std, tgt_std)
            + F.l1_loss(pred_range, tgt_range)
        )

    @staticmethod
    def _acf(x: torch.Tensor, max_lag: int) -> torch.Tensor:
        B, T, C = x.shape
        if T <= 1 or max_lag <= 0:
            return x.new_zeros(B, 0, C)
        centered = x - x.mean(dim=1, keepdim=True)
        var = centered.pow(2).mean(dim=1).clamp_min(1e-6)
        feats = []
        for lag in range(1, min(max_lag, T - 1) + 1):
            cov = (centered[:, :-lag] * centered[:, lag:]).mean(dim=1)
            feats.append(cov / var)
        if not feats:
            return x.new_zeros(B, 0, C)
        return torch.stack(feats, dim=1).clamp(-1.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self._ensure_btc(_as_float(pred))
        target = self._ensure_btc(_as_float(target))
        if pred.shape != target.shape:
            raise ValueError(f"endpoint pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

        loss_l1 = F.l1_loss(pred, target)
        loss_mse = F.mse_loss(pred, target)
        loss_recon = loss_l1 + 0.5 * loss_mse

        d_pred = _finite_difference(pred, order=1)
        d_tgt = _finite_difference(target, order=1)
        loss_delta = F.l1_loss(d_pred, d_tgt) if d_pred.numel() > 0 else pred.new_tensor(0.0)

        loss_moment = self._moment_loss(pred, target)
        pred_acf = self._acf(pred, self.acf_lags)
        tgt_acf = self._acf(target, self.acf_lags)
        loss_acf = F.l1_loss(pred_acf, tgt_acf) if pred_acf.numel() > 0 else pred.new_tensor(0.0)

        total = (
            self.recon_weight * loss_recon
            + self.moment_weight * loss_moment
            + self.delta_weight * loss_delta
            + self.acf_weight * loss_acf
        )
        return {
            "loss_endpoint_l1": loss_l1,
            "loss_endpoint_mse": loss_mse,
            "loss_endpoint_recon": loss_recon,
            "loss_endpoint_moment": loss_moment,
            "loss_endpoint_delta": loss_delta,
            "loss_endpoint_acf": loss_acf,
            "loss_endpoint_total": total,
        }


# -----------------------------------------------------------------------------
# 3) Text-to-layout loss
# -----------------------------------------------------------------------------


class PlanConsistencyLoss(nn.Module):
    """Text latent plan supervision via sequence statistics and contrastive binding."""

    def __init__(
        self,
        plan_dim: int,
        text_dim: int,
        stats_weight: float = 0.2,
        contrast_weight: float = 0.05,
        kl_weight: float = 0.001,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.stats_weight = float(stats_weight)
        self.contrast_weight = float(contrast_weight)
        self.kl_weight = float(kl_weight)
        self.temperature = float(temperature)
        self.stats_head = nn.Sequential(
            nn.LayerNorm(plan_dim),
            nn.Linear(plan_dim, plan_dim),
            nn.SiLU(),
            nn.Linear(plan_dim, 6),
        )
        self.plan_proj = nn.Linear(plan_dim, plan_dim)
        self.text_proj = nn.Linear(text_dim, plan_dim)

    @staticmethod
    def _ensure_btc(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1)
        return x

    @staticmethod
    def _target_stats(x: torch.Tensor) -> torch.Tensor:
        x = PlanConsistencyLoss._ensure_btc(_as_float(x))
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2), unbiased=False)
        value_range = x.amax(dim=(1, 2)) - x.amin(dim=(1, 2))
        delta = _finite_difference(x, order=1)
        if delta.numel() > 0:
            mean_abs_delta = delta.abs().mean(dim=(1, 2))
            max_abs_delta = delta.abs().amax(dim=(1, 2))
        else:
            mean_abs_delta = x.new_zeros(x.shape[0])
            max_abs_delta = x.new_zeros(x.shape[0])

        centered = x - x.mean(dim=1, keepdim=True)
        psd = torch.fft.rfft(centered.float(), dim=1).abs().pow(2)
        if psd.shape[1] > 1:
            low_end = min(4, psd.shape[1])
            low_energy = psd[:, 1:low_end].sum(dim=(1, 2))
            total_energy = psd[:, 1:].sum(dim=(1, 2)).clamp_min(1e-8)
            low_freq_ratio = low_energy / total_energy
        else:
            low_freq_ratio = x.new_zeros(x.shape[0])

        return torch.stack(
            [mean, std, value_range, mean_abs_delta, max_abs_delta, low_freq_ratio.to(x.dtype)],
            dim=-1,
        )

    def forward(
        self,
        plan_global: torch.Tensor,
        pooled_text: torch.Tensor,
        x0: torch.Tensor,
        plan_kl: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        target = self._target_stats(x0).detach().to(plan_global.dtype)
        pred = self.stats_head(plan_global)
        loss_stats = F.smooth_l1_loss(pred, target)

        plan_z = F.normalize(self.plan_proj(plan_global), dim=-1)
        text_z = F.normalize(self.text_proj(pooled_text.to(plan_global.dtype)), dim=-1)
        logits = (plan_z @ text_z.T) / max(self.temperature, 1e-6)
        labels = torch.arange(plan_global.shape[0], device=plan_global.device)
        if plan_global.shape[0] > 1:
            loss_contrast = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
        else:
            loss_contrast = plan_global.new_tensor(0.0)

        loss_kl = plan_kl if torch.is_tensor(plan_kl) else plan_global.new_tensor(0.0)
        total = (
            self.stats_weight * loss_stats
            + self.contrast_weight * loss_contrast
            + self.kl_weight * loss_kl
        )
        return {
            "loss_plan_stats": loss_stats,
            "loss_plan_contrast": loss_contrast,
            "loss_plan_kl": loss_kl,
            "loss_plan_total": total,
            "loss_text_layout": total,
        }


_START, _END, _CENTER, _LENGTH, _LOG_LEN = range(5)
_MASS, _DENSITY, _LOG_DENS, _ORDER = 5, 6, 7, 8


class TextLayoutLoss(nn.Module):
    """Supervise text-predicted patch metadata with tokenizer oracle metadata."""

    def __init__(
        self,
        beta: float = 1.0,
        center_weight: float = 1.0,
        length_weight: float = 2.0,
        mass_weight: float = 1.0,
        density_weight: float = 0.5,
        cdf_weight: float = 2.0,
        distribution_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.center_weight = float(center_weight)
        self.length_weight = float(length_weight)
        self.mass_weight = float(mass_weight)
        self.density_weight = float(density_weight)
        self.cdf_weight = float(cdf_weight)
        self.distribution_weight = float(distribution_weight)

    def forward(self, meta_pred: torch.Tensor, meta_target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = _as_float(meta_pred[..., :9])
        tgt = _as_float(meta_target[..., :9]).detach()

        loss_center = F.smooth_l1_loss(pred[..., _CENTER], tgt[..., _CENTER], beta=self.beta)
        loss_length = F.smooth_l1_loss(pred[..., _LENGTH], tgt[..., _LENGTH], beta=self.beta)
        loss_mass = F.smooth_l1_loss(pred[..., _MASS], tgt[..., _MASS], beta=self.beta)
        loss_log_density = F.smooth_l1_loss(pred[..., _LOG_DENS], tgt[..., _LOG_DENS], beta=self.beta)
        loss_cdf = F.smooth_l1_loss(pred[..., _END], tgt[..., _END], beta=self.beta)
        loss_len_dist = _js_divergence(pred[..., _LENGTH], tgt[..., _LENGTH])
        loss_mass_dist = _js_divergence(pred[..., _MASS], tgt[..., _MASS])
        loss_distribution = 0.5 * (loss_len_dist + loss_mass_dist)

        total = (
            self.center_weight * loss_center
            + self.length_weight * loss_length
            + self.mass_weight * loss_mass
            + self.density_weight * loss_log_density
            + self.cdf_weight * loss_cdf
            + self.distribution_weight * loss_distribution
        )
        return {
            "loss_layout_center": loss_center,
            "loss_layout_length": loss_length,
            "loss_layout_mass": loss_mass,
            "loss_layout_log_density": loss_log_density,
            "loss_layout_cdf": loss_cdf,
            "loss_layout_distribution": loss_distribution,
            "loss_text_layout": total,
        }


# -----------------------------------------------------------------------------
# 4) Causal + semantic + MoE mechanism loss
# -----------------------------------------------------------------------------


class CausalSemanticRouterLoss(nn.Module):
    """Aggregate MATD mechanism losses from causal learner, SCCI and MoE.

    The causal learner and MoE already expose internally weighted aggregate
    losses.  This wrapper makes them part of the main objective and adds a safe
    fallback router KL/balance term if ``moe.last_aux_losses`` is unavailable.
    """

    def __init__(
        self,
        causal_weight: float = 0.01,
        moe_weight: float = 0.01,
        scci_weight: float = 0.0,
        router_fallback_weight: float = 0.01,
        scci_entropy_weight: float = 0.05,
        scci_diversity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.causal_weight = float(causal_weight)
        self.moe_weight = float(moe_weight)
        self.scci_weight = float(scci_weight)
        self.router_fallback_weight = float(router_fallback_weight)
        self.scci_entropy_weight = float(scci_entropy_weight)
        self.scci_diversity_weight = float(scci_diversity_weight)

    @staticmethod
    def _router_fallback(
        router_p: Optional[torch.Tensor],
        prior_p: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if router_p is None:
            return {}
        p = _as_float(router_p).clamp_min(1e-8)
        E = p.shape[-1]
        usage = p.reshape(-1, E).mean(dim=0)
        ideal = torch.full_like(usage, 1.0 / E)
        balance = (usage - ideal).pow(2).sum()
        entropy = -(p * _safe_log(p)).sum(dim=-1).mean()
        if prior_p is not None:
            q = _as_float(prior_p).clamp_min(1e-8)
            kl = (p * (_safe_log(p) - _safe_log(q))).sum(dim=-1).mean()
        else:
            kl = p.new_tensor(0.0)
        return {
            "loss_router_fallback_balance": balance,
            "loss_router_fallback_kl": kl,
            "loss_router_entropy": entropy.detach(),
            "loss_router_fallback": balance + kl,
        }

    def forward(
        self,
        causal_losses: Optional[Mapping[str, torch.Tensor]] = None,
        moe_aux_losses: Optional[Mapping[str, torch.Tensor]] = None,
        scci_aux_losses: Optional[Mapping[str, torch.Tensor]] = None,
        *,
        router_p: Optional[torch.Tensor] = None,
        prior_p: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        ref = router_p if router_p is not None else prior_p
        device = ref.device if torch.is_tensor(ref) else None
        total = _zero_like_reference(ref, device=device)
        out: dict[str, torch.Tensor] = {}

        if causal_losses:
            for k, v in causal_losses.items():
                if torch.is_tensor(v):
                    out[f"causal_{k}"] = v
            causal_main = causal_losses.get("loss_causal")
            if torch.is_tensor(causal_main):
                total = total + self.causal_weight * causal_main
                out["loss_mechanism_causal_part"] = self.causal_weight * causal_main

        moe_main = None
        if moe_aux_losses:
            for k, v in moe_aux_losses.items():
                if torch.is_tensor(v):
                    out[k] = v
            moe_main = moe_aux_losses.get("loss_moe")
        if not torch.is_tensor(moe_main):
            fallback = self._router_fallback(router_p, prior_p)
            out.update(fallback)
            raw_fallback = fallback.get("loss_router_fallback")
            if torch.is_tensor(raw_fallback):
                moe_main = self.router_fallback_weight * raw_fallback
        if torch.is_tensor(moe_main):
            total = total + self.moe_weight * moe_main
            out["loss_mechanism_moe_part"] = self.moe_weight * moe_main

        if scci_aux_losses:
            for k, v in scci_aux_losses.items():
                if torch.is_tensor(v):
                    out[k] = v
            ent = scci_aux_losses.get("loss_scci_attn_entropy")
            div = scci_aux_losses.get("loss_scci_slot_diversity")
            scci_main = _zero_like_reference(ref, device=device)
            if torch.is_tensor(ent):
                scci_main = scci_main + self.scci_entropy_weight * ent
            if torch.is_tensor(div):
                scci_main = scci_main + self.scci_diversity_weight * div
            total = total + self.scci_weight * scci_main
            out["loss_mechanism_scci_part"] = self.scci_weight * scci_main

        out["loss_mechanism"] = total
        return out


# -----------------------------------------------------------------------------
# Loss aggregation
# -----------------------------------------------------------------------------


@dataclass
class LossWeights:
    diffusion_bridge: float = 1.0
    patch_field: float = 0.2
    text_layout: float = 0.5
    mechanism: float = 1.0


def weights_from_config(config: Mapping[str, Any]) -> LossWeights:
    lw = config.get("loss_weights") if isinstance(config, Mapping) else None
    if isinstance(lw, Mapping):
        base = LossWeights()
        return LossWeights(**{k: float(lw.get(k, getattr(base, k))) for k in LossWeights.__dataclass_fields__})
    return LossWeights(
        diffusion_bridge=float(config.get("lambda_diffusion", 1.0)),
        patch_field=float(config.get("lambda_x0", config.get("lambda_recon", 0.2))),
        text_layout=float(config.get("lambda_plan", 0.5)),
        mechanism=1.0,
    )


def compute_total_loss(
    loss_dicts: Mapping[str, Mapping[str, torch.Tensor]],
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine the four MATD loss groups into one scalar."""
    device: Optional[torch.device] = None
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
        ("diffusion_bridge", weights.diffusion_bridge, "loss_diffusion_bridge"),
        ("patch_field", weights.patch_field, "loss_patch_field"),
        ("text_layout", weights.text_layout, "loss_text_layout"),
        ("mechanism", weights.mechanism, "loss_mechanism"),
    ]
    for group_name, weight, agg_key in group_spec:
        sub = loss_dicts.get(group_name)
        if sub is None:
            continue
        for k, v in sub.items():
            all_terms[k] = v
        if float(weight) != 0.0 and agg_key in sub:
            total = total + float(weight) * sub[agg_key]

    all_terms["loss_total"] = total
    return total, all_terms
