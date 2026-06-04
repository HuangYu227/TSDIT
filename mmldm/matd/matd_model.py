"""High-standard end-to-end MATD model.

This file assembles tokenizer, planner, SCCI, causal learner, MoE, causal-aware
DiT denoiser and neural-field decoder into one coherent training/generation
module.  It keeps the public entry points expected by the project:

    model.forward_train(x0, texts)
    model.generate(texts, target_length, ...)
    model.submodules
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal import DynamicCausalMechanismLearner
from .decoder import VariablePatchDecoder
from .dit import T2PDenoiser
from .losses import (
    ConsistencyLoss,
    DiffusionLoss,
    LossWeights,
    ReconstructionLoss,
    compute_total_loss,
)
from .losses import weights_from_config
from .moe import SemanticCausalTemporalMoE
from .planner import PlannerLoss, TextToPatchPlanner
from .scci import SemanticCausalConditionInjector, TextSemanticSlotExtractor
from .text_encoder import MATDTextEncoder, NullTextEncoder


def _import_tokenizer():
    try:
        from .tokenizer import DensityAwareAdaptivePatch
        return DensityAwareAdaptivePatch
    except ImportError as exc:
        raise ImportError("Expected .tokenizer.DensityAwareAdaptivePatch") from exc


@dataclass
class MATDConfig:
    # Tokenizer
    embed_dim: int = 256
    target_tokens: int | None = None
    ref_len: int = 16
    min_len: int = 4
    max_len: int = 64
    rfft_win: int = 32
    tau: float = 0.5
    base_score: float = 0.05
    target_seg_len: int = 8
    min_tokens: int = 8
    max_tokens: int = 128
    encoder_context_depth: int = 2
    encoder_num_heads: int = 8
    encoder_spectral_bins: int = 8
    encoder_meta_fourier_bands: int = 4

    # Text
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim: int = 384
    text_frozen: bool = True
    model_dim: int = 256
    text_max_length: int = 128

    # Planner / SCCI
    n_slots: int = 6
    slot_iters: int = 2
    planner_heads: int = 8
    scci_heads: int = 4
    scci_dropout: float = 0.0

    # MoE
    n_experts: int = 6
    top_k: int = 2
    moe_hidden_mult: int = 4
    moe_router_hidden_mult: int = 2
    moe_dropout: float = 0.0
    moe_noisy_gating: bool = True
    moe_capacity_factor_train: float = 1.25
    moe_capacity_factor_eval: float = 2.0
    moe_prior_scale: float = 1.0
    moe_balance_weight: float = 0.01
    moe_z_loss_weight: float = 0.001
    moe_prior_kl_weight: float = 0.05
    moe_router_smooth_weight: float = 0.01
    moe_capacity_weight: float = 0.1
    moe_residual_scale: float = 0.1

    # Causal
    n_mech: int = 6
    n_segments: int = 8
    max_lag: int = 2
    causal_predict_weight: float = 1.0
    causal_dag_weight: float = 0.1
    causal_sparsity_weight: float = 0.01
    causal_smooth_weight: float = 0.01
    causal_disentangle_weight: float = 0.01
    causal_return_segment_lag_graph: bool = False

    # DiT
    dit_depth: int = 8
    dit_heads: int = 8
    dit_dim: int = 256
    mlp_ratio: float = 4.0
    pred_mode: str = "eps"  # eps or v
    dit_dropout: float = 0.0
    dit_qk_norm: bool = False
    min_snr_gamma: float | None = 5.0
    eval_interval: int = 10
    log_interval: int = 50

    # Decoder
    decoder_hidden: int = 256
    decoder_context_depth: int = 1
    decoder_context_heads: int = 8
    decoder_local_bands: int = 8
    decoder_global_bands: int = 6
    decoder_chunk_size: int = 16
    decoder_field_blocks: int = 3
    decoder_siren_omega: float = 18.0
    decoder_siren_scale: float = 0.1
    decoder_output_activation: str = "none"

    # Diffusion
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    ddim_steps: int = 50

    # Loss weights (simplified to 3)
    lambda_diffusion: float = 1.0
    lambda_x0: float = 0.2
    lambda_consistency: float = 0.5

    # Optimizer / CFG
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100_000
    batch_size: int = 32
    grad_clip: float = 1.0
    p_drop_text: float = 0.1
    cfg_scale: float = 5.0
    align_temperature: float = 0.07
    planner_beta: float = 1.0
    use_oracle_meta_prob: float = 0.5
    use_causal_guidance_in_sampling: bool = False


def _make_alpha_bar(num_steps: int, schedule: str = "cosine", beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    if schedule == "cosine":
        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        x = (steps / num_steps + s) / (1.0 + s)
        ab = torch.cos(x * math.pi / 2) ** 2
        ab = ab / ab[0]
        return ab[1:]
    if schedule == "linear":
        beta = torch.linspace(beta_start, beta_end, num_steps)
        return torch.cumprod(1.0 - beta, dim=0)
    if schedule == "quad":
        beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        return torch.cumprod(1.0 - beta, dim=0)
    raise ValueError(f"Unknown beta schedule: {schedule}")


class MATDModel(nn.Module):
    def __init__(self, cfg: MATDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim
        text_work_dim = cfg.model_dim

        # Validate all embed_dim / num_heads divisibility upfront so the
        # user gets a clear error instead of a cryptic shape-mismatch deep
        # inside a submodule constructor.
        if D % cfg.encoder_num_heads != 0:
            raise ValueError(
                f"embed_dim={D} must be divisible by encoder_num_heads={cfg.encoder_num_heads}"
            )
        if D % cfg.planner_heads != 0:
            raise ValueError(
                f"embed_dim={D} must be divisible by planner_heads={cfg.planner_heads}"
            )
        if D % cfg.scci_heads != 0:
            raise ValueError(
                f"embed_dim={D} must be divisible by scci_heads={cfg.scci_heads}"
            )
        if cfg.dit_dim % cfg.dit_heads != 0:
            raise ValueError(
                f"dit_dim={cfg.dit_dim} must be divisible by dit_heads={cfg.dit_heads}"
            )
        if cfg.decoder_hidden % cfg.decoder_context_heads != 0:
            raise ValueError(
                f"decoder_hidden={cfg.decoder_hidden} must be divisible by "
                f"decoder_context_heads={cfg.decoder_context_heads}"
            )
        # Causal module uses n_heads=4 by default (hardcoded, not in config).
        _causal_default_heads = 4
        if D % _causal_default_heads != 0:
            raise ValueError(
                f"embed_dim={D} must be divisible by the causal module's "
                f"default n_heads={_causal_default_heads}"
            )

        self.text_encoder = MATDTextEncoder(cfg.text_model, model_dim=text_work_dim, freeze=cfg.text_frozen, max_length=cfg.text_max_length)
        self.null_encoder = NullTextEncoder(dim=text_work_dim)

        Tokenizer = _import_tokenizer()
        tok_kwargs = dict(embed_dim=D, target_tokens=cfg.target_tokens, ref_len=cfg.ref_len, min_len=cfg.min_len, max_len=cfg.max_len, rfft_win=cfg.rfft_win, tau=cfg.tau, base_score=cfg.base_score, target_seg_len=cfg.target_seg_len, min_tokens=cfg.min_tokens, max_tokens=cfg.max_tokens)
        # New tokenizer accepts extra kwargs; old tokenizer will not.  Keep robust.
        try:
            self.encoder = Tokenizer(**tok_kwargs, context_depth=cfg.encoder_context_depth, num_heads=cfg.encoder_num_heads, spectral_bins=cfg.encoder_spectral_bins, meta_fourier_bands=cfg.encoder_meta_fourier_bands)
        except TypeError:
            self.encoder = Tokenizer(**tok_kwargs)

        self.planner = TextToPatchPlanner(text_dim=text_work_dim, hidden_dim=D, n_heads=cfg.planner_heads, n_patches=cfg.max_tokens)
        self.slot_extractor = TextSemanticSlotExtractor(text_dim=text_work_dim, dim=D, n_slots=cfg.n_slots, n_heads=cfg.scci_heads, n_iters=cfg.slot_iters, dropout=cfg.scci_dropout)
        self.injector = SemanticCausalConditionInjector(dim=D, meta_dim=9, n_heads=cfg.scci_heads, dropout=cfg.scci_dropout)
        self.causal = DynamicCausalMechanismLearner(dim=D, n_mech=cfg.n_mech, n_segments=cfg.n_segments, max_lag=cfg.max_lag, predict_weight=cfg.causal_predict_weight, dag_weight=cfg.causal_dag_weight, sparsity_weight=cfg.causal_sparsity_weight, smooth_weight=cfg.causal_smooth_weight, disentangle_weight=cfg.causal_disentangle_weight, return_segment_lag_graph=cfg.causal_return_segment_lag_graph)
        self.moe = SemanticCausalTemporalMoE(dim=D, meta_dim=9, n_experts=cfg.n_experts, top_k=cfg.top_k, hidden_mult=cfg.moe_hidden_mult, router_hidden_mult=cfg.moe_router_hidden_mult, dropout=cfg.moe_dropout, noisy_gating=cfg.moe_noisy_gating, capacity_factor_train=cfg.moe_capacity_factor_train, capacity_factor_eval=cfg.moe_capacity_factor_eval, prior_scale=cfg.moe_prior_scale, balance_weight=cfg.moe_balance_weight, z_loss_weight=cfg.moe_z_loss_weight, prior_kl_weight=cfg.moe_prior_kl_weight, smooth_weight=cfg.moe_router_smooth_weight, capacity_weight=cfg.moe_capacity_weight, residual_scale=cfg.moe_residual_scale)
        self.denoiser = T2PDenoiser(input_dim=D, output_dim=D, text_dim=text_work_dim, hidden_dim=cfg.dit_dim, n_heads=cfg.dit_heads, n_layers=cfg.dit_depth, mlp_expand=int(cfg.mlp_ratio), dropout=cfg.dit_dropout, qk_norm=cfg.dit_qk_norm, prediction_type="epsilon" if cfg.pred_mode == "eps" else cfg.pred_mode)
        self.decoder = VariablePatchDecoder(
            latent_dim=D,
            meta_dim=9,
            hidden_dim=cfg.decoder_hidden,
            local_bands=getattr(cfg, "decoder_local_bands", 8),
            global_bands=getattr(cfg, "decoder_global_bands", 6),
            context_depth=getattr(cfg, "decoder_context_depth", 1),
            context_heads=getattr(cfg, "decoder_context_heads", 8),
            chunk_size=getattr(cfg, "decoder_chunk_size", 16),
            field_blocks=getattr(cfg, "decoder_field_blocks", 3),
            siren_omega=getattr(cfg, "decoder_siren_omega", 18.0),
            siren_scale=getattr(cfg, "decoder_siren_scale", 0.1),
            output_activation=getattr(cfg, "decoder_output_activation", "none"),
        )

        self.register_buffer("alpha_bar", _make_alpha_bar(cfg.timesteps, cfg.beta_schedule), persistent=False)
        self.loss_diffusion = DiffusionLoss(min_snr_gamma=cfg.min_snr_gamma, prediction_type="epsilon" if cfg.pred_mode == "eps" else cfg.pred_mode)
        self.loss_recon = ReconstructionLoss(lambda_mse=0.5)
        self.loss_consistency = ConsistencyLoss()

    @property
    def submodules(self) -> dict[str, nn.Module]:
        return {
            "text_encoder": self.text_encoder,
            "null_encoder": self.null_encoder,
            "planner": self.planner,
            "encoder": self.encoder,
            "slot_extractor": self.slot_extractor,
            "injector": self.injector,
            "causal": self.causal,
            "moe": self.moe,
            "denoiser": self.denoiser,
            "decoder": self.decoder,
        }

    def _extract(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.alpha_bar.to(device=x.device, dtype=x.dtype).gather(0, t)
        return out.view(x.shape[0], *([1] * (x.dim() - 1)))

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self._extract(t, x0)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def _predict_x0_from_eps(self, z_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ab = self._extract(t, z_t)
        return (z_t - (1.0 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-8)

    def _predict_x0_from_v(self, z_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ab = self._extract(t, z_t)
        return ab.sqrt() * z_t - (1.0 - ab).sqrt() * v

    def _encode_text_cfg(self, texts: list[str], training: bool = True):
        token_hidden, pooled, attention_mask = self.text_encoder(texts)
        null_tokens, null_pooled = self.null_encoder(len(texts))
        drop_mask = None
        if training and self.cfg.p_drop_text > 0:
            drop_mask = torch.rand(len(texts), device=pooled.device) < self.cfg.p_drop_text
        return token_hidden, pooled, null_tokens, null_pooled, attention_mask, drop_mask

    def _denoise_train_with_cfg_dropout(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        token_hidden: torch.Tensor,
        pooled: torch.Tensor,
        attention_mask: torch.Tensor,
        null_tokens: torch.Tensor,
        null_pooled: torch.Tensor,
        meta_9: torch.Tensor,
        causal_feat: Optional[torch.Tensor],
        drop_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Denoiser forward with per-sample CFG dropout.

        Planner, alignment, slots, and causal losses must always see the real
        text condition.  CFG dropout is only for the denoiser branch, where a
        dropped sample should match the inference-time unconditional branch:
        null text tokens, null pooled text, and no causal feature.
        """
        text_key_padding_mask = attention_mask == 0
        if drop_mask is None or not bool(drop_mask.any()):
            return self.denoiser(
                z_t, t, token_hidden, meta_9, pooled,
                causal_feat=causal_feat,
                text_padding_mask=text_key_padding_mask,
            )

        eps_pred = torch.empty_like(z_t)
        keep_mask = ~drop_mask
        if bool(keep_mask.any()):
            keep_eps = self.denoiser(
                z_t[keep_mask],
                t[keep_mask],
                token_hidden[keep_mask],
                meta_9[keep_mask],
                pooled[keep_mask],
                causal_feat=causal_feat[keep_mask] if causal_feat is not None else None,
                text_padding_mask=text_key_padding_mask[keep_mask],
            )
            eps_pred[keep_mask] = keep_eps.to(dtype=eps_pred.dtype)

        if bool(drop_mask.any()):
            null_padding_mask = torch.zeros(
                (int(drop_mask.sum().item()), null_tokens.shape[1]),
                device=z_t.device,
                dtype=torch.bool,
            )
            drop_eps = self.denoiser(
                z_t[drop_mask],
                t[drop_mask],
                null_tokens[drop_mask],
                meta_9[drop_mask],
                null_pooled[drop_mask],
                causal_feat=None,
                text_padding_mask=null_padding_mask,
            )
            eps_pred[drop_mask] = drop_eps.to(dtype=eps_pred.dtype)
        return eps_pred

    def _encode_oracle(self, x0: torch.Tensor):
        x_in = x0.squeeze(-1) if x0.dim() == 3 and x0.shape[-1] == 1 else x0  # (B,T) or (B,T,C)
        out = self.encoder(x_in)
        if isinstance(out, tuple):
            return out
        raise RuntimeError("MATD encoder/tokenizer must return (z0, meta_oracle)")

    def _stage_weights(self, stage: int) -> LossWeights:
        """Return loss weights tuned for each training stage."""
        if stage == 0:  # joint training
            return LossWeights(
                diffusion=self.cfg.lambda_diffusion,
                reconstruction=self.cfg.lambda_x0,
                consistency=self.cfg.lambda_consistency,
            )
        if stage == 1:  # autoencoder: encoder + decoder only
            return LossWeights(
                diffusion=0.0,
                reconstruction=1.0,
                consistency=0.0,
            )
        if stage == 2:  # planner (no dedicated loss; planner trains via reconstruction gradient)
            return LossWeights(
                diffusion=0.0,
                reconstruction=1.0,
                consistency=0.0,
            )
        if stage == 4:  # joint finetune
            return LossWeights(
                diffusion=self.cfg.lambda_diffusion,
                reconstruction=self.cfg.lambda_x0,
                consistency=self.cfg.lambda_consistency,
            )
        # stage 3 (default): diffusion with soft reconstruction + consistency
        return LossWeights(
            diffusion=self.cfg.lambda_diffusion,
            reconstruction=self.cfg.lambda_x0 * 0.5,
            consistency=self.cfg.lambda_consistency * 0.5,
        )

    def forward_train(self, x0: torch.Tensor, texts: list[str], meta_override: Optional[torch.Tensor] = None, stage: int = 3, training: Optional[bool] = None) -> dict[str, Any]:
        B = x0.shape[0]
        training_mode = self.training if training is None else training
        token_hidden, pooled, null_tokens, null_pooled, attention_mask, drop_mask = self._encode_text_cfg(texts, training=training_mode)
        text_key_padding_mask = attention_mask == 0

        z0, meta_oracle = self._encode_oracle(x0)
        K = z0.shape[1]
        meta_pred = self.planner(token_hidden, text_padding_mask=text_key_padding_mask, n_patches=K)
        use_oracle = (torch.rand((), device=z0.device).item() < self.cfg.use_oracle_meta_prob)
        if meta_override is not None:
            meta = meta_override.to(z0.device)
        elif use_oracle:
            meta = meta_oracle.detach()
        else:
            meta = meta_pred
        meta_9 = meta[..., :9].to(z0.dtype)

        # Stage 1 pure autoencoder: skip causal/SCCI/MoE, use z0 directly
        if stage == 1:
            z_moe = z0
            causal_feat = None
            A0 = Alags = None
            causal_losses = {}
            slot_attn = None
            router_p = prior_p = None
        else:
            t = torch.randint(0, self.cfg.timesteps, (B,), device=z0.device, dtype=torch.long)
            t_emb = self.denoiser.timestep_embed(t)
            slots = self.slot_extractor(token_hidden, text_padding_mask=text_key_padding_mask)
            causal_feat, A0, Alags, causal_losses = self.causal(z0, pooled, meta=meta_9)
            z_cond, slot_attn, text_cond = self.injector(z0, meta_9, slots, t_emb, causal_feat)
            t_emb_k = t_emb.unsqueeze(1).expand(-1, K, -1)
            z_moe, router_p, prior_p = self.moe(z_cond, meta_9, text_cond, t_emb_k, causal_feat)

        # Stage 1: pure autoencoder, no diffusion
        if stage == 1:
            noise = None
            z_t = None
            eps_pred = None
            diff_target = None
        else:
            noise = torch.randn_like(z_moe)
            z_t = self._q_sample(z_moe, t, noise)
            eps_pred = self._denoise_train_with_cfg_dropout(
                z_t,
                t,
                token_hidden,
                pooled,
                attention_mask,
                null_tokens,
                null_pooled,
                meta_9,
                causal_feat,
                drop_mask,
            )
            if self.cfg.pred_mode == "v":
                ab = self._extract(t, z_moe)
                diff_target = ab.sqrt() * noise - (1.0 - ab).sqrt() * z_moe
            else:
                diff_target = noise

        target_len = x0.shape[1]
        x_hat = self.decoder(z_moe, meta_9, target_len)
        x0_seq = x0.unsqueeze(-1) if x0.dim() == 2 else x0  # ensure (B,T,C)

        loss_dicts: dict[str, dict[str, torch.Tensor]] = {}
        if stage != 1:
            loss_dicts["diffusion"] = self.loss_diffusion(eps_pred, diff_target, t=t, alpha_bar=self.alpha_bar)

        # Reconstruction: decoder output from clean latent vs ground truth
        ts_recon = self.loss_recon(x_hat, x0_seq)
        loss_dicts["reconstruction"] = {
            "loss_ts_recon_l1": ts_recon.get("loss_recon_l1", torch.tensor(0.0, device=x0.device)),
            "loss_ts_recon_mse": ts_recon.get("loss_recon_mse", torch.tensor(0.0, device=x0.device)),
            "loss_recon": ts_recon["loss_recon"],
        }

        # Consistency: decoder trained on noisy latent to close train/inference gap
        if stage != 1 and z_t is not None:
            loss_dicts["consistency"] = self.loss_consistency(
                z_t, z_moe, meta_9, target_len, x0_seq, self.decoder,
            )

        weights = self._stage_weights(stage)
        total_loss, all_terms = compute_total_loss(loss_dicts, weights)
        return {
            "loss": total_loss,
            "loss_total": total_loss,
            "loss_terms": all_terms,
            "loss_dicts": loss_dicts,
            "z0": z0,
            "z_moe": z_moe,
            "z_t": z_t,
            "noise": noise,
            "eps_pred": eps_pred,
            "x_hat": x_hat,
            "meta_pred": meta_pred,
            "meta_oracle": meta_oracle,
            "meta": meta,
            "token_hidden": token_hidden,
            "pooled": pooled,
        }

    @torch.no_grad()
    def generate(self, texts: list[str], target_length: int = 96, K: Optional[int] = None, ddim_steps: Optional[int] = None, cfg_scale: Optional[float] = None, eta: float = 0.0, use_causal_guidance: Optional[bool] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        B = len(texts)
        ddim_steps = self.cfg.ddim_steps if ddim_steps is None else ddim_steps
        cfg_scale = self.cfg.cfg_scale if cfg_scale is None else cfg_scale
        use_causal_guidance = self.cfg.use_causal_guidance_in_sampling if use_causal_guidance is None else use_causal_guidance

        text_tokens, pooled, attention_mask = self.text_encoder(texts)
        null_tokens, null_pooled = self.null_encoder(B)
        text_key_padding_mask = attention_mask == 0
        null_padding_mask = torch.zeros(null_tokens.shape[:2], device=device, dtype=torch.bool)
        if K is None:
            K = self.encoder._choose_k(target_length) if hasattr(self.encoder, "_choose_k") else max(self.cfg.min_tokens, math.ceil(target_length / self.cfg.target_seg_len))
        meta = self.planner(text_tokens, text_padding_mask=text_key_padding_mask, n_patches=K).to(device)
        meta_9 = meta[..., :9]
        D = self.cfg.embed_dim
        z = torch.randn(B, K, D, device=device)
        steps = torch.linspace(self.cfg.timesteps - 1, 0, ddim_steps, device=device).long()

        for i, step in enumerate(steps):
            t = torch.full((B,), int(step.item()), device=device, dtype=torch.long)
            causal_feat = None
            if use_causal_guidance:
                causal_feat = self.causal(z, pooled, meta=meta_9)[0]
            eps_cond = self.denoiser(z, t, text_tokens, meta_9, pooled, causal_feat=causal_feat, text_padding_mask=text_key_padding_mask)
            eps_uncond = self.denoiser(z, t, null_tokens, meta_9, null_pooled, causal_feat=None, text_padding_mask=null_padding_mask)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            ab_t = self.alpha_bar.to(device=device, dtype=z.dtype)[step]
            if self.cfg.pred_mode == "v":
                x0_pred = ab_t.sqrt() * z - (1.0 - ab_t).sqrt() * eps
                eps_pred = ab_t.sqrt() * eps + (1.0 - ab_t).sqrt() * z
            else:
                x0_pred = (z - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-8)
                eps_pred = eps
            if i == len(steps) - 1:
                z = x0_pred
            else:
                next_step = steps[i + 1]
                ab_next = self.alpha_bar.to(device=device, dtype=z.dtype)[next_step]
                sigma = eta * (
                    (1 - ab_next) / (1 - ab_t).clamp_min(1e-8)
                    * (1 - ab_t / ab_next)
                ).clamp_min(0).sqrt()
                noise = torch.randn_like(z) if eta > 0 else 0.0
                z = ab_next.sqrt() * x0_pred + (1.0 - ab_next - sigma ** 2).clamp_min(0).sqrt() * eps_pred + sigma * noise
        x = self.decoder(z, meta_9, target_length)
        return x
