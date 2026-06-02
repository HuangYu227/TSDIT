"""MATD Trainer -- Multi-stage training for the Multimodal Adaptive Temporal Diffusion framework.

Supports four progressive training stages:
    Stage 1: Autoencoder (tokenizer + decoder) -- reconstruction fidelity.
    Stage 2: Planner + alignment -- text-to-layout and contrastive alignment.
    Stage 3: Full diffusion -- denoising in patch-latent space.
    Stage 4: Joint finetuning -- oracle/predicted meta mixing for robustness.

Reference: MATD framework design doc.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from .losses import (
    AlignmentLoss,
    CausalLosses,
    DeltaLoss,
    DiffusionLoss,
    FFTLoss,
    LossWeights,
    MoELosses,
    ReconstructionLoss,
    compute_total_loss,
)
from .planner import PlannerLoss, TextToPatchPlanner
from .text_encoder import MATDTextEncoder, NullTextEncoder

logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average for model parameters.

    Maintains a shadow copy of all trainable parameters that is updated
    after every optimizer step.

    Args:
        model: The nn.Module whose parameters to track.
        decay: Smoothing factor in [0, 1).  Typical range: 0.999 -- 0.9999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters after each optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Replace model parameters with EMA shadow (call before eval)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original parameters (call after eval)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine schedule with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_alpha_bar(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    schedule: str = "cosine",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build the alpha_bar (cumulative product of (1-beta)) schedule."""
    if schedule == "cosine":
        steps = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        s = 0.008
        x = (steps / num_steps + s) / (1 + s)
        alpha_bar = torch.cos(x * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        return alpha_bar[1:]
    elif schedule == "linear":
        beta = torch.linspace(beta_start, beta_end, num_steps, device=device)
        return torch.cumprod(1.0 - beta, dim=0)
    elif schedule == "quad":
        beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, device=device) ** 2
        return torch.cumprod(1.0 - beta, dim=0)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class MATDTrainer:
    """Trainer for the MATD model with multi-stage training support.

    The model is expected to be a dict holding sub-modules:

        text_encoder:   MATDTextEncoder
        null_encoder:   NullTextEncoder
        planner:        TextToPatchPlanner
        slot_extractor: TextSemanticSlotExtractor
        injector:       SemanticCausalConditionInjector
        causal:         DynamicCausalMechanismLearner
        moe:            SemanticCausalTemporalMoE
        denoiser:       T2PDenoiser
        decoder:        VariablePatchDecoder | LinearPatchDecoder
        encoder:        (optional) patch encoder for x0 -> z0

    Args:
        model: Dict of named nn.Module sub-components.
        config: Configuration dict.
        device: Target device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model: dict[str, nn.Module],
        config: dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.config = config
        self.model = model
        for name in list(self.model.keys()):
            self.model[name] = self.model[name].to(self.device)

        diff_cfg = config.get("diffusion", {})
        self.num_steps = diff_cfg.get("num_steps", 1000)
        self.alpha_bar = _make_alpha_bar(
            self.num_steps,
            beta_start=diff_cfg.get("beta_start", 1e-4),
            beta_end=diff_cfg.get("beta_end", 0.02),
            schedule=diff_cfg.get("schedule", "cosine"),
            device=self.device,
        )
        self.prediction_type = diff_cfg.get("prediction_type", "epsilon")

        opt_cfg = config.get("optimizer", {})
        self._build_optimizer(opt_cfg)
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._build_losses(config)

        self.use_amp = config.get("use_amp", True)
        self.grad_scaler = GradScaler(enabled=self.use_amp)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        ema_cfg = config.get("ema", {})
        self.use_ema = ema_cfg.get("enabled", True)
        self.ema_decay = ema_cfg.get("decay", 0.9999)
        self.ema: Optional[EMA] = None
        if self.use_ema:
            self._init_ema()

        self.log_interval = config.get("log_interval", 50)
        self.cfg_dropout = config.get("cfg_dropout", 0.1)
        self.global_step = 0

        logger.info(
            "MATDTrainer initialized: %d diffusion steps, AMP=%s, EMA=%s",
            self.num_steps, self.use_amp, self.use_ema,
        )

    # ------------------------------------------------------------------
    #  Internal builders
    # ------------------------------------------------------------------

    def _build_optimizer(self, opt_cfg: dict[str, Any]) -> None:
        """Collect all trainable parameters and build AdamW."""
        params: list[nn.Parameter] = []
        for module in self.model.values():
            params.extend(p for p in module.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg.get("lr", 1e-4),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
            weight_decay=opt_cfg.get("weight_decay", 1e-6),
        )

    def _build_losses(self, config: dict[str, Any]) -> None:
        """Instantiate all loss modules."""
        self.loss_diffusion = DiffusionLoss()
        self.loss_recon = ReconstructionLoss(lambda_mse=config.get("lambda_mse", 0.5))
        self.loss_delta = DeltaLoss()
        self.loss_fft = FFTLoss()
        self.loss_align = AlignmentLoss(temperature=config.get("align_temperature", 0.07))
        self.loss_planner = PlannerLoss(beta=config.get("planner_beta", 1.0))
        self.loss_causal = CausalLosses(
            dag_weight=config.get("dag_weight", 1.0),
            sparsity_weight=config.get("sparsity_weight", 0.1),
            mechanism_weight=config.get("mechanism_weight", 1.0),
        )
        self.loss_moe = MoELosses(
            n_experts=config.get("n_experts", 6),
            lb_weight=config.get("moe_lb_weight", 0.01),
            prior_weight=config.get("moe_prior_weight", 0.01),
            entropy_weight=config.get("moe_entropy_weight", 0.01),
        )
        lw_cfg = config.get("loss_weights", {})
        self.loss_weights = LossWeights(
            diffusion=lw_cfg.get("diffusion", 1.0),
            reconstruction=lw_cfg.get("reconstruction", 0.0),
            delta=lw_cfg.get("delta", 0.0),
            fft=lw_cfg.get("fft", 0.0),
            density_weighted=lw_cfg.get("density_weighted", 0.0),
            alignment=lw_cfg.get("alignment", 0.0),
            moe=lw_cfg.get("moe", 0.0),
            causal=lw_cfg.get("causal", 0.0),
            planner=lw_cfg.get("planner", 0.0),
        )

    def _init_ema(self) -> None:
        """Build EMA over all trainable parameters."""
        container = nn.Module()
        for name, module in self.model.items():
            container.add_module(name, module)
        self.ema = EMA(container, decay=self.ema_decay)

    # ------------------------------------------------------------------
    #  Diffusion helpers
    # ------------------------------------------------------------------

    def _extract(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gather alpha_bar[t] and reshape for broadcasting."""
        out = self.alpha_bar.gather(0, t).to(device=x.device, dtype=x.dtype)
        return out.view(x.shape[0], *([1] * (x.dim() - 1)))

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(z_t | z_0)."""
        ab = self._extract(t, x0)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def _predict_x0_from_eps(self, z_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Recover x_0 from predicted epsilon."""
        ab = self._extract(t, z_t)
        return (z_t - (1.0 - ab).sqrt() * eps_pred) / ab.sqrt().clamp(min=1e-8)

    def _predict_x0_from_v(self, z_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Recover x_0 from predicted velocity."""
        ab = self._extract(t, z_t)
        return ab.sqrt() * z_t - (1.0 - ab).sqrt() * v_pred

    # ------------------------------------------------------------------
    #  Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze(self, *names: str) -> None:
        """Freeze parameters of named sub-modules."""
        for name in names:
            if name in self.model:
                for p in self.model[name].parameters():
                    p.requires_grad = False

    def _unfreeze(self, *names: str) -> None:
        """Unfreeze parameters of named sub-modules."""
        for name in names:
            if name in self.model:
                for p in self.model[name].parameters():
                    p.requires_grad = True

    def _rebuild_optimizer(self, opt_cfg: Optional[dict[str, Any]] = None) -> None:
        """Rebuild optimizer to pick up newly unfrozen parameters."""
        if opt_cfg is None:
            opt_cfg = self.config.get("optimizer", {})
        self._build_optimizer(opt_cfg)

    def _build_scheduler(self, total_steps: int) -> None:
        """Build cosine warmup scheduler for the current stage."""
        sched_cfg = self.config.get("scheduler", {})
        self.scheduler = _cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=sched_cfg.get("warmup_steps", 500),
            total_steps=total_steps,
            min_lr_ratio=sched_cfg.get("min_lr_ratio", 0.01),
        )

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------

    def _encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text through the text encoder.

        Returns:
            token_hidden:  (B, M, D)
            pooled:        (B, D)
            attention_mask: (B, M) HuggingFace-style (1=real, 0=pad)
        """
        return self.model["text_encoder"](texts)

    def _get_null_cond(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get null (unconditional) embeddings for CFG."""
        return self.model["null_encoder"](batch_size)

    def _encode_text_cfg(
        self, texts: list[str], training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text with optional CFG dropout.

        Returns:
            token_hidden, pooled, null_tokens, null_pooled, attention_mask
        """
        B = len(texts)
        token_hidden, pooled, attention_mask = self._encode_text(texts)
        null_tokens, null_pooled = self._get_null_cond(B)
        if training and self.cfg_dropout > 0:
            mask = torch.rand(B, device=self.device) < self.cfg_dropout
            if mask.any():
                token_hidden = torch.where(mask[:, None, None], null_tokens, token_hidden)
                pooled = torch.where(mask[:, None], null_pooled, pooled)
        return token_hidden, pooled, null_tokens, null_pooled, attention_mask

    def _forward_model(
        self,
        x0: torch.Tensor,
        texts: list[str],
        training: bool = True,
        meta_override: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through the MATD pipeline.

        1. Encode text (with CFG dropout if training).
        2. Planner predicts patch metadata from text (or use override).
        3. Encode x0 into patch latents via the encoder.
        4. SCCI conditioning + causal + MoE refinement.
        5. Diffusion: sample t, add noise, predict with denoiser.
        6. Decode back for reconstruction losses.
        """
        B = x0.shape[0]

        # 1. Text encoding
        token_hidden, pooled, null_tokens, null_pooled, attention_mask = self._encode_text_cfg(texts, training)

        # 2. Encode x0 into patch latents (must come before planner for K)
        encoder = self.model.get("encoder")
        if encoder is not None:
            enc_out = encoder(x0)
            if isinstance(enc_out, tuple):
                z0, meta_oracle = enc_out
            else:
                z0, meta_oracle = enc_out, None
        else:
            z0, meta_oracle = x0, None
        K, D = z0.shape[1], z0.shape[2]

        # 3. Planner (with padding mask, aligned to encoder's K)
        text_key_padding_mask = (attention_mask == 0)  # (B, M) bool
        meta_pred = self.model["planner"](
            token_hidden, text_padding_mask=text_key_padding_mask, n_patches=K,
        )
        if meta_override is not None:
            meta = meta_override
        elif meta_oracle is not None:
            meta = meta_oracle.detach()
        else:
            meta = meta_pred

        # 4. SCCI conditioning
        denoiser = self.model["denoiser"]
        slots = self.model["slot_extractor"](token_hidden)

        t_dummy = torch.zeros(B, dtype=torch.long, device=self.device)
        t_emb = denoiser._timestep_embed(t_dummy)

        meta_9 = meta[:, :, :9].to(z0.dtype)
        causal_out = self.model["causal"](z0, pooled, meta=meta_9)
        causal_feat = causal_out[0]
        causal_losses = causal_out[3]

        z_cond, attn_w, u = self.model["injector"](
            z0, meta_9, slots, t_emb, causal_feat
        )

        text_cond = u
        t_emb_expand = t_emb.unsqueeze(1).expand(-1, K, -1)
        meta_embed = meta[:, :, :9].mean(dim=1)
        z_moe, router_p, prior_p = self.model["moe"](
            z_cond, meta_embed, text_cond, t_emb_expand, causal_feat
        )

        # 5. Diffusion
        t = torch.randint(0, self.num_steps, (B,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(z_moe)
        z_t = self._q_sample(z_moe, t, noise)

        # 6. Denoiser prediction
        eps_pred = denoiser(z_t, t, token_hidden, meta.to(z_t.dtype), pooled)

        # 7. Decode for reconstruction
        target_len = x0.shape[1] if x0.dim() >= 2 else K * D
        x_hat = self.model["decoder"](z_moe, meta.to(z_moe.dtype), target_len)

        return {
            "z0": z0, "z_t": z_t, "z_moe": z_moe, "noise": noise,
            "eps_pred": eps_pred, "t": t, "x_hat": x_hat,
            "meta_pred": meta_pred, "meta": meta, "meta_oracle": meta_oracle,
            "token_hidden": token_hidden, "pooled": pooled,
            "null_tokens": null_tokens, "null_pooled": null_pooled,
            "router_p": router_p, "prior_p": prior_p,
            "causal_losses": causal_losses, "slots": slots, "attn_w": attn_w,
        }

    # ------------------------------------------------------------------
    #  Loss computation
    # ------------------------------------------------------------------

    def _compute_stage_losses(
        self,
        fwd: dict[str, torch.Tensor],
        x0: torch.Tensor,
        stage: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute losses appropriate for the given training stage."""
        loss_dicts: dict[str, dict[str, torch.Tensor]] = {}

        if stage == 1:
            x_hat = fwd["x_hat"].squeeze(-1)
            x0_seq = x0.squeeze(-1) if x0.dim() == 3 else x0
            T = min(x_hat.shape[1], x0_seq.shape[1])
            loss_dicts["reconstruction"] = self.loss_recon(x_hat[:, :T], x0_seq[:, :T])
            loss_dicts["delta"] = self.loss_delta(x_hat[:, :T], x0_seq[:, :T])
            loss_dicts["fft"] = self.loss_fft(x_hat[:, :T], x0_seq[:, :T])

        elif stage == 2:
            planner_target = fwd.get("meta_oracle")
            if planner_target is not None:
                loss_dicts["planner"] = self.loss_planner(
                    fwd["meta_pred"], planner_target.detach().to(fwd["meta_pred"].dtype),
                )
            ts_embed = fwd["z0"].mean(dim=1)  # (B, D)
            loss_dicts["alignment"] = self.loss_align(ts_embed, fwd["pooled"])

        elif stage in (3, 4):
            eps_pred, noise = fwd["eps_pred"], fwd["noise"]
            loss_dicts["diffusion"] = self.loss_diffusion(eps_pred, noise)

            if self.prediction_type == "epsilon":
                x0_hat = self._predict_x0_from_eps(fwd["z_t"], eps_pred, fwd["t"])
            else:
                x0_hat = self._predict_x0_from_v(fwd["z_t"], eps_pred, fwd["t"])

            loss_dicts["reconstruction"] = self.loss_recon(x0_hat, fwd["z0"])
            planner_target = fwd.get("meta_oracle")
            if planner_target is not None:
                loss_dicts["planner"] = self.loss_planner(
                    fwd["meta_pred"], planner_target.detach().to(fwd["meta_pred"].dtype),
                )
            ts_embed = fwd["z0"].mean(dim=1)  # (B, D)
            loss_dicts["alignment"] = self.loss_align(ts_embed, fwd["pooled"])

            if fwd.get("router_p") is not None:
                loss_dicts["moe"] = self.loss_moe.forward(fwd["router_p"])

            if fwd.get("causal_losses"):
                cl = fwd["causal_losses"]
                loss_dicts["causal"] = {
                    "loss_mechanism": cl["predict"],
                    "loss_dag": cl["notears"],
                    "loss_sparsity": cl["sparsity"],
                    "loss_smooth": cl["smooth"],
                    "loss_disentangle": cl["disentangle"],
                    "loss_causal": cl["loss_causal"],
                }

        if stage == 1:
            sw = LossWeights(reconstruction=1.0, delta=1.0, fft=1.0)
        elif stage == 2:
            sw = LossWeights(planner=1.0, alignment=1.0)
        else:
            sw = self.loss_weights

        total, all_terms = compute_total_loss(loss_dicts, sw)
        return total, all_terms

    # ------------------------------------------------------------------
    #  Single training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: tuple[torch.Tensor, list[str]],
        stage: int = 3,
        meta_override: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Execute one training step.

        Args:
            batch: (x0_tensor, text_strings).
            stage: Current training stage (1--4).
            meta_override: Optional oracle metadata.

        Returns:
            Dict of loss name -> float value for logging.
        """
        x0, texts = batch
        x0 = x0.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.use_amp):
            fwd = self._forward_model(x0, texts, training=True, meta_override=meta_override)
            total_loss, all_terms = self._compute_stage_losses(fwd, x0, stage)

        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self._all_parameters(), self.max_grad_norm)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()
        if self.use_ema and self.ema is not None:
            self.ema.update(self._ema_module())

        self.global_step += 1
        return {
            k: v.detach().item() if torch.is_tensor(v) else float(v)
            for k, v in all_terms.items()
        }

    def _all_parameters(self) -> list[nn.Parameter]:
        """Collect all trainable parameters."""
        params: list[nn.Parameter] = []
        for module in self.model.values():
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def _ema_module(self) -> nn.Module:
        """Return a flat Module for EMA updates."""
        container = nn.Module()
        for name, module in self.model.items():
            container.add_module(name, module)
        return container

    # ------------------------------------------------------------------
    #  Stage 1: Autoencoder training
    # ------------------------------------------------------------------

    def train_stage1(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> list[dict[str, float]]:
        """Stage 1: Train autoencoder only (encoder + decoder).

        Freezes: text_encoder, planner, scci, moe, causal, denoiser.
        Loss: L_rec + L_delta + L_fft.
        """
        logger.info("=== Stage 1: Autoencoder training (%d epochs) ===", epochs)
        self._freeze("text_encoder", "null_encoder", "planner", "slot_extractor", "injector", "causal", "moe", "denoiser")
        self._unfreeze("encoder", "decoder")
        self._rebuild_optimizer()
        self._build_scheduler(epochs * len(dataloader))
        return self._run_stage(dataloader, epochs, stage=1, val_dataloader=val_dataloader)

    def train_stage2(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> list[dict[str, float]]:
        """Stage 2: Train planner + alignment.

        Freezes: encoder, decoder, denoiser, moe, causal.
        Loss: L_plan + L_align.
        """
        logger.info("=== Stage 2: Planner + alignment (%d epochs) ===", epochs)
        self._freeze("encoder", "decoder", "denoiser", "moe", "causal")
        self._unfreeze("text_encoder", "null_encoder", "planner", "slot_extractor", "injector")
        self._rebuild_optimizer()
        self._build_scheduler(epochs * len(dataloader))
        return self._run_stage(dataloader, epochs, stage=2, val_dataloader=val_dataloader)

    def train_stage3(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> list[dict[str, float]]:
        """Stage 3: Full diffusion training.

        All modules trainable.  Loss: L_diff + L_x0 + L_plan + all.
        """
        logger.info("=== Stage 3: Full diffusion training (%d epochs) ===", epochs)
        self._unfreeze(*self.model.keys())
        self._rebuild_optimizer()
        self._build_scheduler(epochs * len(dataloader))
        return self._run_stage(dataloader, epochs, stage=3, val_dataloader=val_dataloader)

    def train_stage4(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        mix_ratio: float = 0.5,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> list[dict[str, float]]:
        """Stage 4: Joint finetuning with oracle/predicted meta mixing.

        With probability mix_ratio the planner's predicted metadata is
        used; otherwise the oracle (ground-truth) metadata.

        Args:
            dataloader: Yields (x0, texts) or (x0, texts, meta_oracle).
            epochs: Number of epochs.
            mix_ratio: Probability of using predicted meta (vs oracle).
            val_dataloader: Optional validation dataloader.
        """
        logger.info("=== Stage 4: Joint finetuning (mix_ratio=%.2f, %d epochs) ===", mix_ratio, epochs)
        self._unfreeze(*self.model.keys())
        self._rebuild_optimizer()
        self._build_scheduler(epochs * len(dataloader))

        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            self._set_train(*self.model.keys())
            epoch_losses: dict[str, float] = {}
            n_batches = 0

            for batch in dataloader:
                if len(batch) == 3:
                    x0, texts, meta_oracle = batch
                    meta_oracle = meta_oracle.to(self.device)
                    meta_override = None if torch.rand(1).item() < mix_ratio else meta_oracle
                else:
                    x0, texts = batch
                    meta_override = None

                loss_dict = self.train_step((x0, texts), stage=4, meta_override=meta_override)
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                n_batches += 1

                if self.global_step % self.log_interval == 0:
                    logger.info(
                        "[Stage4] step=%d  loss_diff=%.4f  loss_total=%.4f",
                        self.global_step, loss_dict.get("loss_diffusion", 0.0), loss_dict.get("loss_total", 0.0),
                    )

            avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
            avg["epoch"] = epoch
            history.append(avg)

            if val_dataloader is not None and (epoch + 1) % 5 == 0:
                val_loss = self._validate(val_dataloader, stage=4)
                logger.info("[Stage4] val_loss_total=%.4f", val_loss.get("loss_total", 0.0))

        return history

    # ------------------------------------------------------------------
    #  Generic stage runner (shared by stages 1-3)
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        stage: int,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    ) -> list[dict[str, float]]:
        """Run a training stage for the given number of epochs."""
        tag = f"[Stage{stage}]"
        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            self._set_train(*self.model.keys())
            epoch_losses: dict[str, float] = {}
            n_batches = 0

            for batch in dataloader:
                loss_dict = self.train_step(batch, stage=stage)
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                n_batches += 1

                if self.global_step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info("%s step=%d  loss_total=%.4f  lr=%.2e", tag, self.global_step, loss_dict.get("loss_total", 0.0), lr)

            avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
            avg["epoch"] = epoch
            history.append(avg)
            logger.info("%s epoch=%d  avg_loss_total=%.4f", tag, epoch, avg.get("loss_total", 0.0))

            if val_dataloader is not None and (epoch + 1) % 5 == 0:
                val_loss = self._validate(val_dataloader, stage=stage)
                logger.info("%s val_loss_total=%.4f", tag, val_loss.get("loss_total", 0.0))

        return history

    # ------------------------------------------------------------------
    #  Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, dataloader: torch.utils.data.DataLoader, stage: int) -> dict[str, float]:
        """Run validation and return average losses."""
        if self.use_ema and self.ema is not None:
            self.ema.apply(self._ema_module())
        self._set_eval(*self.model.keys())
        total_losses: dict[str, float] = {}
        n_batches = 0

        for batch in dataloader:
            x0, texts = batch[:2]
            x0 = x0.to(self.device)
            with autocast(enabled=self.use_amp):
                fwd = self._forward_model(x0, texts, training=False)
                _, all_terms = self._compute_stage_losses(fwd, x0, stage)
            for k, v in all_terms.items():
                if torch.is_tensor(v):
                    total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1

        if self.use_ema and self.ema is not None:
            self.ema.restore(self._ema_module())
        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    # ------------------------------------------------------------------
    #  Train / eval mode helpers
    # ------------------------------------------------------------------

    def _set_train(self, *names: str) -> None:
        for name in names:
            if name in self.model:
                self.model[name].train()

    def _set_eval(self, *names: str) -> None:
        for name in names:
            if name in self.model:
                self.model[name].eval()

    # ------------------------------------------------------------------
    #  Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save model, optimizer, scheduler, EMA, and scaler state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "global_step": self.global_step,
            "config": self.config,
            "model": {name: m.state_dict() for name, m in self.model.items()},
            "optimizer": self.optimizer.state_dict(),
            "grad_scaler": self.grad_scaler.state_dict(),
            "alpha_bar": self.alpha_bar,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.use_ema and self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, path)
        logger.info("Checkpoint saved to %s (step=%d)", path, self.global_step)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        for name, sd in state["model"].items():
            if name in self.model:
                self.model[name].load_state_dict(sd)
        self.optimizer.load_state_dict(state["optimizer"])
        if "grad_scaler" in state:
            self.grad_scaler.load_state_dict(state["grad_scaler"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.use_ema and self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])
        self.global_step = state.get("global_step", 0)
        logger.info("Checkpoint loaded from %s (step=%d)", path, self.global_step)

    # ------------------------------------------------------------------
    #  Convenience: full 4-stage training
    # ------------------------------------------------------------------

    def train_all_stages(
        self,
        train_loaders: dict[int, torch.utils.data.DataLoader],
        epochs_per_stage: dict[int, int],
        val_loaders: Optional[dict[int, torch.utils.data.DataLoader]] = None,
        save_dir: Optional[str] = None,
        mix_ratio: float = 0.5,
    ) -> dict[int, list[dict[str, float]]]:
        """Run all four training stages sequentially.

        Args:
            train_loaders: Mapping stage_id -> DataLoader.
            epochs_per_stage: Mapping stage_id -> number of epochs.
            val_loaders: Optional mapping stage_id -> val DataLoader.
            save_dir: Directory to save checkpoints after each stage.
            mix_ratio: Stage 4 mix ratio for oracle/predicted meta.

        Returns:
            Mapping stage_id -> list of epoch loss dicts.
        """
        all_history: dict[int, list[dict[str, float]]] = {}
        for stage_id in (1, 2, 3, 4):
            if stage_id not in train_loaders:
                logger.info("Skipping stage %d (no dataloader)", stage_id)
                continue
            loader = train_loaders[stage_id]
            n_epochs = epochs_per_stage.get(stage_id, 10)
            val_loader = val_loaders.get(stage_id) if val_loaders else None

            if stage_id == 1:
                history = self.train_stage1(loader, n_epochs, val_dataloader=val_loader)
            elif stage_id == 2:
                history = self.train_stage2(loader, n_epochs, val_dataloader=val_loader)
            elif stage_id == 3:
                history = self.train_stage3(loader, n_epochs, val_dataloader=val_loader)
            else:
                history = self.train_stage4(loader, n_epochs, mix_ratio=mix_ratio, val_dataloader=val_loader)

            all_history[stage_id] = history
            if save_dir is not None:
                ckpt_path = os.path.join(save_dir, f"stage{stage_id}.pt")
                self.save_checkpoint(ckpt_path)

        return all_history
