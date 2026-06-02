"""MATD Model -- end-to-end assembly of all MATD components.

Combines the text encoder, planner, oracle tokenizer, semantic slot
extractor, SCCI injector, MoE router, causal mechanism learner, DiT
denoiser, and patch decoder into a single nn.Module with:

- forward_train(x0, texts) for the full training forward pass.
- generate(texts, ...) for DDIM-based inference with CFG.

The module exposes a submodules property that returns a
dict[str, nn.Module] compatible with MATDTrainer and MATDGenerator.

Reference: MATD framework design doc.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal import DynamicCausalMechanismLearner
from .decoder import VariablePatchDecoder
from .dit import T2PDenoiser, _get_sinusoidal_embedding
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
from .moe import SemanticCausalTemporalMoE
from .planner import PlannerLoss, TextToPatchPlanner
from .scci import SemanticCausalConditionInjector, TextSemanticSlotExtractor
from .text_encoder import MATDTextEncoder, NullTextEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Lazy import for the oracle tokenizer
# ---------------------------------------------------------------------------


def _import_tokenizer():
    """Import DensityAwareAdaptivePatch, raising a clear error if missing."""
    try:
        from .tokenizer import DensityAwareAdaptivePatch
        return DensityAwareAdaptivePatch
    except ImportError:
        raise ImportError(
            "mmldm.matd.tokenizer not found.  Create tokenizer.py with "
            "class DensityAwareAdaptivePatch(nn.Module) that maps "
            "(B, T, 1) -> (z0: B,K,D, meta: B,K,9)."
        ) from None


# ===========================================================================
#  1. Configuration
# ===========================================================================


@dataclass
class MATDConfig:
    """Hyperparameters for the MATD framework.

    Every field has a sensible default so that MATDConfig() produces
    a working configuration out of the box.  Field groups and defaults
    mirror MATD_DEFAULT_CONFIG from mmldm.matd.config.
    """

    # -- Tokenizer (DA-ATP) --
    embed_dim: int = 256
    target_tokens: int | None = None
    ref_len: int = 16
    min_len: int = 4
    max_len: int = 64
    rfft_win: int = 32
    tau: float = 0.5
    base_score: float = 0.05
    target_seg_len: int = 8
    min_tokens: int = 6
    max_tokens: int = 128

    # -- Text encoder --
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_dim: int = 384
    text_frozen: bool = True
    model_dim: int = 256

    # -- Semantic slots --
    n_slots: int = 6

    # -- Planner --
    planner_heads: int = 8

    # -- Mixture-of-Experts --
    n_experts: int = 6
    top_k: int = 2

    # -- Causal discovery --
    n_mech: int = 6
    n_segments: int = 8
    max_lag: int = 2
    causal_predict_weight: float = 1.0
    causal_dag_weight: float = 0.1
    causal_sparsity_weight: float = 0.01
    causal_smooth_weight: float = 0.01
    causal_disentangle_weight: float = 0.01
    causal_return_segment_lag_graph: bool = False

    # -- DiT denoiser --
    dit_depth: int = 8
    dit_heads: int = 8
    dit_dim: int = 256
    mlp_ratio: float = 4.0
    pred_mode: str = "eps"  # "eps" or "v"

    # -- Decoder --
    decoder_hidden: int = 256

    # -- Diffusion schedule --
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # "cosine", "linear", or "quad"
    ddim_steps: int = 50

    # -- Loss weights --
    lambda_x0: float = 0.2
    lambda_delta: float = 0.1
    lambda_fft: float = 0.05
    lambda_plan: float = 0.5
    lambda_align: float = 0.05
    lambda_moe: float = 0.01
    lambda_causal: float = 0.01

    # -- Optimizer --
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100_000
    batch_size: int = 32
    grad_clip: float = 1.0

    # -- Classifier-Free Guidance --
    p_drop_text: float = 0.1
    cfg_scale: float = 5.0

    # -- Loss-module tuning --
    align_temperature: float = 0.07
    planner_beta: float = 1.0

    # -- Staged meta training --
    use_oracle_meta_prob: float = 1.0  # anneal 1.0 -> 0.0 over training


# ===========================================================================
#  2. Noise-schedule helpers
# ===========================================================================


def _make_alpha_bar(
    num_steps: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Build the cumulative product-of-alphas noise schedule.

    Returns a tensor of shape (num_steps,) indexed from 0 to
    num_steps - 1.  alpha_bar[0] is the least noisy step and
    alpha_bar[-1] is the most noisy.

    Identical to the helper used in MATDTrainer and MATDGenerator.
    """
    if schedule == "cosine":
        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        x = (steps / num_steps + s) / (1.0 + s)
        alpha_bar = torch.cos(x * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        return alpha_bar[1:]  # drop sentinel -> length = num_steps
    elif schedule == "linear":
        beta = torch.linspace(beta_start, beta_end, num_steps)
        return torch.cumprod(1.0 - beta, dim=0)
    elif schedule == "quad":
        beta = (
            torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        )
        return torch.cumprod(1.0 - beta, dim=0)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule!r}")


# ===========================================================================
#  3. Timestep embedding (shared by SCCI / MoE)
# ===========================================================================


class TimestepEmbedding(nn.Module):
    """Sinusoidal + MLP timestep embedding for conditioning modules.

    Produces a (B, dim) vector from integer timesteps, suitable for
    injection into SCCI and MoE.
    """

    def __init__(self, dim: int, sinusoidal_dim: int = 256) -> None:
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t: (B,) integer timesteps.  Returns: (B, dim)."""
        emb = _get_sinusoidal_embedding(t.float(), self.sinusoidal_dim)
        return self.mlp(emb)

# ===========================================================================
#  4. MATD Model
# ===========================================================================


class MATDModel(nn.Module):
    """Multimodal Adaptive Temporal Diffusion model.

    Assembles every MATD sub-module into a single nn.Module that
    supports both training (forward_train / forward) and inference
    (generate).

    Sub-modules are stored as named attributes so that the trainer can
    extract them via the submodules property.

    Args:
        cfg: MATDConfig with all hyperparameters.

    Example::

        cfg = MATDConfig(dit_depth=12, lr=5e-5)
        model = MATDModel(cfg).to("cuda")

        # Training
        out = model(x0_tensor, ["rising trend", "flat line"])
        loss = out["loss_total"]
        loss.backward()

        # Inference
        x_hat = model.generate(["sharp spike"], target_length=64)
    """

    def __init__(self, cfg: MATDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim
        M = cfg.model_dim

        # -- Text encoder (pretrained, frozen by default) --
        self.text_encoder = MATDTextEncoder(
            model_name=cfg.text_model,
            model_dim=M,
            freeze=cfg.text_frozen,
        )

        # -- Null text encoder for CFG --
        self.null_encoder = NullTextEncoder(dim=M, n_tokens=16)

        # -- Planner --
        self.planner = TextToPatchPlanner(
            text_dim=M,
            hidden_dim=D,
            n_heads=cfg.planner_heads,
            n_patches=cfg.max_tokens,  # max possible K, slices to actual K in forward
        )

        # -- Oracle tokenizer (training only) --
        DensityAwareAdaptivePatch = _import_tokenizer()
        self.encoder = DensityAwareAdaptivePatch(
            embed_dim=D,
            target_tokens=cfg.target_tokens,
            ref_len=cfg.ref_len,
            min_len=cfg.min_len,
            max_len=cfg.max_len,
            rfft_win=cfg.rfft_win,
            tau=cfg.tau,
            base_score=cfg.base_score,
            target_seg_len=cfg.target_seg_len,
            min_tokens=cfg.min_tokens,
            max_tokens=cfg.max_tokens,
        )

        # -- Semantic slot extractor --
        self.slot_extractor = TextSemanticSlotExtractor(
            text_dim=M,
            dim=D,
            n_slots=cfg.n_slots,
        )

        # -- SCCI injector --
        self.injector = SemanticCausalConditionInjector(
            dim=D,
            meta_dim=9,
            n_heads=max(1, cfg.planner_heads // 2),
        )

        # -- Causal mechanism learner --
        self.causal = DynamicCausalMechanismLearner(
            dim=D,
            n_mech=cfg.n_mech,
            n_segments=cfg.n_segments,
            max_lag=cfg.max_lag,
            n_heads=max(1, cfg.planner_heads // 2),
            predict_weight=cfg.causal_predict_weight,
            dag_weight=cfg.causal_dag_weight,
            sparsity_weight=cfg.causal_sparsity_weight,
            smooth_weight=cfg.causal_smooth_weight,
            disentangle_weight=cfg.causal_disentangle_weight,
            return_segment_lag_graph=cfg.causal_return_segment_lag_graph,
        )

        # -- Mixture-of-Experts --
        self.moe = SemanticCausalTemporalMoE(
            dim=D,
            meta_dim=9,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
        )

        # -- Timestep embedding for SCCI / MoE --
        self.time_embed = TimestepEmbedding(dim=D)

        # -- DiT denoiser --
        self.denoiser = T2PDenoiser(
            input_dim=D,
            output_dim=D,
            text_dim=M,
            hidden_dim=cfg.dit_dim,
            n_heads=cfg.dit_heads,
            n_layers=cfg.dit_depth,
            mlp_expand=int(cfg.mlp_ratio),
            prediction_type="epsilon" if cfg.pred_mode == "eps" else "v",
        )

        # -- Decoder --
        self.decoder = VariablePatchDecoder(
            latent_dim=D,
            meta_dim=9,
            hidden_dim=cfg.decoder_hidden,
        )

        # -- Noise schedule (buffer -> moves with .to(device)) --
        alpha_bar = _make_alpha_bar(cfg.timesteps, schedule=cfg.beta_schedule)
        self.register_buffer("alpha_bar", alpha_bar)

        # -- Loss modules --
        self.loss_diffusion = DiffusionLoss()
        self.loss_recon = ReconstructionLoss()
        self.loss_delta = DeltaLoss()
        self.loss_fft = FFTLoss()
        self.loss_align = AlignmentLoss(temperature=cfg.align_temperature)
        self.loss_planner = PlannerLoss(beta=cfg.planner_beta)
        self.loss_moe = MoELosses(n_experts=cfg.n_experts)
        self.loss_causal = CausalLosses()

    # ------------------------------------------------------------------
    #  Sub-module dict (MATDTrainer / MATDGenerator compatibility)
    # ------------------------------------------------------------------

    @property
    def submodules(self) -> dict[str, nn.Module]:
        """Return a dict of named sub-modules for the trainer / generator.

        Keys match what MATDTrainer and MATDGenerator expect:
        text_encoder, null_encoder, planner, encoder, slot_extractor,
        injector, causal, moe, denoiser, decoder.
        """
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

    # ------------------------------------------------------------------
    #  Diffusion helpers
    # ------------------------------------------------------------------

    def _extract(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gather alpha_bar[t] and reshape for broadcasting with x."""
        out = self.alpha_bar.gather(0, t).to(device=x.device, dtype=x.dtype)
        return out.view(x.shape[0], *([1] * (x.dim() - 1)))

    def _q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: q(z_t | z_0)."""
        ab = self._extract(t, x0)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def _predict_x0_from_eps(
        self, z_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from predicted epsilon."""
        ab = self._extract(t, z_t)
        return (z_t - (1.0 - ab).sqrt() * eps_pred) / ab.sqrt().clamp(min=1e-8)

    def _predict_x0_from_v(
        self, z_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from predicted velocity."""
        ab = self._extract(t, z_t)
        return ab.sqrt() * z_t - (1.0 - ab).sqrt() * v_pred

    # ------------------------------------------------------------------
    #  Text encoding helpers
    # ------------------------------------------------------------------

    def _encode_text(
        self, texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text through the text encoder.

        Returns:
            token_hidden:  (B, M, model_dim)
            pooled:        (B, model_dim)
            attention_mask: (B, M) HuggingFace-style (1=real, 0=pad)
        """
        return self.text_encoder(texts)

    def _encode_text_cfg(
        self, texts: list[str], training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text with optional CFG dropout.

        During training, with probability p_drop_text the real text
        tokens are replaced with learned null embeddings.

        Returns:
            token_hidden, pooled, null_tokens, null_pooled, attention_mask
        """
        B = len(texts)
        device = next(self.parameters()).device
        token_hidden, pooled, attention_mask = self._encode_text(texts)
        null_tokens, null_pooled = self.null_encoder(B)

        # Align null_tokens sequence length to token_hidden
        M_real = token_hidden.shape[1]
        M_null = null_tokens.shape[1]
        if M_null < M_real:
            # Pad null tokens with repeat
            pad = null_tokens[:, -1:, :].expand(-1, M_real - M_null, -1)
            null_tokens = torch.cat([null_tokens, pad], dim=1)
        elif M_null > M_real:
            null_tokens = null_tokens[:, :M_real, :]

        if training and self.cfg.p_drop_text > 0:
            mask = torch.rand(B, device=device) < self.cfg.p_drop_text
            if mask.any():
                token_hidden = torch.where(
                    mask[:, None, None], null_tokens, token_hidden,
                )
                pooled = torch.where(mask[:, None], null_pooled, pooled)

        return token_hidden, pooled, null_tokens, null_pooled, attention_mask

    # ------------------------------------------------------------------
    #  forward / forward_train
    # ------------------------------------------------------------------

    def forward(
        self, x0: torch.Tensor, texts: list[str],
    ) -> dict[str, Any]:
        """Training forward pass (alias for forward_train)."""
        return self.forward_train(x0, texts)

    def forward_train(
        self,
        x0: torch.Tensor,
        texts: list[str],
        meta_override: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Full training forward pass.

        Pipeline:
            1. Encode text        -> text_tokens, pooled
            2. Planner             -> meta_pred (B, K, 9)
            3. Oracle encode x0   -> z0 (B, K, D)
            4. SCCI + causal + MoE on z0 -> z_moe (conditioned clean latent)
            5. Sample t, add noise -> z_t
            6. Denoiser predicts noise from z_t
            7. Decode z_moe       -> x_hat (for reconstruction loss)

        Conditioning is applied to the clean latent before noise is
        added, matching the MATDTrainer pipeline.

        Args:
            x0: (B, T, 1) ground-truth time series.
            texts: List of B text descriptions.
            meta_override: Optional (B, K, 9) oracle metadata.

        Returns:
            Dict with loss_total, individual loss terms, and all
            intermediates (z0, z_moe, eps_pred, meta_pred, ...).
        """
        B = x0.shape[0]
        device = x0.device

        # -- 1. Text encoding (with CFG dropout during training) --
        token_hidden, pooled, null_tokens, null_pooled, attention_mask = (
            self._encode_text_cfg(texts, training=self.training)
        )

        # -- 2. Oracle tokenizer: x0 -> patch latents --
        if x0.dim() == 3 and x0.shape[-1] == 1:
            x0_for_encoder = x0.squeeze(-1)
        elif x0.dim() == 2:
            x0_for_encoder = x0
        else:
            raise ValueError(f"Expected x0 shape (B,T) or (B,T,1), got {tuple(x0.shape)}")
        z0, meta_oracle = self.encoder(x0_for_encoder)  # (B, K, D), (B, K, 9)
        K, D = z0.shape[1], z0.shape[2]

        # -- 3. Planner: text -> patch metadata --
        # HuggingFace mask: 1=real, 0=pad  -->  PyTorch key_padding_mask: True=ignore
        text_key_padding_mask = (attention_mask == 0)  # (B, M) bool
        meta_pred = self.planner(
            token_hidden, text_padding_mask=text_key_padding_mask,
            n_patches=z0.shape[1],
        )  # (B, K, 9)

        assert meta_pred.shape == meta_oracle.shape, (
            f"Planner/tokenizer mismatch: "
            f"meta_pred={tuple(meta_pred.shape)}, meta_oracle={tuple(meta_oracle.shape)}"
        )

        # Staged meta: gradually transition from oracle to planner predictions
        if meta_override is not None:
            meta = meta_override
        elif self.training and hasattr(self.cfg, 'use_oracle_meta_prob'):
            if torch.rand(()) < self.cfg.use_oracle_meta_prob:
                meta = meta_oracle.detach()
            else:
                meta = meta_pred
        else:
            meta = meta_pred

        # -- 4. SCCI + causal + MoE (on clean z0) --
        t_dummy = torch.zeros(B, dtype=torch.long, device=device)
        t_emb = self.time_embed(t_dummy)  # (B, D)

        slots = self.slot_extractor(token_hidden)  # (B, n_slots, D)

        # meta_9 must be computed BEFORE causal so patch-aligned segments work
        meta_9 = meta[:, :, :9].to(z0.dtype)

        causal_out = self.causal(z0, pooled, meta=meta_9)
        causal_feat = causal_out[0]     # (B, K, D)
        A0, Alags = causal_out[1], causal_out[2]
        causal_losses = causal_out[3]   # dict
        z_cond, attn_w, u = self.injector(
            z0, meta_9, slots, t_emb, causal_feat,
        )

        meta_embed = meta[:, :, :9].mean(dim=1)  # (B, 9) pool over K
        t_emb_k = t_emb.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)
        z_moe, router_p, prior_p = self.moe(
            z_cond, meta_embed, u, t_emb_k, causal_feat,
        )

        # -- 5. Diffusion: sample t, add noise --
        t = torch.randint(
            0, self.cfg.timesteps, (B,), device=device, dtype=torch.long,
        )
        noise = torch.randn_like(z_moe)
        z_t = self._q_sample(z_moe, t, noise)

        # -- 6. Denoiser predicts noise / velocity --
        eps_pred = self.denoiser(
            z_t, t, token_hidden, meta.to(z_t.dtype), pooled,
        )

        if self.cfg.pred_mode == "eps":
            target = noise
        else:  # v-prediction
            ab = self._extract(t, z_moe)
            target = ab.sqrt() * noise - (1.0 - ab).sqrt() * z_moe

        # -- 7. Decode for reconstruction loss --
        target_len = x0.shape[1] if x0.dim() >= 2 else K * D
        x_hat = self.decoder(z_moe, meta.to(z_moe.dtype), target_len)

        # -- 8. Compute losses --
        loss_dicts: dict[str, dict[str, torch.Tensor]] = {}

        loss_dicts["diffusion"] = self.loss_diffusion(eps_pred, target)
        loss_dicts["reconstruction"] = self.loss_recon(z_moe, z0)

        x_hat_sq = x_hat.squeeze(-1) if x_hat.dim() == 3 else x_hat
        x0_sq = x0.squeeze(-1) if x0.dim() == 3 else x0
        T_sig = min(x_hat_sq.shape[1], x0_sq.shape[1])
        loss_dicts["delta"] = self.loss_delta(
            x_hat_sq[:, :T_sig], x0_sq[:, :T_sig],
        )
        loss_dicts["fft"] = self.loss_fft(
            x_hat_sq[:, :T_sig], x0_sq[:, :T_sig],
        )

        planner_target = meta_oracle.detach().to(meta_pred.dtype)
        loss_dicts["planner"] = self.loss_planner(meta_pred, planner_target)

        ts_embed = z0.mean(dim=1)  # (B, D)
        loss_dicts["alignment"] = self.loss_align(ts_embed, pooled)

        loss_dicts["moe"] = self.loss_moe(router_p)

        loss_dicts["causal"] = {
            "loss_mechanism": causal_losses["predict"],
            "loss_dag": causal_losses["notears"],
            "loss_sparsity": causal_losses["sparsity"],
            "loss_smooth": causal_losses["smooth"],
            "loss_disentangle": causal_losses["disentangle"],
            "loss_causal": causal_losses["loss_causal"],
        }

        weights = LossWeights(
            diffusion=1.0,
            reconstruction=self.cfg.lambda_x0,
            delta=self.cfg.lambda_delta,
            fft=self.cfg.lambda_fft,
            alignment=self.cfg.lambda_align,
            moe=self.cfg.lambda_moe,
            causal=self.cfg.lambda_causal,
            planner=self.cfg.lambda_plan,
        )

        total_loss, all_terms = compute_total_loss(loss_dicts, weights)

        return {
            "loss_total": total_loss,
            **all_terms,
            "z0": z0, "z_moe": z_moe, "z_t": z_t, "noise": noise,
            "eps_pred": eps_pred, "t": t, "x_hat": x_hat,
            "meta_pred": meta_pred, "meta": meta, "meta_oracle": meta_oracle,
            "token_hidden": token_hidden, "pooled": pooled,
            "null_tokens": null_tokens, "null_pooled": null_pooled,
            "router_p": router_p, "prior_p": prior_p,
            "causal_losses": causal_losses, "causal_feat": causal_feat,
            "A0": A0, "Alags": Alags, "slots": slots, "attn_w": attn_w,
        }

    # ------------------------------------------------------------------
    #  Inference: DDIM generation with CFG
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        texts: list[str] | str,
        target_length: int = 64,
        K: int | None = None,
        n_samples: int | None = None,
        cfg_scale: float | None = None,
        ddim_steps: int | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate time series from text captions via DDIM + CFG.

        Pipeline:
            1. Encode text  -> text_tokens, pooled
            2. Planner       -> meta (B, K, 9)
            3. Sample z_T ~ N(0, I)
            4. DDIM reverse loop with CFG
            5. Decode z_0   -> x_hat (n_samples, target_length, 1)

        The denoiser operates directly on z_t with text conditioning;
        SCCI / MoE / causal are training-time only.

        Args:
            texts: One or more text descriptions.
            target_length: Output time-series length T.
            K: Number of latent patches (default cfg.target_tokens).
            n_samples: Samples per text (default len(texts)).
            cfg_scale: CFG guidance scale (default cfg.cfg_scale).
            ddim_steps: DDIM steps (default cfg.ddim_steps).
            eta: DDIM eta (0 = deterministic, 1 = DDPM).

        Returns:
            (n_samples, target_length, 1) generated time series.
        """
        if isinstance(texts, str):
            texts = [texts]

        B_text = len(texts)
        if n_samples is None:
            n_samples = B_text
        if K is None:
            K = self.encoder._choose_k(target_length)
        if cfg_scale is None:
            cfg_scale = self.cfg.cfg_scale
        if ddim_steps is None:
            ddim_steps = self.cfg.ddim_steps

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        try:
            # -- 1. Encode text --
            text_tokens, pooled, attention_mask = self._encode_text(texts)
            null_tokens, null_pooled = self.null_encoder(B_text)

            # -- 2. Planner -> meta --
            text_key_padding_mask = (attention_mask == 0)  # (B, M) bool
            meta = self.planner(
                text_tokens, text_padding_mask=text_key_padding_mask,
                n_patches=K,
            )  # (B_text, K, 9)

            # Broadcast if n_samples != B_text
            if n_samples != B_text:
                text_tokens = text_tokens[:1].expand(n_samples, -1, -1)
                pooled = pooled[:1].expand(n_samples, -1)
                null_tokens = null_tokens[:1].expand(n_samples, -1, -1)
                null_pooled = null_pooled[:1].expand(n_samples, -1)
                meta = meta[:1].expand(n_samples, -1, -1)

            # -- 3. Sample z_T --
            D = self.cfg.embed_dim
            z = torch.randn(n_samples, K, D, device=device)

            # -- 4. DDIM reverse loop --
            total = self.cfg.timesteps
            if ddim_steps >= total:
                timesteps = torch.arange(
                    total - 1, -1, -1, device=device, dtype=torch.long,
                )
            else:
                indices = torch.linspace(
                    0, total - 1, ddim_steps, device=device,
                ).long()
                timesteps = indices.flip(0)  # descending

            for i, t_cur in enumerate(timesteps):
                t = t_cur.expand(n_samples)

                # Conditional prediction
                eps_cond = self.denoiser(
                    z, t, text_tokens, meta.to(z.dtype), pooled,
                )

                # Unconditional prediction (for CFG)
                if cfg_scale != 1.0:
                    eps_uncond = self.denoiser(
                        z, t, null_tokens, meta.to(z.dtype), null_pooled,
                    )
                    eps_guided = (
                        eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                    )
                else:
                    eps_guided = eps_cond

                # Predict x_0
                ab_t = self._extract(t, z)
                if self.cfg.pred_mode == "eps":
                    x0_pred = (
                        (z - (1.0 - ab_t).sqrt() * eps_guided)
                        / ab_t.sqrt().clamp(min=1e-8)
                    )
                else:
                    x0_pred = (
                        ab_t.sqrt() * z - (1.0 - ab_t).sqrt() * eps_guided
                    )
                x0_pred = x0_pred.clamp(-5.0, 5.0)

                # Previous alpha_bar
                if i + 1 < len(timesteps):
                    t_prev = timesteps[i + 1]
                    ab_prev = self.alpha_bar[t_prev].to(
                        device=device, dtype=z.dtype,
                    )
                else:
                    ab_prev = torch.tensor(1.0, device=device, dtype=z.dtype)

                # DDIM update
                sigma = eta * (
                    (1.0 - ab_prev) / (1.0 - ab_t)
                    * (1.0 - ab_t / ab_prev)
                ).clamp(min=0).sqrt()

                direction = (
                    (1.0 - ab_prev - sigma ** 2).clamp(min=0).sqrt()
                    * eps_guided
                )
                noise = (
                    torch.randn_like(z) if eta > 0 else torch.zeros_like(z)
                )
                z = ab_prev.sqrt() * x0_pred + direction + sigma * noise

            # -- 5. Decode --
            x_hat = self.decoder(z, meta.to(z.dtype), target_length)

        finally:
            if was_training:
                self.train()

        return x_hat  # (n_samples, target_length, 1)
