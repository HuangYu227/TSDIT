"""MATD Generator -- DDIM inference for the Multimodal Adaptive Temporal Diffusion framework.

Generates time series from text captions via classifier-free guidance
DDIM sampling in patch-latent space, followed by decoding to the final
time series.

Reference: MATD framework design doc.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import T2PDenoiser
from .text_encoder import MATDTextEncoder, NullTextEncoder


def _make_alpha_bar(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    schedule: str = "cosine",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build the alpha_bar cumulative product schedule."""
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


class MATDGenerator:
    """Generator for the MATD model with DDIM sampling.

    Generates time series from text captions via classifier-free
    guidance (CFG) DDIM sampling in patch-latent space.

    The model dict should contain:
        text_encoder:   MATDTextEncoder
        null_encoder:   NullTextEncoder
        planner:        TextToPatchPlanner
        slot_extractor: TextSemanticSlotExtractor
        injector:       SemanticCausalConditionInjector
        causal:         DynamicCausalMechanismLearner
        moe:            SemanticCausalTemporalMoE
        denoiser:       T2PDenoiser
        decoder:        VariablePatchDecoder | LinearPatchDecoder

    Args:
        model: Dict of trained nn.Module sub-components.
        config: Configuration dict (must contain diffusion.num_steps,
                diffusion.schedule, etc.).
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

        # Move to device and set eval mode
        for name in list(self.model.keys()):
            self.model[name] = self.model[name].to(self.device).eval()

        # Diffusion schedule
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

        # Planner config
        self.n_patches = config.get("n_patches", 16)

    # ------------------------------------------------------------------
    #  Schedule helpers
    # ------------------------------------------------------------------

    def _extract(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gather alpha_bar[t] and reshape for broadcasting."""
        out = self.alpha_bar.gather(0, t).to(device=x.device, dtype=x.dtype)
        return out.view(x.shape[0], *([1] * (x.dim() - 1)))

    # ------------------------------------------------------------------
    #  Conditioning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_text(
        self, texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text through the text encoder.

        Returns:
            token_hidden:  (B, M, D)
            pooled:        (B, D)
            attention_mask: (B, M) HuggingFace-style (1=real, 0=pad)
        """
        return self.model["text_encoder"](texts)

    @torch.no_grad()
    def _get_null_cond(
        self, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get null (unconditional) embeddings for CFG."""
        return self.model["null_encoder"](batch_size)

    @torch.no_grad()
    def _get_planner_meta(
        self, token_hidden: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
        K: int | None = None,
    ) -> torch.Tensor:
        """Predict patch metadata from text tokens.

        Args:
            token_hidden: (B, M, D) text token embeddings.
            text_padding_mask: (B, M) bool mask (True = padding).
            K: Number of patches to generate. Must match z_T's K
               to avoid shape mismatch in denoiser/decoder.

        Returns:
            meta: (B, K, 9)
        """
        return self.model["planner"](
            token_hidden, text_padding_mask=text_padding_mask, n_patches=K,
        )

    @torch.no_grad()
    def _build_conditioning(
        self,
        token_hidden: torch.Tensor,
        pooled: torch.Tensor,
        z_t: torch.Tensor,
        meta: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the full conditioning stack for the denoiser.

        Runs SCCI + causal + MoE to produce a conditioned latent,
        but for DDIM sampling we use the denoiser directly with the
        text tokens and pooled embedding.

        Returns:
            token_hidden, pooled (passed through to denoiser)
        """
        # For DDIM sampling the denoiser takes (z_t, t, text_tokens, meta, pooled).
        # The SCCI/MoE/causal modules are used during training to refine z;
        # at inference the denoiser operates directly on z_t with text conditioning.
        return token_hidden, pooled

    # ------------------------------------------------------------------
    #  DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ddim_sample(
        self,
        z_T: torch.Tensor,
        text_tokens: torch.Tensor,
        null_tokens: torch.Tensor,
        meta: torch.Tensor,
        pooled: torch.Tensor,
        null_pooled: torch.Tensor,
        cfg_scale: float,
        steps: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM reverse sampling loop with classifier-free guidance.

        Implements the deterministic (eta=0) DDIM sampler with CFG:

            eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        Args:
            z_T: (B, K, D) initial noise.
            text_tokens: (B, M, D) conditional text token embeddings.
            null_tokens: (B, M, D) unconditional (null) token embeddings.
            meta: (B, K, 9) patch metadata from planner.
            pooled: (B, D) conditional pooled text embedding.
            null_pooled: (B, D) unconditional pooled embedding.
            cfg_scale: Classifier-free guidance scale.
            steps: Number of DDIM steps.
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM).

        Returns:
            z_0: (B, K, D) denoised patch latents.
        """
        B = z_T.shape[0]
        denoiser: T2PDenoiser = self.model["denoiser"]

        # Build sub-sequence of timesteps
        # DDIM uses a sub-sequence of the full schedule
        total = self.num_steps
        if steps >= total:
            timesteps = torch.arange(total - 1, -1, -1, device=self.device, dtype=torch.long)
        else:
            # Evenly spaced sub-sequence
            indices = torch.linspace(0, total - 1, steps, device=self.device).long()
            timesteps = indices.flip(0)  # descending

        x = z_T

        for i, t_cur in enumerate(timesteps):
            t = t_cur.expand(B)

            # Conditional prediction
            eps_cond = denoiser(x, t, text_tokens, meta, pooled)

            # Unconditional prediction (for CFG)
            if cfg_scale != 1.0:
                eps_uncond = denoiser(x, t, null_tokens, meta, null_pooled)
                eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_guided = eps_cond

            # Predict x_0
            ab_t = self._extract(t, x)
            if self.prediction_type == "epsilon":
                x0_pred = (x - (1.0 - ab_t).sqrt() * eps_guided) / ab_t.sqrt().clamp(min=1e-8)
            else:
                # v-prediction: v = alpha_t * eps - sigma_t * x0
                x0_pred = ab_t.sqrt() * x - (1.0 - ab_t).sqrt() * eps_guided

            # x0 clipping for stability
            x0_pred = x0_pred.clamp(-5.0, 5.0)

            # Get previous alpha_bar
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                ab_prev = self.alpha_bar[t_prev].to(device=x.device, dtype=x.dtype)
            else:
                ab_prev = torch.tensor(1.0, device=x.device, dtype=x.dtype)

            # DDIM update
            sigma = eta * ((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)).clamp(min=0).sqrt()
            direction = (1.0 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps_guided
            noise = torch.randn_like(x) if (sigma > 0).any() else torch.zeros_like(x)
            x = ab_prev.sqrt() * x0_pred + direction + sigma * noise

        return x

    # ------------------------------------------------------------------
    #  Public generation API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        caption: str,
        target_length: int,
        K: int = 32,
        n_samples: int = 1,
        cfg_scale: float = 5.0,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> np.ndarray:
        """Generate time series from a text caption.

        Args:
            caption: Text description of the desired time series.
            target_length: Output time series length.
            K: Number of patches.
            n_samples: Number of samples to generate.
            cfg_scale: Classifier-free guidance scale.
            ddim_steps: Number of DDIM sampling steps.
            eta: DDIM eta (0 = deterministic).

        Returns:
            x_hat: numpy array of shape (n_samples, target_length).
        """
        texts = [caption] * n_samples
        x_hat = self.generate_batch(
            texts, target_length, K=K, cfg_scale=cfg_scale,
            ddim_steps=ddim_steps, eta=eta,
        )
        return x_hat

    @torch.no_grad()
    def generate_batch(
        self,
        captions: list[str],
        target_length: int,
        K: int = 32,
        cfg_scale: float = 5.0,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> np.ndarray:
        """Generate time series for a batch of captions.

        Args:
            captions: List of text descriptions.
            target_length: Output time series length.
            K: Number of patches.
            cfg_scale: Classifier-free guidance scale.
            ddim_steps: Number of DDIM steps.
            eta: DDIM eta parameter.

        Returns:
            x_hat: numpy array of shape (B, target_length).
        """
        B = len(captions)

        # 1. Encode text
        text_tokens, pooled, attention_mask = self._encode_text(captions)   # (B, M, D), (B, D)
        null_tokens, null_pooled = self._get_null_cond(B)   # (B, M, D), (B, D)

        # 2. Predict patch metadata (with padding mask, aligned to K)
        text_key_padding_mask = (attention_mask == 0)  # (B, M) bool
        meta = self._get_planner_meta(text_tokens, text_key_padding_mask, K=K)  # (B, K, 9)

        # 3. Sample initial noise in patch-latent space
        D = text_tokens.shape[-1]
        z_T = torch.randn(B, K, D, device=self.device)

        # 4. DDIM reverse sampling
        z_0 = self._ddim_sample(
            z_T, text_tokens, null_tokens, meta, pooled, null_pooled,
            cfg_scale=cfg_scale, steps=ddim_steps, eta=eta,
        )

        # 5. Decode to time series
        decoder = self.model["decoder"]
        x_hat_tensor = decoder(z_0, meta.to(z_0.dtype), target_length)

        # (B, T, 1) -> (B, T)
        if x_hat_tensor.dim() == 3:
            x_hat_tensor = x_hat_tensor.squeeze(-1)

        return x_hat_tensor.cpu().numpy()
