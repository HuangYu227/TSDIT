"""High-standard MATD DDIM generator.

Prefer using ``MATDModel.generate`` directly.  This wrapper exists for scripts
that manage a model object and config separately.  Unlike the older generator,
this version can keep causal conditioning active during sampling.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn


def _make_alpha_bar(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02, schedule: str = "cosine", device: torch.device = torch.device("cpu")) -> torch.Tensor:
    if schedule == "cosine":
        steps = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        s = 0.008
        x = (steps / num_steps + s) / (1 + s)
        ab = torch.cos(x * math.pi / 2) ** 2
        ab = ab / ab[0]
        return ab[1:]
    if schedule == "linear":
        beta = torch.linspace(beta_start, beta_end, num_steps, device=device)
        return torch.cumprod(1.0 - beta, dim=0)
    if schedule == "quad":
        beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, device=device) ** 2
        return torch.cumprod(1.0 - beta, dim=0)
    raise ValueError(f"Unknown schedule: {schedule}")


class MATDGenerator:
    """DDIM sampler for MATD models or MATD submodule dictionaries."""

    def __init__(self, model: nn.Module | dict[str, nn.Module], config: dict[str, Any], device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.config = config
        self.model = model
        if isinstance(model, dict):
            self.module_dict = {k: v.to(self.device).eval() for k, v in model.items()}
            self.full_model = None
        else:
            self.full_model = model.to(self.device).eval()
            self.module_dict = None
        diff_cfg = config.get("diffusion", {}) if isinstance(config.get("diffusion", {}), dict) else {}
        self.num_steps = int(diff_cfg.get("num_steps", config.get("timesteps", 1000)))
        self.prediction_type = diff_cfg.get("prediction_type", config.get("pred_mode", "eps"))
        self.alpha_bar = _make_alpha_bar(self.num_steps, schedule=diff_cfg.get("schedule", config.get("beta_schedule", "cosine")), device=self.device)
        self.default_k = int(config.get("n_patches", config.get("max_tokens", 16)))

    @torch.no_grad()
    def generate_batch(
        self,
        texts: list[str],
        target_length: int = 96,
        K: Optional[int] = None,
        ddim_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        eta: float = 0.0,
        use_causal_guidance: Optional[bool] = None,
    ) -> torch.Tensor:
        if self.full_model is not None and hasattr(self.full_model, "generate"):
            return self.full_model.generate(texts, target_length=target_length, K=K, ddim_steps=ddim_steps, cfg_scale=cfg_scale, eta=eta, use_causal_guidance=use_causal_guidance)
        return self._generate_from_modules(texts, target_length, K, ddim_steps, cfg_scale, eta, use_causal_guidance)

    __call__ = generate_batch

    @torch.no_grad()
    def _generate_from_modules(self, texts: list[str], target_length: int, K: Optional[int], ddim_steps: Optional[int], cfg_scale: Optional[float], eta: float, use_causal_guidance: Optional[bool]) -> torch.Tensor:
        m = self.module_dict
        assert m is not None
        B = len(texts)
        ddim_steps = int(ddim_steps or self.config.get("ddim_steps", 50))
        cfg_scale = float(cfg_scale if cfg_scale is not None else self.config.get("cfg_scale", 5.0))
        use_causal_guidance = bool(self.config.get("use_causal_guidance_in_sampling", True) if use_causal_guidance is None else use_causal_guidance)
        text_tokens, pooled, attention_mask = m["text_encoder"](texts)
        null_tokens, null_pooled = m["null_encoder"](B)
        text_mask = attention_mask == 0
        null_mask = torch.zeros(null_tokens.shape[:2], device=self.device, dtype=torch.bool)
        if K is None:
            enc = m.get("encoder")
            if enc is not None and hasattr(enc, "_choose_k"):
                K = enc._choose_k(target_length)
            else:
                K = self.default_k
        meta = m["planner"](text_tokens, text_padding_mask=text_mask, n_patches=K).to(self.device)
        meta_9 = meta[..., :9]
        denoiser = m["denoiser"]
        D = getattr(denoiser, "input_dim", self.config.get("embed_dim", 256))
        z = torch.randn(B, K, D, device=self.device)
        steps = torch.linspace(self.num_steps - 1, 0, ddim_steps, device=self.device).long()

        for i, step in enumerate(steps):
            t = torch.full((B,), int(step.item()), device=self.device, dtype=torch.long)
            causal_feat = None
            if use_causal_guidance and "causal" in m:
                causal_feat = m["causal"](z, pooled, meta=meta_9)[0]
            eps_cond = denoiser(z, t, text_tokens, meta_9, pooled, causal_feat=causal_feat, text_padding_mask=text_mask)
            eps_uncond = denoiser(z, t, null_tokens, meta_9, null_pooled, causal_feat=None, text_padding_mask=null_mask)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            ab_t = self.alpha_bar[step].to(z.dtype)
            if self.prediction_type == "v":
                x0_pred = ab_t.sqrt() * z - (1.0 - ab_t).sqrt() * eps
                eps_pred = ab_t.sqrt() * eps + (1.0 - ab_t).sqrt() * z
            else:
                x0_pred = (z - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp_min(1e-8)
                eps_pred = eps
            if i == len(steps) - 1:
                z = x0_pred
            else:
                ab_next = self.alpha_bar[steps[i + 1]].to(z.dtype)
                sigma = eta * ((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next)).clamp_min(0).sqrt()
                noise = torch.randn_like(z) if eta > 0 else 0.0
                z = ab_next.sqrt() * x0_pred + (1.0 - ab_next - sigma ** 2).clamp_min(0).sqrt() * eps_pred + sigma * noise
        return m["decoder"](z, meta_9, target_length)
