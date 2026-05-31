"""Minimal DDPM/DDIM samplers compatible with TIGERGenerator."""

from __future__ import annotations

import torch


def _make_beta_schedule(num_steps: int, beta_start: float, beta_end: float, schedule: str, device):
    if schedule == "linear":
        beta = torch.linspace(beta_start, beta_end, num_steps, device=device)
    elif schedule == "quad":
        beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, device=device) ** 2
    elif schedule == "cosine":
        steps = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        s = 0.008
        x = (steps / num_steps + s) / (1 + s)
        alpha_bar = torch.cos(x * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        beta = beta.clamp(1e-5, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return beta


class DDPMSampler:
    def __init__(self, num_steps: int, beta_start: float, beta_end: float, schedule: str, device):
        self.num_steps = num_steps
        self.device = device
        self.beta = _make_beta_schedule(num_steps, beta_start, beta_end, schedule, device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = arr.gather(0, t).to(device=x.device, dtype=x.dtype)
        return out.view(x.shape[0], *([1] * (x.dim() - 1)))

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self._extract(self.alpha_bar, t, x0)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def reverse(self, xt: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        beta_t = self._extract(self.beta, t, xt)
        alpha_t = self._extract(self.alpha, t, xt)
        ab_t = self._extract(self.alpha_bar, t, xt)
        mean = (xt - beta_t / (1.0 - ab_t).sqrt() * pred_noise) / alpha_t.sqrt()
        nonzero = (t > 0).to(dtype=xt.dtype).view(xt.shape[0], *([1] * (xt.dim() - 1)))
        return mean + nonzero * beta_t.sqrt() * noise


class DDIMSampler(DDPMSampler):
    def reverse(
        self,
        xt: torch.Tensor,
        pred_noise: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        is_determin: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        ab_t = self._extract(self.alpha_bar, t, xt)
        prev_t = (t - 1).clamp(min=0)
        ab_prev = self._extract(self.alpha_bar, prev_t, xt)
        x0 = (xt - (1.0 - ab_t).sqrt() * pred_noise) / ab_t.sqrt()
        if is_determin or eta == 0.0:
            sigma = torch.zeros_like(ab_t)
        else:
            sigma = eta * (((1.0 - ab_prev) / (1.0 - ab_t)) * (1.0 - ab_t / ab_prev)).clamp(min=0).sqrt()
        direction = (1.0 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise
        prev = ab_prev.sqrt() * x0 + direction + sigma * noise
        nonzero = (t > 0).to(dtype=xt.dtype).view(xt.shape[0], *([1] * (xt.dim() - 1)))
        return torch.where(nonzero.bool(), prev, x0)
