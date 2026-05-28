import torch
import numpy as np


class BaseSampler:
    """Base class for diffusion samplers. Adapted from VerbalTS."""

    def __init__(self, num_steps=50, beta_start=0.0001, beta_end=0.02,
                 schedule="quad", device="cuda:0"):
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
            raise ValueError(
                f"beta_start/beta_end must be in (0, 1), got {beta_start}, {beta_end}"
            )
        if beta_start > beta_end:
            raise ValueError(
                f"beta_start must be <= beta_end, got {beta_start} > {beta_end}"
            )
        self.num_steps = num_steps
        self.device = device

        if schedule == "quad":
            self.beta = np.linspace(
                beta_start**0.5, beta_end**0.5, self.num_steps, dtype=np.float32
            )**2
        elif schedule == "linear":
            self.beta = np.linspace(beta_start, beta_end, self.num_steps, dtype=np.float32)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

        self.beta = torch.tensor(self.beta).reshape(self.num_steps, 1, 1).to(self.device)
        self.alpha = torch.tensor(self.alpha).reshape(self.num_steps, 1, 1).to(self.device)
        self.alpha_bar = torch.tensor(self.alpha_bar).reshape(self.num_steps, 1, 1).to(self.device)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(torch.clamp(1 - self.alpha_bar, min=1e-12))

    def forward(self, x, t, noise):
        raise NotImplementedError

    def reverse(self, x, pred_noise, t, noise, **kwargs):
        raise NotImplementedError
