import torch
from .base import BaseSampler


class DDPMSampler(BaseSampler):
    """DDPM sampler. Adapted from VerbalTS."""

    def forward(self, x, t, noise):
        """Forward diffusion: q(x_t | x_0).
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        alpha_bar_t = self.alpha_bar[t]
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

    def reverse(self, x, pred_noise, t, noise):
        """Reverse diffusion: p(x_{t-1} | x_t).
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * pred_noise) + sigma_t * noise
        """
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.beta[t]

        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise
