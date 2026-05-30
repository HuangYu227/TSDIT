import torch
from .base import BaseSampler


class DDPMSampler(BaseSampler):
    """DDPM sampler. Adapted from VerbalTS."""

    def _get_alpha(self, alpha, t, x):
        """Index alpha schedule and reshape to match x dims (3D or 4D)."""
        a = alpha[t]
        while a.dim() < x.dim():
            a = a.unsqueeze(-1)
        return a

    def forward(self, x, t, noise):
        """Forward diffusion: q(x_t | x_0).
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        alpha_bar_t = self._get_alpha(self.alpha_bar, t, x)
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

    def reverse(self, x, pred_noise, t, noise):
        """Reverse diffusion: p(x_{t-1} | x_t).
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * pred_noise) + sigma_t * noise
        """
        alpha_t = self._get_alpha(self.alpha, t, x)
        alpha_bar_t = self._get_alpha(self.alpha_bar, t, x)
        beta_t = self._get_alpha(self.beta, t, x)

        denom = torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-12))
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / denom) * pred_noise)
        sigma = torch.sqrt(beta_t)
        # t=0 is the final step: return clean prediction without noise
        mask = (t > 0).float().view(-1, 1, 1, 1)
        return mask * (mean + sigma * noise) + (1 - mask) * mean
