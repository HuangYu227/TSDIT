import torch
from .base import BaseSampler


class DDIMSampler(BaseSampler):
    """DDIM sampler. Adapted from VerbalTS."""

    def _get_alpha(self, alpha, t, x):
        """Index alpha schedule and reshape to match x dims (3D or 4D)."""
        a = alpha[t]
        while a.dim() < x.dim():
            a = a.unsqueeze(-1)
        return a

    def predict_x0(self, x, pred_noise, t):
        """Predict x_0 from x_t and predicted noise.
        x_0 = (x_t - sqrt(1-alpha_bar_t) * pred_noise) / sqrt(alpha_bar_t)
        """
        alpha_bar_t = self._get_alpha(self.alpha_bar, t, x)
        return (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

    def reverse(self, x, pred_noise, t, noise, is_determin=True):
        """DDIM reverse step (eta=0 for deterministic).
        x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_hat + sqrt(1 - alpha_bar_{t-1}) * pred_noise_dir
        """
        alpha_bar_t = self._get_alpha(self.alpha_bar, t, x)
        x0 = self.predict_x0(x, pred_noise, t)

        # For t=0, just return x0
        t_prev = t - 1
        t_prev = torch.clamp(t_prev, min=0)
        alpha_bar_prev = self._get_alpha(self.alpha_bar, t_prev, x)

        # Direction pointing to x_t
        sigma = 0.0 if is_determin else torch.sqrt(self._get_alpha(self.beta, t, x))
        pred_noise_dir = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise

        x_prev = torch.sqrt(alpha_bar_prev) * x0 + pred_noise_dir
        if not is_determin:
            x_prev = x_prev + sigma * noise

        # Handle t=0 case
        mask = (t == 0).float()
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        x_prev = mask * x0 + (1 - mask) * x_prev
        return x_prev
