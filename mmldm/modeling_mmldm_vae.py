# Copyright 2026 MMLDM Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MMLDM VAE v2 — Spectral Dual-Latent + Engineering Improvements.

Innovations:
- A: Spectral Dual-Latent (trend + residual subspaces)
- C: Temporal Contrastive Latent Regularization (TCLR)
- Engineering: Spectral reconstruction loss, latent standardization

Architecture:
- Encoder: FFT decomposition -> trend (low-freq) + residual (high-freq)
  -> Conv1d encoders -> separate latent projections
- Decoder: merge [z_trend; z_residual] -> Conv1d decoder -> time domain
- Text encoder: MLP projection (SBERT -> latent space)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration_mmldm import MMLDMVAEConfig


# ---------------------------------------------------------------------------
# Diagonal Gaussian Distribution
# ---------------------------------------------------------------------------


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior for the VAE latent space."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        assert parameters.ndim in (2, 3)
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * sample

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other_mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        if other_mean is None:
            other_mean = torch.zeros_like(self.mean)
        return 0.5 * (
            self.mean.pow(2) + self.var - 1.0 - self.logvar
        ).sum(dim=-1).mean()


@dataclass
class TextVAEEncoderOutput:
    latents_list: list[torch.Tensor]
    latent_dists: Optional[tuple] = None  # (trend_dists, residual_dists) or (trend_dists, period_dists, residual_dists)
    text_latents: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# FFT decomposition helpers (Innovation A)
# ---------------------------------------------------------------------------


def fft_decompose(x: torch.Tensor, cutoff_ratio: float = 0.3):
    """Decompose time series into low-freq (trend) and high-freq (residual).

    Args:
        x: (B, C, L) time series in time domain.
        cutoff_ratio: fraction of frequencies to keep as low-freq.

    Returns:
        x_low: (B, C, L) low-frequency component (trend).
        x_high: (B, C, L) high-frequency component (residual).
    """
    B, C, L = x.shape
    X = torch.fft.rfft(x, dim=-1)  # (B, C, L//2+1)
    freq_len = X.shape[-1]
    cutoff = max(1, int(freq_len * cutoff_ratio))

    X_low = X.clone()
    X_low[:, :, cutoff:] = 0
    x_low = torch.fft.irfft(X_low, n=L, dim=-1)

    x_high = x - x_low
    return x_low, x_high


def fft_tri_decompose(x: torch.Tensor, band_boundaries: torch.Tensor):
    """Decompose time series into three frequency bands with learnable boundaries.

    Args:
        x: (B, C, L) time series in time domain.
        band_boundaries: (2,) learnable raw boundaries (pre-sigmoid).

    Returns:
        x_low:  (B, C, L) low-frequency component (trend).
        x_mid:  (B, C, L) mid-frequency component (periodic).
        x_high: (B, C, L) high-frequency component (residual).
    """
    B, C, L = x.shape
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(L, device=x.device, dtype=x.dtype)

    low_bound = torch.sigmoid(band_boundaries[0])
    high_bound = torch.sigmoid(band_boundaries[1])

    # Soft masks via steep sigmoid for differentiability
    low_mask = torch.sigmoid(-50.0 * (freqs - low_bound))
    mid_mask = torch.sigmoid(50.0 * (freqs - low_bound)) * torch.sigmoid(-50.0 * (freqs - high_bound))
    high_mask = torch.sigmoid(50.0 * (freqs - high_bound))

    x_low = torch.fft.irfft(X * low_mask, n=L, dim=-1)
    x_mid = torch.fft.irfft(X * mid_mask, n=L, dim=-1)
    x_high = torch.fft.irfft(X * high_mask, n=L, dim=-1)

    return x_low, x_mid, x_high


# ---------------------------------------------------------------------------
# Product of Experts (P2-1)
# ---------------------------------------------------------------------------


class ProductOfExperts(nn.Module):
    """Product of Experts fusion over diagonal Gaussian distributions.

    Given K experts with means μ_k and variances σ²_k, the PoE posterior is:
        σ²_poe = 1 / Σ_k (1/σ²_k)
        μ_poe  = σ²_poe * Σ_k (μ_k / σ²_k)

    This weights each expert by its precision (inverse variance), so
    uncertain experts contribute less to the fused distribution.
    """

    def forward(self, distributions: list[DiagonalGaussianDistribution]) -> torch.Tensor:
        """Fuse K expert distributions via PoE.

        Args:
            distributions: list of DiagonalGaussianDistribution, each with
                .mean (*, D) and .var (*, D).

        Returns:
            Fused parameters (*, 2D) = [poe_mean; log(poe_var)].
        """
        if not distributions:
            raise ValueError("ProductOfExperts requires at least one distribution")
        ref_shape = distributions[0].mean.shape
        for d in distributions[1:]:
            if d.mean.shape != ref_shape:
                raise ValueError(
                    "All PoE experts must describe the same latent space; "
                    f"got {ref_shape} and {d.mean.shape}"
                )
        precisions = [1.0 / (d.var + 1e-8) for d in distributions]
        total_precision = sum(precisions)
        poe_var = 1.0 / total_precision
        poe_mean = poe_var * sum(d.mean * p for d, p in zip(distributions, precisions))
        poe_logvar = torch.log(poe_var + 1e-8)
        return torch.cat([poe_mean, poe_logvar], dim=-1)


# ---------------------------------------------------------------------------
# Conv1d building blocks
# ---------------------------------------------------------------------------


class ConvResidual(nn.Module):
    """Conv1d(k=3,p=1) -> SiLU -> Conv1d(k=1) + skip connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)  # InstanceNorm for stability
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(self.norm(x))))


class ConvResidualStack(nn.Module):
    """Stack of ConvResidual blocks."""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([ConvResidual(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ConvEncoder(nn.Module):
    """Conv1d encoder: preserves sequence length (stride=1, k=3, p=1)."""

    def __init__(self, in_channels: int, dim: int, num_layers: int):
        super().__init__()
        self.proj_in = nn.Conv1d(in_channels, dim, kernel_size=3, padding=1)
        self.stack = ConvResidualStack(dim, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.stack(x)
        return x


class ConvDecoder(nn.Module):
    """Conv1d decoder: preserves sequence length (stride=1, k=3, p=1)."""

    def __init__(self, out_channels: int, dim: int, num_layers: int):
        super().__init__()
        self.proj_in = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.stack = ConvResidualStack(dim, num_layers)
        self.proj_out = nn.Conv1d(dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.stack(x)
        x = self.proj_out(x)
        return x


# ---------------------------------------------------------------------------
# Multi-Resolution FFT Loss (P3-2)
# ---------------------------------------------------------------------------


class MultiResolutionFFTLoss(nn.Module):
    """Multi-resolution STFT loss for finer spectral fidelity.

    Computes Short-Time Fourier Transform at multiple window sizes and
    averages the L1 loss on real and imaginary parts.  Smaller windows
    capture fine-grained time-localised detail; larger windows capture
    global spectral shape.
    """

    def __init__(self, fft_sizes: Optional[list[int]] = None, hop_ratio: float = 0.5):
        super().__init__()
        self.fft_sizes = fft_sizes or [96, 48, 24, 12]
        self.hop_ratio = hop_ratio

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-resolution STFT L1 loss.

        Args:
            recon:  (B, C, L) reconstructed signal.
            target: (B, C, L) ground-truth signal.
        Returns:
            Scalar loss.
        """
        losses = []
        L = recon.shape[-1]
        for n_fft in self.fft_sizes:
            if n_fft > L:
                continue  # skip windows larger than the signal
            hop = max(1, int(n_fft * self.hop_ratio))
            win = torch.hann_window(n_fft, device=recon.device, dtype=recon.dtype)
            # Average over channels
            loss_n = torch.tensor(0.0, device=recon.device, dtype=recon.dtype)
            for c in range(recon.shape[1]):
                s_r = torch.stft(recon[:, c, :], n_fft=n_fft, hop_length=hop,
                                 win_length=n_fft, window=win, return_complex=True)
                s_t = torch.stft(target[:, c, :], n_fft=n_fft, hop_length=hop,
                                 win_length=n_fft, window=win, return_complex=True)
                loss_n = loss_n + F.l1_loss(s_r.real, s_t.real) + F.l1_loss(s_r.imag, s_t.imag)
            losses.append(loss_n / recon.shape[1])
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=recon.device, dtype=recon.dtype)


# ---------------------------------------------------------------------------
# Text Spectral Hypernetwork (innovative text conditioning)
# ---------------------------------------------------------------------------


class FourierFeatureMapping(nn.Module):
    """Random Fourier feature mapping (Tancik et al., NeurIPS 2020).

    Projects low-dimensional input into a high-dimensional Fourier feature
    space, enabling the network to learn high-frequency functions.
    Crucially, the mapping is fixed (not learned), so it acts as a
    deterministic kernel that preserves input topology.
    """

    def __init__(self, in_dim: int, n_frequencies: int = 64, sigma: float = 10.0):
        super().__init__()
        # Fixed random projection — not trainable
        B = torch.randn(in_dim, n_frequencies) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → (B, 2*n_frequencies)"""
        proj = x @ self.B  # (B, n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TextSpectralHypernetwork(nn.Module):
    """Text-conditioned spectral modulation hypernetwork.

    Instead of producing a single conditioning vector, this module
    predicts per-band FiLM parameters (γ, β) that directly modulate
    the VAE's frequency-decomposed representations.

    Key innovation: the text embedding controls WHICH frequencies are
    emphasized/suppressed in each band, grounding the conditioning
    in the VAE's spectral structure rather than an abstract latent space.

    Architecture:
        text_emb → FourierFeatureMapping → MLP → {γ_trend, β_trend, γ_resid, β_resid, ...}

    The Fourier mapping is critical: it enables the network to learn
    sharp, frequency-selective modulation patterns that a standard MLP
    cannot represent (spectral bias / F-principle).

    Additionally produces a latent conditioning vector for downstream
    DiT compatibility (TGFM, alignment loss).
    """

    def __init__(self, text_dim: int, hidden_dim: int, latent_dim: int,
                 n_bands: int = 2, n_frequencies: int = 64, dropout: float = 0.0,
                 num_tokens: int = 1):
        super().__init__()
        self.n_bands = n_bands
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim

        # Fourier feature mapping to overcome spectral bias
        self.fourier = FourierFeatureMapping(text_dim, n_frequencies, sigma=10.0)
        fourier_dim = n_frequencies * 2

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-band FiLM heads: each predicts (gamma, beta) for its frequency band
        self.film_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2)  # (γ, β) per band
            for _ in range(n_bands)
        ])
        # Initialize to identity transform (γ=1, β=0)
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, 0.0)
            # Set γ (first output) to ~1 via bias
            head.bias.data[0] = 1.0

        # Latent conditioning head (for DiT compatibility)
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.token_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * latent_dim),
        )
        self.token_queries = nn.Parameter(torch.randn(num_tokens, latent_dim) * 0.02)
        self.token_norm = nn.LayerNorm(latent_dim)

    def forward(self, text_embs: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Encode text and predict spectral modulation parameters.

        Args:
            text_embs: (B, text_dim) SBERT embeddings.

        Returns:
            latent: (B, latent_dim) conditioning vector for DiT.
            film_params: list of (gamma, beta) tuples, one per band.
                Each gamma, beta is (B, 1) — broadcast over channels and time.
        """
        z_fourier = self.fourier(text_embs)  # (B, 2*n_freq)
        h = self.trunk(z_fourier)            # (B, hidden_dim)

        film_params = []
        for head in self.film_heads:
            params = head(h)  # (B, 2)
            gamma = params[:, 0:1]  # (B, 1)
            beta = params[:, 1:2]   # (B, 1)
            film_params.append((gamma, beta))

        latent = self.latent_head(h)  # (B, latent_dim)
        return latent, film_params

    def encode_tokens(self, text_embs: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Encode text into multiple semantic condition tokens.

        Returns:
            tokens: (B, K, latent_dim), where K is ``num_tokens``.
            film_params: same per-band FiLM parameters as ``forward``.
        """
        z_fourier = self.fourier(text_embs)
        h = self.trunk(z_fourier)

        film_params = []
        for head in self.film_heads:
            params = head(h)
            gamma = params[:, 0:1]
            beta = params[:, 1:2]
            film_params.append((gamma, beta))

        token_delta = self.token_head(h).view(text_embs.shape[0], self.num_tokens, self.latent_dim)
        tokens = self.token_norm(token_delta + self.token_queries.unsqueeze(0))
        return tokens, film_params


# ---------------------------------------------------------------------------
# MMLDM VAE Model v2 — Spectral Dual-Latent
# ---------------------------------------------------------------------------


class MMLDMVAEModel(PreTrainedModel):
    """Spectral Dual-Latent VAE for MMLDM.

    Innovation A: Splits latent into trend (low-freq) and residual (high-freq)
    subspaces with FFT decomposition and separate encoders.

    Text conditioning: TextSpectralHypernetwork predicts per-band FiLM
    parameters (γ, β) that modulate frequency-decomposed representations.

    Engineering: Spectral reconstruction loss + latent standardization.
    Innovation C: Temporal Contrastive Latent Regularization (TCLR).
    """

    config_class = MMLDMVAEConfig
    base_model_prefix = "mmldm_vae"

    def __init__(self, config: MMLDMVAEConfig):
        super().__init__(config)
        self.config = config
        self.use_variation = config.use_variation
        self.use_tri_band = getattr(config, "use_tri_band", False)

        if self.use_tri_band:
            # Tri-band PoE: each expert models the same latent_dim space.
            self.trend_dim = config.latent_dim
            self.period_dim = config.latent_dim
            self.residual_dim = config.latent_dim

            self.trend_encoder = ConvEncoder(config.ts_channels, config.dim, config.num_conv_layers)
            self.period_encoder = ConvEncoder(config.ts_channels, config.dim, config.num_conv_layers)
            self.residual_encoder = ConvEncoder(config.ts_channels, config.dim, config.num_conv_layers)

            mul = 2 if config.use_variation else 1
            self.trend_proj = nn.Conv1d(config.dim, self.trend_dim * mul, kernel_size=1)
            self.period_proj = nn.Conv1d(config.dim, self.period_dim * mul, kernel_size=1)
            self.residual_proj = nn.Conv1d(config.dim, self.residual_dim * mul, kernel_size=1)

            # Pre-sigmoid band boundaries for tri-band frequency decomposition.
            # After sigmoid, boundaries map to actual frequency fractions.
            # sigmoid(-1.4) ≈ 0.20, sigmoid(-0.5) ≈ 0.38 — both within [0, 0.5] (Nyquist).
            # Trend: [0, 0.20), Period: [0.20, 0.38), Residual: [0.38, 0.5]
            self.band_boundaries = nn.Parameter(torch.tensor([-1.4, -0.5]))
            self.poe_fusion = ProductOfExperts()
        else:
            # Dual-band: trend (low) + residual (high)
            self.trend_dim = config.latent_dim // 2
            self.residual_dim = config.latent_dim - self.trend_dim

            self.trend_encoder = ConvEncoder(config.ts_channels, config.dim, config.num_conv_layers)
            self.residual_encoder = ConvEncoder(config.ts_channels, config.dim, config.num_conv_layers)

            if config.use_variation:
                self.trend_proj = nn.Conv1d(config.dim, self.trend_dim * 2, kernel_size=1)
                self.residual_proj = nn.Conv1d(config.dim, self.residual_dim * 2, kernel_size=1)
            else:
                self.trend_proj = nn.Conv1d(config.dim, self.trend_dim, kernel_size=1)
                self.residual_proj = nn.Conv1d(config.dim, self.residual_dim, kernel_size=1)

        # Decoder (merged latent)
        self.decoder_in_layer = nn.Conv1d(config.latent_dim, config.dim, kernel_size=1)
        self.decoder = ConvDecoder(config.ts_channels, config.dim, config.decoder_num_blocks)

        # Text Spectral Hypernetwork: predicts per-band FiLM parameters
        # that directly modulate frequency-decomposed representations.
        n_bands = 3 if self.use_tri_band else 2
        self.text_encoder = TextSpectralHypernetwork(
            text_dim=config.text_dim,
            hidden_dim=config.dim,
            latent_dim=config.latent_dim,
            n_bands=n_bands,
            dropout=config.dropout,
            num_tokens=getattr(config, "text_num_tokens", 1),
        )

        # Multi-Resolution FFT Loss (P3-2)
        self.multi_res_fft_loss = MultiResolutionFFTLoss()

        # Latent standardization buffers
        self.register_buffer("latent_mean", torch.zeros(config.latent_dim))
        self.register_buffer("latent_std", torch.ones(config.latent_dim))
        self.register_buffer("_latent_stats_computed", torch.tensor(False))

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, ot_list: list[torch.Tensor]) -> TextVAEEncoderOutput:
        """Encode with FFT decomposition into frequency-band latents.

        Dual-band (default): trend + residual -> concat.
        Tri-band (use_tri_band=True): trend + periodic + residual -> PoE fusion.
        """
        if self.use_tri_band:
            return self._encode_tri_band(ot_list)
        return self._encode_dual_band(ot_list)

    def _encode_dual_band(self, ot_list: list[torch.Tensor]) -> TextVAEEncoderOutput:
        trend_params, residual_params = [], []
        for ot in ot_list:
            x = ot.unsqueeze(0).permute(0, 2, 1)
            x_low, x_high = fft_decompose(x, cutoff_ratio=self.config.fft_cutoff_ratio)
            p_trend = self.trend_proj(self.trend_encoder(x_low)).permute(0, 2, 1).squeeze(0)
            p_residual = self.residual_proj(self.residual_encoder(x_high)).permute(0, 2, 1).squeeze(0)
            trend_params.append(p_trend)
            residual_params.append(p_residual)

        trend_dists = [DiagonalGaussianDistribution(p) for p in trend_params]
        residual_dists = [DiagonalGaussianDistribution(p) for p in residual_params]
        return TextVAEEncoderOutput(latents_list=[], latent_dists=(trend_dists, residual_dists))

    def _encode_tri_band(self, ot_list: list[torch.Tensor]) -> TextVAEEncoderOutput:
        trend_params, period_params, residual_params = [], [], []
        for ot in ot_list:
            x = ot.unsqueeze(0).permute(0, 2, 1)
            x_low, x_mid, x_high = fft_tri_decompose(x, self.band_boundaries)
            p_trend = self.trend_proj(self.trend_encoder(x_low)).permute(0, 2, 1).squeeze(0)
            p_period = self.period_proj(self.period_encoder(x_mid)).permute(0, 2, 1).squeeze(0)
            p_residual = self.residual_proj(self.residual_encoder(x_high)).permute(0, 2, 1).squeeze(0)
            trend_params.append(p_trend)
            period_params.append(p_period)
            residual_params.append(p_residual)

        trend_dists = [DiagonalGaussianDistribution(p) for p in trend_params]
        period_dists = [DiagonalGaussianDistribution(p) for p in period_params]
        residual_dists = [DiagonalGaussianDistribution(p) for p in residual_params]
        return TextVAEEncoderOutput(latents_list=[], latent_dists=(trend_dists, period_dists, residual_dists))

    def decode(self, z: torch.Tensor, ts_shape: torch.LongTensor) -> torch.Tensor:
        """Decode merged latents into time series."""
        lengths = ts_shape.flatten().tolist()
        z_list = z.split(lengths, dim=0)
        out_list = []
        for z_i in z_list:
            x = z_i.unsqueeze(0).permute(0, 2, 1)
            x = self.decoder_in_layer(x)
            x = self.decoder(x)
            out_list.append(x.permute(0, 2, 1).squeeze(0))
        return torch.cat(out_list, dim=0).unsqueeze(0)

    def encode_text_condition(self, text_embs: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Encode text via spectral hypernetwork.

        Args:
            text_embs: (B, text_dim) SBERT embeddings.

        Returns:
            latent: (B, latent_dim) conditioning vector for DiT.
            film_params: list of (gamma, beta) per frequency band.
                Each is (B, 1), to modulate z0 via: z0 = gamma * z0 + beta.
        """
        return self.text_encoder(text_embs)

    def encode_text_tokens(self, text_embs: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Encode text into K semantic condition tokens for DiT cross-attention."""
        return self.text_encoder.encode_tokens(text_embs)

    def compute_latent_stats(self, dataset_latents: list[torch.Tensor]):
        """Compute mean/std for latent standardization (in-place buffer update)."""
        all_latents = torch.cat(dataset_latents, dim=0)
        self.latent_mean.copy_(all_latents.mean(dim=0))
        self.latent_std.copy_(all_latents.std(dim=0).clamp(min=1e-6))
        self._latent_stats_computed.fill_(True)

    def standardize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self._latent_stats_computed.item():
            return (z - self.latent_mean) / self.latent_std
        return z

    def unstandardize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self._latent_stats_computed.item():
            return z * self.latent_std + self.latent_mean
        return z

    def forward(self, ot_list: list[torch.Tensor], tclr_weight: float = 0.1) -> dict:
        """Full forward with spectral loss and TCLR."""
        enc_output = self.encode(ot_list)

        if self.use_tri_band:
            trend_dists, period_dists, residual_dists = enc_output.latent_dists
            # PoE fusion per sample
            z_list = []
            for td, pd, rd in zip(trend_dists, period_dists, residual_dists):
                fused_params = self.poe_fusion([td, pd, rd])  # (L, 2*latent_dim)
                z_list.append(DiagonalGaussianDistribution(fused_params).sample())
        else:
            trend_dists, residual_dists = enc_output.latent_dists
            trend_samples = [d.sample() for d in trend_dists]
            residual_samples = [d.sample() for d in residual_dists]
            z_list = [torch.cat([t, r], dim=-1) for t, r in zip(trend_samples, residual_samples)]

        z = torch.cat(z_list, dim=0)
        # Standardize latent using dataset-level stats (computed after first epoch)
        if self._latent_stats_computed.item():
            z = self.standardize_latent(z)
        # Before stats are available, KL loss naturally constrains z ~ N(0,I)
        ts_shape = torch.tensor([[z_i.shape[0]] for z_i in z_list], dtype=torch.long, device=z.device)
        recon = self.decode(z, ts_shape)

        result = {"recon": recon, "latent_dists": enc_output.latent_dists, "latents": z_list}

        # Spectral reconstruction loss — per-sample to avoid cross-boundary artifacts
        spectral_losses = []
        offset = 0
        for i, ot_i in enumerate(ot_list):
            L_i = ot_i.shape[0]
            x_i = ot_i.permute(1, 0).unsqueeze(0)            # (1, C, L)
            r_i = recon[:, offset:offset + L_i, :].permute(0, 2, 1)  # (1, C, L)
            fft_x = torch.fft.rfft(x_i, dim=-1)
            fft_r = torch.fft.rfft(r_i, dim=-1)
            spectral_losses.append(
                F.l1_loss(fft_r.real, fft_x.real) + F.l1_loss(fft_r.imag, fft_x.imag)
            )
            offset += L_i
        result["spectral_loss"] = torch.stack(spectral_losses).mean()

        # Multi-Resolution FFT Loss (P3-2) — finer spectral fidelity
        mr_losses = []
        offset = 0
        for i, ot_i in enumerate(ot_list):
            L_i = ot_i.shape[0]
            x_i = ot_i.permute(1, 0).unsqueeze(0)            # (1, C, L)
            r_i = recon[:, offset:offset + L_i, :].permute(0, 2, 1)  # (1, C, L)
            mr_losses.append(self.multi_res_fft_loss(r_i, x_i))
            offset += L_i
        result["multi_res_fft_loss"] = torch.stack(mr_losses).mean()

        # Innovation C: TCLR loss
        if tclr_weight > 0:
            result["tclr_loss"] = self._compute_tclr(z_list)
        else:
            result["tclr_loss"] = torch.tensor(0.0, device=z.device)

        return result

    def _compute_tclr(self, z_list: list[torch.Tensor], margin: float = 1.0) -> torch.Tensor:
        """Temporal Contrastive Latent Regularization.

        Vectorized implementation for gradient-correct accumulation.
        """
        losses = []
        for z in z_list:
            L = z.shape[0]
            if L < 3:
                continue
            k = max(1, L // 4)
            # Positive pairs: adjacent timesteps
            d_pos = torch.norm(z[:-1] - z[1:], p=2, dim=-1)  # (L-1,)
            # Negative pairs: k-step apart
            neg_indices = torch.clamp(torch.arange(1, L, device=z.device) + k - 1, max=L - 1)
            # Ensure negative index != positive index (i+1)
            neg_indices = torch.where(neg_indices == torch.arange(1, L, device=z.device),
                                      torch.clamp(neg_indices + 1, max=L - 1), neg_indices)
            d_neg = torch.norm(z[:-1] - z[neg_indices], p=2, dim=-1)  # (L-1,)
            hinge = torch.clamp(d_pos - d_neg + margin, min=0.0)
            losses.append(hinge.mean())
        if not losses:
            return torch.tensor(0.0, device=z_list[0].device, requires_grad=True)
        return torch.stack(losses).mean()


AutoConfig.register("mmldm_vae", MMLDMVAEConfig)
AutoModel.register(MMLDMVAEConfig, MMLDMVAEModel)
