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
- Text encoder: standalone TransformerBlock for Stage 2 conditioning
"""

from __future__ import annotations

import math
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
    latent_dists: Optional[tuple[list[DiagonalGaussianDistribution], list[DiagonalGaussianDistribution]]] = None
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
# Standalone text encoder modules
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        return freqs.to(dtype)


def _apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_rot = x.float()
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    d = x_rot.shape[-1] // 2
    x1, x2 = x_rot[..., :d], x_rot[..., d:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TextTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, ffn_dim: int,
                 dropout: float = 0.0, layer_norm_eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.attn_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.qkv_proj = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=True)
        self.q_norm = nn.LayerNorm(head_dim, eps=layer_norm_eps)
        self.k_norm = nn.LayerNorm(head_dim, eps=layer_norm_eps)
        self.rope = RotaryEmbedding(head_dim, theta=10000)
        self.ffn_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.ffn = SwiGLUFFN(dim, ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        freqs = self.rope(L, device=x.device, dtype=q.dtype)
        q, k = _apply_rotary_emb(q, freqs), _apply_rotary_emb(k, freqs)

        d_head = q.shape[-1]
        attn = q.mul(1.0 / (d_head ** 0.5)) @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask.to(attn.dtype)
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, L, -1)
        out = self.out_proj(out)
        x = x + out
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# MMLDM VAE Model v2 — Spectral Dual-Latent
# ---------------------------------------------------------------------------


class MMLDMVAEModel(PreTrainedModel):
    """Spectral Dual-Latent VAE for MMLDM.

    Innovation A: Splits latent into trend (low-freq) and residual (high-freq)
    subspaces with FFT decomposition and separate encoders.

    Engineering: Spectral reconstruction loss + latent standardization.
    Innovation C: Temporal Contrastive Latent Regularization (TCLR).
    """

    config_class = MMLDMVAEConfig
    base_model_prefix = "mmldm_vae"

    def __init__(self, config: MMLDMVAEConfig):
        super().__init__(config)
        self.config = config
        self.use_variation = config.use_variation

        # Innovation A: Dual-latent dimensions
        self.trend_dim = config.latent_dim // 2
        self.residual_dim = config.latent_dim - self.trend_dim

        # Dual Conv1d Encoders
        self.trend_encoder = ConvEncoder(config.ts_channels, config.dim, config.num_conv_layers)
        self.residual_encoder = ConvEncoder(config.ts_channels, config.dim, config.num_conv_layers)

        # Dual latent projections
        if config.use_variation:
            self.trend_proj = nn.Conv1d(config.dim, self.trend_dim * 2, kernel_size=1)
            self.residual_proj = nn.Conv1d(config.dim, self.residual_dim * 2, kernel_size=1)
        else:
            self.trend_proj = nn.Conv1d(config.dim, self.trend_dim, kernel_size=1)
            self.residual_proj = nn.Conv1d(config.dim, self.residual_dim, kernel_size=1)

        # Decoder (merged latent)
        self.decoder_in_layer = nn.Conv1d(config.latent_dim, config.dim, kernel_size=1)
        self.decoder = ConvDecoder(config.ts_channels, config.dim, config.decoder_num_blocks)

        # Standalone Text Encoder
        self.text_proj = nn.Linear(config.text_dim, config.dim)
        self.text_encoder_blocks = nn.ModuleList([
            TextTransformerBlock(dim=config.dim, num_heads=config.num_heads, head_dim=config.head_dim,
                                 ffn_dim=config.ffn_dim, dropout=config.dropout, layer_norm_eps=config.layer_norm_eps)
            for _ in range(config.encoder_num_blocks)
        ])
        self.text_final_layer = nn.Linear(config.dim, config.latent_dim)
        self.text_final_norm = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)

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
        """Encode with FFT decomposition -> trend + residual dual latents.

        Returns distributions only — callers should sample once via:
            trend_dists, residual_dists = output.latent_dists
            z_list = [torch.cat([td.sample(), rd.sample()], dim=-1)
                      for td, rd in zip(trend_dists, residual_dists)]
        """
        trend_params, residual_params = [], []

        for ot in ot_list:
            x = ot.unsqueeze(0).permute(0, 2, 1)  # (1, C, L)
            x_low, x_high = fft_decompose(x, cutoff_ratio=self.config.fft_cutoff_ratio)

            h_trend = self.trend_encoder(x_low)
            h_residual = self.residual_encoder(x_high)

            p_trend = self.trend_proj(h_trend).permute(0, 2, 1).squeeze(0)
            p_residual = self.residual_proj(h_residual).permute(0, 2, 1).squeeze(0)

            trend_params.append(p_trend)
            residual_params.append(p_residual)

        trend_dists = [DiagonalGaussianDistribution(p) for p in trend_params]
        residual_dists = [DiagonalGaussianDistribution(p) for p in residual_params]

        return TextVAEEncoderOutput(latents_list=[], latent_dists=(trend_dists, residual_dists))

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

    def encode_text_condition(self, text_embs: torch.Tensor) -> torch.Tensor:
        """Encode text embeddings for Stage 2 conditioning."""
        x = self.text_proj(text_embs).unsqueeze(0)
        for block in self.text_encoder_blocks:
            x = block(x, attn_mask=None)
        return self.text_final_norm(self.text_final_layer(x.squeeze(0)))

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
        trend_dists, residual_dists = enc_output.latent_dists

        trend_samples = [d.sample() for d in trend_dists]
        residual_samples = [d.sample() for d in residual_dists]
        z_list = [torch.cat([t, r], dim=-1) for t, r in zip(trend_samples, residual_samples)]

        z = torch.cat(z_list, dim=0)
        ts_shape = torch.tensor([[z_i.shape[0]] for z_i in z_list], dtype=torch.long, device=z.device)
        recon = self.decode(z, ts_shape)

        result = {"recon": recon, "latent_dists": (trend_dists, residual_dists), "latents": z_list}

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
