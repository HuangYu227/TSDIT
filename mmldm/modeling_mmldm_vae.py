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

"""MMLDM VAE — Stage 1 model with Conv1d encoder/decoder.

Implements a per-sample Conv1d VAE for time series:
- Encoder: Conv1d(k=3,s=1) + Residual stack → latent mean/logvar
- Decoder: Conv1d(k=3,s=1) + Residual stack → reconstructed TS

Also provides a standalone text encoder (encode_text_condition) for
Stage 2 conditioning, using TransformerBlock with RoPE attention.
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
    """Diagonal Gaussian posterior for the VAE latent space.

    Splits the encoder output into mean and logvar, provides sampling
    via the reparameterization trick.
    """

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
        """KL divergence against N(0, I) or another Gaussian."""
        if other_mean is None:
            other_mean = torch.zeros_like(self.mean)
        return 0.5 * (
            self.mean.pow(2) + self.var - 1.0 - self.logvar
        ).sum(dim=-1).mean()


@dataclass
class TextVAEEncoderOutput:
    latents_list: list[torch.Tensor]
    latent_dists: Optional[list[DiagonalGaussianDistribution]] = None
    text_latents: Optional[torch.Tensor] = None  # (B, latent_dim)


# ---------------------------------------------------------------------------
# Conv1d building blocks
# ---------------------------------------------------------------------------


class ConvResidual(nn.Module):
    """Conv1d(k=3,p=1) -> ReLU -> Conv1d(k=1) + skip connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


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
        # x: (B, in_channels, L)
        x = self.proj_in(x)
        x = self.stack(x)
        return x  # (B, dim, L)


class ConvDecoder(nn.Module):
    """Conv1d decoder: preserves sequence length (stride=1, k=3, p=1)."""

    def __init__(self, out_channels: int, dim: int, num_layers: int):
        super().__init__()
        self.proj_in = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.stack = ConvResidualStack(dim, num_layers)
        self.proj_out = nn.Conv1d(dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, L)
        x = self.proj_in(x)
        x = self.stack(x)
        x = self.proj_out(x)
        return x  # (B, out_channels, L)


# ---------------------------------------------------------------------------
# Standalone text encoder modules (for encode_text_condition)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding."""

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
    """Apply rotary embedding. x: (B, H, L, D), freqs: (L, D/2)."""
    x_rot = x.float()
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    d = x_rot.shape[-1] // 2
    x1, x2 = x_rot[..., :d], x_rot[..., d:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TextTransformerBlock(nn.Module):
    """Pre-norm Transformer block for text-only encoding (no KV cache)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        freqs = self.rope(L, device=x.device, dtype=q.dtype)
        q = _apply_rotary_emb(q, freqs)
        k = _apply_rotary_emb(k, freqs)

        d_head = q.shape[-1]
        scale = 1.0 / (d_head ** 0.5)
        attn = q.mul(scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask.to(attn.dtype)
        attn_weight = attn.softmax(dim=-1)
        out = attn_weight @ v
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = self.out_proj(out)

        x = x + out
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


# ---------------------------------------------------------------------------
# MMLDM VAE Model
# ---------------------------------------------------------------------------


class MMLDMVAEModel(PreTrainedModel):
    """Conv1d-based VAE for MMLDM Stage 1.

    Encodes time series into a continuous latent space via per-sample
    Conv1d encoding, then decodes back to time series via Conv1d decoding.

    Text encoding is handled separately by ``encode_text_condition()``
    for Stage 2 conditioning — it does NOT participate in VAE encode/decode.
    """

    config_class = MMLDMVAEConfig
    base_model_prefix = "mmldm_vae"

    def __init__(self, config: MMLDMVAEConfig):
        super().__init__(config)
        self.config = config
        self.use_variation = config.use_variation

        # ---- Conv1d TS Encoder ----
        self.encoder = ConvEncoder(config.ts_channels, config.dim, config.num_conv_layers)

        # ---- Latent projection (Conv1d k=1: dim → latent_dim*2) ----
        if config.use_variation:
            self.final_layer = nn.Conv1d(config.dim, config.latent_dim * 2, kernel_size=1)
        else:
            self.final_layer = nn.Conv1d(config.dim, config.latent_dim, kernel_size=1)

        # ---- Conv1d TS Decoder ----
        self.decoder_in_layer = nn.Conv1d(config.latent_dim, config.dim, kernel_size=1)
        self.decoder = ConvDecoder(config.ts_channels, config.dim, config.decoder_num_blocks)

        # ---- Standalone Text Encoder (for Stage 2 conditioning) ----
        self.text_proj = nn.Linear(config.text_dim, config.dim)
        self.text_encoder_blocks = nn.ModuleList([
            TextTransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.encoder_num_blocks)
        ])
        self.text_final_layer = nn.Linear(config.dim, config.latent_dim)
        self.text_final_norm = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)

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

    # ---------------------------------------------------------------
    # Encode (TS only — no text)
    # ---------------------------------------------------------------

    def encode(self, ot_list: list[torch.Tensor]) -> TextVAEEncoderOutput:
        """Encode time series into latent space (per-sample Conv1d).

        Args:
            ot_list: list of (L_i, C) time series tensors.

        Returns:
            TextVAEEncoderOutput with per-sample latents and distributions.
        """
        per_sample_params = []
        for ot in ot_list:
            # ot: (L, C) → (1, C, L)
            x = ot.unsqueeze(0).permute(0, 2, 1)
            x = self.encoder(x)  # (1, dim, L)
            x = self.final_layer(x)  # (1, latent_dim*2, L)
            x = x.permute(0, 2, 1).squeeze(0)  # (L, latent_dim*2)
            per_sample_params.append(x)

        latent_dists: Optional[list[DiagonalGaussianDistribution]] = None
        if self.use_variation:
            latent_dists = [DiagonalGaussianDistribution(p) for p in per_sample_params]
            latents_mode = [d.mode() for d in latent_dists]
        else:
            latents_mode = per_sample_params

        return TextVAEEncoderOutput(
            latents_list=latents_mode,
            latent_dists=latent_dists,
        )

    # ---------------------------------------------------------------
    # Decode (TS only — no text, no block-causal)
    # ---------------------------------------------------------------

    def decode(self, z: torch.Tensor, ts_shape: torch.LongTensor) -> torch.Tensor:
        """Decode latents into time series (per-sample Conv1d).

        Args:
            z: (L_total, latent_dim) flat latent tensor.
            ts_shape: (B, 1) per-sample token counts.

        Returns:
            (1, L_total, ts_channels) reconstructed time series.
        """
        lengths = ts_shape.flatten().tolist()
        z_list = z.split(lengths, dim=0)
        out_list = []
        for z_i in z_list:
            # z_i: (L_i, latent_dim) → (1, latent_dim, L_i)
            x = z_i.unsqueeze(0).permute(0, 2, 1)
            x = self.decoder_in_layer(x)  # (1, dim, L_i)
            x = self.decoder(x)  # (1, C, L_i)
            x = x.permute(0, 2, 1).squeeze(0)  # (L_i, C)
            out_list.append(x)
        recon = torch.cat(out_list, dim=0)  # (L_total, C)
        return recon.unsqueeze(0)  # (1, L_total, C)

    # ---------------------------------------------------------------
    # Text condition encoding (for Stage 2 — no TS leakage)
    # ---------------------------------------------------------------

    def encode_text_condition(self, text_embs: torch.Tensor) -> torch.Tensor:
        """Encode text embeddings without seeing the target time series.

        Stage 2 and inference use this path for conditioning. It runs
        standalone TransformerBlocks over per-sample text tokens.

        Args:
            text_embs: (B, text_dim) raw text embeddings.

        Returns:
            (B, latent_dim) text latent tokens (one per sample).
        """
        B = text_embs.shape[0]
        x = self.text_proj(text_embs)  # (B, dim)
        x = x.unsqueeze(0)  # (1, B, dim) — each sample is one "token"

        # Bidirectional attention across the batch of text tokens
        for block in self.text_encoder_blocks:
            x = block(x, attn_mask=None)

        x = self.text_final_norm(self.text_final_layer(x.squeeze(0)))  # (B, latent_dim)
        return x

    # ---------------------------------------------------------------
    # Forward (for Stage 1 training)
    # ---------------------------------------------------------------

    def forward(self, ot_list: list[torch.Tensor]) -> dict:
        """Full forward pass: encode → sample → decode.

        Args:
            ot_list: list of (L_i, C) time series tensors.

        Returns:
            dict with keys: recon, latent_dists, latents.
        """
        enc_output = self.encode(ot_list)

        # Sample from posterior
        if self.use_variation:
            z_list = [d.sample() for d in enc_output.latent_dists]
        else:
            z_list = enc_output.latents_list

        z = torch.cat(z_list, dim=0)  # (L_total, latent_dim)
        ts_shape = torch.tensor(
            [[z_i.shape[0]] for z_i in z_list],
            dtype=torch.long,
            device=z.device,
        )

        recon = self.decode(z, ts_shape)

        return {
            "recon": recon,  # (1, L_total, ts_channels)
            "latent_dists": enc_output.latent_dists,
            "latents": z_list,
        }


# Register with HuggingFace
AutoConfig.register("mmldm_vae", MMLDMVAEConfig)
AutoModel.register(MMLDMVAEConfig, MMLDMVAEModel)
