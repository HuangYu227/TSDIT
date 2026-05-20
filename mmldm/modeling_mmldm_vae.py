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

"""MMLDM Multimodal VAE — Stage 1 model.

Implements the shared multimodal latent space with:
- TS Encoder: time series → latent tokens
- Text Encoder: text embedding → latent tokens
- Joint Encoder: MMDiT-style JointAttention fusion
- Block-Causal Decoder: latent → time series reconstruction

The encoder produces ``q_phi(z_0 | x, c)`` and the decoder produces
``p_theta(x | z_0)``.  Stage 1 training minimizes::

    L_VAE = L_recon + beta * KL + lambda_mask * L_mask
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .attention_utils import create_multimodal_joint_mask
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
    text_latents: Optional[torch.Tensor] = None  # (L_text_total, latent_dim)


# ---------------------------------------------------------------------------
# Rotary Embedding
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding for the VAE blocks."""

    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        return freqs.to(dtype)  # (L, dim/2)


def _apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to input tensor.

    Args:
        x: ``(B, H, L, D)`` where ``D = head_dim``.
        freqs: ``(L, D/2)`` sinusoidal frequencies.

    Returns:
        Rotated tensor of same shape as ``x``.
    """
    x_rot = x.float()
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, L, D/2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, L, D/2)
    d = x_rot.shape[-1] // 2
    x1, x2 = x_rot[..., :d], x_rot[..., d:]  # each (B, H, L, D/2)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1).to(x.dtype)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Transformer Block (used by TS encoder, text encoder, and decoder)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional RoPE and KV cache.

    Adapted from Cola-DLM's TextVAEBlock, simplified for MMLDM.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_bias: bool = False,
        post_norm: bool = True,
        rope_theta: int = 10000,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.post_norm = post_norm

        inner_dim = num_heads * head_dim

        # Attention
        self.attn_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.qkv_proj = nn.Linear(dim, inner_dim * 3, bias=qk_bias)
        self.out_proj = nn.Linear(inner_dim, dim, bias=True)

        if qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=layer_norm_eps)
            self.k_norm = nn.LayerNorm(head_dim, eps=layer_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rope = RotaryEmbedding(head_dim, theta=rope_theta)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        # FFN
        self.ffn_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.ffn = SwiGLUFFN(dim, ffn_dim, dropout=dropout)

        # KV cache
        self._k_cache: Optional[list[torch.Tensor]] = None
        self._v_cache: Optional[list[torch.Tensor]] = None

    def set_kv_cache(self, flag: bool) -> None:
        self._k_cache = None if not flag else self._k_cache
        self._v_cache = None if not flag else self._v_cache

    def _slow_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        d_head = q.shape[-1]
        scale = 1.0 / (d_head ** 0.5)
        attn = q.mul(scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask.to(attn.dtype)
        attn_weight = attn.softmax(dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        return attn_weight @ v

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        update_kv: bool = False,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(1, L, d)`` input tensor.
            attn_mask: ``(1, 1, L_q, L_k)`` additive mask.
            update_kv: if True, append K/V to cache.
            use_kv_cache: if True, use cached K/V.

        Returns:
            ``(1, L, d)`` output tensor.
        """
        B, L, _ = x.shape

        # Attention
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, L, H, D)
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        freqs = self.rope(L, device=x.device, dtype=q.dtype)
        q = _apply_rotary_emb(q, freqs)
        k = _apply_rotary_emb(k, freqs)

        # KV cache
        if update_kv:
            if self._k_cache is None:
                self._k_cache = [k.clone()]
                self._v_cache = [v.clone()]
            else:
                self._k_cache.append(k.clone())
                self._v_cache.append(v.clone())
            # Use full cached K/V
            full_k = torch.cat(self._k_cache, dim=2)
            full_v = torch.cat(self._v_cache, dim=2)
        elif use_kv_cache and self._k_cache is not None:
            full_k = torch.cat(self._k_cache + [k], dim=2)
            full_v = torch.cat(self._v_cache + [v], dim=2)
        else:
            full_k = k
            full_v = v

        # Attention
        out = self._slow_attn(q, full_k, full_v, attn_mask)
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = self.out_proj(out)

        if self.post_norm:
            out = self.attn_norm(out)

        x = x + out

        # FFN
        h = self.ffn_norm(x)
        h = self.ffn(h)
        if self.post_norm:
            h = self.ffn_norm(h)
        x = x + h

        return x


# ---------------------------------------------------------------------------
# Joint Attention (MMDiT-style)
# ---------------------------------------------------------------------------


class JointAttention(nn.Module):
    """MMDiT-style joint attention across two modalities.

    TS tokens and text tokens are projected to Q/K/V independently,
    packed into a single sequence, attend together, then unpacked.
    """

    def __init__(
        self,
        ts_dim: int,
        text_dim: int,
        num_heads: int,
        head_dim: int,
        qk_bias: bool = False,
        qk_norm: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        # Per-modality QKV projections
        self.ts_to_qkv = nn.Linear(ts_dim, inner_dim * 3, bias=qk_bias)
        self.text_to_qkv = nn.Linear(text_dim, inner_dim * 3, bias=qk_bias)

        # Per-modality output projections
        self.ts_to_out = nn.Linear(inner_dim, ts_dim, bias=True)
        self.text_to_out = nn.Linear(inner_dim, text_dim, bias=True)

        # QK norm
        if qk_norm:
            self.ts_q_norm = nn.LayerNorm(head_dim, eps=layer_norm_eps)
            self.ts_k_norm = nn.LayerNorm(head_dim, eps=layer_norm_eps)
            self.text_q_norm = nn.LayerNorm(head_dim, eps=layer_norm_eps)
            self.text_k_norm = nn.LayerNorm(head_dim, eps=layer_norm_eps)
        else:
            self.ts_q_norm = self.ts_k_norm = None
            self.text_q_norm = self.text_k_norm = None

    def forward(
        self,
        ts_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ts_tokens: ``(1, L_ts, d_ts)``
            text_tokens: ``(1, L_text, d_text)``
            attn_mask: optional additive mask for the packed sequence.

        Returns:
            ``(ts_out, text_out)`` tensors of same shapes as inputs.
        """
        B = ts_tokens.shape[0]
        L_ts = ts_tokens.shape[1]
        L_text = text_tokens.shape[1]

        # Project to QKV
        ts_qkv = self.ts_to_qkv(ts_tokens).reshape(B, L_ts, 3, self.num_heads, self.head_dim)
        text_qkv = self.text_to_qkv(text_tokens).reshape(B, L_text, 3, self.num_heads, self.head_dim)

        ts_q, ts_k, ts_v = ts_qkv.unbind(2)   # (B, L, H, D)
        text_q, text_k, text_v = text_qkv.unbind(2)

        # Transpose to (B, H, L, D)
        ts_q = ts_q.transpose(1, 2)
        ts_k = ts_k.transpose(1, 2)
        ts_v = ts_v.transpose(1, 2)
        text_q = text_q.transpose(1, 2)
        text_k = text_k.transpose(1, 2)
        text_v = text_v.transpose(1, 2)

        # QK norm
        if self.ts_q_norm is not None:
            ts_q = self.ts_q_norm(ts_q)
            ts_k = self.ts_k_norm(ts_k)
            text_q = self.text_q_norm(text_q)
            text_k = self.text_k_norm(text_k)

        # Pack all modalities into single sequence dimension
        # Use einops pack: (B, H, *, D) where * is the sequence dimension
        all_q, packed_shape_q = pack([ts_q, text_q], "b h * d")
        all_k, packed_shape_k = pack([ts_k, text_k], "b h * d")
        all_v, packed_shape_v = pack([ts_v, text_v], "b h * d")

        # Joint attention
        d_head = all_q.shape[-1]
        scale = 1.0 / (d_head ** 0.5)
        attn = all_q.mul(scale) @ all_k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask.to(attn.dtype)
        attn_weight = attn.softmax(dim=-1)
        outs = attn_weight @ all_v  # (B, H, L_ts+L_text, D)

        # Merge heads
        outs = outs.transpose(1, 2).reshape(B, L_ts + L_text, -1)

        # Unpack back to per-modality
        ts_out, text_out = unpack(outs, packed_shape_q, "b * d")

        # Per-modality output projection
        ts_out = self.ts_to_out(ts_out)
        text_out = self.text_to_out(text_out)

        return ts_out, text_out


# ---------------------------------------------------------------------------
# Joint Encoder Block
# ---------------------------------------------------------------------------


class JointEncoderBlock(nn.Module):
    """Joint encoder block with MMDiT-style attention.

    Each modality has its own LayerNorm and FFN, but they share the
    joint attention computation.
    """

    def __init__(
        self,
        ts_dim: int,
        text_dim: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        qk_bias: bool = False,
        qk_norm: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Attention norms
        self.ts_attn_norm = nn.LayerNorm(ts_dim, eps=layer_norm_eps)
        self.text_attn_norm = nn.LayerNorm(text_dim, eps=layer_norm_eps)

        # Joint attention
        self.joint_attn = JointAttention(
            ts_dim=ts_dim,
            text_dim=text_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm=qk_norm,
            layer_norm_eps=layer_norm_eps,
        )

        # FFN norms
        self.ts_ffn_norm = nn.LayerNorm(ts_dim, eps=layer_norm_eps)
        self.text_ffn_norm = nn.LayerNorm(text_dim, eps=layer_norm_eps)

        # Per-modality FFN
        self.ts_ffn = SwiGLUFFN(ts_dim, ffn_dim, dropout=dropout)
        self.text_ffn = SwiGLUFFN(text_dim, ffn_dim, dropout=dropout)

    def forward(
        self,
        ts_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention branch
        ts_normed = self.ts_attn_norm(ts_tokens)
        text_normed = self.text_attn_norm(text_tokens)

        ts_attn_out, text_attn_out = self.joint_attn(
            ts_normed, text_normed, attn_mask=attn_mask
        )

        ts_tokens = ts_tokens + ts_attn_out
        text_tokens = text_tokens + text_attn_out

        # FFN branch
        ts_tokens = ts_tokens + self.ts_ffn(self.ts_ffn_norm(ts_tokens))
        text_tokens = text_tokens + self.text_ffn(self.text_ffn_norm(text_tokens))

        return ts_tokens, text_tokens


# ---------------------------------------------------------------------------
# MMLDM VAE Model
# ---------------------------------------------------------------------------


class MMLDMVAEModel(PreTrainedModel):
    """Multimodal VAE for MMLDM Stage 1.

    Encodes time series and text into a shared latent space, then
    decodes from the latent back to time series.
    """

    config_class = MMLDMVAEConfig
    base_model_prefix = "mmldm_vae"

    def __init__(self, config: MMLDMVAEConfig):
        super().__init__(config)
        self.config = config
        self.block_causal = config.block_causal
        self.block_size = config.block_size
        self.use_variation = config.use_variation
        self.patch_size = config.patch_size

        # ---- TS Encoder ----
        self.ts_proj = nn.Linear(config.ts_channels, config.dim)
        self.ts_patch_embedder = nn.Conv1d(
            config.dim, config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.ts_encoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                qk_norm=config.qk_norm,
                qk_bias=config.qk_bias,
                post_norm=config.post_norm,
                rope_theta=config.rope_theta,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.encoder_num_blocks)
        ])

        # ---- Text Encoder ----
        self.text_proj = nn.Linear(config.text_dim, config.dim)
        self.text_encoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                qk_norm=config.qk_norm,
                qk_bias=config.qk_bias,
                post_norm=config.post_norm,
                rope_theta=config.rope_theta,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.encoder_num_blocks)
        ])

        # ---- Joint Encoder ----
        self.joint_encoder_blocks = nn.ModuleList([
            JointEncoderBlock(
                ts_dim=config.dim,
                text_dim=config.dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                qk_bias=config.qk_bias,
                qk_norm=config.qk_norm,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.joint_num_blocks)
        ])

        # ---- Latent projection ----
        if config.use_variation:
            self.final_layer = nn.Linear(config.dim, config.latent_dim * 2, bias=config.bias)
            self.final_norm = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)
        else:
            self.final_layer = nn.Linear(config.dim, config.latent_dim, bias=config.bias)
            self.final_norm = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)

        # Text-side projection to latent_dim (mirrors final_layer for TS)
        self.text_final_layer = nn.Linear(config.dim, config.latent_dim, bias=config.bias)
        self.text_final_norm = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)

        # ---- Decoder ----
        self.decoder_in_layer = nn.Linear(config.latent_dim, config.dim, bias=config.bias)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                qk_norm=config.qk_norm,
                qk_bias=config.qk_bias,
                post_norm=config.post_norm,
                rope_theta=config.rope_theta,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.decoder_num_blocks)
        ])
        self.decoder_unpatch = nn.Linear(config.dim, config.dim * config.patch_size)
        self.decoder_norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.decoder_out = nn.Linear(config.dim, config.ts_channels, bias=config.bias)

        self.post_init()

    def _init_weights(self, module):
        std = self.config.init_std
        cutoff = self.config.init_cutoff_factor
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=std, a=-cutoff * std, b=cutoff * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ---------------------------------------------------------------
    # Encode
    # ---------------------------------------------------------------

    def _encode_ts_per_sample(
        self, ot_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Run TS embedding + patch Conv1d one sample at a time."""
        out_list = []
        for ot in ot_list:
            # ot: (L, C)
            x = self.ts_proj(ot.unsqueeze(0))  # (1, L, d)
            x = x.permute(0, 2, 1)  # (1, d, L)
            x = self.ts_patch_embedder(x)  # (1, d, n)
            x = x.permute(0, 2, 1).squeeze(0)  # (n, d)
            out_list.append(x)
        return out_list

    def _encode_text_per_sample(
        self, text_embs: torch.Tensor
    ) -> list[torch.Tensor]:
        """Encode text embeddings to latent tokens."""
        # text_embs: (B, text_dim)
        B = text_embs.shape[0]
        x = self.text_proj(text_embs)  # (B, d)
        return [x[i : i + 1] for i in range(B)]  # list of (1, d)

    def encode(
        self,
        ot_list: list[torch.Tensor],
        text_embs: torch.Tensor,
    ) -> TextVAEEncoderOutput:
        """Encode time series and text into shared latent space.

        Args:
            ot_list: list of ``(L_i, C)`` time series tensors.
            text_embs: ``(B, text_dim)`` text embedding tensor.

        Returns:
            :class:`TextVAEEncoderOutput` with per-sample latents.
        """
        B = len(ot_list)

        # TS encoding
        ts_per_sample = self._encode_ts_per_sample(ot_list)
        ts_shape = torch.tensor(
            [[x.shape[0]] for x in ts_per_sample],
            dtype=torch.long,
            device=ts_per_sample[0].device,
        )

        # Text encoding
        text_per_sample = self._encode_text_per_sample(text_embs)
        text_shape = torch.tensor(
            [[x.shape[0]] for x in text_per_sample],
            dtype=torch.long,
            device=text_per_sample[0].device,
        )

        # Run through individual encoder blocks
        ts_x = torch.cat(ts_per_sample, dim=0).unsqueeze(0)  # (1, L_ts_total, d)
        for block in self.ts_encoder_blocks:
            ts_x = block(ts_x)

        text_x = torch.cat(text_per_sample, dim=0).unsqueeze(0)  # (1, L_text_total, d)
        for block in self.text_encoder_blocks:
            text_x = block(text_x)

        # Joint encoder — no block-causal mask needed for encoding
        for block in self.joint_encoder_blocks:
            ts_x, text_x = block(ts_x, text_x)

        # Combine TS and text latents for the final projection
        ts_latents = ts_x.squeeze(0)  # (L_ts_total, d)
        text_latents_raw = text_x.squeeze(0)  # (L_text_total, d)

        # Project TS latents to latent space
        ts_latents = self.final_layer(ts_latents)
        if self.use_variation:
            mean, logvar = torch.chunk(ts_latents, 2, dim=-1)
            mean = self.final_norm(mean)
            ts_latents = torch.cat((mean, logvar), dim=-1)

        # Project text latents to the same latent_dim space
        text_latents = self.text_final_norm(
            self.text_final_layer(text_latents_raw),
        )  # (L_text_total, latent_dim)

        # Split back to per-sample latents
        split_sizes = ts_shape.flatten().tolist()
        per_sample_latents = list(ts_latents.split(split_sizes, dim=0))

        latent_dists: Optional[list[DiagonalGaussianDistribution]] = None
        if self.use_variation:
            latent_dists = [DiagonalGaussianDistribution(p) for p in per_sample_latents]
            latents_mode = [d.mode() for d in latent_dists]
        else:
            latents_mode = per_sample_latents

        return TextVAEEncoderOutput(
            latents_list=latents_mode,
            latent_dists=latent_dists,
            text_latents=text_latents,
        )

    # ---------------------------------------------------------------
    # Decode
    # ---------------------------------------------------------------

    def decode(
        self,
        z: torch.Tensor,
        txt_shape: torch.LongTensor,
        txt_q_shape: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latents into time series.

        Args:
            z: ``(L_total, latent_dim)`` latent tensor.
            txt_shape: ``(B, 1)`` K lengths.
            txt_q_shape: ``(B, 1)`` Q lengths.
            attn_mask: optional block-causal mask.

        Returns:
            ``(1, L_total * patch_size, ts_channels)`` output tensor.
        """
        z = self.decoder_in_layer(z)
        z = z.unsqueeze(0)  # (1, L_total, d)

        for block in self.decoder_blocks:
            z = block(z, attn_mask=attn_mask)

        z = self.decoder_unpatch(z)  # (1, L, d * patch_size)
        z = rearrange(z, "b l (c ps) -> b (l ps) c", ps=self.patch_size)
        z = self.decoder_norm(z)
        z = self.decoder_out(z)
        return z

    # ---------------------------------------------------------------
    # Forward (for training)
    # ---------------------------------------------------------------

    def forward(
        self,
        ot_list: list[torch.Tensor],
        text_embs: torch.Tensor,
    ) -> dict:
        """Full forward pass: encode → sample → decode.

        Returns dict with keys: ``recon``, ``latent_dists``, ``latents``.
        """
        enc_output = self.encode(ot_list, text_embs)

        # Sample from posterior
        if self.use_variation:
            z_list = [d.sample() for d in enc_output.latent_dists]
        else:
            z_list = enc_output.latents_list

        # Build txt_shape for decoder
        z = torch.cat(z_list, dim=0)  # (L_total, latent_dim)
        txt_shape = torch.tensor(
            [[z_i.shape[0]] for z_i in z_list],
            dtype=torch.long,
            device=z.device,
        )

        # Build block-causal mask for decoder
        attn_mask = None
        if self.block_causal:
            lengths = txt_shape.flatten().tolist()
            block_sizes = []
            for l in lengths:
                n_full = l // self.block_size
                sizes = [self.block_size] * n_full
                remainder = l - n_full * self.block_size
                if remainder > 0:
                    sizes.append(remainder)
                block_sizes.append(sizes if sizes else [l])

            text_shape_zero = torch.zeros_like(txt_shape)
            attn_mask = create_multimodal_joint_mask(
                ts_shape=txt_shape,
                text_shape=text_shape_zero,
                block_sizes=block_sizes,
                dtype=torch.bfloat16 if z.is_cuda else z.dtype,
                device=z.device,
            )

        # Decode
        recon = self.decode(z, txt_shape, txt_shape, attn_mask=attn_mask)

        return {
            "recon": recon,  # (1, L_total * patch_size, ts_channels)
            "latent_dists": enc_output.latent_dists,
            "latents": z_list,
        }


# Register with HuggingFace
AutoConfig.register("mmldm_vae", MMLDMVAEConfig)
AutoModel.register(MMLDMVAEConfig, MMLDMVAEModel)
