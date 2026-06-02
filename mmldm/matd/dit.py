"""Causal-aware Patch Latent Diffusion Transformer for MATD.

The public class ``T2PDenoiser`` preserves the existing call shape and adds an
optional ``causal_feat`` argument:

    eps = denoiser(z_t, t, text_tokens, meta, pooled_text, causal_feat=None)

Design upgrades:
- adaLN-Zero style modulation in every residual path;
- relative temporal attention driven by adaptive patch metadata;
- optional density and causal attention biases;
- explicit causal feature injection into both token states and global condition;
- supports epsilon and v prediction modes by returning the requested target type.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_sinusoidal_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Standard sinusoidal timestep embedding."""
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be (B,), got {tuple(timesteps.shape)}")
    half = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    freq = torch.exp(exponent)
    emb = timesteps.float()[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class RelativeTemporalSelfAttention(nn.Module):
    """Patch self-attention with metadata-derived relative time and density bias."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        num_buckets: int = 32,
        max_rel_dist: float = 1.0,
        use_density_bias: bool = True,
        use_causal_bias: bool = True,
        dropout: float = 0.0,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.num_buckets = num_buckets
        self.max_rel_dist = max_rel_dist
        self.use_density_bias = use_density_bias
        self.use_causal_bias = use_causal_bias

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        self.rel_time_bias = nn.Embedding(num_buckets, n_heads)
        if use_density_bias:
            self.density_proj = nn.Linear(1, n_heads)
        if use_causal_bias:
            self.causal_bias_proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, n_heads))
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
        self.attn_dropout = nn.Dropout(dropout)

    def _bucketize_centers(self, centers: torch.Tensor) -> torch.Tensor:
        delta = centers[:, :, None] - centers[:, None, :]
        bucket_size = 2.0 * self.max_rel_dist / self.num_buckets
        bucket = ((delta + self.max_rel_dist) / bucket_size).long().clamp(0, self.num_buckets - 1)
        return self.rel_time_bias(bucket).permute(0, 3, 1, 2)

    def forward(
        self,
        x: torch.Tensor,
        centers: torch.Tensor,
        log_density: Optional[torch.Tensor] = None,
        causal_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, K, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + self._bucketize_centers(centers)
        if self.use_density_bias and log_density is not None:
            den = self.density_proj(log_density.unsqueeze(-1)).permute(0, 2, 1).unsqueeze(-1)
            attn = attn + den
        if self.use_causal_bias and causal_feat is not None:
            cb = self.causal_bias_proj(causal_feat).permute(0, 2, 1).unsqueeze(-1)
            attn = attn + cb
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, K, self.dim)
        return self.to_out(out)


class TextCrossAttention(nn.Module):
    """Cross-attention from patch tokens to text tokens."""

    def __init__(self, dim: int, text_dim: int, n_heads: int = 8, dropout: float = 0.0, qk_norm: bool = False, norm_eps: float = 1e-5) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(text_dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, text_tokens: torch.Tensor, text_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, _ = x.shape
        N = text_tokens.shape[1]
        q = self.to_q(x).reshape(B, K, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.to_kv(text_tokens).reshape(B, N, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if text_padding_mask is not None:
            attn = attn.masked_fill(text_padding_mask[:, None, None, :], torch.finfo(attn.dtype).min)
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, K, self.dim)
        return self.to_out(out)


class TextTemporalDiTBlock(nn.Module):
    """Pre-norm DiT block with adaLN-Zero gates for all residual paths."""

    def __init__(
        self,
        dim: int,
        text_dim: int,
        n_heads: int = 8,
        mlp_expand: int = 4,
        num_buckets: int = 32,
        max_rel_dist: float = 1.0,
        use_density_bias: bool = True,
        use_causal_bias: bool = True,
        dropout: float = 0.0,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.ada_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        _zero_module(self.ada_proj[1])
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.self_attn = RelativeTemporalSelfAttention(dim, n_heads, num_buckets, max_rel_dist, use_density_bias, use_causal_bias, dropout, qk_norm, norm_eps)
        self.cross_attn = TextCrossAttention(dim, text_dim, n_heads, dropout, qk_norm, norm_eps)
        hidden = dim * mlp_expand
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        cond: torch.Tensor,
        centers: torch.Tensor,
        log_density: Optional[torch.Tensor] = None,
        causal_feat: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        params = self.ada_proj(cond).chunk(9, dim=-1)
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = params
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.self_attn(h, centers=centers, log_density=log_density, causal_feat=causal_feat)
        x = x + gate1.unsqueeze(1) * h
        h = modulate(self.norm2(x), shift2, scale2)
        h = self.cross_attn(h, text_tokens, text_padding_mask=text_padding_mask)
        x = x + gate2.unsqueeze(1) * h
        h = modulate(self.norm3(x), shift3, scale3)
        x = x + gate3.unsqueeze(1) * self.mlp(h)
        return x


class T2PDenoiser(nn.Module):
    """Text-to-patch DiT denoiser with optional causal conditioning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_expand: int = 4,
        num_buckets: int = 32,
        max_rel_dist: float = 1.0,
        use_density_bias: bool = True,
        use_causal_bias: bool = True,
        dropout: float = 0.0,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
        sinusoidal_dim: int = 256,
        prediction_type: str = "epsilon",
    ) -> None:
        super().__init__()
        if prediction_type not in ("epsilon", "eps", "v"):
            raise ValueError("prediction_type must be 'epsilon'/'eps' or 'v'")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.prediction_type = "epsilon" if prediction_type == "eps" else prediction_type
        self.sinusoidal_dim = sinusoidal_dim
        self.timestep_mlp = nn.Sequential(nn.Linear(sinusoidal_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.causal_token_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.causal_pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_mlp = nn.Sequential(nn.Linear(hidden_dim + text_dim + hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([
            TextTemporalDiTBlock(hidden_dim, text_dim, n_heads, mlp_expand, num_buckets, max_rel_dist, use_density_bias, use_causal_bias, dropout, qk_norm, norm_eps)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        _zero_module(self.output_proj)

    def _timestep_embed(self, t: torch.Tensor) -> torch.Tensor:
        return self.timestep_mlp(_get_sinusoidal_embedding(t.float(), self.sinusoidal_dim))

    def timestep_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Public wrapper for timestep embedding."""
        return self._timestep_embed(t)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        text_tokens: torch.Tensor,
        meta: torch.Tensor,
        pooled_text: torch.Tensor,
        causal_feat: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        centers = meta[..., 2].clamp(0.0, 1.0)
        log_density = meta[..., 7] if meta.shape[-1] > 7 else None
        timestep_emb = self._timestep_embed(t)
        x = self.input_proj(z_t)
        if causal_feat is not None:
            causal_tokens = self.causal_token_proj(causal_feat)
            x = x + causal_tokens
            causal_pool = causal_tokens.mean(dim=1)
        else:
            causal_tokens = None
            causal_pool = torch.zeros_like(timestep_emb)
        cond = self.global_cond_mlp(torch.cat([timestep_emb, pooled_text, self.causal_pool_proj(causal_pool)], dim=-1))
        for block in self.blocks:
            x = block(x, text_tokens, cond, centers, log_density, causal_tokens, text_padding_mask)
        return self.output_proj(self.final_norm(x))
