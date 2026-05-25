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

"""MMLDM Multimodal DiT — Stage 2 prior model.

Implements the block-causal multimodal diffusion transformer prior
``p_psi(z_0 | c)`` with Flow Matching and DCD dual-condition denoising.

The DiT learns the vector field ``v_psi(z_t, t; z_0^{(<b)}, c)`` under
the visible set ``V_b = {sg(z_0^{(<b)}), z_t^(b), c}``.

Stage 2 training minimizes::

    L = L_FM + gamma1 * L_DCD_mix + gamma2 * L_DCD_aux
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .attention_utils import create_dit_readonly_text_mask
from .configuration_mmldm import MMLDMDiTConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(hid_list: list[torch.Tensor]):
    """``List[Tensor(*, c)]`` → ``(Tensor(L_total, c), shape (B, 1))``."""
    shape = torch.stack([
        torch.tensor(x.shape[:-1], device=hid_list[0].device) for x in hid_list
    ])
    hid = torch.cat([x.flatten(0, -2) for x in hid_list])
    return hid, shape


def _unflatten(hid: torch.Tensor, hid_shape: torch.LongTensor):
    """Inverse of :func:`_flatten`."""
    hid_len = hid_shape.prod(-1)
    hid = hid.split(hid_len.tolist())
    return [x.unflatten(0, s.tolist()) for x, s in zip(hid, hid_shape)]


def _to_1d_timestep(
    timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=device, dtype=dtype)
    return timestep.to(device=device, dtype=dtype).flatten()


def _expand_ts_timestep(
    timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
    ts_shape: torch.LongTensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand scalar/per-sample/per-token timesteps to TS-token timesteps."""
    t = _to_1d_timestep(timestep, device=device, dtype=dtype)
    lengths = ts_shape.flatten().to(device=device)
    total = int(lengths.sum().item())
    batch = int(lengths.numel())

    if t.numel() == 1:
        return t.expand(total)
    if t.numel() == total:
        return t
    if t.numel() == batch:
        return torch.repeat_interleave(t, lengths)
    raise ValueError(
        f"timestep must be scalar, per-sample ({batch}), or per-token ({total}); "
        f"got {t.numel()}"
    )


def _expand_text_timestep(
    timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
    ts_shape: torch.LongTensor,
    text_shape: torch.LongTensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand conditioning timesteps to text-token timesteps."""
    t = _to_1d_timestep(timestep, device=device, dtype=dtype)
    ts_lengths = ts_shape.flatten().to(device=device)
    text_lengths = text_shape.flatten().to(device=device)
    total_ts = int(ts_lengths.sum().item())
    total_text = int(text_lengths.sum().item())
    batch = int(ts_lengths.numel())

    if total_text == 0:
        return torch.empty(0, device=device, dtype=dtype)
    if t.numel() == 1:
        per_sample = t.expand(batch)
    elif t.numel() == batch:
        per_sample = t
    elif t.numel() == total_text:
        return t
    elif t.numel() == total_ts:
        end_indices = ts_lengths.cumsum(0) - 1
        per_sample = t[end_indices]
    else:
        raise ValueError(
            f"text timestep must be scalar, per-sample ({batch}), per-text-token "
            f"({total_text}), or per-TS-token ({total_ts}); got {t.numel()}"
        )
    return torch.repeat_interleave(per_sample, text_lengths)


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------


def _get_sinusoidal_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (diffusers convention)."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / half_dim
    emb = torch.exp(exponent)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """Three-layer MLP for timestep conditioning."""

    def __init__(self, sinusoidal_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def initialize_weights(self):
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_hid.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)

    def forward(self, timestep, device, dtype):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]
        emb = _get_sinusoidal_embedding(timestep, self.sinusoidal_dim).to(dtype)
        emb = self.act(self.proj_in(emb))
        emb = self.act(self.proj_hid(emb))
        emb = self.proj_out(emb)
        return emb


# ---------------------------------------------------------------------------
# Patch In/Out
# ---------------------------------------------------------------------------


class PatchIn1D(nn.Module):
    """1-D patch embedding."""

    def __init__(self, in_channels: int, patch_size: int, dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size, dim)

    def forward(self, txt: torch.Tensor, txt_shape: torch.LongTensor):
        txt_shape_before_patchify = txt_shape
        if self.patch_size != 1:
            batch_list = _unflatten(txt, txt_shape)
            for i in range(len(batch_list)):
                batch_list[i] = rearrange(
                    batch_list[i], "(T t) c -> T (t c)", t=self.patch_size
                )
            txt, txt_shape = _flatten(batch_list)
        txt = self.proj(txt)
        return txt, txt_shape, txt_shape_before_patchify


class PatchOut1D(nn.Module):
    """1-D patch un-embedding."""

    def __init__(self, out_channels: int, patch_size: int, dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(dim, out_channels * patch_size)

    def forward(
        self,
        txt: torch.Tensor,
        txt_shape: torch.LongTensor,
        txt_shape_before_patchify: torch.LongTensor,
    ):
        txt = self.proj(txt)
        if self.patch_size != 1:
            batch_list = _unflatten(txt, txt_shape)
            for i in range(len(batch_list)):
                batch_list[i] = rearrange(
                    batch_list[i], "T (t c) -> (T t) c", t=self.patch_size
                )
            txt, txt_shape = _flatten(batch_list)
        return txt, txt_shape


# ---------------------------------------------------------------------------
# Adaptive LayerNorm (from MMDiT)
# ---------------------------------------------------------------------------


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm with time conditioning.

    Produces gamma/beta from the conditioning signal, applied after
    standard LayerNorm.  Gamma initialized to 1, beta to 0.
    """

    def __init__(self, dim: int, dim_cond: Optional[int] = None):
        super().__init__()
        has_cond = dim_cond is not None
        self.has_cond = has_cond
        self.ln = nn.LayerNorm(dim, elementwise_affine=not has_cond)

        if has_cond:
            cond_linear = nn.Linear(dim_cond, dim * 2)
            self.to_cond = nn.Sequential(
                nn.SiLU(),
                cond_linear,
            )
            nn.init.zeros_(cond_linear.weight)
            nn.init.constant_(cond_linear.bias[:dim], 1.0)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        x = self.ln(x)
        if self.has_cond and cond is not None:
            # x: (*, D), cond: (*, dim_cond) — may differ in leading dims.
            # Squeeze to 2D for MLP, then restore original shape.
            orig_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
            cond_flat = cond.reshape(-1, cond.shape[-1])
            if cond_flat.shape[0] == 1 and x_flat.shape[0] != 1:
                cond_flat = cond_flat.expand(x_flat.shape[0], -1)
            elif cond_flat.shape[0] != x_flat.shape[0]:
                raise ValueError(
                    f"AdaptiveLayerNorm condition length {cond_flat.shape[0]} "
                    f"does not match token length {x_flat.shape[0]}"
                )
            gamma, beta = self.to_cond(cond_flat).chunk(2, dim=-1)
            x_flat = x_flat * gamma + beta
            x = x_flat.reshape(orig_shape)
        return x


# ---------------------------------------------------------------------------
# MLP (GELU, tanh approximation)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim: int, expand_ratio: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = nn.Linear(dim * expand_ratio, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.act(self.proj_in(x)))


# ---------------------------------------------------------------------------
# Text-Guided Feature Modulation (TGFM)
# ---------------------------------------------------------------------------


class TextModulator(nn.Module):
    """Per-block text-guided feature modulation.

    Generates (scale, shift) from a text latent vector to modulate
    TS features.  This provides a direct text-conditioning pathway
    independent of the timestep-based adaLN used in the DiT blocks.

    The modulation is: output = scale * input + shift, where scale/shift
    are derived from the text latent via a small MLP.
    """

    def __init__(self, dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
        )
        # Zero-init for safe residual start
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, ts: torch.Tensor, text_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts: ``(B, L, dim)`` TS token features.
            text_latent: ``(B, text_dim)`` global text vector, or
                ``(B, L, text_dim)`` per-token expanded text latents.
        Returns:
            ``(B, L, dim)`` modulation output (scale * ts + shift).
        """
        if text_latent.ndim == 3:
            # Already per-token expanded: (B, L, D) — no unsqueeze needed
            params = self.proj(text_latent)  # (B, L, dim*2)
        else:
            # Global vector: (B, D) → (B, 1, D) for broadcast over L
            params = self.proj(text_latent.unsqueeze(1))  # (B, 1, dim*2)
        scale, shift = params.chunk(2, dim=-1)
        return scale * ts + shift


class MultiViewTextPooler(nn.Module):
    """Generate K orthogonal text views from a single SBERT embedding.

    Inspired by CaTSG's Environment Bank but driven by text semantics
    rather than data distribution. Each linear projection extracts a
    different semantic "view" of the same text (e.g., trend description,
    amplitude description, periodicity description).

    Orthogonal initialization ensures views are diverse by default.
    """

    def __init__(self, text_dim: int = 128, n_views: int = 4):
        super().__init__()
        self.n_views = n_views
        # Each view is a learnable rotation in text_dim space (no dimension change).
        # TextModulator in each block expects text_dim, so views must stay in that space.
        self.views = nn.ModuleList([
            nn.Linear(text_dim, text_dim) for _ in range(n_views)
        ])
        for v in self.views:
            nn.init.orthogonal_(v.weight)
            nn.init.zeros_(v.bias)

    def forward(self, text_emb: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            text_emb: ``(B, text_dim)`` raw SBERT embedding.
        Returns:
            List of K ``(B, latent_dim)`` text views.
        """
        return [v(text_emb) for v in self.views]


# ---------------------------------------------------------------------------
# Multimodal Joint Attention (for DiT)
# ---------------------------------------------------------------------------


class MultimodalJointAttention(nn.Module):
    """Joint attention for the DiT, with per-modality QKV and RoPE."""

    def __init__(
        self,
        ts_dim: int,
        text_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool = False,
        qk_norm_eps: float = 1e-5,
        rope_dim: int = 32,
    ):
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim

        # Per-modality QKV
        self.ts_to_qkv = nn.Linear(ts_dim, inner_dim * 3, bias=qk_bias)
        self.text_to_qkv = nn.Linear(text_dim, inner_dim * 3, bias=qk_bias)

        # Per-modality output
        self.ts_to_out = nn.Linear(inner_dim, ts_dim, bias=qk_bias)
        self.text_to_out = nn.Linear(inner_dim, text_dim, bias=qk_bias)

        # QK norm
        self.ts_q_norm = nn.LayerNorm(head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.ts_k_norm = nn.LayerNorm(head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.text_q_norm = nn.LayerNorm(head_dim, eps=qk_norm_eps, elementwise_affine=True)
        self.text_k_norm = nn.LayerNorm(head_dim, eps=qk_norm_eps, elementwise_affine=True)

        # RoPE (simplified sinusoidal)
        self._rope_cache: Optional[torch.Tensor] = None

    def _get_rope_freqs(self, seq_len: int, device: torch.device, dtype: torch.dtype, pos_offset: int = 0):
        half_dim = self.rope_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        t = torch.arange(pos_offset, pos_offset + seq_len, device=device, dtype=inv_freq.dtype)
        return torch.outer(t, inv_freq)  # (L, rope_dim//2)

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to first ``rope_dim`` channels."""
        d = self.rope_dim
        x_rope = x[..., :d].float()
        x_pass = x[..., d:]
        half = d // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        # Truncate freqs to half so cos/sin match x1, x2
        freqs_half = freqs[..., :half]
        cos = freqs_half.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs_half.sin().unsqueeze(0).unsqueeze(0)
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([torch.cat([out1, out2], dim=-1).to(x.dtype), x_pass], dim=-1)

    def forward(
        self,
        ts_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prefix_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        pos_offset: int = 0,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B = ts_tokens.shape[0]
        L_ts = ts_tokens.shape[1]
        L_text = text_tokens.shape[1]

        # Project to QKV
        ts_qkv_flat = self.ts_to_qkv(ts_tokens)
        if ts_qkv_flat.numel() != B * L_ts * 3 * self.heads * self.head_dim:
            raise RuntimeError(
                f"SHAPE MISMATCH: ts_tokens.shape={tuple(ts_tokens.shape)}, "
                f"text_tokens.shape={tuple(text_tokens.shape)}, "
                f"B={B}, L_ts={L_ts}, L_text={L_text}, "
                f"ts_to_qkv_out.shape={tuple(ts_qkv_flat.shape)}, "
                f"expected={B * L_ts * 3 * self.heads * self.head_dim}, "
                f"got={ts_qkv_flat.numel()}"
            )
        ts_qkv = ts_qkv_flat.reshape(B, L_ts, 3, self.heads, self.head_dim)
        text_qkv = self.text_to_qkv(text_tokens).reshape(B, L_text, 3, self.heads, self.head_dim)

        ts_q, ts_k, ts_v = ts_qkv.unbind(2)
        text_q, text_k, text_v = text_qkv.unbind(2)

        # Transpose to (B, H, L, D)
        ts_q, ts_k, ts_v = ts_q.transpose(1, 2), ts_k.transpose(1, 2), ts_v.transpose(1, 2)
        text_q, text_k, text_v = text_q.transpose(1, 2), text_k.transpose(1, 2), text_v.transpose(1, 2)

        # QK norm
        ts_q = self.ts_q_norm(ts_q)
        ts_k = self.ts_k_norm(ts_k)
        text_q = self.text_q_norm(text_q)
        text_k = self.text_k_norm(text_k)

        # Apply RoPE to TS only
        ts_freqs = self._get_rope_freqs(L_ts, ts_q.device, ts_q.dtype, pos_offset=pos_offset)
        ts_q = self._apply_rope(ts_q, ts_freqs)
        ts_k = self._apply_rope(ts_k, ts_freqs)

        # Concatenate prefix KV if provided (only TS KV is cached)
        if prefix_kv is not None:
            pre_ts_k, pre_ts_v = prefix_kv
            ts_k = torch.cat([pre_ts_k, ts_k], dim=2)
            ts_v = torch.cat([pre_ts_v, ts_v], dim=2)

        # Pack
        all_q, _ = pack([ts_q, text_q], "b h * d")
        all_k, _ = pack([ts_k, text_k], "b h * d")
        all_v, packed_shape_kv = pack([ts_v, text_v], "b h * d")

        # For unpacking output: q has (L_ts + L_text) tokens, kv may have more
        packed_shape_q = ([L_ts], [L_text])

        # Attention
        d_head = all_q.shape[-1]
        scale = 1.0 / (d_head ** 0.5)
        attn = all_q.mul(scale) @ all_k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask.to(attn.dtype)
        attn_weight = attn.softmax(dim=-1)
        outs = attn_weight @ all_v

        # Merge heads
        outs = outs.transpose(1, 2).reshape(B, L_ts + L_text, -1)

        # Unpack — use q shape since output matches query length
        ts_out, text_out = unpack(outs, packed_shape_q, "b * d")

        # Output projection
        ts_out = self.ts_to_out(ts_out)
        text_out = self.text_to_out(text_out)

        if return_kv:
            # Return current TS KV (without prefix) for caching; text KV is discarded
            current_kv = (ts_k[:, :, -L_ts:], ts_v[:, :, -L_ts:])
            return ts_out, text_out, current_kv

        return ts_out, text_out


# ---------------------------------------------------------------------------
# Multimodal DiT Block
# ---------------------------------------------------------------------------


class MultimodalDiTBlock(nn.Module):
    """DiT block with MMDiT-style joint attention and AdaLN conditioning.

    Structure per block:
    1. AdaLN pre-norm + JointAttention + post-attn gamma + residual
    2. AdaLN pre-norm + FeedForward + post-ff gamma + residual
    """

    def __init__(
        self,
        ts_dim: int,
        text_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int = 4,
        norm_eps: float = 1e-5,
        qk_bias: bool = False,
        rope_dim: int = 32,
        text_latent_dim: int = 0,
    ):
        super().__init__()
        # Attention norms (adaptive)
        self.ts_attn_norm = AdaptiveLayerNorm(ts_dim, dim_cond=emb_dim)
        self.text_attn_norm = AdaptiveLayerNorm(text_dim, dim_cond=emb_dim)

        # Joint attention
        self.joint_attn = MultimodalJointAttention(
            ts_dim=ts_dim,
            text_dim=text_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm_eps=norm_eps,
            rope_dim=rope_dim,
        )

        # Post-attn gamma (from conditioning)
        self.ts_post_attn_gamma = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, ts_dim))
        self.text_post_attn_gamma = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, text_dim))
        nn.init.zeros_(self.ts_post_attn_gamma[-1].weight)
        nn.init.zeros_(self.ts_post_attn_gamma[-1].bias)
        nn.init.zeros_(self.text_post_attn_gamma[-1].weight)
        nn.init.zeros_(self.text_post_attn_gamma[-1].bias)

        # FFN norms (adaptive)
        self.ts_ffn_norm = AdaptiveLayerNorm(ts_dim, dim_cond=emb_dim)
        self.text_ffn_norm = AdaptiveLayerNorm(text_dim, dim_cond=emb_dim)

        # FFN
        self.ts_ffn = MLP(ts_dim, expand_ratio)
        self.text_ffn = MLP(text_dim, expand_ratio)

        # Post-ff gamma
        self.ts_post_ff_gamma = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, ts_dim))
        self.text_post_ff_gamma = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, text_dim))
        nn.init.zeros_(self.ts_post_ff_gamma[-1].weight)
        nn.init.zeros_(self.ts_post_ff_gamma[-1].bias)
        nn.init.zeros_(self.text_post_ff_gamma[-1].weight)
        nn.init.zeros_(self.text_post_ff_gamma[-1].bias)

        # Text-Guided Feature Modulation (TGFM)
        self.use_tgfm = text_latent_dim > 0
        if self.use_tgfm:
            self.text_modulator = TextModulator(ts_dim, text_latent_dim)
            # Init to 0.1 (not 0) — zero gate + zero-modulator = double dead gradient
            self.text_mod_gate = nn.Parameter(torch.full((1,), 0.1))

    def forward(
        self,
        ts_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        ts_emb: torch.Tensor,
        text_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prefix_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        pos_offset: int = 0,
        return_kv: bool = False,
        text_latent: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, tuple]:

        # Attention branch
        ts_normed = self.ts_attn_norm(ts_tokens, cond=ts_emb)
        text_normed = self.text_attn_norm(text_tokens, cond=text_emb)

        if return_kv:
            ts_attn_out, text_attn_out, kv = self.joint_attn(
                ts_normed, text_normed, attn_mask=attn_mask,
                prefix_kv=prefix_kv, pos_offset=pos_offset, return_kv=True,
            )
        else:
            ts_attn_out, text_attn_out = self.joint_attn(
                ts_normed, text_normed, attn_mask=attn_mask,
                prefix_kv=prefix_kv, pos_offset=pos_offset,
            )

        # Post-attn gamma + residual
        ts_attn_out = ts_attn_out * self.ts_post_attn_gamma(ts_emb)
        text_attn_out = text_attn_out * self.text_post_attn_gamma(text_emb)

        ts_tokens = ts_tokens + ts_attn_out
        text_tokens = text_tokens + text_attn_out

        # FFN branch
        ts_ff = self.ts_ffn(self.ts_ffn_norm(ts_tokens, cond=ts_emb))
        text_ff = self.text_ffn(self.text_ffn_norm(text_tokens, cond=text_emb))

        # Post-ff gamma + residual
        ts_ff = ts_ff * self.ts_post_ff_gamma(ts_emb)
        text_ff = text_ff * self.text_post_ff_gamma(text_emb)

        ts_tokens = ts_tokens + ts_ff
        text_tokens = text_tokens + text_ff

        # Text-Guided Feature Modulation (TGFM) — after FFN, before return
        if self.use_tgfm and text_latent is not None:
            ts_tokens = ts_tokens + self.text_mod_gate * self.text_modulator(ts_tokens, text_latent)

        if return_kv:
            return ts_tokens, text_tokens, kv
        return ts_tokens, text_tokens


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class MMLDMDiTOutput:
    ts_sample: torch.Tensor
    text_sample: torch.Tensor


@dataclass
class PrefixKVCache:
    """Cached KV pairs from prefix blocks for inference acceleration."""
    # List of (ts_k, ts_v) per layer — text KV is not cached
    layers: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    n_prefix_ts: int = 0


# ---------------------------------------------------------------------------
# Main Model: MMLDMDiTModel
# ---------------------------------------------------------------------------


class MMLDMDiTModel(PreTrainedModel):
    """Multimodal DiT prior for MMLDM Stage 2.

    Learns the vector field ``v_psi(z_t, t; z_0^{(<b)}, c)`` with
    block-causal attention and MMDiT-style joint attention.
    """

    config_class = MMLDMDiTConfig
    base_model_prefix = "mmldm_dit"

    def __init__(self, config: MMLDMDiTConfig):
        super().__init__(config)
        self.config = config
        self.block_size = config.block_size
        self.heads = config.heads
        self.text_latent_dim = getattr(config, 'text_latent_dim', 0)
        self.n_text_views = getattr(config, 'n_text_views', 1)

        # Multi-View Text Pooler (MVTC)
        if self.text_latent_dim > 0 and self.n_text_views > 1:
            self.text_pooler = MultiViewTextPooler(
                text_dim=self.text_latent_dim,
                n_views=self.n_text_views,
            )
        else:
            self.text_pooler = None

        # Patch embedding
        self.ts_in = PatchIn1D(
            in_channels=config.ts_in_channels,
            patch_size=config.patch_size,
            dim=config.txt_dim,
        )
        self.text_in = PatchIn1D(
            in_channels=config.text_in_channels,
            patch_size=config.patch_size,
            dim=config.txt_dim,
        )

        # Timestep embedding
        self.emb_in = TimestepEmbedding(
            sinusoidal_dim=256,
            hidden_dim=config.txt_dim,
            output_dim=config.emb_dim,
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            MultimodalDiTBlock(
                ts_dim=config.txt_dim,
                text_dim=config.txt_dim,
                emb_dim=config.emb_dim,
                heads=config.heads,
                head_dim=config.head_dim,
                expand_ratio=config.expand_ratio,
                norm_eps=config.norm_eps,
                qk_bias=config.qk_bias,
                rope_dim=config.rope_dim,
                text_latent_dim=self.text_latent_dim,
            )
            for _ in range(config.num_layers)
        ])

        # Output
        self.ts_out_norm = AdaptiveLayerNorm(config.txt_dim, dim_cond=config.emb_dim)
        self.text_out_norm = AdaptiveLayerNorm(config.txt_dim, dim_cond=config.emb_dim)

        self.ts_out = PatchOut1D(
            out_channels=config.ts_out_channels,
            patch_size=config.patch_size,
            dim=config.txt_dim,
        )
        self.text_out = PatchOut1D(
            out_channels=config.text_out_channels,
            patch_size=config.patch_size,
            dim=config.txt_dim,
        )

        self.post_init()
        self._reset_conditioning_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _reset_conditioning_init(self):
        """Restore DiT-style stable init after HF ``post_init`` recursion."""
        for module in self.modules():
            if isinstance(module, AdaptiveLayerNorm) and module.has_cond:
                cond_linear = module.to_cond[-1]
                dim = cond_linear.out_features // 2
                nn.init.zeros_(cond_linear.weight)
                nn.init.constant_(cond_linear.bias[:dim], 1.0)
                nn.init.zeros_(cond_linear.bias[dim:])

        for block in self.blocks:
            for gate in (
                block.ts_post_attn_gamma[-1],
                block.text_post_attn_gamma[-1],
                block.ts_post_ff_gamma[-1],
                block.text_post_ff_gamma[-1],
            ):
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

        nn.init.zeros_(self.ts_out.proj.weight)
        nn.init.zeros_(self.ts_out.proj.bias)
        nn.init.zeros_(self.text_out.proj.weight)
        nn.init.zeros_(self.text_out.proj.bias)

    def _expand_text_latent_to_ts(
        self,
        text_latent: Optional[torch.Tensor],
        ts_shape: torch.LongTensor,
    ) -> Optional[torch.Tensor]:
        """Align per-sample text conditions to the flat TS token layout."""
        if text_latent is None or text_latent.ndim != 2:
            return text_latent
        lengths = ts_shape.flatten().tolist()
        if text_latent.shape[0] == 1 or text_latent.shape[0] != len(lengths):
            return text_latent
        expanded = [
            text_latent[i].unsqueeze(0).expand(int(length), -1)
            for i, length in enumerate(lengths)
            if int(length) > 0
        ]
        if not expanded:
            return text_latent[:1]
        return torch.cat(expanded, dim=0).unsqueeze(0)

    def forward(
        self,
        ts: torch.FloatTensor,
        text: torch.FloatTensor,
        ts_shape: torch.LongTensor,
        text_shape: torch.LongTensor,
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
        attn_mask: Optional[torch.Tensor] = None,
        prefix_kv: Optional[PrefixKVCache] = None,
        pos_offset: int = 0,
        text_timestep: Optional[Union[int, float, torch.IntTensor, torch.FloatTensor]] = None,
        text_latent: Optional[torch.FloatTensor] = None,
    ) -> MMLDMDiTOutput:
        """NA-form forward pass.

        Args:
            ts: ``(L_ts_total, ts_channels)`` TS latent tokens.
            text: ``(L_text_total, text_channels)`` text latent tokens.
            ts_shape: ``(B, 1)`` per-sample TS lengths.
            text_shape: ``(B, 1)`` per-sample text lengths.
            timestep: Flow Matching time ``t``.
            attn_mask: optional block-causal mask.
            text_latent: ``(B, text_latent_dim)`` raw text embedding for TGFM.
                If None or text_latent_dim=0, TGFM is skipped.

        Returns:
            :class:`MMLDMDiTOutput` with velocity field predictions.
        """
        # Patch embed
        ts, ts_shape_patched, ts_shape_before = self.ts_in(ts, ts_shape)
        text, text_shape_patched, text_shape_before = self.text_in(text, text_shape)

        # Per-modality timestep embeddings.
        ts_t_expanded = _expand_ts_timestep(
            timestep, ts_shape_patched, device=ts.device, dtype=ts.dtype,
        )
        text_t_source = timestep if text_timestep is None else text_timestep
        text_t_expanded = _expand_text_timestep(
            text_t_source,
            ts_shape_patched,
            text_shape_patched,
            device=ts.device,
            dtype=ts.dtype,
        )

        ts_emb = self.emb_in(ts_t_expanded, device=ts.device, dtype=ts.dtype)      # (L_ts, emb_dim)
        text_emb = self.emb_in(text_t_expanded, device=ts.device, dtype=ts.dtype)  # (L_text, emb_dim)

        # Build mask if not provided — multimodal [ts; text] joint layout
        if attn_mask is None:
            # Default: fixed block_size for each sample
            lengths = ts_shape_patched.flatten().tolist()
            block_sizes_fallback = []
            for l in lengths:
                n_full = l // self.block_size
                sizes = [self.block_size] * n_full
                remainder = l - n_full * self.block_size
                if remainder > 0:
                    sizes.append(remainder)
                block_sizes_fallback.append(sizes if sizes else [l])

            attn_mask = create_dit_readonly_text_mask(
                ts_shape=ts_shape_patched,
                text_shape=text_shape_patched,
                block_sizes=block_sizes_fallback,
                dtype=torch.bfloat16 if ts.is_cuda else ts.dtype,
                device=ts.device,
            )

        # Add batch dim for attention
        ts = ts.unsqueeze(0)    # (1, L_ts, d)
        text = text.unsqueeze(0)  # (1, L_text, d)

        # Add batch dim to embeddings
        ts_emb = ts_emb.unsqueeze(0)      # (1, L_ts, emb_dim)
        text_emb = text_emb.unsqueeze(0)  # (1, 1, emb_dim)

        # DiT blocks
        # If MVTC is active, pre-compute K text views and cycle through them.
        # Each block sees a different semantic "view" of the same text.
        text_views = None
        if self.text_pooler is not None and text_latent is not None:
            text_views = self.text_pooler(text_latent)  # K × (B, text_dim)

        for i, block in enumerate(self.blocks):
            layer_prefix_kv = prefix_kv.layers[i] if prefix_kv is not None and i < len(prefix_kv.layers) else None
            # Cycle through views if available, else use original text_latent
            block_text_latent = text_views[i % len(text_views)] if text_views is not None else text_latent
            block_text_latent = self._expand_text_latent_to_ts(block_text_latent, ts_shape_patched)
            ts, text = block(
                ts, text,
                ts_emb=ts_emb,
                text_emb=text_emb,
                attn_mask=attn_mask,
                prefix_kv=layer_prefix_kv,
                pos_offset=pos_offset,
                text_latent=block_text_latent,
            )

        # Output norm + unpatch
        ts = self.ts_out_norm(ts.squeeze(0), cond=ts_emb.squeeze(0))
        text = self.text_out_norm(text.squeeze(0), cond=text_emb.squeeze(0))

        ts, _ = self.ts_out(ts, ts_shape_patched, ts_shape_before)
        text, _ = self.text_out(text, text_shape_patched, text_shape_before)

        return MMLDMDiTOutput(ts_sample=ts, text_sample=text)

    @torch.no_grad()
    def compute_prefix_kv(
        self,
        ts: torch.FloatTensor,
        text: torch.FloatTensor,
        ts_shape: torch.LongTensor,
        text_shape: torch.LongTensor,
        timestep: Union[float, torch.FloatTensor],
        attn_mask: Optional[torch.Tensor] = None,
        text_timestep: Optional[Union[int, float, torch.IntTensor, torch.FloatTensor]] = None,
        text_latent: Optional[torch.FloatTensor] = None,
    ) -> PrefixKVCache:
        """Compute and cache KV for prefix tokens (inference only)."""
        # Patch embed
        ts, ts_shape_patched, ts_shape_before = self.ts_in(ts, ts_shape)
        text, text_shape_patched, text_shape_before = self.text_in(text, text_shape)

        # Timestep embeddings
        ts_t_expanded = _expand_ts_timestep(
            timestep, ts_shape_patched, device=ts.device, dtype=ts.dtype,
        )
        text_t_source = timestep if text_timestep is None else text_timestep
        text_t_expanded = _expand_text_timestep(
            text_t_source,
            ts_shape_patched,
            text_shape_patched,
            device=ts.device,
            dtype=ts.dtype,
        )
        ts_emb = self.emb_in(ts_t_expanded, device=ts.device, dtype=ts.dtype)
        text_emb = self.emb_in(text_t_expanded, device=ts.device, dtype=ts.dtype)

        # Build mask if not provided
        if attn_mask is None:
            lengths = ts_shape_patched.flatten().tolist()
            block_sizes_fallback = []
            for l in lengths:
                n_full = l // self.block_size
                sizes = [self.block_size] * n_full
                remainder = l - n_full * self.block_size
                if remainder > 0:
                    sizes.append(remainder)
                block_sizes_fallback.append(sizes if sizes else [l])
            attn_mask = create_dit_readonly_text_mask(
                ts_shape=ts_shape_patched,
                text_shape=text_shape_patched,
                block_sizes=block_sizes_fallback,
                dtype=torch.bfloat16 if ts.is_cuda else ts.dtype,
                device=ts.device,
            )

        ts = ts.unsqueeze(0)
        text = text.unsqueeze(0)
        ts_emb = ts_emb.unsqueeze(0)
        text_emb = text_emb.unsqueeze(0)

        # Run all blocks, collecting KV from each
        text_views = None
        if self.text_pooler is not None and text_latent is not None:
            text_views = self.text_pooler(text_latent)

        cache = PrefixKVCache()
        for i, block in enumerate(self.blocks):
            block_text_latent = text_views[i % len(text_views)] if text_views is not None else text_latent
            block_text_latent = self._expand_text_latent_to_ts(block_text_latent, ts_shape_patched)
            ts, text, kv = block(
                ts, text,
                ts_emb=ts_emb,
                text_emb=text_emb,
                attn_mask=attn_mask,
                return_kv=True,
                text_latent=block_text_latent,
            )
            cache.layers.append(kv)

        cache.n_prefix_ts = int(ts_shape_patched.sum().item())
        return cache

    @torch.no_grad()
    def extend_prefix_kv(
        self,
        existing_cache: PrefixKVCache,
        new_ts: torch.FloatTensor,
        text: torch.FloatTensor,
        new_ts_shape: torch.LongTensor,
        text_shape: torch.LongTensor,
        timestep: Union[float, torch.FloatTensor],
        pos_offset: int = 0,
        text_timestep: Optional[Union[int, float, torch.IntTensor, torch.FloatTensor]] = None,
        text_latent: Optional[torch.FloatTensor] = None,
    ) -> PrefixKVCache:
        """Incrementally extend a PrefixKVCache with new tokens.

        Only runs new_ts through all blocks and appends their KV to
        existing_cache, instead of reprocessing all prefix tokens.
        """
        # Patch embed
        ts, ts_shape_patched, _ = self.ts_in(new_ts, new_ts_shape)
        text, text_shape_patched, _ = self.text_in(text, text_shape)

        # Timestep embeddings
        ts_t_expanded = _expand_ts_timestep(
            timestep, ts_shape_patched, device=ts.device, dtype=ts.dtype,
        )
        text_t_source = timestep if text_timestep is None else text_timestep
        text_t_expanded = _expand_text_timestep(
            text_t_source, ts_shape_patched, text_shape_patched,
            device=ts.device, dtype=ts.dtype,
        )
        ts_emb = self.emb_in(ts_t_expanded, device=ts.device, dtype=ts.dtype)
        text_emb = self.emb_in(text_t_expanded, device=ts.device, dtype=ts.dtype)

        ts = ts.unsqueeze(0)
        text = text.unsqueeze(0)
        ts_emb = ts_emb.unsqueeze(0)
        text_emb = text_emb.unsqueeze(0)

        # Run new tokens through all blocks, collecting KV
        text_views = None
        if self.text_pooler is not None and text_latent is not None:
            text_views = self.text_pooler(text_latent)

        new_cache = PrefixKVCache()
        for i, block in enumerate(self.blocks):
            block_text_latent = text_views[i % len(text_views)] if text_views is not None else text_latent
            block_text_latent = self._expand_text_latent_to_ts(block_text_latent, ts_shape_patched)
            ts, text, kv = block(
                ts, text,
                ts_emb=ts_emb,
                text_emb=text_emb,
                pos_offset=pos_offset,
                return_kv=True,
                text_latent=block_text_latent,
            )
            # Append to existing layer KV
            old_ts_k, old_ts_v = existing_cache.layers[i]
            new_ts_k, new_ts_v = kv
            extended = (torch.cat([old_ts_k, new_ts_k], dim=2),
                        torch.cat([old_ts_v, new_ts_v], dim=2))
            new_cache.layers.append(extended)

        new_cache.n_prefix_ts = existing_cache.n_prefix_ts + int(new_ts_shape.sum().item())
        return new_cache


# Register
AutoConfig.register("mmldm_dit", MMLDMDiTConfig)
AutoModel.register(MMLDMDiTConfig, MMLDMDiTModel)
