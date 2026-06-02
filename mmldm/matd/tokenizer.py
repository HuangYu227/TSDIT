"""Density-Aware Adaptive Temporal Patching (DA-ATP) tokenizer."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityAwareAdaptivePatch(nn.Module):
    _META_DIM = 9

    def __init__(self, embed_dim=256, target_tokens=None, ref_len=16,
                 min_len=4, max_len=64, rfft_win=32, tau=0.5, base_score=0.05,
                 target_seg_len=8, min_tokens=6, max_tokens=128):
        super().__init__()
        self.embed_dim = embed_dim
        self._target_tokens = target_tokens
        self.ref_len = ref_len
        self.min_len = min_len
        self.max_len = max_len
        self.rfft_win = rfft_win
        self.tau = tau
        self.base_score = base_score
        self.target_seg_len = target_seg_len
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.content_mlp = nn.Sequential(
            nn.Linear(ref_len + 9, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.pos_mlp = nn.Sequential(
            nn.Linear(9, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))

    def _choose_k(self, T: int) -> int:
        """Length-aware token number selection."""
        if self._target_tokens is None:
            K = math.ceil(T / self.target_seg_len)
            K = max(K, self.min_tokens)
            K = min(K, self.max_tokens)
        else:
            K = int(self._target_tokens)

        max_allowed_k = max(1, T // self.min_len)
        min_required_k = math.ceil(T / self.max_len)

        K = min(K, max_allowed_k)
        K = max(K, min_required_k)

        return int(K)

    @staticmethod
    def _safe_norm(x, eps=1e-6):
        return x / (x.mean(dim=-1, keepdim=True) + eps)

    def compute_rfft_score(self, x):
        B, T = x.shape
        W = min(self.rfft_win, T)
        if W < 4:
            return torch.zeros_like(x)
        windows = x.unfold(1, W, 1)
        N = windows.shape[1]
        hann = torch.hann_window(W, device=x.device, dtype=x.dtype)
        windows = windows * hann.view(1, 1, W)
        X = torch.fft.rfft(windows, dim=-1)
        P = X.abs().pow(2) + 1e-8
        Pn = P / P.sum(-1, keepdim=True)
        entropy = -(Pn * Pn.log()).sum(-1) / math.log(P.shape[-1])
        hf = max(1, int(P.shape[-1] * 0.5))
        hf_ratio = P[..., hf:].sum(-1) / P.sum(-1)
        win_score = entropy + 0.5 * hf_ratio
        score = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        count = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        half = W // 2
        for i in range(N):
            c = min(i + half, T - 1)
            score[:, c] += win_score[:, i]
            count[:, c] += 1
        score = score / count.clamp_min(1.0)
        if half < T:
            score[:, :half] = score[:, half:half+1]
        last = min(T-1, N-1+half)
        score[:, last:] = score[:, last:last+1]
        return score

    def compute_raw_score(self, x):
        d1 = torch.zeros_like(x)
        d1[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
        d2 = torch.zeros_like(x)
        d2[:, 1:-1] = (x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]).abs()
        d1 = self._safe_norm(d1)
        d2 = self._safe_norm(d2)
        freq = self._safe_norm(self.compute_rfft_score(x))
        raw = d1 + 0.5*d2 + freq + 1e-4
        raw = F.avg_pool1d(raw.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        return raw.clamp_min(1e-4)

    def make_boundaries(self, raw_score):
        T = raw_score.shape[0]
        K = self._choose_k(T)

        if K * self.min_len > T:
            raise ValueError(f"K*min_len={K*self.min_len} > T={T}")
        if K * self.max_len < T:
            raise ValueError(f"K*max_len={K*self.max_len} < T={T}")

        alloc = raw_score.pow(self.tau) + self.base_score
        cdf = torch.cumsum(alloc, 0)
        total = cdf[-1].clamp_min(1e-6)

        boundaries = [0]
        prev = 0

        for k in range(1, K):
            idx = torch.searchsorted(cdf, total * (k / K)).item()

            lower = max(
                prev + self.min_len,
                T - (K - k) * self.max_len,
            )
            upper = min(
                prev + self.max_len,
                T - (K - k) * self.min_len,
            )

            if lower > upper:
                raise ValueError(
                    f"No feasible boundary: T={T}, K={K}, "
                    f"k={k}, lower={lower}, upper={upper}"
                )

            idx = min(max(idx, lower), upper)
            boundaries.append(idx)
            prev = idx

        boundaries.append(T)
        return boundaries

    def encode_segment(self, segment, score_seg, total_score, T):
        L = segment.shape[0]
        seg = F.interpolate(segment.view(1,1,L), self.ref_len, mode="linear", align_corners=False).view(self.ref_len)
        mass = score_seg.sum() / total_score
        length = torch.tensor(L/T, device=segment.device, dtype=segment.dtype)
        density = mass / (length + 1e-6)
        stats = torch.stack([segment.mean(), segment.std(unbiased=False),
                             segment.min(), segment.max(), segment[0], segment[-1],
                             segment[-1]-segment[0], mass, density])
        return torch.cat([seg, stats]), mass, density

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected (B,T), got {tuple(x.shape)}")
        B, T = x.shape
        K = self._choose_k(T)
        raw_score = self.compute_raw_score(x)
        all_tokens, all_meta = [], []
        for b in range(B):
            score_b = raw_score[b]
            bd = self.make_boundaries(score_b)
            total = score_b.sum().clamp_min(1e-6)
            cfeats, mfeats = [], []
            for k in range(K):
                s, e = bd[k], bd[k+1]
                feat, mass, density = self.encode_segment(x[b,s:e], score_b[s:e], total, T)
                start = torch.tensor(s/T, device=x.device, dtype=x.dtype)
                end_v = torch.tensor(e/T, device=x.device, dtype=x.dtype)
                center = torch.tensor(((s+e-1)/2)/max(T-1,1), device=x.device, dtype=x.dtype)
                length = torch.tensor((e-s)/T, device=x.device, dtype=x.dtype)
                meta = torch.stack([start, end_v, center, length, torch.log(length+1e-6),
                                    mass, density, torch.log(density+1e-6),
                                    torch.tensor(k/max(K-1,1), device=x.device, dtype=x.dtype)])
                cfeats.append(feat)
                mfeats.append(meta)
            cfeats = torch.stack(cfeats)
            mfeats = torch.stack(mfeats)
            tokens = self.content_mlp(cfeats) + self.pos_mlp(mfeats)
            all_tokens.append(tokens)
            all_meta.append(mfeats)
        return torch.stack(all_tokens), torch.stack(all_meta)


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_buckets=33, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_buckets = num_buckets
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)
        self.rel_time_bias = nn.Embedding(num_buckets, num_heads)

    def forward(self, x, meta=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        if meta is not None:
            centers = meta[..., 2]
            delta = centers[:,None,:] - centers[:,:,None]
            bucket = ((delta+1)*0.5*(self.num_buckets-1)).long().clamp(0, self.num_buckets-1)
            bias = self.rel_time_bias(bucket).permute(0,3,1,2)
            attn[:,:,1:,1:] = attn[:,:,1:,1:] + bias
        attn = self.attn_drop(attn.softmax(-1))
        out = (attn @ v).transpose(1,2).reshape(B,N,D)
        return self.proj_drop(self.proj(out))


class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalSelfAttention(dim, num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(drop))
    def forward(self, x, meta=None):
        x = x + self.attn(self.norm1(x), meta)
        x = x + self.mlp(self.norm2(x))
        return x


class AdaptiveTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=256, out_dim=128, target_tokens=32, depth=4, num_heads=8, drop=0.1):
        super().__init__()
        self.patch_embed = DensityAwareAdaptivePatch(embed_dim=embed_dim, target_tokens=target_tokens)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.blocks = nn.ModuleList([TemporalTransformerBlock(embed_dim, num_heads, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        tokens, meta = self.patch_embed(x)
        cls = self.cls_token.expand(B,-1,-1) + self.cls_pos
        h = torch.cat([cls, tokens], 1)
        for block in self.blocks:
            h = block(h, meta)
        return self.proj(self.norm(h)), meta
