"""Patch-aligned dynamic causal mechanism learner for MATD.

This module is a drop-in replacement for the existing ``causal.py`` while
adding stronger theoretical alignment with adaptive temporal patching and
text-conditioned latent diffusion.

Core conventions
----------------
- ``H`` is a segment-level latent mechanism state tensor with shape
  ``(B, S, M, D)``.
- All adjacency matrices use the same direction convention:
  ``A[..., i, j] = source mechanism j -> target mechanism i``.
- ``A0[b, s]`` is the instantaneous mechanism graph inside segment ``s``.
- ``Alags[b, s, lag]`` is the dynamic lag graph from segment
  ``s - 1 - lag`` to segment ``s``.

Public interface
----------------
The forward return protocol is intentionally compatible with the current MATD
codebase:

    causal_feat, A0_ret, Alags_ret, losses = learner(patches, text_feat)

A new optional ``meta`` argument is supported and should be used whenever the
adaptive patch metadata is available:

    causal_feat, A0_ret, Alags_ret, losses = learner(patches, text_feat, meta)

Inputs:
    patches:   (B, K, D) patch latent tokens.
    text_feat: (B, D) pooled text embedding.
    meta:      optional (B, K, 9) patch metadata.  Channel 2 is patch center.

Outputs:
    causal_feat: (B, K, D) patch-level causal conditioning feature.
    A0_ret:      (S, M, M) detached mean instantaneous graph.
    Alags_ret:   by default (L, M, M) detached mean lag graph for backward
                 compatibility.  If ``return_segment_lag_graph=True`` in the
                 constructor, returns (S, L, M, M).
    losses:      differentiable scalars: notears, predict, sparsity, smooth,
                 disentangle, and loss_causal.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicCausalMechanismLearner(nn.Module):
    """Dynamic latent causal mechanism learner for adaptive temporal patches.

    Args:
        dim: Feature dimension D.
        n_mech: Number of latent mechanisms M.
        n_segments: Number of temporal causal segments S.
        max_lag: Number of lagged segment graphs L.
        n_heads: Multi-head count for segment-local cross-attention.
        hidden_mult: Width multiplier for transition/update MLPs.
        dropout: Dropout probability.
        edge_bias_init: Initial logit bias for instantaneous edges.
        lag_bias_init: Initial logit bias for lagged edges.
        segment_bias_strength: Initial strength of the metadata-based
            segment-local attention bias.
        segment_bias_sigma: Initial temporal bandwidth for segment bias.
            If None, uses 0.5 / n_segments.
        detach_transition_target: If True, transition prediction loss trains
            graph/message/update functions without moving the target H[:, s].
        return_segment_lag_graph: If True, ``Alags_ret`` is (S, L, M, M).
            If False, ``Alags_ret`` is averaged over segments as (L, M, M),
            matching the previous implementation's logging shape.
        predict_weight, dag_weight, sparsity_weight, smooth_weight,
            disentangle_weight: Internal weights used only for the convenience
            aggregate ``losses["loss_causal"]``.  You can still combine
            individual terms manually in MATDModel/trainer.
        predict_weight: Weight of transition prediction inside losses["loss_causal"].
        dag_weight: Weight of NOTEARS penalty inside losses["loss_causal"].
        sparsity_weight: Weight of graph sparsity inside losses["loss_causal"].
        smooth_weight: Weight of temporal graph smoothness inside losses["loss_causal"].
        disentangle_weight: Weight of weak causal/style disentanglement inside losses["loss_causal"].
    """

    def __init__(
        self,
        dim: int,
        n_mech: int = 6,
        n_segments: int = 8,
        max_lag: int = 2,
        n_heads: int = 4,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        edge_bias_init: float = -4.0,
        lag_bias_init: float = -3.0,
        segment_bias_strength: float = 1.0,
        segment_bias_sigma: Optional[float] = None,
        detach_transition_target: bool = True,
        return_segment_lag_graph: bool = False,
        predict_weight: float = 1.0,
        dag_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        smooth_weight: float = 0.01,
        disentangle_weight: float = 0.01,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if n_mech <= 1:
            raise ValueError(f"n_mech must be > 1, got {n_mech}")
        if n_segments <= 0:
            raise ValueError(f"n_segments must be positive, got {n_segments}")
        if max_lag <= 0:
            raise ValueError(f"max_lag must be positive, got {max_lag}")
        if n_heads <= 0 or dim % n_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by positive n_heads={n_heads}"
            )

        self.dim = dim
        self.n_mech = n_mech
        self.n_segments = n_segments
        self.max_lag = max_lag
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.detach_transition_target = detach_transition_target
        self.return_segment_lag_graph = return_segment_lag_graph
        self.predict_weight = float(predict_weight)
        self.dag_weight = float(dag_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.smooth_weight = float(smooth_weight)
        self.disentangle_weight = float(disentangle_weight)

        # ------------------------------------------------------------------
        # Segment-local Perceiver-style mechanism extraction.
        # ------------------------------------------------------------------
        self.query = nn.Parameter(torch.randn(1, n_segments * n_mech, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm_h = nn.LayerNorm(dim)
        self.mech_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        sigma = segment_bias_sigma if segment_bias_sigma is not None else 0.5 / n_segments
        sigma = max(float(sigma), 1e-4)
        self.segment_log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.segment_bias_strength = nn.Parameter(torch.tensor(float(segment_bias_strength)))

        # ------------------------------------------------------------------
        # Dynamic graph learners. Direction convention:
        # A[..., i, j] means source j -> target i.
        # ------------------------------------------------------------------
        self.base_A0 = nn.Parameter(torch.tensor(float(edge_bias_init)))
        self.base_Alag = nn.Parameter(torch.tensor(float(lag_bias_init)))
        self.text_graph_proj = nn.Linear(dim, dim)

        self.edge_score = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

        self.lag_embed = nn.Embedding(max_lag, dim)
        self.lag_edge_score = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

        # ------------------------------------------------------------------
        # Edge-message transition decoder.
        # ------------------------------------------------------------------
        self.mech_embed = nn.Parameter(torch.randn(n_mech, dim) * 0.02)
        self.msg_lag_embed = nn.Embedding(max_lag + 1, dim)  # last id = A0
        self.edge_msg = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        hidden = dim * hidden_mult
        self.node_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim * 2, hidden),
                    nn.GELU(approximate="tanh"),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, dim),
                )
                for _ in range(n_mech)
            ]
        )

        # ------------------------------------------------------------------
        # Patch-level causal intervention feature projection.
        # ------------------------------------------------------------------
        self.local_patch_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.patch_mech_q = nn.Linear(dim, dim)
        self.patch_mech_k = nn.Linear(dim, dim)
        self.segment_config_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.meta_gate_proj = nn.Sequential(
            nn.Linear(9, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.intervention_gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.causal_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Style/non-causal proxy used only for an auxiliary disentanglement loss.
        self.style_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    # ------------------------------------------------------------------
    # Utility masks / metadata helpers
    # ------------------------------------------------------------------

    def _offdiag_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        eye = torch.eye(self.n_mech, device=device, dtype=dtype)
        return 1.0 - eye

    def _valid_lag_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return valid temporal lag mask with shape (S, L)."""
        S, L = self.n_segments, self.max_lag
        mask = torch.zeros(S, L, device=device, dtype=dtype)
        for s in range(S):
            for lag in range(L):
                if s - 1 - lag >= 0:
                    mask[s, lag] = 1.0
        return mask

    def _check_inputs(
        self,
        patches: torch.Tensor,
        text_feat: torch.Tensor,
        meta: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if patches.dim() != 3:
            raise ValueError(f"patches must have shape (B,K,D), got {tuple(patches.shape)}")
        if text_feat.dim() != 2:
            raise ValueError(f"text_feat must have shape (B,D), got {tuple(text_feat.shape)}")

        B, K, D = patches.shape
        if D != self.dim:
            raise ValueError(f"patch dim D={D} does not match self.dim={self.dim}")
        if text_feat.shape != (B, D):
            raise ValueError(f"text_feat must have shape {(B, D)}, got {tuple(text_feat.shape)}")

        if meta is None:
            return None
        if meta.dim() != 3 or meta.shape[0] != B or meta.shape[1] != K:
            raise ValueError(
                f"meta must be None or shape (B,K,C) aligned to patches; "
                f"got meta={tuple(meta.shape)}, patches={tuple(patches.shape)}"
            )
        if meta.shape[-1] < 9:
            raise ValueError(f"meta must have at least 9 channels, got {meta.shape[-1]}")
        return meta[..., :9].to(device=patches.device, dtype=patches.dtype)

    def _segment_attention_bias(self, meta: torch.Tensor) -> torch.Tensor:
        """Metadata-based temporal bias for mechanism extraction.

        Args:
            meta: (B, K, 9), where channel 2 is patch center in [0, 1].
        Returns:
            bias: (B, S*M, K). Larger values mean stronger attention.
        """
        centers = meta[..., 2].clamp(0.0, 1.0)
        B, K = centers.shape
        S, M = self.n_segments, self.n_mech

        seg_centers = torch.linspace(
            0.5 / S,
            1.0 - 0.5 / S,
            S,
            device=centers.device,
            dtype=centers.dtype,
        )
        dist = centers[:, None, :] - seg_centers[None, :, None]  # (B, S, K)
        sigma = self.segment_log_sigma.exp().clamp(min=1e-4, max=1.0).to(centers.dtype)
        strength = F.softplus(self.segment_bias_strength).to(centers.dtype)
        bias = -0.5 * (dist / sigma).pow(2) * strength
        return bias[:, :, None, :].expand(B, S, M, K).reshape(B, S * M, K)

    # ------------------------------------------------------------------
    # Mechanism extraction
    # ------------------------------------------------------------------

    def _extract_mechanisms(
        self,
        patches: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract segment-mechanism states H with shape (B, S, M, D)."""
        B, K, D = patches.shape
        N = self.n_segments * self.n_mech

        q0 = self.query.expand(B, -1, -1)
        q = self.q_proj(self.norm_q(q0))
        kv = self.norm_kv(patches)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = q.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, K)
        if meta is not None:
            attn = attn + self._segment_attention_bias(meta).unsqueeze(1)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.attn_out(out)

        H = self.mech_proj(out + q0)
        H = self.norm_h(H)
        return H.reshape(B, self.n_segments, self.n_mech, D)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _instantaneous_graph(self, H: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Learn A0 where A0[b, s, i, j] means source j -> target i."""
        B, S, M, D = H.shape
        mask = self._offdiag_mask(H.device, H.dtype)

        target = H.unsqueeze(3).expand(B, S, M, M, D)  # target i
        source = H.unsqueeze(2).expand(B, S, M, M, D)  # source j
        text_h = self.text_graph_proj(text_feat).to(dtype=H.dtype)
        text_h = text_h.view(B, 1, 1, 1, D).expand(B, S, M, M, D)

        pair = torch.cat([source, target, text_h], dim=-1)
        logits = self.edge_score(pair).squeeze(-1)
        return torch.sigmoid(self.base_A0.to(H.dtype) + logits) * mask

    def _lagged_graph(self, H: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Learn Alags[b, s, lag, i, j] = source j -> target i."""
        B, S, M, D = H.shape
        mask = self._offdiag_mask(H.device, H.dtype)
        text_h = self.text_graph_proj(text_feat).to(dtype=H.dtype)

        rows: list[torch.Tensor] = []
        for s in range(S):
            lag_rows: list[torch.Tensor] = []
            for lag in range(self.max_lag):
                src_s = s - 1 - lag
                if src_s < 0:
                    lag_rows.append(H.new_zeros(B, M, M))
                    continue

                target = H[:, s].unsqueeze(2).expand(B, M, M, D)      # target i
                source = H[:, src_s].unsqueeze(1).expand(B, M, M, D)  # source j
                text = text_h[:, None, None, :].expand(B, M, M, D)
                lag_e = self.lag_embed.weight[lag].to(H.dtype).view(1, 1, 1, D)
                lag_e = lag_e.expand(B, M, M, D)

                pair = torch.cat([source, target, text, lag_e], dim=-1)
                logits = self.lag_edge_score(pair).squeeze(-1)
                A = torch.sigmoid(self.base_Alag.to(H.dtype) + logits) * mask
                lag_rows.append(A)
            rows.append(torch.stack(lag_rows, dim=1))  # (B, L, M, M)

        return torch.stack(rows, dim=1)  # (B, S, L, M, M)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _notears_loss(self, A0: torch.Tensor) -> torch.Tensor:
        """NOTEARS acyclicity penalty on every instantaneous graph A0[b, s]."""
        _B, _S, M, _ = A0.shape
        A = A0.float()  # matrix_exp is safer in float32 under AMP.
        mat_exp = torch.linalg.matrix_exp(A * A)
        traces = mat_exp.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        h = traces - float(M)
        return h.pow(2).mean().to(dtype=A0.dtype)

    def _message_pass(self, A: torch.Tensor, source_H: torch.Tensor, lag_index: int) -> torch.Tensor:
        """Edge-level message passing.

        Args:
            A: (B, M, M), A[i, j] means source j -> target i.
            source_H: (B, M, D) source mechanism states.
            lag_index: 0..max_lag-1 for lagged edges, max_lag for A0.
        Returns:
            (B, M, D) aggregated incoming messages per target.
        """
        B, M, D = source_H.shape
        if lag_index < 0 or lag_index > self.max_lag:
            raise ValueError(f"lag_index must be in [0, {self.max_lag}], got {lag_index}")

        source_state = source_H.unsqueeze(1).expand(B, M, M, D)  # source j
        target_id = self.mech_embed.to(source_H.dtype).view(1, M, 1, D).expand(B, M, M, D)
        source_id = self.mech_embed.to(source_H.dtype).view(1, 1, M, D).expand(B, M, M, D)
        lag_id = self.msg_lag_embed.weight[lag_index].to(source_H.dtype).view(1, 1, 1, D)
        lag_id = lag_id.expand(B, M, M, D)

        msg_in = torch.cat([source_state, target_id, source_id, lag_id], dim=-1)
        msg = self.edge_msg(msg_in)
        return (msg * A.unsqueeze(-1)).sum(dim=2)

    def _transition_loss(self, H: torch.Tensor, A0: torch.Tensor, Alags: torch.Tensor) -> torch.Tensor:
        """Predict current mechanisms from lagged and instantaneous parents."""
        B, S, M, _D = H.shape
        if S <= 1:
            return H.new_tensor(0.0)

        total = H.new_tensor(0.0)
        count = 0
        for s in range(1, S):
            agg = H.new_zeros(B, M, self.dim)

            for lag in range(min(s, self.max_lag)):
                source_H = H[:, s - 1 - lag]
                agg = agg + self._message_pass(Alags[:, s, lag], source_H, lag_index=lag)

            # Same-segment instantaneous parents. Diagonal is already zero.
            agg = agg + self._message_pass(A0[:, s], H[:, s], lag_index=self.max_lag)

            # Previous self state helps avoid a trivial identity solution from H[:, s].
            prev_self = H[:, s - 1]
            pred = torch.stack(
                [
                    self.node_update[i](torch.cat([prev_self[:, i], agg[:, i]], dim=-1))
                    for i in range(M)
                ],
                dim=1,
            )

            target = H[:, s].detach() if self.detach_transition_target else H[:, s]
            total = total + F.smooth_l1_loss(pred, target)
            count += 1

        return total / max(count, 1)

    def _sparsity_loss(self, A0: torch.Tensor, Alags: torch.Tensor) -> torch.Tensor:
        """L1 sparsity on valid off-diagonal graph weights."""
        B, S, M, _ = A0.shape
        L = self.max_lag
        offdiag = self._offdiag_mask(A0.device, A0.dtype)

        a0_denom = max(B * S * M * (M - 1), 1)
        a0_sparse = (A0.abs() * offdiag).sum() / a0_denom

        valid = self._valid_lag_mask(A0.device, A0.dtype).view(1, S, L, 1, 1)
        lag_denom = (valid.sum() * M * (M - 1)).clamp_min(1.0)
        lag_sparse = (Alags.abs() * valid * offdiag.view(1, 1, 1, M, M)).sum() / lag_denom
        return a0_sparse + lag_sparse

    def _smoothness_loss(self, A0: torch.Tensor, Alags: torch.Tensor) -> torch.Tensor:
        """Temporal smoothness for dynamic graphs across adjacent segments."""
        if self.n_segments <= 1:
            return A0.new_tensor(0.0)

        smooth_a0 = (A0[:, 1:] - A0[:, :-1]).abs().mean()

        valid = self._valid_lag_mask(A0.device, A0.dtype)  # (S, L)
        valid_pair = (valid[1:] * valid[:-1]).view(1, self.n_segments - 1, self.max_lag, 1, 1)
        lag_diff = (Alags[:, 1:] - Alags[:, :-1]).abs() * valid_pair
        denom = valid_pair.sum().clamp_min(1.0)
        smooth_lag = lag_diff.sum() / (denom * self.n_mech * self.n_mech)
        return smooth_a0 + smooth_lag

    @staticmethod
    def _cross_cov_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1:
            return x.new_tensor(0.0)
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        cov = x.transpose(0, 1) @ y / max(x.shape[0] - 1, 1)
        return cov.pow(2).mean()

    def _disentangle_loss(self, H: torch.Tensor, patches: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Softly decorrelate causal mechanism pool and style/text residual proxy."""
        causal_pool = H.mean(dim=(1, 2))
        style_proxy = self.style_proj(torch.cat([patches.mean(dim=1), text_feat], dim=-1))
        cos = (F.normalize(causal_pool, dim=-1) * F.normalize(style_proxy, dim=-1))
        cos_loss = cos.sum(dim=-1).pow(2).mean()
        return cos_loss + self._cross_cov_loss(causal_pool, style_proxy)

    # ------------------------------------------------------------------
    # Patch-level causal feature reconstruction / intervention
    # ------------------------------------------------------------------

    def _segment_indices(self, B: int, K: int, device: torch.device, meta: Optional[torch.Tensor]) -> torch.Tensor:
        if meta is not None:
            centers = meta[..., 2].clamp(0.0, 1.0)
            return (centers * self.n_segments).long().clamp(0, self.n_segments - 1)
        pos = torch.arange(K, device=device, dtype=torch.float32)
        seg = (pos / max(K, 1) * self.n_segments).long().clamp(0, self.n_segments - 1)
        return seg.unsqueeze(0).expand(B, -1)

    def _patch_causal_features(
        self,
        patches: torch.Tensor,
        H: torch.Tensor,
        A0: torch.Tensor,
        meta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Map mechanism states and graph context back to patch-level features."""
        B, K, D = patches.shape
        device = patches.device

        seg_idx = self._segment_indices(B, K, device, meta)
        b_idx = torch.arange(B, device=device).unsqueeze(1)

        local = self.local_patch_proj(patches)  # (B, K, D)
        seg_mech = H[b_idx, seg_idx]            # (B, K, M, D)

        # Patch-specific attention over mechanisms within its assigned segment.
        q = self.patch_mech_q(local).unsqueeze(2)  # (B, K, 1, D)
        k = self.patch_mech_k(seg_mech)            # (B, K, M, D)
        mech_logits = (q * k).sum(dim=-1) / math.sqrt(D)
        mech_w = F.softmax(mech_logits, dim=-1)
        seg_feat = (mech_w.unsqueeze(-1) * seg_mech).sum(dim=2)  # (B, K, D)

        # Dynamic graph configuration for the assigned segment.
        incoming = A0.abs().mean(dim=-1)  # target incoming profile, (B, S, M)
        outgoing = A0.abs().mean(dim=-2)  # source outgoing profile, (B, S, M)
        strength = incoming + outgoing
        strength = strength / strength.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        seg_summary = (H * strength.unsqueeze(-1)).mean(dim=2)  # (B, S, D)
        seg_config_all = self.segment_config_proj(seg_summary)
        seg_config = seg_config_all[b_idx, seg_idx]  # (B, K, D)

        meta9 = meta if meta is not None else patches.new_zeros(B, K, 9)
        meta_h = self.meta_gate_proj(meta9.to(dtype=patches.dtype))

        gate = self.intervention_gate(torch.cat([local, seg_feat, seg_config, meta_h], dim=-1))
        fused = local + gate * (seg_feat + seg_config)
        return self.causal_proj(fused)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        patches: torch.Tensor,
        text_feat: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run adaptive latent causal mechanism learning."""
        meta_9 = self._check_inputs(patches, text_feat, meta)
        text_feat = text_feat.to(device=patches.device, dtype=patches.dtype)

        H = self._extract_mechanisms(patches, meta=meta_9)  # (B, S, M, D)
        A0 = self._instantaneous_graph(H, text_feat)        # (B, S, M, M)
        Alags = self._lagged_graph(H, text_feat)            # (B, S, L, M, M)

        predict = self._transition_loss(H, A0, Alags)
        notears = self._notears_loss(A0)
        sparsity = self._sparsity_loss(A0, Alags)
        smooth = self._smoothness_loss(A0, Alags)
        disentangle = self._disentangle_loss(H, patches, text_feat)

        causal_feat = self._patch_causal_features(patches, H, A0, meta=meta_9)

        A0_ret = A0.mean(dim=0).detach()  # (S, M, M)
        if self.return_segment_lag_graph:
            Alags_ret = Alags.mean(dim=0).detach()  # (S, L, M, M)
        else:
            Alags_ret = Alags.mean(dim=(0, 1)).detach()  # (L, M, M)

        losses: Dict[str, torch.Tensor] = {
            "predict": predict,
            "notears": notears,
            "sparsity": sparsity,
            "smooth": smooth,
            "disentangle": disentangle,
            # A conservative default aggregate.  Prefer overriding weights in
            # MATDModel / trainer configs as described in the companion notes.
            "loss_causal": (
                self.predict_weight * predict
                + self.dag_weight * notears
                + self.sparsity_weight * sparsity
                + self.smooth_weight * smooth
                + self.disentangle_weight * disentangle
            ),
        }
        return causal_feat, A0_ret, Alags_ret, losses


AdaptiveLatentCausalIntervention = DynamicCausalMechanismLearner
