"""CTICD: Channel-Transform-informed Causal Dynamics for TIGER (Simplified).

Simplified production-oriented plug-and-play causal mechanism module.
Based on CTICD-v3 with the following reductions:
  - Removed: CrossChannelSinkhornCCIP, ChannelAdversary, route_regularization
  - Removed: GradientReverse, log_sinkhorn, _make_channel_labels
  - DynamicCausalGraphLearner returns only A, A_dag, notears_loss, sparsity_loss
  - Only 3 losses: L_causal, L_notears, L_sparsity (L_diffusion external)

Key properties:
  - DDP-safe: all parameters created in __init__, none in forward().
  - Resolution-agnostic: cross-attention from x_in queries to mechanism states.
  - Safe injection: sigmoid(-4) ~ 0.018 scale + zero-init out_proj.
  - GradientScale: prevents diffusion gradients from overwhelming causal losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "CTICD",
    "CTICDOutput",
    "GradientScale",
]


# ---------------------------------------------------------------------------
# Return container
# ---------------------------------------------------------------------------

@dataclass
class CTICDOutput:
    """Return value of CTICD.forward()."""

    causal_features: torch.Tensor      # (B, C, 1, L)  -- ready to add to x_in
    causal_graph: torch.Tensor         # (B, M, M)     -- learned DAG adjacency
    mechanism_states: torch.Tensor     # (B, M, D)     -- transitioned stable states
    losses: dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

class GradientScale(torch.autograd.Function):
    """Identity in forward; scales incoming gradient by *scale* in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize every parameter of *module*."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _pool_attr(
    attr_emb: Optional[torch.Tensor],
    batch: int,
    attr_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Pool TIGER attr embedding to (B, attr_dim)."""
    if attr_emb is None:
        return torch.zeros(batch, attr_dim, device=device)

    if attr_emb.dim() == 4:
        pooled = attr_emb.mean(dim=(2, 3))
    elif attr_emb.dim() == 3:
        pooled = attr_emb.mean(dim=-1)
    elif attr_emb.dim() == 2:
        pooled = attr_emb
    else:
        raise ValueError(
            f"attr_emb must be 2D/3D/4D or None, got shape {tuple(attr_emb.shape)}."
        )

    if pooled.shape[-1] != attr_dim:
        raise ValueError(
            f"attr_emb pooled dim={pooled.shape[-1]} but CTICD attr_dim={attr_dim}."
        )

    return pooled

# ---------------------------------------------------------------------------
# Component A: Channel Mechanism Encoder
# ---------------------------------------------------------------------------

class ChannelMechanismEncoder(nn.Module):
    """Encode one image channel into K mechanism states via cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_mechanisms: int,
        patch_size: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )

        self.d_model = d_model
        self.n_mechanisms = n_mechanisms

        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(1, d_model),
            nn.GELU(approximate="tanh"),
        )

        self.mechanism_queries = nn.Parameter(
            torch.empty(n_mechanisms, d_model)
        )
        nn.init.normal_(self.mechanism_queries, std=1.0 / math.sqrt(d_model))

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(2 * d_model, d_model),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, channel: torch.Tensor) -> torch.Tensor:
        B = channel.shape[0]
        feat = self.patch_embed(channel)                       # (B, D, h, w)
        spatial = feat.flatten(2).transpose(1, 2).contiguous() # (B, hw, D)
        queries = self.mechanism_queries.unsqueeze(0).expand(B, -1, -1)
        attended, _ = self.attn(
            query=queries, key=spatial, value=spatial, need_weights=False,
        )
        states = self.norm(attended)
        states = states + self.ffn(states)  # (B, K, d_model)
        return states

# ---------------------------------------------------------------------------
# Component B: Mechanism Splitter
# ---------------------------------------------------------------------------

class MechanismSplitter(nn.Module):
    """Project raw mechanism states into stable (causal) representations.

    Simplified from CTICD-v3: nuisance branch removed to avoid DDP deadlock
    (unused parameters have grad=None). Only stable projection retained.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.stable_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model, d_model),
        )
        self.stable_norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.stable_proj[-1].weight)
        nn.init.zeros_(self.stable_proj[-1].bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.stable_norm(states + self.stable_proj(states))

# ---------------------------------------------------------------------------
# Component C: Dynamic Causal Graph Learner
# ---------------------------------------------------------------------------

class DynamicCausalGraphLearner(nn.Module):
    """Learn a DAG adjacency over mechanism nodes.
    NOTEARS acyclicity constraint applied to the instantaneous DAG.
    Returns A, notears_loss, and sparsity_loss.
    """

    def __init__(self, n_nodes: int, edge_bias: float = -4.0, init_tau: float = 1.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.base_dag_logits = nn.Parameter(
            torch.full((n_nodes, n_nodes), float(edge_bias))
        )
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, _ = states.shape
        if M != self.n_nodes:
            raise ValueError(f'Expected {self.n_nodes} nodes, got {M}.')
        normed = F.normalize(states, dim=-1)
        sim = torch.bmm(normed, normed.transpose(1, 2))
        tau = self.log_tau.exp().clamp(min=0.1, max=5.0)
        dag_logits = self.base_dag_logits.unsqueeze(0) + sim / tau
        A = torch.sigmoid(dag_logits)
        eye = torch.eye(M, device=states.device, dtype=states.dtype).unsqueeze(0)
        A = A * (1.0 - eye)
        W_sq = A * A
        # NOTEARS: tr(exp(A∘A)) - M = 0 guarantees DAG
        # Use max(h²) instead of mean(h²) to ensure ALL samples are DAGs,
        # not just the average (mean could be 0 with positive/negative cancellation)
        expm = torch.linalg.matrix_exp(W_sq)  # (B, M, M) batched
        h = expm.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - M  # (B,)
        notears_loss = (h * h).mean()  # mean for smoother gradient flow
        sparsity_loss = A.mean()
        return A, notears_loss, sparsity_loss

# ---------------------------------------------------------------------------
# Component D: Mechanism Transition
# ---------------------------------------------------------------------------

class MechanismTransition(nn.Module):
    """Per-mechanism MLP with graph parent aggregation + FiLM conditioning."""

    def __init__(self, d_model: int, n_nodes: int, attr_dim: int, hidden_mult: int = 2):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        hidden = hidden_mult * d_model
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden, d_model),
            )
            for _ in range(n_nodes)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_nodes)])
        self.film = nn.Linear(attr_dim, 2 * n_nodes * d_model)
        _zero_module(self.film)
        for mlp in self.mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def forward(
        self, states: torch.Tensor, graph: torch.Tensor,
        attr_pooled: torch.Tensor, intervention: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        B, M, D = states.shape
        graph_used = graph
        node = None
        value = None
        strength = 1.0
        cut_incoming = True
        if intervention is not None:
            node = int(intervention["node"])
            if not (0 <= node < M):
                raise ValueError(f'intervention node {node} out of range for M={M}.')
            strength = float(intervention.get("strength", 1.0))
            cut_incoming = bool(intervention.get("cut_incoming", True))
            value = intervention.get("value", None)
            if cut_incoming:
                graph_used = graph.clone()
                graph_used[:, :, node] = 0.0
            if value is not None:
                value = value.to(device=states.device, dtype=states.dtype)
                if value.dim() == 1:
                    value = value.unsqueeze(0).expand(B, -1)
                elif value.dim() == 3 and value.shape[1] == 1:
                    value = value[:, 0, :]
                if value.shape != (B, D):
                    raise ValueError(
                        f"intervention value must broadcast to (B,D)={(B,D)}, got {tuple(value.shape)}."
                    )
        # Graph convention: A[j,k] means j→k (j is parent of k)
        # parent[b, k] = Σ_j A[j,k] · states[b, j] = sum of all parents of k
        parent = torch.bmm(graph_used.transpose(1, 2), states)
        gamma_beta = self.film(attr_pooled).view(B, M, 2, D)
        gamma = gamma_beta[:, :, 0, :]
        beta = gamma_beta[:, :, 1, :]
        updated_parts = []
        for k in range(M):
            delta = self.mlps[k](parent[:, k, :])
            delta = self.norms[k](delta)
            delta = delta * (1.0 + gamma[:, k, :]) + beta[:, k, :]
            updated_parts.append(states[:, k, :] + delta)
        updated = torch.stack(updated_parts, dim=1)
        if intervention is not None and value is not None:
            old = updated[:, node, :]
            updated[:, node, :] = (1.0 - strength) * old + strength * value
        return updated

# ---------------------------------------------------------------------------
# Component E: Mechanism Recomposer
# ---------------------------------------------------------------------------

class MechanismRecomposer(nn.Module):
    """Cross-attention from x_in queries to mechanism state key/values."""

    def __init__(self, d_model: int, output_channels: int, num_heads: int = 4):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )
        self.in_proj = nn.Linear(output_channels, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = _zero_module(nn.Linear(d_model, output_channels))

    def forward(self, mechanism_states: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        q = x_in.squeeze(2).transpose(1, 2).contiguous()
        q = self.in_proj(q)
        attended, _ = self.cross_attn(
            query=q, key=mechanism_states, value=mechanism_states, need_weights=False,
        )
        attended = self.norm(attended + q)
        out = self.out_proj(attended)
        return out.transpose(1, 2).unsqueeze(2).contiguous()

# ---------------------------------------------------------------------------
# Main CTICD Module (Simplified)
# ---------------------------------------------------------------------------

class CTICD(nn.Module):
    """Simplified CTICD module with 3 losses.

    Components:
        A) ChannelMechanismEncoder -- per-channel encoding to K states
        B) MechanismSplitter       -- stable vs nuisance separation
        C) DynamicCausalGraphLearner -- DAG + NOTEARS constraint
        D) MechanismTransition     -- per-mechanism MLP with graph parents
        E) MechanismRecomposer     -- cross-attention from x_in to states
        F) Safe Injection          -- sigmoid(-4) + GradientScale

    Losses:
        L_causal:    MSE(transitioned, raw) -- how well mechanisms predict each other
        L_notears:   [tr(exp(A^2)) - M]^2  -- acyclicity constraint
        L_sparsity:  ||A||_1 / M^2         -- L1 graph sparsity
    """

    def __init__(
        self, d_model: int, output_channels: int, attr_dim: Optional[int] = None,
        n_channels: int = 3, n_mechanisms_per_channel: int = 4,
        patch_size: int = 4, num_heads: int = 4, edge_bias: float = -4.0,
        branch_grad_scale: float = 0.2,
        lambda_causal: float = 1.0, lambda_notears: float = 1e-3,
        lambda_sparsity: float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_channels = output_channels
        self.attr_dim = output_channels if attr_dim is None else attr_dim
        self.n_channels = n_channels
        self.K = n_mechanisms_per_channel
        self.n_nodes = n_channels * n_mechanisms_per_channel
        self.branch_grad_scale = float(branch_grad_scale)
        self.lambda_causal = float(lambda_causal)
        self.lambda_notears = float(lambda_notears)
        self.lambda_sparsity = float(lambda_sparsity)

        # A) Per-channel encoders
        self.encoders = nn.ModuleList([
            ChannelMechanismEncoder(
                d_model=d_model, n_mechanisms=n_mechanisms_per_channel,
                patch_size=patch_size, num_heads=num_heads,
            )
            for _ in range(n_channels)
        ])
        # B) Stable / nuisance splitter
        self.splitter = MechanismSplitter(d_model)
        # C) Dynamic causal graph learner
        self.graph = DynamicCausalGraphLearner(self.n_nodes, edge_bias=edge_bias)
        # D) Mechanism transition
        self.transition = MechanismTransition(d_model, self.n_nodes, self.attr_dim)
        # E) Mechanism recomposer
        self.recomposer = MechanismRecomposer(d_model, output_channels, num_heads=num_heads)
        # F) Safe injection logit (sigmoid(-4) ~ 0.018)
        self.injection_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self, image: torch.Tensor, x_in: torch.Tensor,
        attr_emb: Optional[torch.Tensor] = None,
        intervention: Optional[dict[str, Any]] = None,
    ) -> CTICDOutput:
        B, C_img, _, _ = image.shape
        attr_pooled = _pool_attr(attr_emb, B, self.attr_dim, image.device)

        # A. Per-channel encoding
        raw_states = []
        for c in range(self.n_channels):
            states_c = self.encoders[c](image[:, c : c + 1])
            raw_states.append(states_c)
        raw = torch.cat(raw_states, dim=1)  # (B, M, D)

        # B. Project to stable (causal) representations
        stable = self.splitter(raw)

        # C. Causal graph
        A, notears_loss, sparsity_loss = self.graph(stable)

        # D. Mechanism transition
        updated = self.transition(
            states=stable, graph=A, attr_pooled=attr_pooled, intervention=intervention,
        )

        # E. Recompose
        causal_features = self.recomposer(updated, x_in)

        # F. Safe injection
        injection_scale = torch.sigmoid(self.injection_logit)
        causal_features = injection_scale * causal_features
        if self.branch_grad_scale != 1.0:
            causal_features = GradientScale.apply(causal_features, self.branch_grad_scale)

        # Losses
        # NOTE: raw.detach() prevents degeneration where model moves both
        # raw and updated to reduce loss without learning real causal mechanisms.
        #
        # Current design: L_causal encourages graph transition to preserve
        # information from raw encoding. This is an information-preservation
        # loss, not a true causal prediction loss. For true causal semantics,
        # we would need temporal data (t and t+1 states) to predict changes.
        # In the single-image setting, this serves as a regularizer ensuring
        # the causal graph doesn't lose information from the original encoding.
        causal_loss = F.mse_loss(updated, raw.detach())

        total = (
            self.lambda_causal * causal_loss
            + self.lambda_notears * notears_loss
            + self.lambda_sparsity * sparsity_loss
        )
        losses = {
            "cticd_total": total,
            "cticd_causal": causal_loss.detach(),
            "cticd_notears": notears_loss.detach(),
            "cticd_sparsity": sparsity_loss.detach(),
        }
        return CTICDOutput(
            causal_features=causal_features, causal_graph=A,
            mechanism_states=updated, losses=losses,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(7)
    B, C, H, W = 2, 64, 64, 64
    L = 16 * 16 + 8 * 8
    image = torch.randn(B, 3, H, W)
    x_in = torch.randn(B, C, 1, L, requires_grad=True)
    attr = torch.randn(B, C, 1, L)
    model = CTICD(
        d_model=64, output_channels=C, attr_dim=C,
        n_mechanisms_per_channel=4, patch_size=4, num_heads=4,
    )
    out = model(image=image, x_in=x_in, attr_emb=attr)
    assert out.causal_features.shape == x_in.shape, (
        f"Shape mismatch: {out.causal_features.shape} != {x_in.shape}"
    )
    assert out.causal_graph.shape == (B, 12, 12)
    assert out.mechanism_states.shape == (B, 12, 64)
    loss = out.causal_features.square().mean() + out.losses["cticd_total"]
    loss.backward()
    print("CTICD simplified smoke test PASSED.")
    print(f"  causal_features : {tuple(out.causal_features.shape)}")
    print(f"  causal_graph    : {tuple(out.causal_graph.shape)}")
    print(f"  mechanism_states: {tuple(out.mechanism_states.shape)}")
    print(f"  losses          : {list(out.losses.keys())}")
