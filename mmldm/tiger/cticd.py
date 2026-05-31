"""Dynamic CTICD for TIGER.

CTICD = Channel-Transform-informed Causal Dynamics.

This paper-ready version fixes the earlier static reconstruction bottleneck by
learning lagged mechanism graphs and predicting future mechanism states.  It is
still a neural causal-inductive-bias module rather than a fully identifiable
causal discovery algorithm; the point is that graph direction is now grounded in
ordered temporal segments instead of symmetric same-frame similarity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["CTICD", "CTICDOutput", "GradientScale"]


@dataclass
class CTICDOutput:
    """Return value of ``CTICD.forward``.

    causal_features:
        ``(B, C, 1, L)`` tensor ready to add to DiT tokens.
    causal_graph:
        ``(B, M, M)`` instantaneous DAG adjacency ``A0``.
    lagged_graphs:
        ``(B, P, M, M)`` lagged dynamic graphs where ``A[:,p,j,i]`` means
        mechanism ``j`` at time ``t-p-1`` influences mechanism ``i`` at time
        ``t``.
    mechanism_states:
        ``(B, S, M, D)`` stable temporal mechanism states.
    losses:
        Auxiliary losses used by the diffusion trainer.
    """

    causal_features: torch.Tensor
    causal_graph: torch.Tensor
    lagged_graphs: torch.Tensor
    mechanism_states: torch.Tensor
    losses: dict[str, torch.Tensor]


class GradientScale(torch.autograd.Function):
    """Identity in forward; scales incoming gradient by ``scale`` in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def _zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _pool_attr(attr_emb: Optional[torch.Tensor], batch: int, attr_dim: int, device: torch.device) -> torch.Tensor:
    if attr_emb is None:
        return torch.zeros(batch, attr_dim, device=device)
    if attr_emb.dim() == 4:
        pooled = attr_emb.mean(dim=(2, 3))
    elif attr_emb.dim() == 3:
        pooled = attr_emb.mean(dim=-1)
    elif attr_emb.dim() == 2:
        pooled = attr_emb
    else:
        raise ValueError(f"attr_emb must be 2D/3D/4D or None, got {tuple(attr_emb.shape)}")
    if pooled.shape[-1] != attr_dim:
        raise ValueError(f"pooled attr dim={pooled.shape[-1]} but expected {attr_dim}")
    return pooled


class ChannelTemporalMechanismEncoder(nn.Module):
    """Encode one transform channel into temporal mechanism states.

    The TS-image width axis is treated as the ordered time axis.  The encoder
    patches the channel image, splits patch columns into ``n_segments`` temporal
    bins, and lets K learnable mechanism queries attend to each bin.
    """

    def __init__(
        self,
        d_model: int,
        n_mechanisms: int,
        n_segments: int,
        patch_size: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.n_mechanisms = n_mechanisms
        self.n_segments = n_segments
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(1, d_model),
            nn.GELU(approximate="tanh"),
        )
        self.mechanism_queries = nn.Parameter(torch.empty(n_mechanisms, d_model))
        nn.init.normal_(self.mechanism_queries, std=1.0 / math.sqrt(d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(2 * d_model, d_model),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, channel: torch.Tensor) -> torch.Tensor:
        if channel.dim() != 4 or channel.shape[1] != 1:
            raise ValueError(f"channel must be (B,1,H,W), got {tuple(channel.shape)}")
        B = channel.shape[0]
        feat = self.patch_embed(channel)  # (B,D,h,w)
        _, D, h, w = feat.shape
        segments = min(self.n_segments, max(1, w))
        edges = torch.linspace(0, w, steps=segments + 1, device=feat.device)
        edges = edges.round().long()
        states = []
        queries = self.mechanism_queries.unsqueeze(0).expand(B, -1, -1)
        for s in range(segments):
            start = int(edges[s].item())
            end = int(edges[s + 1].item())
            if end <= start:
                end = min(w, start + 1)
            spatial = feat[:, :, :, start:end].flatten(2).transpose(1, 2).contiguous()
            attended, _ = self.attn(query=queries, key=spatial, value=spatial, need_weights=False)
            state = self.norm(attended)
            state = state + self.ffn(state)
            states.append(state)
        # If n_segments > patch width, repeat the last state to keep a fixed shape.
        # Detach to prevent gradient duplication (Nx gradient on the last real segment).
        while len(states) < self.n_segments:
            states.append(states[-1].detach())
        return torch.stack(states, dim=1)  # (B,S,K,D)


class MechanismSplitter(nn.Module):
    """Stable mechanism representation with zero-init residual projection."""

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


class DynamicCausalGraphLearner(nn.Module):
    """Learn instantaneous DAG and directed lagged graphs over mechanisms."""

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        max_lag: int = 2,
        edge_bias: float = -4.0,
        lag_edge_bias: float = -2.5,
        init_tau: float = 1.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.max_lag = max_lag
        self.base_dag_logits = nn.Parameter(torch.full((n_nodes, n_nodes), float(edge_bias)))
        self.lag_logits = nn.Parameter(torch.full((max_lag, n_nodes, n_nodes), float(lag_edge_bias)))
        self.q_child = nn.Linear(d_model, d_model, bias=False)
        self.k_parent = nn.Linear(d_model, d_model, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))

    @staticmethod
    def _acyclicity_loss(A: torch.Tensor) -> torch.Tensor:
        B, M, _ = A.shape
        W_sq = (A * A).clamp(max=2.0)  # prevent matrix_exp overflow (e^24 → Inf)
        expm = torch.linalg.matrix_exp(W_sq)
        h = expm.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - M
        h = h.clamp(max=100.0)  # prevent h² gradient explosion
        return (h * h).mean()

    def _asym_score(self, parent_states: torch.Tensor, child_states: torch.Tensor) -> torch.Tensor:
        # Returns score[j,i] = parent_j -> child_i.
        # eps=1e-8 prevents NaN when mechanism states degenerate to near-zero.
        parent = F.normalize(self.k_parent(parent_states), dim=-1, eps=1e-8)
        child = F.normalize(self.q_child(child_states), dim=-1, eps=1e-8)
        tau = self.log_tau.exp().clamp(min=0.1, max=5.0)
        score = torch.bmm(parent, child.transpose(1, 2)) / tau
        # Clamp to prevent extreme logits → sigmoid saturation → matrix_exp overflow
        return score.clamp(-20.0, 20.0)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if states.dim() != 4:
            raise ValueError(f"states must be (B,S,M,D), got {tuple(states.shape)}")
        B, S, M, _ = states.shape
        if M != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {M}")

        mean_states = states.mean(dim=1)
        score0 = self._asym_score(mean_states, mean_states)
        A0 = torch.sigmoid(self.base_dag_logits.unsqueeze(0) + score0)
        eye = torch.eye(M, device=states.device, dtype=states.dtype).unsqueeze(0)
        A0 = A0 * (1.0 - eye)
        notears_loss = self._acyclicity_loss(A0)

        lag_graphs = []
        for lag in range(1, self.max_lag + 1):
            if S > lag:
                parent_states = states[:, :-lag].mean(dim=1)
                child_states = states[:, lag:].mean(dim=1)
                score_lag = self._asym_score(parent_states, child_states)
            else:
                score_lag = torch.zeros(B, M, M, device=states.device, dtype=states.dtype)
            Al = torch.sigmoid(self.lag_logits[lag - 1].unsqueeze(0) + score_lag)
            lag_graphs.append(Al)
        Alags = torch.stack(lag_graphs, dim=1)  # (B,P,M,M)

        sparsity_loss = 0.5 * A0.mean() + 0.5 * Alags.mean()
        return A0, Alags, notears_loss, sparsity_loss


class LaggedMechanismPredictor(nn.Module):
    """Predict future mechanism states from lagged graph parents."""

    def __init__(self, d_model: int, n_nodes: int, attr_dim: int, max_lag: int = 2, hidden_mult: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = n_nodes
        self.max_lag = max_lag
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
        self.film = _zero_module(nn.Linear(attr_dim, 2 * n_nodes * d_model))
        for mlp in self.mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def _aggregate_lagged(self, states: torch.Tensor, lagged_graphs: torch.Tensor) -> torch.Tensor:
        B, S, M, D = states.shape
        P = lagged_graphs.shape[1]
        start = min(max(1, P), S - 1)
        if S <= start:
            return states[:, -1:]  # keep grad; constructor validates S > P
        target_len = S - start
        parent = torch.zeros(B, target_len, M, D, device=states.device, dtype=states.dtype)
        for lag in range(1, P + 1):
            graph = lagged_graphs[:, lag - 1]  # (B,M,M), j -> i
            past = states[:, start - lag:S - lag]
            parent = parent + torch.einsum("bji,btjd->btid", graph, past)
        return parent / float(max(1, P))

    def forward(
        self,
        states: torch.Tensor,
        lagged_graphs: torch.Tensor,
        attr_pooled: torch.Tensor,
        intervention: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, M, D = states.shape
        P = lagged_graphs.shape[1]
        start = min(max(1, P), S - 1)

        graph_used = lagged_graphs
        node = None
        value = None
        strength = 1.0
        if intervention is not None:
            node = int(intervention["node"])
            if not (0 <= node < M):
                raise ValueError(f"intervention node {node} out of range for M={M}")
            strength = float(intervention.get("strength", 1.0))
            if bool(intervention.get("cut_incoming", True)):
                graph_used = lagged_graphs.clone()
                graph_used[:, :, :, node] = 0.0
            value = intervention.get("value", None)
            if value is not None:
                value = value.to(device=states.device, dtype=states.dtype)
                if value.dim() == 1:
                    value = value.unsqueeze(0).expand(B, -1)
                elif value.dim() == 3 and value.shape[1] == 1:
                    value = value[:, 0, :]
                if value.shape != (B, D):
                    raise ValueError(f"intervention value must broadcast to {(B,D)}, got {tuple(value.shape)}")

        parent = self._aggregate_lagged(states, graph_used)  # (B,T,M,D)
        target = states[:, start:] if S > start else states[:, -1:]

        gamma_beta = self.film(attr_pooled).view(B, M, 2, D)
        gamma = gamma_beta[:, :, 0, :]
        beta = gamma_beta[:, :, 1, :]

        pred_parts = []
        for k in range(M):
            x = parent[:, :, k, :]
            delta = self.mlps[k](x)
            delta = self.norms[k](delta)
            delta = delta * (1.0 + gamma[:, None, k, :]) + beta[:, None, k, :]
            pred_parts.append(parent[:, :, k, :] + delta)
        pred = torch.stack(pred_parts, dim=2)

        if intervention is not None and value is not None:
            # Out-of-place to avoid autograd inplace-modification errors.
            last_frame = pred[:, -1:]
            last_frame_modified = last_frame.clone()
            last_frame_modified[:, 0, node, :] = (
                (1.0 - strength) * last_frame[:, 0, node, :] + strength * value
            )
            pred = torch.cat([pred[:, :-1], last_frame_modified], dim=1)

        # Last predicted state is used for causal feature injection.
        causal_state = pred[:, -1]
        return pred, target, causal_state


class MechanismRecomposer(nn.Module):
    """Cross-attention from DiT token queries to mechanism state key/values."""

    def __init__(self, d_model: int, output_channels: int, num_heads: int = 4):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.in_proj = nn.Linear(output_channels, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = _zero_module(nn.Linear(d_model, output_channels))

    def forward(self, mechanism_states: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
        q = x_in.squeeze(2).transpose(1, 2).contiguous()
        q = self.in_proj(q)
        attended, _ = self.cross_attn(query=q, key=mechanism_states, value=mechanism_states, need_weights=False)
        attended = self.norm(attended + q)
        out = self.out_proj(attended)
        return out.transpose(1, 2).unsqueeze(2).contiguous()


class CTICD(nn.Module):
    """Dynamic causal mechanism module for TIGER-DiT.

    Losses
    ------
    ``cticd_pred``
        Future mechanism prediction loss from lagged graph parents.
    ``cticd_notears``
        NOTEARS acyclicity penalty on the instantaneous graph ``A0``.
    ``cticd_sparsity``
        L1-style sparsity pressure on both instantaneous and lagged graphs.
    ``cticd_smooth``
        Small temporal smoothness regularizer on mechanism states.
    """

    def __init__(
        self,
        d_model: int,
        output_channels: int,
        attr_dim: Optional[int] = None,
        n_channels: int = 3,
        n_mechanisms_per_channel: int = 4,
        n_segments: int = 8,
        max_lag: int = 2,
        patch_size: int = 4,
        num_heads: int = 4,
        edge_bias: float = -4.0,
        lag_edge_bias: float = -2.5,
        branch_grad_scale: float = 0.2,
        lambda_causal: float = 1.0,
        lambda_notears: float = 1e-3,
        lambda_sparsity: float = 1e-2,
        lambda_smooth: float = 1e-3,
        injection_init: float = -4.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_channels = output_channels
        self.attr_dim = attr_dim or output_channels
        self.n_channels = n_channels
        self.n_mechanisms_per_channel = n_mechanisms_per_channel
        self.n_nodes = n_channels * n_mechanisms_per_channel
        self.n_segments = n_segments
        self.max_lag = max_lag
        self.branch_grad_scale = float(branch_grad_scale)
        self.lambda_causal = float(lambda_causal)
        self.lambda_notears = float(lambda_notears)
        self.lambda_sparsity = float(lambda_sparsity)
        self.lambda_smooth = float(lambda_smooth)

        self.channel_encoders = nn.ModuleList([
            ChannelTemporalMechanismEncoder(
                d_model=d_model,
                n_mechanisms=n_mechanisms_per_channel,
                n_segments=n_segments,
                patch_size=patch_size,
                num_heads=num_heads,
            )
            for _ in range(n_channels)
        ])
        self.splitter = MechanismSplitter(d_model)
        self.graph_learner = DynamicCausalGraphLearner(
            n_nodes=self.n_nodes,
            d_model=d_model,
            max_lag=max_lag,
            edge_bias=edge_bias,
            lag_edge_bias=lag_edge_bias,
        )
        # Validate temporal constraint: need more segments than lags for
        # meaningful lagged prediction.  Without this, _aggregate_lagged
        # produces empty slices and crashes with shape mismatches.
        if n_segments <= max_lag:
            raise ValueError(
                f"n_segments ({n_segments}) must be > max_lag ({max_lag}) "
                f"to have enough temporal bins for lagged parent aggregation."
            )

        self.predictor = LaggedMechanismPredictor(d_model, self.n_nodes, self.attr_dim, max_lag=max_lag)
        self.recomposer = MechanismRecomposer(d_model, output_channels, num_heads=num_heads)
        self.injection_logit = nn.Parameter(torch.tensor(float(injection_init)))

    def encode_mechanisms(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(f"image must be (B,C,H,W), got {tuple(image.shape)}")
        if image.shape[1] < self.n_channels:
            raise ValueError(f"image has {image.shape[1]} channels, expected at least {self.n_channels}")
        states = []
        for c in range(self.n_channels):
            states.append(self.channel_encoders[c](image[:, c:c + 1]))
        return torch.cat(states, dim=2)  # (B,S,M,D)

    def forward(
        self,
        image: torch.Tensor,
        x_in: torch.Tensor,
        attr_emb: Optional[torch.Tensor] = None,
        clean_image: Optional[torch.Tensor] = None,
        diffusion_emb: Optional[torch.Tensor] = None,
        intervention: Optional[dict[str, Any]] = None,
    ) -> CTICDOutput:
        del diffusion_emb  # kept for a stable public signature
        B = image.shape[0]
        device = image.device
        source_image = clean_image if clean_image is not None else image

        raw = self.encode_mechanisms(source_image.float())
        # NaN guard: if encoder produces NaN, return zero features + zero losses
        if not torch.isfinite(raw).all():
            return CTICDOutput(
                causal_features=torch.zeros_like(x_in),
                causal_graph=torch.zeros(B, self.n_nodes, self.n_nodes, device=device),
                lagged_graphs=torch.zeros(B, self.max_lag, self.n_nodes, self.n_nodes, device=device),
                mechanism_states=torch.zeros(B, self.n_segments, self.n_nodes, self.d_model, device=device),
                losses={"cticd_total": torch.tensor(0.0, device=device),
                        "cticd_pred": torch.tensor(0.0, device=device),
                        "cticd_notears": torch.tensor(0.0, device=device),
                        "cticd_sparsity": torch.tensor(0.0, device=device),
                        "cticd_smooth": torch.tensor(0.0, device=device)},
            )
        stable = self.splitter(raw)
        A0, Alags, notears_loss, sparsity_loss = self.graph_learner(stable)
        # NaN guard: if graph produces NaN/Inf, use zero graph
        if not torch.isfinite(A0).all():
            A0 = torch.zeros(B, self.n_nodes, self.n_nodes, device=device)
            Alags = torch.zeros(B, self.max_lag, self.n_nodes, self.n_nodes, device=device)
            notears_loss = torch.tensor(0.0, device=device)
            sparsity_loss = torch.tensor(0.0, device=device)
        attr_pooled = _pool_attr(attr_emb, B, self.attr_dim, device)

        pred, target, causal_state = self.predictor(stable, Alags, attr_pooled, intervention=intervention)
        pred_loss = F.mse_loss(pred, target.detach())
        if stable.shape[1] > 1:
            smooth_loss = (stable[:, 1:] - stable[:, :-1]).pow(2).mean()
        else:
            smooth_loss = stable.new_zeros(())

        causal_features = self.recomposer(causal_state, x_in)
        injection_scale = torch.sigmoid(self.injection_logit)
        causal_features = injection_scale * causal_features
        if self.branch_grad_scale != 1.0:
            causal_features = GradientScale.apply(causal_features, self.branch_grad_scale)

        total = (
            self.lambda_causal * pred_loss
            + self.lambda_notears * notears_loss
            + self.lambda_sparsity * sparsity_loss
            + self.lambda_smooth * smooth_loss
        )
        losses = {
            "cticd_total": total,
            "cticd_pred": pred_loss.detach(),
            "cticd_causal": pred_loss.detach(),  # backward-compatible log key
            "cticd_notears": notears_loss.detach(),
            "cticd_sparsity": sparsity_loss.detach(),
            "cticd_smooth": smooth_loss.detach(),
            "cticd_edge_density": A0.detach().mean(),
            "cticd_lag_edge_density": Alags.detach().mean(),
        }
        return CTICDOutput(
            causal_features=causal_features,
            causal_graph=A0,
            lagged_graphs=Alags,
            mechanism_states=stable,
            losses=losses,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 64, 64
    image = torch.rand(B, C, H, W)
    x_in = torch.randn(B, 32, 1, 64)
    attr = torch.randn(B, 32, 1, 64)
    model = CTICD(d_model=32, output_channels=32, attr_dim=32, n_segments=4, max_lag=2)
    out = model(image=image, clean_image=image, x_in=x_in, attr_emb=attr)
    assert out.causal_features.shape == x_in.shape
    assert out.causal_graph.shape == (B, 12, 12)
    assert out.lagged_graphs.shape == (B, 2, 12, 12)
    loss = out.causal_features.square().mean() + out.losses["cticd_total"]
    loss.backward()
    print("CTICD dynamic smoke test passed")
