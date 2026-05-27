"""Spectral Causal Mechanism Operator Network (SCMON).

Discovers causal structure in the VAE latent space by decomposing it into
independent mechanism subspaces, learning spectral signatures for each, and
building a causal graph from spectral compatibility.

Reference design doc: 综合创新之CausalDiscoveryModule (CDM).md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class SCMONOutput:
    """Return value of :meth:`SCMON.forward`."""

    causal_features: torch.Tensor   # (L_total, latent_dim) -- fused output
    causal_graph: torch.Tensor      # (K, K) soft adjacency
    mechanism_states: list[torch.Tensor]  # K x (L_total, latent_dim)
    spectral_signatures: list[torch.Tensor]  # K x (n_freq_bins,)
    losses: dict[str, torch.Tensor]  # individual loss terms


# ---------------------------------------------------------------------------
# Component A: Mechanism Subspace Decomposition
# ---------------------------------------------------------------------------

class MechanismSubspaceDecomposer(nn.Module):
    """Learnable soft assignment of latent tokens to K mechanism subspaces.

    Each mechanism produces a soft mask over the latent dimensions:

        mask_k = softmax(A_k . z / tau)   over latent_dim
        z_k = mask_k * z                   soft selection of dimensions
        z_k = to_mechanism_k(z_k)          project to mechanism subspace
        z_k = out_proj_k(z_k)              project back to latent_dim

    Orthogonality regularisation encourages subspaces to be disjoint.
    """

    def __init__(self, latent_dim: int, n_mechanisms: int, mechanism_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mechanisms = n_mechanisms
        self.mechanism_dim = mechanism_dim

        # Per-mechanism soft-mask projection: latent_dim -> latent_dim
        self.projections = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(n_mechanisms)
        ])
        # Temperature for softmax (learnable)
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # tau = 1.0

        # Projection to mechanism subspace: latent_dim -> mechanism_dim
        self.to_mechanism = nn.ModuleList([
            nn.Linear(latent_dim, mechanism_dim) for _ in range(n_mechanisms)
        ])

        # Output projection: mechanism_dim -> latent_dim (recompose back)
        self.output_projs = nn.ModuleList([
            nn.Linear(mechanism_dim, latent_dim) for _ in range(n_mechanisms)
        ])

        self._init_weights()

    def _init_weights(self):
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
        for proj in self.to_mechanism:
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
        for proj in self.output_projs:
            nn.init.xavier_uniform_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)

    def forward(
        self, z0: torch.Tensor, ts_shape: torch.LongTensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Args:
            z0: (L_total, latent_dim) detached VAE latent.
            ts_shape: (B, 1) per-sample lengths.

        Returns:
            mechanism_states: K x (L_total, latent_dim) projected states.
            orth_loss: scalar orthogonality regularisation.
        """
        tau = self.log_tau.exp().clamp(min=0.1, max=5.0)

        mechanism_states = []
        for k in range(self.n_mechanisms):
            # Soft assignment mask over latent_dim
            logits = self.projections[k](z0) / tau        # (L, latent_dim)
            mask = torch.softmax(logits, dim=-1)           # (L, latent_dim)
            z_k = mask * z0                                # (L, latent_dim)
            # Project to mechanism subspace
            z_k = self.to_mechanism[k](z_k)                # (L, mechanism_dim)
            # Project back to latent_dim for downstream use
            z_k = self.output_projs[k](z_k)                # (L, latent_dim)
            mechanism_states.append(z_k)

        # Orthogonality loss: encourage subspaces to be disjoint
        orth_loss = z0.new_tensor(0.0)
        for j in range(self.n_mechanisms):
            for k in range(j + 1, self.n_mechanisms):
                sim = F.cosine_similarity(
                    mechanism_states[j], mechanism_states[k], dim=-1,
                )
                orth_loss = orth_loss + sim.pow(2).mean()

        return mechanism_states, orth_loss


# ---------------------------------------------------------------------------
# Component B: Spectral Signature Learning
# ---------------------------------------------------------------------------

class SpectralSignatureLearner(nn.Module):
    """Extract per-mechanism spectral signatures via batched FFT.

    For each mechanism state z_k of shape (L_total, latent_dim):
      1. Pad per-sample segments to max length, do ONE batched FFT.
      2. Adaptive pool frequency dim to n_freq_bins.
      3. Learnable pool across latent dim (latent_dim -> 1 per bin).
      4. Average across batch, L2-normalise.

    The causal graph is then built from pairwise spectral compatibility.
    """

    def __init__(
        self,
        latent_dim: int,
        n_mechanisms: int,
        n_freq_bins: int = 32,
    ):
        super().__init__()
        self.n_mechanisms = n_mechanisms
        self.n_freq_bins = n_freq_bins
        self.latent_dim = latent_dim

        # Learnable frequency pooling: latent_dim -> 1 (per freq bin)
        self.freq_pool = nn.ModuleList([
            nn.Linear(latent_dim, 1) for _ in range(n_mechanisms)
        ])
        # Learnable direction parameters for graph edges
        self.direction_logits = nn.Parameter(
            torch.zeros(n_mechanisms, n_mechanisms),
        )
        # Temperature scaling for graph sigmoid
        self.graph_log_tau = nn.Parameter(torch.tensor(0.0))  # tau = 1.0

        self._init_weights()

    def _init_weights(self):
        for linear in self.freq_pool:
            nn.init.xavier_uniform_(linear.weight, gain=0.5)
            nn.init.zeros_(linear.bias)

    def forward(
        self, mechanism_states: list[torch.Tensor], ts_shape: torch.LongTensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Args:
            mechanism_states: K x (L_total, latent_dim)
            ts_shape: (B, 1) per-sample lengths.

        Returns:
            spectral_signatures: K x (n_freq_bins,)
            causal_graph: (K, K) soft adjacency in [0, 1].
        """
        lengths = ts_shape.flatten().tolist()
        B = len(lengths)

        spectral_signatures = []
        for k in range(self.n_mechanisms):
            z_k = mechanism_states[k]  # (L_total, latent_dim)

            # Split into per-sample segments
            segments = []
            offset = 0
            for b in range(B):
                L_b = int(lengths[b])
                segments.append(z_k[offset:offset + L_b])  # (L_b, latent_dim)
                offset += L_b

            # Pad all segments to max length in the batch
            padded = nn.utils.rnn.pad_sequence(
                segments, batch_first=True, padding_value=0.0,
            )  # (B, max_L, latent_dim)

            # One batched FFT along sequence dim
            fft_out = torch.fft.rfft(padded, dim=1)  # (B, F, latent_dim)
            mag = fft_out.abs()  # (B, F, latent_dim)

            # Adaptive pool frequency dim to n_freq_bins
            mag = F.adaptive_avg_pool1d(
                mag.permute(0, 2, 1),  # (B, latent_dim, F)
                self.n_freq_bins,
            )  # (B, latent_dim, n_freq_bins)
            mag = mag.permute(0, 2, 1)  # (B, n_freq_bins, latent_dim)

            # Learnable pooling across latent dim -> (B, n_freq_bins)
            mag_pooled = self.freq_pool[k](mag).squeeze(-1)  # (B, n_freq_bins)

            # Average across batch: (n_freq_bins,)
            phi_k = mag_pooled.mean(dim=0)

            # L2-normalise for cosine similarity
            phi_k = F.normalize(phi_k, dim=0)
            spectral_signatures.append(phi_k)

        # Build causal graph from spectral compatibility
        causal_graph = self._build_graph(spectral_signatures)

        return spectral_signatures, causal_graph

    def _build_graph(self, signatures: list[torch.Tensor]) -> torch.Tensor:
        """Compute soft causal graph from spectral signatures.

        W_jk = sigma(cos(phi_j, phi_k) * tau_jk / temp)
        """
        K = self.n_mechanisms
        device = signatures[0].device

        # Stack signatures: (K, n_freq_bins)
        Phi = torch.stack(signatures, dim=0)

        # Pairwise cosine similarity: (K, K)
        cos_sim = F.cosine_similarity(
            Phi.unsqueeze(1), Phi.unsqueeze(0), dim=-1,
        )

        # Temperature scaling
        temp = self.graph_log_tau.exp().clamp(min=0.1, max=5.0)

        # Apply learnable direction: asymmetric via tau_jk
        tau = torch.sigmoid(self.direction_logits)  # (K, K)
        raw_graph = cos_sim * tau / temp

        # Sigmoid to [0, 1]
        causal_graph = torch.sigmoid(raw_graph)

        # Zero diagonal (no self-loops)
        causal_graph = causal_graph * (1 - torch.eye(K, device=device))

        return causal_graph


# ---------------------------------------------------------------------------
# Component C: Causal Mechanism Transition
# ---------------------------------------------------------------------------

class CausalMechanismTransition(nn.Module):
    """Per-mechanism transition with per-token AdaLN regime conditioning.

    For each mechanism k:
        input_k = sum_{j in pa(k)} W_jk . z_j
        z_k' = AdaLN(MLP_k(input_k); regime) + z_k

    Each mechanism has an independent MLP (ICM principle).
    Regime is expanded per-token via ts_shape for proper per-sample conditioning.
    """

    def __init__(
        self,
        latent_dim: int,
        n_mechanisms: int,
        hidden_dim: int = 128,
        regime_dim: int = 128,
    ):
        super().__init__()
        self.n_mechanisms = n_mechanisms
        self.latent_dim = latent_dim

        # Per-mechanism MLPs (ICM: independent parameters)
        self.mechanism_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(hidden_dim, latent_dim),
            )
            for _ in range(n_mechanisms)
        ])

        # Per-mechanism AdaLN: gamma, beta from regime encoding
        self.adaln_gamma = nn.ModuleList([
            nn.Linear(regime_dim, latent_dim) for _ in range(n_mechanisms)
        ])
        self.adaln_beta = nn.ModuleList([
            nn.Linear(regime_dim, latent_dim) for _ in range(n_mechanisms)
        ])
        # Layer norm (non-adaptive part)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(n_mechanisms)
        ])

        self._init_weights()

    def _init_weights(self):
        for mlp in self.mechanism_mlps:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        # Init gamma to 1, beta to 0 (identity at start)
        for gamma_linear in self.adaln_gamma:
            nn.init.zeros_(gamma_linear.weight)
            nn.init.ones_(gamma_linear.bias)
        for beta_linear in self.adaln_beta:
            nn.init.zeros_(beta_linear.weight)
            nn.init.zeros_(beta_linear.bias)

    def _expand_regime(
        self, regime: torch.Tensor, ts_shape: torch.LongTensor,
    ) -> torch.Tensor:
        """Expand per-sample regime to per-token regime.

        Args:
            regime: (B, regime_dim) per-sample regime encoding.
            ts_shape: (B, 1) per-sample lengths.

        Returns:
            regime_expanded: (L_total, regime_dim) per-token regime.
        """
        lengths = ts_shape.flatten().tolist()
        parts = []
        for i, L_i in enumerate(lengths):
            parts.append(regime[i].unsqueeze(0).expand(int(L_i), -1))
        return torch.cat(parts, dim=0)  # (L_total, regime_dim)

    def forward(
        self,
        mechanism_states: list[torch.Tensor],
        causal_graph: torch.Tensor,
        regime: torch.Tensor,
        ts_shape: torch.LongTensor,
    ) -> list[torch.Tensor]:
        """
        Args:
            mechanism_states: K x (L_total, latent_dim)
            causal_graph: (K, K) soft adjacency
            regime: (regime_dim,) or (B, regime_dim) regime encoding
            ts_shape: (B, 1) per-sample lengths.

        Returns:
            updated_states: K x (L_total, latent_dim)
        """
        # Ensure regime is 2-D: (B, regime_dim)
        if regime.ndim == 1:
            regime = regime.unsqueeze(0)  # (1, regime_dim)

        # Expand regime to per-token: (L_total, regime_dim)
        regime_expanded = self._expand_regime(regime, ts_shape)

        updated = []
        for k in range(self.n_mechanisms):
            # Weighted aggregation of parent states
            parent_input = mechanism_states[k].new_zeros(
                mechanism_states[k].shape,
            )
            for j in range(self.n_mechanisms):
                if j == k:
                    continue
                w_jk = causal_graph[j, k]
                parent_input = parent_input + w_jk * mechanism_states[j]

            # Mechanism MLP
            h = self.mechanism_mlps[k](parent_input)  # (L, latent_dim)

            # AdaLN modulation per-token
            gamma = self.adaln_gamma[k](regime_expanded)  # (L_total, latent_dim)
            beta = self.adaln_beta[k](regime_expanded)    # (L_total, latent_dim)
            h = self.layer_norms[k](h)
            h = gamma * h + beta

            # Residual connection
            updated.append(mechanism_states[k] + h)

        return updated


# ---------------------------------------------------------------------------
# Component D: Mechanism Recomposition
# ---------------------------------------------------------------------------

class MechanismRecomposer(nn.Module):
    """Recombine K mechanism states via cross-attention fusion.

    Each mechanism queries all others via multi-head attention, plus a
    learned importance gate as residual path.
    """

    def __init__(self, latent_dim: int, n_mechanisms: int, num_heads: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mechanisms = n_mechanisms

        # Cross-attention: each mechanism queries all others
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Learnable mechanism importance weights (residual gate)
        self.importance = nn.Linear(latent_dim * n_mechanisms, n_mechanisms)
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.importance.weight)
        nn.init.zeros_(self.importance.bias)
        # Zero-init for stable residual start
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)
        # Xavier init for cross-attention
        nn.init.xavier_uniform_(self.cross_attn.in_proj_weight)
        nn.init.zeros_(self.cross_attn.in_proj_bias)
        nn.init.xavier_uniform_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, mechanism_states: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            mechanism_states: K x (L_total, latent_dim)

        Returns:
            causal_features: (L_total, latent_dim)
        """
        # Stack: (L, K, latent_dim)
        stacked = torch.stack(mechanism_states, dim=1)
        L = stacked.shape[0]

        # Cross-attention: each mechanism queries all mechanism states
        attended, _ = self.cross_attn(
            query=stacked, key=stacked, value=stacked,
        )  # (L, K, latent_dim)

        # Importance gate (residual path)
        flat = stacked.reshape(L, -1)  # (L, K * latent_dim)
        weights = torch.softmax(self.importance(flat), dim=-1)  # (L, K)

        # Gated fusion: weighted sum of attended outputs
        fused = (attended * weights.unsqueeze(-1)).sum(dim=1)  # (L, latent_dim)

        # Output projection (zero-init -> identity at start)
        return self.out_proj(fused)


# ---------------------------------------------------------------------------
# Graph regularisation losses
# ---------------------------------------------------------------------------

def compute_notears_acyclicity(graph: torch.Tensor) -> torch.Tensor:
    """NOTEARS acyclicity constraint: tr(exp(W .* W)) - K = 0 iff DAG.

    Args:
        graph: (K, K) soft adjacency matrix.
    Returns:
        Scalar loss, >= 0, = 0 iff graph is acyclic.
    """
    K = graph.shape[0]
    W_sq = graph * graph
    return torch.trace(torch.matrix_exp(W_sq)) - K


def compute_graph_sparsity(graph: torch.Tensor) -> torch.Tensor:
    """L1 sparsity penalty on causal graph."""
    return graph.abs().mean()


# ---------------------------------------------------------------------------
# Main SCMON Module
# ---------------------------------------------------------------------------

class SCMON(nn.Module):
    """Spectral Causal Mechanism Operator Network.

    Full pipeline: Subspace Decomposition -> Spectral Signatures ->
    Causal Graph -> Mechanism Transition -> Recomposition.

    Operates on z0.detach() from the VAE.  Returns causal_features
    that can be injected into DiT blocks.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_mechanisms: int = 8,
        mechanism_dim: int = 8,
        hidden_dim: int = 128,
        regime_dim: int = 128,
        n_freq_bins: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mechanisms = n_mechanisms
        self.regime_dim = regime_dim

        # Stage A
        self.decomposer = MechanismSubspaceDecomposer(
            latent_dim, n_mechanisms, mechanism_dim,
        )
        # Stage B
        self.spectral_learner = SpectralSignatureLearner(
            latent_dim, n_mechanisms, n_freq_bins,
        )
        # Stage C
        self.transition = CausalMechanismTransition(
            latent_dim, n_mechanisms, hidden_dim, regime_dim,
        )
        # Stage D
        self.recomposer = MechanismRecomposer(latent_dim, n_mechanisms)

        # Loss weights (fixed)
        self.lambda_graph = 0.1
        self.lambda_orth = 0.01
        self.lambda_spectral = 0.05

    def forward(
        self,
        z0: torch.Tensor,
        ts_shape: torch.LongTensor,
        regime: torch.Tensor | None = None,
    ) -> SCMONOutput:
        """
        Args:
            z0: (L_total, latent_dim) -- detached VAE latent.
            ts_shape: (B, 1) per-sample lengths.
            regime: (regime_dim,) or (B, regime_dim) regime encoding
                (e.g. from text encoder).  If None, uses zeros.

        Returns:
            SCMONOutput with causal_features, graph, losses, etc.
        """
        L_total = z0.shape[0]
        device = z0.device

        # Default regime: zeros (no modulation)
        if regime is None:
            regime = z0.new_zeros(self.regime_dim)

        # Stage A: Decompose into mechanism subspaces
        mechanism_states, orth_loss = self.decomposer(z0, ts_shape)

        # Stage B: Learn spectral signatures and build causal graph
        signatures, causal_graph = self.spectral_learner(
            mechanism_states, ts_shape,
        )

        # Stage C: Causal mechanism transitions
        updated_states = self.transition(
            mechanism_states, causal_graph, regime, ts_shape,
        )

        # Stage D: Recompose into unified causal features
        causal_features = self.recomposer(updated_states)

        # Compute losses
        notears_loss = compute_notears_acyclicity(causal_graph)
        sparsity_loss = compute_graph_sparsity(causal_graph)

        # Spectral consistency: updated signatures should resemble original
        spectral_loss = z0.new_tensor(0.0)
        updated_sigs, _ = self.spectral_learner(updated_states, ts_shape)
        for k in range(self.n_mechanisms):
            spectral_loss = spectral_loss + F.mse_loss(
                updated_sigs[k], signatures[k].detach(),
            )

        losses = {
            "scmon_notears": notears_loss,
            "scmon_sparsity": sparsity_loss,
            "scmon_orth": orth_loss,
            "scmon_spectral": spectral_loss,
            "scmon_graph_total": (
                self.lambda_graph * (notears_loss + sparsity_loss)
                + self.lambda_orth * orth_loss
                + self.lambda_spectral * spectral_loss
            ),
        }

        return SCMONOutput(
            causal_features=causal_features,
            causal_graph=causal_graph,
            mechanism_states=updated_states,
            spectral_signatures=signatures,
            losses=losses,
        )
