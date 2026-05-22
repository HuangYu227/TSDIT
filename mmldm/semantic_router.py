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

"""Semantic Router for Adaptive Semantic Patching.

Implements the text-guided dynamic block allocation mechanism.
Given a text description, the router determines how to partition
the latent sequence into variable-sized blocks based on semantic
complexity.

The router is lightweight (~5M params) and operates with stop-gradient,
so it does not participate in the main model's backpropagation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticBoundaryDetector(nn.Module):
    """Lightweight Transformer that detects semantic boundaries in text.

    Takes text tokens and outputs per-token boundary logits indicating
    where semantic transitions occur.
    """

    def __init__(
        self,
        text_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.boundary_head = nn.Linear(hidden_dim, 1)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Detect semantic boundaries.

        Args:
            text_tokens: ``(B, L_text, text_dim)`` text token embeddings.

        Returns:
            ``(B, L_text)`` boundary logits (higher = more likely boundary).
        """
        x = self.text_proj(text_tokens)
        x = self.transformer(x)
        logits = self.boundary_head(x).squeeze(-1)  # (B, L_text)
        return logits


class AlignmentProjector(nn.Module):
    """Projects text-token-level boundaries to latent-token-level boundaries.

    Uses a learned linear projection to map from the text token sequence
    length to the latent token sequence length.
    """

    def __init__(self, text_dim: int = 128, latent_len: int = 96):
        super().__init__()
        # Simple linear projection: text features → latent-length boundary scores
        self.proj = nn.Linear(text_dim, latent_len)

    def forward(
        self,
        text_tokens: torch.Tensor,
        boundary_logits: torch.Tensor,
        n_latent: int,
    ) -> torch.Tensor:
        """Project boundaries to latent space.

        Args:
            text_tokens: ``(B, L_text, text_dim)``
            boundary_logits: ``(B, L_text)`` from SemanticBoundaryDetector
            n_latent: target latent sequence length.

        Returns:
            ``(B, n_latent)`` boundary scores for each latent position.
        """
        # Weight text tokens by boundary probability
        weights = torch.softmax(boundary_logits, dim=-1)  # (B, L_text)
        weighted_text = (text_tokens * weights.unsqueeze(-1)).sum(dim=1)  # (B, text_dim)

        # Project to latent-length scores
        scores = self.proj(weighted_text)  # (B, latent_len)
        if n_latent > scores.shape[-1]:
            scores = scores.repeat(1, (n_latent // scores.shape[-1]) + 1)[:, :n_latent]
        else:
            scores = scores[:, :n_latent]
        return scores


class BlockAllocator(nn.Module):
    """Allocates variable-sized blocks based on semantic density.

    Uses CDF-based equal-probability splitting to determine block
    boundaries.  High-density regions get smaller blocks (more tokens),
    low-density regions get larger blocks.
    """

    def __init__(
        self,
        min_block_size: int = 1,
        max_block_size: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.temperature = temperature

    def forward(
        self,
        boundary_scores: torch.Tensor,
        n_blocks: int,
    ) -> list[list[int]]:
        """Allocate blocks based on boundary scores.

        Args:
            boundary_scores: ``(B, n_latent)`` semantic density scores.
            n_blocks: target number of blocks per sample.

        Returns:
            List of B lists, each containing block sizes for that sample.
        """
        B, n_latent = boundary_scores.shape
        device = boundary_scores.device

        block_sizes_batch = []
        for b in range(B):
            scores = boundary_scores[b]  # (n_latent,)

            # Convert to density via softmax with temperature
            density = torch.softmax(scores / self.temperature, dim=-1)

            # Cumulative distribution
            cdf = torch.cumsum(density, dim=-1)

            # Equal-probability split points
            split_points = torch.linspace(0, 1, n_blocks + 1, device=device)[1:-1]

            # Find block boundaries by searching CDF
            boundaries = [0]
            for sp in split_points:
                idx = torch.searchsorted(cdf, sp)
                boundaries.append(min(int(idx.item()), n_latent))
            boundaries.append(n_latent)

            # Compute block sizes
            block_sizes = []
            for i in range(len(boundaries) - 1):
                bs = boundaries[i + 1] - boundaries[i]
                bs = max(self.min_block_size, min(self.max_block_size, bs))
                block_sizes.append(bs)

            # Adjust to sum to n_latent
            total = sum(block_sizes)
            if total != n_latent:
                # Adjust the last block
                diff = n_latent - total
                block_sizes[-1] = max(self.min_block_size, block_sizes[-1] + diff)
                # If still not matching, redistribute
                if sum(block_sizes) != n_latent:
                    block_sizes = self._redistribute(block_sizes, n_latent)

            block_sizes_batch.append(block_sizes)

        return block_sizes_batch

    def _redistribute(self, block_sizes: list[int], target: int) -> list[int]:
        """Redistribute block sizes to match target total (bounded greedy)."""
        current = sum(block_sizes)
        diff = target - current

        while diff != 0:
            for i in range(len(block_sizes)):
                if diff > 0 and block_sizes[i] < self.max_block_size:
                    block_sizes[i] += 1
                    diff -= 1
                elif diff < 0 and block_sizes[i] > self.min_block_size:
                    block_sizes[i] -= 1
                    diff += 1
                if diff == 0:
                    break
            else:
                # Safety: if no block can absorb, force into last block
                if diff > 0:
                    block_sizes[-1] += diff
                elif diff < 0:
                    block_sizes[-1] = max(self.min_block_size, block_sizes[-1] + diff)
                break

        return block_sizes


class SemanticRouter(nn.Module):
    """Semantic Router for Adaptive Semantic Patching.

    Orchestrates the boundary detection, alignment, and block allocation.
    Operates with stop-gradient to not affect main model training.

    Args:
        text_dim: dimension of input text embeddings.
        hidden_dim: hidden dimension of the boundary detector.
        n_latent: expected latent sequence length.
        min_block_size: minimum block size.
        max_block_size: maximum block size.
        temperature: softmax temperature for block allocation.
    """

    def __init__(
        self,
        text_dim: int = 128,
        hidden_dim: int = 256,
        n_latent: int = 96,
        min_block_size: int = 1,
        max_block_size: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.boundary_detector = SemanticBoundaryDetector(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
        )
        self.alignment_projector = AlignmentProjector(
            text_dim=text_dim,
            latent_len=n_latent,
        )
        self.block_allocator = BlockAllocator(
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            temperature=temperature,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        n_latent: int,
        n_blocks: int,
    ) -> list[list[int]]:
        """Route text to block sizes.

        Args:
            text_tokens: ``(B, L_text, text_dim)`` or ``(B, text_dim)``.
            n_latent: number of latent tokens to partition.
            n_blocks: target number of blocks.

        Returns:
            List of B lists of block sizes.
        """
        # Handle 2D input (single embedding per sample)
        if text_tokens.ndim == 2:
            text_tokens = text_tokens.unsqueeze(1)  # (B, 1, text_dim)

        boundary_logits = self.boundary_detector(text_tokens)
        boundary_scores = self.alignment_projector(
            text_tokens, boundary_logits, n_latent,
        )

        block_sizes = self.block_allocator(boundary_scores, n_blocks)
        return block_sizes
