"""Text encoders for MATD framework.

Two text encoder implementations:

1. **MATDTextEncoder** -- wraps a pretrained sentence-transformers model
   (default ``all-MiniLM-L6-v2``, 384-dim) with optional linear projection
   to the model's working dimension.  Returns both per-token hidden states
   and a pooled sentence embedding.
2. **NullTextEncoder** -- learned null embeddings for classifier-free
   guidance (CFG) unconditional branch.

Reference: MATD framework design doc.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ==========================================================================
#  1. MATD Text Encoder
# ==========================================================================


class MATDTextEncoder(nn.Module):
    """Text encoder wrapping a pretrained transformer.

    Uses ``transformers.AutoTokenizer`` and ``transformers.AutoModel`` to
    encode a batch of text strings.  Supports:

    - **Frozen mode** (``freeze=True``): encoder parameters are frozen and
      require no gradients -- only the optional projection is trained.
    - **Projection**: a linear layer maps from the encoder's hidden
      dimension to the target ``model_dim`` when they differ.

    Outputs:

    - ``token_hidden``: (B, M, D) per-token hidden states (after projection
      if applicable).  M is the padded sequence length.
    - ``pooled``: (B, D) mean-pooled sentence embedding over non-padding
      positions.

    Example::

        enc = MATDTextEncoder(model_dim=256)
        token_hidden, pooled = enc(["a rising trend", "sharp spike at t=50"])
        # token_hidden: (2, M, 256),  pooled: (2, 256)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_dim: Optional[int] = None,
        freeze: bool = True,
        max_length: int = 128,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model identifier.
            model_dim:  Target output dimension.  If ``None`` or equal to
                        the encoder's hidden size, no projection is applied.
            freeze:     If ``True``, freeze all encoder parameters.
            max_length: Maximum tokenisation length (truncation bound).
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        encoder_dim: int = self.encoder.config.hidden_size

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Optional projection when encoder_dim != model_dim
        if model_dim is not None and model_dim != encoder_dim:
            self.projection = nn.Linear(encoder_dim, model_dim)
            self.output_dim = model_dim
        else:
            self.projection = nn.Identity()
            self.output_dim = encoder_dim

    # ------------------------------------------------------------------

    def forward(
        self,
        texts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of text strings.

        Args:
            texts: List of B text strings.

        Returns:
            token_hidden:  (B, M, D) per-token hidden states after projection.
            pooled:        (B, D) mean-pooled embedding over non-padding tokens.
            attention_mask: (B, M) HuggingFace-style mask (1 = real, 0 = pad).
        """
        device = next(self.parameters()).device

        # Tokenise with padding
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)           # (B, M)
        attention_mask = encoded["attention_mask"].to(device)  # (B, M)

        # Forward through the transformer encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (B, M, encoder_dim)

        # Project to model_dim
        token_hidden = self.projection(hidden)  # (B, M, D)

        # Mean pooling over non-padding positions
        mask_f = attention_mask.unsqueeze(-1).float()  # (B, M, 1)
        pooled = (token_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
        # pooled: (B, D)

        return token_hidden, pooled, attention_mask


# ==========================================================================
#  2. Null Text Encoder (for CFG)
# ==========================================================================


class NullTextEncoder(nn.Module):
    """Learned null text embeddings for classifier-free guidance.

    Provides a fixed, learned embedding that replaces real text conditioning
    during unconditional generation.  Both the per-token and pooled
    embeddings are learnable parameters broadcast to the requested batch
    size.

    Usage in CFG::

        null_enc = NullTextEncoder(dim=256, n_tokens=16)
        null_tok, null_pool = null_enc(batch_size=B)
        # Use null_tok, null_pool in place of real text embeddings
        # during unconditional forward passes.
    """

    def __init__(
        self,
        dim: int,
        n_tokens: int = 16,
    ) -> None:
        """
        Args:
            dim:      Embedding dimension (should match the model D).
            n_tokens: Number of null token positions M.
        """
        super().__init__()
        self.dim = dim
        self.n_tokens = n_tokens

        self.null_token = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.null_pool = nn.Parameter(torch.randn(1, dim) * 0.02)

    def forward(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate null embeddings for a batch.

        Args:
            batch_size: B -- number of samples in the batch.

        Returns:
            null_tokens: (B, M, dim) per-token null embeddings.
            null_pooled: (B, dim) pooled null embedding.
        """
        null_tokens = self.null_token.expand(batch_size, -1, -1)
        null_pooled = self.null_pool.expand(batch_size, -1)
        return null_tokens, null_pooled

