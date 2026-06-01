"""TIGER DiT -- Diffusion Image Transformer.

Adapted from VerbalTS (E:\\Research\\TSG\\VerbalTS\\models\\diffusion\\verbalts.py)
for image+text conditioned generation.

Architecture mapping (VerbalTS -> TIGER):
    n_var (K)   -> n_patches_h for square images; K=1 for row-raster
    L (time)    -> n_patches_w / row-major temporal patch index
    TsPatchEmbedding    -> ImagePatchEmbedding or RowRasterPatchEmbedding
    SideEncoder_Var     -> ImageSideEncoder      (2D sinusoidal PE + learnable)
    PatchDecoder        -> ImagePatchDecoder or RowRasterPatchDecoder
    ResidualBlock       -> ResidualBlock          (dual-axis or temporal-only)
    multipatch_mixer    -> multipatch_mixer       (per-pixel scale mixing)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformer helpers (copied from VerbalTS)
# ---------------------------------------------------------------------------

def get_torch_trans(
    heads: int = 8,
    layers: int = 1,
    channels: int = 64,
    dim_feedforward: int | None = None,
):
    if dim_feedforward is None:
        dim_feedforward = 4 * channels
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=dim_feedforward,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_torch_cross_trans(
    heads: int = 8,
    layers: int = 1,
    channels: int = 64,
    dim_feedforward: int | None = None,
):
    if dim_feedforward is None:
        dim_feedforward = 4 * channels
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=dim_feedforward,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerDecoder(decoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


# ---------------------------------------------------------------------------
# Diffusion timestep embedding (identical to VerbalTS)
# ---------------------------------------------------------------------------

class DiffusionEmbedding(nn.Module):
    """Sinusoidal embedding + two-layer MLP for diffusion timestep."""

    def __init__(self, num_steps: int, embedding_dim: int = 128,
                 projection_dim: int | None = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x

    @staticmethod
    def _build_embedding(num_steps: int, dim: int = 64) -> torch.Tensor:
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


# ---------------------------------------------------------------------------
# Image patch embedding / decoding
# ---------------------------------------------------------------------------

class ImagePatchEmbedding(nn.Module):
    """Patch a 2D image into tokens.

    Input:  (B, C, H, W)
    Output: (B, d_model, n_h, n_w)

    Uses ``nn.Unfold`` to extract non-overlapping patches of size
    ``patch_size x patch_size``, then linearly projects each flattened
    patch to ``d_model`` dimensions.
    """

    def __init__(self, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.projection = nn.Sequential(
            nn.Linear(in_channels * patch_size * patch_size, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size

        # Pad H, W to be divisible by patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')  # (left, right, top, bottom)

        # Unfold: (B, C*ps*ps, n_h*n_w)
        x = F.unfold(x, kernel_size=ps, stride=ps)
        n_h = (H + pad_h) // ps
        n_w = (W + pad_w) // ps

        # Project each flattened patch
        x = x.permute(0, 2, 1).contiguous()     # (B, n_h*n_w, C*ps*ps)
        x = self.projection(x)                   # (B, n_h*n_w, d_model)
        x = x.permute(0, 2, 1).contiguous()      # (B, d_model, n_h*n_w)
        x = x.reshape(B, -1, n_h, n_w)          # (B, d_model, n_h, n_w)
        return x


class ImagePatchDecoder(nn.Module):
    """Unpatch tokens back to image.

    Input:  (B, d_model, n_h, n_w)
    Output: (B, C, H, W)

    Inverse of :class:`ImagePatchEmbedding`: linearly projects each token
    back to ``patch_size x patch_size x C`` pixels, then uses ``nn.Fold``
    to reconstruct the image.
    """

    def __init__(self, patch_size: int, d_model: int, out_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.linear = nn.Linear(d_model, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _D, n_h, n_w = x.shape
        ps = self.patch_size

        # Project back to pixel patches
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, n_h, n_w, D)
        x = self.linear(x)                         # (B, n_h, n_w, C*ps*ps)
        x = x.reshape(B, n_h * n_w, -1)           # (B, n_h*n_w, C*ps*ps)
        x = x.permute(0, 2, 1).contiguous()        # (B, C*ps*ps, n_h*n_w)

        # Fold back to padded image
        H_pad, W_pad = n_h * ps, n_w * ps
        x = F.fold(x, (H_pad, W_pad), kernel_size=ps, stride=ps)

        # Crop to original resolution
        x = x[:, :, :H, :W]
        return x


class ConvRasterStem(nn.Module):
    """Convolutional stem for row-raster images.

    The horizontal branch learns adjacent-time patterns within a row, the
    vertical branch learns cross-row/periodic offsets, and the local branch
    captures joint 2D motifs before Transformer tokenization.
    """

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        branch_dim = max(8, d_model // 2)
        self.branch_dim = branch_dim
        self.horizontal = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(1, branch_dim),
            nn.SiLU(),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.GroupNorm(1, branch_dim),
            nn.SiLU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, branch_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * branch_dim, d_model, kernel_size=1),
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
        )
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3 * branch_dim, 3 * branch_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.branch_gate[1].weight)
        nn.init.constant_(self.branch_gate[1].bias, 2.0)
        self.skip = nn.Conv2d(in_channels, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.horizontal(x), self.vertical(x), self.local(x)], dim=1)
        feat = feat * self.branch_gate(feat)
        return self.fuse(feat) + self.skip(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention (SE-Net).

    Global average pooling → bottleneck FC → per-channel sigmoid gate.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.fc[4].weight)
        nn.init.constant_(self.fc[4].bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class MultiScaleConvRasterStem(ConvRasterStem):
    """Multi-scale stem with dilated temporal branches + SE fusion.

    Extends :class:`ConvRasterStem` with two dilated branches (dilation=2,4)
    for medium/long-range temporal context.  A :class:`SqueezeExcitation`
    module re-weights all 5 branches before fusion.
    """

    def __init__(self, in_channels: int, d_model: int):
        super().__init__(in_channels, d_model)
        branch_dim = self.branch_dim
        self.dilated_medium = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=3,
                      padding=2, dilation=2),
            nn.GroupNorm(1, branch_dim),
            nn.SiLU(),
        )
        self.dilated_long = nn.Sequential(
            nn.Conv2d(in_channels, branch_dim, kernel_size=3,
                      padding=4, dilation=4),
            nn.GroupNorm(1, branch_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * branch_dim, d_model, kernel_size=1),
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
        )
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(5 * branch_dim, 5 * branch_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.branch_gate[1].weight)
        nn.init.constant_(self.branch_gate[1].bias, 2.0)
        self.se = SqueezeExcitation(5 * branch_dim, reduction=4)
        _log.debug(
            "MultiScaleConvRasterStem: in=%d d_model=%d branch_dim=%d "
            "(5 branches: h/v/local/dil_med/dil_long + SE)",
            in_channels, d_model, branch_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([
            self.horizontal(x),
            self.vertical(x),
            self.local(x),
            self.dilated_medium(x),
            self.dilated_long(x),
        ], dim=1)
        feat = self.se(feat)
        feat = feat * self.branch_gate(feat)
        return self.fuse(feat) + self.skip(x)


class RowRasterPatchEmbedding(nn.Module):
    """Convolutional row-raster embedding with horizontal temporal patches.

    Output preserves the raster row axis: ``(B, d_model, H, ceil(W / patch))``.
    Each token sees a contiguous horizontal time span, while the stem has
    already mixed nearby rows for periodic/cross-row structure.
    """

    def __init__(self, patch_size: int, in_channels: int, d_model: int,
                 use_multiscale_stem: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        if use_multiscale_stem:
            self.stem = MultiScaleConvRasterStem(in_channels, d_model)
        else:
            self.stem = ConvRasterStem(in_channels, d_model)
        self.patch = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, patch_size), stride=(1, patch_size)),
            nn.GroupNorm(1, d_model),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size
        pad_w = (ps - W % ps) % ps
        if pad_w:
            x = F.pad(x, (0, pad_w, 0, 0), mode="replicate")
        return self.patch(self.stem(x))


class RowRasterPatchDecoder(nn.Module):
    """Inverse projection for :class:`RowRasterPatchEmbedding`."""

    def __init__(self, patch_size: int, d_model: int, out_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.linear = nn.Linear(d_model, patch_size * out_channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _D, n_h, n_t = x.shape
        ps = self.patch_size
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, n_h, n_t, D)
        x = self.linear(x).reshape(B, n_h, n_t, self.out_channels, ps)
        x = x.permute(0, 3, 1, 2, 4).reshape(B, self.out_channels, n_h, n_t * ps)
        return x[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Image side encoder (position encoding for 2D patch grids)
# ---------------------------------------------------------------------------

class ImageSideEncoder(nn.Module):
    """Encode 2D patch positions as side information.

    Combines sinusoidal position encoding (row + col + flattened row-major
    time index) with an optional
    learnable spatial embedding that captures dataset-specific positional
    patterns beyond what sinusoidal frequencies can represent.

    Output: ``(1, row_dim + col_dim + time_dim, n_h, n_w)``
    """

    def __init__(self, row_dim: int, col_dim: int, time_dim: int | None = None,
                 max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.time_dim = col_dim if time_dim is None else int(time_dim)
        self.total_emb_dim = row_dim + col_dim + self.time_dim
        self.max_h = max_h
        self.max_w = max_w

        # Learnable spatial embedding (the "channel embedding" analogue)
        self.spatial_emb = nn.Parameter(
            torch.zeros(1, self.total_emb_dim, max_h, max_w)
        )
        nn.init.trunc_normal_(self.spatial_emb, std=0.02)

    @staticmethod
    def _sinusoidal_pe(positions: torch.Tensor, d_model: int) -> torch.Tensor:
        """Sinusoidal position encoding.

        Args:
            positions: (N,) integer position indices.
            d_model:   encoding dimension (must be even).

        Returns:
            pe: (N, d_model)
        """
        device = positions.device
        pe = torch.zeros(len(positions), d_model, device=device)
        pos = positions.float().unsqueeze(1)                          # (N, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * -(math.log(10000.0) / d_model)
        )                                                             # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, n_h: int, n_w: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            n_h:   number of row patches.
            n_w:   number of column patches.
            device: target device.

        Returns:
            side_emb: ``(1, row_dim + col_dim + time_dim, n_h, n_w)``
        """
        row_ids = torch.arange(n_h, device=device)
        col_ids = torch.arange(n_w, device=device)
        flat_ids = torch.arange(n_h * n_w, device=device)

        row_pe = self._sinusoidal_pe(row_ids, self.row_dim)  # (n_h, row_dim)
        col_pe = self._sinusoidal_pe(col_ids, self.col_dim)  # (n_w, col_dim)
        time_pe = self._sinusoidal_pe(flat_ids, self.time_dim).reshape(n_h, n_w, self.time_dim)

        # Broadcast: rows vary along dim=2, cols along dim=3
        row_pe = row_pe.T.unsqueeze(0).unsqueeze(-1)         # (1, row_dim, n_h, 1)
        col_pe = col_pe.T.unsqueeze(0).unsqueeze(2)          # (1, col_dim, 1, n_w)
        time_pe = time_pe.permute(2, 0, 1).unsqueeze(0)      # (1, time_dim, n_h, n_w)
        row_pe = row_pe.expand(-1, -1, -1, n_w)              # (1, row_dim, n_h, n_w)
        col_pe = col_pe.expand(-1, -1, n_h, -1)              # (1, col_dim, n_h, n_w)

        sinusoidal = torch.cat([row_pe, col_pe, time_pe], dim=1)

        # Learnable spatial embedding (interpolate if grid exceeds max)
        spatial = self.spatial_emb.to(device)
        if n_h > self.max_h or n_w > self.max_w:
            spatial = F.interpolate(
                spatial, size=(n_h, n_w), mode="bilinear", align_corners=False,
            )
        else:
            spatial = spatial[:, :, :n_h, :n_w]

        return sinusoidal + spatial


# ---------------------------------------------------------------------------
# Residual Block (structure identical to VerbalTS)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Dual-axis Transformer block with conditioning.

    This block is structurally identical to the VerbalTS ResidualBlock:

    1. **Condition injection** (before attention):
       - ``"add"``:            ``x = x + attr_emb``
       - ``"cross_attention"``: cross-attention from x to attr_emb
       - ``"adaLN"``:           adaptive layer-norm modulation
    2. **Diffusion timestep** injection via additive projection.
    3. **Dual-axis attention**:
       - ``forward_time``:    Transformer over L (W-patches / columns)
       - ``forward_feature``: Transformer over K (H-patches / rows)
    4. **Side projection** + gate/filter mechanism.
    5. Residual connection with ``/ sqrt(2)`` stabilisation.
    """

    def __init__(self, side_dim: int, channels: int,
                 diffusion_embedding_dim: int, nheads: int,
                 condition_type: str = "adaLN",
                 dim_feedforward: int | None = None,
                 use_feature_axis: bool = True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.norm_mid = nn.GroupNorm(1, channels)      # normalize before mid_projection
        self.side_pre_projection = Conv1d_with_init(side_dim, channels, 1)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.norm_out = nn.GroupNorm(1, channels)      # normalize before output_projection
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(
            heads=nheads, layers=1, channels=channels, dim_feedforward=dim_feedforward,
        )
        self.feature_layer = (
            get_torch_trans(
                heads=nheads, layers=1, channels=channels, dim_feedforward=dim_feedforward,
            )
            if use_feature_axis
            else None
        )

        self.condition_type = condition_type
        if condition_type == "cross_attention":
            self.condition_cross_attention = get_torch_cross_trans(
                heads=nheads, layers=1, channels=channels, dim_feedforward=dim_feedforward,
            )
        elif condition_type == "adaLN":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 3 * channels, bias=True),
            )
            # adaLN-Zero: zero-init so gamma=0, beta=0, alpha=0 at start
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    # -- axis attention ----------------------------------------------------------

    def forward_time(self, y: torch.Tensor, base_shape: tuple,
                     attention_mask: torch.Tensor | dict | None = None) -> torch.Tensor:
        """Transformer attention over L (column / W-patch) dimension.

        Reshapes ``(B, C, K, L)`` to ``(B*K, L, C)``, applies self-attention
        along L, then reshapes back.
        """
        B, C, K, L = base_shape
        if L == 1:
            return y
        attn_mask = attention_mask
        key_padding_mask = None
        if isinstance(attention_mask, dict):
            attn_mask = attention_mask.get("time_attn_mask", None)
            key_padding_mask = attention_mask.get("time_key_padding_mask", None)
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y = self.time_layer(
            y.permute(0, 2, 1),
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        ).permute(0, 2, 1)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)
        return y

    def forward_feature(self, y: torch.Tensor, base_shape: tuple,
                        attention_mask: torch.Tensor | dict | None = None) -> torch.Tensor:
        """Transformer attention over K (row / H-patch) dimension.

        Reshapes ``(B, C, K, L)`` to ``(B*L, K, C)``, applies self-attention
        along K, then reshapes back.
        """
        B, C, K, L = base_shape
        if K == 1 or self.feature_layer is None:
            return y
        attn_mask = attention_mask
        key_padding_mask = None
        if isinstance(attention_mask, dict):
            attn_mask = attention_mask.get("feature_attn_mask", None)
            key_padding_mask = attention_mask.get("feature_key_padding_mask", None)
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        y = self.feature_layer(
            y.permute(0, 2, 1),
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        ).permute(0, 2, 1)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)
        return y

    def forward_cross_attention(self, y: torch.Tensor, cond: torch.Tensor,
                                attention_mask: torch.Tensor | dict | None = None) -> torch.Tensor:
        """Cross-attention from y to cond (both shaped (B, C, K, L))."""
        B, C, K, L = y.shape
        memory_mask = attention_mask
        memory_key_padding_mask = None
        if isinstance(attention_mask, dict):
            memory_mask = attention_mask.get("time_attn_mask", None)
            memory_key_padding_mask = attention_mask.get("time_key_padding_mask", None)
        y = y.reshape(B, C, K, L).permute(0, 2, 3, 1).reshape(B * K, L, C)
        cond = cond.reshape(B, C, K, L).permute(0, 2, 3, 1).reshape(B * K, L, C)
        y = self.condition_cross_attention(
            tgt=y,
            memory=cond,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ).permute(0, 2, 1)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3)
        return y

    # -- adaLN helpers -----------------------------------------------------------

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor,
                 scale: torch.Tensor) -> torch.Tensor:
        """Adaptive layer-norm style modulation: ``x * (1 + scale) + shift``."""
        return x * (1 + scale) + shift

    # -- forward -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,             # (B, channels, K, L)
        side_emb: torch.Tensor,      # (B, side_dim,   K, L)
        attr_emb: torch.Tensor,      # (B, channels, K, L)
        diffusion_emb: torch.Tensor, # (B, diffusion_embedding_dim)
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            ``(x + residual) / sqrt(2)`` and ``skip_connection``.
        """
        ct = self.condition_type

        # -- 1. condition injection (before attention) ---------------------------
        if ct == "add":
            x = x + attr_emb
        elif ct == "cross_attention":
            x = self.forward_cross_attention(x, attr_emb, attention_mask)
        elif ct == "adaLN":
            # attr_emb: (B, C, K, L) -> permute for linear -> (B, K, L, C)
            gamma, beta, alpha = self.adaLN_modulation(
                attr_emb.permute(0, 2, 3, 1)
            ).chunk(3, dim=-1)
            gamma = gamma.permute(0, 3, 1, 2)  # (B, C, K, L)
            beta  = beta.permute(0, 3, 1, 2)
            alpha = alpha.permute(0, 3, 1, 2)

        # -- 2. diffusion timestep injection -------------------------------------
        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb)   # (B, channels)
        diffusion_emb = diffusion_emb.unsqueeze(-1).unsqueeze(-1)  # (B, channels, 1, 1)
        y = x + diffusion_emb
        side_pre = self.side_pre_projection(side_emb.reshape(B, side_emb.shape[1], K * L))
        y = y + side_pre.reshape(B, channel, K, L)

        if ct == "adaLN":
            y = self.modulate(y, gamma, beta)

        # -- 3. dual-axis attention ----------------------------------------------
        y = self.forward_time(y, base_shape, attention_mask)     # over L (columns)
        y = self.forward_feature(y, base_shape, attention_mask)  # over K (rows)

        if ct == "adaLN":
            y = y.reshape(B, channel, K, L)
            y = alpha * y
            y = y.reshape(B, channel, K * L)

        # -- 4. side projection + gate / filter ----------------------------------
        y = y.reshape(B, channel, K * L)
        y = self.norm_mid(y)
        y = self.mid_projection(y)

        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        y = y + side_emb

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.norm_out(y)
        y = self.output_projection(y)

        # -- 5. residual + skip --------------------------------------------------
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


# ---------------------------------------------------------------------------
# Simple CTICD fallback for univariate data
# ---------------------------------------------------------------------------

@dataclass
class _SimpleCTICDOutput:
    """Lightweight return type matching CTICDOutput interface for n_channels==1."""
    causal_features: torch.Tensor
    causal_graph: torch.Tensor
    lagged_graphs: torch.Tensor
    mechanism_states: torch.Tensor
    losses: dict[str, torch.Tensor]


class SimpleCTICDFallback(nn.Module):
    """Segment-level causal self-attention fallback for univariate data.

    CTICD's NOTEARS causal graph learning is designed for cross-channel causal
    discovery.  For univariate data (n_channels==1) there are no cross-channel
    edges to discover, so this lightweight module provides temporal causal
    masking instead: each segment can only attend to itself and earlier segments.
    """

    def __init__(
        self,
        output_channels: int,
        n_segments: int = 8,
        num_heads: int = 4,
        injection_init: float = -4.0,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.n_segments = n_segments
        self.num_heads = num_heads

        self.seg_proj_in = nn.Linear(output_channels, output_channels)
        self.temporal_pe = nn.Parameter(torch.zeros(n_segments, output_channels))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        self.norm1 = nn.LayerNorm(output_channels)
        self.self_attn = nn.MultiheadAttention(
            output_channels, num_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(output_channels)
        self.ffn = nn.Sequential(
            nn.Linear(output_channels, 2 * output_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(2 * output_channels, output_channels),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

        self.out_proj = nn.Linear(output_channels, output_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.injection_logit = nn.Parameter(torch.tensor(float(injection_init)))

    def forward(
        self,
        image: torch.Tensor,
        x_in: torch.Tensor,
        attr_emb: torch.Tensor | None = None,
        clean_image: torch.Tensor | None = None,
        diffusion_emb: torch.Tensor | None = None,
        intervention=None,
    ) -> _SimpleCTICDOutput:
        B, C, K, L = x_in.shape
        device = x_in.device
        n_seg = self.n_segments

        # Pool spatial tokens into n_segments temporal bins
        seq_len = K * L
        seg_edges = torch.linspace(0, seq_len, steps=n_seg + 1, device=device).long()
        seg_feats = []
        x_flat = x_in.reshape(B, C, seq_len).permute(0, 2, 1)  # (B, seq_len, C)
        for s in range(n_seg):
            lo, hi = int(seg_edges[s]), int(seg_edges[s + 1])
            hi = max(hi, lo + 1)
            seg_feats.append(x_flat[:, lo:hi].mean(dim=1))
        tokens = torch.stack(seg_feats, dim=1)  # (B, n_seg, C)
        tokens = self.seg_proj_in(tokens) + self.temporal_pe.unsqueeze(0)

        # Causal self-attention: segment s attends only to segments <= s
        causal_mask = torch.triu(
            torch.ones(n_seg, n_seg, device=device, dtype=torch.bool), diagonal=1,
        )
        h = self.norm1(tokens)
        attended, _ = self.self_attn(
            query=h, key=h, value=h, attn_mask=causal_mask, need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Broadcast back to (B, C, K, L)
        out = self.out_proj(tokens.mean(dim=1, keepdim=True))  # (B, 1, C)
        causal_features = out.unsqueeze(-1).expand(B, C, K, L)

        scale = torch.sigmoid(self.injection_logit)
        causal_features = scale * causal_features

        zero_graph = torch.zeros(B, 1, 1, device=device)
        return _SimpleCTICDOutput(
            causal_features=causal_features,
            causal_graph=zero_graph,
            lagged_graphs=torch.zeros(B, 1, 1, 1, device=device),
            mechanism_states=tokens.unsqueeze(2),
            losses={
                "cticd_total": torch.tensor(0.0, device=device),
                "cticd_pred": torch.tensor(0.0, device=device),
                "cticd_notears": torch.tensor(0.0, device=device),
                "cticd_sparsity": torch.tensor(0.0, device=device),
                "cticd_smooth": torch.tensor(0.0, device=device),
            },
        )


# ---------------------------------------------------------------------------
# TIGER DiT -- main model
# ---------------------------------------------------------------------------

class TIGERDiT(nn.Module):
    """Text+Image Guided Encoding for Recomposition -- DiT backbone.

    Takes a noisy TS-image and predicts the added noise, conditioned on
    diffusion timestep and text/image attributes via adaLN (or add /
    cross-attention).  For row-raster inputs, ``patch_mode="row_raster"``
    uses contiguous temporal patches instead of square image patches.

    Multi-patch support: multiple patch sizes produce grids that are
    **flattened and concatenated** along the sequence dimension with a
    block-diagonal parallel attention mask (same strategy as VerbalTS
    ``multipatch_num``).  In ``patch_mode="row_raster"`` the recommended
    first-stage path uses a single horizontal patch scale and preserves the
    raster row axis as the feature axis for dual-axis attention.

    Expected ``config`` keys::

        channels              : int   -- hidden dimension (e.g. 256)
        nheads                : int   -- attention heads (e.g. 4)
        layers                : int   -- number of ResidualBlocks
        num_steps             : int   -- diffusion timesteps
        diffusion_embedding_dim: int  -- timestep embedding dim
        base_patch            : int   -- base patch size (e.g. 4)
        multipatch_num        : int   -- number of patch scales (default 1)
        patch_scale           : int   -- scale factor between patches (default 2)
        in_channels           : int   -- input image channels (default 3)
        row_dim               : int   -- row PE dim (default 32)
        col_dim               : int   -- col PE dim (default 32)
        condition_type        : str   -- "adaLN" | "add" | "cross_attention"
        attention_mask_type   : str   -- "parallel" | "full"
        attr_dim              : int   -- attr_emb channel dim (default = channels)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.channels: int = config["channels"]
        self.nheads: int = config["nheads"]
        self.condition_type: str = config.get("condition_type", "adaLN")
        self.attention_mask_type: str = config.get("attention_mask_type", "parallel")
        self.multipatch_num: int = config.get("multipatch_num", 1)
        self.patch_mode: str = str(config.get("patch_mode", "square")).lower()
        if self.patch_mode == "row_raster" and self.multipatch_num > 1:
            raise ValueError(
                "row_raster currently requires multipatch_num=1. "
                "Flattening multiple row-raster scales would collapse the row axis "
                "back to K=1 and disable dual-axis attention."
            )
        self.signal_length: int | None = config.get("signal_length", None)
        if self.signal_length is not None:
            self.signal_length = int(self.signal_length)
        ff_mult = int(config.get("ff_mult", 4))
        dim_feedforward = int(config.get("dim_feedforward", ff_mult * self.channels))
        use_feature_axis = bool(config.get("use_feature_axis", True))

        # -- diffusion timestep embedding ----------------------------------------
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # -- side encoder (shared across scales) ---------------------------------
        row_dim: int = config.get("row_dim", 32)
        col_dim: int = config.get("col_dim", 32)
        time_dim: int = config.get("time_dim", col_dim)
        side_dim: int = row_dim + col_dim + time_dim

        # -- attr_emb projection (if attr_dim != channels) -----------------------
        attr_dim: int = config.get("attr_dim", self.channels)
        if attr_dim != self.channels:
            self.attr_proj = Conv1d_with_init(attr_dim, self.channels, 1)
        else:
            self.attr_proj = nn.Identity()

        # -- multi-patch embedders / decoders / side encoders --------------------
        base_patch: int = config["base_patch"]
        patch_scale: int = config.get("patch_scale", 2)
        in_channels: int = config.get("in_channels", 3)
        if self.patch_mode == "row_raster" and "image_size_w" in config:
            image_size_w = max(1, int(config["image_size_w"]))
            if base_patch > image_size_w:
                base_patch = image_size_w
                self.config["base_patch"] = base_patch

        self.image_downsample = nn.ModuleList()
        self.side_downsample = nn.ModuleList()
        self.patch_decoder = nn.ModuleList()

        use_multiscale_stem: bool = config.get("use_multiscale_stem", False)

        for i in range(self.multipatch_num):
            ps = base_patch * (patch_scale ** i)
            patch_embed_cls = RowRasterPatchEmbedding if self.patch_mode == "row_raster" else ImagePatchEmbedding
            patch_decoder_cls = RowRasterPatchDecoder if self.patch_mode == "row_raster" else ImagePatchDecoder
            if self.patch_mode == "row_raster":
                self.image_downsample.append(
                    patch_embed_cls(ps, in_channels, self.channels,
                                    use_multiscale_stem=use_multiscale_stem),
                )
            else:
                self.image_downsample.append(
                    patch_embed_cls(ps, in_channels, self.channels),
                )
            self.patch_decoder.append(
                patch_decoder_cls(ps, self.channels, in_channels),
            )
            # Each scale gets its own side encoder instance
            # (ImageSideEncoder is stateless for sinusoidal + learnable spatial)
            self.side_downsample.append(
                ImageSideEncoder(row_dim=row_dim, col_dim=col_dim, time_dim=time_dim),
            )

        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)
        if self.multipatch_num == 1:
            with torch.no_grad():
                self.multipatch_mixer.weight.fill_(1.0)
                self.multipatch_mixer.bias.zero_()

        # -- output projection ---------------------------------------------------
        self.output_projection = Conv1d_with_init(self.channels, self.channels, 1)

        # -- residual transformer layers -----------------------------------------
        _base_layers = nn.ModuleList([
            ResidualBlock(
                side_dim=side_dim,
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=self.nheads,
                condition_type=self.condition_type,
                dim_feedforward=dim_feedforward,
                use_feature_axis=use_feature_axis,
            )
            for _ in range(config["layers"])
        ])

        # --- CSA-MoE: optional channel-structure-aware MoE (ablation toggle) ---
        moe_cfg = config.get("csa_moe", None)
        self._csa_moe_enabled = moe_cfg is not None and moe_cfg.get("enabled", True)
        self._csa_moe_losses = None

        if self._csa_moe_enabled:
            from .csa_moe import ChannelAwareResidualBlock

            image_size_h = config.get("image_size_h", config.get("image_size", 64))
            image_size_w = config.get("image_size_w", config.get("image_size", 64))
            moe_grids = []
            for i in range(self.multipatch_num):
                ps = base_patch * (patch_scale ** i)
                if self.patch_mode == "row_raster":
                    # RowRasterPatchEmbedding keeps rows intact and only
                    # patches along width, so CSA-MoE/SCCA must see the same
                    # H x ceil(W / ps) token grid as the DiT backbone.
                    n_h = image_size_h
                    n_w = (image_size_w + ps - 1) // ps
                else:
                    # Match ImagePatchEmbedding padding: ceil division
                    n_h = (image_size_h + ps - 1) // ps
                    n_w = (image_size_w + ps - 1) // ps
                moe_grids.append((n_h, n_w))

            self.residual_layers = nn.ModuleList()
            for base_layer in _base_layers:
                self.residual_layers.append(ChannelAwareResidualBlock(
                    base=base_layer,
                    grids=moe_grids,
                    ch=self.channels,
                    t_dim=config["diffusion_embedding_dim"],
                    k=moe_cfg.get("k", 1),
                    alpha=moe_cfg.get("alpha", 0.01),
                    inject_aux=moe_cfg.get("inject_aux", False),
                    scca_heads=moe_cfg.get("scca_heads", self.nheads),
                ))
        else:
            self.residual_layers = _base_layers

        # --- CTICD: optional causal module ---
        cticd_cfg = config.get("cticd", None)
        self.cticd = None
        self._cticd_losses = None
        self._cticd_graph = None

        if cticd_cfg is not None and cticd_cfg.get("enabled", True):
            cticd_n_channels = cticd_cfg.get("n_channels", 3)

            # Guard: CTICD's NOTEARS causal graph learning is designed for
            # cross-channel (multivariate) causal discovery.  With n_channels==1
            # there are no cross-channel edges to learn, so we fall back to a
            # lightweight segment-level causal self-attention module.
            if cticd_n_channels == 1:
                _log.warning(
                    "CTICD disabled for univariate data (n_channels=1). "
                    "Falling back to SimpleCTICDFallback with segment-level "
                    "causal self-attention. CTICD requires n_channels > 1 "
                    "for meaningful cross-channel causal graph learning."
                )
                self.cticd = SimpleCTICDFallback(
                    output_channels=self.channels,
                    n_segments=cticd_cfg.get("n_segments", 8),
                    num_heads=cticd_cfg.get("num_heads", self.nheads),
                    injection_init=cticd_cfg.get("injection_init", -4.0),
                )
            else:
                from .cticd import CTICD

                cticd_patch_size = cticd_cfg.get("patch_size", config.get("base_patch", 4))
                if self.patch_mode == "row_raster":
                    cticd_patch_size = min(int(cticd_patch_size), int(config.get("image_size_w", cticd_patch_size)))
                self.cticd = CTICD(
                    d_model=cticd_cfg.get("d_model", self.channels),
                    output_channels=self.channels,
                    attr_dim=self.channels,
                    n_channels=cticd_n_channels,
                    n_mechanisms_per_channel=cticd_cfg.get("n_mechanisms_per_channel", 4),
                    patch_size=cticd_patch_size,
                    num_heads=cticd_cfg.get("num_heads", self.nheads),
                    edge_bias=cticd_cfg.get("edge_bias", -4.0),
                    branch_grad_scale=cticd_cfg.get("branch_grad_scale", 0.2),
                    n_segments=cticd_cfg.get("n_segments", 8),
                    max_lag=cticd_cfg.get("max_lag", 2),
                    lag_edge_bias=cticd_cfg.get("lag_edge_bias", -2.5),
                    lambda_causal=cticd_cfg.get("lambda_causal", 1.0),
                    lambda_notears=cticd_cfg.get("lambda_notears", 1e-3),
                    lambda_sparsity=cticd_cfg.get("lambda_sparsity", 1e-2),
                    lambda_smooth=cticd_cfg.get("lambda_smooth", 1e-3),
                    lambda_prior=cticd_cfg.get("lambda_prior", 1e-1),
                    use_lag_prior=cticd_cfg.get("use_lag_prior", True),
                    lag_prior_method=cticd_cfg.get("lag_prior_method", "acf"),
                    lag_prior_mode=cticd_cfg.get("lag_prior_mode", "bias_gate"),
                    lag_prior_strength=cticd_cfg.get("lag_prior_strength", 1.0),
                    lag_prior_significance=cticd_cfg.get("lag_prior_significance", True),
                    lag_prior_bins=cticd_cfg.get("lag_prior_bins", 16),
                    lag_topk=cticd_cfg.get("lag_topk", 0),
                    signal_length=cticd_cfg.get("signal_length", None),
                    injection_init=cticd_cfg.get("injection_init", -4.0),
                )

    # -- mask builder -----------------------------------------------------------

    @staticmethod
    def _build_parallel_mask(
        len_list: list[int],
        device: torch.device,
        valid_list: list[int] | None = None,
    ) -> torch.Tensor:
        """Block-diagonal mask: each block attends only within itself.

        Replicates the VerbalTS ``get_mask`` logic with ``attr_len=0``.
        """
        total = sum(len_list)
        mask = torch.zeros(total, total, device=device) - float("inf")
        start = 0
        for idx, length in enumerate(len_list):
            mask[start:start + length, start:start + length] = 0
            if valid_list is not None:
                valid = max(1, min(int(valid_list[idx]), length))
                mask[start:start + length, start + valid:start + length] = -float("inf")
            start += length
        return mask

    def _valid_tokens_for_scale(self, patch_size: int, token_count: int) -> int:
        if self.patch_mode != "row_raster" or self.signal_length is None:
            return token_count
        return min(token_count, max(1, (self.signal_length + patch_size - 1) // patch_size))

    def _build_row_raster_padding_masks(
        self,
        batch_size: int,
        n_h: int,
        n_w: int,
        image_w: int,
        patch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        """Build key-padding masks for row-raster dual-axis attention.

        ``True`` entries are ignored by PyTorch attention.  Fully padded rows or
        columns are given one unmasked anchor token to avoid all-masked attention
        rows producing NaN on some PyTorch versions.
        """
        if self.patch_mode != "row_raster" or self.signal_length is None:
            return None

        rows = torch.arange(n_h, device=device).unsqueeze(1)
        cols = torch.arange(n_w, device=device).unsqueeze(0)
        patch_start = rows * int(image_w) + cols * int(patch_size)
        valid = patch_start < int(self.signal_length)  # (n_h, n_w)
        if bool(valid.all()):
            return None
        token_valid_mask = valid.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size, n_h * n_w)

        time_key_padding_mask = (~valid).unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * n_h, n_w)
        all_masked_time = time_key_padding_mask.all(dim=1)
        if bool(all_masked_time.any()):
            time_key_padding_mask[all_masked_time, 0] = False

        feature_valid = valid.transpose(0, 1).contiguous()  # (n_w, n_h)
        feature_key_padding_mask = (
            ~feature_valid
        ).unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * n_w, n_h)
        all_masked_feature = feature_key_padding_mask.all(dim=1)
        if bool(all_masked_feature.any()):
            feature_key_padding_mask[all_masked_feature, 0] = False

        return {
            "time_key_padding_mask": time_key_padding_mask,
            "feature_key_padding_mask": feature_key_padding_mask,
            "token_valid_mask": token_valid_mask,
        }

    # -- forward ----------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,                    # (B, 3, H, W)
        diffusion_step: torch.Tensor,           # (B,)
        attr_emb: torch.Tensor | None = None,   # (B, attr_dim, n_h, n_w) or None
        clean_image: torch.Tensor | None = None, # clean target image for CTICD during training
        intervention: dict | None = None,        # optional do-intervention for causal sampling
        enable_cticd: bool = True,               # whether to run CTICD (skip at high noise)
    ) -> torch.Tensor:
        """
        Args:
            image:          ``(B, in_channels, H, W)`` noisy image.
            diffusion_step: ``(B,)`` integer timestep indices.
            attr_emb:       ``(B, attr_dim, n_h, n_w)`` from TextOnlyProjector or
                            MultiModalConditioner, or ``None`` for unconditional generation.
            clean_image:    optional clean image used by CTICD to avoid learning
                            graphs purely from noisy diffusion states.
            intervention:   optional causal intervention dictionary passed to CTICD.
            enable_cticd:   if ``False``, skip CTICD entirely and inject zero features.
                            Used during sampling to disable CTICD at high noise levels.

        Returns:
            noise_pred: ``(B, in_channels, H, W)`` predicted noise.
        """
        B, C_in, H, W = image.shape
        device = image.device

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # ------------------------------------------------------------------
        # 1. Multi-patch encoding
        # ------------------------------------------------------------------
        x_list: list[torch.Tensor] = []       # each: (B, channels, n_tok_i)
        side_list: list[torch.Tensor] = []    # each: (B, side_dim,  n_tok_i)
        token_counts: list[int] = []
        valid_counts: list[int] = []
        grids: list[tuple[int, int]] = []     # (n_h_i, n_w_i) per scale

        for i in range(self.multipatch_num):
            x_i = self.image_downsample[i](image)            # (B, ch, n_h_i, n_w_i)
            n_h_i, n_w_i = x_i.shape[2], x_i.shape[3]
            ps = self.config["base_patch"] * (self.config.get("patch_scale", 2) ** i)

            side_i = self.side_downsample[i](n_h_i, n_w_i, device)  # (1, sd, n_h_i, n_w_i)
            side_i = side_i.expand(B, -1, -1, -1)                   # (B, sd, n_h_i, n_w_i)

            x_list.append(x_i)
            side_list.append(side_i)
            token_counts.append(n_h_i * n_w_i)
            valid_counts.append(self._valid_tokens_for_scale(ps, n_h_i * n_w_i))
            grids.append((n_h_i, n_w_i))

        # ------------------------------------------------------------------
        # 2. Attention mask
        # ------------------------------------------------------------------
        if self.multipatch_num == 1:
            if self.patch_mode == "row_raster":
                ps = self.config["base_patch"]
                n_h_i, n_w_i = grids[0]
                attention_mask = self._build_row_raster_padding_masks(
                    batch_size=B,
                    n_h=n_h_i,
                    n_w=n_w_i,
                    image_w=W,
                    patch_size=ps,
                    device=device,
                )
            else:
                attention_mask = None
        elif self.attention_mask_type == "parallel":
            attention_mask = self._build_parallel_mask(token_counts, device, valid_counts)
        else:
            attention_mask = None

        # ------------------------------------------------------------------
        # 3. Build token grid
        # ------------------------------------------------------------------
        if self.multipatch_num == 1:
            x_in = x_list[0]       # (B, channels, K, L)
            side_in = side_list[0] # (B, side_dim, K, L)
            total_tokens = token_counts[0]
        else:
            x_in = torch.cat(
                [x_i.reshape(B, self.channels, -1) for x_i in x_list],
                dim=-1,
            ).unsqueeze(2)
            side_in = torch.cat(
                [s_i.reshape(B, s_i.shape[1], -1) for s_i in side_list],
                dim=-1,
            ).unsqueeze(2)
            total_tokens = x_in.shape[-1]

        # ------------------------------------------------------------------
        # 4. attr_emb handling
        # ------------------------------------------------------------------
        if attr_emb is None:
            attr_cat = torch.zeros(
                B, self.channels, *x_in.shape[2:], device=device,
            )
        else:
            # Treat condition anchors as a low-resolution condition map and
            # resize them to each patch scale. This keeps n_var x n_scale
            # anchors usable instead of collapsing back to a global vector.
            attr_for_resize = attr_emb.float()
            if self.multipatch_num == 1:
                n_h_i, n_w_i = grids[0]
                attr_cat = F.interpolate(
                    attr_for_resize,
                    size=(n_h_i, n_w_i),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=attr_emb.dtype)
            else:
                attr_parts: list[torch.Tensor] = []
                for i in range(self.multipatch_num):
                    n_h_i, n_w_i = grids[i]
                    attr_i = F.interpolate(
                        attr_for_resize,
                        size=(n_h_i, n_w_i),
                        mode="bilinear",
                        align_corners=False,
                    ).to(dtype=attr_emb.dtype)
                    attr_parts.append(attr_i.reshape(B, attr_i.shape[1], -1))
                attr_cat = torch.cat(attr_parts, dim=-1).unsqueeze(2)

            # Project if attr_dim != channels
            attr_shape = attr_cat.shape
            attr_cat = self.attr_proj(attr_cat.reshape(B, attr_shape[1], -1))
            attr_cat = attr_cat.reshape(B, self.channels, *attr_shape[2:])

        attr_in = attr_cat

        # --- CTICD causal feature injection ---
        self._cticd_losses = None
        self._cticd_graph = None

        if self.cticd is not None and enable_cticd:
            cticd_out = self.cticd(
                image=image,
                clean_image=clean_image,
                x_in=x_in,
                attr_emb=attr_in,
                diffusion_emb=diffusion_emb,
                intervention=intervention,
            )

            self._cticd_losses = cticd_out.losses
            self._cticd_graph = cticd_out.causal_graph

            x_in = x_in + cticd_out.causal_features

        # ------------------------------------------------------------------
        # 6. Residual layers with skip connections
        # ------------------------------------------------------------------
        skips: list[torch.Tensor] = []
        moe_aux_losses: list[torch.Tensor] = []
        for layer in self.residual_layers:
            if self._csa_moe_enabled:
                x_in, skip, aux = layer(
                    x_in, side_in, attr_in, diffusion_emb,
                    t_emb=diffusion_emb, am=attention_mask,
                )
                if aux is not None:
                    moe_aux_losses.append(aux)
            else:
                x_in, skip = layer(
                    x_in, side_in, attr_in, diffusion_emb,
                    attention_mask=attention_mask,
                )
            skips.append(skip)

        # Store CSA-MoE losses for generator to pick up
        if self._csa_moe_enabled and moe_aux_losses:
            self._csa_moe_losses = torch.stack(moe_aux_losses).mean()
        else:
            self._csa_moe_losses = None

        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(skips))

        # ------------------------------------------------------------------
        # 7. Output projection
        # ------------------------------------------------------------------
        x = x.reshape(B, self.channels, total_tokens)
        x = self.output_projection(x)

        # ------------------------------------------------------------------
        # 8. Split back per scale, decode, and mix
        # ------------------------------------------------------------------
        all_out: list[torch.Tensor] = []
        start = 0
        for i in range(self.multipatch_num):
            n_tok = token_counts[i]
            n_h_i, n_w_i = grids[i]

            x_i = x[:, :, start:start + n_tok]                    # (B, ch, n_tok)
            x_i = x_i.reshape(B, self.channels, n_h_i, n_w_i)    # (B, ch, n_h_i, n_w_i)
            start += n_tok

            out_i = self.patch_decoder[i](x_i, H, W)              # (B, C_in, H, W)
            all_out.append(out_i)

        # multipatch_mixer: per-pixel weighted combination across scales
        # Stack: (B, mp_num, C_in, H, W) -> permute -> mix -> squeeze
        all_out = torch.stack(all_out, dim=1)                      # (B, mp, C_in, H, W)
        all_out = all_out.permute(0, 3, 4, 2, 1)                  # (B, H, W, C_in, mp)
        all_out = self.multipatch_mixer(all_out)                   # (B, H, W, C_in, 1)
        all_out = all_out.squeeze(-1).permute(0, 3, 1, 2)         # (B, C_in, H, W)

        return all_out
