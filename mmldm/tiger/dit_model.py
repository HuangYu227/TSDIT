"""TIGER DiT -- Diffusion Image Transformer.

Adapted from VerbalTS (E:\\Research\\TSG\\VerbalTS\\models\\diffusion\\verbalts.py)
for image+text conditioned generation.

Architecture mapping (VerbalTS -> TIGER):
    n_var (K)   -> n_patches_h   (H dimension patches)
    L (time)    -> n_patches_w   (W dimension patches)
    TsPatchEmbedding    -> ImagePatchEmbedding   (2D unfold)
    SideEncoder_Var     -> ImageSideEncoder      (2D sinusoidal PE + learnable)
    PatchDecoder        -> ImagePatchDecoder      (2D fold)
    ResidualBlock       -> ResidualBlock          (identical structure)
    multipatch_mixer    -> multipatch_mixer       (per-pixel scale mixing)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Transformer helpers (copied from VerbalTS)
# ---------------------------------------------------------------------------

def get_torch_trans(heads: int = 8, layers: int = 1, channels: int = 64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_torch_cross_trans(heads: int = 8, layers: int = 1, channels: int = 64):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
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
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size

        # Pad H, W to be divisible by patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

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


# ---------------------------------------------------------------------------
# Image side encoder (position encoding for 2D patch grids)
# ---------------------------------------------------------------------------

class ImageSideEncoder(nn.Module):
    """Encode 2D patch positions (row, col) as side information.

    Combines sinusoidal position encoding (row + col) with an optional
    learnable spatial embedding that captures dataset-specific positional
    patterns beyond what sinusoidal frequencies can represent.

    Output: ``(1, row_dim + col_dim, n_h, n_w)``
    """

    def __init__(self, row_dim: int, col_dim: int,
                 max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.total_emb_dim = row_dim + col_dim
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
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, n_h: int, n_w: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            n_h:   number of row patches.
            n_w:   number of column patches.
            device: target device.

        Returns:
            side_emb: ``(1, row_dim + col_dim, n_h, n_w)``
        """
        row_ids = torch.arange(n_h, device=device)
        col_ids = torch.arange(n_w, device=device)

        row_pe = self._sinusoidal_pe(row_ids, self.row_dim)  # (n_h, row_dim)
        col_pe = self._sinusoidal_pe(col_ids, self.col_dim)  # (n_w, col_dim)

        # Broadcast: rows vary along dim=2, cols along dim=3
        row_pe = row_pe.T.unsqueeze(0).unsqueeze(-1)         # (1, row_dim, n_h, 1)
        col_pe = col_pe.T.unsqueeze(0).unsqueeze(2)          # (1, col_dim, 1, n_w)
        row_pe = row_pe.expand(-1, -1, -1, n_w)              # (1, row_dim, n_h, n_w)
        col_pe = col_pe.expand(-1, -1, n_h, -1)              # (1, col_dim, n_h, n_w)

        sinusoidal = torch.cat([row_pe, col_pe], dim=1)      # (1, rd+cd, n_h, n_w)

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
                 condition_type: str = "adaLN"):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.norm_mid = nn.GroupNorm(1, channels)      # normalize before mid_projection
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.norm_out = nn.GroupNorm(1, channels)      # normalize before output_projection
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        self.condition_type = condition_type
        if condition_type == "cross_attention":
            self.condition_cross_attention = get_torch_cross_trans(
                heads=nheads, layers=1, channels=channels,
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
                     attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Transformer attention over L (column / W-patch) dimension.

        Reshapes ``(B, C, K, L)`` to ``(B*K, L, C)``, applies self-attention
        along L, then reshapes back.
        """
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)
        return y

    def forward_feature(self, y: torch.Tensor, base_shape: tuple,
                        attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Transformer attention over K (row / H-patch) dimension.

        Reshapes ``(B, C, K, L)`` to ``(B*L, K, C)``, applies self-attention
        along K, then reshapes back.
        """
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        y = self.feature_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)
        return y

    def forward_cross_attention(self, y: torch.Tensor, cond: torch.Tensor,
                                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Cross-attention from y to cond (both shaped (B, C, K, L))."""
        B, C, K, L = y.shape
        y = y.reshape(B, C, K, L).permute(0, 2, 3, 1).reshape(B * K, L, C)
        cond = cond.reshape(B, C, K, L).permute(0, 2, 3, 1).reshape(B * K, L, C)
        y = self.condition_cross_attention(
            tgt=y, memory=cond, memory_mask=attention_mask,
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

        if ct == "adaLN":
            y = self.modulate(y, gamma, beta)

        # -- 3. dual-axis attention ----------------------------------------------
        y = self.forward_time(y, base_shape, attention_mask)     # over L (columns)
        y = self.forward_feature(y, base_shape, None)            # over K (rows)

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
# TIGER DiT -- main model
# ---------------------------------------------------------------------------

class TIGERDiT(nn.Module):
    """Text+Image Guided Encoding for Recomposition -- DiT backbone.

    Takes a noisy 3-channel image (GAF + STFT + RP) and predicts the
    added noise, conditioned on diffusion timestep and text/image
    attributes via adaLN (or add / cross-attention).

    Multi-patch support: multiple patch sizes produce grids that are
    **flattened and concatenated** along the sequence dimension with a
    block-diagonal parallel attention mask (same strategy as VerbalTS
    ``multipatch_num``).

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

        # -- diffusion timestep embedding ----------------------------------------
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # -- side encoder (shared across scales) ---------------------------------
        row_dim: int = config.get("row_dim", 32)
        col_dim: int = config.get("col_dim", 32)
        self.side_encoder = ImageSideEncoder(row_dim=row_dim, col_dim=col_dim)
        side_dim: int = self.side_encoder.total_emb_dim

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

        self.image_downsample = nn.ModuleList()
        self.side_downsample = nn.ModuleList()
        self.patch_decoder = nn.ModuleList()

        for i in range(self.multipatch_num):
            ps = base_patch * (patch_scale ** i)
            self.image_downsample.append(
                ImagePatchEmbedding(ps, in_channels, self.channels),
            )
            self.patch_decoder.append(
                ImagePatchDecoder(ps, self.channels, in_channels),
            )
            # Each scale gets its own side encoder instance
            # (ImageSideEncoder is stateless for sinusoidal + learnable spatial)
            self.side_downsample.append(
                ImageSideEncoder(row_dim=row_dim, col_dim=col_dim),
            )

        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)

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
            )
            for _ in range(config["layers"])
        ])

        # --- CSA-MoE: optional channel-structure-aware MoE (ablation toggle) ---
        moe_cfg = config.get("csa_moe", None)
        self._csa_moe_enabled = moe_cfg is not None and moe_cfg.get("enabled", True)
        self._csa_moe_losses = None

        if self._csa_moe_enabled:
            from .csa_moe import ChannelAwareResidualBlock

            image_size = config.get("image_size", 64)
            moe_grids = []
            for i in range(self.multipatch_num):
                ps = base_patch * (patch_scale ** i)
                # Match ImagePatchEmbedding padding: ceil division
                n_h = (image_size + ps - 1) // ps
                n_w = (image_size + ps - 1) // ps
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
            from .cticd import CTICD

            self.cticd = CTICD(
                d_model=cticd_cfg.get("d_model", self.channels),
                output_channels=self.channels,
                attr_dim=self.channels,
                n_channels=cticd_cfg.get("n_channels", 3),
                n_mechanisms_per_channel=cticd_cfg.get("n_mechanisms_per_channel", 4),
                patch_size=cticd_cfg.get("patch_size", config.get("base_patch", 4)),
                num_heads=cticd_cfg.get("num_heads", self.nheads),
                edge_bias=cticd_cfg.get("edge_bias", -4.0),
                branch_grad_scale=cticd_cfg.get("branch_grad_scale", 0.2),
                lambda_causal=cticd_cfg.get("lambda_causal", 0.1),
                lambda_notears=cticd_cfg.get("lambda_notears", 1e-3),
                lambda_sparsity=cticd_cfg.get("lambda_sparsity", 1e-2),
            )

    # -- mask builder -----------------------------------------------------------

    @staticmethod
    def _build_parallel_mask(
        len_list: list[int], device: torch.device,
    ) -> torch.Tensor:
        """Block-diagonal mask: each block attends only within itself.

        Replicates the VerbalTS ``get_mask`` logic with ``attr_len=0``.
        """
        total = sum(len_list)
        mask = torch.zeros(total, total, device=device) - float("inf")
        start = 0
        for length in len_list:
            mask[start:start + length, start:start + length] = 0
            start += length
        return mask

    # -- forward ----------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,                    # (B, 3, H, W)
        diffusion_step: torch.Tensor,           # (B,)
        attr_emb: torch.Tensor | None = None,   # (B, attr_dim, n_h, n_w) or None
    ) -> torch.Tensor:
        """
        Args:
            image:          ``(B, in_channels, H, W)`` noisy image.
            diffusion_step: ``(B,)`` integer timestep indices.
            attr_emb:       ``(B, attr_dim, n_h, n_w)`` from TextOnlyProjector,
                            or ``None`` for unconditional generation.

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
        grids: list[tuple[int, int]] = []     # (n_h_i, n_w_i) per scale

        for i in range(self.multipatch_num):
            x_i = self.image_downsample[i](image)            # (B, ch, n_h_i, n_w_i)
            n_h_i, n_w_i = x_i.shape[2], x_i.shape[3]

            side_i = self.side_downsample[i](n_h_i, n_w_i, device)  # (1, sd, n_h_i, n_w_i)
            side_i = side_i.expand(B, -1, -1, -1)                   # (B, sd, n_h_i, n_w_i)

            x_list.append(x_i.reshape(B, self.channels, -1))
            side_list.append(side_i.reshape(B, side_i.shape[1], -1))
            token_counts.append(n_h_i * n_w_i)
            grids.append((n_h_i, n_w_i))

        # ------------------------------------------------------------------
        # 2. Attention mask
        # ------------------------------------------------------------------
        if (self.attention_mask_type == "parallel"
                and self.multipatch_num > 1):
            attention_mask = self._build_parallel_mask(token_counts, device)
        else:
            attention_mask = None

        # ------------------------------------------------------------------
        # 3. Concatenate along sequence dimension
        # ------------------------------------------------------------------
        x_in    = torch.cat(x_list,    dim=-1)   # (B, channels, total_tokens)
        side_in = torch.cat(side_list, dim=-1)   # (B, side_dim,   total_tokens)

        total_tokens = x_in.shape[-1]

        # ------------------------------------------------------------------
        # 4. attr_emb handling
        # ------------------------------------------------------------------
        if attr_emb is None:
            attr_cat = torch.zeros(
                B, self.channels, total_tokens, device=device,
            )
        else:
            # Treat condition anchors as a low-resolution condition map and
            # resize them to each patch scale. This keeps n_var x n_scale
            # anchors usable instead of collapsing back to a global vector.
            attr_parts: list[torch.Tensor] = []
            attr_for_resize = attr_emb.float()
            for i in range(self.multipatch_num):
                n_h_i, n_w_i = grids[i]
                attr_i = F.interpolate(
                    attr_for_resize,
                    size=(n_h_i, n_w_i),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=attr_emb.dtype)
                attr_parts.append(attr_i.reshape(B, attr_i.shape[1], -1))
            attr_cat = torch.cat(attr_parts, dim=-1)    # (B, attr_dim, total_tokens)

            # Project if attr_dim != channels
            attr_cat = self.attr_proj(attr_cat)          # (B, channels, total_tokens)

        # ------------------------------------------------------------------
        # 5. Reshape for ResidualBlock: (B, C, K=1, L=total_tokens)
        #    K=1 because multi-patch grids are flattened into a single
        #    sequence; the block-diagonal mask ensures cross-scale isolation.
        # ------------------------------------------------------------------
        x_in    = x_in.unsqueeze(2)       # (B, channels, 1, total_tokens)
        side_in = side_in.unsqueeze(2)    # (B, side_dim,   1, total_tokens)
        attr_in = attr_cat.unsqueeze(2)   # (B, channels,   1, total_tokens)

        # --- CTICD causal feature injection ---
        self._cticd_losses = None
        self._cticd_graph = None

        if self.cticd is not None:
            cticd_out = self.cticd(
                image=image,
                x_in=x_in,
                attr_emb=attr_in,
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
        x = F.relu(self.output_projection(x))

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
