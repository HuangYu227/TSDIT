"""Condition projectors for TIGER.

This module keeps the original text-only projector for fair baselines and adds
``MultiModalConditioner`` for the paper-ready text+image setting.  The
multimodal conditioner treats a reference/degraded TS-image as an actual
conditioning modality, not as the noisy diffusion state.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: build a cross-attention stack (reused by every anchor group)
# ---------------------------------------------------------------------------

def _make_cross_attn(
    dim_in: int,
    n_heads: int = 8,
    n_layers: int = 2,
    dim_ff: int = 256,
) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=dim_in,
        nhead=n_heads,
        dim_feedforward=dim_ff,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=n_layers)


# ---------------------------------------------------------------------------
# Core building block: anchor projector
# ---------------------------------------------------------------------------

class _AnchorProjector(nn.Module):
    """Variable + scale + diffusion-step anchor projector.

    Parameters
    ----------
    n_var, n_scale, n_steps, n_stages : int
        Anchor counts. ``n_var`` and ``n_scale`` determine the spatial condition
        map consumed by TIGER-DiT.
    dim_in : int
        Input token embedding dimension.
    dim_out : int
        Output embedding dimension.
    """

    def __init__(
        self,
        n_var: int,
        n_scale: int,
        n_steps: int,
        n_stages: int,
        dim_in: int = 128,
        dim_out: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_ff: int = 256,
    ):
        super().__init__()
        if n_stages < 1:
            raise ValueError(f"n_stages must be >= 1, got {n_stages}")
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_var = n_var
        self.n_scale = n_scale
        self.n_stages = n_stages
        self.seg_size = n_steps // max(1, n_stages) + 1

        self.var_emb = nn.Parameter(torch.zeros((1, n_var, dim_in)))
        self.scale_emb = nn.Parameter(torch.zeros((1, n_scale, dim_in)))
        self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))

        self.var_cross_attn = _make_cross_attn(dim_in, n_heads, n_layers, dim_ff)
        self.scale_cross_attn = _make_cross_attn(dim_in, n_heads, n_layers, dim_ff)
        self.step_cross_attn = _make_cross_attn(dim_in, n_heads, n_layers, dim_ff)

        self.proj_out = nn.Linear(dim_in, dim_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.var_emb, std=0.02)
        nn.init.trunc_normal_(self.scale_emb, std=0.02)
        nn.init.trunc_normal_(self.step_emb, std=0.02)

    def forward(self, attr: torch.Tensor, diffusion_step: torch.Tensor) -> torch.Tensor:
        """Project conditioning tokens into TIGER condition anchors.

        Args:
            attr: ``(B, T_cond, dim_in)`` conditioning token sequence.
            diffusion_step: ``(B,)`` integer diffusion timestep.

        Returns:
            ``(B, n_var, n_scale, dim_out)``.
        """
        if attr.dim() != 3:
            raise ValueError(f"attr must be 3D (B,T,D), got {tuple(attr.shape)}")
        B = attr.shape[0]

        var_emb = self.var_emb.expand(B, -1, -1)
        mvar_attr = self.var_cross_attn(tgt=var_emb, memory=attr)
        mvar_attr = mvar_attr[:, :, None, :]

        scale_emb = self.scale_emb.expand(B, -1, -1)
        mscale_attr = self.scale_cross_attn(tgt=scale_emb, memory=attr)
        mscale_attr = mscale_attr[:, None, :, :].expand(-1, self.n_var, -1, -1)

        step_emb = self.step_emb.expand(B, -1, -1)
        mstep_attr = self.step_cross_attn(tgt=step_emb, memory=attr)
        indices = (diffusion_step // self.seg_size).clamp(0, self.n_stages - 1)
        indices = indices[:, None, None]
        mstep_attr = torch.gather(
            mstep_attr,
            dim=1,
            index=indices.expand(-1, -1, mstep_attr.shape[-1]),
        )
        mstep_attr = mstep_attr[:, None, :, :].expand(-1, self.n_var, -1, -1)

        mix_attr = mvar_attr + mscale_attr + mstep_attr
        return self.proj_out(mix_attr)


# ---------------------------------------------------------------------------
# Text-only baseline projector
# ---------------------------------------------------------------------------

class TextOnlyProjector(nn.Module):
    """Text-only conditioning projector for fair comparison with T2S/VerbalTS."""

    def __init__(
        self,
        n_var: int,
        n_scale: int,
        n_steps: int,
        n_stages: int,
        dim_in_text: int = 512,
        dim_out: int = 128,
    ):
        super().__init__()
        self.dim_out = dim_out
        self.n_var = n_var
        self.n_scale = n_scale
        self.text_projector = _AnchorProjector(
            n_var=n_var,
            n_scale=n_scale,
            n_steps=n_steps,
            n_stages=n_stages,
            dim_in=dim_in_text,
            dim_out=dim_out,
        )

    def forward(self, text_emb: torch.Tensor, diffusion_step: torch.Tensor) -> torch.Tensor:
        out = self.text_projector(text_emb, diffusion_step)
        return out.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Paper-ready multimodal conditioner
# ---------------------------------------------------------------------------

class MultiModalConditioner(nn.Module):
    """Text+image conditioner with modality dropout and anchor fusion.

    This is the key change needed to make the model genuinely multimodal.  The
    image branch consumes an external reference/degraded TS-image, while the
    DiT still receives the noisy target image through the diffusion path.

    The conditioner supports four training conditions via dropout masks:
    ``(text,image)``, ``(text,empty)``, ``(empty,image)``, and
    ``(empty,empty)``.  These conditions are also exposed to the sampler for
    multimodal classifier-free guidance.
    """

    def __init__(
        self,
        n_var: int,
        n_scale: int,
        n_steps: int,
        n_stages: int,
        dim_in_text: int = 128,
        dim_out: int = 128,
        dim_joint: Optional[int] = None,
        image_cfg: Optional[dict] = None,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_ff: int = 256,
    ):
        super().__init__()
        self.dim_out = dim_out
        self.n_var = n_var
        self.n_scale = n_scale
        self.dim_joint = int(dim_joint or dim_out)

        image_cfg = dict(image_cfg or {})
        image_cfg.setdefault("image_emb", self.dim_joint)
        enc_type = str(image_cfg.get("encoder", image_cfg.get("encoder_type", "vit"))).lower()

        # Import lazily so text-only use does not require vision dependencies.
        if enc_type == "clip":
            from .image_encoder import ImageEncoder
            self.image_encoder = ImageEncoder(image_cfg)
            image_dim = image_cfg["image_emb"]
        elif enc_type == "cnn":
            from .image_encoder import CNNEncoder
            self.image_encoder = CNNEncoder(image_cfg)
            image_dim = image_cfg["image_emb"]
        elif enc_type == "vit":
            from .image_encoder import ViTEncoder
            self.image_encoder = ViTEncoder(image_cfg)
            image_dim = image_cfg["image_emb"]
        else:
            raise ValueError(f"Unknown image encoder type: {enc_type}")

        self.text_proj = nn.Linear(dim_in_text, self.dim_joint)
        self.image_proj = nn.Linear(image_dim, self.dim_joint)

        self.type_emb = nn.Embedding(2, self.dim_joint)  # 0=text, 1=image
        self.null_text = nn.Parameter(torch.zeros(1, 1, self.dim_joint))
        self.null_image = nn.Parameter(torch.zeros(1, 1, self.dim_joint))

        self.fuse_norm = nn.LayerNorm(self.dim_joint)
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.dim_joint, self.dim_joint),
            nn.SiLU(),
            nn.Linear(self.dim_joint, self.dim_joint),
        )
        nn.init.zeros_(self.fuse_gate[-1].weight)
        nn.init.zeros_(self.fuse_gate[-1].bias)

        self.anchor_projector = _AnchorProjector(
            n_var=n_var,
            n_scale=n_scale,
            n_steps=n_steps,
            n_stages=n_stages,
            dim_in=self.dim_joint,
            dim_out=dim_out,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
        )

    @staticmethod
    def _as_batch_mask(mask: Optional[torch.Tensor], batch: int, device: torch.device) -> torch.Tensor:
        if mask is None:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.dim() == 0:
            mask = mask.expand(batch)
        if mask.shape != (batch,):
            raise ValueError(f"drop mask must have shape {(batch,)}, got {tuple(mask.shape)}")
        return mask

    def _replace_with_null(
        self,
        tokens: torch.Tensor,
        null_token: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not drop_mask.any():
            return tokens
        B, T, D = tokens.shape
        null = null_token.expand(B, T, D)
        return torch.where(drop_mask[:, None, None], null, tokens)

    def forward(
        self,
        text_emb: Optional[torch.Tensor],
        diffusion_step: torch.Tensor,
        ref_image: Optional[torch.Tensor] = None,
        drop_text_mask: Optional[torch.Tensor] = None,
        drop_image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = diffusion_step.shape[0]
        device = diffusion_step.device

        drop_text_mask = self._as_batch_mask(drop_text_mask, B, device)
        drop_image_mask = self._as_batch_mask(drop_image_mask, B, device)

        if text_emb is None or drop_text_mask.all():
            # Skip text_proj when all samples drop text modality.
            text_tokens = self.null_text.expand(B, 1, self.dim_joint)
            drop_text_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            text_tokens = self.text_proj(text_emb.to(device))
            text_tokens = text_tokens + self.type_emb.weight[0].view(1, 1, -1)
        text_tokens = self._replace_with_null(text_tokens, self.null_text, drop_text_mask)

        if ref_image is None or drop_image_mask.all():
            # Skip encoder entirely when all samples drop image modality.
            image_tokens = self.null_image.expand(B, 1, self.dim_joint)
            drop_image_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            image_tokens = self.image_encoder(ref_image.to(device).float())
            image_tokens = self.image_proj(image_tokens)
            image_tokens = image_tokens + self.type_emb.weight[1].view(1, 1, -1)
        image_tokens = self._replace_with_null(image_tokens, self.null_image, drop_image_mask)

        tokens = torch.cat([text_tokens, image_tokens], dim=1)
        # Zero-initialized residual gate: starts close to stable projected tokens.
        tokens = self.fuse_norm(tokens + torch.tanh(self.fuse_gate(tokens)))

        out = self.anchor_projector(tokens, diffusion_step)
        return out.permute(0, 3, 1, 2).contiguous()
