"""Condition Projector for TIGER — fuse text and image embeddings for conditioning.

Adapted from VerbalTS TextProjectorMVarMScaleMStep to support both text-only
and multimodal (text + image) conditioning paths with learnable gated fusion.

Classes
-------
ImageTextProjector
    Dual-path projector: text path + image path with gated fusion.
ImageOnlyProjector
    Single-path projector for image-only conditioning (ablation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: build a cross-attention stack (reused by every anchor group)
# ---------------------------------------------------------------------------

def _make_cross_attn(dim_in: int, n_heads: int = 8, n_layers: int = 2,
                     dim_ff: int = 64) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=dim_in, nhead=n_heads,
        dim_feedforward=dim_ff, activation="gelu", batch_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=n_layers)


# ---------------------------------------------------------------------------
# Core building block: single-modality anchor projector
# ---------------------------------------------------------------------------
# This mirrors VerbalTS TextProjectorMVarMScaleMStep exactly, but is
# factored out so we can instantiate it twice (once for text, once for image).

class _AnchorProjector(nn.Module):
    """Variable + Scale + Diffusion-step anchor projector.

    Identical to VerbalTS ``TextProjectorMVarMScaleMStep``.

    Parameters
    ----------
    n_var, n_scale, n_steps, n_stages : int
        Anchor counts (same semantics as VerbalTS).
    dim_in : int
        Input token embedding dimension.
    dim_out : int
        Output embedding dimension.
    """

    def __init__(self, n_var: int, n_scale: int, n_steps: int,
                 n_stages: int, dim_in: int = 128, dim_out: int = 128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_var = n_var
        self.seg_size = n_steps // n_stages + 1

        # Learnable anchor queries
        self.var_emb = nn.Parameter(torch.zeros((1, n_var, dim_in)))
        self.scale_emb = nn.Parameter(torch.zeros((1, n_scale, dim_in)))
        self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))

        # Cross-attention stacks (anchors attend to conditioning tokens)
        self.var_cross_attn = _make_cross_attn(dim_in)
        self.scale_cross_attn = _make_cross_attn(dim_in)
        self.step_cross_attn = _make_cross_attn(dim_in)

        # Output projection
        self.proj_out = nn.Linear(dim_in, dim_out)

    def forward(self, attr: torch.Tensor,
                diffusion_step: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        attr : torch.Tensor
            Conditioning tokens, shape ``(B, T, dim_in)``.
        diffusion_step : torch.Tensor
            Integer diffusion timestep per sample, shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, dim_out, n_var, Nl)`` where ``Nl`` is the number of
            scale anchors (``n_scale``) — exactly matching VerbalTS output.
        """
        B = attr.shape[0]

        # --- Variable anchors: (B, n_var, dim_in) -> (B, n_var, 1, dim_in)
        var_emb = self.var_emb.expand(B, -1, -1)
        mvar_attr = self.var_cross_attn(tgt=var_emb, memory=attr)
        mvar_attr = mvar_attr[:, :, None, :]  # (B, n_var, 1, dim_in)

        # --- Scale anchors: (B, n_scale, dim_in) -> (B, 1, n_scale, dim_in)
        #     then broadcast to (B, n_var, n_scale, dim_in)
        scale_emb = self.scale_emb.expand(B, -1, -1)
        mscale_attr = self.scale_cross_attn(tgt=scale_emb, memory=attr)
        mscale_attr = mscale_attr[:, None, :, :].expand(-1, self.n_var, -1, -1)

        # --- Diffusion-step anchors: select by step, then broadcast
        step_emb = self.step_emb.expand(B, -1, -1)
        mstep_attr = self.step_cross_attn(tgt=step_emb, memory=attr)
        indices = diffusion_step // self.seg_size
        indices = indices[:, None, None]
        mstep_attr = torch.gather(
            mstep_attr, dim=1,
            index=indices.expand(-1, -1, mstep_attr.shape[-1]),
        )
        mstep_attr = mstep_attr[:, None, :, :].expand(-1, self.n_var, -1, -1)

        # --- Combine: additive fusion of three anchor types
        mix_attr = mvar_attr + mscale_attr + mstep_attr  # (B, n_var, n_scale, dim_in)

        # --- Project to output dim
        out = self.proj_out(mix_attr)  # (B, n_var, n_scale, dim_out)
        return out


# ---------------------------------------------------------------------------
# Public: dual-path multimodal projector
# ---------------------------------------------------------------------------

class ImageTextProjector(nn.Module):
    """Fuse text and image embeddings for conditioning the diffusion model.

    Two parallel anchor-projector branches (text path and image path) each
    produce a tensor of shape ``(B, dim_out, n_var, n_scale)``.  A learnable
    sigmoid gate fuses them element-wise:

        gate = sigmoid(W @ [text_proj ; image_proj])
        out  = gate * text_proj + (1 - gate) * image_proj

    Parameters
    ----------
    n_var : int
        Number of variable anchors (e.g. number of channels/locations).
    n_scale : int
        Number of scale anchors (multi-resolution).
    n_steps : int
        Total number of diffusion timesteps (for stage segmentation).
    n_stages : int
        Number of diffusion-stage anchors.
    dim_in_text : int
        Dimension of text token embeddings (from CLIPTextEncoder).
    dim_in_image : int
        Dimension of image token embeddings (from ImageEncoder).
    dim_out : int
        Output conditioning dimension fed to the diffusion backbone.
    """

    def __init__(self, n_var: int, n_scale: int, n_steps: int,
                 n_stages: int, dim_in_text: int = 512,
                 dim_in_image: int = 512, dim_out: int = 128):
        super().__init__()
        self.dim_out = dim_out
        self.n_var = n_var
        self.n_scale = n_scale

        # --- Text path
        self.text_projector = _AnchorProjector(
            n_var=n_var, n_scale=n_scale,
            n_steps=n_steps, n_stages=n_stages,
            dim_in=dim_in_text, dim_out=dim_out,
        )

        # --- Image path
        self.image_projector = _AnchorProjector(
            n_var=n_var, n_scale=n_scale,
            n_steps=n_steps, n_stages=n_stages,
            dim_in=dim_in_image, dim_out=dim_out,
        )

        # --- Learnable gated fusion (operates on last-dim concatenation)
        self.gate_linear = nn.Linear(dim_out * 2, dim_out)

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        text_emb : torch.Tensor
            Text token embeddings, shape ``(B, N_text, dim_in_text)``.
        image_emb : torch.Tensor
            Image token embeddings, shape ``(B, N_image, dim_in_image)``.
        diffusion_step : torch.Tensor
            Integer diffusion timestep per sample, shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, dim_out, n_var, n_scale)``.
        """
        # Text path:  (B, n_var, n_scale, dim_out)
        text_proj = self.text_projector(text_emb, diffusion_step)

        # Image path: (B, n_var, n_scale, dim_out)
        image_proj = self.image_projector(image_emb, diffusion_step)

        # Gated fusion
        gate = torch.sigmoid(
            self.gate_linear(torch.cat([text_proj, image_proj], dim=-1))
        )  # (B, n_var, n_scale, dim_out)
        out = gate * text_proj + (1.0 - gate) * image_proj

        # Permute to match VerbalTS convention: (B, dim_out, n_var, n_scale)
        out = out.permute(0, 3, 1, 2)
        return out


# ---------------------------------------------------------------------------
# Public: image-only projector (ablation — no text path)
# ---------------------------------------------------------------------------

class ImageOnlyProjector(nn.Module):
    """Image-only conditioning projector (for ablation studies).

    Uses a single :class:`_AnchorProjector` on image embeddings with no text
    path and no gating.  Output shape is identical to
    :class:`ImageTextProjector`.

    Parameters
    ----------
    n_var, n_scale, n_steps, n_stages : int
        Anchor counts (same semantics as VerbalTS).
    dim_in_image : int
        Dimension of image token embeddings.
    dim_out : int
        Output conditioning dimension.
    """

    def __init__(self, n_var: int, n_scale: int, n_steps: int,
                 n_stages: int, dim_in_image: int = 512,
                 dim_out: int = 128):
        super().__init__()
        self.dim_out = dim_out
        self.n_var = n_var
        self.n_scale = n_scale

        self.image_projector = _AnchorProjector(
            n_var=n_var, n_scale=n_scale,
            n_steps=n_steps, n_stages=n_stages,
            dim_in=dim_in_image, dim_out=dim_out,
        )

    def forward(
        self,
        image_emb: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image_emb : torch.Tensor
            Image token embeddings, shape ``(B, N_image, dim_in_image)``.
        diffusion_step : torch.Tensor
            Integer diffusion timestep per sample, shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, dim_out, n_var, n_scale)``.
        """
        out = self.image_projector(image_emb, diffusion_step)
        out = out.permute(0, 3, 1, 2)  # (B, dim_out, n_var, n_scale)
        return out


# ---------------------------------------------------------------------------
# Public: text-only projector (for fair comparison with T2S/VerbalTS)
# ---------------------------------------------------------------------------

class TextOnlyProjector(nn.Module):
    """Text-only conditioning projector (no image path).

    Identical to :class:`ImageOnlyProjector` but operates on text embeddings.
    Used for fair comparison with text-only baselines (T2S, VerbalTS).
    """

    def __init__(self, n_var: int, n_scale: int, n_steps: int,
                 n_stages: int, dim_in_text: int = 512,
                 dim_out: int = 128):
        super().__init__()
        self.dim_out = dim_out
        self.n_var = n_var
        self.n_scale = n_scale

        self.text_projector = _AnchorProjector(
            n_var=n_var, n_scale=n_scale,
            n_steps=n_steps, n_stages=n_stages,
            dim_in=dim_in_text, dim_out=dim_out,
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> torch.Tensor:
        out = self.text_projector(text_emb, diffusion_step)
        out = out.permute(0, 3, 1, 2)  # (B, dim_out, n_var, n_scale)
        return out
