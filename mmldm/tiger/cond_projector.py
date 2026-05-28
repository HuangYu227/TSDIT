"""Condition Projector for TIGER — text-only conditioning for diffusion.

Adapted from VerbalTS TextProjectorMVarMScaleMStep to support text-only
conditioning with multi-focal anchor projection.

Classes
-------
TextOnlyProjector
    Text-only conditioning projector (no image path).
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
# Public: text-only projector
# ---------------------------------------------------------------------------

class TextOnlyProjector(nn.Module):
    """Text-only conditioning projector (no image path).

    Text-only conditioning projector for fair comparison with text-only baselines (T2S, VerbalTS).
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
