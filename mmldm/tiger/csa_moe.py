"""Channel-Structure-Aware Mixture-of-Experts (CSA-MoE) for TIGER.

True sparse token-dispatch MoE with heterogeneous experts, structural
consistency cross-attention, and auxiliary load-balancing loss.

Reference: Diff-MoE (ICML 2025), TIGER dit_model.py

Usage:
    1. Wrap each ResidualBlock with ``ChannelAwareResidualBlock``.
    2. Pass ``grids: list[tuple[int,int]]`` so it can split the flat
       token sequence by scale and reshape to 2D.
    3. The auxiliary loss is returned alongside the output and should be
       added to the total training loss.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ===========================================================================
#  MoE Gate (from Diff-MoE)
# ===========================================================================


class MoEGate(nn.Module):
    """Top-k sparse router with load-balancing auxiliary loss.

    Input:
        x: [B, L, C]

    Output:
        ti:  [B, L, k] selected expert indices
        tw:  [B, L, k] selected expert weights
        aux: scalar load-balancing loss or None
    """

    def __init__(self, dim: int, n_exp: int = 3, k: int = 1, alpha: float = 0.01):
        super().__init__()
        assert n_exp >= 1
        assert 1 <= k <= n_exp

        self.dim = dim
        self.n_exp = n_exp
        self.k = k
        self.alpha = alpha

        self.w = nn.Parameter(torch.empty(n_exp, dim))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        flat = x.reshape(B * L, C)

        prob = F.linear(flat, self.w).softmax(dim=-1)  # [B*L, E]
        tw, ti = torch.topk(prob, self.k, dim=-1, sorted=False)

        if self.k > 1:
            tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)

        aux = None
        if self.training and self.alpha > 0:
            selected = F.one_hot(ti, num_classes=self.n_exp).float().sum(dim=1)
            selected = selected / float(self.k)

            load = selected.mean(dim=0)       # actual selected expert ratio
            importance = prob.mean(dim=0)     # router probability mass

            aux = (importance * load * self.n_exp).sum() * self.alpha

        ti = ti.view(B, L, self.k)
        tw = tw.view(B, L, self.k)

        return ti, tw, aux


class AddAuxiliaryLoss(torch.autograd.Function):
    """Autograd trick: forward value unchanged, backward injects aux gradient.

    Use only if you do NOT add aux externally in the trainer.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, loss: torch.Tensor):
        ctx.requires_loss_grad = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_loss = None
        if ctx.requires_loss_grad:
            grad_loss = grad_output.new_ones(())
        return grad_output, grad_loss


# ===========================================================================
#  Heterogeneous Experts (with sparse forward)
# ===========================================================================


class GASFExpert(nn.Module):
    """Sparse row/column structural expert for Gramian-like symmetric patterns.

    Computes row/column context once, but only projects and mixes routed tokens.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.row_proj = nn.Linear(dim, dim)
        self.col_proj = nn.Linear(dim, dim)

        self.mix = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward_sparse(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        b_idx: torch.Tensor,
        l_idx: torch.Tensor,
    ) -> torch.Tensor:
        if b_idx.numel() == 0:
            return x.new_empty(0, x.shape[-1])

        B, L, C = x.shape
        if L != H * W:
            raise ValueError(f"GASFExpert expects L == H*W, got L={L}, H*W={H * W}.")

        x2d = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)

        row_ctx = x2d.mean(dim=2)  # [B, H, C]
        col_ctx = x2d.mean(dim=1)  # [B, W, C]

        row = l_idx // W
        col = l_idx % W

        r = self.row_proj(row_ctx[b_idx, row])
        c = self.col_proj(col_ctx[b_idx, col])

        return self.mix(torch.cat([r, c], dim=-1))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape

        b_idx = torch.arange(B, device=x.device).repeat_interleave(L)
        l_idx = torch.arange(L, device=x.device).repeat(B)

        y = self.forward_sparse(x, H, W, b_idx, l_idx)
        return y.view(B, L, C)


class STFTExpert(nn.Module):
    """Sparse local frequency-time expert.

    Extracts local patches only at routed token positions and applies
    the depthwise kernel there, instead of convolving every token.
    """

    def __init__(self, dim: int, ks: int = 3):
        super().__init__()
        assert ks % 2 == 1, "STFTExpert kernel size should be odd."

        self.ks = ks
        self.dw = nn.Conv2d(dim, dim, ks, padding=ks // 2, groups=dim)
        self.pw = nn.Linear(dim, dim)

        nn.init.kaiming_normal_(self.dw.weight)
        if self.dw.bias is not None:
            nn.init.zeros_(self.dw.bias)

    def forward_sparse(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        b_idx: torch.Tensor,
        l_idx: torch.Tensor,
    ) -> torch.Tensor:
        if b_idx.numel() == 0:
            return x.new_empty(0, x.shape[-1])

        B, L, C = x.shape
        if L != H * W:
            raise ValueError(f"STFTExpert expects L == H*W, got L={L}, H*W={H * W}.")

        ks = self.ks
        pad = ks // 2

        x2d = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        xpad = F.pad(x2d, (pad, pad, pad, pad))

        row = l_idx // W
        col = l_idx % W

        patches = []
        for dy in range(ks):
            for dx in range(ks):
                patches.append(xpad[b_idx, :, row + dy, col + dx])

        patches = torch.stack(patches, dim=-1).view(-1, C, ks, ks)

        weight = self.dw.weight[:, 0, :, :].unsqueeze(0)
        y = (patches * weight).sum(dim=(2, 3))

        if self.dw.bias is not None:
            y = y + self.dw.bias

        return self.pw(y)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape

        b_idx = torch.arange(B, device=x.device).repeat_interleave(L)
        l_idx = torch.arange(L, device=x.device).repeat(B)

        y = self.forward_sparse(x, H, W, b_idx, l_idx)
        return y.view(B, L, C)


class RPExpert(nn.Module):
    """Sparse token-wise gated MLP expert for recurrence/state-transition patterns."""

    def __init__(self, dim: int):
        super().__init__()

        self.g = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, dim)
        self.d = nn.Linear(dim, dim)

        nn.init.zeros_(self.d.weight)
        nn.init.zeros_(self.d.bias)

    def forward_sparse(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        b_idx: torch.Tensor,
        l_idx: torch.Tensor,
    ) -> torch.Tensor:
        if b_idx.numel() == 0:
            return x.new_empty(0, x.shape[-1])

        z = x[b_idx, l_idx]
        return self.d(torch.sigmoid(self.g(z)) * F.silu(self.u(z)))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return self.d(torch.sigmoid(self.g(x)) * F.silu(self.u(x)))


class CrossChannelRefine(nn.Module):
    """Delta-only cross-expert refinement.

    Returns only a zero-initialized refinement delta.
    Does not directly pass original tokens through.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.p = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

        nn.init.zeros_(self.p[-1].weight)
        nn.init.zeros_(self.p[-1].bias)

    def forward(
        self,
        g: torch.Tensor,
        s: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        return self.p(torch.cat([g, s, r], dim=-1))


# ===========================================================================
#  CSA-MoE Layer (true sparse token dispatch)
# ===========================================================================


class CSAMoELayer(nn.Module):
    """Channel-Structure-Aware Sparse MoE.

    True token-dispatch MoE:
        1. Gate produces top-k expert indices and weights.
        2. Tokens are physically dispatched to selected experts.
        3. Each expert computes only its routed tokens.
        4. Outputs are scattered back to the original [B, L, C] layout.
    """

    def __init__(
        self,
        dim: int,
        t_dim: Optional[int] = None,
        use_cc: bool = True,
        k: int = 1,
        alpha: float = 0.01,
        inject_aux: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.use_cc = use_cc
        self.inject_aux = inject_aux

        self.gasf = GASFExpert(dim)
        self.stft = STFTExpert(dim)
        self.rp = RPExpert(dim)

        self.experts = nn.ModuleList([
            self.gasf,
            self.stft,
            self.rp,
        ])

        self.gate = MoEGate(dim=dim, n_exp=3, k=k, alpha=alpha)
        self.router_norm = nn.LayerNorm(dim)

        if t_dim is None or t_dim == dim:
            self.t_proj = None
        else:
            self.t_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_dim, dim),
            )

        if use_cc:
            self.cc = CrossChannelRefine(dim)

        self.last_route_frac = None

    def _project_t(self, t_emb: Optional[torch.Tensor], tok: torch.Tensor):
        if t_emb is None:
            return None

        if self.t_proj is None:
            if t_emb.shape[-1] != tok.shape[-1]:
                raise ValueError(
                    f"t_emb dim {t_emb.shape[-1]} != token dim {tok.shape[-1]}. "
                    f"Set t_dim={t_emb.shape[-1]} when constructing CSAMoELayer."
                )
            return t_emb

        return self.t_proj(t_emb)

    def forward(
        self,
        tok: torch.Tensor,
        H: int,
        W: int,
        t_emb: Optional[torch.Tensor] = None,
        return_streams: bool = False,
    ):
        """
        Args:
            tok: [B, L, C]
            H, W: spatial grid size, require L == H * W
            t_emb: optional timestep embedding, [B, t_dim] or [B, C]
            return_streams: whether to return expert-specific sparse pools

        Returns:
            out: [B, L, C]
            aux: scalar aux loss or None
            optionally:
                expert_pools: list of 3 tensors, each [B, L, C]
                ti: [B, L, k]
        """
        B, L, C = tok.shape

        if L != H * W:
            raise ValueError(f"CSAMoELayer expects L == H*W, got L={L}, H*W={H * W}.")

        t = self._project_t(t_emb, tok)

        ri = tok if t is None else tok + t.unsqueeze(1)
        ri = self.router_norm(ri)

        ti, tw, aux = self.gate(ri)

        out = tok.new_zeros(B, L, C)
        expert_pools = [tok.new_zeros(B, L, C) for _ in range(3)]

        for k_idx in range(ti.shape[-1]):
            idx_k = ti[:, :, k_idx]
            w_k = tw[:, :, k_idx]

            for e_id, expert in enumerate(self.experts):
                mask = idx_k == e_id
                b_idx, l_idx = torch.where(mask)

                if b_idx.numel() == 0:
                    continue

                y = expert.forward_sparse(tok, H, W, b_idx, l_idx)
                y = y * w_k[b_idx, l_idx].unsqueeze(-1)

                out.index_put_((b_idx, l_idx), y, accumulate=True)
                expert_pools[e_id].index_put_((b_idx, l_idx), y, accumulate=True)

        if self.use_cc:
            out = out + self.cc(expert_pools[0], expert_pools[1], expert_pools[2])

        if self.training:
            with torch.no_grad():
                self.last_route_frac = torch.stack([
                    (ti == e_id).float().mean()
                    for e_id in range(3)
                ])

        if self.inject_aux and aux is not None and self.training:
            out = AddAuxiliaryLoss.apply(out, aux)

        if return_streams:
            return out, aux, expert_pools, ti

        return out, aux


# ===========================================================================
#  Structural Consistency Cross-Attention (SCCA)
# ===========================================================================


class StructuralConsistencyCrossAttention(nn.Module):
    """Geometric alignment + ring cross-attention for structural consistency.

    Input streams:
        gf: GASF-aware stream
        sf: STFT-aware stream
        rf: RP-aware stream
    """

    def __init__(self, dim: int, nh: int = 4):
        super().__init__()

        if dim % nh != 0:
            raise ValueError(f"dim={dim} must be divisible by nh={nh}.")

        self.a1 = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.a2 = nn.MultiheadAttention(dim, nh, batch_first=True)
        self.a3 = nn.MultiheadAttention(dim, nh, batch_first=True)

        self.dp = nn.Linear(dim, dim)
        self.fp = nn.Linear(dim, dim)

        self.ag = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        gf: torch.Tensor,
        sf: torch.Tensor,
        rf: torch.Tensor,
        H: int,
        W: int,
    ):
        B, L, C = gf.shape

        if L != H * W:
            raise ValueError(f"SCCA expects L == H*W, got L={L}, H*W={H * W}.")

        g2 = rearrange(gf, "b (h w) c -> b h w c", h=H, w=W)
        s2 = rearrange(sf, "b (h w) c -> b h w c", h=H, w=W)

        diag_len = min(H, W)
        diag = torch.arange(diag_len, device=gf.device)

        g_diag = g2[:, diag, diag, :]
        s_time = s2.mean(dim=1)

        n = min(g_diag.shape[1], s_time.shape[1])

        aligned_diag = g_diag[:, :n, :] + self.ag.tanh() * self.fp(s_time[:, :n, :])

        g2_aligned = g2.clone()
        for i in range(n):
            g2_aligned[:, i, i, :] = g2[:, i, i, :] + self.dp(aligned_diag[:, i, :])

        gfa = rearrange(g2_aligned, "b h w c -> b (h w) c")

        g, _ = self.a1(query=gfa, key=sf, value=sf, need_weights=False)
        s, _ = self.a2(query=sf, key=rf, value=rf, need_weights=False)
        r, _ = self.a3(query=rf, key=gfa, value=gfa, need_weights=False)

        return g, s, r


# ===========================================================================
#  Channel-Aware ResidualBlock Wrapper
# ===========================================================================


class ChannelAwareResidualBlock(nn.Module):
    """Wrap TIGER ResidualBlock with true sparse CSA-MoE + SCCA.

    Expected base block:
        bo, sk = base(x, se, ae, de, am)

    Expected bo:
        [B, C, 1, T]
        where T == sum(h * w for h, w in grids)
    """

    def __init__(
        self,
        base: nn.Module,
        grids: List[Tuple[int, int]],
        ch: int,
        t_dim: Optional[int] = None,
        use_cc: bool = True,
        k: int = 1,
        alpha: float = 0.01,
        inject_aux: bool = False,
        scca_heads: int = 4,
    ):
        super().__init__()

        self.base = base
        self.grids = grids
        self.ns = len(grids)

        self.moe = nn.ModuleList([
            CSAMoELayer(
                dim=ch,
                t_dim=t_dim,
                use_cc=use_cc,
                k=k,
                alpha=alpha,
                inject_aux=inject_aux,
            )
            for _ in range(self.ns)
        ])

        self.scca = nn.ModuleList([
            StructuralConsistencyCrossAttention(ch, nh=scca_heads)
            for _ in range(self.ns)
        ])

        self.gate = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(self.ns)
        ])

        self.scca_gate = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(self.ns)
        ])

    def forward(
        self,
        x: torch.Tensor,
        se: torch.Tensor,
        ae: torch.Tensor,
        de: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        am: Optional[torch.Tensor] = None,
    ):
        bo, sk = self.base(x, se, ae, de, am)

        if bo.dim() != 4:
            raise ValueError(f"Expected base output bo to be 4D [B,C,1,T], got {bo.shape}.")

        if bo.shape[2] != 1:
            raise ValueError(f"Expected bo.shape[2] == 1, got bo.shape={bo.shape}.")

        f = bo.squeeze(2).permute(0, 2, 1).contiguous()  # [B, L, C]

        expected_tokens = sum(h * w for h, w in self.grids)
        if expected_tokens != f.shape[1]:
            raise ValueError(
                f"Grid token count {expected_tokens} does not match feature length {f.shape[1]}."
            )

        outs = []
        aux_list = []
        offset = 0

        for scale_id, (nh, nw) in enumerate(self.grids):
            nt = nh * nw
            tok = f[:, offset:offset + nt, :].contiguous()

            moe_delta, aux, expert_pools, _ = self.moe[scale_id](
                tok,
                H=nh,
                W=nw,
                t_emb=t_emb,
                return_streams=True,
            )

            g_stream = tok + expert_pools[0]
            s_stream = tok + expert_pools[1]
            r_stream = tok + expert_pools[2]

            gs, ss, rs = self.scca[scale_id](
                g_stream,
                s_stream,
                r_stream,
                nh,
                nw,
            )

            scca_delta = ((gs + ss + rs) / 3.0) - tok
            delta = moe_delta + self.scca_gate[scale_id].tanh() * scca_delta

            out_tok = tok + self.gate[scale_id].tanh() * delta
            outs.append(out_tok)

            if aux is not None:
                aux_list.append(aux)

            offset += nt

        out = torch.cat(outs, dim=1)
        out = out.permute(0, 2, 1).unsqueeze(2).contiguous()

        aux_total = None
        if len(aux_list) > 0:
            aux_total = torch.stack(aux_list).mean()

        return out, sk, aux_total
