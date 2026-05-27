"""MMLDM Architecture Overview — Multi-Branch Comparison Diagram.

Generates a publication-quality architecture diagram showing the full MMLDM
pipeline with branch-specific annotations.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ─── Color Palette ──────────────────────────────────────────────────────────
C = {
    'bg':           '#FFFFFF',
    'text_in':      '#E8F4FD',
    'ts_in':        '#FDEBD0',
    'vae_enc':      '#D4E6F1',
    'vae_dec':      '#D5F5E3',
    'latent':       '#FADBD8',
    'dit':          '#FCF3CF',
    'text_enc':     '#E8DAEF',
    'innov':        '#F5CBA7',
    'loss':         '#F9E79F',
    'output':       '#ABEBC6',
    'eval':         '#D2B4DE',
    'arrow':        '#2C3E50',
    'border':       '#7F8C8D',
    'header':       '#2C3E50',
    'sub':          '#5D6D7E',
    'branch_v2':    '#2980B9',
    'branch_verb':  '#27AE60',
    'branch_v4':    '#8E44AD',
}


def box(ax, x, y, w, h, text, color, fs=8, bold=False, alpha=0.9, z=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       facecolor=color, edgecolor=C['border'],
                       linewidth=1.0, alpha=alpha, zorder=z)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal',
            color=C['header'], zorder=z+1)
    return (x + w/2, y + h/2)


def arrow(ax, a, b, color=None, lw=1.2, z=4, rad=0.0):
    c = color or C['arrow']
    p = FancyArrowPatch(a, b, arrowstyle='->', color=c, lw=lw, mutation_scale=10,
                        connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(p)


def dashed_box(ax, x, y, w, h, text, color, fs=8):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                       facecolor='none', edgecolor=color, linewidth=1.5,
                       linestyle='--', alpha=0.6, zorder=1)
    ax.add_patch(b)
    ax.text(x + 0.02, y + h - 0.015, text, ha='left', va='top',
            fontsize=fs, fontweight='bold', color=color, style='italic', zorder=3)


def create_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(C['bg'])

    # ─── Title ────────────────────────────────────────────────────────────
    ax.text(0.5, 0.985, 'MMLDM: Multimodal Latent Diffusion Model for Text-to-Time-Series',
            ha='center', va='top', fontsize=15, fontweight='bold', color=C['header'])
    ax.text(0.5, 0.955, 'Spectral Dual-Latent VAE  +  DiT Flow Matching  +  Text-Guided Feature Modulation',
            ha='center', va='top', fontsize=10, color=C['sub'])

    # ─── STAGE 1 ──────────────────────────────────────────────────────────
    dashed_box(ax, 0.015, 0.22, 0.465, 0.68, 'Stage 1: Spectral Dual-Latent VAE', C['branch_v2'], 9)

    # TS Input
    ts = box(ax, 0.06, 0.82, 0.11, 0.05, 'Time Series\nx shape (L, C)', C['ts_in'], 7, True)

    # FFT
    fft = box(ax, 0.06, 0.72, 0.11, 0.05, 'FFT\nDecomposition', '#AED6F1', 7, True)
    arrow(ax, (ts[0], ts[1] - 0.025), (fft[0], fft[1] + 0.025))

    # Dual paths
    lo = box(ax, 0.04, 0.61, 0.09, 0.05, 'Low-Freq\n(Trend)', '#85C1E9', 6)
    hi = box(ax, 0.15, 0.61, 0.09, 0.05, 'High-Freq\n(Residual)', '#F1948A', 6)
    arrow(ax, (fft[0] - 0.02, fft[1] - 0.025), (lo[0], lo[1] + 0.025))
    arrow(ax, (fft[0] + 0.02, fft[1] - 0.025), (hi[0], hi[1] + 0.025))

    # Dual Encoders
    e1 = box(ax, 0.04, 0.51, 0.09, 0.05, 'Trend\nEncoder', C['vae_enc'], 6, True)
    e2 = box(ax, 0.15, 0.51, 0.09, 0.05, 'Residual\nEncoder', C['vae_enc'], 6, True)
    arrow(ax, (lo[0], lo[1] - 0.025), (e1[0], e1[1] + 0.025))
    arrow(ax, (hi[0], hi[1] - 0.025), (e2[0], e2[1] + 0.025))

    # Latent dists
    q1 = box(ax, 0.04, 0.42, 0.09, 0.05, 'q(zt | x-lo)', C['latent'], 6)
    q2 = box(ax, 0.15, 0.42, 0.09, 0.05, 'q(zr | x-hi)', C['latent'], 6)
    arrow(ax, (e1[0], e1[1] - 0.025), (q1[0], q1[1] + 0.025))
    arrow(ax, (e2[0], e2[1] - 0.025), (q2[0], q2[1] + 0.025))

    # Merge
    mg = box(ax, 0.06, 0.34, 0.18, 0.05, 'z = [zt; zr] shape (L, d)', C['latent'], 7, True)
    arrow(ax, (q1[0], q1[1] - 0.025), (mg[0] - 0.04, mg[1] + 0.025))
    arrow(ax, (q2[0], q2[1] - 0.025), (mg[0] + 0.04, mg[1] + 0.025))

    # KL + Spectral + TCLR
    box(ax, 0.04, 0.38, 0.20, 0.03,
        'KL + Spectral + TCLR Loss', C['loss'], 5.5, alpha=0.7)

    # Decoder
    dec = box(ax, 0.28, 0.34, 0.12, 0.05, 'Conv1d Decoder\n(Dual-path)', C['vae_dec'], 7, True)
    arrow(ax, (mg[0] + 0.09, mg[1]), (dec[0] - 0.06, dec[1]))

    # Recon
    rec = box(ax, 0.28, 0.26, 0.12, 0.05, 'Reconstructed\nx_hat', C['output'], 7, True)
    arrow(ax, (dec[0], dec[1] - 0.025), (rec[0], rec[1] + 0.025))

    # VAE Loss formula
    box(ax, 0.04, 0.255, 0.36, 0.045,
        'Loss-VAE = Recon + beta*KL + gamma-s*Spectral + gamma-t*TCLR',
        C['loss'], 6)

    # ─── STAGE 2 ──────────────────────────────────────────────────────────
    dashed_box(ax, 0.52, 0.22, 0.465, 0.68, 'Stage 2: Multimodal DiT with Flow Matching', C['branch_v2'], 9)

    # Text input
    tx = box(ax, 0.55, 0.82, 0.10, 0.05, 'Text\nDescription', C['text_in'], 7, True)

    # SBERT + Projector
    te = box(ax, 0.55, 0.72, 0.10, 0.05, 'SBERT Encoder\n+ Projector', C['text_enc'], 7, True)
    arrow(ax, (tx[0], tx[1] - 0.025), (te[0], te[1] + 0.025))

    # Text latent
    tl = box(ax, 0.55, 0.63, 0.10, 0.045, 'c shape (D,)', C['text_enc'], 6)
    arrow(ax, (te[0], te[1] - 0.025), (tl[0], tl[1] + 0.022))

    # MVTC
    mv = box(ax, 0.67, 0.63, 0.10, 0.045, 'MVTC: 4-View\nText Expansion', C['text_enc'], 6, True)
    arrow(ax, (tl[0] + 0.05, tl[1]), (mv[0] - 0.05, mv[1]))

    # TGFM
    tgfm = box(ax, 0.55, 0.54, 0.10, 0.045, 'TGFM\n(scale, shift)', C['innov'], 7, True)
    arrow(ax, (mv[0], mv[1] - 0.022), (tgfm[0], tgfm[1] + 0.022))

    # TS latent from VAE
    z0 = box(ax, 0.79, 0.63, 0.14, 0.045, 'z0 (VAE Latent)\nfrom Stage 1', C['latent'], 6)
    arrow(ax, (mg[0] + 0.09, mg[1]), (z0[0] - 0.09, z0[1]),
          color=C['branch_v2'], rad=-0.15)

    # Flow matching
    fm = box(ax, 0.67, 0.54, 0.14, 0.045, 'Flow Matching\nzt = (1-t)*z0 + t*eps', C['dit'], 7, True)
    arrow(ax, (z0[0], z0[1] - 0.022), (fm[0] + 0.07, fm[1] + 0.022))

    # DiT
    dit_box = box(ax, 0.67, 0.43, 0.14, 0.07,
                  'DiT Transformer\nv(zt, t; c)\n(Block-Causal + TGFM)', C['dit'], 7, True)
    arrow(ax, (fm[0], fm[1] - 0.022), (dit_box[0], dit_box[1] + 0.035))
    arrow(ax, (tgfm[0] + 0.05, tgfm[1]), (dit_box[0] - 0.07, dit_box[1] + 0.015),
          color=C['innov'], rad=0.15)

    # Velocity
    vel_box = box(ax, 0.85, 0.43, 0.10, 0.045, 'Velocity\nField ut', '#F5B7B1', 6)
    arrow(ax, (dit_box[0] + 0.07, dit_box[1]), (vel_box[0] - 0.05, vel_box[1]))

    # Euler integration
    euler = box(ax, 0.85, 0.35, 0.10, 0.045, 'Euler ODE\nIntegration', '#FADBD8', 7)
    arrow(ax, (vel_box[0], vel_box[1] - 0.022), (euler[0], euler[1] + 0.022))

    # Generated TS
    gen = box(ax, 0.72, 0.26, 0.14, 0.05, 'Generated\nTime Series x_hat', C['output'], 7, True)
    arrow(ax, (euler[0] - 0.03, euler[1] - 0.022), (gen[0] + 0.07, gen[1] + 0.025))

    # DiT Loss
    box(ax, 0.55, 0.255, 0.22, 0.05,
        'Loss-DiT = FM + CFG(0.3) + gamma-cons*Cons + FreqLoss',
        C['loss'], 6)

    # ─── Inference Flow ───────────────────────────────────────────────────
    dashed_box(ax, 0.015, 0.07, 0.97, 0.10, 'Inference Pipeline', '#E67E22', 9)

    steps = [
        ('1. Encode\nText to c', 0.075, 0.085),
        ('2. Sample\neps from N(0,I)', 0.195, 0.085),
        ('3. Euler ODE\nt=T to 0', 0.315, 0.085),
        ('4. CFG Guidance\nscale=7.0', 0.435, 0.085),
        ('5. VAE\nDecode', 0.555, 0.085),
        ('6. Un-normalize\n(Raw TS)', 0.675, 0.085),
        ('7. Evaluate\nMSE / WAPE', 0.795, 0.085),
    ]
    for label, x, y in steps:
        box(ax, x, y, 0.095, 0.055, label, C['innov'], 5.5, True, alpha=0.75)

    # ─── Innovations Badges ───────────────────────────────────────────────
    innovations = [
        ('A', 'Spectral\nDual-Latent'),
        ('B', 'TGFM +\nMVTC'),
        ('C', 'TCLR\nRegularization'),
        ('D', 'Semantic\nCurriculum'),
        ('E', 'Text-Adaptive\nSNR'),
        ('F', 'Consistency\nDistill'),
        ('Eng', 'EMA + KL\nAnneal'),
    ]
    for i, (label, name) in enumerate(innovations):
        x = 0.08 + i * 0.095
        box(ax, x, 0.008, 0.08, 0.042, f'({label})\n{name}', C['innov'], 5, True, alpha=0.85)

    # ─── Branch Annotations ───────────────────────────────────────────────
    branch_info = [
        ('[V2]  feature/mmv2-spectral-dual-latent', C['branch_v2'],
         'Full V2 pipeline: Stage 1+2, all innovations A-F'),
        ('[Verb] myverbal', C['branch_verb'],
         'V2 + Weather dataset (21-var) + CTTP/FID/JFTSD evaluation'),
        ('[V4]  mmldm-v4', C['branch_v4'],
         'V2 codebase + warmup fix (clean training baseline)'),
    ]
    for i, (name, color, desc) in enumerate(branch_info):
        y_b = 0.205 - i * 0.018
        ax.text(0.04, y_b, name, ha='left', va='center',
                fontsize=8, color=color, fontweight='bold')
        ax.text(0.38, y_b, desc, ha='left', va='center',
                fontsize=7, color=C['sub'])

    # ─── Legend ───────────────────────────────────────────────────────────
    leg = [
        mpatches.Patch(facecolor=C['vae_enc'], label='VAE Encoder'),
        mpatches.Patch(facecolor=C['vae_dec'], label='VAE Decoder'),
        mpatches.Patch(facecolor=C['latent'], label='Latent Space'),
        mpatches.Patch(facecolor=C['dit'], label='DiT Model'),
        mpatches.Patch(facecolor=C['text_enc'], label='Text Processing'),
        mpatches.Patch(facecolor=C['innov'], label='Innovations'),
        mpatches.Patch(facecolor=C['loss'], label='Loss Terms'),
        mpatches.Patch(facecolor=C['output'], label='Output'),
    ]
    ax.legend(handles=leg, loc='upper right', fontsize=6, framealpha=0.9,
              edgecolor=C['border'], ncol=2)

    out_base = 'E:/Research/TSG/myTSG_V0/mmldm_architecture'
    plt.savefig(out_base + '.png', dpi=300, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    plt.savefig(out_base + '.pdf', bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    print(f"Saved: {out_base}.png and {out_base}.pdf")
    plt.close()


if __name__ == '__main__':
    create_diagram()
