"""MMLDM v2 Overview Diagram — Publication Quality."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─── Color Palette (Paper-friendly) ───────────────────────────────────────────
COLORS = {
    'bg':           '#FFFFFF',
    'text_input':   '#E8F4FD',  # Light blue
    'ts_input':     '#FDF2E8',  # Light orange
    'vae_enc':      '#D4E6F1',  # Blue
    'vae_dec':      '#D5F5E3',  # Green
    'latent':       '#FADBD8',  # Pink
    'dit':          '#FCF3CF',  # Yellow
    'text_enc':     '#E8DAEF',  # Purple
    'innovation':   '#F5CBA7',  # Orange
    'loss':         '#F9E79F',  # Light yellow
    'output':       '#ABEBC6',  # Green
    'arrow':        '#2C3E50',  # Dark gray
    'border':       '#7F8C8D',  # Gray
    'header':       '#2C3E50',  # Dark blue
    'subheader':    '#5D6D7E',  # Medium gray
}

def draw_rounded_box(ax, x, y, w, h, label, color, fontsize=9, bold=False, alpha=0.9):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=COLORS['border'],
                         linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=COLORS['header'], zorder=3)
    return (x + w/2, y + h/2)

def draw_arrow(ax, start, end, color=None, style='->', lw=1.5, connectionstyle="arc3,rad=0.0"):
    """Draw an arrow between two points."""
    c = color or COLORS['arrow']
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                           color=c, lw=lw, mutation_scale=12,
                           connectionstyle=connectionstyle, zorder=4)
    ax.add_patch(arrow)

def draw_dashed_box(ax, x, y, w, h, label, color, fontsize=8):
    """Draw a dashed boundary box."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                         facecolor='none', edgecolor=color,
                         linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)
    ax.add_patch(box)
    ax.text(x + 0.02, y + h - 0.02, label, ha='left', va='top',
            fontsize=fontsize, fontweight='bold', color=color,
            style='italic', zorder=3)

def create_overview():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['bg'])

    # ─── Title ─────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97, 'MMLDM v2: Spectral Dual-Latent Diffusion for Text-to-Time-Series Generation',
            ha='center', va='top', fontsize=14, fontweight='bold', color=COLORS['header'])
    ax.text(0.5, 0.94, 'Multimodal Latent Diffusion Model with FFT Decomposition & Adaptive Semantic Patching',
            ha='center', va='top', fontsize=10, color=COLORS['subheader'])

    # ─── Stage 1: VAE Training (Left) ─────────────────────────────────────
    draw_dashed_box(ax, 0.02, 0.15, 0.46, 0.72, 'Stage 1: Spectral Dual-Latent VAE', '#3498DB', 9)

    # Input: Time Series
    ts_x, ts_y = 0.08, 0.78
    draw_rounded_box(ax, ts_x, ts_y, 0.12, 0.06, 'Time Series\n$x \\in \\mathbb{R}^{L \\times C}$',
                    COLORS['ts_input'], 8, True)

    # FFT Decomposition
    fft_x, fft_y = 0.08, 0.66
    draw_rounded_box(ax, fft_x, fft_y, 0.12, 0.06, 'FFT\nDecomposition', '#AED6F1', 8, True)
    draw_arrow(ax, (ts_x + 0.06, ts_y), (fft_x + 0.06, fft_y + 0.06))

    # Low-freq / High-freq
    low_x, low_y = 0.05, 0.55
    high_x, high_y = 0.17, 0.55
    draw_rounded_box(ax, low_x, low_y, 0.10, 0.05, 'Low-freq\n(Trend)', '#85C1E9', 7)
    draw_rounded_box(ax, high_x, high_y, 0.10, 0.05, 'High-freq\n(Residual)', '#F1948A', 7)
    draw_arrow(ax, (fft_x + 0.04, fft_y), (low_x + 0.05, low_y + 0.05))
    draw_arrow(ax, (fft_x + 0.08, fft_y), (high_x + 0.05, high_y + 0.05))

    # Dual Encoders
    enc1_x, enc1_y = 0.05, 0.44
    enc2_x, enc2_y = 0.17, 0.44
    draw_rounded_box(ax, enc1_x, enc1_y, 0.10, 0.05, 'Conv1d\nEncoder', COLORS['vae_enc'], 7, True)
    draw_rounded_box(ax, enc2_x, enc2_y, 0.10, 0.05, 'Conv1d\nEncoder', COLORS['vae_enc'], 7, True)
    draw_arrow(ax, (low_x + 0.05, low_y), (enc1_x + 0.05, enc1_y + 0.05))
    draw_arrow(ax, (high_x + 0.05, high_y), (enc2_x + 0.05, enc2_y + 0.05))

    # Latent Projections
    proj1_x, proj1_y = 0.05, 0.35
    proj2_x, proj2_y = 0.17, 0.35
    draw_rounded_box(ax, proj1_x, proj1_y, 0.10, 0.04, '$q(z_t|x_{low})$', COLORS['latent'], 7)
    draw_rounded_box(ax, proj2_x, proj2_y, 0.10, 0.04, '$q(z_r|x_{high})$', COLORS['latent'], 7)
    draw_arrow(ax, (enc1_x + 0.05, enc1_y), (proj1_x + 0.05, proj1_y + 0.04))
    draw_arrow(ax, (enc2_x + 0.05, enc2_y), (proj2_x + 0.05, proj2_y + 0.04))

    # Merged Latent
    merge_x, merge_y = 0.08, 0.26
    draw_rounded_box(ax, merge_x, merge_y, 0.16, 0.05, '$z = [z_t; z_r] \\in \\mathbb{R}^{L \\times d}$',
                    COLORS['latent'], 8, True)
    draw_arrow(ax, (proj1_x + 0.05, proj1_y), (merge_x + 0.06, merge_y + 0.05))
    draw_arrow(ax, (proj2_x + 0.05, proj2_y), (merge_x + 0.10, merge_y + 0.05))

    # Decoder
    dec_x, dec_y = 0.28, 0.26
    draw_rounded_box(ax, dec_x, dec_y, 0.14, 0.05, 'Conv1d\nDecoder', COLORS['vae_dec'], 8, True)
    draw_arrow(ax, (merge_x + 0.16, merge_y + 0.025), (dec_x, dec_y + 0.025))

    # Reconstruction
    recon_x, recon_y = 0.28, 0.18
    draw_rounded_box(ax, recon_x, recon_y, 0.14, 0.05, 'Reconstruction\n$\\hat{x}$', COLORS['output'], 8, True)
    draw_arrow(ax, (dec_x + 0.07, dec_y), (recon_x + 0.07, recon_y + 0.05))

    # VAE Losses
    loss_x, loss_y = 0.05, 0.18
    draw_rounded_box(ax, loss_x, loss_y, 0.18, 0.05,
                    '$\\mathcal{L}_{VAE} = \\mathcal{L}_{recon} + \\beta\\mathcal{L}_{KL} + \\mathcal{L}_{spectral} + \\mathcal{L}_{TCLR}$',
                    COLORS['loss'], 7)

    # ─── Stage 2: DiT Training (Right) ────────────────────────────────────
    draw_dashed_box(ax, 0.52, 0.15, 0.46, 0.72, 'Stage 2: Multimodal DiT with Flow Matching', '#E74C3C', 9)

    # Text Input
    txt_x, txt_y = 0.56, 0.78
    draw_rounded_box(ax, txt_x, txt_y, 0.12, 0.06, 'Text\nDescription', COLORS['text_input'], 8, True)

    # Text Encoder
    txt_enc_x, txt_enc_y = 0.56, 0.67
    draw_rounded_box(ax, txt_enc_x, txt_enc_y, 0.12, 0.05, 'Text Encoder\n(SBERT + Proj)', COLORS['text_enc'], 7, True)
    draw_arrow(ax, (txt_x + 0.06, txt_y), (txt_enc_x + 0.06, txt_enc_y + 0.05))

    # Text Latent
    txt_lat_x, txt_lat_y = 0.56, 0.57
    draw_rounded_box(ax, txt_lat_x, txt_lat_y, 0.12, 0.04, '$c \\in \\mathbb{R}^{L_t \\times d}$',
                    COLORS['text_enc'], 7)
    draw_arrow(ax, (txt_enc_x + 0.06, txt_enc_y), (txt_lat_x + 0.06, txt_lat_y + 0.04))

    # TS Latent (from VAE)
    ts_lat_x, ts_lat_y = 0.74, 0.57
    draw_rounded_box(ax, ts_lat_x, ts_lat_y, 0.12, 0.04, '$z_0 \\in \\mathbb{R}^{L \\times d}$',
                    COLORS['latent'], 7)
    # Arrow from VAE latent
    draw_arrow(ax, (merge_x + 0.16, merge_y + 0.025), (ts_lat_x, ts_lat_y + 0.02),
              color='#E74C3C', connectionstyle="arc3,rad=-0.2")

    # Flow Matching
    flow_x, flow_y = 0.65, 0.47
    draw_rounded_box(ax, flow_x, flow_y, 0.14, 0.05, 'Flow Matching\n$z_t = (1-t)z_0 + t\\epsilon$',
                    COLORS['dit'], 8, True)
    draw_arrow(ax, (ts_lat_x + 0.06, ts_lat_y), (flow_x + 0.04, flow_y + 0.05))
    draw_arrow(ax, (txt_lat_x + 0.06, txt_lat_y), (flow_x + 0.10, flow_y + 0.05))

    # DiT Model
    dit_x, dit_y = 0.65, 0.37
    draw_rounded_box(ax, dit_x, dit_y, 0.14, 0.06, 'DiT\n$v_\\psi(z_t, t; c)$\n(Block-Causal)',
                    COLORS['dit'], 7, True)
    draw_arrow(ax, (flow_x + 0.07, flow_y), (dit_x + 0.07, dit_y + 0.06))

    # Velocity Field
    vel_x, vel_y = 0.85, 0.47
    draw_rounded_box(ax, vel_x, vel_y, 0.10, 0.05, 'Velocity\nField $u_t$', '#F5B7B1', 7)
    draw_arrow(ax, (dit_x + 0.14, dit_y + 0.03), (vel_x, vel_y + 0.025))

    # Iterative Denoising
    denoise_x, denoise_y = 0.85, 0.37
    draw_rounded_box(ax, denoise_x, denoise_y, 0.10, 0.05, 'Iterative\nDenoising', '#FADBD8', 7)
    draw_arrow(ax, (vel_x + 0.05, vel_y), (denoise_x + 0.05, denoise_y + 0.05))

    # Generated TS
    gen_x, gen_y = 0.72, 0.26
    draw_rounded_box(ax, gen_x, gen_y, 0.14, 0.05, 'Generated\nTime Series', COLORS['output'], 8, True)
    draw_arrow(ax, (denoise_x + 0.05, denoise_y), (gen_x + 0.12, gen_y + 0.05))

    # DiT Losses
    dit_loss_x, dit_loss_y = 0.55, 0.18
    draw_rounded_box(ax, dit_loss_x, dit_loss_y, 0.20, 0.05,
                    '$\\mathcal{L}_{DiT} = \\mathcal{L}_{FM} + \\gamma\\mathcal{L}_{cons} + \\mathcal{L}_{TCLR}$',
                    COLORS['loss'], 7)

    # ─── Innovations (Bottom) ─────────────────────────────────────────────
    innovations = [
        ('A', 'Spectral\nDual-Latent', 0.08),
        ('C', 'TCLR\nRegularization', 0.22),
        ('D', 'Semantic\nCurriculum', 0.38),
        ('E', 'Text-Adaptive\nSNR', 0.54),
        ('F', 'Consistency\nDistillation', 0.70),
        ('Eng', 'EMA + KL Anneal\n+ Standardization', 0.86),
    ]

    for label, name, x in innovations:
        draw_rounded_box(ax, x - 0.04, 0.03, 0.10, 0.08,
                        f'({label})\n{name}', COLORS['innovation'], 6, True, alpha=0.8)

    # Innovation labels header
    ax.text(0.5, 0.125, 'Innovations & Engineering Improvements',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=COLORS['subheader'], style='italic')

    # ─── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['vae_enc'], label='VAE Encoder'),
        mpatches.Patch(facecolor=COLORS['vae_dec'], label='VAE Decoder'),
        mpatches.Patch(facecolor=COLORS['latent'], label='Latent Space'),
        mpatches.Patch(facecolor=COLORS['dit'], label='DiT Model'),
        mpatches.Patch(facecolor=COLORS['text_enc'], label='Text Processing'),
        mpatches.Patch(facecolor=COLORS['innovation'], label='Innovations'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
             framealpha=0.9, edgecolor=COLORS['border'], ncol=2)

    plt.tight_layout()
    plt.savefig('E:/Research/TSG/myTSG_V0/mmv2_overview.png', dpi=300, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.savefig('E:/Research/TSG/myTSG_V0/mmv2_overview.pdf', bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    print("Saved: mmv2_overview.png and mmv2_overview.pdf")

if __name__ == '__main__':
    create_overview()
