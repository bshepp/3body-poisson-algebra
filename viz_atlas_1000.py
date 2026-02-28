#!/usr/bin/env python3
"""
1000x1000 Atlas Visualization Suite
=====================================

Renders the high-resolution (800x1000 clean) 1/r atlas data as:
  1. Static images: rank map, gap ratio, SV116 landscape, S3 symmetry overlay
  2. Interactive Plotly HTML with multiple color modes
  3. Rotating shape sphere animation (GIF)

Usage:
    python viz_atlas_1000.py              # everything
    python viz_atlas_1000.py static       # static images only
    python viz_atlas_1000.py interactive  # plotly HTML only
    python viz_atlas_1000.py animation    # rotating GIF only
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DATA_DIR = os.path.join('atlas_1000', '1_r', 'merged_clean')
EXCL_DIR = os.path.join('atlas_1000', '1_r', 'block_0100_0200')
OUT_DIR = os.path.join('atlas_1000', 'viz')
STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]

MU_FULL = np.linspace(0.05, 5.0, 1000)
PHI_FULL = np.linspace(0.05, np.pi - 0.05, 1000)


def mu_phi_to_shape_sphere(mu, phi):
    w2_sq = mu**2 - mu * np.cos(phi) + 0.25
    N = 1.0 + w2_sq
    s1 = (1.0 - w2_sq) / N
    s2 = 2.0 * (mu * np.cos(phi) - 0.5) / N
    s3 = 2.0 * mu * np.sin(phi) / N
    return s1, s2, s3


def load_clean():
    return {
        'mu':   np.load(os.path.join(DATA_DIR, 'mu_vals.npy')),
        'phi':  np.load(os.path.join(DATA_DIR, 'phi_vals.npy')),
        'rank': np.load(os.path.join(DATA_DIR, 'rank_map.npy')),
        'gap':  np.load(os.path.join(DATA_DIR, 'gap_map.npy')),
        'sv':   np.load(os.path.join(DATA_DIR, 'sv_spectra.npy')),
    }


def draw_isosceles_flat(ax, mu, phi):
    phi_dense = np.linspace(phi[0], phi[-1], 800)

    ax.axhline(1.0, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8,
               label=r'$r_{13}=r_{12}$')

    mu_c2 = 2 * np.cos(phi_dense)
    m2 = (mu_c2 >= mu[0]) & (mu_c2 <= mu[-1])
    if m2.any():
        ax.plot(phi_dense[m2] * 180 / np.pi, mu_c2[m2],
                color='lime', linewidth=1.5, linestyle='--', alpha=0.8,
                label=r'$r_{23}=r_{12}$')

    cos_p = np.cos(phi_dense)
    safe = cos_p > 0.01
    mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
    m3 = np.isfinite(mu_c3) & (mu_c3 >= mu[0]) & (mu_c3 <= mu[-1])
    if m3.any():
        ax.plot(phi_dense[m3] * 180 / np.pi, mu_c3[m3],
                color='magenta', linewidth=1.5, linestyle='--', alpha=0.8,
                label=r'$r_{13}=r_{23}$')

    if 0.4 <= 1.0 <= 1.8:
        ax.plot(60, 1.0, '*', color='white', markersize=14,
                markeredgecolor='black', markeredgewidth=1, zorder=15)
        ax.annotate('Lagrange', (60, 1.0), textcoords='offset points',
                    xytext=(8, 8), color='white', fontsize=10,
                    fontweight='bold', path_effects=STROKE)


def draw_isosceles_3d(ax, mu_range=(0.2, 5.0)):
    phi_c1 = np.linspace(0.1, np.pi - 0.1, 500)
    s1c, s2c, s3c = mu_phi_to_shape_sphere(1.0, phi_c1)
    ax.plot(s1c, s2c, s3c, color='cyan', linewidth=2, alpha=0.9, zorder=5)

    phi_c2 = np.linspace(0.05, np.pi / 2 - 0.05, 500)
    mu_c2 = 2 * np.cos(phi_c2)
    v2 = (mu_c2 >= mu_range[0]) & (mu_c2 <= mu_range[1])
    if v2.any():
        s1c, s2c, s3c = mu_phi_to_shape_sphere(mu_c2[v2], phi_c2[v2])
        ax.plot(s1c, s2c, s3c, color='lime', linewidth=2, alpha=0.9, zorder=5)

    phi_c3 = np.linspace(0.05, np.pi / 2 - 0.05, 500)
    cos_c3 = np.cos(phi_c3)
    mu_c3 = 1.0 / (2 * cos_c3)
    v3 = (mu_c3 >= mu_range[0]) & (mu_c3 <= mu_range[1])
    if v3.any():
        s1c, s2c, s3c = mu_phi_to_shape_sphere(mu_c3[v3], phi_c3[v3])
        ax.plot(s1c, s2c, s3c, color='magenta', linewidth=2, alpha=0.9, zorder=5)

    s1l, s2l, s3l = mu_phi_to_shape_sphere(1.0, np.pi / 3)
    ax.scatter([s1l], [s2l], [s3l], color='white', s=80, zorder=15,
               edgecolors='black', linewidth=1.2, depthshade=False, marker='*')


# ===================================================================
# Static images
# ===================================================================

def render_static():
    print("\n  Loading 1000x1000 clean data...", flush=True)
    D = load_clean()
    mu, phi = D['mu'], D['phi']
    rank, gap, sv = D['rank'], D['gap'], D['sv']

    valid_rank = rank.copy().astype(float)
    valid_rank[rank == -1] = np.nan

    log_gap = np.log10(np.clip(gap, 1.0, None))
    log_gap[rank == -1] = np.nan

    MU, PHI = np.meshgrid(mu, phi, indexing='ij')
    phi_deg = phi * 180 / np.pi

    # Subsample for 3D sphere (full 800x1000 is too heavy for mpl surface)
    step_mu = max(1, len(mu) // 200)
    step_phi = max(1, len(phi) // 200)
    mu_s = mu[::step_mu]
    phi_s = phi[::step_phi]
    MU_s, PHI_s = np.meshgrid(mu_s, phi_s, indexing='ij')
    S1, S2, S3 = mu_phi_to_shape_sphere(MU_s, PHI_s)

    # --- Figure 1: Rank + Gap dual panel ---
    print("  Fig 1: Rank map + gap ratio...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), facecolor='white')

    im1 = ax1.pcolormesh(phi_deg, mu, valid_rank, cmap='RdYlGn',
                         vmin=114, vmax=126, shading='auto')
    draw_isosceles_flat(ax1, mu, phi)
    ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax1.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=13)
    ax1.set_title('Rank Map (800x1000)', fontsize=14)
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.02)
    cb1.set_label('Local rank', fontsize=12)

    im2 = ax2.pcolormesh(phi_deg, mu, log_gap, cmap='inferno', shading='auto')
    draw_isosceles_flat(ax2, mu, phi)
    ax2.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax2.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=13)
    ax2.set_title(r'$\log_{10}$(Gap Ratio) (800x1000)', fontsize=14)
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cb2.set_label(r'$\log_{10}$(gap ratio)', fontsize=12)

    fig.suptitle('1000x1000 Atlas — 1/r Potential (Newton), Level 3\n'
                 r'800 clean rows ($\mu$ = 1.04–5.00), $\varepsilon$ = 5×10$^{-3}$',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'atlas_1000_rank_gap.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("    Saved atlas_1000_rank_gap.png")

    # --- Figure 2: Gap ratio on shape sphere ---
    print("  Fig 2: Shape sphere gap ratio...", flush=True)
    log_gap_s = log_gap[::step_mu, ::step_phi]
    valid_s = np.isfinite(log_gap_s)
    vmin_g = np.nanpercentile(log_gap_s[valid_s], 1)
    vmax_g = np.nanpercentile(log_gap_s[valid_s], 99)
    norm_g = colors.Normalize(vmin=vmin_g, vmax=vmax_g)
    cmap_g = plt.get_cmap('inferno')
    FC = cmap_g(norm_g(np.clip(np.where(valid_s, log_gap_s, vmin_g), vmin_g, vmax_g)))

    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S1, S2, S3, facecolors=FC,
                    rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.95)
    FC_m = FC.copy(); FC_m[:, :, 3] = 0.35
    ax.plot_surface(S1, S2, -S3, facecolors=FC_m,
                    rstride=1, cstride=1, shade=False, antialiased=True)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
            'k-', alpha=0.25, linewidth=0.8)
    draw_isosceles_3d(ax)
    ax.set_xlabel('$s_1$'); ax.set_ylabel('$s_2$'); ax.set_zlabel('$s_3$')
    ax.set_title('Gap Ratio on Shape Sphere — 1/r (Newton)\n'
                 '800×1000 grid, Level 3', fontsize=14, fontweight='bold')
    ax.view_init(elev=25, azim=-50)
    sm = cm.ScalarMappable(norm=norm_g, cmap=cmap_g); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, label=r'$\log_{10}$(gap ratio)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'atlas_1000_sphere_gap.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("    Saved atlas_1000_sphere_gap.png")

    # --- Figure 3: SV116 landscape ---
    print("  Fig 3: SV116 landscape...", flush=True)
    sv116 = sv[:, :, 115]
    sv1 = sv[:, :, 0]
    sv116_norm = np.full_like(sv116, np.nan)
    mask116 = (rank == 116) & (sv1 > 0) & (sv116 > 0)
    sv116_norm[mask116] = np.log10(sv116[mask116] / sv1[mask116])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), facecolor='white')

    vmin_sv = np.nanpercentile(sv116_norm[mask116], 1)
    vmax_sv = np.nanpercentile(sv116_norm[mask116], 99)

    im1 = ax1.pcolormesh(phi_deg, mu, sv116_norm, cmap='viridis',
                         vmin=vmin_sv, vmax=vmax_sv, shading='auto')
    draw_isosceles_flat(ax1, mu, phi)
    ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax1.set_ylabel(r'$\mu$', fontsize=13)
    ax1.set_title(r'$\log_{10}(\sigma_{116} / \sigma_1)$ — Flat view', fontsize=14)
    fig.colorbar(im1, ax=ax1, pad=0.02)

    sv115 = sv[:, :, 114]
    ratio = np.full_like(sv116, np.nan)
    mask_ratio = mask116 & (sv116 > 1e-30)
    ratio[mask_ratio] = np.log10(sv115[mask_ratio] / sv116[mask_ratio])

    im2 = ax2.pcolormesh(phi_deg, mu, ratio, cmap='RdYlBu_r', shading='auto')
    draw_isosceles_flat(ax2, mu, phi)
    ax2.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax2.set_ylabel(r'$\mu$', fontsize=13)
    ax2.set_title(r'$\log_{10}(\sigma_{115} / \sigma_{116})$ — Near-degenerate map',
                  fontsize=14)
    fig.colorbar(im2, ax=ax2, pad=0.02)

    fig.suptitle('Singular Value Structure — 1/r (Newton), 800×1000 Grid\n'
                 'Left: strength of marginal generator | Right: gap sharpness',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'atlas_1000_sv116.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("    Saved atlas_1000_sv116.png")

    # --- Figure 4: S3 symmetry overlay with gap ratio ---
    print("  Fig 4: S3 symmetry overlay...", flush=True)
    fig = plt.figure(figsize=(20, 9), facecolor='white')

    ax1 = fig.add_subplot(121)
    ax1.pcolormesh(phi_deg, mu, log_gap, cmap='inferno', shading='auto')
    draw_isosceles_flat(ax1, mu, phi)
    ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax1.set_ylabel(r'$\mu$', fontsize=13)
    ax1.set_title(r'Gap ratio with $S_3$ isosceles curves', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.8,
               edgecolor='white', facecolor='#333333', labelcolor='white')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(S1, S2, S3, facecolors=FC,
                     rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.9)
    ax2.plot_surface(S1, S2, -S3, facecolors=FC_m,
                     rstride=1, cstride=1, shade=False, antialiased=True)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
             'k-', alpha=0.25, linewidth=0.8)
    draw_isosceles_3d(ax2)
    ax2.set_xlabel('$s_1$'); ax2.set_ylabel('$s_2$'); ax2.set_zlabel('$s_3$')
    ax2.set_title(r'Shape sphere with $S_3$ symmetry walls', fontsize=14)
    ax2.view_init(elev=25, azim=-50)
    sm = cm.ScalarMappable(norm=norm_g, cmap=cmap_g); sm.set_array([])
    fig.colorbar(sm, ax=ax2, shrink=0.55, pad=0.08)

    fig.suptitle(r'$S_3$ Symmetry in the Poisson Algebra — 1/r (Newton), 800×1000'
                 '\nThree isosceles curves (permutation symmetry walls) meet at Lagrange',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, 'atlas_1000_s3_symmetry.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("    Saved atlas_1000_s3_symmetry.png")

    # --- Figure 5: Rank drops and anomalies detail ---
    print("  Fig 5: Rank drops & anomalies...", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor='white')

    ax = axes[0]
    drop_mask = (rank == 115).astype(float)
    drop_mask[rank == -1] = np.nan
    im = ax.pcolormesh(phi_deg, mu, drop_mask, cmap='Reds', shading='auto',
                       vmin=0, vmax=1)
    draw_isosceles_flat(ax, mu, phi)
    ax.set_xlabel(r'$\phi$ (deg)'); ax.set_ylabel(r'$\mu$')
    ax.set_title(f'Rank = 115 (drops)\n{(rank==115).sum():,} points / {rank.size:,}',
                 fontsize=13)

    ax = axes[1]
    excess = (rank > 116).astype(float)
    excess[(rank <= 116) & (rank >= 0)] = 0
    excess[rank == -1] = np.nan
    im = ax.pcolormesh(phi_deg, mu, excess, cmap='Blues', shading='auto',
                       vmin=0, vmax=1)
    draw_isosceles_flat(ax, mu, phi)
    ax.set_xlabel(r'$\phi$ (deg)'); ax.set_ylabel(r'$\mu$')
    ax.set_title(f'Rank > 116 (excess)\n{(rank>116).sum():,} points / {rank.size:,}',
                 fontsize=13)

    ax = axes[2]
    cat = np.zeros_like(rank, dtype=float)
    cat[rank == 116] = 0
    cat[rank == 115] = 1
    cat[rank > 116] = 2
    cat[rank == -1] = 3
    cat[(rank > 0) & (rank < 115)] = 1

    cmap_cat = colors.ListedColormap(['#2d3436', '#e17055', '#0984e3', '#d63031'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_cat = colors.BoundaryNorm(bounds, cmap_cat.N)
    im = ax.pcolormesh(phi_deg, mu, cat, cmap=cmap_cat, norm=norm_cat, shading='auto')
    draw_isosceles_flat(ax, mu, phi)
    ax.set_xlabel(r'$\phi$ (deg)'); ax.set_ylabel(r'$\mu$')
    ax.set_title('Combined classification', fontsize=13)
    cb = fig.colorbar(im, ax=ax, pad=0.02, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels([f'116 ({(rank==116).sum():,})',
                           f'<116 ({((rank>0)&(rank<116)).sum():,})',
                           f'>116 ({(rank>116).sum():,})',
                           f'timeout ({(rank==-1).sum()})'],
                          fontsize=9)

    fig.suptitle('Rank Structure Detail — 1/r (Newton), 800×1000\n'
                 'Drops concentrate at high mu (unequal masses), '
                 'excess near mu~1 (equal masses)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'atlas_1000_rank_detail.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("    Saved atlas_1000_rank_detail.png")

    print("  Static images complete.\n")
    return D


# ===================================================================
# Interactive Plotly HTML
# ===================================================================

def render_interactive(D=None):
    import plotly.graph_objects as go

    print("  Building interactive HTML...", flush=True)
    if D is None:
        D = load_clean()

    mu, phi = D['mu'], D['phi']
    rank, gap, sv = D['rank'], D['gap'], D['sv']

    step_mu = max(1, len(mu) // 150)
    step_phi = max(1, len(phi) // 150)
    mu_s = mu[::step_mu]
    phi_s = phi[::step_phi]
    rank_s = rank[::step_mu, ::step_phi].astype(float)
    gap_s = gap[::step_mu, ::step_phi]
    sv_s = sv[::step_mu, ::step_phi]

    MU_s, PHI_s = np.meshgrid(mu_s, phi_s, indexing='ij')
    S1, S2, S3 = mu_phi_to_shape_sphere(MU_s, PHI_s)

    log_gap = np.log10(np.clip(gap_s, 1.0, None))
    log_gap[rank_s == -1] = np.nan

    hover = np.empty(MU_s.shape, dtype=object)
    for i in range(MU_s.shape[0]):
        for j in range(MU_s.shape[1]):
            hover[i, j] = (f"mu={MU_s[i,j]:.3f}, phi={PHI_s[i,j]*180/np.pi:.1f}deg<br>"
                           f"rank={int(rank_s[i,j])}, gap={gap_s[i,j]:.2e}")

    fig = go.Figure()

    # Gap ratio surface
    fig.add_trace(go.Surface(
        x=S1, y=S2, z=S3, surfacecolor=log_gap,
        colorscale='Inferno', customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="log10(gap)", x=1.02, len=0.5, y=0.75),
        visible=True, name='Gap ratio'))

    # Rank surface
    fig.add_trace(go.Surface(
        x=S1, y=S2, z=S3, surfacecolor=rank_s,
        colorscale='Viridis', customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="rank", x=1.02, len=0.5, y=0.75),
        visible=False, name='Rank'))

    # SV116 normalized
    sv116_n = np.full_like(gap_s, np.nan)
    m = (rank_s == 116) & (sv_s[:, :, 0] > 0) & (sv_s[:, :, 115] > 0)
    sv116_n[m] = np.log10(sv_s[:, :, 115][m] / sv_s[:, :, 0][m])

    fig.add_trace(go.Surface(
        x=S1, y=S2, z=S3, surfacecolor=sv116_n,
        colorscale='Viridis', customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="log10(sv116/sv1)", x=1.02, len=0.5, y=0.75),
        visible=False, name='SV116'))

    # SV115/SV116 ratio
    sv_ratio = np.full_like(gap_s, np.nan)
    m2 = m & (sv_s[:, :, 115] > 1e-30)
    sv_ratio[m2] = np.log10(sv_s[:, :, 114][m2] / sv_s[:, :, 115][m2])

    fig.add_trace(go.Surface(
        x=S1, y=S2, z=S3, surfacecolor=sv_ratio,
        colorscale='RdYlBu_r', customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="log10(sv115/sv116)", x=1.02, len=0.5, y=0.75),
        visible=False, name='SV ratio'))

    # Mirror hemisphere
    fig.add_trace(go.Surface(
        x=S1, y=S2, z=-S3, surfacecolor=log_gap,
        colorscale='Inferno', opacity=0.35, showscale=False,
        hoverinfo='skip', visible=True, name='mirror'))

    n_main = 4

    # Isosceles curves
    for curve_mu, curve_phi, cname, ccolor in _iso_curves():
        s1c, s2c, s3c = mu_phi_to_shape_sphere(curve_mu, curve_phi)
        fig.add_trace(go.Scatter3d(
            x=s1c, y=s2c, z=s3c, mode='lines',
            line=dict(color=ccolor, width=4),
            name=cname, visible=True, showlegend=True))

    # Lagrange point
    s1l, s2l, s3l = mu_phi_to_shape_sphere(1.0, np.pi / 3)
    fig.add_trace(go.Scatter3d(
        x=[s1l], y=[s2l], z=[s3l], mode='markers+text',
        marker=dict(size=8, color='white', symbol='diamond',
                    line=dict(width=2, color='black')),
        text=['Lagrange'], textposition='top center',
        name='Lagrange', visible=True, showlegend=False))

    buttons = []
    labels = ['Gap ratio', 'Rank', 'SV116', 'SV115/SV116 ratio']
    for idx, label in enumerate(labels):
        vis = [False] * len(fig.data)
        vis[idx] = True
        vis[n_main] = True  # mirror
        for k in range(n_main + 1, len(fig.data)):
            vis[k] = True
        buttons.append(dict(label=label, method='update',
                            args=[{'visible': vis}]))

    fig.update_layout(
        title=dict(text='1000×1000 Atlas — 1/r Potential (800 clean rows)',
                   font=dict(size=18)),
        scene=dict(
            xaxis_title='s₁', yaxis_title='s₂', zaxis_title='s₃',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.0, z=0.8))),
        updatemenus=[dict(
            type='buttons', direction='right',
            x=0.5, y=1.08, xanchor='center',
            buttons=buttons, showactive=True,
            font=dict(size=13))],
        width=1100, height=800,
        margin=dict(l=0, r=100, t=100, b=0))

    out_path = os.path.join(OUT_DIR, 'atlas_1000_interactive.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"    Saved {out_path}")


def _iso_curves():
    phi1 = np.linspace(0.1, np.pi - 0.1, 500)
    mu1 = np.ones_like(phi1)

    phi2 = np.linspace(0.05, np.pi / 2 - 0.05, 500)
    mu2 = 2 * np.cos(phi2)
    v2 = (mu2 >= 0.2) & (mu2 <= 5.0)

    phi3 = np.linspace(0.05, np.pi / 2 - 0.05, 500)
    mu3 = 1.0 / (2 * np.cos(phi3))
    v3 = (mu3 >= 0.2) & (mu3 <= 5.0)

    curves = [
        (mu1, phi1, 'r13=r12 (mu=1)', 'cyan'),
        (mu2[v2], phi2[v2], 'r23=r12', 'lime'),
    ]
    if v3.any():
        curves.append((mu3[v3], phi3[v3], 'r13=r23', 'magenta'))
    return curves


# ===================================================================
# Animation: rotating shape sphere
# ===================================================================

def render_animation(D=None):
    print("  Building rotation animation...", flush=True)
    if D is None:
        D = load_clean()

    mu, phi = D['mu'], D['phi']
    rank, gap = D['rank'], D['gap']

    step_mu = max(1, len(mu) // 150)
    step_phi = max(1, len(phi) // 150)

    log_gap = np.log10(np.clip(gap[::step_mu, ::step_phi], 1.0, None))
    rank_s = rank[::step_mu, ::step_phi]
    log_gap[rank_s == -1] = np.nan

    mu_s = mu[::step_mu]
    phi_s = phi[::step_phi]
    MU_s, PHI_s = np.meshgrid(mu_s, phi_s, indexing='ij')
    S1, S2, S3 = mu_phi_to_shape_sphere(MU_s, PHI_s)

    valid = np.isfinite(log_gap)
    vmin = np.nanpercentile(log_gap[valid], 1)
    vmax = np.nanpercentile(log_gap[valid], 99)
    norm_obj = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap('inferno')
    FC = cmap_obj(norm_obj(np.clip(np.where(valid, log_gap, vmin), vmin, vmax)))
    FC_m = FC.copy(); FC_m[:, :, 3] = 0.35

    n_frames = 120
    azimuths = np.linspace(-60, 300, n_frames)

    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor('black')
        ax.plot_surface(S1, S2, S3, facecolors=FC,
                        rstride=1, cstride=1, shade=False,
                        antialiased=True, alpha=0.95)
        ax.plot_surface(S1, S2, -S3, facecolors=FC_m,
                        rstride=1, cstride=1, shade=False, antialiased=True)
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
                color='gray', alpha=0.3, linewidth=0.8)
        draw_isosceles_3d(ax)

        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
        ax.set_xlabel('$s_1$', color='white', fontsize=10)
        ax.set_ylabel('$s_2$', color='white', fontsize=10)
        ax.set_zlabel('$s_3$', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=7)

        elev = 25 + 10 * np.sin(2 * np.pi * frame_idx / n_frames)
        ax.view_init(elev=elev, azim=azimuths[frame_idx])

        ax.set_title('Poisson Algebra Gap Ratio — Shape Sphere\n'
                     '1/r Potential, 800×1000 Grid',
                     color='white', fontsize=14, fontweight='bold', pad=10)
        return []

    print(f"    Rendering {n_frames} frames...", flush=True)
    anim = FuncAnimation(fig, draw_frame, frames=n_frames, blit=False)

    gif_path = os.path.join(OUT_DIR, 'atlas_1000_rotation.gif')
    anim.save(gif_path, writer=PillowWriter(fps=20), dpi=100)
    plt.close()
    print(f"    Saved {gif_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print("=" * 60)
    print("  1000x1000 Atlas Visualization Suite")
    print("=" * 60)

    D = None
    if mode in ('all', 'static'):
        D = render_static()
    if mode in ('all', 'interactive'):
        render_interactive(D)
    if mode in ('all', 'animation'):
        render_animation(D)

    print(f"\n  All outputs in {OUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
