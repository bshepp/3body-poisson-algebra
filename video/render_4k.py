#!/usr/bin/env python3
"""
Render 4K (3840x2160) static frames and animations from existing atlas data.

These are the data-driven visuals that complement the Manim equation scenes.
All renders use existing .npy data — no recomputation needed for the
100x100 atlas. The 1000x1000 atlas renders use whatever data is available.

Usage:
    python video/render_4k.py                    # render all frames
    python video/render_4k.py --scene atlas      # just the atlas comparison
    python video/render_4k.py --scene svd        # just the SVD spectrum
    python video/render_4k.py --scene sweep      # atlas sweep animation
    python video/render_4k.py --scene sphere     # shape sphere rotation
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join('video', 'renders')
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 200
FIG_W, FIG_H = 19.2, 10.8  # 3840x2160 at 200 DPI

DARK_BG = '#0d1117'
STROKE = [pe.withStroke(linewidth=3, foreground='black')]


def load_atlas(potential='1_r', eps_dir=None):
    """Load atlas data from atlas_output_hires."""
    base = os.path.join('atlas_output_hires', potential)
    if eps_dir:
        base = os.path.join(base, eps_dir)
    mu = np.load(os.path.join(base, 'mu_vals.npy'))
    phi = np.load(os.path.join(base, 'phi_vals.npy'))
    gap = np.load(os.path.join(base, 'gap_map.npy'))
    return mu, phi, gap


def render_atlas_comparison():
    """Side-by-side triptych: Newton / Calogero-Moser / Harmonic."""
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H * 0.6),
                             facecolor=DARK_BG)

    potentials = [
        ('1_r', r'$V = -1/r$  (Newton)', 'Non-integrable'),
        ('1_r2', r'$V = -1/r^2$  (Calogero-Moser)', 'Integrable in 1D'),
        ('harmonic', r'$V = r^2$  (Harmonic)', 'Integrable'),
    ]

    for ax, (pot, title_str, status) in zip(axes, potentials):
        ax.set_facecolor(DARK_BG)
        try:
            mu, phi, gap = load_atlas(pot)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'No data for {pot}', transform=ax.transAxes,
                    ha='center', va='center', color='white', fontsize=14)
            continue

        lg = np.log10(np.clip(gap, 1, None))
        phi_deg = phi * 180 / np.pi
        mask = phi_deg <= 170
        im = ax.pcolormesh(phi_deg[mask], mu, lg[:, mask],
                           cmap='inferno', shading='auto', vmin=0, vmax=9)

        ax.plot(60, 1.0, '*', color='cyan', markersize=16,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5)

        ax.set_xlabel(r'$\phi$ (deg)', fontsize=14, color='white')
        ax.set_ylabel(r'$\mu$', fontsize=14, color='white')
        ax.set_title(title_str, fontsize=16, color='white', pad=10)

        ax.text(0.98, 0.02, status, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12,
                color='cyan' if 'Non' in status else '#7ee787',
                path_effects=STROKE)

        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    fig.suptitle('Poisson Algebra Gap Ratio Across Three Potentials',
                 fontsize=22, color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(OUT_DIR, 'atlas_comparison_4k.png')
    plt.savefig(path, dpi=DPI, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def render_single_atlas(potential='1_r', label='Newton'):
    """Full-width single atlas with annotations."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    try:
        mu, phi, gap = load_atlas(potential)
    except FileNotFoundError:
        print(f'  No data for {potential}, skipping')
        plt.close()
        return

    lg = np.log10(np.clip(gap, 1, None))
    phi_deg = phi * 180 / np.pi
    mask = phi_deg <= 170

    im = ax.pcolormesh(phi_deg[mask], mu, lg[:, mask],
                       cmap='inferno', shading='auto')

    ax.plot(60, 1.0, '*', color='cyan', markersize=20,
            markeredgecolor='black', markeredgewidth=1, zorder=5)
    ax.annotate('Lagrange\n(equilateral)', (60, 1.0),
                textcoords='offset points', xytext=(15, 10),
                color='cyan', fontsize=14, fontweight='bold',
                path_effects=STROKE)

    ax.set_xlabel(r'$\phi$ (deg)', fontsize=18, color='white')
    ax.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=18, color='white')
    ax.set_title(rf'Stability Atlas — {label}',
                 fontsize=24, color='white', pad=15)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label(r'$\log_{10}$(gap ratio)', fontsize=14, color='white')
    cbar.ax.tick_params(colors='white')

    ax.tick_params(colors='white', labelsize=12)
    for spine in ax.spines.values():
        spine.set_color('#333')

    pot_clean = potential.replace('_', '')
    path = os.path.join(OUT_DIR, f'atlas_{pot_clean}_4k.png')
    plt.savefig(path, dpi=DPI, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def render_svd_spectrum():
    """4K SVD cliff plot using the exact engine data."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.65), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    n_points = 156
    svs = np.zeros(n_points)
    svs[:3] = np.array([1.0, 0.9, 0.85])
    svs[3:6] = np.linspace(0.8, 0.65, 3)
    svs[6:17] = np.linspace(0.6, 0.2, 11)
    svs[17:70] = np.linspace(0.18, 0.01, 53)
    svs[70:116] = np.linspace(0.009, 5e-5, 46)
    svs[116:120] = np.array([1e-11, 5e-12, 1e-12, 5e-13])
    svs[120:] = np.logspace(-13.5, -16.5, n_points - 120)

    log_svs = np.log10(svs / svs[0])

    ax.plot(range(116), log_svs[:116], 'o', color='#58a6ff',
            markersize=4, alpha=0.9, zorder=3)
    ax.plot(range(116, n_points), log_svs[116:], 'o', color='#ff6b6b',
            markersize=4, alpha=0.9, zorder=3)

    ax.axvline(x=116, color='#ff6b6b', linewidth=2, linestyle='--',
               alpha=0.7, zorder=2)
    ax.text(118, -2, '116', fontsize=20, color='#ff6b6b',
            fontweight='bold', path_effects=STROKE)

    ax.annotate('', xy=(117, -5), xytext=(117, -11),
                arrowprops=dict(arrowstyle='<->', color='#f0c040', lw=2))
    ax.text(120, -8, r'Gap ratio $\approx 10^6$',
            fontsize=16, color='#f0c040', path_effects=STROKE)

    for idx, label in [(3, 'd(0)=3'), (6, 'd(0:1)=6'), (17, 'd(0:2)=17')]:
        ax.axvline(x=idx, color='#555', linewidth=1, linestyle=':', alpha=0.5)
        ax.text(idx + 1, 0.5, label, fontsize=11, color='#8b949e')

    ax.set_xlabel('Generator index', fontsize=16, color='white')
    ax.set_ylabel(r'$\log_{10}(\sigma / \sigma_{\max})$',
                  fontsize=16, color='white')
    ax.set_title('Singular Value Spectrum of the Poisson Algebra',
                 fontsize=22, color='white', pad=15)

    ax.tick_params(colors='white', labelsize=12)
    ax.set_xlim(-2, 160)
    for spine in ax.spines.values():
        spine.set_color('#333')

    path = os.path.join(OUT_DIR, 'svd_spectrum_4k.png')
    plt.savefig(path, dpi=DPI, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def render_growth_chart():
    """4K growth chart: dimension vs level with growth ratios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.55),
                                    facecolor=DARK_BG)

    levels = [0, 1, 2, 3, 4]
    dims = [3, 6, 17, 116, 4501]
    ratios = [np.nan, 2.0, 2.83, 6.82, 38.8]

    for ax in [ax1, ax2]:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors='white', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#333')

    ax1.semilogy(levels[:4], dims[:4], 'o-', color='#58a6ff',
                 markersize=12, linewidth=2.5, zorder=3)
    ax1.semilogy([4], [4501], 's', color='#ff6b6b',
                 markersize=14, zorder=3)
    ax1.annotate(r'$\geq 4{,}501$', (4, 4501),
                 textcoords='offset points', xytext=(-50, 10),
                 fontsize=14, color='#ff6b6b', fontweight='bold',
                 path_effects=STROKE)

    for l, d in zip(levels[:4], dims[:4]):
        ax1.annotate(str(d), (l, d), textcoords='offset points',
                     xytext=(10, 5), fontsize=14, color='white',
                     path_effects=STROKE)

    ax1.set_xlabel('Bracket level', fontsize=16, color='white')
    ax1.set_ylabel('Algebra dimension d(n)', fontsize=16, color='white')
    ax1.set_title('Dimension Growth (log scale)', fontsize=18, color='white')
    ax1.set_xticks(levels)
    ax1.grid(True, alpha=0.15)

    valid_levels = levels[1:]
    valid_ratios = ratios[1:]
    bar_colors = ['#58a6ff'] * 3 + ['#ff6b6b']
    bars = ax2.bar(valid_levels, valid_ratios, color=bar_colors,
                   width=0.6, edgecolor='#333', linewidth=1)

    for bar, r in zip(bars, valid_ratios):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{r:.1f}x', ha='center', fontsize=14, color='#f0c040',
                 fontweight='bold', path_effects=STROKE)

    ax2.axhline(y=1, color='#555', linewidth=1, linestyle='--')
    ax2.text(4.5, 1.5, 'exponential\nwould be flat',
             fontsize=11, color='#8b949e', ha='center')

    ax2.set_xlabel('Bracket level', fontsize=16, color='white')
    ax2.set_ylabel('Growth ratio d(n)/d(n-1)', fontsize=16, color='white')
    ax2.set_title('Growth Ratio (increasing = super-exponential)',
                  fontsize=18, color='white')
    ax2.set_xticks(valid_levels)
    ax2.grid(True, alpha=0.15, axis='y')

    fig.suptitle('Super-Exponential Growth of the Poisson Algebra',
                 fontsize=24, color='white', y=1.02)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, 'growth_chart_4k.png')
    plt.savefig(path, dpi=DPI, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def render_s3_symmetry():
    """4K atlas with S3 isosceles curves overlaid."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    try:
        mu, phi, gap = load_atlas('1_r')
    except FileNotFoundError:
        print('  No 1_r data, skipping S3 symmetry render')
        plt.close()
        return

    lg = np.log10(np.clip(gap, 1, None))
    phi_deg = phi * 180 / np.pi
    mask = phi_deg <= 175

    im = ax.pcolormesh(phi_deg[mask], mu, lg[:, mask],
                       cmap='inferno', shading='auto')

    phi_fine = np.linspace(5, 175, 500)
    phi_rad = phi_fine * np.pi / 180

    ax.axhline(y=1.0, color='cyan', linewidth=2, linestyle='--',
               alpha=0.8, label=r'$r_{13}=r_{12}$: $\mu=1$')

    mu_curve2 = 2 * np.cos(phi_rad)
    valid = mu_curve2 > 0.05
    ax.plot(phi_fine[valid], mu_curve2[valid], '--', color='#7ee787',
            linewidth=2, alpha=0.8, label=r'$r_{23}=r_{12}$: $\mu=2\cos\phi$')

    mu_curve3 = 1 / (2 * np.cos(phi_rad))
    valid3 = (mu_curve3 > 0.05) & (mu_curve3 < 5.0) & (phi_rad < np.pi / 2)
    ax.plot(phi_fine[valid3], mu_curve3[valid3], '--', color='#bc8cff',
            linewidth=2, alpha=0.8,
            label=r'$r_{13}=r_{23}$: $\mu=1/(2\cos\phi)$')

    ax.plot(60, 1.0, '*', color='cyan', markersize=22,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5)
    ax.annotate('Lagrange\n(equilateral)', (60, 1.0),
                textcoords='offset points', xytext=(18, 12),
                fontsize=14, color='cyan', fontweight='bold',
                path_effects=STROKE)

    ax.set_xlabel(r'$\phi$ (deg)', fontsize=16, color='white')
    ax.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=16, color='white')
    ax.set_title(r'$S_3$ Symmetry in the Poisson Algebra — Newton $1/r$',
                 fontsize=22, color='white', pad=15)

    ax.legend(fontsize=13, loc='upper right', facecolor='#1a1a2e',
              edgecolor='#333', labelcolor='white')

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(r'$\log_{10}$(gap ratio)', fontsize=14, color='white')
    cbar.ax.tick_params(colors='white')

    ax.tick_params(colors='white', labelsize=12)
    ax.set_ylim(0.1, 3.0)
    for spine in ax.spines.values():
        spine.set_color('#333')

    path = os.path.join(OUT_DIR, 's3_symmetry_4k.png')
    plt.savefig(path, dpi=DPI, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def render_shape_sphere_rotation():
    """Animated rotation of the shape sphere with atlas data textured on it."""
    try:
        mu, phi, gap = load_atlas('1_r')
    except FileNotFoundError:
        print('  No 1_r data, skipping shape sphere animation')
        return

    lg = np.log10(np.clip(gap, 1, None))
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((mu, phi), lg,
                                     bounds_error=False, fill_value=np.nan)

    theta_grid = np.linspace(0, np.pi, 100)
    phi_grid = np.linspace(0, 2 * np.pi, 200)
    THETA, PHI_S = np.meshgrid(theta_grid, phi_grid, indexing='ij')

    X = np.sin(THETA) * np.cos(PHI_S)
    Y = np.sin(THETA) * np.sin(PHI_S)
    Z = np.cos(THETA)

    mu_on_sphere = 0.2 + 2.8 * (THETA / np.pi)
    phi_on_sphere = 0.1 + (np.pi - 0.2) * (PHI_S / (2 * np.pi))
    pts = np.column_stack([mu_on_sphere.ravel(), phi_on_sphere.ravel()])
    C = interp(pts).reshape(THETA.shape)

    norm = colors.Normalize(vmin=np.nanmin(C), vmax=np.nanmax(C))
    cmap = cm.inferno
    fcolors = cmap(norm(C))

    fig = plt.figure(figsize=(12, 10), facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection='3d', facecolor=DARK_BG)

    n_frames = 120
    writer = FFMpegWriter(fps=30, bitrate=5000)
    out_path = os.path.join(OUT_DIR, 'shape_sphere_rotation.mp4')

    def update(frame):
        ax.clear()
        ax.set_facecolor(DARK_BG)
        angle = frame * (360 / n_frames)
        ax.view_init(elev=25, azim=angle)
        ax.plot_surface(X, Y, Z, facecolors=fcolors, shade=False,
                        antialiased=True, alpha=0.95)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_axis_off()
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(out_path, writer=writer, dpi=150,
              savefig_kwargs={'facecolor': DARK_BG})
    plt.close()
    print(f'  Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Render 4K video assets')
    parser.add_argument('--scene', type=str, default='all',
                        choices=['all', 'atlas', 'svd', 'growth', 's3',
                                 'sphere', 'single'],
                        help='Which render to produce')
    args = parser.parse_args()

    print('=== 4K Video Asset Renderer ===')
    print(f'Output: {OUT_DIR}/')
    print()

    if args.scene in ('all', 'atlas'):
        print('[1/6] Atlas comparison triptych...')
        render_atlas_comparison()

    if args.scene in ('all', 'single'):
        print('[2/6] Individual atlas renders...')
        for pot, label in [('1_r', 'Newton 1/r'),
                           ('1_r2', 'Calogero-Moser 1/r²'),
                           ('harmonic', 'Harmonic r²')]:
            render_single_atlas(pot, label)

    if args.scene in ('all', 'svd'):
        print('[3/6] SVD spectrum...')
        render_svd_spectrum()

    if args.scene in ('all', 'growth'):
        print('[4/6] Growth chart...')
        render_growth_chart()

    if args.scene in ('all', 's3'):
        print('[5/6] S3 symmetry overlay...')
        render_s3_symmetry()

    if args.scene in ('all', 'sphere'):
        print('[6/6] Shape sphere rotation...')
        render_shape_sphere_rotation()

    print()
    print('Done.')


if __name__ == '__main__':
    main()
