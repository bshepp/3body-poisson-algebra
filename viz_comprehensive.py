#!/usr/bin/env python3
"""
Comprehensive Visualization Suite
====================================

Renders all appropriate visuals from available data:
  A. 100x100 full-sphere rotating animations (all 3 potentials)
  B. Multi-epsilon cross-potential comparison + epsilon-rotation animation
  C. Level 4 comparison dashboard
  D. (1000x1000 half-sphere images handled separately)

Usage:
    python viz_comprehensive.py              # everything
    python viz_comprehensive.py rotate       # Group A only
    python viz_comprehensive.py multieps     # Group B only
    python viz_comprehensive.py level4       # Group C only
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

HIRES_DIR = 'atlas_output_hires'
RESULTS_DIR = 'results'
OUT_A = os.path.join(HIRES_DIR, 'animations')
OUT_B = os.path.join(HIRES_DIR, 'multi_epsilon')
OUT_C = os.path.join(RESULTS_DIR, 'dashboards')

STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]

POT_DIRS = {'1/r': '1_r', '1/r2': '1_r2', 'harmonic': 'harmonic'}
POT_LABELS = {
    '1/r':      '1/r (Newton)',
    '1/r2':     r'1/r$^2$ (Calogero-Moser)',
    'harmonic': r'r$^2$ (Harmonic)',
}


def mu_phi_to_sphere(mu, phi):
    w2_sq = mu**2 - mu * np.cos(phi) + 0.25
    N = 1.0 + w2_sq
    return (1.0 - w2_sq) / N, 2.0 * (mu * np.cos(phi) - 0.5) / N, 2.0 * mu * np.sin(phi) / N


def load_atlas(pot_dir, eps_subdir=None):
    d = os.path.join(HIRES_DIR, pot_dir)
    if eps_subdir:
        d = os.path.join(d, eps_subdir)
    return {
        'mu':   np.load(os.path.join(d, 'mu_vals.npy')),
        'phi':  np.load(os.path.join(d, 'phi_vals.npy')),
        'rank': np.load(os.path.join(d, 'rank_map.npy')),
        'gap':  np.load(os.path.join(d, 'gap_map.npy')),
        'sv':   np.load(os.path.join(d, 'sv_spectra.npy')),
    }


def draw_iso_3d(ax, mu_range=(0.05, 5.0)):
    phi_c1 = np.linspace(0.1, np.pi - 0.1, 300)
    s1, s2, s3 = mu_phi_to_sphere(1.0, phi_c1)
    ax.plot(s1, s2, s3, color='cyan', linewidth=2, alpha=0.8, zorder=5)

    phi_c2 = np.linspace(0.05, np.pi / 2 - 0.05, 300)
    mu_c2 = 2 * np.cos(phi_c2)
    v = (mu_c2 >= mu_range[0]) & (mu_c2 <= mu_range[1])
    if v.any():
        s1, s2, s3 = mu_phi_to_sphere(mu_c2[v], phi_c2[v])
        ax.plot(s1, s2, s3, color='lime', linewidth=2, alpha=0.8, zorder=5)

    phi_c3 = np.linspace(0.05, np.pi / 2 - 0.05, 300)
    mu_c3 = 1.0 / (2 * np.cos(phi_c3))
    v = (mu_c3 >= mu_range[0]) & (mu_c3 <= mu_range[1])
    if v.any():
        s1, s2, s3 = mu_phi_to_sphere(mu_c3[v], phi_c3[v])
        ax.plot(s1, s2, s3, color='magenta', linewidth=2, alpha=0.8, zorder=5)

    s1l, s2l, s3l = mu_phi_to_sphere(1.0, np.pi / 3)
    ax.scatter([s1l], [s2l], [s3l], color='white', s=60, zorder=15,
               edgecolors='black', linewidth=1, depthshade=False, marker='*')


def prepare_sphere(D, cmap_name='inferno', value='gap'):
    mu, phi = D['mu'], D['phi']
    MU, PHI = np.meshgrid(mu, phi, indexing='ij')
    S1, S2, S3 = mu_phi_to_sphere(MU, PHI)

    if value == 'gap':
        data = np.log10(np.clip(D['gap'], 1.0, None))
    elif value == 'rank':
        data = D['rank'].astype(float)
    else:
        data = value

    valid = np.isfinite(data)
    vmin = np.nanpercentile(data[valid], 1) if valid.any() else 0
    vmax = np.nanpercentile(data[valid], 99) if valid.any() else 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap_name)
    FC = cmap_obj(norm(np.clip(np.where(valid, data, vmin), vmin, vmax)))
    FC_m = FC.copy()
    FC_m[:, :, 3] = 0.35
    return S1, S2, S3, FC, FC_m, norm, cmap_obj


def draw_sphere(ax, S1, S2, S3, FC, FC_m, elev=25, azim=-50):
    ax.plot_surface(S1, S2, S3, facecolors=FC,
                    rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.95)
    ax.plot_surface(S1, S2, -S3, facecolors=FC_m,
                    rstride=1, cstride=1, shade=False, antialiased=True)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
            color='gray', alpha=0.3, linewidth=0.8)
    draw_iso_3d(ax)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('$s_1$', fontsize=9, labelpad=-4)
    ax.set_ylabel('$s_2$', fontsize=9, labelpad=-4)
    ax.set_zlabel('$s_3$', fontsize=9, labelpad=-4)
    ax.tick_params(labelsize=6)


# ===================================================================
# GROUP A: 100x100 full-sphere rotating animations
# ===================================================================

def render_per_potential_rotations():
    """Individual rotating GIF for each potential."""
    os.makedirs(OUT_A, exist_ok=True)

    for pot_key, pot_dir in POT_DIRS.items():
        path = os.path.join(HIRES_DIR, pot_dir, 'rank_map.npy')
        if not os.path.exists(path):
            print(f"  Skipping {pot_dir} -- no data")
            continue

        print(f"  Rendering {pot_dir} rotation...", flush=True)
        D = load_atlas(pot_dir)
        S1, S2, S3, FC, FC_m, norm, cmap_obj = prepare_sphere(D)

        n_frames = 90
        azimuths = np.linspace(-60, 300, n_frames)

        fig = plt.figure(figsize=(8, 8), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        def draw_frame(idx):
            ax.clear()
            ax.set_facecolor('black')
            draw_sphere(ax, S1, S2, S3, FC, FC_m,
                        elev=25 + 8 * np.sin(2 * np.pi * idx / n_frames),
                        azim=azimuths[idx])
            ax.set_title(f'{POT_LABELS[pot_key]}\nGap Ratio on Shape Sphere',
                         color='white', fontsize=13, fontweight='bold', pad=8)
            ax.tick_params(colors='white', labelsize=6)
            ax.set_xlabel('$s_1$', color='white', fontsize=9)
            ax.set_ylabel('$s_2$', color='white', fontsize=9)
            ax.set_zlabel('$s_3$', color='white', fontsize=9)
            return []

        anim = FuncAnimation(fig, draw_frame, frames=n_frames, blit=False)
        out_path = os.path.join(OUT_A, f'rotation_{pot_dir}.gif')
        anim.save(out_path, writer=PillowWriter(fps=18), dpi=100)
        plt.close()
        print(f"    Saved {out_path}")


def render_three_potential_rotation():
    """Side-by-side-by-side: all 3 potentials rotating in sync."""
    os.makedirs(OUT_A, exist_ok=True)
    print("  Rendering three-potential rotation...", flush=True)

    data = {}
    spheres = {}
    for pot_key, pot_dir in POT_DIRS.items():
        if not os.path.exists(os.path.join(HIRES_DIR, pot_dir, 'rank_map.npy')):
            print(f"    Missing {pot_dir}, skipping three-potential")
            return
        D = load_atlas(pot_dir)
        data[pot_key] = D
        spheres[pot_key] = prepare_sphere(D)

    n_frames = 90
    azimuths = np.linspace(-60, 300, n_frames)

    fig = plt.figure(figsize=(22, 7), facecolor='black')

    axes = {}
    for i, pot_key in enumerate(['1/r', '1/r2', 'harmonic']):
        axes[pot_key] = fig.add_subplot(1, 3, i + 1, projection='3d', facecolor='black')

    def draw_frame(idx):
        elev = 25 + 8 * np.sin(2 * np.pi * idx / n_frames)
        azim = azimuths[idx]
        for pot_key in ['1/r', '1/r2', 'harmonic']:
            ax = axes[pot_key]
            ax.clear()
            ax.set_facecolor('black')
            S1, S2, S3, FC, FC_m, _, _ = spheres[pot_key]
            draw_sphere(ax, S1, S2, S3, FC, FC_m, elev=elev, azim=azim)
            ax.set_title(POT_LABELS[pot_key], color='white', fontsize=12,
                         fontweight='bold', pad=6)
            ax.tick_params(colors='white', labelsize=5)
            ax.set_xlabel('$s_1$', color='white', fontsize=8)
            ax.set_ylabel('$s_2$', color='white', fontsize=8)
            ax.set_zlabel('$s_3$', color='white', fontsize=8)
        return []

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, blit=False)
    out_path = os.path.join(OUT_A, 'three_potential_rotation.gif')
    anim.save(out_path, writer=PillowWriter(fps=18), dpi=100)
    plt.close()
    print(f"    Saved {out_path}")


# ===================================================================
# GROUP B: Multi-epsilon visuals
# ===================================================================

def render_cross_potential_epsilon():
    """Side-by-side 1/r vs 1/r² at matching epsilons."""
    os.makedirs(OUT_B, exist_ok=True)
    print("  Rendering cross-potential epsilon comparison...", flush=True)

    eps_list = ['5e-3', '2e-3', '1e-3', '5e-4', '2e-4', '1e-4']
    eps_subdirs = [None, 'eps_2e-03', 'eps_1e-03', 'eps_5e-04', 'eps_2e-04', 'eps_1e-04']

    available = []
    for eps_label, eps_sub in zip(eps_list, eps_subdirs):
        path_1r = os.path.join(HIRES_DIR, '1_r', eps_sub or '', 'rank_map.npy')
        path_1r2 = os.path.join(HIRES_DIR, '1_r2', eps_sub or '', 'rank_map.npy')
        if eps_sub is None:
            path_1r = os.path.join(HIRES_DIR, '1_r', 'rank_map.npy')
            path_1r2 = os.path.join(HIRES_DIR, '1_r2', 'rank_map.npy')
        if os.path.exists(path_1r) and os.path.exists(path_1r2):
            available.append((eps_label, eps_sub))

    n = len(available)
    if n == 0:
        print("    No matching epsilon pairs found")
        return

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), facecolor='white')
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (eps_label, eps_sub) in enumerate(available):
        D1 = load_atlas('1_r', eps_sub)
        D2 = load_atlas('1_r2', eps_sub)

        lg1 = np.log10(np.clip(D1['gap'], 1.0, None))
        lg2 = np.log10(np.clip(D2['gap'], 1.0, None))
        phi_deg = D1['phi'] * 180 / np.pi

        vmin = min(lg1[lg1 > 0].min() if (lg1 > 0).any() else 0,
                   lg2[lg2 > 0].min() if (lg2 > 0).any() else 0)
        vmax = max(lg1.max(), lg2.max())

        ax1 = axes[0, col]
        ax1.pcolormesh(phi_deg, D1['mu'], lg1, cmap='inferno',
                       vmin=vmin, vmax=vmax, shading='auto')
        ax1.set_title(f'1/r, eps={eps_label}', fontsize=10)
        ax1.set_ylabel(r'$\mu$', fontsize=9) if col == 0 else None

        ax2 = axes[1, col]
        ax2.pcolormesh(phi_deg, D2['mu'], lg2, cmap='inferno',
                       vmin=vmin, vmax=vmax, shading='auto')
        ax2.set_title(f'1/r², eps={eps_label}', fontsize=10)
        ax2.set_xlabel(r'$\phi$ (deg)', fontsize=9)
        ax2.set_ylabel(r'$\mu$', fontsize=9) if col == 0 else None

    fig.suptitle('Cross-Potential Epsilon Comparison: 1/r vs 1/r²\n'
                 'Gap ratio landscape at matching epsilon values',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUT_B, 'cross_potential_epsilon.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved {out_path}")


def render_epsilon_rotation():
    """Sphere rotates while epsilon transitions through all values."""
    os.makedirs(OUT_B, exist_ok=True)
    print("  Rendering epsilon-rotation animation (1/r)...", flush=True)

    eps_order = [
        ('5e-3', None), ('2e-3', 'eps_2e-03'), ('1e-3', 'eps_1e-03'),
        ('5e-4', 'eps_5e-04'), ('2e-4', 'eps_2e-04'), ('1e-4', 'eps_1e-04'),
    ]

    available = []
    for label, sub in eps_order:
        p = os.path.join(HIRES_DIR, '1_r', sub or '', 'rank_map.npy')
        if sub is None:
            p = os.path.join(HIRES_DIR, '1_r', 'rank_map.npy')
        if os.path.exists(p):
            available.append((label, sub))

    if len(available) < 2:
        print("    Need at least 2 epsilon datasets")
        return

    sphere_data = []
    for label, sub in available:
        D = load_atlas('1_r', sub)
        S1, S2, S3, FC, FC_m, _, _ = prepare_sphere(D)
        sphere_data.append((label, S1, S2, S3, FC, FC_m))

    frames_per_eps = 20
    n_eps = len(sphere_data)
    n_frames = frames_per_eps * n_eps
    azimuths = np.linspace(-60, 300, n_frames)

    fig = plt.figure(figsize=(9, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    def draw_frame(idx):
        eps_idx = min(idx // frames_per_eps, n_eps - 1)
        label, S1, S2, S3, FC, FC_m = sphere_data[eps_idx]

        ax.clear()
        ax.set_facecolor('black')
        elev = 25 + 8 * np.sin(2 * np.pi * idx / n_frames)
        draw_sphere(ax, S1, S2, S3, FC, FC_m, elev=elev, azim=azimuths[idx])
        ax.set_title(f'1/r Gap Ratio — epsilon = {label}\n'
                     f'({eps_idx + 1}/{n_eps} epsilon values)',
                     color='white', fontsize=13, fontweight='bold', pad=8)
        ax.tick_params(colors='white', labelsize=6)
        ax.set_xlabel('$s_1$', color='white', fontsize=9)
        ax.set_ylabel('$s_2$', color='white', fontsize=9)
        ax.set_zlabel('$s_3$', color='white', fontsize=9)
        return []

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, blit=False)
    out_path = os.path.join(OUT_B, 'epsilon_rotation_1r.gif')
    anim.save(out_path, writer=PillowWriter(fps=15), dpi=100)
    plt.close()
    print(f"    Saved {out_path}")


# ===================================================================
# GROUP C: Level 4 dashboard
# ===================================================================

def render_level4_dashboard():
    os.makedirs(OUT_C, exist_ok=True)
    print("  Rendering Level 4 dashboard...", flush=True)

    configs = ['lagrange', 'euler', 'scalene', 'global']
    samples = [5000, 10000, 20000]
    config_labels = {
        'lagrange': 'Lagrange (equilateral)',
        'euler': 'Euler (collinear)',
        'scalene': 'Scalene (generic)',
        'global': 'Global (generic)',
    }

    all_results = {}
    for cfg in configs:
        all_results[cfg] = {}
        for n in samples:
            rdir = os.path.join(RESULTS_DIR, f'level4_{cfg}_{n}')
            rfile = os.path.join(rdir, 'results.json')
            if os.path.exists(rfile):
                with open(rfile) as f:
                    all_results[cfg][n] = json.load(f)

    # Also check for 30K global
    r30k = os.path.join(RESULTS_DIR, 'level4_global_30000', 'results.json')
    if os.path.exists(r30k):
        with open(r30k) as f:
            all_results['global'][30000] = json.load(f)

    fig = plt.figure(figsize=(20, 12), facecolor='white')

    # Top row: SVD spectra at 20K samples for each config
    for i, cfg in enumerate(configs):
        ax = fig.add_subplot(2, 4, i + 1)
        sdir = os.path.join(RESULTS_DIR, f'level4_{cfg}_20000')
        svd_path = os.path.join(sdir, 'svd_spectrum.npy')

        if os.path.exists(svd_path):
            sv = np.load(svd_path)
            sv_norm = sv / sv[0] if sv[0] > 0 else sv
            ax.semilogy(np.arange(1, len(sv_norm) + 1), sv_norm,
                        linewidth=1.5, color='steelblue')
            ax.axhline(1e-12, color='gray', linestyle=':', alpha=0.5)

            res = all_results[cfg].get(20000, {})
            n_gens = res.get('n_generators', len(sv))
            gap = res.get('max_gap_ratio', 0)
            ax.set_title(f'{config_labels[cfg]}\n'
                         f'n_gen={n_gens}, gap={gap:.1f}',
                         fontsize=11)
        else:
            ax.set_title(f'{config_labels[cfg]}\n(no 20K data)', fontsize=11)
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')

        ax.set_xlabel('SV index', fontsize=9)
        if i == 0:
            ax.set_ylabel(r'$\sigma_k / \sigma_1$', fontsize=10)
        ax.set_ylim(1e-16, 2)
        ax.grid(True, alpha=0.3)

    # Bottom left: Convergence plot -- d(4) lower bound vs samples
    ax_conv = fig.add_subplot(2, 2, 3)
    markers = {'lagrange': 'o', 'euler': 's', 'scalene': '^', 'global': 'D'}
    clrs = {'lagrange': '#e74c3c', 'euler': '#3498db', 'scalene': '#2ecc71', 'global': '#9b59b6'}

    for cfg in configs:
        sample_list = sorted(all_results[cfg].keys())
        if not sample_list:
            continue
        n_gens_list = [all_results[cfg][s].get('n_generators', 0) for s in sample_list]
        ax_conv.plot(sample_list, n_gens_list, f'-{markers[cfg]}',
                     color=clrs[cfg], linewidth=2, markersize=8,
                     label=config_labels[cfg])

    ax_conv.set_xlabel('Phase-space samples', fontsize=12)
    ax_conv.set_ylabel('d(4) lower bound (n_generators)', fontsize=12)
    ax_conv.set_title('Level 4 Convergence: d(4) vs Sample Count', fontsize=13,
                      fontweight='bold')
    ax_conv.legend(fontsize=10, loc='lower right')
    ax_conv.grid(True, alpha=0.3)

    # Bottom right: Gap ratios vs samples
    ax_gap = fig.add_subplot(2, 2, 4)
    for cfg in configs:
        sample_list = sorted(all_results[cfg].keys())
        if not sample_list:
            continue
        gaps = [all_results[cfg][s].get('max_gap_ratio', 0) for s in sample_list]
        ax_gap.semilogy(sample_list, gaps, f'-{markers[cfg]}',
                        color=clrs[cfg], linewidth=2, markersize=8,
                        label=config_labels[cfg])

    ax_gap.set_xlabel('Phase-space samples', fontsize=12)
    ax_gap.set_ylabel('Max gap ratio', fontsize=12)
    ax_gap.set_title('Gap Ratio Convergence\n'
                     '(higher = sharper rank boundary)', fontsize=13, fontweight='bold')
    ax_gap.legend(fontsize=10)
    ax_gap.grid(True, alpha=0.3)

    fig.suptitle('Level 4 Analysis: Poisson Algebra Growth at Different Configurations\n'
                 '1/r potential, Level 4 bracket computation',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUT_C, 'level4_dashboard.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved {out_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print("=" * 60)
    print("  Comprehensive Visualization Suite")
    print("=" * 60)

    if mode in ('all', 'rotate'):
        print("\n--- Group A: 100x100 Full-Sphere Rotations ---")
        render_per_potential_rotations()
        render_three_potential_rotation()

    if mode in ('all', 'multieps'):
        print("\n--- Group B: Multi-Epsilon Visuals ---")
        render_cross_potential_epsilon()
        render_epsilon_rotation()

    if mode in ('all', 'level4'):
        print("\n--- Group C: Level 4 Dashboard ---")
        render_level4_dashboard()

    print(f"\n{'='*60}")
    print("  Complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
