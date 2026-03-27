#!/usr/bin/env python3
"""
Render shape-sphere atlas triptychs for all completed full_atlas configs.

For each completed config in aws_results/atlas_full/, produces:
  1. Clean two-panel figure (heatmap + sphere, no annotations)
  2. Triptych: [1:1:1 gravitational baseline] | [this config] | [difference]

Usage:
    python render_full_atlas.py                   # all completed configs
    python render_full_atlas.py 1r_q+1_-1_-1      # specific config
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RegularGridInterpolator

DATA_DIR = os.path.join('aws_results', 'atlas_full')
OUT_DIR = os.path.join('aws_results', 'atlas_figures')
BASELINE = '1r_q+1_+1_+1'


def mu_phi_to_sphere(mu, phi):
    w2_sq = mu**2 - mu * np.cos(phi) + 0.25
    N = 1.0 + w2_sq
    return (1.0 - w2_sq) / N, 2.0 * (mu * np.cos(phi) - 0.5) / N, 2.0 * mu * np.sin(phi) / N


def load_config(name):
    d = os.path.join(DATA_DIR, name)
    mu = np.load(os.path.join(d, 'mu_vals.npy'))
    phi = np.load(os.path.join(d, 'phi_vals.npy'))
    rank = np.load(os.path.join(d, 'rank_map.npy'))
    gap = np.load(os.path.join(d, 'gap_map.npy'))
    with open(os.path.join(d, 'config.json')) as f:
        cfg = json.load(f)
    with open(os.path.join(d, 'summary.json')) as f:
        summary = json.load(f)
    return mu, phi, rank, gap, cfg, summary


def interpolate(mu, phi, data, n_fine=200):
    interp = RegularGridInterpolator(
        (mu, phi), data, method='cubic',
        bounds_error=False, fill_value=None
    )
    mu_f = np.linspace(mu[0], mu[-1], n_fine)
    phi_f = np.linspace(phi[0], phi[-1], n_fine)
    MU, PHI = np.meshgrid(mu_f, phi_f, indexing='ij')
    pts = np.stack([MU.ravel(), PHI.ravel()], axis=-1)
    return MU, PHI, interp(pts).reshape(MU.shape)


def _heatmap(ax, phi, mu, data, cmap, vmin, vmax, title):
    """Render a clean flat heatmap with no annotations."""
    im = ax.pcolormesh(
        phi * 180 / np.pi, mu, data, cmap=cmap,
        vmin=vmin, vmax=vmax, shading='auto'
    )
    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=11)
    ax.set_ylabel(r'$\mu$', fontsize=11)
    ax.set_title(title, fontsize=12)
    return im


def _sphere(ax, MU, PHI, LG, norm, cmap):
    """Render a clean 3D shape sphere with no annotations."""
    S1, S2, S3 = mu_phi_to_sphere(MU, PHI)
    valid = np.isfinite(LG)
    FC = cmap(norm(np.where(valid, LG, norm.vmin)))

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color='lightsteelblue', alpha=0.08, shade=False, antialiased=False)

    ax.plot_surface(S1, S2, S3, facecolors=FC,
                    rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.95)
    FC_m = FC.copy()
    FC_m[:, :, 3] = 0.5
    ax.plot_surface(S1, S2, -S3, facecolors=FC_m,
                    rstride=1, cstride=1, shade=False, antialiased=True)

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
            'k-', alpha=0.25, linewidth=0.8)
    ax.set_xlabel('$s_1$', fontsize=9)
    ax.set_ylabel('$s_2$', fontsize=9)
    ax.set_zlabel('$s_3$', fontsize=9)
    ax.view_init(elev=30, azim=-55)


def render_single(name):
    """Clean two-panel figure (heatmap + sphere), no markers or contours."""
    mu, phi, rank, gap, cfg, summary = load_config(name)
    label = cfg.get('label', name)
    r116 = summary.get('rank_116_fraction', 0)
    uniq = sorted(summary.get('unique_ranks', []))
    rank_str = ', '.join(str(r) for r in uniq)

    log_gap = np.log10(np.clip(gap, 1.0, None))
    MU, PHI, LG = interpolate(mu, phi, log_gap)

    valid = np.isfinite(LG)
    vmin = np.nanpercentile(LG[valid], 1)
    vmax = np.nanpercentile(LG[valid], 99)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.inferno

    fig = plt.figure(figsize=(18, 8), facecolor='white')

    ax1 = fig.add_subplot(121)
    im = _heatmap(ax1, phi, mu, log_gap, cmap, vmin, vmax,
                  r'Shape parameter space $(\mu, \phi)$')
    cb1 = fig.colorbar(im, ax=ax1, pad=0.02)
    cb1.set_label(r'$\log_{10}$(gap ratio)', fontsize=11)

    ax2 = fig.add_subplot(122, projection='3d')
    _sphere(ax2, MU, PHI, LG, norm, cmap)
    ax2.set_title('Shape sphere $S^2$', fontsize=12)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=ax2, shrink=0.55, pad=0.08)
    cb2.set_label(r'$\log_{10}$(gap ratio)', fontsize=11)

    fig.suptitle(
        f'{label}  —  rank 116: {r116:.1%}  |  unique ranks: {{{rank_str}}}',
        fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = os.path.join(OUT_DIR, f'atlas_{name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return fname


def render_triptych(name, base_gap, base_mu, base_phi, base_cfg, base_summary):
    """Three-panel comparison: baseline | config | difference."""
    mu, phi, rank, gap, cfg, summary = load_config(name)
    label = cfg.get('label', name)
    base_label = base_cfg.get('label', BASELINE)
    r116 = summary.get('rank_116_fraction', 0)
    base_r116 = base_summary.get('rank_116_fraction', 0)

    log_gap = np.log10(np.clip(gap, 1.0, None))
    log_base = np.log10(np.clip(base_gap, 1.0, None))

    if log_gap.shape != log_base.shape:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (base_mu, base_phi), log_base, method='linear',
            bounds_error=False, fill_value=np.nan)
        MU_g, PHI_g = np.meshgrid(mu, phi, indexing='ij')
        log_base_resampled = interp(
            np.stack([MU_g.ravel(), PHI_g.ravel()], axis=-1)
        ).reshape(log_gap.shape)
        diff = log_gap - log_base_resampled
    else:
        diff = log_gap - log_base

    all_vals = np.concatenate([log_gap.ravel(), log_base.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    shared_vmin = np.percentile(all_vals, 1)
    shared_vmax = np.percentile(all_vals, 99)

    diff_abs = np.nanmax(np.abs(diff[np.isfinite(diff)]))
    diff_lim = max(diff_abs, 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor='white')
    cmap_gap = cm.inferno
    cmap_diff = cm.RdBu_r

    # Panel 1: baseline
    im0 = _heatmap(axes[0], base_phi, base_mu, log_base, cmap_gap,
                   shared_vmin, shared_vmax,
                   f'{base_label}\nrank 116: {base_r116:.1%}')
    fig.colorbar(im0, ax=axes[0], pad=0.02).set_label(
        r'$\log_{10}$(gap ratio)', fontsize=10)

    # Panel 2: this config
    im1 = _heatmap(axes[1], phi, mu, log_gap, cmap_gap,
                   shared_vmin, shared_vmax,
                   f'{label}\nrank 116: {r116:.1%}')
    fig.colorbar(im1, ax=axes[1], pad=0.02).set_label(
        r'$\log_{10}$(gap ratio)', fontsize=10)

    # Panel 3: difference
    im2 = _heatmap(axes[2], phi, mu, diff, cmap_diff,
                   -diff_lim, diff_lim,
                   f'Difference ({label}\nminus {base_label})')
    fig.colorbar(im2, ax=axes[2], pad=0.02).set_label(
        r'$\Delta\;\log_{10}$(gap ratio)', fontsize=10)

    fig.suptitle(
        f'Atlas Comparison — {label}  vs  {base_label}',
        fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f'triptych_{name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return fname


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = sys.argv[1:] if len(sys.argv) > 1 else None

    configs = []
    for name in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, name)
        if not os.path.isdir(d):
            continue
        if not os.path.exists(os.path.join(d, 'summary.json')):
            continue
        if targets and name not in targets:
            continue
        configs.append(name)

    print(f'Loading baseline: {BASELINE}')
    b_mu, b_phi, b_rank, b_gap, b_cfg, b_sum = load_config(BASELINE)

    print(f'Rendering {len(configs)} configs...\n')
    for i, name in enumerate(configs, 1):
        print(f'[{i}/{len(configs)}] {name}')
        fname = render_single(name)
        print(f'    single -> {fname}')
        if name != BASELINE:
            fname_t = render_triptych(name, b_gap, b_mu, b_phi, b_cfg, b_sum)
            print(f'  triptych -> {fname_t}')
        else:
            print(f'  triptych -> (skipped, is baseline)')

    print(f'\nDone. Figures saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
