#!/usr/bin/env python3
"""
CG Atlas Comparison — Phase 1 Item 1.4
=======================================

Extends the Clebsch-Gordan doublet analysis (previously done only at
the Lagrange point) to the full shape-sphere atlas. Tests whether the
S₃ representation-theoretic prediction of doublet structure holds
only at symmetric configurations or persists more broadly.

Output goes to spectral_depth/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HIRES_DIR = 'atlas_output_hires'
OUT_DIR = 'spectral_depth'
os.makedirs(OUT_DIR, exist_ok=True)

SPECIALS = {
    'Lagrange':  (1.0, np.pi / 3),
    'Euler':     (0.5, np.pi),
    'Isos-90':   (1.0, np.pi / 2),
}
STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]

# Tier boundaries from clebsch_gordan_analysis.py
TIER_BOUNDS = [0, 52, 96, 112, 116]
TIER_NAMES  = ['Tier 1 (0–52)', 'Tier 2 (52–96)', 'Tier 3 (96–112)', 'Tier 4 (112–116)']

DOUBLET_THRESHOLD = 1.05  # ratio < this => near-degenerate pair


def mark_specials(ax):
    for name, (m, p) in SPECIALS.items():
        px = np.degrees(p)
        ax.plot(px, m, '*', color='cyan', markersize=12,
                markeredgecolor='white', markeredgewidth=0.5, zorder=10)
        ax.annotate(name, (px, m), textcoords='offset points',
                    xytext=(6, 4), color='white', fontsize=8,
                    fontweight='bold', path_effects=STROKE)


def count_doublets(spectrum, start, end, threshold=DOUBLET_THRESHOLD):
    """Count near-degenerate pairs in a sorted SV slice."""
    seg = np.sort(spectrum[start:end])[::-1]  # descending
    count = 0
    i = 0
    while i < len(seg) - 1:
        if seg[i] > 0 and seg[i + 1] / seg[i] > (1.0 / threshold):
            count += 1
            i += 2  # skip pair
        else:
            i += 1
    return count


def load_sv():
    """Load 1/r sv_spectra (use base epsilon, not epsilon subdirs)."""
    d = os.path.join(HIRES_DIR, '1_r')
    return {
        'mu':   np.load(os.path.join(d, 'mu_vals.npy')),
        'phi':  np.load(os.path.join(d, 'phi_vals.npy')),
        'rank': np.load(os.path.join(d, 'rank_map.npy')),
        'sv':   np.load(os.path.join(d, 'sv_spectra.npy')),
    }


def compute_doublet_maps(sv, rank):
    """Compute doublet counts at every grid point for each tier."""
    n_mu, n_phi = sv.shape[:2]
    mask = rank >= 116

    total_doublets = np.full((n_mu, n_phi), np.nan)
    tier_doublets = {name: np.full((n_mu, n_phi), np.nan) for name in TIER_NAMES}

    for i in range(n_mu):
        for j in range(n_phi):
            if not mask[i, j]:
                continue
            spec = sv[i, j, :116]
            total = 0
            for t, (lo, hi) in enumerate(zip(TIER_BOUNDS[:-1], TIER_BOUNDS[1:])):
                c = count_doublets(spec, lo, hi)
                tier_doublets[TIER_NAMES[t]][i, j] = c
                total += c
            total_doublets[i, j] = total

    return total_doublets, tier_doublets


def fig_doublet_landscape(mu, phi, total_doublets, tier_doublets):
    """Figure 1: Doublet count heatmaps — total + per tier."""
    phi_deg = np.degrees(phi)

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), facecolor='white')

    # Total doublets
    ax = axes[0]
    valid = total_doublets[np.isfinite(total_doublets)]
    im = ax.pcolormesh(phi_deg, mu, total_doublets, cmap='hot',
                       vmin=valid.min(), vmax=valid.max(), shading='auto')
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\mu$')
    ax.set_title('Total Doublets', fontsize=11)
    plt.colorbar(im, ax=ax, pad=0.02)

    # Per-tier
    for ax, name in zip(axes[1:], TIER_NAMES):
        data = tier_doublets[name]
        valid_t = data[np.isfinite(data)]
        if len(valid_t) == 0:
            continue
        im = ax.pcolormesh(phi_deg, mu, data, cmap='hot',
                           vmin=valid_t.min(), vmax=valid_t.max(), shading='auto')
        mark_specials(ax)
        ax.set_xlabel(r'$\phi$ (deg)')
        ax.set_ylabel(r'$\mu$')
        ax.set_title(name, fontsize=10)
        plt.colorbar(im, ax=ax, pad=0.02)

    fig.suptitle('Near-Degenerate Doublet Count Across Shape Sphere — 1/r',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = 'cg_doublet_landscape.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


def fig_predicted_vs_observed(mu, phi, total_doublets):
    """Figure 2: Doublet count vs distance from symmetry points + E-fraction."""
    phi_deg = np.degrees(phi)
    MU, PHI = np.meshgrid(mu, phi, indexing='ij')
    mask = np.isfinite(total_doublets)

    # Distance from Lagrange in (mu, phi) space
    d_lag = np.sqrt((MU - 1.0)**2 + (PHI - np.pi / 3)**2)

    # E-fraction: fraction of 116 SVs in doublets → 2 * total_doublets / 116
    e_frac = 2.0 * total_doublets / 116.0

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='white')

    # Panel 1: scatter of doublet count vs distance from Lagrange
    ax = axes[0]
    ax.scatter(d_lag[mask], total_doublets[mask], s=4, alpha=0.3, c='steelblue')
    ax.set_xlabel('Distance from Lagrange point', fontsize=11)
    ax.set_ylabel('Total doublet count', fontsize=11)
    ax.set_title('Doublets vs Distance from Lagrange', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel 2: E-fraction heatmap
    ax = axes[1]
    valid_ef = e_frac[np.isfinite(e_frac)]
    im = ax.pcolormesh(phi_deg, mu, e_frac, cmap='RdYlBu_r',
                       vmin=0, vmax=min(1.0, valid_ef.max()), shading='auto')
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\mu$')
    ax.set_title('E-fraction (frac. of SVs in doublets)', fontsize=11)
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('2 × doublets / 116', fontsize=9)

    # Panel 3: Lagrange residual — count at Lagrange minus count elsewhere
    def nearest_idx(arr, val):
        return np.argmin(np.abs(arr - val))

    i_lag = nearest_idx(mu, 1.0)
    j_lag = nearest_idx(phi, np.pi / 3)
    lag_count = total_doublets[i_lag, j_lag]
    residual = total_doublets - lag_count
    residual[~mask] = np.nan

    ax = axes[2]
    rlim = max(abs(np.nanmin(residual)), abs(np.nanmax(residual)))
    if rlim == 0:
        rlim = 1
    im = ax.pcolormesh(phi_deg, mu, residual, cmap='RdBu_r',
                       vmin=-rlim, vmax=rlim, shading='auto')
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\mu$')
    ax.set_title(f'Residual (count − Lagrange count={lag_count:.0f})', fontsize=11)
    plt.colorbar(im, ax=ax, pad=0.02)

    fig.suptitle('CG Doublet Predictions vs Observations — 1/r',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = 'cg_predicted_vs_observed.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


if __name__ == '__main__':
    print("\n--- 1.4 CG Atlas Comparison ---")
    D = load_sv()
    total_doublets, tier_doublets = compute_doublet_maps(D['sv'], D['rank'])
    fig_doublet_landscape(D['mu'], D['phi'], total_doublets, tier_doublets)
    fig_predicted_vs_observed(D['mu'], D['phi'], total_doublets)

    # Print summary at special points
    for name, (m_val, p_val) in SPECIALS.items():
        i = np.argmin(np.abs(D['mu'] - m_val))
        j = np.argmin(np.abs(D['phi'] - p_val))
        print(f"  {name}: total doublets = {total_doublets[i, j]:.0f}, "
              f"E-fraction = {2 * total_doublets[i, j] / 116:.2f}")

    print(f"\nAll output in {OUT_DIR}/")
