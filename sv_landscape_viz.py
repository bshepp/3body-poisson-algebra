#!/usr/bin/env python3
"""
Singular Value Landscape Visualizations
========================================

Dedicated visualizations for the hidden structure discovered in the
high-resolution (100x100) shape sphere atlas:

1. SV116 landscape — the 116th singular value as a continuous function
   on the shape sphere, revealing where the algebra is "closest to closing"
2. Near-degenerate pairs — map of points where SVs 115 and 116 are
   nearly equal (marginally independent generators)
3. 1/r vs 1/r² spectral fingerprint comparison — showing they share
   the same rank but have different spectral shapes
4. Harmonic SV15 hidden structure — variation in the last meaningful
   SV of the integrable potential
5. Spectral profile comparison — full SV spectra at representative points

Usage:
    python sv_landscape_viz.py          # generate all figures
    python sv_landscape_viz.py --open   # generate and open output folder
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HIRES_DIR = 'atlas_output_hires'
OUT_DIR = os.path.join(HIRES_DIR, 'sv_analysis')

SPECIALS = {
    'Lagrange':  (1.0, np.pi / 3),
    'Euler':     (0.5, np.pi),
    'Isos-90':   (1.0, np.pi / 2),
}

STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]


def mu_phi_to_shape_sphere(mu, phi):
    w2_sq = mu**2 - mu * np.cos(phi) + 0.25
    N = 1.0 + w2_sq
    s1 = (1.0 - w2_sq) / N
    s2 = 2.0 * (mu * np.cos(phi) - 0.5) / N
    s3 = 2.0 * mu * np.sin(phi) / N
    return s1, s2, s3


def load(potential_dir):
    d = os.path.join(HIRES_DIR, potential_dir)
    return {
        'mu':   np.load(os.path.join(d, 'mu_vals.npy')),
        'phi':  np.load(os.path.join(d, 'phi_vals.npy')),
        'rank': np.load(os.path.join(d, 'rank_map.npy')),
        'gap':  np.load(os.path.join(d, 'gap_map.npy')),
        'sv':   np.load(os.path.join(d, 'sv_spectra.npy')),
    }


def mark_specials(ax, deg=True):
    for name, (m, p) in SPECIALS.items():
        px = p * 180 / np.pi if deg else p
        ax.plot(px, m, '*', color='cyan', markersize=14,
                markeredgecolor='white', markeredgewidth=0.5, zorder=10)
        ax.annotate(name, (px, m), textcoords='offset points',
                    xytext=(8, 5), color='white', fontsize=9,
                    fontweight='bold', path_effects=STROKE)


def mark_specials_3d(ax):
    for name, (m, p) in SPECIALS.items():
        s1, s2, s3 = mu_phi_to_shape_sphere(m, p)
        ax.scatter([s1], [s2], [s3], color='cyan', s=60, zorder=10,
                   edgecolors='white', linewidth=1, depthshade=False)


def make_dual_panel(mu, phi, data_2d, cmap_name, label, title,
                    fname, vmin=None, vmax=None, specials=True, pct_clip=(1, 99)):
    """Reusable flat-heatmap + shape-sphere dual panel."""
    MU, PHI = np.meshgrid(mu, phi, indexing='ij')
    S1, S2, S3 = mu_phi_to_shape_sphere(MU, PHI)

    valid = np.isfinite(data_2d)
    if vmin is None:
        vmin = np.nanpercentile(data_2d[valid], pct_clip[0])
    if vmax is None:
        vmax = np.nanpercentile(data_2d[valid], pct_clip[1])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap_name)
    FC = cmap_obj(norm(np.clip(np.where(valid, data_2d, vmin), vmin, vmax)))

    fig = plt.figure(figsize=(18, 8), facecolor='white')

    ax1 = fig.add_subplot(121)
    im = ax1.pcolormesh(phi * 180 / np.pi, mu, data_2d,
                        cmap=cmap_name, vmin=vmin, vmax=vmax, shading='auto')
    if specials:
        mark_specials(ax1)
    ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax1.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=13)
    ax1.set_title(r'Shape parameter space $(\mu, \phi)$', fontsize=14)
    cb1 = fig.colorbar(im, ax=ax1, pad=0.02)
    cb1.set_label(label, fontsize=12)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(S1, S2, S3, facecolors=FC,
                     rstride=1, cstride=1, shade=False,
                     antialiased=True, alpha=0.95)
    FC_m = FC.copy()
    FC_m[:, :, 3] = 0.4
    ax2.plot_surface(S1, S2, -S3, facecolors=FC_m,
                     rstride=1, cstride=1, shade=False, antialiased=True)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
             'k-', alpha=0.25, linewidth=0.8)
    if specials:
        mark_specials_3d(ax2)
    ax2.set_xlabel('$s_1$', fontsize=11)
    ax2.set_ylabel('$s_2$', fontsize=11)
    ax2.set_zlabel('$s_3$', fontsize=11)
    ax2.set_title('Shape sphere $S^2$', fontsize=14)
    ax2.view_init(elev=30, azim=-55)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=ax2, shrink=0.55, pad=0.08)
    cb2.set_label(label, fontsize=12)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


# ===================================================================
# Figure 1: SV116 landscape for 1/r
# ===================================================================
def fig_sv116_landscape():
    print("\n--- Figure 1: SV #116 Landscape (1/r) ---")
    D = load('1_r')
    sv116 = D['sv'][:, :, 115]
    sv1   = D['sv'][:, :, 0]
    normed = np.log10(sv116 / sv1)
    mask = D['rank'] == 116
    normed[~mask] = np.nan

    make_dual_panel(
        D['mu'], D['phi'], normed,
        cmap_name='viridis',
        label=r'$\log_{10}(\sigma_{116} / \sigma_1)$',
        title='116th Singular Value Landscape — 1/r (Newton)\n'
              'Where the algebra is closest to closing at dim 115',
        fname='sv116_landscape_1r.png',
    )

    make_dual_panel(
        D['mu'], D['phi'], normed,
        cmap_name='viridis',
        label=r'$\log_{10}(\sigma_{116} / \sigma_1)$',
        title='116th Singular Value Landscape — 1/r (Newton)\n'
              'View 2: front',
        fname='sv116_landscape_1r_front.png',
    )


# ===================================================================
# Figure 2: SV115/SV116 ratio — near-degenerate pairs
# ===================================================================
def fig_near_degenerate():
    print("\n--- Figure 2: Near-Degenerate Pairs Map ---")
    D = load('1_r')
    sv115 = D['sv'][:, :, 114]
    sv116 = D['sv'][:, :, 115]
    ratio = sv115 / np.clip(sv116, 1e-30, None)
    mask = D['rank'] == 116
    ratio[~mask] = np.nan

    log_ratio = np.log10(ratio)

    make_dual_panel(
        D['mu'], D['phi'], log_ratio,
        cmap_name='RdYlBu_r',
        label=r'$\log_{10}(\sigma_{115} / \sigma_{116})$',
        title='Ratio of 115th to 116th Singular Values — 1/r (Newton)\n'
              'Low values (blue) = near-degenerate pair, generator barely independent',
        fname='sv_ratio_115_116.png',
    )


# ===================================================================
# Figure 3: 1/r vs 1/r² spectral fingerprint comparison
# ===================================================================
def fig_spectral_comparison():
    print("\n--- Figure 3: 1/r vs 1/r² Spectral Fingerprint ---")
    D1 = load('1_r')
    D2 = load('1_r2')

    sv116_1 = np.log10(D1['sv'][:, :, 115] / D1['sv'][:, :, 0])
    sv116_2 = np.log10(D2['sv'][:, :, 115] / D2['sv'][:, :, 0])
    mask = (D1['rank'] == 116) & (D2['rank'] == 116)
    sv116_1[~mask] = np.nan
    sv116_2[~mask] = np.nan

    diff = sv116_1 - sv116_2
    diff[~mask] = np.nan

    shared_vmin = min(np.nanpercentile(sv116_1, 1), np.nanpercentile(sv116_2, 1))
    shared_vmax = max(np.nanpercentile(sv116_1, 99), np.nanpercentile(sv116_2, 99))

    fig = plt.figure(figsize=(22, 14), facecolor='white')

    # Row 1: SV116 for 1/r and 1/r², same color scale
    ax1 = fig.add_subplot(231)
    im1 = ax1.pcolormesh(D1['phi'] * 180 / np.pi, D1['mu'], sv116_1,
                         cmap='viridis', vmin=shared_vmin, vmax=shared_vmax, shading='auto')
    mark_specials(ax1)
    ax1.set_xlabel(r'$\phi$ (deg)', fontsize=11)
    ax1.set_ylabel(r'$\mu$', fontsize=11)
    ax1.set_title(r'$\log_{10}(\sigma_{116}/\sigma_1)$ — 1/r', fontsize=13)
    fig.colorbar(im1, ax=ax1, pad=0.02)

    ax2 = fig.add_subplot(232)
    im2 = ax2.pcolormesh(D2['phi'] * 180 / np.pi, D2['mu'], sv116_2,
                         cmap='viridis', vmin=shared_vmin, vmax=shared_vmax, shading='auto')
    mark_specials(ax2)
    ax2.set_xlabel(r'$\phi$ (deg)', fontsize=11)
    ax2.set_ylabel(r'$\mu$', fontsize=11)
    ax2.set_title(r'$\log_{10}(\sigma_{116}/\sigma_1)$ — 1/r$^2$', fontsize=13)
    fig.colorbar(im2, ax=ax2, pad=0.02)

    # Difference map
    ax3 = fig.add_subplot(233)
    dlim = max(abs(np.nanpercentile(diff, 2)), abs(np.nanpercentile(diff, 98)))
    im3 = ax3.pcolormesh(D1['phi'] * 180 / np.pi, D1['mu'], diff,
                         cmap='RdBu_r', vmin=-dlim, vmax=dlim, shading='auto')
    mark_specials(ax3)
    ax3.set_xlabel(r'$\phi$ (deg)', fontsize=11)
    ax3.set_ylabel(r'$\mu$', fontsize=11)
    ax3.set_title(r'Difference: $\Delta\log_{10}(\sigma_{116}/\sigma_1)$', fontsize=13)
    fig.colorbar(im3, ax=ax3, pad=0.02)

    # Row 2: Gap ratio comparison and correlation scatter
    g1 = np.log10(np.clip(D1['gap'], 1, None))
    g2 = np.log10(np.clip(D2['gap'], 1, None))

    ax4 = fig.add_subplot(234)
    im4 = ax4.pcolormesh(D1['phi'] * 180 / np.pi, D1['mu'], g1,
                         cmap='inferno', shading='auto')
    mark_specials(ax4)
    ax4.set_xlabel(r'$\phi$ (deg)', fontsize=11)
    ax4.set_ylabel(r'$\mu$', fontsize=11)
    ax4.set_title(r'$\log_{10}$(gap ratio) — 1/r', fontsize=13)
    fig.colorbar(im4, ax=ax4, pad=0.02)

    ax5 = fig.add_subplot(235)
    im5 = ax5.pcolormesh(D2['phi'] * 180 / np.pi, D2['mu'], g2,
                         cmap='inferno', shading='auto')
    mark_specials(ax5)
    ax5.set_xlabel(r'$\phi$ (deg)', fontsize=11)
    ax5.set_ylabel(r'$\mu$', fontsize=11)
    ax5.set_title(r'$\log_{10}$(gap ratio) — 1/r$^2$', fontsize=13)
    fig.colorbar(im5, ax=ax5, pad=0.02)

    # Correlation scatter
    ax6 = fig.add_subplot(236)
    both_clean = mask.flatten()
    ax6.scatter(g1.flatten()[both_clean], g2.flatten()[both_clean],
                s=1, alpha=0.3, c='steelblue')
    lims = [min(g1[mask].min(), g2[mask].min()), max(g1[mask].max(), g2[mask].max())]
    ax6.plot(lims, lims, 'r--', linewidth=1, label='y = x')
    corr = np.corrcoef(g1[mask], g2[mask])[0, 1]
    ax6.set_xlabel(r'$\log_{10}$(gap) — 1/r', fontsize=11)
    ax6.set_ylabel(r'$\log_{10}$(gap) — 1/r$^2$', fontsize=11)
    ax6.set_title(f'Gap Ratio Correlation (r = {corr:.4f})', fontsize=13)
    ax6.legend(fontsize=10)

    fig.suptitle(
        'Spectral Fingerprint Comparison: 1/r (Newton) vs 1/r$^2$ (Calogero-Moser)\n'
        'Same rank (116) everywhere, but detectably different spectral shapes',
        fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'spectral_fingerprint_1r_vs_1r2.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved spectral_fingerprint_1r_vs_1r2.png")


# ===================================================================
# Figure 4: Harmonic SV15 hidden structure
# ===================================================================
def fig_harmonic_hidden():
    print("\n--- Figure 4: Harmonic SV #15 Hidden Structure ---")
    D = load('harmonic')
    sv15 = D['sv'][:, :, 14]
    sv1  = D['sv'][:, :, 0]
    normed = np.log10(sv15 / sv1)

    make_dual_panel(
        D['mu'], D['phi'], normed,
        cmap_name='magma',
        label=r'$\log_{10}(\sigma_{15} / \sigma_1)$',
        title='15th Singular Value Landscape — Harmonic (r$^2$)\n'
              'Hidden structure within a perfectly integrable system',
        fname='sv15_landscape_harmonic.png',
    )


# ===================================================================
# Figure 5: Full spectral profiles at representative configurations
# ===================================================================
def fig_spectral_profiles():
    print("\n--- Figure 5: Full Spectral Profiles ---")
    D1 = load('1_r')
    D2 = load('1_r2')
    DH = load('harmonic')

    configs = [
        ('Generic scalene',  0.5,  0.8),
        ('Near equilateral', 1.0,  np.pi / 3),
        ('Large mass ratio', 2.5,  0.8),
        ('Near collinear',   0.7,  2.8),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')

    for ax, (label, target_mu, target_phi) in zip(axes.flat, configs):
        i1 = np.argmin(np.abs(D1['mu'] - target_mu))
        j1 = np.argmin(np.abs(D1['phi'] - target_phi))

        s1 = D1['sv'][i1, j1, :]
        s2 = D2['sv'][i1, j1, :]
        sh = DH['sv'][i1, j1, :]

        s1_n = s1 / s1[0]
        s2_n = s2 / s2[0]
        sh_n = sh / sh[0]

        idx = np.arange(1, len(s1) + 1)
        ax.semilogy(idx, s1_n, 'b-', linewidth=1.5, alpha=0.8, label='1/r (Newton)')
        ax.semilogy(idx, s2_n, 'r--', linewidth=1.5, alpha=0.8, label='1/r$^2$ (Calogero)')
        ax.semilogy(idx[:len(sh)], sh_n, 'g-', linewidth=1.5, alpha=0.8, label='r$^2$ (Harmonic)')

        ax.axhline(1e-12, color='gray', linestyle=':', alpha=0.5, label='Noise floor')
        ax.axvline(116, color='blue', linestyle=':', alpha=0.4)
        ax.axvline(15, color='green', linestyle=':', alpha=0.4)

        r1 = D1['rank'][i1, j1]
        r2 = D2['rank'][i1, j1]
        rh = DH['rank'][i1, j1]

        ax.set_title(f'{label}\n'
                     rf'$\mu$={D1["mu"][i1]:.2f}, $\phi$={D1["phi"][j1]:.2f} '
                     f'(ranks: {r1}/{r2}/{rh})',
                     fontsize=12)
        ax.set_xlabel('Singular value index', fontsize=11)
        ax.set_ylabel(r'$\sigma_k / \sigma_1$', fontsize=11)
        ax.set_ylim(1e-18, 2)
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Full Singular Value Spectra at Representative Configurations\n'
        'Normalized by largest SV; vertical lines at rank 15 (harmonic) and 116 (singular)',
        fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'spectral_profiles.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved spectral_profiles.png")


# ===================================================================
# Figure 6: Three-potential SV comparison on shape sphere
# ===================================================================
def fig_three_potential_sv():
    print("\n--- Figure 6: Three-Potential Last-SV Comparison ---")
    D1 = load('1_r')
    D2 = load('1_r2')
    DH = load('harmonic')

    datasets = [
        ('1/r (Newton)', D1, 115, 116),
        ('1/r$^2$ (Calogero-Moser)', D2, 115, 116),
        ('r$^2$ (Harmonic)', DH, 14, 15),
    ]

    fig = plt.figure(figsize=(21, 7), facecolor='white')

    for idx, (label, D, sv_idx, expected_rank) in enumerate(datasets):
        sv_last = D['sv'][:, :, sv_idx]
        sv_first = D['sv'][:, :, 0]
        normed = np.log10(sv_last / sv_first)
        mask = D['rank'] == expected_rank
        normed[~mask] = np.nan

        MU, PHI = np.meshgrid(D['mu'], D['phi'], indexing='ij')
        S1, S2, S3 = mu_phi_to_shape_sphere(MU, PHI)

        valid = np.isfinite(normed)
        vmin = np.nanpercentile(normed[valid], 1)
        vmax = np.nanpercentile(normed[valid], 99)
        norm_obj = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap('viridis')
        FC = cmap_obj(norm_obj(np.clip(np.where(valid, normed, vmin), vmin, vmax)))

        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.plot_surface(S1, S2, S3, facecolors=FC,
                        rstride=1, cstride=1, shade=False,
                        antialiased=True, alpha=0.95)
        FC_m = FC.copy()
        FC_m[:, :, 3] = 0.4
        ax.plot_surface(S1, S2, -S3, facecolors=FC_m,
                        rstride=1, cstride=1, shade=False, antialiased=True)
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
                'k-', alpha=0.25, linewidth=0.8)
        mark_specials_3d(ax)

        ax.set_xlabel('$s_1$', fontsize=10, labelpad=-2)
        ax.set_ylabel('$s_2$', fontsize=10, labelpad=-2)
        ax.set_zlabel('$s_3$', fontsize=10, labelpad=-2)
        ax.set_title(f'{label}\n$\\sigma_{{{sv_idx+1}}} / \\sigma_1$', fontsize=12, pad=10)
        ax.view_init(elev=30, azim=-55)

        sm = cm.ScalarMappable(norm=norm_obj, cmap=cmap_obj)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
        cb.set_label(r'$\log_{10}$', fontsize=10)

    fig.suptitle(
        'Last Meaningful Singular Value on the Shape Sphere\n'
        'Strength of the marginal generator across all three potentials',
        fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'three_potential_sv_comparison.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved three_potential_sv_comparison.png")


# ===================================================================
# Figure 7: Anomaly map overlay — where 1/r and 1/r² break
# ===================================================================
def fig_anomaly_map():
    print("\n--- Figure 7: Anomaly Overlay Map ---")
    D1 = load('1_r')
    D2 = load('1_r2')

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), facecolor='white')

    # 1/r rank map
    ax = axes[0]
    im = ax.pcolormesh(D1['phi'] * 180 / np.pi, D1['mu'], D1['rank'],
                       cmap='RdYlGn_r', vmin=115, vmax=126, shading='auto')
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)', fontsize=11)
    ax.set_ylabel(r'$\mu$', fontsize=11)
    ax.set_title('Rank Map — 1/r', fontsize=13)
    fig.colorbar(im, ax=ax, pad=0.02)

    # 1/r² rank map
    ax = axes[1]
    im = ax.pcolormesh(D2['phi'] * 180 / np.pi, D2['mu'], D2['rank'],
                       cmap='RdYlGn_r', vmin=115, vmax=126, shading='auto')
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)', fontsize=11)
    ax.set_ylabel(r'$\mu$', fontsize=11)
    ax.set_title('Rank Map — 1/r$^2$', fontsize=13)
    fig.colorbar(im, ax=ax, pad=0.02)

    # Overlay: which points are anomalous in each
    ax = axes[2]
    both_ok = (D1['rank'] == 116) & (D2['rank'] == 116)
    only_1r = (D1['rank'] > 116) & (D2['rank'] == 116)
    only_1r2 = (D1['rank'] == 116) & (D2['rank'] > 116)
    both_anom = (D1['rank'] > 116) & (D2['rank'] > 116)

    overlay = np.zeros_like(D1['rank'], dtype=float)
    overlay[both_ok] = 0
    overlay[only_1r] = 1
    overlay[only_1r2] = 2
    overlay[both_anom] = 3

    cmap_cat = colors.ListedColormap(['#2d3436', '#e17055', '#00b894', '#d63031'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_cat = colors.BoundaryNorm(bounds, cmap_cat.N)
    im = ax.pcolormesh(D1['phi'] * 180 / np.pi, D1['mu'], overlay,
                       cmap=cmap_cat, norm=norm_cat, shading='auto')
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)', fontsize=11)
    ax.set_ylabel(r'$\mu$', fontsize=11)
    ax.set_title('Anomaly Overlap', fontsize=13)
    cb = fig.colorbar(im, ax=ax, pad=0.02, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels([
        f'Both OK ({both_ok.sum()})',
        f'Only 1/r ({only_1r.sum()})',
        f'Only 1/r$^2$ ({only_1r2.sum()})',
        f'Both ({both_anom.sum()})',
    ], fontsize=9)

    fig.suptitle(
        'Numerical Anomalies (rank > 116): Location and Overlap\n'
        'All anomalies cluster near isosceles line ($\\mu \\approx 1$) at small $\\phi$',
        fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'anomaly_overlay.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved anomaly_overlay.png")


# ===================================================================
# Figure 8: S3 symmetry — isosceles curves overlaid on gap landscape
# ===================================================================
def _draw_isosceles_flat(ax, mu, phi):
    """Overlay the three isosceles curves on a flat (phi, mu) heatmap."""
    phi_dense = np.linspace(phi[0], phi[-1], 500)

    ax.axhline(1.0, color='cyan', linewidth=2, linestyle='--', alpha=0.9,
               label=r'$r_{13}=r_{12}$: $\mu=1$')

    mu_c2 = 2 * np.cos(phi_dense)
    m2 = (mu_c2 >= mu[0]) & (mu_c2 <= mu[-1])
    ax.plot(phi_dense[m2] * 180 / np.pi, mu_c2[m2],
            color='lime', linewidth=2, linestyle='--', alpha=0.9,
            label=r'$r_{23}=r_{12}$: $\mu=2\cos\phi$')

    cos_p = np.cos(phi_dense)
    safe = cos_p > 0.01
    mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
    m3 = np.isfinite(mu_c3) & (mu_c3 >= mu[0]) & (mu_c3 <= mu[-1])
    ax.plot(phi_dense[m3] * 180 / np.pi, mu_c3[m3],
            color='magenta', linewidth=2, linestyle='--', alpha=0.9,
            label=r'$r_{13}=r_{23}$: $\mu=1/(2\cos\phi)$')

    ax.plot(60, 1.0, '*', color='white', markersize=18,
            markeredgecolor='black', markeredgewidth=1, zorder=15)
    ax.annotate('Lagrange\n(equilateral)', (60, 1.0),
                textcoords='offset points', xytext=(12, 10),
                color='white', fontsize=11, fontweight='bold',
                path_effects=STROKE)


def _draw_isosceles_3d(ax):
    """Overlay the three isosceles curves on a 3D shape sphere."""
    phi_c1 = np.linspace(0.1, np.pi - 0.1, 300)
    s1c, s2c, s3c = mu_phi_to_shape_sphere(1.0, phi_c1)
    ax.plot(s1c, s2c, s3c, color='cyan', linewidth=2.5, alpha=0.9, zorder=5)

    phi_c2 = np.linspace(0.05, np.pi / 2 - 0.05, 300)
    mu_c2 = 2 * np.cos(phi_c2)
    v2 = (mu_c2 >= 0.2) & (mu_c2 <= 3.0)
    s1c, s2c, s3c = mu_phi_to_shape_sphere(mu_c2[v2], phi_c2[v2])
    ax.plot(s1c, s2c, s3c, color='lime', linewidth=2.5, alpha=0.9, zorder=5)

    phi_c3 = np.linspace(0.05, np.pi / 2 - 0.05, 300)
    cos_c3 = np.cos(phi_c3)
    mu_c3 = 1.0 / (2 * cos_c3)
    v3 = (mu_c3 >= 0.2) & (mu_c3 <= 3.0)
    s1c, s2c, s3c = mu_phi_to_shape_sphere(mu_c3[v3], phi_c3[v3])
    ax.plot(s1c, s2c, s3c, color='magenta', linewidth=2.5, alpha=0.9, zorder=5)

    s1l, s2l, s3l = mu_phi_to_shape_sphere(1.0, np.pi / 3)
    ax.scatter([s1l], [s2l], [s3l], color='white', s=120, zorder=15,
               edgecolors='black', linewidth=1.5, depthshade=False, marker='*')


def fig_s3_symmetry():
    print("\n--- Figure 8: S3 Symmetry Overlay ---")

    for pot_dir, pot_label in [('1_r', '1/r (Newton)'),
                                ('1_r2', r'1/r$^2$ (Calogero-Moser)')]:
        D = load(pot_dir)
        lg = np.log10(np.clip(D['gap'], 1, None))
        MU, PHI = np.meshgrid(D['mu'], D['phi'], indexing='ij')
        S1, S2, S3_coord = mu_phi_to_shape_sphere(MU, PHI)

        valid = np.isfinite(lg)
        vmin = np.nanpercentile(lg[valid], 1)
        vmax = np.nanpercentile(lg[valid], 99)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = cm.inferno
        FC = cmap_obj(norm(np.where(valid, lg, vmin)))

        fig = plt.figure(figsize=(20, 9), facecolor='white')

        ax1 = fig.add_subplot(121)
        ax1.pcolormesh(D['phi'] * 180 / np.pi, D['mu'], lg,
                       cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
        _draw_isosceles_flat(ax1, D['mu'], D['phi'])
        ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
        ax1.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=13)
        ax1.set_title(r'Gap ratio with $S_3$ isosceles curves', fontsize=14)
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.8,
                   edgecolor='white', facecolor='#333333', labelcolor='white')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(S1, S2, S3_coord, facecolors=FC,
                         rstride=1, cstride=1, shade=False,
                         antialiased=True, alpha=0.9)
        FC_m = FC.copy()
        FC_m[:, :, 3] = 0.35
        ax2.plot_surface(S1, S2, -S3_coord, facecolors=FC_m,
                         rstride=1, cstride=1, shade=False, antialiased=True)
        theta = np.linspace(0, 2 * np.pi, 300)
        ax2.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
                 'k-', alpha=0.25, linewidth=0.8)
        _draw_isosceles_3d(ax2)
        ax2.set_xlabel('$s_1$', fontsize=11)
        ax2.set_ylabel('$s_2$', fontsize=11)
        ax2.set_zlabel('$s_3$', fontsize=11)
        ax2.set_title(r'Shape sphere with $S_3$ symmetry walls', fontsize=14)
        ax2.view_init(elev=25, azim=-50)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        fig.colorbar(sm, ax=ax2, shrink=0.55, pad=0.08)

        fig.suptitle(
            f'$S_3$ Symmetry in the Poisson Algebra — {pot_label}\n'
            r'Three isosceles curves (permutation symmetry walls) meet at Lagrange',
            fontsize=15, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(OUT_DIR, f's3_symmetry_{pot_dir}.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved s3_symmetry_{pot_dir}.png")

    # Zoomed view near Lagrange
    D = load('1_r2')
    lg = np.log10(np.clip(D['gap'], 1, None))
    mu_mask = (D['mu'] >= 0.4) & (D['mu'] <= 1.8)
    phi_mask = (D['phi'] >= 0.3) & (D['phi'] <= 1.8)
    mu_z = D['mu'][mu_mask]
    phi_z = D['phi'][phi_mask]
    lg_z = lg[np.ix_(mu_mask, phi_mask)]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.pcolormesh(phi_z * 180 / np.pi, mu_z, lg_z, cmap='inferno', shading='auto')
    _draw_isosceles_flat(ax, mu_z, phi_z)
    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=14)
    ax.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=14)
    ax.set_title(r'Zoom: $S_3$ Symmetry Walls Meet at Lagrange — 1/r$^2$', fontsize=15)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.85,
              edgecolor='white', facecolor='#333333', labelcolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 's3_symmetry_zoom_lagrange.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved s3_symmetry_zoom_lagrange.png")


# ===================================================================
# Main
# ===================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Singular Value Landscape Analysis")
    print("=" * 60)

    fig_sv116_landscape()
    fig_near_degenerate()
    fig_spectral_comparison()
    fig_harmonic_hidden()
    fig_spectral_profiles()
    fig_three_potential_sv()
    fig_anomaly_map()
    fig_s3_symmetry()

    print(f"\n  All figures saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
