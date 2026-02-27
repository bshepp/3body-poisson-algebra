#!/usr/bin/env python3
"""
Shape Sphere Gap Ratio Landscape
=================================

Maps the SVD gap ratio onto the shape sphere of the three-body problem.

The shape sphere is the reduced configuration space after removing
center-of-mass (2 DOF), rotation (1 DOF), and scale (1 DOF), leaving
a 2-dimensional surface topologically equivalent to S².

Each point represents a distinct triangle shape. The gap ratio measures
how cleanly the SVD separates independent Poisson algebra generators from
dependent ones — the algebraic "sharpness" at that configuration.

Coordinate mapping (equal masses, body 1 at origin, body 2 on x-axis):
    Jacobi vectors:  w1 = r2 - r1 = (1, 0)
                     w2 = r3 - (r1+r2)/2
    Dragt shape coordinates on S²:
        s1 = (|w1|² - |w2|²) / (|w1|² + |w2|²)
        s2 = 2(w1 · w2) / (|w1|² + |w2|²)
        s3 = 2(w1 × w2) / (|w1|² + |w2|²)

    Lagrange equilateral  →  near north pole (s3 ≈ 1)
    Collinear configs     →  equator (s3 = 0)
    Collisions            →  three isolated points

Usage:
    python shape_sphere.py              # 1/r potential, existing data
    python shape_sphere.py 1_r2         # 1/r² potential
    python shape_sphere.py all          # all three potentials side by side
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import os
import sys


POTENTIAL_LABELS = {
    '1_r':      '1/r (Newton)',
    '1_r2':     r'1/r² (Calogero-Moser)',
    'harmonic': r'r² (Harmonic)',
}


def mu_phi_to_shape_sphere(mu, phi):
    """
    Map (mu, phi) shape parameters to Dragt coordinates on S².

    With body positions r1=(0,0), r2=(1,0), r3=(mu*cos(phi), mu*sin(phi)):
        w1 = (1, 0),  w2 = (mu*cos(phi) - 1/2,  mu*sin(phi))

    Returns (s1, s2, s3) satisfying s1² + s2² + s3² = 1.
    """
    w2_sq = mu**2 - mu * np.cos(phi) + 0.25
    N = 1.0 + w2_sq
    s1 = (1.0 - w2_sq) / N
    s2 = 2.0 * (mu * np.cos(phi) - 0.5) / N
    s3 = 2.0 * mu * np.sin(phi) / N
    return s1, s2, s3


def load_data(potential='1_r'):
    base = os.path.join('atlas_output', potential)
    return (
        np.load(os.path.join(base, 'mu_vals.npy')),
        np.load(os.path.join(base, 'phi_vals.npy')),
        np.load(os.path.join(base, 'rank_map.npy')),
        np.load(os.path.join(base, 'gap_map.npy')),
    )


def interpolate_to_fine_grid(mu, phi, data, n_fine=200):
    interp = RegularGridInterpolator(
        (mu, phi), data, method='cubic',
        bounds_error=False, fill_value=None
    )
    mu_f = np.linspace(mu[0], mu[-1], n_fine)
    phi_f = np.linspace(phi[0], phi[-1], n_fine)
    MU, PHI = np.meshgrid(mu_f, phi_f, indexing='ij')
    pts = np.stack([MU.ravel(), PHI.ravel()], axis=-1)
    DATA = interp(pts).reshape(MU.shape)
    return MU, PHI, DATA


def draw_reference_sphere(ax, alpha=0.08):
    """Draw a translucent reference sphere."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='lightsteelblue', alpha=alpha,
                    shade=False, antialiased=False)


def draw_equator(ax):
    """Draw the collinear great circle (equator)."""
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
            'k-', alpha=0.25, linewidth=0.8)


def make_single_sphere(potential='1_r', n_fine=200, save=True):
    """Two-panel figure: flat heatmap + 3D shape sphere."""
    mu, phi, rank, gap = load_data(potential)
    log_gap = np.log10(np.clip(gap, 1.0, None))

    MU, PHI, LG = interpolate_to_fine_grid(mu, phi, log_gap, n_fine)
    S1, S2, S3 = mu_phi_to_shape_sphere(MU, PHI)

    valid = np.isfinite(LG)
    vmin = np.nanpercentile(LG[valid], 1)
    vmax = np.nanpercentile(LG[valid], 99)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.inferno

    FC = cmap(norm(np.where(valid, LG, vmin)))

    fig = plt.figure(figsize=(18, 8), facecolor='white')

    # --- Panel 1: Flat heatmap ---
    ax1 = fig.add_subplot(121)
    im = ax1.pcolormesh(
        phi * 180 / np.pi, mu, log_gap, cmap=cmap,
        vmin=vmin, vmax=vmax, shading='auto'
    )

    specials = {
        'Lagrange':     (1.0,  np.pi / 3),
        'Euler':        (0.5,  np.pi),
        'Isosceles-90': (1.0,  np.pi / 2),
        'Lagrange-obt': (1.0,  2 * np.pi / 3),
    }
    for name, (m, p) in specials.items():
        if mu[0] <= m <= mu[-1] and phi[0] <= p <= phi[-1]:
            ax1.plot(p * 180 / np.pi, m, '*', color='cyan', markersize=14,
                     markeredgecolor='white', markeredgewidth=0.5)
            ax1.annotate(name, (p * 180 / np.pi, m),
                         textcoords='offset points', xytext=(8, 5),
                         color='white', fontsize=9, fontweight='bold',
                         path_effects=[
                             pe.withStroke(linewidth=2, foreground='black')
                         ])

    ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax1.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=13)
    ax1.set_title(r'Shape parameter space $(\mu, \phi)$', fontsize=14)
    cb1 = fig.colorbar(im, ax=ax1, pad=0.02)
    cb1.set_label(r'$\log_{10}$(gap ratio)', fontsize=12)

    # --- Panel 2: Shape sphere ---
    ax2 = fig.add_subplot(122, projection='3d')

    draw_reference_sphere(ax2)

    ax2.plot_surface(
        S1, S2, S3, facecolors=FC,
        rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.95
    )
    # Mirror to lower hemisphere (reflected triangles, same gap)
    FC_mirror = FC.copy()
    FC_mirror[:, :, 3] = 0.5
    ax2.plot_surface(
        S1, S2, -S3, facecolors=FC_mirror,
        rstride=1, cstride=1, shade=False, antialiased=True
    )

    draw_equator(ax2)

    # Mark special points
    for name, (m, p) in specials.items():
        if mu[0] <= m <= mu[-1] and phi[0] <= p <= phi[-1]:
            s1, s2, s3 = mu_phi_to_shape_sphere(m, p)
            ax2.scatter([s1], [s2], [s3], color='cyan', s=80, zorder=10,
                        edgecolors='white', linewidth=1, depthshade=False)
            ax2.text(s1 + 0.06, s2 + 0.06, s3 + 0.04, name,
                     color='cyan', fontsize=9, fontweight='bold')

    # Collision points (for reference)
    collision_labels = ['1-2 collision', '1-3 collision']
    coll_coords = [
        mu_phi_to_shape_sphere(100, np.pi / 3),
        mu_phi_to_shape_sphere(0.01, np.pi / 3),
    ]
    for label, (cs1, cs2, cs3) in zip(collision_labels, coll_coords):
        ax2.scatter([cs1], [cs2], [cs3], color='red', s=40, marker='x',
                    zorder=10, depthshade=False)

    ax2.set_xlabel('$s_1$', fontsize=11)
    ax2.set_ylabel('$s_2$', fontsize=11)
    ax2.set_zlabel('$s_3$', fontsize=11)
    ax2.set_title('Shape sphere $S^2$', fontsize=14)
    ax2.view_init(elev=30, azim=-55)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb2 = fig.colorbar(sm, ax=ax2, shrink=0.55, pad=0.08)
    cb2.set_label(r'$\log_{10}$(gap ratio)', fontsize=12)

    label = POTENTIAL_LABELS.get(potential, potential)
    fig.suptitle(
        f'Gap Ratio Landscape on the Shape Sphere — {label}',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'shape_sphere_{potential}.png'
    if save:
        plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  Saved {fname}")
    plt.close()
    return fname


def make_comparison(n_fine=200):
    """Three-panel shape sphere comparison across potentials."""
    potentials = []
    for p in ['1_r', '1_r2', 'harmonic']:
        if os.path.exists(os.path.join('atlas_output', p, 'gap_map.npy')):
            potentials.append(p)

    fig = plt.figure(figsize=(7 * len(potentials), 7), facecolor='white')

    for idx, pot in enumerate(potentials):
        mu, phi, rank, gap = load_data(pot)
        log_gap = np.log10(np.clip(gap, 1.0, None))
        MU, PHI, LG = interpolate_to_fine_grid(mu, phi, log_gap, n_fine)
        S1, S2, S3 = mu_phi_to_shape_sphere(MU, PHI)

        valid = np.isfinite(LG)
        vmin = np.nanpercentile(LG[valid], 1)
        vmax = np.nanpercentile(LG[valid], 99)
        norm_v = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.inferno
        FC = cmap(norm_v(np.where(valid, LG, vmin)))

        ax = fig.add_subplot(1, len(potentials), idx + 1, projection='3d')
        draw_reference_sphere(ax, alpha=0.05)
        ax.plot_surface(
            S1, S2, S3, facecolors=FC,
            rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.95
        )
        FC_m = FC.copy()
        FC_m[:, :, 3] = 0.45
        ax.plot_surface(
            S1, S2, -S3, facecolors=FC_m,
            rstride=1, cstride=1, shade=False, antialiased=True
        )
        draw_equator(ax)

        # Lagrange marker
        s1, s2, s3 = mu_phi_to_shape_sphere(1.0, np.pi / 3)
        ax.scatter([s1], [s2], [s3], color='cyan', s=70, zorder=10,
                   edgecolors='white', linewidth=1, depthshade=False)
        ax.text(s1 + 0.07, s2, s3 + 0.04, 'L', color='cyan',
                fontsize=11, fontweight='bold')

        ax.set_xlabel('$s_1$', fontsize=10, labelpad=-2)
        ax.set_ylabel('$s_2$', fontsize=10, labelpad=-2)
        ax.set_zlabel('$s_3$', fontsize=10, labelpad=-2)
        ax.set_title(POTENTIAL_LABELS.get(pot, pot), fontsize=13, pad=10)
        ax.view_init(elev=30, azim=-55)

        sm = cm.ScalarMappable(norm=norm_v, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
        cb.set_label(r'$\log_{10}$(gap)', fontsize=10)

    fig.suptitle(
        'Gap Ratio Landscape on the Shape Sphere — Potential Comparison',
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    fname = 'shape_sphere_comparison.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved {fname}")
    plt.close()
    return fname


def make_orthographic_views(potential='1_r', n_fine=200):
    """Four views of the shape sphere: top, front, side, tilted."""
    mu, phi, rank, gap = load_data(potential)
    log_gap = np.log10(np.clip(gap, 1.0, None))
    MU, PHI, LG = interpolate_to_fine_grid(mu, phi, log_gap, n_fine)
    S1, S2, S3 = mu_phi_to_shape_sphere(MU, PHI)

    valid = np.isfinite(LG)
    vmin = np.nanpercentile(LG[valid], 1)
    vmax = np.nanpercentile(LG[valid], 99)
    norm_v = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.inferno
    FC = cmap(norm_v(np.where(valid, LG, vmin)))

    views = [
        ('North pole (Lagrange)', 90, 0),
        ('Equatorial (collinear)', 0, -90),
        ('Side view', 0, 0),
        ('Tilted', 30, -55),
    ]

    fig = plt.figure(figsize=(20, 5), facecolor='white')

    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        draw_reference_sphere(ax, alpha=0.05)
        ax.plot_surface(
            S1, S2, S3, facecolors=FC,
            rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.95
        )
        FC_m = FC.copy()
        FC_m[:, :, 3] = 0.4
        ax.plot_surface(
            S1, S2, -S3, facecolors=FC_m,
            rstride=1, cstride=1, shade=False, antialiased=True
        )
        draw_equator(ax)

        s1, s2, s3 = mu_phi_to_shape_sphere(1.0, np.pi / 3)
        ax.scatter([s1], [s2], [s3], color='cyan', s=60, zorder=10,
                   edgecolors='white', linewidth=1, depthshade=False)

        ax.set_title(title, fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('$s_1$', fontsize=8, labelpad=-4)
        ax.set_ylabel('$s_2$', fontsize=8, labelpad=-4)
        ax.set_zlabel('$s_3$', fontsize=8, labelpad=-4)

    label = POTENTIAL_LABELS.get(potential, potential)
    fig.suptitle(
        f'Shape Sphere — Four Views — {label}',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    fname = f'shape_sphere_views_{potential}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved {fname}")
    plt.close()
    return fname


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '1_r'

    if target == 'all':
        print("Generating comparison figure...")
        make_comparison()
        for pot in ['1_r', '1_r2', 'harmonic']:
            if os.path.exists(os.path.join('atlas_output', pot, 'gap_map.npy')):
                print(f"\nGenerating {pot} detailed views...")
                make_single_sphere(pot)
                make_orthographic_views(pot)
    else:
        print(f"Generating {target} shape sphere...")
        make_single_sphere(target)
        make_orthographic_views(target)

    print("\nDone.")
