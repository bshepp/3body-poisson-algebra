#!/usr/bin/env python3
"""
Animate the stability atlas as the potential exponent sweeps from 1/r to 1/r^2.

For each exponent n in [1.0, 2.0], computes the level-3 Poisson algebra
for V = -u_ij^n, evaluates the SVD gap ratio over shape space, and renders
a frame.  Frames are stitched into an animated GIF.

Usage:
    python animate_atlas.py                     # default: 11 frames, 50x50 grid
    python animate_atlas.py --frames 21         # smoother animation
    python animate_atlas.py --grid 100          # higher resolution (slower)
    python animate_atlas.py --test              # quick test: 3 frames, 20x20
"""

import argparse
import os
import sys
import numpy as np
from numpy.linalg import svd
from time import time
from sympy import nsimplify

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from exact_growth import (
    symbols, diff, Integer, cancel, expand,
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    u12, u13, u23,
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    total_deriv, poisson_bracket, simplify_generator,
    lambdify_generators,
    T1, T2, T3,
)

OUT_DIR = 'atlas_animation'
STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]


def build_hamiltonians(n):
    """Build pairwise Hamiltonians for V = -u_ij^n potential."""
    n_sym = nsimplify(n, rational=False)
    H12 = T1 + T2 - u12**n_sym
    H13 = T1 + T3 - u13**n_sym
    H23 = T2 + T3 - u23**n_sym
    return H12, H13, H23


def build_algebra(H12, H13, H23, max_level=3):
    """
    Compute Poisson bracket generators through max_level.
    Returns (exprs, levels, evaluate_fn).
    """
    all_exprs = [H12, H13, H23]
    all_names = ["H12", "H13", "H23"]
    all_levels = [0, 0, 0]
    computed_pairs = {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}

    for short, i, j in [("K1", 0, 1), ("K2", 0, 2), ("K3", 1, 2)]:
        expr = simplify_generator(
            poisson_bracket(all_exprs[i], all_exprs[j]))
        all_exprs.append(expr)
        all_names.append(short)
        all_levels.append(1)

    for level in range(2, max_level + 1):
        frontier = [i for i, lv in enumerate(all_levels) if lv == level - 1]
        n_existing = len(all_exprs)
        new_exprs = []
        new_names = []

        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)
                if all_exprs[i] == 0 or all_exprs[j] == 0:
                    continue
                expr = simplify_generator(
                    poisson_bracket(all_exprs[i], all_exprs[j]))
                new_exprs.append(expr)
                new_names.append(f"{{{all_names[i]},{all_names[j]}}}")

        for expr, name in zip(new_exprs, new_names):
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(level)

        n_zero = sum(1 for e in new_exprs if e == 0)
        print(f"      Level {level}: {len(new_exprs)} brackets ({n_zero} zero)")

    nonzero_mask = [i for i, e in enumerate(all_exprs) if e != 0]
    exprs = [all_exprs[i] for i in nonzero_mask]
    levels = [all_levels[i] for i in nonzero_mask]

    print(f"      Total: {len(all_exprs)} generators, {len(exprs)} non-zero")
    print(f"      Compiling evaluator...", flush=True)
    evaluate = lambdify_generators(exprs)

    return exprs, levels, evaluate


def sample_local(positions, n_samples, epsilon, rng,
                 mom_range=0.5, min_sep=0.1):
    """Generate local phase-space samples around a configuration."""
    base_q = positions.flatten()
    Z_qp = np.zeros((n_samples, 12))
    Z_u = np.zeros((n_samples, 3))
    accepted = 0

    while accepted < n_samples:
        q = base_q + rng.randn(6) * epsilon
        p = rng.randn(6) * mom_range

        dx12 = q[0] - q[2]; dy12 = q[1] - q[3]
        dx13 = q[0] - q[4]; dy13 = q[1] - q[5]
        dx23 = q[2] - q[4]; dy23 = q[3] - q[5]

        r12 = np.sqrt(dx12**2 + dy12**2)
        r13 = np.sqrt(dx13**2 + dy13**2)
        r23 = np.sqrt(dx23**2 + dy23**2)

        if min(r12, r13, r23) < min_sep:
            continue

        Z_qp[accepted, :6] = q
        Z_qp[accepted, 6:] = p
        Z_u[accepted] = [1.0 / r12, 1.0 / r13, 1.0 / r23]
        accepted += 1

    return Z_qp, Z_u


def shape_to_positions(mu, phi, scale=1.0):
    """Convert (mu, phi) shape parameters to body positions."""
    positions = np.zeros((3, 2))
    positions[0] = [0.0, 0.0]
    positions[1] = [scale, 0.0]
    positions[2] = [mu * scale * np.cos(phi), mu * scale * np.sin(phi)]
    return positions


def compute_gap_ratio(singular_values):
    """Compute the maximum gap ratio in the singular value spectrum."""
    s = singular_values
    if len(s) < 2:
        return 1.0
    above_noise = s[s > 1e-10 * s[0]]
    if len(above_noise) < 2:
        return s[0] / max(s[-1], 1e-300)
    ratios = above_noise[:-1] / above_noise[1:]
    return float(np.max(ratios))


def compute_atlas_for_exponent(n, evaluate, mu_vals, phi_vals,
                               n_samples=200, epsilon=0.01, seed=42):
    """
    Evaluate gap ratio over shape space for a given compiled algebra.
    Returns gap_map of shape (len(mu_vals), len(phi_vals)).
    """
    rng = np.random.RandomState(seed)
    gap_map = np.zeros((len(mu_vals), len(phi_vals)))

    for i, mu in enumerate(mu_vals):
        for j, phi in enumerate(phi_vals):
            positions = shape_to_positions(mu, phi)

            r_min = min(
                np.linalg.norm(positions[0] - positions[1]),
                np.linalg.norm(positions[0] - positions[2]),
                np.linalg.norm(positions[1] - positions[2]),
            )
            if r_min < 0.05:
                gap_map[i, j] = 1.0
                continue

            eps_local = min(epsilon, 0.1 * r_min)
            try:
                Z_qp, Z_u = sample_local(positions, n_samples, eps_local, rng)
                matrix = evaluate(Z_qp, Z_u)

                norms = np.linalg.norm(matrix, axis=0)
                norms[norms < 1e-15] = 1.0
                matrix = matrix / norms

                _, S, _ = svd(matrix, full_matrices=False)
                gap_map[i, j] = compute_gap_ratio(S)
            except Exception:
                gap_map[i, j] = 1.0

    return gap_map


def render_frame(gap_map, mu_vals, phi_vals, n, frame_idx, out_dir):
    """Render a single atlas frame in the teaser style."""
    lg = np.log10(np.clip(gap_map, 1, None))

    phi_deg = phi_vals * 180 / np.pi
    phi_max = 170
    phi_mask = phi_deg <= phi_max
    phi_trim = phi_deg[phi_mask]
    lg_trim = lg[:, phi_mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(phi_trim, mu_vals, lg_trim,
                       cmap='inferno', shading='auto',
                       vmin=0, vmax=9)

    lx = 60.0
    ly = 1.0
    ax.plot(lx, ly, '*', color='cyan', markersize=14,
            markeredgecolor='black', markeredgewidth=0.8, zorder=5)
    ax.annotate('Lagrange', (lx, ly), textcoords='offset points',
                xytext=(10, -4), color='cyan', fontsize=10,
                fontweight='bold', path_effects=STROKE)

    ax.set_xlim(phi_trim[0], phi_max)
    ax.set_xlabel(r'$\phi$ (deg)', fontsize=13)
    ax.set_ylabel(r'$\mu$', fontsize=13)

    if n == int(n):
        n_str = f'{int(n)}'
    else:
        n_str = f'{n:.2f}'
    ax.set_title(
        r'$\log_{10}$(gap ratio) $-$ $1/r^{' + n_str + r'}$',
        fontsize=14)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$\log_{10}$(gap ratio)', fontsize=12)

    plt.tight_layout()
    path = os.path.join(out_dir, f'frame_{frame_idx:03d}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return path


def stitch_gif(frame_paths, out_path, duration_ms=500):
    """Stitch PNG frames into an animated GIF using Pillow."""
    from PIL import Image

    frames = [Image.open(p) for p in frame_paths]
    forward = list(frames)
    backward = list(reversed(frames[1:-1]))
    all_frames = forward + backward

    all_frames[0].save(
        out_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"\n  Animation saved: {out_path}")
    print(f"  {len(all_frames)} frames, {duration_ms}ms per frame "
          f"({len(all_frames) * duration_ms / 1000:.1f}s loop)")


def main():
    parser = argparse.ArgumentParser(
        description='Animate stability atlas over 1/r^n potential sweep')
    parser.add_argument('--frames', type=int, default=11,
                        help='Number of exponent values (default: 11)')
    parser.add_argument('--grid', type=int, default=50,
                        help='Grid resolution NxN (default: 50)')
    parser.add_argument('--samples', type=int, default=200,
                        help='Phase-space samples per grid point')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Local ball radius')
    parser.add_argument('--n-min', type=float, default=1.0,
                        help='Minimum exponent')
    parser.add_argument('--n-max', type=float, default=2.0,
                        help='Maximum exponent')
    parser.add_argument('--level', type=int, default=3,
                        help='Max bracket level')
    parser.add_argument('--test', action='store_true',
                        help='Quick test: 3 frames, 20x20 grid, level 2')
    parser.add_argument('--duration', type=int, default=500,
                        help='GIF frame duration in ms')
    args = parser.parse_args()

    if args.test:
        args.frames = 3
        args.grid = 20
        args.level = 2
        args.samples = 100

    os.makedirs(OUT_DIR, exist_ok=True)

    n_values = np.linspace(args.n_min, args.n_max, args.frames)

    mu_vals = np.linspace(0.2, 3.0, args.grid)
    phi_vals = np.linspace(0.1, np.pi - 0.1, args.grid)

    print("=" * 60)
    print(f"  ATLAS ANIMATION: 1/r^n sweep")
    print(f"  n range: [{args.n_min}, {args.n_max}], {args.frames} frames")
    print(f"  Grid: {args.grid}x{args.grid}, Level: {args.level}")
    print(f"  Samples/point: {args.samples}, epsilon: {args.epsilon}")
    print("=" * 60)

    frame_paths = []
    total_t0 = time()

    for idx, n in enumerate(n_values):
        gap_file = os.path.join(OUT_DIR, f'gap_map_n{n:.4f}.npy')
        frame_file = os.path.join(OUT_DIR, f'frame_{idx:03d}.png')

        if os.path.exists(gap_file):
            gap_map = np.load(gap_file)
            if gap_map.shape == (args.grid, args.grid):
                print(f"\n--- Frame {idx+1}/{len(n_values)}: n = {n:.4f} "
                      f"[CACHED] ---")
                path = render_frame(gap_map, mu_vals, phi_vals,
                                    n, idx, OUT_DIR)
                frame_paths.append(path)
                continue

        print(f"\n--- Frame {idx+1}/{len(n_values)}: n = {n:.4f} ---")
        frame_t0 = time()

        print(f"    Building Hamiltonians for V = -u^{n:.4f}...")
        H12, H13, H23 = build_hamiltonians(n)

        print(f"    Computing Poisson algebra (level {args.level})...")
        t0 = time()
        exprs, levels, evaluate = build_algebra(
            H12, H13, H23, max_level=args.level)
        algebra_time = time() - t0
        print(f"    Algebra: {algebra_time:.1f}s")

        print(f"    Computing atlas ({args.grid}x{args.grid})...",
              flush=True)
        t0 = time()
        gap_map = compute_atlas_for_exponent(
            n, evaluate, mu_vals, phi_vals,
            n_samples=args.samples, epsilon=args.epsilon)
        atlas_time = time() - t0
        print(f"    Atlas: {atlas_time:.1f}s")

        np.save(gap_file, gap_map)

        path = render_frame(gap_map, mu_vals, phi_vals, n, idx, OUT_DIR)
        frame_paths.append(path)

        frame_time = time() - frame_t0
        remaining = frame_time * (len(n_values) - idx - 1)
        print(f"    Frame total: {frame_time:.1f}s "
              f"(est. remaining: {remaining/60:.1f} min)")

    np.save(os.path.join(OUT_DIR, 'mu_vals.npy'), mu_vals)
    np.save(os.path.join(OUT_DIR, 'phi_vals.npy'), phi_vals)
    np.save(os.path.join(OUT_DIR, 'n_values.npy'), n_values)

    print(f"\n  Total compute time: {(time() - total_t0)/60:.1f} min")

    gif_path = os.path.join(OUT_DIR, 'atlas_sweep_1r_to_1r2.gif')
    stitch_gif(frame_paths, gif_path, duration_ms=args.duration)


if __name__ == '__main__':
    main()
