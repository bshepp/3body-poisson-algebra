#!/usr/bin/env python3
"""
High-Resolution Lagrange Region Scan
======================================
Focused 1000x1000 grid scan centered on the equilateral (Lagrange) point
at epsilon=2e-4 for the 1/r^2 (Calogero-Moser) potential.

Region: mu=[0.3, 2.0], phi=[15deg, 105deg]
This captures the concentric ring structure visible in the multi-epsilon atlas.

Usage:
    python hires_lagrange_scan.py              # run scan
    python hires_lagrange_scan.py --test       # timing test (1 row only)
    python hires_lagrange_scan.py --render     # render from existing data
"""

import os
import sys
import json
import argparse
import numpy as np
from numpy.linalg import svd
from time import time, strftime
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

GRID_N = 300
N_SAMPLES = 400
LEVEL = 3
EPSILON = 2e-4

MU_RANGE = (0.3, 2.0)
PHI_RANGE_DEG = (15.0, 105.0)
PHI_RANGE = (PHI_RANGE_DEG[0] * np.pi / 180, PHI_RANGE_DEG[1] * np.pi / 180)

OUT_DIR = os.path.join('atlas_output_hires', 'lagrange_hires')
POTENTIAL = '1/r2'


def load_checkpoint():
    cp_file = os.path.join(OUT_DIR, 'checkpoint.json')
    if os.path.exists(cp_file):
        with open(cp_file) as f:
            return json.load(f)
    return None


def save_checkpoint(completed_rows, n_generators):
    cp_file = os.path.join(OUT_DIR, 'checkpoint.json')
    with open(cp_file, 'w') as f:
        json.dump({
            'completed_rows': completed_rows,
            'n_generators': n_generators,
            'epsilon': EPSILON,
            'timestamp': strftime('%Y-%m-%d %H:%M:%S'),
        }, f)


def flush_arrays(mu_vals, phi_vals, rank_map, gap_map, sv_spectra):
    np.save(os.path.join(OUT_DIR, 'mu_vals.npy'), mu_vals)
    np.save(os.path.join(OUT_DIR, 'phi_vals.npy'), phi_vals)
    np.save(os.path.join(OUT_DIR, 'rank_map.npy'), rank_map)
    np.save(os.path.join(OUT_DIR, 'gap_map.npy'), gap_map)
    np.save(os.path.join(OUT_DIR, 'sv_spectra.npy'), sv_spectra)


def run_scan(test_mode=False):
    from stability_atlas import AtlasConfig, PoissonAlgebra, ShapeSpace

    os.makedirs(OUT_DIR, exist_ok=True)

    grid_n = 10 if test_mode else GRID_N

    print(f"\n{'='*70}")
    print(f"  HIGH-RES LAGRANGE SCAN: 1/r^2 (Calogero-Moser)")
    print(f"  Grid: {grid_n}x{grid_n} = {grid_n**2} points")
    print(f"  Region: mu=[{MU_RANGE[0]}, {MU_RANGE[1]}], "
          f"phi=[{PHI_RANGE_DEG[0]}, {PHI_RANGE_DEG[1]}] deg")
    print(f"  Epsilon: {EPSILON:.0e}")
    print(f"  Samples/point: {N_SAMPLES}, Level: {LEVEL}")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*70}\n")

    # Build algebra
    config = AtlasConfig(
        potential_type=POTENTIAL,
        max_level=LEVEL,
        n_phase_samples=N_SAMPLES,
        epsilon=EPSILON,
        svd_gap_threshold=1e4,
    )
    print(f"  Building symbolic algebra...")
    t_build = time()
    algebra = PoissonAlgebra(config)
    n_gen = algebra._n_generators
    build_time = time() - t_build
    print(f"  Algebra ready ({build_time:.1f}s, {n_gen} generators)\n")

    mu_vals = np.linspace(MU_RANGE[0], MU_RANGE[1], grid_n)
    phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], grid_n)
    total = grid_n * grid_n

    cp = load_checkpoint() if not test_mode else None
    start_row = cp['completed_rows'] if cp and cp.get('completed_rows', 0) < grid_n else 0

    if start_row > 0 and not test_mode:
        rank_map = np.load(os.path.join(OUT_DIR, 'rank_map.npy'))
        gap_map = np.load(os.path.join(OUT_DIR, 'gap_map.npy'))
        sv_spectra = np.load(os.path.join(OUT_DIR, 'sv_spectra.npy'))
        print(f"  Resuming from row {start_row} ({start_row * grid_n} points done)")
    else:
        rank_map = np.zeros((grid_n, grid_n), dtype=int)
        gap_map = np.zeros((grid_n, grid_n))
        sv_spectra = np.zeros((grid_n, grid_n, n_gen), dtype=np.float64)

    with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
        json.dump({
            'potential': POTENTIAL,
            'grid_n': grid_n,
            'epsilon': EPSILON,
            'n_samples': N_SAMPLES,
            'level': LEVEL,
            'n_generators': n_gen,
            'mu_range': list(MU_RANGE),
            'phi_range': list(PHI_RANGE),
            'phi_range_deg': list(PHI_RANGE_DEG),
            'test_mode': test_mode,
        }, f, indent=2)

    t_scan_start = time()

    for i in range(start_row, grid_n):
        t_row_start = time()
        mu = mu_vals[i]

        for j in range(grid_n):
            phi_val = phi_vals[j]
            positions = ShapeSpace.shape_to_positions(mu, phi_val)

            try:
                rank, svs, info = algebra.compute_rank_at_configuration(
                    positions, LEVEL, epsilon=EPSILON
                )
                rank_map[i, j] = rank
                gap_map[i, j] = info['max_gap_ratio']
                sv_spectra[i, j, :len(svs)] = svs
            except Exception as e:
                rank_map[i, j] = -1
                gap_map[i, j] = 0
                print(f"  WARN: Failed at ({i},{j}) mu={mu:.3f} "
                      f"phi={phi_val*180/np.pi:.1f}: {e}")

        if not test_mode:
            flush_arrays(mu_vals, phi_vals, rank_map, gap_map, sv_spectra)
            save_checkpoint(i + 1, n_gen)

        done = (i + 1) * grid_n
        elapsed = time() - t_scan_start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        row_time = time() - t_row_start

        print(f"  Row {i+1:4d}/{grid_n}  mu={mu:.4f}  "
              f"[{done:7d}/{total}]  "
              f"row={row_time:.1f}s  "
              f"ETA={remaining/60:.0f}m ({remaining/3600:.1f}h)  "
              f"ranks=[{rank_map[i,:].min()},{rank_map[i,:].max()}]  "
              f"gap=[{gap_map[i,:].min():.1e},{gap_map[i,:].max():.1e}]",
              flush=True)

        if test_mode and i == 0:
            per_point = row_time / grid_n
            est_1000 = per_point * 1000 * 1000
            print(f"\n  === TIMING ESTIMATE ===")
            print(f"  Per point: {per_point:.2f}s")
            print(f"  1000x1000 estimate: {est_1000:.0f}s = {est_1000/3600:.1f}h = {est_1000/86400:.1f}d")
            print(f"  500x500 estimate: {per_point*500*500/3600:.1f}h")
            print(f"  300x300 estimate: {per_point*300*300/3600:.1f}h")
            print(f"  200x200 estimate: {per_point*200*200/3600:.1f}h")
            return

    if not test_mode:
        total_time = time() - t_scan_start
        print(f"\n  Scan complete: {total_time:.0f}s ({total_time/60:.1f}m = {total_time/3600:.1f}h)")
        print(f"  Rank range: [{rank_map.min()}, {rank_map.max()}]")
        print(f"  Gap range:  [{gap_map.min():.2e}, {gap_map.max():.2e}]")


def run_render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rm = np.load(os.path.join(OUT_DIR, 'rank_map.npy'))
    gm = np.load(os.path.join(OUT_DIR, 'gap_map.npy'))
    mu = np.load(os.path.join(OUT_DIR, 'mu_vals.npy'))
    phi = np.load(os.path.join(OUT_DIR, 'phi_vals.npy'))
    lg = np.log10(np.clip(gm, 1, None))

    grid_n = rm.shape[0]
    print(f"  Data: {grid_n}x{grid_n}, rank=[{rm.min()}, {rm.max()}]")
    print(f"  Unique ranks: {sorted(set(rm.flatten()))}")

    # Single panel rank map (super high res)
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    im = ax.pcolormesh(phi * 180 / np.pi, mu, rm, cmap='RdYlGn',
                       vmin=rm.min(), vmax=rm.max(), shading='auto')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Rank', fontsize=14)
    ax.plot(60, 1.0, '*', color='white', markersize=22,
            markeredgecolor='black', markeredgewidth=1.5, zorder=15)
    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=15)
    ax.set_ylabel(r'$\mu$', fontsize=15)
    ax.set_title(f'1/r\u00b2 (Calogero-Moser) \u2014 \u03b5=2\u00d710\u207b\u2074 \u2014 '
                 f'{grid_n}\u00d7{grid_n} Lagrange Region\n'
                 f'Rank Map', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'rank_map_{grid_n}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved rank_map_{grid_n}.png')

    # Gap map
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    im = ax.pcolormesh(phi * 180 / np.pi, mu, lg, cmap='inferno',
                       vmin=0, vmax=lg.max(), shading='auto')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('log\u2081\u2080(gap ratio)', fontsize=14)
    ax.plot(60, 1.0, '*', color='cyan', markersize=22,
            markeredgecolor='white', markeredgewidth=1.5, zorder=15)
    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=15)
    ax.set_ylabel(r'$\mu$', fontsize=15)
    ax.set_title(f'1/r\u00b2 (Calogero-Moser) \u2014 \u03b5=2\u00d710\u207b\u2074 \u2014 '
                 f'{grid_n}\u00d7{grid_n} Lagrange Region\n'
                 f'Gap Ratio Landscape', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'gap_map_{grid_n}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved gap_map_{grid_n}.png')

    # Combined side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 11), facecolor='white')

    im1 = ax1.pcolormesh(phi * 180 / np.pi, mu, rm, cmap='RdYlGn',
                         vmin=rm.min(), vmax=rm.max(), shading='auto')
    fig.colorbar(im1, ax=ax1, pad=0.02, label='Rank')
    ax1.plot(60, 1.0, '*', color='white', markersize=18,
             markeredgecolor='black', markeredgewidth=1.5, zorder=15)
    ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax1.set_ylabel(r'$\mu$', fontsize=13)
    ax1.set_title('Rank Map', fontsize=14, fontweight='bold')

    im2 = ax2.pcolormesh(phi * 180 / np.pi, mu, lg, cmap='inferno',
                         vmin=0, vmax=lg.max(), shading='auto')
    fig.colorbar(im2, ax=ax2, pad=0.02, label='log\u2081\u2080(gap ratio)')
    ax2.plot(60, 1.0, '*', color='cyan', markersize=18,
             markeredgecolor='white', markeredgewidth=1.5, zorder=15)
    ax2.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
    ax2.set_ylabel(r'$\mu$', fontsize=13)
    ax2.set_title('Gap Ratio Landscape', fontsize=14, fontweight='bold')

    fig.suptitle(f'1/r\u00b2 (Calogero-Moser) \u2014 \u03b5=2\u00d710\u207b\u2074 \u2014 '
                 f'{grid_n}\u00d7{grid_n} Lagrange Region',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'combined_{grid_n}.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved combined_{grid_n}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run timing test (1 row of 10)')
    parser.add_argument('--render', action='store_true', help='Render from existing data')
    parser.add_argument('--grid', type=int, default=1000, help='Grid size (default 1000)')
    args = parser.parse_args()

    if args.grid != 1000:
        GRID_N = args.grid

    if args.render:
        run_render()
    else:
        run_scan(test_mode=args.test)
        if not args.test:
            run_render()
