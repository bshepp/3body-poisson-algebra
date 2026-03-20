#!/usr/bin/env python3
"""
Multi-Epsilon Shape Sphere Atlas
==================================

Scans the 100x100 shape sphere grid at multiple epsilon values to map
how the Poisson algebra rank evolves with sampling scale.

Architecture: builds the symbolic algebra ONCE per potential (~13 min),
then loops over epsilon values, evaluating the grid at each.  Row-level
checkpointing ensures a crash loses at most 100 grid points (~2 min).

After scanning, computes derived analysis (rank-drop onset, differential
rank-drop, gap sensitivity) and generates animations.

Usage:
    python multi_epsilon_atlas.py                          # scan all
    python multi_epsilon_atlas.py scan                     # scan only
    python multi_epsilon_atlas.py scan --potential 1/r     # one potential
    python multi_epsilon_atlas.py scan --charges 2 -1 -1   # helium Coulomb
    python multi_epsilon_atlas.py analyze                  # derived analysis
    python multi_epsilon_atlas.py animate                  # animation only
    python multi_epsilon_atlas.py all                      # scan + analyze + animate
"""

import os
import sys
import json
import signal
import hashlib
import argparse
import faulthandler
import numpy as np
from numpy.linalg import svd
from time import time, strftime
from pathlib import Path
from multiprocessing import Pool, TimeoutError as MPTimeoutError, cpu_count

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
faulthandler.enable()

S3_BUCKET = os.environ.get("S3_BUCKET", "")

def s3_sync(local_dir, s3_prefix=""):
    """If S3_BUCKET is set, sync local dir to S3 (non-blocking best-effort)."""
    if not S3_BUCKET:
        return
    import subprocess
    dest = f"s3://{S3_BUCKET}/{s3_prefix}" if s3_prefix else f"s3://{S3_BUCKET}/{local_dir}"
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, dest,
             "--exclude", "*.html", "--exclude", "*.png"],
            capture_output=True, timeout=120
        )
    except Exception as e:
        print(f"  [S3 sync warning: {e}]", flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POT_DIR = {'1/r': '1_r', '1/r2': '1_r2', 'harmonic': 'harmonic'}
POT_LABEL = {
    '1/r':      '1/r (Newton)',
    '1/r2':     '1/r^2 (Calogero-Moser)',
    'harmonic': 'r^2 (Harmonic)',
}

HIRES_DIR = 'atlas_output_hires'
GRID_N = 100
N_SAMPLES = 400
LEVEL = 3
MU_RANGE = (0.2, 3.0)
PHI_RANGE = (0.1, np.pi - 0.1)

EPSILONS = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]

SINGULAR_POTENTIALS = ['1/r', '1/r2']


def charges_dir_tag(charges, potential_type='1/r'):
    """Filesystem-safe tag, e.g. 'coulomb_+2_-1_-1' or 'coulomb_1r2_+2_-1_-1'."""
    if charges is None:
        return None
    charge_part = "_".join(f"{c:+d}" for c in charges)
    if potential_type == '1/r':
        return f"coulomb_{charge_part}"
    pot_slug = POT_DIR.get(potential_type, potential_type.replace('/', ''))
    return f"coulomb_{pot_slug}_{charge_part}"


def charges_label(charges):
    """Human-readable label for a charge configuration."""
    if charges is None:
        return None
    return "Coulomb (" + ", ".join(f"{c:+d}" for c in charges) + ")"


def eps_tag(eps):
    """Filesystem-safe epsilon label, e.g. 'eps_5e-3'."""
    return f"eps_{eps:.0e}".replace("+", "")


def pot_dir_key(potential_type, charges=None):
    """Directory name for a potential+charges combination."""
    if charges is not None:
        return charges_dir_tag(charges, potential_type)
    return POT_DIR[potential_type]


def pot_label_key(potential_type, charges=None):
    """Human-readable label for a potential+charges combination."""
    if charges is not None:
        pot_name = POT_LABEL.get(potential_type, potential_type)
        return f"{pot_name} {charges_label(charges)}"
    return POT_LABEL[potential_type]


def eps_dir(potential_type, eps, charges=None):
    """Output directory for a given potential, epsilon, and optional charges."""
    base = pot_dir_key(potential_type, charges)
    if eps == 5e-3:
        return os.path.join(HIRES_DIR, base)
    return os.path.join(HIRES_DIR, base, eps_tag(eps))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def load_checkpoint(out_dir):
    cp_file = os.path.join(out_dir, 'checkpoint.json')
    if os.path.exists(cp_file):
        with open(cp_file) as f:
            return json.load(f)
    return None


def save_checkpoint(out_dir, completed_rows, n_generators, eps):
    cp_file = os.path.join(out_dir, 'checkpoint.json')
    with open(cp_file, 'w') as f:
        json.dump({
            'completed_rows': completed_rows,
            'n_generators': n_generators,
            'epsilon': eps,
            'timestamp': strftime('%Y-%m-%d %H:%M:%S'),
        }, f)


def save_checkpoint_atomic(out_dir, completed_rows, n_generators, eps,
                           extra=None):
    """Write checkpoint via tmp + os.replace to survive spot termination."""
    cp_file = os.path.join(out_dir, 'checkpoint.json')
    tmp_file = cp_file + '.tmp'
    data = {
        'completed_rows': completed_rows,
        'n_generators': n_generators,
        'epsilon': eps,
        'timestamp': strftime('%Y-%m-%d %H:%M:%S'),
    }
    if extra:
        data.update(extra)
    with open(tmp_file, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, cp_file)


def flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map, sv_spectra):
    np.save(os.path.join(out_dir, 'mu_vals.npy'), mu_vals)
    np.save(os.path.join(out_dir, 'phi_vals.npy'), phi_vals)
    np.save(os.path.join(out_dir, 'rank_map.npy'), rank_map)
    np.save(os.path.join(out_dir, 'gap_map.npy'), gap_map)
    np.save(os.path.join(out_dir, 'sv_spectra.npy'), sv_spectra)


# ---------------------------------------------------------------------------
# Grid scan: build once, scan at multiple epsilons
# ---------------------------------------------------------------------------
def run_multi_epsilon_scan(potential_type, epsilons=None, charges=None):
    from stability_atlas import AtlasConfig, PoissonAlgebra, ShapeSpace

    if epsilons is None:
        epsilons = EPSILONS

    label = pot_label_key(potential_type, charges)
    eps_to_run = []
    for eps in epsilons:
        d = eps_dir(potential_type, eps, charges)
        cp = load_checkpoint(d) if os.path.exists(d) else None
        if cp is not None and cp.get('completed_rows', 0) >= GRID_N:
            print(f"  [{label}] eps={eps:.0e}: COMPLETE (skipping)")
        else:
            done = cp['completed_rows'] if cp else 0
            eps_to_run.append((eps, done))

    if not eps_to_run:
        print(f"  [{label}] All epsilon scans already complete!")
        return

    print(f"\n{'='*70}")
    print(f"  MULTI-EPSILON SCAN: {label}")
    if charges is not None:
        print(f"  Charges: {charges}")
    print(f"  Grid: {GRID_N}x{GRID_N} = {GRID_N**2} points/epsilon")
    print(f"  Epsilons to run: {[f'{e:.0e}' for e, _ in eps_to_run]}")
    print(f"  Samples/point: {N_SAMPLES}, Level: {LEVEL}")
    print(f"{'='*70}\n")

    charges_tuple = tuple(charges) if charges is not None else None

    config = AtlasConfig(
        potential_type=potential_type,
        max_level=LEVEL,
        n_phase_samples=N_SAMPLES,
        epsilon=EPSILONS[0],
        svd_gap_threshold=1e4,
        charges=charges_tuple,
    )
    print(f"  Building symbolic algebra for {label}...")
    t_build = time()
    algebra = PoissonAlgebra(config)
    n_gen = algebra._n_generators
    print(f"  Algebra ready ({time()-t_build:.1f}s, {n_gen} generators)\n")

    mu_vals = np.linspace(MU_RANGE[0], MU_RANGE[1], GRID_N)
    phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], GRID_N)
    total = GRID_N * GRID_N

    for eps, start_row in eps_to_run:
        out_dir = eps_dir(potential_type, eps, charges)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  {'='*60}")
        print(f"  EPSILON = {eps:.0e}")
        print(f"  Output: {out_dir}/")
        print(f"  {'='*60}")

        if start_row > 0:
            rank_map = np.load(os.path.join(out_dir, 'rank_map.npy'))
            gap_map = np.load(os.path.join(out_dir, 'gap_map.npy'))
            sv_spectra = np.load(os.path.join(out_dir, 'sv_spectra.npy'))
            print(f"  Resuming from row {start_row} ({start_row * GRID_N} points done)")
        else:
            rank_map = np.zeros((GRID_N, GRID_N), dtype=int)
            gap_map = np.zeros((GRID_N, GRID_N))
            sv_spectra = np.zeros((GRID_N, GRID_N, n_gen), dtype=np.float64)

        np.save(os.path.join(out_dir, 'mu_vals.npy'), mu_vals)
        np.save(os.path.join(out_dir, 'phi_vals.npy'), phi_vals)

        config_data = {
            'potential': potential_type,
            'grid_n': GRID_N,
            'epsilon': eps,
            'n_samples': N_SAMPLES,
            'level': LEVEL,
            'n_generators': n_gen,
            'mu_range': list(MU_RANGE),
            'phi_range': list(PHI_RANGE),
        }
        if charges is not None:
            config_data['charges'] = list(charges)
        with open(os.path.join(out_dir, 'config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)

        t_scan_start = time()

        for i in range(start_row, GRID_N):
            t_row_start = time()
            mu = mu_vals[i]

            for j in range(GRID_N):
                phi_val = phi_vals[j]
                positions = ShapeSpace.shape_to_positions(mu, phi_val)

                try:
                    rank, svs, info = algebra.compute_rank_at_configuration(
                        positions, LEVEL, epsilon=eps
                    )
                    rank_map[i, j] = rank
                    gap_map[i, j] = info['max_gap_ratio']
                    sv_spectra[i, j, :len(svs)] = svs
                except Exception as e:
                    rank_map[i, j] = -1
                    gap_map[i, j] = 0
                    print(f"  WARN: Failed at ({i},{j}) mu={mu:.3f} "
                          f"phi={phi_val:.3f}: {e}")

            flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map, sv_spectra)
            save_checkpoint(out_dir, i + 1, n_gen, eps)

            if (i + 1) % 10 == 0:
                s3_sync(out_dir)

            done = (i + 1) * GRID_N
            elapsed = time() - t_scan_start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            row_time = time() - t_row_start

            print(f"  Row {i+1:3d}/{GRID_N}  mu={mu:.3f}  "
                  f"[{done:5d}/{total}]  "
                  f"row={row_time:.1f}s  "
                  f"ETA={remaining/60:.0f}m  "
                  f"ranks=[{rank_map[i,:].min()},{rank_map[i,:].max()}]  "
                  f"gap=[{gap_map[i,:].min():.1e},{gap_map[i,:].max():.1e}]",
                  flush=True)

        total_time = time() - t_scan_start
        print(f"\n  eps={eps:.0e} complete: {total_time:.0f}s ({total_time/60:.1f}m)")
        print(f"  Rank range: [{rank_map.min()}, {rank_map.max()}]")
        print(f"  Gap range:  [{gap_map.min():.2e}, {gap_map.max():.2e}]\n")

        s3_sync(out_dir)


# ---------------------------------------------------------------------------
# Derived analysis
# ---------------------------------------------------------------------------
_POT_SLUG_TO_TYPE = {v: k for k, v in POT_DIR.items()}


def _discover_charged_configs():
    """Find all charged configurations that have scan data in HIRES_DIR.

    Returns list of (potential_type, charges) tuples.  Directory names
    follow the convention produced by charges_dir_tag():
      coulomb_+2_-1_-1          -> ('1/r', [2, -1, -1])
      coulomb_1_r2_+2_-1_-1    -> ('1/r2', [2, -1, -1])
    """
    configs = []
    if not os.path.isdir(HIRES_DIR):
        return configs
    for name in os.listdir(HIRES_DIR):
        if not name.startswith('coulomb_'):
            continue
        base = os.path.join(HIRES_DIR, name)
        if not os.path.isdir(base):
            continue
        remainder = name[len('coulomb_'):]

        potential_type = '1/r'
        for slug, ptype in _POT_SLUG_TO_TYPE.items():
            prefix = slug + '_'
            if remainder.startswith(prefix):
                potential_type = ptype
                remainder = remainder[len(prefix):]
                break

        parts = remainder.split('_')
        try:
            charges = [int(p) for p in parts]
        except ValueError:
            continue
        configs.append((potential_type, charges))
    return configs


def run_analysis():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib import cm, colors

    STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]
    out = os.path.join(HIRES_DIR, 'multi_epsilon')
    os.makedirs(out, exist_ok=True)

    def draw_isosceles(ax, mu_range, phi_range):
        phi_d = np.linspace(phi_range[0], phi_range[1], 500)
        ax.axhline(1.0, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)
        mu_c2 = 2 * np.cos(phi_d)
        m2 = (mu_c2 >= mu_range[0]) & (mu_c2 <= mu_range[1])
        ax.plot(phi_d[m2] * 180 / np.pi, mu_c2[m2],
                color='lime', linewidth=1.5, linestyle='--', alpha=0.7)
        cos_p = np.cos(phi_d)
        safe = cos_p > 0.01
        mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
        m3 = np.isfinite(mu_c3) & (mu_c3 >= mu_range[0]) & (mu_c3 <= mu_range[1])
        ax.plot(phi_d[m3] * 180 / np.pi, mu_c3[m3],
                color='magenta', linewidth=1.5, linestyle='--', alpha=0.7)

    scan_configs = []
    for pot in SINGULAR_POTENTIALS:
        scan_configs.append((pot, None))
    for pot_type, charges in _discover_charged_configs():
        scan_configs.append((pot_type, charges))

    for potential_type, charges in scan_configs:
        pot_d = pot_dir_key(potential_type, charges)
        pot_l = pot_label_key(potential_type, charges)
        print(f"\n  Analyzing {pot_l}...")

        eps_data = {}
        for eps in EPSILONS:
            d = eps_dir(potential_type, eps, charges)
            if not os.path.exists(os.path.join(d, 'rank_map.npy')):
                print(f"    Missing data for eps={eps:.0e}, skipping")
                continue
            eps_data[eps] = {
                'rank': np.load(os.path.join(d, 'rank_map.npy')),
                'gap': np.load(os.path.join(d, 'gap_map.npy')),
                'sv': np.load(os.path.join(d, 'sv_spectra.npy')),
            }

        first_avail = None
        for eps in EPSILONS:
            d = eps_dir(potential_type, eps, charges)
            if os.path.exists(os.path.join(d, 'mu_vals.npy')):
                first_avail = eps
                break
        if first_avail is None:
            print("    No data found, skipping")
            continue
        mu = np.load(os.path.join(eps_dir(potential_type, first_avail, charges), 'mu_vals.npy'))
        phi = np.load(os.path.join(eps_dir(potential_type, first_avail, charges), 'phi_vals.npy'))

        avail_eps = sorted(eps_data.keys(), reverse=True)
        if len(avail_eps) < 2:
            print("    Not enough epsilon data for analysis")
            continue

        # --- 1. Rank-drop onset map ---
        onset_map = np.full((GRID_N, GRID_N), np.nan)
        for i in range(GRID_N):
            for j in range(GRID_N):
                for eps in avail_eps:
                    if eps_data[eps]['rank'][i, j] < 116:
                        onset_map[i, j] = np.log10(eps)
                        break

        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        cmap_onset = plt.get_cmap('RdYlGn')
        has_drop = np.isfinite(onset_map)
        if has_drop.any():
            vmin_o = np.nanmin(onset_map)
            vmax_o = np.nanmax(onset_map)
        else:
            vmin_o, vmax_o = -4, -2.3

        display = np.where(has_drop, onset_map, vmax_o + 0.5)
        im = ax.pcolormesh(phi * 180 / np.pi, mu, display,
                           cmap=cmap_onset, vmin=vmin_o, vmax=vmax_o + 0.5,
                           shading='auto')
        draw_isosceles(ax, (mu[0], mu[-1]), (phi[0], phi[-1]))
        ax.plot(60, 1.0, '*', color='white', markersize=16,
                markeredgecolor='black', markeredgewidth=1, zorder=15)
        ax.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
        ax.set_ylabel(r'$\mu$', fontsize=13)
        ax.set_title(f'Rank-Drop Onset Map -- {pot_l}\n'
                     'Largest epsilon where rank first drops below 116',
                     fontsize=14)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r'$\log_{10}(\varepsilon_{\mathrm{onset}})$', fontsize=12)
        n_drops = has_drop.sum()
        ax.annotate(f'{n_drops} / {GRID_N**2} points show rank drops',
                    xy=(0.02, 0.02), xycoords='axes fraction', color='white',
                    fontsize=11, fontweight='bold', path_effects=STROKE)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'rank_drop_onset_{pot_d}.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved rank_drop_onset_{pot_d}.png")

        # --- 2. Differential rank-drop at smallest epsilon ---
        smallest_eps = min(avail_eps)
        rank_small = eps_data[smallest_eps]['rank']
        generic_floor = np.percentile(rank_small, 75)
        diff_drop = generic_floor - rank_small.astype(float)
        diff_drop = np.clip(diff_drop, 0, None)

        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        im = ax.pcolormesh(phi * 180 / np.pi, mu, diff_drop,
                           cmap='hot_r', vmin=0, vmax=max(diff_drop.max(), 1),
                           shading='auto')
        draw_isosceles(ax, (mu[0], mu[-1]), (phi[0], phi[-1]))
        ax.plot(60, 1.0, '*', color='cyan', markersize=16,
                markeredgecolor='black', markeredgewidth=1, zorder=15)
        ax.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
        ax.set_ylabel(r'$\mu$', fontsize=13)
        ax.set_title(f'Differential Rank Drop -- {pot_l}\n'
                     f'Extra ranks lost beyond generic floor '
                     f'(floor={generic_floor:.0f} at eps={smallest_eps:.0e})',
                     fontsize=14)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label('Extra rank reduction', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'differential_rank_drop_{pot_d}.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved differential_rank_drop_{pot_d}.png")

        # --- 3. Gap sensitivity: d(log gap)/d(log eps) ---
        if len(avail_eps) >= 3:
            log_eps = np.array([np.log10(e) for e in avail_eps])
            log_gaps = np.zeros((len(avail_eps), GRID_N, GRID_N))
            for k, eps in enumerate(avail_eps):
                log_gaps[k] = np.log10(np.clip(eps_data[eps]['gap'], 1, None))

            sensitivity = np.zeros((GRID_N, GRID_N))
            for i in range(GRID_N):
                for j in range(GRID_N):
                    y = log_gaps[:, i, j]
                    coeffs = np.polyfit(log_eps, y, 1)
                    sensitivity[i, j] = coeffs[0]

            fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
            vlim = max(abs(np.percentile(sensitivity, 2)),
                       abs(np.percentile(sensitivity, 98)))
            im = ax.pcolormesh(phi * 180 / np.pi, mu, sensitivity,
                               cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                               shading='auto')
            draw_isosceles(ax, (mu[0], mu[-1]), (phi[0], phi[-1]))
            ax.plot(60, 1.0, '*', color='white', markersize=16,
                    markeredgecolor='black', markeredgewidth=1, zorder=15)
            ax.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
            ax.set_ylabel(r'$\mu$', fontsize=13)
            ax.set_title(f'Gap Sensitivity -- {pot_l}\n'
                         r'$d(\log_{10}\mathrm{gap}) / d(\log_{10}\varepsilon)$',
                         fontsize=14)
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label('Sensitivity (slope)', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(out, f'gap_sensitivity_{pot_d}.png'),
                        dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    Saved gap_sensitivity_{pot_d}.png")

        # --- 4. Rank-vs-epsilon curves at notable configurations ---
        notable = {
            'Lagrange': (1.0, np.pi / 3),
            'Euler': (0.5, np.pi),
            'Right-isos': (1.0, np.pi / 2),
            'Generic-1': (0.6, 0.8),
            'Generic-2': (1.5, 1.2),
            'Generic-3': (2.0, 0.6),
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
        cmap_lines = plt.get_cmap('tab10')

        for idx, (name, (target_mu, target_phi)) in enumerate(notable.items()):
            i = np.argmin(np.abs(mu - target_mu))
            j = np.argmin(np.abs(phi - target_phi))
            color = cmap_lines(idx)

            ranks_vs_eps = []
            gaps_vs_eps = []
            for eps in avail_eps:
                ranks_vs_eps.append(eps_data[eps]['rank'][i, j])
                gaps_vs_eps.append(np.log10(max(eps_data[eps]['gap'][i, j], 1)))

            ax1.plot(avail_eps, ranks_vs_eps, 'o-', color=color, linewidth=2,
                     markersize=6, label=f'{name} ({mu[i]:.2f},{phi[j]*180/np.pi:.0f})')
            ax2.plot(avail_eps, gaps_vs_eps, 'o-', color=color, linewidth=2,
                     markersize=6, label=name)

        ax1.set_xscale('log')
        ax1.invert_xaxis()
        ax1.set_xlabel(r'$\varepsilon$', fontsize=13)
        ax1.set_ylabel('Rank', fontsize=13)
        ax1.set_title('Rank vs Epsilon', fontsize=14)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.set_xscale('log')
        ax2.invert_xaxis()
        ax2.set_xlabel(r'$\varepsilon$', fontsize=13)
        ax2.set_ylabel(r'$\log_{10}$(gap ratio)', fontsize=13)
        ax2.set_title('Gap Ratio vs Epsilon', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Multi-Epsilon Profiles -- {pot_l}', fontsize=15,
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'rank_vs_epsilon_{pot_d}.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved rank_vs_epsilon_{pot_d}.png")

        # --- 5. Summary stats ---
        summary_file = os.path.join(out, f'summary_{pot_d}.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Multi-Epsilon Atlas Summary: {pot_l}\n")
            f.write("=" * 60 + "\n\n")
            for eps in avail_eps:
                rm = eps_data[eps]['rank']
                gm = eps_data[eps]['gap']
                lg = np.log10(np.clip(gm, 1, None))
                f.write(f"eps = {eps:.0e}:\n")
                f.write(f"  Rank range: [{rm.min()}, {rm.max()}]\n")
                f.write(f"  Unique ranks: {sorted(set(rm.flatten()))}\n")
                f.write(f"  Points at rank 116: {np.sum(rm==116)} "
                        f"({100*np.sum(rm==116)/rm.size:.1f}%)\n")
                f.write(f"  Points below 116: {np.sum(rm<116)} "
                        f"({100*np.sum(rm<116)/rm.size:.1f}%)\n")
                f.write(f"  log10(gap) p5={np.percentile(lg,5):.1f} "
                        f"p50={np.percentile(lg,50):.1f} "
                        f"p95={np.percentile(lg,95):.1f}\n\n")
            f.write(f"\nRank-drop onset: {has_drop.sum()} / {GRID_N**2} points\n")
        print(f"    Saved summary_{pot_d}.txt")

    print(f"\n  All analysis saved to {out}/")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def run_animation():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib import cm, colors
    from matplotlib.animation import FuncAnimation, PillowWriter

    STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]
    out = os.path.join(HIRES_DIR, 'multi_epsilon')
    os.makedirs(out, exist_ok=True)

    def draw_isosceles(ax, mu_range, phi_range):
        phi_d = np.linspace(phi_range[0], phi_range[1], 500)
        ax.axhline(1.0, color='cyan', linewidth=1.5, linestyle='--', alpha=0.6)
        mu_c2 = 2 * np.cos(phi_d)
        m2 = (mu_c2 >= mu_range[0]) & (mu_c2 <= mu_range[1])
        ax.plot(phi_d[m2] * 180 / np.pi, mu_c2[m2],
                color='lime', linewidth=1.5, linestyle='--', alpha=0.6)
        cos_p = np.cos(phi_d)
        safe = cos_p > 0.01
        mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
        m3 = np.isfinite(mu_c3) & (mu_c3 >= mu_range[0]) & (mu_c3 <= mu_range[1])
        ax.plot(phi_d[m3] * 180 / np.pi, mu_c3[m3],
                color='magenta', linewidth=1.5, linestyle='--', alpha=0.6)

    anim_configs = []
    for pot in SINGULAR_POTENTIALS:
        anim_configs.append((pot, None))
    for pot_type, charges in _discover_charged_configs():
        anim_configs.append((pot_type, charges))

    for potential_type, charges in anim_configs:
        pot_d = pot_dir_key(potential_type, charges)
        pot_l = pot_label_key(potential_type, charges)
        print(f"\n  Animating {pot_l}...")

        eps_data = {}
        for eps in EPSILONS:
            d = eps_dir(potential_type, eps, charges)
            rm_path = os.path.join(d, 'rank_map.npy')
            if os.path.exists(rm_path):
                eps_data[eps] = {
                    'rank': np.load(rm_path),
                    'gap': np.load(os.path.join(d, 'gap_map.npy')),
                }

        first_avail = None
        for eps in EPSILONS:
            d = eps_dir(potential_type, eps, charges)
            if os.path.exists(os.path.join(d, 'mu_vals.npy')):
                first_avail = eps
                break
        if first_avail is None:
            print("    No data found, skipping")
            continue
        mu = np.load(os.path.join(eps_dir(potential_type, first_avail, charges), 'mu_vals.npy'))
        phi = np.load(os.path.join(eps_dir(potential_type, first_avail, charges), 'phi_vals.npy'))

        avail_eps = sorted(eps_data.keys(), reverse=True)
        if len(avail_eps) < 2:
            print("    Not enough data for animation")
            continue

        # --- Animated GIF: rank map evolution ---
        fig, (ax_rank, ax_gap) = plt.subplots(1, 2, figsize=(18, 7),
                                               facecolor='white')

        all_ranks = np.stack([eps_data[e]['rank'] for e in avail_eps])
        rank_min = max(all_ranks.min(), 100)
        rank_max = 118

        def animate(frame_idx):
            ax_rank.clear()
            ax_gap.clear()

            eps = avail_eps[frame_idx]
            rm = eps_data[eps]['rank']
            gm = eps_data[eps]['gap']
            lg = np.log10(np.clip(gm, 1, None))

            im_r = ax_rank.pcolormesh(phi * 180 / np.pi, mu, rm,
                                       cmap='RdYlGn', vmin=rank_min,
                                       vmax=rank_max, shading='auto')
            draw_isosceles(ax_rank, (mu[0], mu[-1]), (phi[0], phi[-1]))
            ax_rank.plot(60, 1.0, '*', color='white', markersize=14,
                         markeredgecolor='black', markeredgewidth=1, zorder=15)
            ax_rank.set_xlabel(r'$\phi$ (deg)', fontsize=12)
            ax_rank.set_ylabel(r'$\mu$', fontsize=12)
            n116 = np.sum(rm == 116)
            n_below = np.sum(rm < 116)
            ax_rank.set_title(f'Rank Map  |  rank=116: {n116}  |  '
                              f'below: {n_below}', fontsize=12)

            im_g = ax_gap.pcolormesh(phi * 180 / np.pi, mu, lg,
                                      cmap='inferno', vmin=0,
                                      vmax=9, shading='auto')
            draw_isosceles(ax_gap, (mu[0], mu[-1]), (phi[0], phi[-1]))
            ax_gap.plot(60, 1.0, '*', color='cyan', markersize=14,
                        markeredgecolor='white', markeredgewidth=1, zorder=15)
            ax_gap.set_xlabel(r'$\phi$ (deg)', fontsize=12)
            ax_gap.set_ylabel(r'$\mu$', fontsize=12)
            ax_gap.set_title(f'log10(gap)  |  median={np.median(lg):.1f}',
                             fontsize=12)

            fig.suptitle(
                f'{pot_l} -- '
                r'$\varepsilon$'
                f' = {eps:.0e}   '
                f'[frame {frame_idx+1}/{len(avail_eps)}]',
                fontsize=15, fontweight='bold', y=0.98)
            return []

        anim = FuncAnimation(fig, animate, frames=len(avail_eps),
                             interval=1500, blit=False)

        gif_path = os.path.join(out, f'epsilon_sweep_{pot_d}.gif')
        anim.save(gif_path, writer=PillowWriter(fps=1))
        plt.close()
        print(f"    Saved {gif_path}")

        # --- Static multi-panel: all epsilons side by side ---
        n_eps = len(avail_eps)
        fig, axes = plt.subplots(2, n_eps, figsize=(5 * n_eps, 12),
                                 facecolor='white')
        if n_eps == 1:
            axes = axes.reshape(2, 1)

        for k, eps in enumerate(avail_eps):
            rm = eps_data[eps]['rank']
            gm = eps_data[eps]['gap']
            lg = np.log10(np.clip(gm, 1, None))

            ax_r = axes[0, k]
            ax_r.pcolormesh(phi * 180 / np.pi, mu, rm, cmap='RdYlGn',
                            vmin=rank_min, vmax=rank_max, shading='auto')
            draw_isosceles(ax_r, (mu[0], mu[-1]), (phi[0], phi[-1]))
            ax_r.plot(60, 1.0, '*', color='white', markersize=10,
                      markeredgecolor='black', markeredgewidth=0.8, zorder=15)
            n_below = np.sum(rm < 116)
            ax_r.set_title(f'eps={eps:.0e}\nrank<116: {n_below}', fontsize=11)
            if k == 0:
                ax_r.set_ylabel(r'$\mu$', fontsize=12)
            else:
                ax_r.set_yticklabels([])

            ax_g = axes[1, k]
            ax_g.pcolormesh(phi * 180 / np.pi, mu, lg, cmap='inferno',
                            vmin=0, vmax=9, shading='auto')
            draw_isosceles(ax_g, (mu[0], mu[-1]), (phi[0], phi[-1]))
            ax_g.plot(60, 1.0, '*', color='cyan', markersize=10,
                      markeredgecolor='white', markeredgewidth=0.8, zorder=15)
            ax_g.set_xlabel(r'$\phi$ (deg)', fontsize=11)
            if k == 0:
                ax_g.set_ylabel(r'$\mu$', fontsize=12)
            else:
                ax_g.set_yticklabels([])

        fig.suptitle(f'Multi-Epsilon Atlas -- {pot_l}\n'
                     'Top: Rank maps  |  Bottom: Gap ratio landscapes',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'multi_epsilon_panels_{pot_d}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved multi_epsilon_panels_{pot_d}.png")

        # --- Static multi-panel (CLEAN: no overlay lines) ---
        fig, axes = plt.subplots(2, n_eps, figsize=(5 * n_eps, 12),
                                 facecolor='white')
        if n_eps == 1:
            axes = axes.reshape(2, 1)

        for k, eps in enumerate(avail_eps):
            rm = eps_data[eps]['rank']
            gm = eps_data[eps]['gap']
            lg = np.log10(np.clip(gm, 1, None))

            ax_r = axes[0, k]
            ax_r.pcolormesh(phi * 180 / np.pi, mu, rm, cmap='RdYlGn',
                            vmin=rank_min, vmax=rank_max, shading='auto')
            n_below = np.sum(rm < 116)
            ax_r.set_title(f'eps={eps:.0e}\nrank<116: {n_below}', fontsize=11)
            if k == 0:
                ax_r.set_ylabel(r'$\mu$', fontsize=12)
            else:
                ax_r.set_yticklabels([])

            ax_g = axes[1, k]
            ax_g.pcolormesh(phi * 180 / np.pi, mu, lg, cmap='inferno',
                            vmin=0, vmax=9, shading='auto')
            ax_g.set_xlabel(r'$\phi$ (deg)', fontsize=11)
            if k == 0:
                ax_g.set_ylabel(r'$\mu$', fontsize=12)
            else:
                ax_g.set_yticklabels([])

        fig.suptitle(f'Multi-Epsilon Atlas -- {pot_l}\n'
                     'Top: Rank maps  |  Bottom: Gap ratio landscapes',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'multi_epsilon_panels_{pot_d}_clean.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved multi_epsilon_panels_{pot_d}_clean.png")

    # --- Interactive Plotly with epsilon slider ---
    try:
        import plotly.graph_objects as go

        plotly_configs = list(anim_configs)
        for potential_type, charges in plotly_configs:
            pot_d = pot_dir_key(potential_type, charges)
            pot_l = pot_label_key(potential_type, charges)

            eps_data = {}
            for eps in EPSILONS:
                d = eps_dir(potential_type, eps, charges)
                rm_path = os.path.join(d, 'rank_map.npy')
                if os.path.exists(rm_path):
                    eps_data[eps] = {
                        'rank': np.load(rm_path),
                        'gap': np.load(os.path.join(d, 'gap_map.npy')),
                    }

            first_avail = None
            for eps in EPSILONS:
                d = eps_dir(potential_type, eps, charges)
                if os.path.exists(os.path.join(d, 'mu_vals.npy')):
                    first_avail = eps
                    break
            if first_avail is None:
                continue
            mu = np.load(os.path.join(eps_dir(potential_type, first_avail, charges), 'mu_vals.npy'))
            phi = np.load(os.path.join(eps_dir(potential_type, first_avail, charges), 'phi_vals.npy'))
            avail_eps = sorted(eps_data.keys(), reverse=True)

            if len(avail_eps) < 2:
                continue

            fig = go.Figure()
            frames = []

            for k, eps in enumerate(avail_eps):
                rm = eps_data[eps]['rank']
                n_below = int(np.sum(rm < 116))
                heatmap = go.Heatmap(
                    z=rm, x=phi * 180 / np.pi, y=mu,
                    colorscale='RdYlGn', zmin=100, zmax=118,
                    colorbar=dict(title='Rank'),
                    hovertemplate=('phi=%{x:.1f} deg<br>mu=%{y:.3f}<br>'
                                  'rank=%{z}<extra></extra>'),
                )
                if k == 0:
                    fig.add_trace(heatmap)
                frames.append(go.Frame(
                    data=[heatmap],
                    name=f'{eps:.0e}',
                    layout=go.Layout(
                        title=f'{pot_l} -- eps={eps:.0e} -- '
                              f'rank<116: {n_below}'
                    ),
                ))

            fig.frames = frames

            steps = []
            for k, eps in enumerate(avail_eps):
                step = dict(
                    method='animate',
                    args=[[f'{eps:.0e}'],
                          dict(frame=dict(duration=800, redraw=True),
                               mode='immediate')],
                    label=f'{eps:.0e}',
                )
                steps.append(step)

            sliders = [dict(
                active=0, steps=steps,
                currentvalue=dict(prefix='epsilon = '),
                pad=dict(t=50),
            )]

            fig.update_layout(
                title=f'Multi-Epsilon Rank Atlas -- {pot_l}',
                xaxis_title='phi (degrees)',
                yaxis_title='mu',
                sliders=sliders,
                width=900, height=700,
            )

            html_path = os.path.join(out, f'interactive_epsilon_{pot_d}.html')
            fig.write_html(html_path)
            print(f"    Saved {html_path}")

    except ImportError:
        print("  Plotly not available, skipping interactive HTML")

    print(f"\n  All animations saved to {out}/")


# ---------------------------------------------------------------------------
# Adaptive epsilon scan
# ---------------------------------------------------------------------------
MAX_TIERS = 8

# Module-level globals for fork-based multiprocessing (set before Pool).
# Workers inherit these via copy-on-write after fork.
_adaptive_algebra = None
_adaptive_level = LEVEL
_adaptive_eps_range = (1e-4, 5e-3)
_adaptive_n_eps = 8
_adaptive_tier_threshold = 10.0
_adaptive_max_tiers = MAX_TIERS

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    """Handle SIGTERM from AWS spot reclamation (2-min warning)."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  [SIGTERM] Spot reclamation warning -- finishing current row...",
          flush=True)


def _eval_adaptive_point(args):
    """Worker: compute adaptive rank at one grid point (fork-inherited algebra)."""
    mu, phi = args
    from stability_atlas import ShapeSpace
    positions = ShapeSpace.shape_to_positions(mu, phi)
    try:
        rank, svs, info = _adaptive_algebra.compute_adaptive_rank(
            positions, _adaptive_level,
            eps_range=_adaptive_eps_range,
            n_eps=_adaptive_n_eps,
            tier_threshold=_adaptive_tier_threshold,
            max_tiers=_adaptive_max_tiers,
        )
        tiers = info['tier_boundaries']
        return (rank, svs, info['max_gap_ratio'], info['optimal_eps'],
                info['gap_score'], info['n_eps_tested'], tiers)
    except Exception:
        return (-1, np.array([]), 0.0, 0.0, 0.0, 0, [])


def _adaptive_block_dir(base_dir, start_row, end_row):
    """Block directory for distributed row-range execution."""
    return os.path.join(base_dir, f'block_{start_row:04d}_{end_row:04d}')


def run_adaptive_scan(potential_type, charges=None,
                      eps_range=(1e-4, 5e-3), n_eps=8,
                      tier_threshold=10.0,
                      n_workers=1, start_row=0, end_row=None,
                      point_timeout=600):
    global _adaptive_algebra, _adaptive_level, _adaptive_eps_range
    global _adaptive_n_eps, _adaptive_tier_threshold, _adaptive_max_tiers
    global _shutdown_requested

    from stability_atlas import AtlasConfig, PoissonAlgebra, ShapeSpace

    if end_row is None:
        end_row = GRID_N

    label = pot_label_key(potential_type, charges)
    charges_tuple = tuple(charges) if charges is not None else None

    base = pot_dir_key(potential_type, charges)
    is_block = (start_row != 0 or end_row != GRID_N)
    if is_block:
        out_dir = _adaptive_block_dir(
            os.path.join(HIRES_DIR, base, 'adaptive'), start_row, end_row)
    else:
        out_dir = os.path.join(HIRES_DIR, base, 'adaptive')
    os.makedirs(out_dir, exist_ok=True)

    n_rows = end_row - start_row
    s3_prefix = f"{HIRES_DIR}/{base}/adaptive"
    if is_block:
        s3_prefix += f"/block_{start_row:04d}_{end_row:04d}"

    cp = load_checkpoint(out_dir)
    cp_done = cp.get('completed_rows', 0) if cp else 0
    if cp_done >= n_rows:
        print(f"  [{label}] adaptive rows {start_row}-{end_row}: "
              f"COMPLETE (skipping)")
        return

    signal.signal(signal.SIGTERM, _sigterm_handler)
    _shutdown_requested = False

    print(f"\n{'='*70}")
    print(f"  ADAPTIVE EPSILON SCAN: {label}")
    if charges is not None:
        print(f"  Charges: {charges}")
    print(f"  Rows: {start_row}-{end_row} ({n_rows} rows, "
          f"{n_rows * GRID_N} points)")
    print(f"  Grid columns: {GRID_N}")
    print(f"  Epsilon range: [{eps_range[0]:.0e}, {eps_range[1]:.0e}], "
          f"{n_eps} candidates, tier threshold={tier_threshold}")
    print(f"  Samples/point: {N_SAMPLES}, Level: {LEVEL}")
    print(f"  Workers: {n_workers}, Point timeout: {point_timeout}s")
    print(f"{'='*70}\n")

    config = AtlasConfig(
        potential_type=potential_type,
        max_level=LEVEL,
        n_phase_samples=N_SAMPLES,
        epsilon=eps_range[1],
        svd_gap_threshold=1e4,
        charges=charges_tuple,
    )
    print(f"  Building symbolic algebra for {label}...")
    t_build = time()
    algebra = PoissonAlgebra(config)
    n_gen = len(algebra._names)
    build_time = time() - t_build
    print(f"  Algebra ready ({build_time:.1f}s, {n_gen} generators)")

    _adaptive_algebra = algebra
    _adaptive_level = LEVEL
    _adaptive_eps_range = eps_range
    _adaptive_n_eps = n_eps
    _adaptive_tier_threshold = tier_threshold
    _adaptive_max_tiers = MAX_TIERS

    mu_vals = np.linspace(MU_RANGE[0], MU_RANGE[1], GRID_N)
    phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], GRID_N)
    mu_slice = mu_vals[start_row:end_row]
    total = n_rows * GRID_N

    if cp_done > 0:
        rank_map = np.load(os.path.join(out_dir, 'rank_map.npy'))
        gap_map = np.load(os.path.join(out_dir, 'gap_map.npy'))
        sv_spectra = np.load(os.path.join(out_dir, 'sv_spectra.npy'))
        optimal_eps_map = np.load(os.path.join(out_dir, 'optimal_eps_map.npy'))
        gap_score_map = np.load(os.path.join(out_dir, 'gap_score_map.npy'))
        tier_map = np.load(os.path.join(out_dir, 'tier_map.npy'))
        n_tested_map = np.load(os.path.join(out_dir, 'n_tested_map.npy'))
        print(f"  Resuming from local row {cp_done} "
              f"({cp_done * GRID_N} points done)")
    else:
        rank_map = np.zeros((n_rows, GRID_N), dtype=np.int32)
        gap_map = np.zeros((n_rows, GRID_N))
        sv_spectra = np.zeros((n_rows, GRID_N, n_gen), dtype=np.float64)
        optimal_eps_map = np.zeros((n_rows, GRID_N))
        gap_score_map = np.zeros((n_rows, GRID_N))
        tier_map = np.full((n_rows, GRID_N, MAX_TIERS, 2), -1,
                           dtype=np.float64)
        n_tested_map = np.zeros((n_rows, GRID_N), dtype=np.int32)

    np.save(os.path.join(out_dir, 'mu_vals.npy'), mu_slice)
    np.save(os.path.join(out_dir, 'phi_vals.npy'), phi_vals)

    config_data = {
        'potential': potential_type,
        'grid_n': GRID_N,
        'n_rows': n_rows,
        'start_row': start_row,
        'end_row': end_row,
        'eps_range': list(eps_range),
        'n_eps': n_eps,
        'tier_threshold': tier_threshold,
        'n_samples': N_SAMPLES,
        'level': LEVEL,
        'n_generators': n_gen,
        'n_workers': n_workers,
        'point_timeout': point_timeout,
        'mu_range': list(MU_RANGE),
        'phi_range': list(PHI_RANGE),
        'mode': 'adaptive',
    }
    if charges is not None:
        config_data['charges'] = list(charges)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2)

    use_mp = (n_workers > 1)
    pool = None
    if use_mp:
        print(f"  Starting worker pool ({n_workers} workers)...")
        pool = Pool(processes=n_workers)

    def _save_all():
        for arr_name, arr in [('rank_map', rank_map), ('gap_map', gap_map),
                              ('sv_spectra', sv_spectra),
                              ('optimal_eps_map', optimal_eps_map),
                              ('gap_score_map', gap_score_map),
                              ('tier_map', tier_map),
                              ('n_tested_map', n_tested_map)]:
            np.save(os.path.join(out_dir, f'{arr_name}.npy'), arr)

    def _unpack_result(local_i, j, result):
        rank, svs, max_gap, opt_e, g_score, n_tried, tiers = result
        rank_map[local_i, j] = rank
        gap_map[local_i, j] = max_gap
        if len(svs) > 0:
            sv_spectra[local_i, j, :len(svs)] = svs
        optimal_eps_map[local_i, j] = opt_e
        gap_score_map[local_i, j] = g_score
        n_tested_map[local_i, j] = n_tried
        for t_idx, (t_pos, t_gap) in enumerate(tiers[:MAX_TIERS]):
            tier_map[local_i, j, t_idx, 0] = t_pos
            tier_map[local_i, j, t_idx, 1] = min(t_gap, 1e16)

    t_scan_start = time()

    try:
        for local_i in range(cp_done, n_rows):
            t_row_start = time()
            global_i = start_row + local_i
            mu = mu_vals[global_i]
            n_timeout = 0

            if use_mp:
                tasks = [(mu, phi_vals[j]) for j in range(GRID_N)]
                async_results = [pool.apply_async(_eval_adaptive_point, (t,))
                                 for t in tasks]

                for j, ar in enumerate(async_results):
                    try:
                        result = ar.get(timeout=point_timeout)
                    except MPTimeoutError:
                        result = (-1, np.array([]), 0.0, 0.0, 0.0, 0, [])
                        n_timeout += 1
                    except Exception:
                        result = (-1, np.array([]), 0.0, 0.0, 0.0, 0, [])
                    _unpack_result(local_i, j, result)
            else:
                for j in range(GRID_N):
                    result = _eval_adaptive_point((mu, phi_vals[j]))
                    _unpack_result(local_i, j, result)

            _save_all()
            save_checkpoint_atomic(out_dir, local_i + 1, n_gen, -1, extra={
                'start_row': start_row,
                'end_row': end_row,
                'n_workers': n_workers,
                'global_row': global_i + 1,
            })

            if (local_i + 1) % 5 == 0:
                s3_sync(out_dir, s3_prefix)

            done = (local_i + 1) * GRID_N
            elapsed = time() - t_scan_start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            row_time = time() - t_row_start

            avg_eps_tested = n_tested_map[local_i, :].mean()
            timeout_str = f"  timeouts={n_timeout}" if n_timeout else ""
            print(f"  Row {local_i+1:3d}/{n_rows}  mu={mu:.3f}  "
                  f"[{done:5d}/{total}]  "
                  f"row={row_time:.1f}s  "
                  f"ETA={remaining/60:.0f}m  "
                  f"ranks=[{rank_map[local_i,:].min()},"
                  f"{rank_map[local_i,:].max()}]  "
                  f"gap=[{gap_map[local_i,:].min():.1e},"
                  f"{gap_map[local_i,:].max():.1e}]  "
                  f"avg_eps_tried={avg_eps_tested:.1f}"
                  f"{timeout_str}",
                  flush=True)

            if _shutdown_requested:
                print(f"  [SHUTDOWN] Saved through row {local_i+1}/{n_rows}. "
                      f"Safe to resume.", flush=True)
                s3_sync(out_dir, s3_prefix)
                break

    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    total_time = time() - t_scan_start
    completed_all = (not _shutdown_requested
                     and cp_done + (local_i - cp_done + 1) >= n_rows)

    print(f"\n  Adaptive scan {'complete' if completed_all else 'interrupted'}: "
          f"{total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Rank range: [{rank_map[rank_map >= 0].min() if (rank_map >= 0).any() else -1}, "
          f"{rank_map.max()}]")
    print(f"  Gap range:  [{gap_map.min():.2e}, {gap_map.max():.2e}]")
    valid_eps = optimal_eps_map[optimal_eps_map > 0]
    if valid_eps.size > 0:
        print(f"  Optimal eps range: [{valid_eps.min():.2e}, {valid_eps.max():.2e}]")
    print(f"  Gap score range: [{gap_score_map.min():.2f}, "
          f"{gap_score_map.max():.2f}]")
    n_failed = (rank_map == -1).sum()
    if n_failed > 0:
        print(f"  WARNING: {n_failed} points failed ({100*n_failed/total:.1f}%)")
    print()

    s3_sync(out_dir, s3_prefix)


# ---------------------------------------------------------------------------
# Adaptive analysis & visualization
# ---------------------------------------------------------------------------
def run_adaptive_analysis(potential_type=None, charges=None):
    """Generate visualizations from adaptive scan data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    def draw_isosceles(ax, mu_range, phi_range):
        phi_d = np.linspace(phi_range[0], phi_range[1], 500)
        ax.axhline(1.0, color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)
        mu_c2 = 2 * np.cos(phi_d)
        m2 = (mu_c2 >= mu_range[0]) & (mu_c2 <= mu_range[1])
        ax.plot(phi_d[m2] * 180 / np.pi, mu_c2[m2],
                color='lime', linewidth=1.5, linestyle='--', alpha=0.7)
        cos_p = np.cos(phi_d)
        safe = cos_p > 0.01
        mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
        m3 = np.isfinite(mu_c3) & (mu_c3 >= mu_range[0]) & (mu_c3 <= mu_range[1])
        ax.plot(phi_d[m3] * 180 / np.pi, mu_c3[m3],
                color='magenta', linewidth=1.5, linestyle='--', alpha=0.7)

    configs_to_plot = []
    if potential_type is not None:
        configs_to_plot.append((potential_type, charges))
    else:
        for pot in SINGULAR_POTENTIALS:
            base = os.path.join(HIRES_DIR, POT_DIR[pot], 'adaptive')
            if os.path.isdir(base):
                configs_to_plot.append((pot, None))
        for pt, ch in _discover_charged_configs():
            base = os.path.join(HIRES_DIR, pot_dir_key(pt, ch), 'adaptive')
            if os.path.isdir(base):
                configs_to_plot.append((pt, ch))

    if not configs_to_plot:
        print("  No adaptive scan data found.")
        return

    for pot_type, ch in configs_to_plot:
        pot_d = pot_dir_key(pot_type, ch)
        pot_l = pot_label_key(pot_type, ch)
        ad_dir = os.path.join(HIRES_DIR, pot_d, 'adaptive')

        if not os.path.isfile(os.path.join(ad_dir, 'rank_map.npy')):
            print(f"  No adaptive data for {pot_l}")
            continue

        print(f"\n  Visualizing adaptive: {pot_l}...")

        rank = np.load(os.path.join(ad_dir, 'rank_map.npy'))
        gap = np.load(os.path.join(ad_dir, 'gap_map.npy'))
        opt_eps = np.load(os.path.join(ad_dir, 'optimal_eps_map.npy'))
        gap_score = np.load(os.path.join(ad_dir, 'gap_score_map.npy'))
        tier_data = np.load(os.path.join(ad_dir, 'tier_map.npy'))
        mu = np.load(os.path.join(ad_dir, 'mu_vals.npy'))
        phi = np.load(os.path.join(ad_dir, 'phi_vals.npy'))

        phi_deg = np.degrees(phi)
        out = os.path.join(ad_dir, 'plots')
        os.makedirs(out, exist_ok=True)

        # --- 1. Four-panel overview ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Adaptive Atlas: {pot_l}', fontsize=16, fontweight='bold')

        # Panel 1: optimal epsilon
        log_eps = np.log10(np.where(opt_eps > 0, opt_eps, 1e-5))
        im0 = axes[0, 0].pcolormesh(phi_deg, mu, log_eps, cmap='viridis',
                                      shading='auto')
        axes[0, 0].set_title('Optimal log₁₀(ε)')
        axes[0, 0].set_ylabel('μ = r₁₃/r₁₂')
        plt.colorbar(im0, ax=axes[0, 0])
        draw_isosceles(axes[0, 0], MU_RANGE, PHI_RANGE)

        # Panel 2: gap score
        im1 = axes[0, 1].pcolormesh(phi_deg, mu, gap_score, cmap='inferno',
                                      shading='auto')
        axes[0, 1].set_title('Gap Score (information content)')
        plt.colorbar(im1, ax=axes[0, 1])
        draw_isosceles(axes[0, 1], MU_RANGE, PHI_RANGE)

        # Panel 3: log gap ratio
        log_gap = np.log10(np.where(gap > 1, gap, 1))
        im2 = axes[1, 0].pcolormesh(phi_deg, mu, log_gap, cmap='hot',
                                      shading='auto')
        axes[1, 0].set_title('log₁₀(best gap ratio)')
        axes[1, 0].set_xlabel('φ (degrees)')
        axes[1, 0].set_ylabel('μ = r₁₃/r₁₂')
        plt.colorbar(im2, ax=axes[1, 0])
        draw_isosceles(axes[1, 0], MU_RANGE, PHI_RANGE)

        # Panel 4: number of tiers detected
        n_tiers = np.sum(tier_data[:, :, :, 0] >= 0, axis=2)
        im3 = axes[1, 1].pcolormesh(phi_deg, mu, n_tiers, cmap='YlOrRd',
                                      shading='auto', vmin=1, vmax=6)
        axes[1, 1].set_title('Number of tier boundaries')
        axes[1, 1].set_xlabel('φ (degrees)')
        plt.colorbar(im3, ax=axes[1, 1])
        draw_isosceles(axes[1, 1], MU_RANGE, PHI_RANGE)

        plt.tight_layout()
        path = os.path.join(out, 'adaptive_overview.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {path}")

        # --- 2. Tier boundary map ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Tier Boundaries: {pot_l}', fontsize=15, fontweight='bold')

        tier_labels = ['1st tier', '2nd tier', '3rd tier']
        cmaps = ['Blues', 'Oranges', 'Greens']
        for t_idx in range(3):
            ax = axes[t_idx]
            t_pos = tier_data[:, :, t_idx, 0].copy()
            valid = t_pos >= 0
            t_pos[~valid] = np.nan
            im = ax.pcolormesh(phi_deg, mu, t_pos, cmap=cmaps[t_idx],
                               shading='auto')
            ax.set_title(f'{tier_labels[t_idx]} index')
            ax.set_xlabel('φ (degrees)')
            if t_idx == 0:
                ax.set_ylabel('μ = r₁₃/r₁₂')
            plt.colorbar(im, ax=ax)
            draw_isosceles(ax, MU_RANGE, PHI_RANGE)

        plt.tight_layout()
        path = os.path.join(out, 'tier_boundaries.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {path}")

        # --- 3. Print statistics ---
        valid_mask = opt_eps > 0
        print(f"\n    Statistics for {pot_l}:")
        print(f"      Rank range: [{rank[rank>0].min()}, {rank.max()}]")
        print(f"      Optimal eps range: [{opt_eps[valid_mask].min():.2e}, "
              f"{opt_eps[valid_mask].max():.2e}]")
        print(f"      Median optimal eps: {np.median(opt_eps[valid_mask]):.2e}")
        print(f"      Gap score range: [{gap_score.min():.2f}, "
              f"{gap_score.max():.2f}]")
        print(f"      Median gap score: {np.median(gap_score):.2f}")
        median_tiers = np.median(n_tiers)
        print(f"      Median tier count: {median_tiers:.1f}")


# ---------------------------------------------------------------------------
# Verification & data integrity
# ---------------------------------------------------------------------------
ADAPTIVE_ARRAYS = [
    ('rank_map', np.int32, 2),
    ('gap_map', np.float64, 2),
    ('sv_spectra', np.float64, 3),
    ('optimal_eps_map', np.float64, 2),
    ('gap_score_map', np.float64, 2),
    ('tier_map', np.float64, 4),
    ('n_tested_map', np.int32, 2),
]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_adaptive_scan(data_dir, expected_rows=None, expected_cols=None):
    """Validate adaptive scan output for completeness and integrity.

    Returns (ok: bool, report: dict).
    """
    report = {
        'data_dir': data_dir,
        'timestamp': strftime('%Y-%m-%d %H:%M:%S'),
        'checks': {},
        'warnings': [],
        'errors': [],
        'checksums': {},
    }

    def _check(name, passed, msg=""):
        report['checks'][name] = {'passed': passed, 'detail': msg}
        if not passed:
            report['errors'].append(f"{name}: {msg}")
        return passed

    def _warn(msg):
        report['warnings'].append(msg)

    all_ok = True

    # 1. Checkpoint
    cp = load_checkpoint(data_dir)
    if cp is None:
        all_ok &= _check('checkpoint_exists', False, 'No checkpoint.json')
    else:
        _check('checkpoint_exists', True,
               f"completed_rows={cp.get('completed_rows')}")
        cp_rows = cp.get('completed_rows', 0)
        if expected_rows is not None and cp_rows < expected_rows:
            all_ok &= _check('checkpoint_complete', False,
                             f"checkpoint says {cp_rows}, expected {expected_rows}")
        else:
            _check('checkpoint_complete', True, f"rows={cp_rows}")

    # 2. Config
    cfg_path = os.path.join(data_dir, 'config.json')
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        _check('config_exists', True,
               f"potential={cfg.get('potential')}, mode={cfg.get('mode')}")
        if expected_cols is None:
            expected_cols = cfg.get('grid_n', GRID_N)
        if expected_rows is None:
            expected_rows = cfg.get('n_rows', cfg.get('grid_n', GRID_N))
    else:
        all_ok &= _check('config_exists', False, 'No config.json')
        if expected_rows is None:
            expected_rows = GRID_N
        if expected_cols is None:
            expected_cols = GRID_N

    # 3. Array files
    loaded = {}
    for arr_name, dtype, ndim in ADAPTIVE_ARRAYS:
        fpath = os.path.join(data_dir, f'{arr_name}.npy')
        if not os.path.isfile(fpath):
            all_ok &= _check(f'{arr_name}_exists', False, 'File missing')
            continue

        report['checksums'][arr_name] = _sha256_file(fpath)
        try:
            arr = np.load(fpath)
            loaded[arr_name] = arr
        except Exception as e:
            all_ok &= _check(f'{arr_name}_loadable', False, str(e))
            continue

        _check(f'{arr_name}_exists', True,
               f"shape={arr.shape}, dtype={arr.dtype}")

        if arr.ndim != ndim:
            all_ok &= _check(f'{arr_name}_ndim', False,
                             f"expected {ndim}D, got {arr.ndim}D")
        else:
            _check(f'{arr_name}_ndim', True, f"{ndim}D")

        if arr.shape[0] != expected_rows or arr.shape[1] != expected_cols:
            all_ok &= _check(f'{arr_name}_shape', False,
                             f"expected ({expected_rows}, {expected_cols}, ...), "
                             f"got {arr.shape}")
        else:
            _check(f'{arr_name}_shape', True, str(arr.shape))

        if np.issubdtype(arr.dtype, np.floating):
            n_nan = np.isnan(arr).sum()
            n_inf = np.isinf(arr).sum()
            if n_nan > 0:
                all_ok &= _check(f'{arr_name}_nan', False,
                                 f"{n_nan} NaN values")
            else:
                _check(f'{arr_name}_nan', True, 'No NaN')
            if n_inf > 0:
                _warn(f"{arr_name} contains {n_inf} Inf values")

    # 4. Rank distribution
    if 'rank_map' in loaded:
        rm = loaded['rank_map']
        n_failed = int((rm == -1).sum())
        total_pts = rm.size
        fail_pct = 100 * n_failed / total_pts if total_pts > 0 else 0

        unique, counts = np.unique(rm, return_counts=True)
        rank_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        report['rank_distribution'] = rank_dist
        report['n_failed_points'] = n_failed
        report['fail_pct'] = round(fail_pct, 2)

        if fail_pct > 5:
            _warn(f"High failure rate: {n_failed}/{total_pts} "
                  f"({fail_pct:.1f}%) points have rank=-1")
        _check('rank_failures', fail_pct <= 10,
               f"{n_failed}/{total_pts} ({fail_pct:.1f}%) failed")

    # 5. Statistics
    stats = {}
    for name, arr in loaded.items():
        if np.issubdtype(arr.dtype, np.floating) and arr.ndim == 2:
            flat = arr.flatten()
            valid = flat[np.isfinite(flat) & (flat > 0)] if name != 'gap_score_map' else flat[np.isfinite(flat)]
            if valid.size > 0:
                stats[name] = {
                    'min': float(np.min(valid)),
                    'max': float(np.max(valid)),
                    'mean': float(np.mean(valid)),
                    'std': float(np.std(valid)),
                    'median': float(np.median(valid)),
                }
    report['statistics'] = stats

    # 6. Grid coordinate files
    for coord_name in ('mu_vals', 'phi_vals'):
        fpath = os.path.join(data_dir, f'{coord_name}.npy')
        if os.path.isfile(fpath):
            report['checksums'][coord_name] = _sha256_file(fpath)
            _check(f'{coord_name}_exists', True, '')
        else:
            all_ok &= _check(f'{coord_name}_exists', False, 'Missing')

    report['overall_pass'] = all_ok

    report_path = os.path.join(data_dir, 'verification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    status = "PASS" if all_ok else "FAIL"
    print(f"\n  Verification {status}: {data_dir}")
    if report['errors']:
        for e in report['errors']:
            print(f"    ERROR: {e}")
    if report['warnings']:
        for w in report['warnings']:
            print(f"    WARN:  {w}")
    if 'rank_distribution' in report:
        print(f"    Rank distribution: {report['rank_distribution']}")
    if 'statistics' in report:
        for name, s in report['statistics'].items():
            print(f"    {name}: min={s['min']:.3e} max={s['max']:.3e} "
                  f"mean={s['mean']:.3e}")
    print(f"    Report saved: {report_path}")

    return all_ok, report


# ---------------------------------------------------------------------------
# Block merge for distributed execution
# ---------------------------------------------------------------------------
def merge_adaptive_blocks(potential_type, charges=None):
    """Discover and merge adaptive/block_SSSS_EEEE/ dirs into adaptive/merged/."""

    base = pot_dir_key(potential_type, charges)
    adaptive_dir = os.path.join(HIRES_DIR, base, 'adaptive')
    label = pot_label_key(potential_type, charges)

    if not os.path.isdir(adaptive_dir):
        print(f"  No adaptive directory for {label}")
        return False

    blocks = []
    for name in sorted(os.listdir(adaptive_dir)):
        if not name.startswith('block_'):
            continue
        bdir = os.path.join(adaptive_dir, name)
        if not os.path.isdir(bdir):
            continue
        parts = name.replace('block_', '').split('_')
        if len(parts) != 2:
            continue
        try:
            sr, er = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        blocks.append((sr, er, bdir))

    if not blocks:
        print(f"  No blocks found in {adaptive_dir}")
        return False

    blocks.sort(key=lambda x: x[0])
    print(f"\n  Merging {len(blocks)} adaptive blocks for {label}:")
    for sr, er, bd in blocks:
        print(f"    rows {sr}-{er}: {bd}")

    # Validate each block is complete
    for sr, er, bdir in blocks:
        expected = er - sr
        cp = load_checkpoint(bdir)
        if cp is None:
            print(f"  ERROR: No checkpoint in {bdir}")
            return False
        done = cp.get('completed_rows', 0)
        if done < expected:
            print(f"  ERROR: Block {sr}-{er} incomplete: "
                  f"{done}/{expected} rows")
            return False

    # Verify row continuity
    expected_start = blocks[0][0]
    total_rows = 0
    for sr, er, _ in blocks:
        if sr != expected_start:
            print(f"  ERROR: Gap in blocks at row {expected_start} "
                  f"(next block starts at {sr})")
            return False
        total_rows += er - sr
        expected_start = er

    # Load and concatenate
    merged_dir = os.path.join(adaptive_dir, 'merged')
    os.makedirs(merged_dir, exist_ok=True)

    first_cfg_path = os.path.join(blocks[0][2], 'config.json')
    if os.path.isfile(first_cfg_path):
        with open(first_cfg_path) as f:
            cfg = json.load(f)
        n_gen = cfg.get('n_generators', 156)
    else:
        n_gen = 156

    print(f"  Total rows to merge: {total_rows}")

    all_arrays = {}
    for arr_name, dtype, ndim in ADAPTIVE_ARRAYS:
        parts_list = []
        for sr, er, bdir in blocks:
            fpath = os.path.join(bdir, f'{arr_name}.npy')
            if not os.path.isfile(fpath):
                print(f"  ERROR: Missing {arr_name}.npy in block {sr}-{er}")
                return False
            parts_list.append(np.load(fpath))
        merged = np.concatenate(parts_list, axis=0)
        all_arrays[arr_name] = merged
        np.save(os.path.join(merged_dir, f'{arr_name}.npy'), merged)
        print(f"    {arr_name}: shape {merged.shape}")

    # Merge coordinate arrays
    mu_parts = []
    for sr, er, bdir in blocks:
        fpath = os.path.join(bdir, 'mu_vals.npy')
        if os.path.isfile(fpath):
            mu_parts.append(np.load(fpath))
    if mu_parts:
        np.save(os.path.join(merged_dir, 'mu_vals.npy'),
                np.concatenate(mu_parts))

    phi_path = os.path.join(blocks[0][2], 'phi_vals.npy')
    if os.path.isfile(phi_path):
        np.save(os.path.join(merged_dir, 'phi_vals.npy'),
                np.load(phi_path))

    # Write merged config
    merged_cfg = dict(cfg) if os.path.isfile(first_cfg_path) else {}
    merged_cfg['n_rows'] = total_rows
    merged_cfg['start_row'] = blocks[0][0]
    merged_cfg['end_row'] = blocks[-1][1]
    merged_cfg['merged_from'] = len(blocks)
    merged_cfg['merge_timestamp'] = strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(merged_dir, 'config.json'), 'w') as f:
        json.dump(merged_cfg, f, indent=2)

    save_checkpoint_atomic(merged_dir, total_rows, n_gen, -1, extra={
        'merged': True,
        'n_blocks': len(blocks),
        'total_rows': total_rows,
    })

    print(f"\n  Merged results saved to {merged_dir}")

    # Run verification on merged data
    ok, _ = verify_adaptive_scan(merged_dir,
                                  expected_rows=total_rows,
                                  expected_cols=GRID_N)

    s3_sync(merged_dir, f"{HIRES_DIR}/{base}/adaptive/merged")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Multi-Epsilon Shape Sphere Atlas')
    parser.add_argument('mode', nargs='?', default='all',
                        choices=['all', 'scan', 'analyze', 'animate',
                                 'adaptive', 'adaptive-analyze',
                                 'adaptive-verify', 'adaptive-merge'],
                        help='Which phase(s) to run')
    parser.add_argument('--potential', type=str, default=None,
                        choices=['1/r', '1/r2', 'harmonic'],
                        help='Run only one potential (default: singular potentials)')
    parser.add_argument('--charges', nargs='+', type=int, default=None,
                        metavar='Q',
                        help='Charges for Coulomb potential, e.g. --charges 2 -1 -1')
    parser.add_argument('--workers', type=int,
                        default=max(1, cpu_count() - 1),
                        help='Number of multiprocessing workers (adaptive mode)')
    parser.add_argument('--start-row', type=int, default=0,
                        help='First grid row to scan (adaptive mode)')
    parser.add_argument('--end-row', type=int, default=None,
                        help='Last grid row (exclusive, adaptive mode)')
    parser.add_argument('--point-timeout', type=int, default=600,
                        help='Per-point timeout in seconds (adaptive mode)')
    args = parser.parse_args()

    charges = args.charges

    if args.mode in ('adaptive', 'adaptive-analyze'):
        pot = args.potential or '1/r'
        if args.mode == 'adaptive':
            print(f"\n{'#'*70}")
            print(f"# ADAPTIVE EPSILON SCAN")
            print(f"# Grid: {GRID_N}x{GRID_N}, samples={N_SAMPLES}, level={LEVEL}")
            print(f"# Workers: {args.workers}")
            if charges is not None:
                print(f"# Charges: {charges}")
            print(f"{'#'*70}")
            run_adaptive_scan(pot, charges=charges,
                              n_workers=args.workers,
                              start_row=args.start_row,
                              end_row=args.end_row,
                              point_timeout=args.point_timeout)

            # Verify after scan completes
            base = pot_dir_key(pot, charges)
            is_block = (args.start_row != 0 or
                        (args.end_row is not None and args.end_row != GRID_N))
            if is_block:
                data_dir = _adaptive_block_dir(
                    os.path.join(HIRES_DIR, base, 'adaptive'),
                    args.start_row, args.end_row or GRID_N)
            else:
                data_dir = os.path.join(HIRES_DIR, base, 'adaptive')
            verify_adaptive_scan(data_dir)

        print(f"\n{'#'*70}")
        print(f"# ADAPTIVE ANALYSIS")
        print(f"{'#'*70}")
        run_adaptive_analysis(pot, charges=charges)

    elif args.mode == 'adaptive-verify':
        pot = args.potential or '1/r'
        base = pot_dir_key(pot, charges)
        is_block = (args.start_row != 0 or
                    (args.end_row is not None and args.end_row != GRID_N))
        if is_block:
            data_dir = _adaptive_block_dir(
                os.path.join(HIRES_DIR, base, 'adaptive'),
                args.start_row, args.end_row or GRID_N)
        else:
            data_dir = os.path.join(HIRES_DIR, base, 'adaptive')
        ok, report = verify_adaptive_scan(data_dir)
        sys.exit(0 if ok else 1)

    elif args.mode == 'adaptive-merge':
        pot = args.potential or '1/r'
        ok = merge_adaptive_blocks(pot, charges=charges)
        sys.exit(0 if ok else 1)

    if args.mode in ('all', 'scan'):
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: MULTI-EPSILON GRID SCANS")
        print(f"# Epsilons: {[f'{e:.0e}' for e in EPSILONS]}")
        print(f"# Grid: {GRID_N}x{GRID_N}, samples={N_SAMPLES}, level={LEVEL}")
        if charges is not None:
            print(f"# Charges: {charges}")
        print(f"{'#'*70}")

        if charges is not None:
            pot = args.potential or '1/r'
            run_multi_epsilon_scan(pot, charges=charges)
        elif args.potential:
            pots = [args.potential]
            for pot in pots:
                if pot == 'harmonic':
                    run_multi_epsilon_scan(pot, epsilons=[5e-3, 1e-4])
                else:
                    run_multi_epsilon_scan(pot)
        else:
            for pot in SINGULAR_POTENTIALS:
                run_multi_epsilon_scan(pot)

    if args.mode in ('all', 'analyze'):
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: DERIVED ANALYSIS")
        print(f"{'#'*70}")
        run_analysis()

    if args.mode in ('all', 'animate'):
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: ANIMATION")
        print(f"{'#'*70}")
        run_animation()

    print(f"\n{'#'*70}")
    print(f"# COMPLETE")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
