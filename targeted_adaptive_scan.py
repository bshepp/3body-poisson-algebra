#!/usr/bin/env python3
"""
Targeted High-Resolution Adaptive Scans
==========================================

Runs adaptive epsilon scans on specific regions of interest identified
from the 100x100 atlas data.  Each region is a small patch (30x30 to
50x50) with enhanced parameters (more epsilon candidates, more samples)
to resolve fine structure.

Regions are selected based on:
  - Unusual ranks (rank != 116)
  - Richest tier structure (4-5 tiers)
  - Largest charge-sensitivity (gap score difference)
  - Finest-scale configurations (smallest optimal epsilon)
  - Known special configurations (Lagrange, Euler, isosceles)

Usage:
    python targeted_adaptive_scan.py                    # scan all regions
    python targeted_adaptive_scan.py --region lagrange  # one region
    python targeted_adaptive_scan.py --list             # list regions
    python targeted_adaptive_scan.py --region euler --potential 1/r
    python targeted_adaptive_scan.py --workers 15       # multiprocessing
    python targeted_adaptive_scan.py --analyze          # visualize results
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time, strftime

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from multiprocessing import Pool, TimeoutError as MPTimeoutError, cpu_count

from multi_epsilon_atlas import (
    save_checkpoint_atomic, load_checkpoint, s3_sync,
    pot_dir_key, pot_label_key, verify_adaptive_scan,
    MAX_TIERS, _eval_adaptive_point,
    _adaptive_algebra, _adaptive_level,
    _sigterm_handler,
)
import multi_epsilon_atlas as mea

# ---------------------------------------------------------------------------
# Target regions
# ---------------------------------------------------------------------------
REGIONS = {
    'lagrange': {
        'description': 'Lagrange equilateral neighborhood (5-tier point nearby)',
        'mu_range': (0.85, 1.15),
        'phi_range_deg': (50, 95),
        'grid_n': 50,
        'n_samples': 800,
        'n_eps': 12,
        'reason': 'Equilateral triangle, 5-tier point at (0.99, 84deg), '
                  'tier count transition zone',
    },
    'euler_strip': {
        'description': 'Euler collinear / low-phi strip with anomalous ranks',
        'mu_range': (0.7, 1.3),
        'phi_range_deg': (3, 20),
        'grid_n': 40,
        'n_samples': 800,
        'n_eps': 12,
        'reason': 'All unusual ranks (117-124) and lowest gap scores '
                  'cluster here; near-collinear configurations',
    },
    'charge_hotspot': {
        'description': 'Maximum charge-sensitivity region',
        'mu_range': (0.5, 0.85),
        'phi_range_deg': (85, 105),
        'grid_n': 40,
        'n_samples': 800,
        'n_eps': 12,
        'reason': 'Largest gap score difference between charged and '
                  'uncharged (delta=-2.64 at mu=0.65, phi=94deg); '
                  'helium needs finest scale (eps=9.35e-4) here',
    },
    'tier_cluster': {
        'description': 'High-tier helium cluster (asymmetric from Lagrange)',
        'mu_range': (1.45, 1.9),
        'phi_range_deg': (65, 85),
        'grid_n': 40,
        'n_samples': 800,
        'n_eps': 12,
        'reason': '18 points with 4 tiers in helium scan; this region '
                  'has rich internal structure displaced from equilateral',
    },
    'isosceles_ridge': {
        'description': 'mu=1 isosceles symmetry line',
        'mu_range': (0.9, 1.1),
        'phi_range_deg': (10, 170),
        'grid_n': (10, 80),
        'n_samples': 800,
        'n_eps': 12,
        'reason': 'Narrow strip along mu=1 isosceles line; tracks how '
                  'tier structure evolves along the symmetry curve',
    },
    'small_mu': {
        'description': 'Small mu (extreme mass ratio / near-collision)',
        'mu_range': (0.2, 0.4),
        'phi_range_deg': (70, 100),
        'grid_n': 30,
        'n_samples': 800,
        'n_eps': 12,
        'reason': 'Reference scan needed finest epsilon (1.64e-3) near '
                  'mu=0.2, phi=84deg; extreme mass ratios may hide structure',
    },
}

LEVEL = 3
OUTPUT_BASE = 'atlas_targeted'
S3_BUCKET = os.environ.get("S3_BUCKET", "")


def region_out_dir(region_name, potential_type, charges=None):
    pot_d = pot_dir_key(potential_type, charges)
    return os.path.join(OUTPUT_BASE, pot_d, region_name)


def _s3_prefix(potential_type, charges, region_name):
    return f"{OUTPUT_BASE}/{pot_dir_key(potential_type, charges)}/{region_name}"


def _write_status(out_dir, status_dict):
    """Write a machine-readable status.json for remote monitoring."""
    path = os.path.join(out_dir, 'status.json')
    tmp = path + '.tmp'
    status_dict['timestamp'] = strftime('%Y-%m-%d %H:%M:%S')
    with open(tmp, 'w') as f:
        json.dump(status_dict, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sync_down_region(region_name, potential_type, charges=None):
    """Pull existing region data from S3 before starting (for spot resume)."""
    if not S3_BUCKET:
        return
    import subprocess
    prefix = _s3_prefix(potential_type, charges, region_name)
    local = region_out_dir(region_name, potential_type, charges)
    os.makedirs(local, exist_ok=True)
    src = f"s3://{S3_BUCKET}/{prefix}/"
    print(f"  [S3] Downloading {src} -> {local}/")
    try:
        subprocess.run(
            ['aws', 's3', 'sync', src, local + '/'],
            capture_output=True, timeout=300
        )
        cp = load_checkpoint(local)
        if cp:
            print(f"  [S3] Found checkpoint: {cp.get('completed_rows', 0)} "
                  f"rows done")
        else:
            print(f"  [S3] No existing checkpoint found")
    except Exception as e:
        print(f"  [S3] Download warning: {e}")


def sync_up_region(out_dir, potential_type, charges, region_name):
    """Push all region data to S3 (including plots and verification)."""
    prefix = _s3_prefix(potential_type, charges, region_name)
    s3_sync(out_dir, prefix, include_all=True)


def run_region_scan(region_name, potential_type, charges=None,
                    n_workers=1, point_timeout=600):
    """Run an adaptive scan on a single targeted region."""
    import signal
    from stability_atlas import AtlasConfig, PoissonAlgebra, ShapeSpace

    region = REGIONS[region_name]
    label = pot_label_key(potential_type, charges)
    charges_tuple = tuple(charges) if charges is not None else None

    mu_lo, mu_hi = region['mu_range']
    phi_lo_deg, phi_hi_deg = region['phi_range_deg']
    phi_lo = np.radians(phi_lo_deg)
    phi_hi = np.radians(phi_hi_deg)
    n_samples = region.get('n_samples', 800)
    n_eps = region.get('n_eps', 12)
    eps_range = region.get('eps_range', (5e-5, 5e-3))

    grid_spec = region['grid_n']
    if isinstance(grid_spec, tuple):
        grid_mu, grid_phi = grid_spec
    else:
        grid_mu = grid_phi = grid_spec

    out_dir = region_out_dir(region_name, potential_type, charges)
    os.makedirs(out_dir, exist_ok=True)

    n_rows = grid_mu
    n_cols = grid_phi
    total = n_rows * n_cols

    # Pull any existing progress from S3 before checking local state
    sync_down_region(region_name, potential_type, charges)

    cp = load_checkpoint(out_dir)
    cp_done = cp.get('completed_rows', 0) if cp else 0
    if cp_done >= n_rows:
        print(f"  [{region_name}] {label}: COMPLETE (skipping)")
        return

    print(f"\n{'='*70}")
    print(f"  TARGETED ADAPTIVE SCAN: {region_name}")
    print(f"  {region['description']}")
    print(f"  Potential: {label}")
    if charges is not None:
        print(f"  Charges: {charges}")
    print(f"  mu: [{mu_lo:.3f}, {mu_hi:.3f}]  "
          f"phi: [{phi_lo_deg:.0f}, {phi_hi_deg:.0f}] deg")
    print(f"  Grid: {grid_mu}x{grid_phi} = {total} points")
    print(f"  Samples/point: {n_samples}, Eps candidates: {n_eps}, "
          f"Level: {LEVEL}")
    print(f"  Workers: {n_workers}, Point timeout: {point_timeout}s")
    print(f"  Reason: {region['reason']}")
    print(f"{'='*70}\n")

    signal.signal(signal.SIGTERM, _sigterm_handler)
    mea._shutdown_requested = False

    config = AtlasConfig(
        potential_type=potential_type,
        max_level=LEVEL,
        n_phase_samples=n_samples,
        epsilon=eps_range[1],
        svd_gap_threshold=1e4,
        charges=charges_tuple,
    )
    print(f"  Building symbolic algebra for {label}...")
    t_build = time()
    algebra = PoissonAlgebra(config)
    n_gen = len(algebra._names)
    print(f"  Algebra ready ({time()-t_build:.1f}s, {n_gen} generators)")

    mea._adaptive_algebra = algebra
    mea._adaptive_level = LEVEL
    mea._adaptive_eps_range = eps_range
    mea._adaptive_n_eps = n_eps
    mea._adaptive_tier_threshold = 10.0
    mea._adaptive_max_tiers = MAX_TIERS

    mu_vals = np.linspace(mu_lo, mu_hi, grid_mu)
    phi_vals = np.linspace(phi_lo, phi_hi, grid_phi)

    if cp_done > 0:
        try:
            rank_map = np.load(os.path.join(out_dir, 'rank_map.npy'))
            gap_map = np.load(os.path.join(out_dir, 'gap_map.npy'))
            sv_spectra = np.load(os.path.join(out_dir, 'sv_spectra.npy'))
            optimal_eps_map = np.load(os.path.join(out_dir, 'optimal_eps_map.npy'))
            gap_score_map = np.load(os.path.join(out_dir, 'gap_score_map.npy'))
            tier_map = np.load(os.path.join(out_dir, 'tier_map.npy'))
            n_tested_map = np.load(os.path.join(out_dir, 'n_tested_map.npy'))

            if rank_map.shape != (n_rows, n_cols):
                print(f"  [WARN] Shape mismatch: saved {rank_map.shape} vs "
                      f"expected ({n_rows},{n_cols}). Starting fresh.")
                cp_done = 0
                raise ValueError("shape mismatch")

            print(f"  Resuming from row {cp_done} "
                  f"({cp_done * n_cols} points done)")
        except (ValueError, OSError) as e:
            print(f"  [WARN] Could not load checkpoint data: {e}. "
                  f"Starting fresh.")
            cp_done = 0

    if cp_done == 0:
        rank_map = np.zeros((n_rows, n_cols), dtype=np.int32)
        gap_map = np.zeros((n_rows, n_cols))
        sv_spectra = np.zeros((n_rows, n_cols, n_gen), dtype=np.float64)
        optimal_eps_map = np.zeros((n_rows, n_cols))
        gap_score_map = np.zeros((n_rows, n_cols))
        tier_map = np.full((n_rows, n_cols, MAX_TIERS, 2), -1,
                           dtype=np.float64)
        n_tested_map = np.zeros((n_rows, n_cols), dtype=np.int32)

    np.save(os.path.join(out_dir, 'mu_vals.npy'), mu_vals)
    np.save(os.path.join(out_dir, 'phi_vals.npy'), phi_vals)

    config_data = {
        'region': region_name,
        'potential': potential_type,
        'mu_range': [mu_lo, mu_hi],
        'phi_range_deg': [phi_lo_deg, phi_hi_deg],
        'grid_mu': grid_mu,
        'grid_phi': grid_phi,
        'eps_range': list(eps_range),
        'n_eps': n_eps,
        'n_samples': n_samples,
        'level': LEVEL,
        'n_generators': n_gen,
        'mode': 'targeted_adaptive',
        'description': region['description'],
        'reason': region['reason'],
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

    def _unpack(i, j, result):
        rank, svs, max_gap, opt_e, g_score, n_tried, tiers = result
        rank_map[i, j] = rank
        gap_map[i, j] = max_gap
        if len(svs) > 0:
            sv_spectra[i, j, :len(svs)] = svs
        optimal_eps_map[i, j] = opt_e
        gap_score_map[i, j] = g_score
        n_tested_map[i, j] = n_tried
        for t_idx, (t_pos, t_gap) in enumerate(tiers[:MAX_TIERS]):
            tier_map[i, j, t_idx, 0] = t_pos
            tier_map[i, j, t_idx, 1] = min(t_gap, 1e16)

    def _save_all():
        """Save all arrays atomically (write to .tmp.npy, then rename)."""
        for arr_name, arr in [('rank_map', rank_map), ('gap_map', gap_map),
                              ('sv_spectra', sv_spectra),
                              ('optimal_eps_map', optimal_eps_map),
                              ('gap_score_map', gap_score_map),
                              ('tier_map', tier_map),
                              ('n_tested_map', n_tested_map)]:
            dst = os.path.join(out_dir, f'{arr_name}.npy')
            tmp = os.path.join(out_dir, f'{arr_name}.tmp.npy')
            np.save(tmp, arr)
            os.replace(tmp, dst)

    t_scan = time()
    try:
        for i in range(cp_done, n_rows):
            t_row = time()
            mu = mu_vals[i]
            n_timeout = 0

            if use_mp:
                tasks = [(mu, phi_vals[j]) for j in range(n_cols)]
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
                    _unpack(i, j, result)
            else:
                for j in range(n_cols):
                    result = _eval_adaptive_point((mu, phi_vals[j]))
                    _unpack(i, j, result)

            _save_all()
            save_checkpoint_atomic(out_dir, i + 1, n_gen, -1, extra={
                'region': region_name, 'potential': potential_type,
            })

            done = (i + 1) * n_cols
            elapsed = time() - t_scan
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            row_time = time() - t_row

            avg_eps_tested = n_tested_map[i, :].mean()
            timeout_str = f"  timeouts={n_timeout}" if n_timeout else ""
            print(f"  Row {i+1:3d}/{n_rows}  mu={mu:.3f}  "
                  f"[{done:5d}/{total}]  "
                  f"row={row_time:.1f}s  "
                  f"ETA={remaining/60:.0f}m  "
                  f"ranks=[{rank_map[i,:].min()},{rank_map[i,:].max()}]  "
                  f"gap_score=[{gap_score_map[i,:].min():.1f},"
                  f"{gap_score_map[i,:].max():.1f}]  "
                  f"avg_eps={avg_eps_tested:.1f}"
                  f"{timeout_str}",
                  flush=True)

            _write_status(out_dir, {
                'region': region_name, 'potential': potential_type,
                'row': i + 1, 'total_rows': n_rows,
                'done_pts': done, 'total_pts': total,
                'elapsed_s': round(elapsed, 1),
                'eta_min': round(remaining / 60, 1),
                'state': 'running',
            })

            # S3 sync every 3 rows (small data, fast sync)
            if (i + 1) % 3 == 0 or (i + 1) == n_rows:
                sync_up_region(out_dir, potential_type, charges,
                               region_name)

            if mea._shutdown_requested:
                print(f"  [SHUTDOWN] Saved through row {i+1}/{n_rows}.",
                      flush=True)
                _write_status(out_dir, {
                    'region': region_name, 'potential': potential_type,
                    'row': i + 1, 'total_rows': n_rows,
                    'state': 'shutdown_safe',
                })
                sync_up_region(out_dir, potential_type, charges,
                               region_name)
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    total_time = time() - t_scan

    if mea._shutdown_requested:
        # On spot reclamation, skip verification -- just ensure data is synced
        print(f"\n  {region_name} INTERRUPTED after {total_time:.0f}s "
              f"(saved through checkpoint)")
        return

    print(f"\n  {region_name} complete: {total_time:.0f}s ({total_time/60:.1f}m)")

    valid_mask = rank_map > 0
    if valid_mask.any():
        print(f"  Rank range: [{rank_map[valid_mask].min()}, {rank_map.max()}]")
    valid_eps = optimal_eps_map[optimal_eps_map > 0]
    if valid_eps.size > 0:
        print(f"  Optimal eps: [{valid_eps.min():.2e}, {valid_eps.max():.2e}]")
    print(f"  Gap score: [{gap_score_map.min():.2f}, {gap_score_map.max():.2f}]")
    n_tiers = np.sum(tier_map[:, :, :, 0] >= 0, axis=2)
    print(f"  Tier count: [{n_tiers.min()}, {n_tiers.max()}], "
          f"median={np.median(n_tiers):.1f}")
    unique_ranks, counts = np.unique(rank_map[valid_mask], return_counts=True)
    print(f"  Rank distribution: "
          + ", ".join(f"{r}:{c}" for r, c in zip(unique_ranks, counts)))

    verify_adaptive_scan(out_dir, expected_rows=n_rows,
                          expected_cols=n_cols)
    _write_status(out_dir, {
        'region': region_name, 'potential': potential_type,
        'row': n_rows, 'total_rows': n_rows,
        'done_pts': total, 'total_pts': total,
        'elapsed_s': round(total_time, 1),
        'state': 'complete',
    })
    sync_up_region(out_dir, potential_type, charges, region_name)
    print()


def run_analysis(potential_type, charges=None):
    """Generate comparison plots for all completed targeted regions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    label = pot_label_key(potential_type, charges)
    pot_d = pot_dir_key(potential_type, charges)
    base = os.path.join(OUTPUT_BASE, pot_d)

    if not os.path.isdir(base):
        print(f"  No targeted data for {label}")
        return

    completed = []
    for name in REGIONS:
        rdir = os.path.join(base, name)
        if os.path.isfile(os.path.join(rdir, 'rank_map.npy')):
            completed.append(name)

    if not completed:
        print(f"  No completed regions for {label}")
        return

    out = os.path.join(base, 'plots')
    os.makedirs(out, exist_ok=True)

    def draw_isosceles(ax, mu_range, phi_range_rad):
        phi_d = np.linspace(phi_range_rad[0], phi_range_rad[1], 500)
        mu_lo, mu_hi = mu_range
        ax.axhline(1.0, color='cyan', linewidth=1, linestyle='--', alpha=0.5)
        mu_c2 = 2 * np.cos(phi_d)
        m2 = (mu_c2 >= mu_lo) & (mu_c2 <= mu_hi)
        if m2.any():
            ax.plot(np.degrees(phi_d[m2]), mu_c2[m2],
                    color='lime', linewidth=1, linestyle='--', alpha=0.5)
        cos_p = np.cos(phi_d)
        safe = cos_p > 0.01
        mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
        m3 = np.isfinite(mu_c3) & (mu_c3 >= mu_lo) & (mu_c3 <= mu_hi)
        if m3.any():
            ax.plot(np.degrees(phi_d[m3]), mu_c3[m3],
                    color='magenta', linewidth=1, linestyle='--', alpha=0.5)

    for name in completed:
        rdir = os.path.join(base, name)
        region = REGIONS[name]

        rank = np.load(os.path.join(rdir, 'rank_map.npy'))
        gap_score = np.load(os.path.join(rdir, 'gap_score_map.npy'))
        opt_eps = np.load(os.path.join(rdir, 'optimal_eps_map.npy'))
        tier_data = np.load(os.path.join(rdir, 'tier_map.npy'))
        mu = np.load(os.path.join(rdir, 'mu_vals.npy'))
        phi = np.load(os.path.join(rdir, 'phi_vals.npy'))
        phi_deg = np.degrees(phi)
        n_tiers = np.sum(tier_data[:, :, :, 0] >= 0, axis=2)

        mu_range = (mu[0], mu[-1])
        phi_range = (phi[0], phi[-1])
        phi_span = phi_deg[-1] - phi_deg[0]
        mu_span = mu[-1] - mu[0]
        data_aspect = phi_span / (mu_span * 180)
        fig_w = max(14, min(22, 10 * data_aspect))
        fig_h = 10

        fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
        fig.suptitle(f'Targeted: {name} -- {label}\n{region["description"]}',
                     fontsize=13, fontweight='bold')

        log_eps = np.log10(np.where(opt_eps > 0, opt_eps, 1e-5))
        eps_lo = np.percentile(log_eps, 1)
        eps_hi = np.percentile(log_eps, 99)
        if eps_hi - eps_lo < 0.1:
            eps_lo = log_eps.min() - 0.05
            eps_hi = log_eps.max() + 0.05
        im0 = axes[0, 0].pcolormesh(phi_deg, mu, log_eps, cmap='viridis',
                                      shading='auto', vmin=eps_lo, vmax=eps_hi)
        axes[0, 0].set_title('Optimal log10(eps)')
        axes[0, 0].set_ylabel('mu')
        plt.colorbar(im0, ax=axes[0, 0])
        draw_isosceles(axes[0, 0], mu_range, phi_range)

        gap_lo = np.percentile(gap_score, 1)
        gap_hi = np.percentile(gap_score, 99)
        if gap_hi - gap_lo < 0.1:
            gap_lo = gap_score.min() - 0.05
            gap_hi = gap_score.max() + 0.05
        im1 = axes[0, 1].pcolormesh(phi_deg, mu, gap_score, cmap='inferno',
                                      shading='auto', vmin=gap_lo, vmax=gap_hi)
        axes[0, 1].set_title('Gap Score')
        plt.colorbar(im1, ax=axes[0, 1])
        draw_isosceles(axes[0, 1], mu_range, phi_range)

        rank_lo = max(rank.min() - 1, 112)
        rank_hi = max(rank.max() + 1, 118)
        im2 = axes[1, 0].pcolormesh(phi_deg, mu, rank, cmap='RdYlGn',
                                      shading='auto', vmin=rank_lo, vmax=rank_hi)
        axes[1, 0].set_title(f'SVD Rank [{rank.min()}\u2013{rank.max()}]')
        axes[1, 0].set_xlabel('phi (deg)')
        axes[1, 0].set_ylabel('mu')
        plt.colorbar(im2, ax=axes[1, 0])
        draw_isosceles(axes[1, 0], mu_range, phi_range)

        tier_hi = max(n_tiers.max(), 4)
        im3 = axes[1, 1].pcolormesh(phi_deg, mu, n_tiers, cmap='YlOrRd',
                                      shading='auto', vmin=1, vmax=tier_hi)
        axes[1, 1].set_title(f'Number of Tiers [1\u2013{n_tiers.max()}]')
        axes[1, 1].set_xlabel('phi (deg)')
        plt.colorbar(im3, ax=axes[1, 1])
        draw_isosceles(axes[1, 1], mu_range, phi_range)

        for ax in axes.flat:
            ax.set_xlim(phi_deg[0], phi_deg[-1])
            ax.set_ylim(mu[0], mu[-1])

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        path = os.path.join(out, f'{name}_overview.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    # If both reference and charged exist, do a comparison
    ref_base = os.path.join(OUTPUT_BASE, pot_dir_key(potential_type))
    chg_base = os.path.join(OUTPUT_BASE, pot_d) if charges else None

    if charges and os.path.isdir(ref_base):
        for name in completed:
            ref_dir = os.path.join(ref_base, name)
            chg_dir = os.path.join(chg_base, name)
            if not (os.path.isfile(os.path.join(ref_dir, 'gap_score_map.npy'))
                    and os.path.isfile(os.path.join(chg_dir, 'gap_score_map.npy'))):
                continue

            gs_ref = np.load(os.path.join(ref_dir, 'gap_score_map.npy'))
            gs_chg = np.load(os.path.join(chg_dir, 'gap_score_map.npy'))
            mu = np.load(os.path.join(ref_dir, 'mu_vals.npy'))
            phi = np.load(os.path.join(ref_dir, 'phi_vals.npy'))

            if gs_ref.shape != gs_chg.shape:
                continue

            diff = gs_chg - gs_ref
            fig, ax = plt.subplots(figsize=(10, 7))
            vlim = max(abs(diff.min()), abs(diff.max()), 0.5)
            im = ax.pcolormesh(np.degrees(phi), mu, diff, cmap='RdBu_r',
                               vmin=-vlim, vmax=vlim, shading='auto')
            draw_isosceles(ax, (mu[0], mu[-1]), (phi[0], phi[-1]))
            ax.set_xlim(np.degrees(phi[0]), np.degrees(phi[-1]))
            ax.set_ylim(mu[0], mu[-1])
            ax.set_xlabel('phi (deg)')
            ax.set_ylabel('mu')
            ax.set_title(f'Gap Score Difference (charged - ref): {name}\n'
                         f'{label}')
            plt.colorbar(im, ax=ax, label='Delta gap score')
            plt.tight_layout()
            path = os.path.join(out, f'{name}_charge_diff.png')
            fig.savefig(path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {path}")

    print(f"\n  All plots saved to {out}/")


def main():
    parser = argparse.ArgumentParser(
        description='Targeted High-Resolution Adaptive Scans')
    parser.add_argument('--region', type=str, default=None,
                        choices=list(REGIONS.keys()),
                        help='Scan a specific region (default: all)')
    parser.add_argument('--potential', type=str, default='1/r2',
                        choices=['1/r', '1/r2', 'harmonic'],
                        help='Potential type (default: 1/r2)')
    parser.add_argument('--charges', nargs='+', type=int, default=None,
                        metavar='Q',
                        help='Charges, e.g. --charges 2 -1 -1')
    parser.add_argument('--list', action='store_true',
                        help='List available regions and exit')
    parser.add_argument('--analyze', action='store_true',
                        help='Generate plots from completed scans')
    parser.add_argument('--workers', type=int,
                        default=max(1, cpu_count() - 1),
                        help='Number of multiprocessing workers')
    parser.add_argument('--point-timeout', type=int, default=600,
                        help='Per-point timeout in seconds')
    parser.add_argument('--both', action='store_true',
                        help='Run both reference and charged scans')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable target regions:\n")
        for name, r in REGIONS.items():
            grid_spec = r['grid_n']
            if isinstance(grid_spec, tuple):
                grid_str = f"{grid_spec[0]}x{grid_spec[1]}"
                pts = grid_spec[0] * grid_spec[1]
            else:
                grid_str = f"{grid_spec}x{grid_spec}"
                pts = grid_spec ** 2
            est_min = pts * 5.5 / 60
            print(f"  {name:20s}  {grid_str:>7s} = {pts:5d} pts  "
                  f"~{est_min:.0f} min")
            print(f"    mu=[{r['mu_range'][0]:.2f}, {r['mu_range'][1]:.2f}]  "
                  f"phi=[{r['phi_range_deg'][0]:.0f}, "
                  f"{r['phi_range_deg'][1]:.0f}] deg")
            print(f"    {r['description']}")
            print()
        return

    if args.analyze:
        run_analysis(args.potential, args.charges)
        return

    regions_to_run = [args.region] if args.region else list(REGIONS.keys())

    if args.both:
        configs = [
            (args.potential, None),
            (args.potential, args.charges or [2, -1, -1]),
        ]
    elif args.charges:
        configs = [(args.potential, args.charges)]
    else:
        configs = [(args.potential, None)]

    total_pts = 0
    for name in regions_to_run:
        g = REGIONS[name]['grid_n']
        if isinstance(g, tuple):
            total_pts += g[0] * g[1]
        else:
            total_pts += g * g

    print(f"\n{'#'*70}")
    print(f"# TARGETED ADAPTIVE SCANS")
    print(f"# Regions: {', '.join(regions_to_run)}")
    print(f"# Configs: {len(configs)}")
    print(f"# Total points per config: {total_pts}")
    print(f"# Workers: {args.workers}")
    if args.workers > 1:
        est_hrs = total_pts * 5.5 / 3600 / args.workers
        print(f"# Estimated time per config: ~{est_hrs:.1f} hrs "
              f"({args.workers} workers)")
    else:
        print(f"# Estimated time per config: ~{total_pts * 5.5 / 3600:.1f} hrs "
              f"(sequential)")
    print(f"{'#'*70}")

    for pot, charges in configs:
        for name in regions_to_run:
            run_region_scan(name, pot, charges,
                            n_workers=args.workers,
                            point_timeout=args.point_timeout)
            if mea._shutdown_requested:
                print(f"\n  [SHUTDOWN] Stopping after {name} (spot reclaim)")
                break

        if mea._shutdown_requested:
            break

        run_analysis(pot, charges)

    if mea._shutdown_requested:
        print(f"\n{'#'*70}")
        print(f"# SHUTDOWN: Data saved to checkpoint + S3")
        print(f"# Re-launch instance to resume from checkpoint")
        print(f"{'#'*70}")
    else:
        print(f"\n{'#'*70}")
        print(f"# ALL TARGETED SCANS COMPLETE")
        print(f"{'#'*70}")


if __name__ == '__main__':
    main()
