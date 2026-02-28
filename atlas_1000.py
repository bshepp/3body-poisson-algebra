"""
1000x1000 Atlas: Parallelized High-Resolution Shape Sphere Scan

Designed for distributed execution across multiple EC2 spot instances.
Each instance scans a range of rows (mu values) using multiprocessing
to parallelize across grid points within each row.

Usage:
    # Scan rows 0-99 on this machine with 15 workers
    python atlas_1000.py scan --start-row 0 --end-row 100 --workers 15

    # Merge all blocks into final 1000x1000 arrays
    python atlas_1000.py merge

    # Scan with non-default parameters
    python atlas_1000.py scan --potential "1/r2" --epsilon 1e-3 --grid-n 500
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from time import time
from multiprocessing import Pool, TimeoutError as MPTimeoutError

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

S3_BUCKET = os.environ.get('S3_BUCKET', '')
OUTPUT_BASE = 'atlas_1000'

MU_RANGE = (0.05, 5.0)
PHI_RANGE = (0.05, np.pi - 0.05)
N_SAMPLES = 400
LEVEL = 3

# Module-level reference set by the main process before forking
_algebra = None
_level = LEVEL


def s3_upload(local_path, s3_key):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(
            ['aws', 's3', 'cp', local_path,
             f's3://{S3_BUCKET}/{s3_key}'],
            capture_output=True, timeout=120)
    except Exception:
        pass


def s3_sync(local_dir, s3_prefix):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(
            ['aws', 's3', 'sync', local_dir,
             f's3://{S3_BUCKET}/{s3_prefix}',
             '--exclude', '*.html', '--exclude', '*.png'],
            capture_output=True, timeout=120)
    except Exception:
        pass


def block_dir(potential_type, start_row, end_row):
    pot_dir = {'1/r': '1_r', '1/r2': '1_r2', 'harmonic': 'harmonic'}
    return os.path.join(OUTPUT_BASE, pot_dir[potential_type],
                        f'block_{start_row:04d}_{end_row:04d}')


def _eval_point(args):
    """Worker function -- called in forked child, uses inherited _algebra."""
    mu, phi, epsilon = args
    from stability_atlas import ShapeSpace
    positions = ShapeSpace.shape_to_positions(mu, phi)
    try:
        rank, svs, info = _algebra.compute_rank_at_configuration(
            positions, _level, epsilon=epsilon)
        return rank, svs, info['max_gap_ratio']
    except Exception:
        return -1, np.array([]), 0.0


def run_scan(args):
    from stability_atlas import AtlasConfig, PoissonAlgebra

    global _algebra, _level
    _level = LEVEL

    potential = args.potential
    epsilon = args.epsilon
    grid_n = args.grid_n
    start_row = args.start_row
    end_row = args.end_row
    n_workers = args.workers

    out = block_dir(potential, start_row, end_row)
    os.makedirs(out, exist_ok=True)
    s3_prefix = out + '/'

    mu_vals = np.linspace(MU_RANGE[0], MU_RANGE[1], grid_n)
    phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], grid_n)

    n_rows = end_row - start_row
    total_points = n_rows * grid_n

    print(f"{'='*70}")
    print(f"  1000x1000 ATLAS SCAN")
    print(f"  Potential: {potential}   Epsilon: {epsilon:.0e}")
    print(f"  Grid: {grid_n}x{grid_n}   Rows: {start_row}-{end_row-1}")
    print(f"  Workers: {n_workers}   Samples/point: {N_SAMPLES}")
    print(f"  Output: {out}/")
    print(f"{'='*70}\n", flush=True)

    # Check for checkpoint
    cp_file = os.path.join(out, 'checkpoint.json')
    done_rows = 0
    if os.path.exists(cp_file):
        with open(cp_file) as f:
            cp = json.load(f)
        done_rows = cp.get('completed_rows', 0)

    if done_rows > 0:
        rank_map = np.load(os.path.join(out, 'rank_map.npy'))
        gap_map = np.load(os.path.join(out, 'gap_map.npy'))
        sv_spectra = np.load(os.path.join(out, 'sv_spectra.npy'))
        print(f"  Resuming from local row {done_rows} "
              f"({done_rows * grid_n} points done)\n", flush=True)
    else:
        rank_map = None
        gap_map = None
        sv_spectra = None

    # Build algebra once in main process
    print(f"  Building symbolic algebra...", flush=True)
    t_build = time()
    config = AtlasConfig(
        potential_type=potential,
        max_level=LEVEL,
        n_phase_samples=N_SAMPLES,
        epsilon=epsilon,
        svd_gap_threshold=1e4,
    )
    _algebra = PoissonAlgebra(config)
    n_gen = _algebra._n_generators
    print(f"  Algebra ready ({time()-t_build:.1f}s, {n_gen} generators)\n",
          flush=True)

    if rank_map is None:
        rank_map = np.zeros((n_rows, grid_n), dtype=int)
        gap_map = np.zeros((n_rows, grid_n))
        sv_spectra = np.zeros((n_rows, grid_n, n_gen), dtype=np.float64)

    np.save(os.path.join(out, 'mu_vals.npy'), mu_vals[start_row:end_row])
    np.save(os.path.join(out, 'phi_vals.npy'), phi_vals)

    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({
            'potential': potential, 'grid_n': grid_n,
            'epsilon': epsilon, 'n_samples': N_SAMPLES,
            'level': LEVEL, 'n_generators': n_gen,
            'mu_range': list(MU_RANGE), 'phi_range': list(PHI_RANGE),
            'start_row': start_row, 'end_row': end_row,
        }, f, indent=2)

    point_timeout = args.timeout
    print(f"  Per-point timeout: {point_timeout}s\n", flush=True)

    # Fork workers AFTER algebra is built
    pool = Pool(processes=n_workers)

    t_scan = time()
    for local_i in range(done_rows, n_rows):
        global_i = start_row + local_i
        mu = mu_vals[global_i]
        t_row = time()

        work = [(mu, phi_vals[j], epsilon) for j in range(grid_n)]
        async_results = [pool.apply_async(_eval_point, (w,)) for w in work]

        n_timeout = 0
        for j, ar in enumerate(async_results):
            try:
                rank, svs, gap = ar.get(timeout=point_timeout)
            except MPTimeoutError:
                rank, svs, gap = -1, np.array([]), 0.0
                n_timeout += 1
            except Exception:
                rank, svs, gap = -1, np.array([]), 0.0
            rank_map[local_i, j] = rank
            gap_map[local_i, j] = gap
            if len(svs) > 0:
                sv_spectra[local_i, j, :len(svs)] = svs

        # Save after every row
        np.save(os.path.join(out, 'rank_map.npy'), rank_map)
        np.save(os.path.join(out, 'gap_map.npy'), gap_map)
        np.save(os.path.join(out, 'sv_spectra.npy'), sv_spectra)

        with open(cp_file, 'w') as f:
            json.dump({
                'completed_rows': local_i + 1,
                'n_generators': n_gen,
                'epsilon': epsilon,
                'start_row': start_row,
                'end_row': end_row,
                'timestamp': __import__('datetime').datetime.utcnow()
                             .strftime('%Y-%m-%d %H:%M:%S'),
            }, f)

        if (local_i + 1) % 10 == 0:
            s3_sync(out, s3_prefix)

        done = (local_i + 1) * grid_n
        elapsed = time() - t_scan
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total_points - done) / rate if rate > 0 else 0
        row_time = time() - t_row

        timeout_str = f"  timeouts={n_timeout}" if n_timeout else ""
        print(f"  Row {local_i+1:4d}/{n_rows}  "
              f"(global {global_i:4d})  mu={mu:.4f}  "
              f"[{done:7d}/{total_points}]  "
              f"row={row_time:.1f}s  "
              f"ETA={remaining/60:.0f}m  "
              f"ranks=[{rank_map[local_i,:].min()},{rank_map[local_i,:].max()}]  "
              f"gap=[{gap_map[local_i,:].min():.1e},{gap_map[local_i,:].max():.1e}]"
              f"{timeout_str}",
              flush=True)

    pool.close()
    pool.join()

    # Final sync
    s3_sync(out, s3_prefix)

    total_time = time() - t_scan
    print(f"\n{'='*70}")
    print(f"  BLOCK COMPLETE: rows {start_row}-{end_row-1}")
    print(f"  Time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Rank range: [{rank_map.min()}, {rank_map.max()}]")
    print(f"  Gap range:  [{gap_map.min():.2e}, {gap_map.max():.2e}]")
    print(f"{'='*70}", flush=True)


def run_merge(args):
    potential = args.potential
    grid_n = args.grid_n
    pot_dir = {'1/r': '1_r', '1/r2': '1_r2', 'harmonic': 'harmonic'}[potential]
    base = os.path.join(OUTPUT_BASE, pot_dir)

    # Download from S3 if available
    if S3_BUCKET:
        print(f"  Downloading blocks from S3...", flush=True)
        subprocess.run(
            ['aws', 's3', 'sync',
             f's3://{S3_BUCKET}/{base}/', base + '/'],
            timeout=600)

    # Find all block directories
    blocks = sorted([
        d for d in os.listdir(base)
        if d.startswith('block_') and os.path.isdir(os.path.join(base, d))
    ])

    if not blocks:
        print("  ERROR: No block directories found!")
        sys.exit(1)

    print(f"  Found {len(blocks)} blocks: {blocks[0]} ... {blocks[-1]}")

    # Verify all blocks are complete
    all_rank = []
    all_gap = []
    all_sv = []
    total_rows = 0

    for bdir in blocks:
        bpath = os.path.join(base, bdir)
        cp_file = os.path.join(bpath, 'checkpoint.json')
        if not os.path.exists(cp_file):
            print(f"  WARNING: {bdir} has no checkpoint -- skipping")
            continue
        with open(cp_file) as f:
            cp = json.load(f)
        cfg_file = os.path.join(bpath, 'config.json')
        with open(cfg_file) as f:
            cfg = json.load(f)
        expected = cfg['end_row'] - cfg['start_row']
        actual = cp['completed_rows']
        if actual < expected:
            print(f"  WARNING: {bdir} incomplete ({actual}/{expected} rows) "
                  f"-- including partial data")

        r = np.load(os.path.join(bpath, 'rank_map.npy'))[:actual]
        g = np.load(os.path.join(bpath, 'gap_map.npy'))[:actual]
        s = np.load(os.path.join(bpath, 'sv_spectra.npy'))[:actual]
        all_rank.append(r)
        all_gap.append(g)
        all_sv.append(s)
        total_rows += actual
        status = "OK" if actual == expected else f"PARTIAL ({actual}/{expected})"
        print(f"  {bdir}: {actual} rows {status}")

    if total_rows != grid_n:
        print(f"\n  WARNING: merged {total_rows} rows, expected {grid_n}")

    rank_map = np.concatenate(all_rank, axis=0)
    gap_map = np.concatenate(all_gap, axis=0)
    sv_spectra = np.concatenate(all_sv, axis=0)

    mu_vals = np.linspace(MU_RANGE[0], MU_RANGE[1], grid_n)
    phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], grid_n)

    merged_dir = os.path.join(base, 'merged')
    os.makedirs(merged_dir, exist_ok=True)

    np.save(os.path.join(merged_dir, 'rank_map.npy'), rank_map)
    np.save(os.path.join(merged_dir, 'gap_map.npy'), gap_map)
    np.save(os.path.join(merged_dir, 'sv_spectra.npy'), sv_spectra)
    np.save(os.path.join(merged_dir, 'mu_vals.npy'), mu_vals)
    np.save(os.path.join(merged_dir, 'phi_vals.npy'), phi_vals)

    with open(os.path.join(merged_dir, 'config.json'), 'w') as f:
        json.dump({
            'potential': potential, 'grid_n': grid_n,
            'n_generators': sv_spectra.shape[2],
            'mu_range': list(MU_RANGE), 'phi_range': list(PHI_RANGE),
            'total_rows': total_rows, 'n_blocks': len(blocks),
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  MERGE COMPLETE")
    print(f"  Shape: {rank_map.shape}")
    print(f"  Rank range: [{rank_map.min()}, {rank_map.max()}]")
    print(f"  Gap range:  [{gap_map.min():.2e}, {gap_map.max():.2e}]")
    print(f"  Output: {merged_dir}/")
    print(f"{'='*70}")

    if S3_BUCKET:
        s3_sync(merged_dir, f'{base}/merged/')
        print(f"  Uploaded merged results to S3")


def main():
    parser = argparse.ArgumentParser(
        description='1000x1000 Atlas: Parallelized High-Resolution Scan')
    parser.add_argument('mode', choices=['scan', 'merge'],
                        help='scan = compute a block of rows; '
                             'merge = stitch blocks together')
    parser.add_argument('--potential', type=str, default='1/r',
                        choices=['1/r', '1/r2', 'harmonic'])
    parser.add_argument('--epsilon', type=float, default=5e-3)
    parser.add_argument('--grid-n', type=int, default=1000)
    parser.add_argument('--start-row', type=int, default=0)
    parser.add_argument('--end-row', type=int, default=100)
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument('--timeout', type=int, default=600,
                        help='Per-point timeout in seconds (default 600)')
    args = parser.parse_args()

    if args.mode == 'scan':
        run_scan(args)
    elif args.mode == 'merge':
        run_merge(args)


if __name__ == '__main__':
    main()
