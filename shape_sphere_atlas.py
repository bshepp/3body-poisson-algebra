"""
Shape Sphere Atlas — direct sampling on S² (Hsiang–Montgomery / Iwai chart).

Companion to ``atlas_1000.py``.  Where that script samples the Poisson algebra
rank/gap on the (μ, φ) chart of the equal-mass three-body shape space, this
script samples on the canonical *shape sphere* itself by inverting the Jacobi
map.  It therefore covers all three binary-collision points symmetrically,
including the r₁=r₂ cap that the (μ, φ) chart sends to μ→∞.

Math (equal masses, hyperradius R = 1, COM at origin):
    Given (s1, s2, s3) on the unit sphere, recover Jacobi vectors via
        |ρ1|² = (1 + s1)/2,    |ρ2|² = (1 - s1)/2,
        ρ1 = (|ρ1|, 0),
        ρ2 = ( s2/(2|ρ1|),  s3/(2|ρ1|) ),
    then
        r1 = -ρ1/2 - (√3/6) ρ2
        r2 = +ρ1/2 - (√3/6) ρ2
        r3 =        (√3/3) ρ2
    (Derivation: r2-r1 = ρ1; r3 - (r1+r2)/2 = (√3/2) ρ2; r1+r2+r3 = 0.)

    The s1 → -1 limit (binary r1=r2) is the degenerate point; the worker
    returns (-1, [], 0.0) like all other singularities in the existing atlas.

Sampling grid: latitude/longitude (θ, φ_sph) on S² with
    s1 = sin θ cos φ_sph,  s2 = sin θ sin φ_sph,  s3 = cos θ.
Trades minor pole oversampling for drop-in compatibility with the existing
Plotly heatmap / animation infrastructure on the website.

Usage:
    python shape_sphere_atlas.py scan --start-row 0 --end-row 100 --workers 8
    python shape_sphere_atlas.py merge --potential 1/r
    python shape_sphere_atlas.py to-json --potential 1/r --epsilon 1e-3
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
OUTPUT_BASE = 'shape_sphere_atlas'
LEVEL = 3
N_SAMPLES = 400

# Theta excludes the exact poles (well-defined but degenerate Jacobi frame at
# the equilateral; trivially extendable but we skip a thin band for safety).
THETA_RANGE = (0.02, np.pi - 0.02)
# Full longitude range — the missing r1=r2 cap sits at φ_sph = π.
PHI_RANGE = (0.0, 2.0 * np.pi)

SQRT3 = np.sqrt(3.0)

_algebra = None
_level = LEVEL


# --------------------------------------------------------------------------- #
# Geometry — sphere ↔ positions                                               #
# --------------------------------------------------------------------------- #

def thetaphi_to_sphere(theta, phi_sph):
    """Lat/lon (colatitude, azimuth) → (s1, s2, s3) on the unit sphere."""
    st, ct = np.sin(theta), np.cos(theta)
    return st * np.cos(phi_sph), st * np.sin(phi_sph), ct


def sphere_to_positions(s1, s2, s3):
    """Invert the Jacobi map at hyperradius R=1, equal masses, COM at origin.

    Returns a (3, 2) ndarray ``[[x1, y1], [x2, y2], [x3, y3]]`` or ``None``
    if the configuration is too close to the binary r1=r2 collision (s1≈-1).
    """
    rho1_sq = 0.5 * (1.0 + s1)
    if rho1_sq <= 1e-12:
        return None
    rho1 = np.sqrt(rho1_sq)
    rho1_vec = np.array([rho1, 0.0])
    rho2_vec = np.array([s2 / (2.0 * rho1), s3 / (2.0 * rho1)])

    a = SQRT3 / 6.0
    b = SQRT3 / 3.0
    r1 = -0.5 * rho1_vec - a * rho2_vec
    r2 = +0.5 * rho1_vec - a * rho2_vec
    r3 = b * rho2_vec
    return np.stack([r1, r2, r3])


# --------------------------------------------------------------------------- #
# S3 helpers (copied from atlas_1000.py to keep this driver standalone)       #
# --------------------------------------------------------------------------- #

def s3_upload(local_path, s3_key):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(['aws', 's3', 'cp', local_path,
                        f's3://{S3_BUCKET}/{s3_key}'],
                       capture_output=True, timeout=120)
    except Exception:
        pass


def s3_sync(local_dir, s3_prefix):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(['aws', 's3', 'sync', local_dir,
                        f's3://{S3_BUCKET}/{s3_prefix}',
                        '--exclude', '*.html', '--exclude', '*.png'],
                       capture_output=True, timeout=120)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Layout                                                                      #
# --------------------------------------------------------------------------- #

_POT_DIR = {'1/r': '1_r', '1/r2': '1_r2', '1/r3': '1_r3', 'harmonic': 'harmonic'}


def block_dir(potential_type, start_row, end_row):
    return os.path.join(OUTPUT_BASE, _POT_DIR[potential_type],
                        f'block_{start_row:04d}_{end_row:04d}')


def merged_path(potential_type):
    return os.path.join(OUTPUT_BASE, _POT_DIR[potential_type], 'atlas.npz')


# --------------------------------------------------------------------------- #
# Worker                                                                      #
# --------------------------------------------------------------------------- #

def _eval_point(args):
    """Inherited-fork worker: evaluate algebra rank at one (θ, φ_sph)."""
    theta, phi_sph, epsilon = args
    s1, s2, s3 = thetaphi_to_sphere(theta, phi_sph)
    positions = sphere_to_positions(s1, s2, s3)
    if positions is None:
        return -1, np.array([]), 0.0
    try:
        rank, svs, info = _algebra.compute_rank_at_configuration(
            positions, _level, epsilon=epsilon)
        return rank, svs, info['max_gap_ratio']
    except Exception:
        return -1, np.array([]), 0.0


# --------------------------------------------------------------------------- #
# Scan                                                                        #
# --------------------------------------------------------------------------- #

def run_scan(args):
    from stability_atlas import AtlasConfig, PoissonAlgebra
    global _algebra, _level
    _level = LEVEL

    pot = args.potential
    if pot not in _POT_DIR:
        raise SystemExit(f"Unknown potential {pot!r}; supported: {list(_POT_DIR)}")

    n_theta = args.grid_theta
    n_phi = args.grid_phi
    eps = args.epsilon
    sr, er = args.start_row, args.end_row
    workers = args.workers

    out = block_dir(pot, sr, er)
    os.makedirs(out, exist_ok=True)

    theta_vals = np.linspace(THETA_RANGE[0], THETA_RANGE[1], n_theta)
    phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], n_phi, endpoint=False)

    n_rows = er - sr
    total_pts = n_rows * n_phi

    print(f"{'='*70}")
    print(f"  SHAPE SPHERE ATLAS SCAN")
    print(f"  Potential={pot}   Epsilon={eps:.0e}")
    print(f"  Grid: theta x phi_sph = {n_theta} x {n_phi}")
    print(f"  Rows {sr}-{er-1}    Workers={workers}    Samples/pt={N_SAMPLES}")
    print(f"  Output: {out}/")
    print(f"{'='*70}\n", flush=True)

    cp_file = os.path.join(out, 'checkpoint.json')
    done_rows = 0
    if os.path.exists(cp_file):
        with open(cp_file) as f:
            done_rows = json.load(f).get('completed_rows', 0)

    print("  Building symbolic algebra...", flush=True)
    t0 = time()
    cfg = AtlasConfig(potential_type=pot, max_level=LEVEL,
                      n_phase_samples=N_SAMPLES, epsilon=eps,
                      svd_gap_threshold=1e4)
    _algebra = PoissonAlgebra(cfg)
    n_gen = _algebra._n_generators
    print(f"  Algebra ready ({time()-t0:.1f}s, {n_gen} generators)\n",
          flush=True)

    if done_rows > 0:
        rank_map = np.load(os.path.join(out, 'rank_map.npy'))
        gap_map = np.load(os.path.join(out, 'gap_map.npy'))
        sv_spectra = np.load(os.path.join(out, 'sv_spectra.npy'))
        print(f"  Resuming at local row {done_rows}\n", flush=True)
    else:
        rank_map = np.zeros((n_rows, n_phi), dtype=int)
        gap_map = np.zeros((n_rows, n_phi))
        sv_spectra = np.zeros((n_rows, n_phi, n_gen), dtype=np.float64)

    np.save(os.path.join(out, 'theta_vals.npy'), theta_vals[sr:er])
    np.save(os.path.join(out, 'phi_vals.npy'), phi_vals)

    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({
            'schema': 'shape_sphere_v1',
            'potential': pot, 'epsilon': eps,
            'n_samples': N_SAMPLES, 'level': LEVEL,
            'n_generators': n_gen,
            'grid_theta': n_theta, 'grid_phi': n_phi,
            'theta_range': list(THETA_RANGE),
            'phi_range': list(PHI_RANGE),
            'start_row': sr, 'end_row': er,
        }, f, indent=2)

    use_pool = workers > 1
    pool = Pool(processes=workers) if use_pool else None
    timeout = args.timeout
    t_scan = time()
    try:
        for li in range(done_rows, n_rows):
            gi = sr + li
            theta = theta_vals[gi]
            t_row = time()
            work = [(theta, phi_vals[j], eps) for j in range(n_phi)]
            n_to = 0
            if use_pool:
                ar = [pool.apply_async(_eval_point, (w,)) for w in work]
                results = []
                for a in ar:
                    try:
                        results.append(a.get(timeout=timeout))
                    except MPTimeoutError:
                        results.append((-1, np.array([]), 0.0))
                        n_to += 1
                    except Exception:
                        results.append((-1, np.array([]), 0.0))
            else:
                results = [_eval_point(w) for w in work]
            for j, (rank, svs, gap) in enumerate(results):
                rank_map[li, j] = rank
                gap_map[li, j] = gap
                if len(svs):
                    sv_spectra[li, j, :len(svs)] = svs

            np.save(os.path.join(out, 'rank_map.npy'), rank_map)
            np.save(os.path.join(out, 'gap_map.npy'), gap_map)
            np.save(os.path.join(out, 'sv_spectra.npy'), sv_spectra)
            with open(cp_file, 'w') as f:
                json.dump({'completed_rows': li + 1, 'n_generators': n_gen,
                           'epsilon': eps, 'start_row': sr, 'end_row': er},
                          f)
            if (li + 1) % 10 == 0:
                s3_sync(out, out + '/')

            done = (li + 1) * n_phi
            elapsed = time() - t_scan
            rate = done / max(elapsed, 1e-9)
            eta = (total_pts - done) / rate if rate else 0
            tos = f"  timeouts={n_to}" if n_to else ""
            print(f"  Row {li+1:4d}/{n_rows}  (gi={gi:4d})  θ={theta:.4f}  "
                  f"[{done:7d}/{total_pts}]  row={time()-t_row:.1f}s  "
                  f"ETA={eta/60:.0f}m  ranks=[{rank_map[li].min()},"
                  f"{rank_map[li].max()}]  gap=[{gap_map[li].min():.1e},"
                  f"{gap_map[li].max():.1e}]{tos}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    s3_sync(out, out + '/')
    print(f"\n  Done. Block at {out}/", flush=True)


# --------------------------------------------------------------------------- #
# Merge blocks                                                                #
# --------------------------------------------------------------------------- #

def run_merge(args):
    pot = args.potential
    if pot not in _POT_DIR:
        raise SystemExit(f"Unknown potential {pot!r}")
    base = os.path.join(OUTPUT_BASE, _POT_DIR[pot])
    if not os.path.isdir(base):
        raise SystemExit(f"No blocks dir at {base}")

    blocks = sorted(d for d in os.listdir(base) if d.startswith('block_'))
    if not blocks:
        raise SystemExit(f"No blocks in {base}")

    cfg = None
    rank_rows, gap_rows, sv_rows, theta_rows = [], [], [], []
    phi_vals = None
    for bd in blocks:
        bp = os.path.join(base, bd)
        with open(os.path.join(bp, 'config.json')) as f:
            cfg_b = json.load(f)
        if cfg is None:
            cfg = cfg_b
            phi_vals = np.load(os.path.join(bp, 'phi_vals.npy'))
        rank_rows.append(np.load(os.path.join(bp, 'rank_map.npy')))
        gap_rows.append(np.load(os.path.join(bp, 'gap_map.npy')))
        sv_rows.append(np.load(os.path.join(bp, 'sv_spectra.npy')))
        theta_rows.append(np.load(os.path.join(bp, 'theta_vals.npy')))

    rank_map = np.concatenate(rank_rows, axis=0)
    gap_map = np.concatenate(gap_rows, axis=0)
    sv_spectra = np.concatenate(sv_rows, axis=0)
    theta_vals = np.concatenate(theta_rows, axis=0)

    out = merged_path(pot)
    np.savez(out, rank_map=rank_map, gap_map=gap_map,
             sv_spectra=sv_spectra, theta_vals=theta_vals,
             phi_vals=phi_vals, config=json.dumps(cfg))
    print(f"  Merged {len(blocks)} blocks -> {out}")
    print(f"  Shape: rank_map={rank_map.shape}  sv={sv_spectra.shape}")


# --------------------------------------------------------------------------- #
# JSON for the website                                                        #
# --------------------------------------------------------------------------- #

def _eps_tag(eps):
    if eps is None:
        return ''
    return f"_eps_{eps:.0e}".replace('e-0', 'em').replace('e-', 'em')


def run_to_json(args):
    pot = args.potential
    src = args.input or merged_path(pot)
    z = np.load(src, allow_pickle=True)
    cfg = json.loads(str(z['config']))
    rank_map = z['rank_map']
    gap_map = z['gap_map']
    theta_vals = z['theta_vals']
    phi_vals = z['phi_vals']

    payload = {
        'schema': 'shape_sphere_v1',
        'theta': theta_vals.tolist(),
        'phi_sph': phi_vals.tolist(),
        'rank': rank_map.astype(int).tolist(),
        'gap': [[(float(v) if np.isfinite(v) else None) for v in row]
                for row in gap_map],
        'grid_theta': int(rank_map.shape[0]),
        'grid_phi': int(rank_map.shape[1]),
        'potential': cfg.get('potential', pot),
        'epsilon': cfg.get('epsilon'),
        'level': cfg.get('level', LEVEL),
        'n_generators': cfg.get('n_generators'),
    }

    out = args.output
    if out is None:
        eps = cfg.get('epsilon')
        out = os.path.join('website', 'data',
                           f'atlas_sphere_{_POT_DIR[pot]}{_eps_tag(eps)}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(payload, f)
    print(f"  Wrote {out}  ({os.path.getsize(out)/1024:.1f} KB)")
    print(f"  Schema: shape_sphere_v1  shape: {rank_map.shape}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    sc = sub.add_parser('scan', help='Run a row-block scan')
    sc.add_argument('--potential', default='1/r',
                    choices=list(_POT_DIR))
    sc.add_argument('--epsilon', type=float, default=1e-3)
    sc.add_argument('--grid-theta', type=int, default=200)
    sc.add_argument('--grid-phi', type=int, default=400)
    sc.add_argument('--start-row', type=int, default=0)
    sc.add_argument('--end-row', type=int, default=200)
    sc.add_argument('--workers', type=int, default=4)
    sc.add_argument('--timeout', type=float, default=60.0,
                    help='Per-point worker timeout (seconds)')
    sc.set_defaults(func=run_scan)

    mg = sub.add_parser('merge', help='Concatenate blocks into atlas.npz')
    mg.add_argument('--potential', default='1/r', choices=list(_POT_DIR))
    mg.set_defaults(func=run_merge)

    js = sub.add_parser('to-json', help='Produce website JSON from atlas.npz')
    js.add_argument('--potential', default='1/r', choices=list(_POT_DIR))
    js.add_argument('--epsilon', type=float, default=None,
                    help='(metadata only — read from .npz config)')
    js.add_argument('--input')
    js.add_argument('--output')
    js.set_defaults(func=run_to_json)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
