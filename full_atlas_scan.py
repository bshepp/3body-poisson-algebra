#!/usr/bin/env python3
"""
Full shape-sphere atlas scanner with corrected SVD gap analysis and
mass-aware momentum sampling.

Generates NxN global rank maps over the (mu, phi) shape space for any
3-body configuration defined in expansion_configs or via CLI flags.

Usage examples:
    python full_atlas_scan.py --resolution 100 --potential 1/r --samples 400
    python full_atlas_scan.py --resolution 50 --scenario sun_earth_moon
    python full_atlas_scan.py --resolution 100 --potential 1/r --charges 2 -1 -1
    python full_atlas_scan.py --resolution 200 --potential yukawa --yukawa-mu 0.7
    python full_atlas_scan.py --resolution 100 --all-scenarios
"""

import argparse
import json
import os
import sys
import signal
import multiprocessing
import numpy as np
from time import time, strftime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stability_atlas import (
    AtlasConfig, PoissonAlgebra, ShapeSpace,
)

_shutdown_requested = False

# Module-level globals for multiprocessing workers (inherited via fork)
_worker_algebra = None
_worker_level = 3
_worker_n_samples = 400


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[SIGTERM] Graceful shutdown requested — finishing current row...")


def _compute_cell(args):
    """Worker function for parallel grid evaluation."""
    j, mu, phi = args
    positions = ShapeSpace.shape_to_positions(mu, phi)
    try:
        rank, svs, info = _worker_algebra.compute_rank_at_configuration(
            positions, _worker_level, n_samples=_worker_n_samples)
        return j, rank, info.get("max_gap_ratio", 0)
    except Exception:
        return j, -1, 0.0


def _compute_cell_adaptive(args):
    """Worker function for parallel adaptive grid evaluation."""
    j, mu, phi, eps_range, n_eps = args
    positions = ShapeSpace.shape_to_positions(mu, phi)
    try:
        rank, svs, info = _worker_algebra.compute_adaptive_rank(
            positions, _worker_level,
            eps_range=eps_range, n_eps=n_eps,
            n_samples=_worker_n_samples)
        return (j, rank, info.get("max_gap_ratio", 0),
                info.get("optimal_eps", 0), info.get("gap_score", 0),
                info.get("tiers", []))
    except Exception:
        return j, -1, 0.0, 0.0, 0.0, []


def scenario_to_config(scenario_dict, resolution, samples, level):
    """Convert an expansion_configs SCENARIOS entry to AtlasConfig params."""
    masses_dict = scenario_dict.get("masses", {1: 1, 2: 1, 3: 1})
    masses = tuple(float(masses_dict[k]) for k in sorted(masses_dict.keys()))

    charges = None
    charges_dict = scenario_dict.get("charges")
    if charges_dict is not None:
        charges = tuple(charges_dict[k] for k in sorted(charges_dict.keys()))

    potential = scenario_dict["potential"]
    yukawa_mu = None
    pp = scenario_dict.get("potential_params")
    if pp is not None and potential == "yukawa":
        for item in pp:
            if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "mu":
                yukawa_mu = float(item[1])

    return {
        "potential_type": potential,
        "masses": masses,
        "charges": charges,
        "yukawa_mu": yukawa_mu,
        "max_level": level,
        "n_phase_samples": samples,
        "resolution": "custom",
        "grid_sizes": {"custom": resolution},
    }


def output_tag(potential, masses, charges, yukawa_mu):
    """Build a filesystem-safe directory name for this configuration."""
    tag = potential.replace("/", "").replace("^", "")
    if charges is not None:
        tag += "_q" + "_".join(f"{c:+g}" for c in charges)
    if masses != (1.0, 1.0, 1.0) and masses != (1, 1, 1):
        def _m(v):
            s = str(v)
            return s.replace("/", "over").replace(".", "p")
        tag += "_m" + "_".join(_m(v) for v in masses)
    if yukawa_mu is not None:
        tag += f"_mu{yukawa_mu}"
    return tag


def save_checkpoint_atomic(out_dir, last_row, n_rows, extra=None):
    cp_file = os.path.join(out_dir, "checkpoint.json")
    tmp_file = cp_file + ".tmp"
    data = {
        "last_completed_row": last_row,
        "total_rows": n_rows,
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        data.update(extra)
    with open(tmp_file, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, cp_file)


def flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map,
                 optimal_eps_map=None, gap_score_map=None, tier_map=None):
    np.save(os.path.join(out_dir, "mu_vals.npy"), mu_vals)
    np.save(os.path.join(out_dir, "phi_vals.npy"), phi_vals)
    np.save(os.path.join(out_dir, "rank_map.npy"), rank_map)
    np.save(os.path.join(out_dir, "gap_map.npy"), gap_map)
    if optimal_eps_map is not None:
        np.save(os.path.join(out_dir, "optimal_eps_map.npy"), optimal_eps_map)
    if gap_score_map is not None:
        np.save(os.path.join(out_dir, "gap_score_map.npy"), gap_score_map)
    if tier_map is not None:
        np.save(os.path.join(out_dir, "tier_map.npy"), tier_map)


def run_single_scan(config, label, out_dir, adaptive=False,
                    eps_range=(1e-4, 5e-3), n_eps=8, n_workers=1):
    global _shutdown_requested, _worker_algebra, _worker_level, _worker_n_samples

    os.makedirs(out_dir, exist_ok=True)
    grid_n = config.grid_sizes.get(config.resolution, 100)

    print(f"\n{'='*70}")
    print(f"  FULL ATLAS SCAN: {label}")
    print(f"  Grid: {grid_n}x{grid_n}  |  Samples: {config.n_phase_samples}")
    print(f"  Masses: {config.masses}  |  Potential: {config.potential_type}")
    if config.charges:
        print(f"  Charges: {config.charges}")
    if config.yukawa_mu:
        print(f"  Yukawa mu: {config.yukawa_mu}")
    print(f"  Adaptive: {adaptive}  |  Level: {config.max_level}")
    print(f"  Workers: {n_workers}  |  Output: {out_dir}")
    print(f"{'='*70}\n")

    print("  Building symbolic algebra...")
    t_build = time()
    algebra = PoissonAlgebra(config)
    n_gen = algebra._n_generators
    print(f"  Algebra ready ({time()-t_build:.1f}s, {n_gen} generators)\n")

    _worker_algebra = algebra
    _worker_level = config.max_level
    _worker_n_samples = config.n_phase_samples

    mu_range = (0.2, 3.0)
    phi_range = (0.1, np.pi - 0.1)
    mu_vals = np.linspace(mu_range[0], mu_range[1], grid_n)
    phi_vals = np.linspace(phi_range[0], phi_range[1], grid_n)

    rank_map = np.full((grid_n, grid_n), -1, dtype=np.int32)
    gap_map = np.zeros((grid_n, grid_n), dtype=np.float64)
    optimal_eps_map = None
    gap_score_map = None
    tier_map = None

    if adaptive:
        optimal_eps_map = np.zeros((grid_n, grid_n), dtype=np.float64)
        gap_score_map = np.zeros((grid_n, grid_n), dtype=np.float64)
        max_tiers = 8
        tier_map = np.zeros((grid_n, grid_n, max_tiers, 2), dtype=np.float64)

    start_row = 0
    cp_file = os.path.join(out_dir, "checkpoint.json")
    if os.path.exists(cp_file):
        with open(cp_file) as f:
            cp = json.load(f)
        last_done = cp.get("last_completed_row", -1)
        if last_done >= 0:
            print(f"  Resuming from row {last_done + 1} "
                  f"(checkpoint: {cp.get('timestamp', '?')})")
            start_row = last_done + 1
            rank_map = np.load(os.path.join(out_dir, "rank_map.npy"))
            gap_map = np.load(os.path.join(out_dir, "gap_map.npy"))
            if adaptive:
                optimal_eps_map = np.load(
                    os.path.join(out_dir, "optimal_eps_map.npy"))
                gap_score_map = np.load(
                    os.path.join(out_dir, "gap_score_map.npy"))
                tier_map = np.load(os.path.join(out_dir, "tier_map.npy"))

    config_info = {
        "potential": config.potential_type,
        "masses": list(config.masses),
        "charges": list(config.charges) if config.charges else None,
        "yukawa_mu": config.yukawa_mu,
        "level": config.max_level,
        "grid_n": grid_n,
        "n_phase_samples": config.n_phase_samples,
        "adaptive": adaptive,
        "mu_range": list(mu_range),
        "phi_range": list(phi_range),
        "label": label,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    t_scan = time()
    total_points = grid_n * grid_n
    done_before = start_row * grid_n

    use_parallel = n_workers > 1
    pool = None
    if use_parallel:
        pool = multiprocessing.Pool(n_workers)
        print(f"  Worker pool created ({n_workers} processes)\n")

    try:
        for i in range(start_row, grid_n):
            if _shutdown_requested:
                print(f"\n  [SHUTDOWN] Saving at row {i}...")
                flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map,
                             optimal_eps_map, gap_score_map, tier_map)
                save_checkpoint_atomic(out_dir, i - 1, grid_n)
                print(f"  Checkpoint saved. Exiting.")
                return

            row_t0 = time()
            mu = mu_vals[i]

            if use_parallel:
                if adaptive:
                    tasks = [(j, mu, phi_vals[j], eps_range, n_eps)
                             for j in range(grid_n)]
                    results = pool.map(_compute_cell_adaptive, tasks)
                    for res in results:
                        j, rank, gap, opt_eps, gscore, tiers = res
                        rank_map[i, j] = rank
                        gap_map[i, j] = gap
                        optimal_eps_map[i, j] = opt_eps
                        gap_score_map[i, j] = gscore
                        for t_idx, (t_pos, t_ratio) in enumerate(tiers):
                            if t_idx < tier_map.shape[2]:
                                tier_map[i, j, t_idx] = [t_pos, t_ratio]
                else:
                    tasks = [(j, mu, phi_vals[j]) for j in range(grid_n)]
                    results = pool.map(_compute_cell, tasks)
                    for j, rank, gap in results:
                        rank_map[i, j] = rank
                        gap_map[i, j] = gap
            else:
                for j in range(grid_n):
                    phi = phi_vals[j]
                    positions = ShapeSpace.shape_to_positions(mu, phi)

                    try:
                        if adaptive:
                            rank, svs, info = algebra.compute_adaptive_rank(
                                positions, config.max_level,
                                eps_range=eps_range, n_eps=n_eps,
                                n_samples=config.n_phase_samples)
                            rank_map[i, j] = rank
                            gap_map[i, j] = info.get("max_gap_ratio", 0)
                            optimal_eps_map[i, j] = info.get("optimal_eps", 0)
                            gap_score_map[i, j] = info.get("gap_score", 0)
                            tiers = info.get("tiers", [])
                            for t_idx, (t_pos, t_ratio) in enumerate(tiers):
                                if t_idx < tier_map.shape[2]:
                                    tier_map[i, j, t_idx] = [t_pos, t_ratio]
                        else:
                            rank, svs, info = algebra.compute_rank_at_configuration(
                                positions, config.max_level)
                            rank_map[i, j] = rank
                            gap_map[i, j] = info.get("max_gap_ratio", 0)
                    except Exception:
                        rank_map[i, j] = -1
                        gap_map[i, j] = 0

            row_time = time() - row_t0
            done_now = (i + 1) * grid_n
            elapsed = time() - t_scan
            rate = (done_now - done_before) / elapsed if elapsed > 0 else 0
            eta = (total_points - done_now) / rate if rate > 0 else 0

            rank_116 = np.sum(rank_map[i] == 116)
            print(f"  Row {i+1:>4}/{grid_n}  mu={mu:.3f}  "
                  f"rank-116: {rank_116}/{grid_n}  "
                  f"row: {row_time:.1f}s  "
                  f"ETA: {eta/60:.0f}min")

            flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map,
                         optimal_eps_map, gap_score_map, tier_map)
            save_checkpoint_atomic(out_dir, i, grid_n)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    total_time = time() - t_scan
    n_116 = int(np.sum(rank_map == 116))
    n_valid = int(np.sum(rank_map >= 0))

    summary = {
        "label": label,
        "total_points": total_points,
        "valid_points": n_valid,
        "rank_116_count": n_116,
        "rank_116_fraction": n_116 / n_valid if n_valid > 0 else 0,
        "unique_ranks": sorted(set(int(r) for r in rank_map.flatten() if r >= 0)),
        "elapsed_seconds": total_time,
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  DONE: {label}")
    print(f"  rank=116: {n_116}/{n_valid} ({100*n_116/n_valid:.1f}%)")
    print(f"  Unique ranks: {summary['unique_ranks']}")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f}min)\n")


def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)

    parser = argparse.ArgumentParser(
        description="Full shape-sphere atlas scanner for 3-body Poisson algebras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--resolution", type=int, default=100,
                        help="Grid resolution NxN (default: 100)")
    parser.add_argument("--potential", type=str, default=None,
                        help="Potential type (1/r, 1/r^N for any N, harmonic, log, yukawa)")
    parser.add_argument("--masses", nargs=3, type=float, default=None,
                        metavar=("M1", "M2", "M3"),
                        help="Body masses (default: 1 1 1)")
    parser.add_argument("--charges", nargs=3, type=int, default=None,
                        metavar=("Q1", "Q2", "Q3"),
                        help="Body charges for Coulomb potentials")
    parser.add_argument("--yukawa-mu", type=float, default=None,
                        help="Yukawa screening parameter")
    parser.add_argument("--samples", type=int, default=400,
                        help="Phase-space samples per grid point (default: 400)")
    parser.add_argument("--level", type=int, default=3,
                        help="Max bracket level (default: 3)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive epsilon sweep")
    parser.add_argument("--eps-range", nargs=2, type=float, default=None,
                        metavar=("EPS_MIN", "EPS_MAX"),
                        help="Epsilon range for adaptive mode (default: 1e-4 5e-3)")
    parser.add_argument("--n-eps", type=int, default=8,
                        help="Number of epsilon candidates for adaptive (default: 8)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Named scenario from expansion_configs")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run all atlas-enabled scenarios")
    parser.add_argument("--output-dir", type=str, default="atlas_full",
                        help="Base output directory (default: atlas_full)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")

    args = parser.parse_args()

    n_workers = args.workers if args.workers else (os.cpu_count() or 1)

    eps_range = tuple(args.eps_range) if args.eps_range else (1e-4, 5e-3)

    if args.all_scenarios:
        from nbody.expansion_configs import SCENARIOS, get_atlas_scenarios
        for key in get_atlas_scenarios():
            sc = SCENARIOS[key]
            cfg_kw = scenario_to_config(sc, args.resolution, args.samples,
                                        args.level)
            config = AtlasConfig(**cfg_kw)
            tag = output_tag(config.potential_type, config.masses,
                             config.charges, config.yukawa_mu)
            out_dir = os.path.join(args.output_dir, f"{key}_{tag}")
            run_single_scan(config, sc["label"], out_dir,
                            adaptive=args.adaptive,
                            eps_range=eps_range, n_eps=args.n_eps,
                            n_workers=n_workers)
            if _shutdown_requested:
                break

    elif args.scenario:
        from nbody.expansion_configs import SCENARIOS
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        sc = SCENARIOS[args.scenario]
        cfg_kw = scenario_to_config(sc, args.resolution, args.samples,
                                    args.level)
        config = AtlasConfig(**cfg_kw)
        tag = output_tag(config.potential_type, config.masses,
                         config.charges, config.yukawa_mu)
        out_dir = os.path.join(args.output_dir, f"{args.scenario}_{tag}")
        run_single_scan(config, sc["label"], out_dir,
                        adaptive=args.adaptive,
                        eps_range=eps_range, n_eps=args.n_eps,
                        n_workers=n_workers)

    elif args.potential:
        masses = tuple(args.masses) if args.masses else (1.0, 1.0, 1.0)
        charges = tuple(args.charges) if args.charges else None
        config = AtlasConfig(
            potential_type=args.potential,
            masses=masses,
            charges=charges,
            yukawa_mu=args.yukawa_mu,
            max_level=args.level,
            n_phase_samples=args.samples,
            resolution="custom",
            grid_sizes={"custom": args.resolution},
        )
        tag = output_tag(args.potential, masses, charges, args.yukawa_mu)
        label = f"{args.potential}"
        if charges:
            label += f" charges={charges}"
        if masses != (1.0, 1.0, 1.0):
            label += f" masses={masses}"
        out_dir = os.path.join(args.output_dir, tag)
        run_single_scan(config, label, out_dir,
                        adaptive=args.adaptive,
                        eps_range=eps_range, n_eps=args.n_eps,
                        n_workers=n_workers)

    else:
        parser.print_help()
        print("\nSpecify --potential, --scenario, or --all-scenarios")
        sys.exit(1)


if __name__ == "__main__":
    main()
