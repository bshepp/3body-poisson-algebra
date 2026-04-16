#!/usr/bin/env python3
"""
N=4 Atlas: 1D slices through the shape space of four bodies on a line.

For N=4 d=1, after removing translation and scale the shape space is
2-dimensional.  We fix one body at 0, another at 1, and parameterize
the remaining two positions (s, t) with s < t and s,t not colliding
with the fixed bodies.

Slices:
  A – Fix t=2.0, sweep s in (0,1) exclusive  (100 points)
  B – Fix s=0.5, sweep t in (1,5) exclusive   (100 points)
  C – Equal-spacing: bodies at 0, d, 2d, 3d, sweep d in (0.3, 3.0) (100 points)

For each configuration we evaluate pre-compiled L0-L2 generators at
local phase-space samples and determine the SVD rank.  Generic rank
should be 62.  Any drop indicates symmetry enhancement at that shape.

Output: results/n4_atlas_1d.json
"""

import argparse
import json
import os
import sys
import numpy as np
from time import time, strftime
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbody"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exact_growth_nbody import NBodyAlgebra


def local_sample_n4_d1(positions, n_samples, epsilon, mom_range=0.5,
                        min_sep=0.1, seed=None):
    """Sample phase space locally around a 4-body 1D configuration.

    positions: (4,) array of 1D positions
    Returns (Z_qp, Z_u) where Z_qp is (n, 8) [q1..q4, p1..p4]
    and Z_u is (n, 6) [1/r_ij for all 6 pairs].
    """
    N = 4
    d = 1
    rng = np.random.RandomState(seed)
    base_q = np.array(positions, dtype=float).ravel()

    pairs = list(combinations(range(N), 2))
    dists = [abs(base_q[i] - base_q[j]) for i, j in pairs]
    r_min = min(dists)
    eps = min(epsilon, 0.1 * r_min)
    eff_min_sep = min(min_sep, 0.3 * r_min)

    n_phase = 2 * N * d  # 8
    n_pairs = len(pairs)  # 6
    Z_qp = np.zeros((n_samples, n_phase))
    Z_u = np.zeros((n_samples, n_pairs))
    accepted = 0

    for _ in range(n_samples * 300):
        if accepted >= n_samples:
            break

        q = base_q + rng.randn(N * d) * eps
        p = rng.randn(N * d) * mom_range

        ok = True
        u_vals = np.zeros(n_pairs)
        for k, (i, j) in enumerate(pairs):
            r = abs(q[i] - q[j])
            if r < eff_min_sep:
                ok = False
                break
            u_vals[k] = 1.0 / r

        if not ok:
            continue

        Z_qp[accepted, :N * d] = q
        Z_qp[accepted, N * d:] = p
        Z_u[accepted] = u_vals
        accepted += 1

    return Z_qp[:accepted], Z_u[:accepted]


def rank_from_gap(singular_values):
    """Determine rank from SVD gap (same logic as PoissonAlgebra)."""
    s = singular_values
    if len(s) <= 1:
        return len(s), 1.0

    noise_threshold = 1e-8 * s[0]
    n_meaningful = int(np.sum(s > noise_threshold))

    best_gap_ratio = 1.0
    best_gap_idx = -1
    for i in range(min(n_meaningful, len(s) - 1)):
        if s[i + 1] > noise_threshold:
            gap = s[i] / s[i + 1]
        else:
            gap = s[i] / max(s[i + 1], 1e-300)
        if gap > best_gap_ratio:
            best_gap_ratio = gap
            best_gap_idx = i

    below_noise = (best_gap_idx >= 0 and
                   best_gap_idx + 1 < len(s) and
                   s[best_gap_idx + 1] < noise_threshold)

    if best_gap_ratio > 1e4 and below_noise:
        rank = best_gap_idx + 1
    elif best_gap_ratio > 1e4:
        rel_below = s[best_gap_idx + 1] / s[0] if s[0] > 0 else 0
        if rel_below < 1e-6:
            rank = best_gap_idx + 1
        else:
            rank = n_meaningful
    else:
        rank = n_meaningful

    return rank, best_gap_ratio


def compute_rank_at_config(evaluator, levels, positions,
                           n_samples=500, epsilon=0.05, max_level=2,
                           seed=None):
    """Evaluate algebra generators at local samples and return SVD rank."""
    Z_qp, Z_u = local_sample_n4_d1(
        positions, n_samples, epsilon, seed=seed)

    if Z_qp.shape[0] < 50:
        return -1, 0.0, Z_qp.shape[0]

    full_matrix = evaluator(Z_qp, Z_u)

    col_mask = [i for i, lv in enumerate(levels) if lv <= max_level]
    if not col_mask:
        return 0, 0.0, Z_qp.shape[0]

    sub = full_matrix[:, col_mask]

    norms = np.linalg.norm(sub, axis=0)
    norms[norms < 1e-15] = 1.0
    sub = sub / norms

    U, S, Vt = np.linalg.svd(sub, full_matrices=False)
    rank, gap = rank_from_gap(S)

    return rank, gap, Z_qp.shape[0]


def define_slices(n_points=100):
    """Define the 1D slices through N=4 d=1 shape space."""
    slices = {}

    s_vals = np.linspace(0.05, 0.95, n_points)
    slices["slice_A_sweep_s_t2.0"] = {
        "description": "Fix x=(0, s, 1, 2.0), sweep s in (0,1)",
        "param_name": "s",
        "param_values": s_vals.tolist(),
        "positions": [[0.0, s, 1.0, 2.0] for s in s_vals],
    }

    t_vals = np.linspace(1.1, 5.0, n_points)
    slices["slice_B_sweep_t_s0.5"] = {
        "description": "Fix x=(0, 0.5, 1, t), sweep t in (1.1,5)",
        "param_name": "t",
        "param_values": t_vals.tolist(),
        "positions": [[0.0, 0.5, 1.0, t] for t in t_vals],
    }

    d_vals = np.linspace(0.3, 3.0, n_points)
    slices["slice_C_equal_spacing"] = {
        "description": "Equal spacing: x=(0, d, 2d, 3d), sweep d",
        "param_name": "d",
        "param_values": d_vals.tolist(),
        "positions": [[0.0, d, 2*d, 3*d] for d in d_vals],
    }

    return slices


def main():
    parser = argparse.ArgumentParser(
        description="N=4 d=1 atlas: 1D slices through shape space")
    parser.add_argument("--n-points", type=int, default=100)
    parser.add_argument("--samples", type=int, default=500,
                        help="Phase-space samples per grid point")
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--potential", default="1/r")
    parser.add_argument("--output", default="results/n4_atlas_1d.json")
    parser.add_argument("--slices", nargs="*", default=None,
                        help="Which slices to run (default: all)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 70)
    print("N=4 ATLAS: 1D SLICES  (d=1)")
    print(f"  Potential: {args.potential}")
    print(f"  Points per slice: {args.n_points}")
    print(f"  Samples per point: {args.samples}")
    print(f"  Max level: {args.max_level}")
    print("=" * 70)

    # Build algebra and compile evaluator (once)
    print("\nBuilding N=4 d=1 algebra...", flush=True)
    t_build = time()
    alg = NBodyAlgebra(n_bodies=4, d_spatial=1, potential=args.potential)
    dims = alg.compute_growth(
        max_level=args.max_level, n_samples=args.samples,
        seed=42, resume=True)
    build_time = time() - t_build
    print(f"Algebra ready ({build_time:.1f}s)")

    all_exprs = []
    all_levels = []
    ckpt = alg.load_checkpoint()
    if ckpt is None:
        print("ERROR: No checkpoint found after compute_growth")
        sys.exit(1)

    all_exprs = ckpt["exprs"]
    all_levels_raw = ckpt["levels"]

    # Filter zero expressions
    nonzero = [(e, lv) for e, lv in zip(all_exprs, all_levels_raw) if e != 0]
    exprs_nz = [e for e, _ in nonzero]
    levels_nz = [lv for _, lv in nonzero]

    print(f"  {len(exprs_nz)} non-zero generators "
          f"(levels: {dict(sorted(set((lv, sum(1 for l in levels_nz if l == lv)) for lv in set(levels_nz))))})")

    print("Lambdifying generators...", flush=True)
    t_lam = time()
    evaluator = alg.lambdify_generators(exprs_nz)
    print(f"Lambdification done ({time()-t_lam:.1f}s)")

    # Define slices
    all_slices = define_slices(args.n_points)
    if args.slices:
        all_slices = {k: v for k, v in all_slices.items()
                      if any(s in k for s in args.slices)}

    # Load existing results for resume
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        results = {s["slice_name"]: s for s in existing.get("slices", [])}
        print(f"  Loaded {len(results)} existing slices from {args.output}")

    # Run slices
    for slice_name, slice_def in all_slices.items():
        if slice_name in results:
            print(f"\nSkipping {slice_name} (already complete)")
            continue

        positions_list = slice_def["positions"]
        param_values = slice_def["param_values"]
        n_pts = len(positions_list)

        print(f"\n{'='*60}")
        print(f"SLICE: {slice_name}")
        print(f"  {slice_def['description']}")
        print(f"  {n_pts} points, {args.samples} samples each")
        print(f"{'='*60}")

        ranks = []
        gap_ratios = []
        n_accepted_list = []
        t_slice = time()

        for idx, (pos, param) in enumerate(zip(positions_list, param_values)):
            t0 = time()
            rank, gap, n_acc = compute_rank_at_config(
                evaluator, levels_nz, pos,
                n_samples=args.samples, epsilon=0.05,
                max_level=args.max_level, seed=42 + idx)
            dt = time() - t0

            ranks.append(rank)
            gap_ratios.append(float(gap))
            n_accepted_list.append(n_acc)

            if (idx + 1) % 10 == 0 or idx == 0 or idx == n_pts - 1:
                elapsed = time() - t_slice
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (n_pts - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1:4d}/{n_pts}]  "
                      f"{slice_def['param_name']}={param:.4f}  "
                      f"rank={rank:4d}  gap={gap:.1e}  "
                      f"n_acc={n_acc}  "
                      f"{dt:.1f}s  ETA={eta:.0f}s")

        slice_elapsed = time() - t_slice

        unique_ranks = sorted(set(ranks))
        mode_rank = max(set(ranks), key=ranks.count)
        n_mode = ranks.count(mode_rank)

        slice_result = {
            "slice_name": slice_name,
            "description": slice_def["description"],
            "param_name": slice_def["param_name"],
            "param_values": param_values,
            "ranks": ranks,
            "gap_ratios": gap_ratios,
            "n_accepted": n_accepted_list,
            "n_points": n_pts,
            "n_samples": args.samples,
            "max_level": args.max_level,
            "potential": args.potential,
            "N": 4,
            "d": 1,
            "unique_ranks": unique_ranks,
            "mode_rank": mode_rank,
            "pct_mode": round(100 * n_mode / n_pts, 1),
            "rank_min": min(ranks),
            "rank_max": max(ranks),
            "elapsed_s": round(slice_elapsed, 1),
            "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
        }
        results[slice_name] = slice_result

        print(f"\n  Summary: unique ranks = {unique_ranks}")
        print(f"  Mode rank = {mode_rank} ({n_mode}/{n_pts} = {100*n_mode/n_pts:.1f}%)")
        print(f"  Elapsed: {slice_elapsed:.1f}s")

        # Save incrementally
        output_data = {
            "metadata": {
                "N": 4,
                "d": 1,
                "potential": args.potential,
                "max_level": args.max_level,
                "n_samples_per_point": args.samples,
                "n_points_per_slice": args.n_points,
                "script": "n4_atlas_1d.py",
                "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
            },
            "slices": list(results.values()),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Saved to {args.output}")

    # Final summary
    print(f"\n{'='*70}")
    print("ALL SLICES COMPLETE")
    for name, res in results.items():
        print(f"  {name}: ranks {res['unique_ranks']}, "
              f"mode={res['mode_rank']} ({res['pct_mode']}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
