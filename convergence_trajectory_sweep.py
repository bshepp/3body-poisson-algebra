#!/usr/bin/env python3
"""
Convergence Trajectory Sweep
=============================

Track how SVD rank determination converges as sample count increases,
for multiple (N, d, potential, level) configurations.

Uses pre-built checkpoints where available to avoid re-computing
symbolic brackets (which can take hours for Level 3).

Output: results/convergence_trajectories.json
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(100000)

SAMPLE_COUNTS = [50, 100, 200, 500, 1000, 2000, 5000]


def svd_rank_and_gap(eval_matrix):
    """Compute numerical rank and gap ratio from an evaluation matrix."""
    norms = np.linalg.norm(eval_matrix, axis=0)
    norms[norms < 1e-15] = 1.0
    M = eval_matrix / norms

    _, s, _ = np.linalg.svd(M, full_matrices=False)

    noise_threshold = 1e-8 * s[0]
    n_meaningful = int(np.sum(s > noise_threshold))

    best_gap = 1.0
    best_idx = -1
    for i in range(min(n_meaningful, len(s) - 1)):
        if s[i + 1] > noise_threshold:
            gap = s[i] / s[i + 1]
        else:
            gap = s[i] / max(s[i + 1], 1e-300)
        if gap > best_gap and i >= 2:
            best_gap = gap
            best_idx = i

    if best_gap > 1e4:
        rank = best_idx + 1
    else:
        rank = n_meaningful

    gap_ratio = float(best_gap) if best_gap < 1e300 else None
    return rank, gap_ratio


def convergence_sweep(evaluate_fn, sample_fn, all_levels, pot_name,
                      N, d, max_level=3):
    """Run convergence trajectories for one potential configuration."""
    results = []

    for target_level in range(1, max_level + 1):
        mask = [i for i, lv in enumerate(all_levels) if lv <= target_level]
        if len(mask) == 0:
            continue

        print(f"\n  Level {target_level} ({len(mask)} candidates):")

        for n_samples in SAMPLE_COUNTS:
            t0 = time()
            Z_qp, Z_u = sample_fn(n_samples, seed=42)
            eval_matrix = evaluate_fn(Z_qp, Z_u)
            sub = eval_matrix[:, mask]
            rank, gap = svd_rank_and_gap(sub)
            elapsed = time() - t0

            row = {
                "N": N, "d": d, "potential": pot_name,
                "level": target_level, "n_samples": n_samples,
                "n_candidates": len(mask),
                "rank": rank, "gap_ratio": gap,
                "elapsed_s": round(elapsed, 3),
            }
            results.append(row)
            gap_str = f"{gap:.2e}" if gap else "N/A"
            print(f"    n={n_samples:>5d}: rank={rank:>4d}, "
                  f"gap={gap_str}, {elapsed:.1f}s")

    return results


def run_n3_d2_from_checkpoint():
    """Run convergence for 1/r N=3 d=2 using the pre-built L3 checkpoint."""
    from exact_growth import sample_phase_space, lambdify_generators

    ckpt_path = os.path.join("checkpoints", "level_3.pkl")
    if not os.path.exists(ckpt_path):
        print("  WARNING: No L3 checkpoint found, skipping 1/r L3")
        return []

    print(f"\n{'='*60}")
    print(f"N=3, d=2, potential=1/r (from checkpoint)")
    print(f"{'='*60}")

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    all_exprs = ckpt["exprs"]
    all_levels = ckpt["levels"]
    print(f"  Loaded {len(all_exprs)} generators from checkpoint")

    evaluate = lambdify_generators(all_exprs)
    return convergence_sweep(evaluate, sample_phase_space,
                             all_levels, "1/r", 3, 2, max_level=3)


def run_n3_d2_level2():
    """Run convergence for N=3 d=2 at levels 1-2 (fast, no L3 brackets)."""
    from exact_growth import (
        sample_phase_space, lambdify_generators,
        poisson_bracket, simplify_generator,
        build_hamiltonians, H12, H13, H23,
    )

    potentials = {
        "1/r^2": ("1/r2", None, 1),
        "r^2": ("harmonic", None, 1),
    }

    results = []
    for pot_name, (pot_type, masses, coupling) in potentials.items():
        print(f"\n{'='*60}")
        print(f"N=3, d=2, potential={pot_name} (L1-L2)")
        print(f"{'='*60}")

        if pot_type == '1/r' and masses is None and coupling == 1:
            h12, h13, h23 = H12, H13, H23
        else:
            h12, h13, h23 = build_hamiltonians(pot_type, masses, coupling)

        all_exprs = [h12, h13, h23]
        all_levels = [0, 0, 0]
        computed_pairs = {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}

        # Level 1
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            expr = poisson_bracket(all_exprs[i], all_exprs[j])
            expr = simplify_generator(expr)
            all_exprs.append(expr)
            all_levels.append(1)

        # Level 2
        frontier = [i for i, lv in enumerate(all_levels) if lv == 1]
        n_existing = len(all_exprs)
        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                expr = simplify_generator(expr)
                all_exprs.append(expr)
                all_levels.append(2)

        print(f"  Built {len(all_exprs)} candidate generators through L2")
        evaluate = lambdify_generators(all_exprs)

        results.extend(convergence_sweep(
            evaluate, sample_phase_space, all_levels,
            pot_name, 3, 2, max_level=2))

    return results


def run_n4_configs():
    """Run convergence trajectories for N=4 L1-L2 using NBodyAlgebra."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nbody'))
    from exact_growth_nbody import NBodyAlgebra
    from sympy import expand

    results = []
    for d in [1, 2]:
        print(f"\n{'='*60}")
        print(f"N=4, d={d}, potential=1/r (L1-L2)")
        print(f"{'='*60}")

        alg = NBodyAlgebra(n_bodies=4, d_spatial=d, potential='1/r')

        all_exprs = list(alg.hamiltonians.values())
        all_levels = [0] * len(all_exprs)
        computed_pairs = set()
        for i in range(len(all_exprs)):
            for j in range(i + 1, len(all_exprs)):
                computed_pairs.add(frozenset({i, j}))

        # Level 1
        n0 = len(all_exprs)
        for i in range(n0):
            for j in range(i + 1, n0):
                expr = alg.poisson_bracket(all_exprs[i], all_exprs[j])
                expr = expand(expr)
                if expr != 0:
                    all_exprs.append(expr)
                    all_levels.append(1)

        # Level 2
        frontier = [i for i, lv in enumerate(all_levels) if lv == 1]
        n_existing = len(all_exprs)
        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)
                expr = alg.poisson_bracket(all_exprs[i], all_exprs[j])
                expr = expand(expr)
                if expr != 0:
                    all_exprs.append(expr)
                    all_levels.append(2)

        print(f"  Built {len(all_exprs)} candidate generators through L2")
        evaluate = alg.lambdify_generators(all_exprs)

        def _sample(n, seed=42):
            return alg.sample_phase_space(n, seed=seed)

        results.extend(convergence_sweep(
            evaluate, _sample, all_levels,
            "1/r", 4, d, max_level=2))

    return results


def main():
    ap = argparse.ArgumentParser(description="Convergence trajectory sweep")
    ap.add_argument("--skip-n4", action="store_true",
                    help="Skip N=4 configs (faster)")
    ap.add_argument("--output", default="results/convergence_trajectories",
                    help="Output path (without .json)")
    args = ap.parse_args()

    print("=" * 70)
    print("CONVERGENCE TRAJECTORY SWEEP")
    print("=" * 70)
    print(f"  Sample counts: {SAMPLE_COUNTS}")
    print(f"  Output: {args.output}.json")

    t_total = time()
    results = []

    # 1/r from checkpoint (L1-L3, fast)
    results.extend(run_n3_d2_from_checkpoint())

    # Other potentials at L1-L2 (fast)
    results.extend(run_n3_d2_level2())

    # N=4 configs
    if not args.skip_n4:
        results.extend(run_n4_configs())

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_path = f"{args.output}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(results)} data points saved to {out_path}")
    print(f"Total time: {time() - t_total:.1f}s")

    # Summary
    configs = set((r["N"], r["d"], r["potential"], r["level"]) for r in results)
    print(f"Configs: {len(configs)}")
    for cfg in sorted(configs):
        rows = [r for r in results if (r["N"], r["d"], r["potential"], r["level"]) == cfg]
        final_rank = rows[-1]["rank"]
        converged_at = min((r["n_samples"] for r in rows if r["rank"] == final_rank),
                           default=None)
        print(f"  N={cfg[0]} d={cfg[1]} {cfg[2]:>6s} L{cfg[3]}: "
              f"final_rank={final_rank}, converged_at={converged_at} samples")


if __name__ == "__main__":
    main()
