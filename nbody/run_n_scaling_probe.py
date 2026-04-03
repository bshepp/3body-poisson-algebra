#!/usr/bin/env python3
"""
N-body scaling probe: how high can N go at d=1 for levels 0, 1, 2?

Runs N=5,6,7,... at d=1 with 1/r potential, measuring time per level.
Stops when a level takes too long or we hit a keyboard interrupt.
"""

import sys
import os
import json
from time import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra

MAX_LEVEL = 2
N_SAMPLES = 2000
SEED = 42
D_SPATIAL = 1

# Start from N=5 (N=3,4 already known)
N_START = 5
N_MAX = 9  # cap at 9 for this run


def count_candidates(n_bodies, max_level):
    """Estimate the number of bracket candidates at each level."""
    from math import comb
    n_pairs = comb(n_bodies, 2)
    l0 = n_pairs
    l1_new = comb(n_pairs, 2)
    total_l1 = l0 + l1_new
    # Level 2: each level-1 generator bracketed with all through level 1
    l2_candidates = l1_new * total_l1  # upper bound (minus already computed)
    return {
        "N": n_bodies,
        "pairs": n_pairs,
        "phase_dim": 2 * n_bodies * D_SPATIAL,
        "n_u": n_pairs,
        "level_0": l0,
        "level_1_new": l1_new,
        "total_through_1": total_l1,
        "level_2_candidates_upper": l2_candidates,
    }


def main():
    print("=" * 70)
    print("N-BODY SCALING PROBE  (d=1, V=1/r)")
    print("=" * 70)

    # First, show the scaling estimates
    print(f"\n{'N':>3} | {'Pairs':>5} | {'Phase':>5} | {'L0':>4} | "
          f"{'L1 new':>7} | {'Thru L1':>7} | {'L2 cands':>10}")
    print(f"{'---':>3}-+-{'-----':>5}-+-{'-----':>5}-+-{'----':>4}-+-"
          f"{'-------':>7}-+-{'-------':>7}-+-{'----------':>10}")
    for n in range(3, N_MAX + 1):
        c = count_candidates(n, MAX_LEVEL)
        print(f"{c['N']:>3} | {c['pairs']:>5} | {c['phase_dim']:>5} | "
              f"{c['level_0']:>4} | {c['level_1_new']:>7} | "
              f"{c['total_through_1']:>7} | {c['level_2_candidates_upper']:>10}")

    results = []

    for n_bodies in range(N_START, N_MAX + 1):
        c = count_candidates(n_bodies, MAX_LEVEL)
        print(f"\n{'#' * 70}")
        print(f"  N={n_bodies}: {c['pairs']} pairs, "
              f"{c['phase_dim']}D phase space, "
              f"~{c['level_2_candidates_upper']} L2 candidates")
        print(f"{'#' * 70}")

        t_start = time()
        alg = NBodyAlgebra(n_bodies=n_bodies, d_spatial=D_SPATIAL,
                           potential="1/r")

        try:
            dims = alg.compute_growth(
                max_level=MAX_LEVEL,
                n_samples=N_SAMPLES,
                seed=SEED,
                resume=False,
            )
        except (MemoryError, KeyboardInterrupt) as e:
            elapsed = time() - t_start
            print(f"\n  STOPPED at N={n_bodies}: {type(e).__name__} "
                  f"after {elapsed:.1f}s")
            break

        elapsed = time() - t_start
        seq = [dims[lv] for lv in range(MAX_LEVEL + 1)]

        result = {
            "N": n_bodies,
            "d": D_SPATIAL,
            "potential": "1/r",
            "sequence": seq,
            "elapsed_s": round(elapsed, 1),
            "pairs": c["pairs"],
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        print(f"\n  N={n_bodies} RESULT: {seq}  ({elapsed:.1f}s)")

        # Incremental save after each N
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "n_body_scaling_results.json")
        with open(out_path, "w") as f:
            json.dump({
                "experiment": "N-body scaling probe (d=1, 1/r, level 0-2)",
                "results": results,
            }, f, indent=2)
        print(f"  (saved to {out_path})")

        # Bail if level 2 took more than 30 minutes
        if elapsed > 1800:
            print(f"  Stopping: N={n_bodies} took {elapsed:.0f}s")
            break

    # Summary
    print("\n" + "=" * 70)
    print("SCALING SUMMARY  (d=1, V=1/r)")
    print("=" * 70)
    print(f"  {'N':>3} | {'d(0)':>6} | {'d(1)':>6} | {'d(2)':>6} | {'Time':>10}")
    print(f"  {'---':>3}-+-{'------':>6}-+-{'------':>6}-+-{'------':>6}-+-{'----------':>10}")
    # Include known results
    known = [
        {"N": 3, "sequence": [3, 6, 17], "elapsed_s": "<1"},
        {"N": 4, "sequence": [6, 14, 62], "elapsed_s": "~3"},
    ]
    for r in known:
        s = r["sequence"]
        print(f"  {r['N']:>3} | {s[0]:>6} | {s[1]:>6} | {s[2]:>6} | "
              f"{r['elapsed_s']:>10}")
    for r in results:
        s = r["sequence"]
        print(f"  {r['N']:>3} | {s[0]:>6} | {s[1]:>6} | {s[2]:>6} | "
              f"{r['elapsed_s']:>8.1f}s")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "n_body_scaling_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "N-body scaling probe (d=1, 1/r, level 0-2)",
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
