#!/usr/bin/env python3
"""
Level-2 Exponent Sweep
======================

Produces a continuous map of algebra dimension vs potential exponent n,
sweeping both 1/r^n and r^n families at Level 2 only.

Level 2 is fast (~1-5 seconds per exponent), enabling dense sampling.
Checkpoints after each exponent for safe resume.

Usage:
    python level2_exponent_sweep.py                    # full sweep
    python level2_exponent_sweep.py --family 1/r^n     # singular only
    python level2_exponent_sweep.py --family r^n        # polynomial only
    python level2_exponent_sweep.py --step 0.01         # finer step
    python level2_exponent_sweep.py --n-samples 1000    # more samples
"""

import sys
import os
import json
import argparse
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra

N_BODIES = 3
D_SPATIAL = 1
MAX_LEVEL = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "level2_exponent_sweep.json")


def load_checkpoint():
    if os.path.isfile(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        return data
    return None


def save_checkpoint(data):
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUTPUT_PATH)


def get_completed_keys(data):
    completed = set()
    for r in data.get("results", []):
        key = (r["family"], round(r["exponent"], 8))
        completed.add(key)
    return completed


def run_single(family, n, n_samples, seed):
    """Run a single exponent and return result dict."""
    if family == "1/r^n":
        potential_params = [(-1, n)]
        label = f"1/r^{n}"
    else:
        potential_params = [(1, -n)]
        label = f"r^{n}"

    t0 = time()
    try:
        algebra = NBodyAlgebra(
            n_bodies=N_BODIES,
            d_spatial=D_SPATIAL,
            potential="composite",
            potential_params=potential_params,
        )
        level_dims = algebra.compute_growth(
            max_level=MAX_LEVEL,
            n_samples=n_samples,
            seed=seed,
        )
        elapsed = time() - t0
        seq = [level_dims[lv] for lv in range(MAX_LEVEL + 1)]

        return {
            "family": family,
            "exponent": n,
            "potential_label": label,
            "N": N_BODIES,
            "d": D_SPATIAL,
            "max_level": MAX_LEVEL,
            "dimension_sequence": seq,
            "dim_L0": seq[0],
            "dim_L1": seq[1],
            "dim_L2": seq[2],
            "matches_universal_L2": seq == [3, 6, 17],
            "computation_time_s": round(elapsed, 2),
            "method": "numerical_svd",
            "n_samples": n_samples,
        }
    except Exception as e:
        elapsed = time() - t0
        return {
            "family": family,
            "exponent": n,
            "potential_label": label,
            "N": N_BODIES,
            "d": D_SPATIAL,
            "max_level": MAX_LEVEL,
            "error": str(e),
            "computation_time_s": round(elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Level-2 exponent sweep")
    parser.add_argument("--family", choices=["1/r^n", "r^n", "both"],
                        default="both", help="Which potential family")
    parser.add_argument("--step", type=float, default=0.02,
                        help="Step size for exponent (default 0.02)")
    parser.add_argument("--n-min", type=float, default=0.02,
                        help="Minimum exponent (default 0.02)")
    parser.add_argument("--n-max", type=float, default=5.0,
                        help="Maximum exponent (default 5.0)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="SVD sample count (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exponents = []
    n = args.n_min
    while n <= args.n_max + 1e-9:
        exponents.append(round(n, 8))
        n += args.step

    families = []
    if args.family in ("1/r^n", "both"):
        families.append("1/r^n")
    if args.family in ("r^n", "both"):
        families.append("r^n")

    total_jobs = len(families) * len(exponents)

    data = load_checkpoint()
    if data is None:
        data = {
            "title": "Level-2 exponent sweep",
            "description": ("Continuous map of L2 algebra dimension vs "
                            "potential exponent for 1/r^n and r^n families"),
            "N": N_BODIES,
            "d": D_SPATIAL,
            "max_level": MAX_LEVEL,
            "n_samples": args.n_samples,
            "step": args.step,
            "exponent_range": [args.n_min, args.n_max],
            "results": [],
        }

    completed = get_completed_keys(data)
    remaining = [(fam, n) for fam in families for n in exponents
                 if (fam, round(n, 8)) not in completed]

    print("=" * 70)
    print("LEVEL-2 EXPONENT SWEEP")
    print("=" * 70)
    print(f"  Families: {families}")
    print(f"  Exponents: {len(exponents)} values, "
          f"n = {args.n_min} to {args.n_max}, step {args.step}")
    print(f"  Total jobs: {total_jobs}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Output: {OUTPUT_PATH}")
    print("=" * 70, flush=True)

    t_total = time()
    for i, (family, n) in enumerate(remaining):
        print(f"\n  [{i+1}/{len(remaining)}] {family} n={n:.4f} ...", end="",
              flush=True)

        result = run_single(family, n, args.n_samples, args.seed)
        data["results"].append(result)

        if "error" in result:
            print(f" ERROR: {result['error']} ({result['computation_time_s']:.1f}s)")
        else:
            seq = result["dimension_sequence"]
            match = "UNIVERSAL" if result["matches_universal_L2"] else f"{seq}"
            print(f" {match} ({result['computation_time_s']:.1f}s)")

        save_checkpoint(data)

    total_elapsed = time() - t_total

    n_success = sum(1 for r in data["results"] if "error" not in r)
    n_error = sum(1 for r in data["results"] if "error" in r)
    n_universal = sum(1 for r in data["results"]
                      if r.get("matches_universal_L2", False))

    print(f"\n\n{'=' * 70}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total results: {len(data['results'])}")
    print(f"  Successful: {n_success}")
    print(f"  Errors: {n_error}")
    print(f"  Universal [3,6,17]: {n_universal}")
    print(f"  Non-universal: {n_success - n_universal}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output: {OUTPUT_PATH}")

    for family in families:
        fam_results = [r for r in data["results"]
                       if r["family"] == family and "error" not in r]
        non_univ = [r for r in fam_results if not r["matches_universal_L2"]]
        print(f"\n  {family}: {len(fam_results)} successful, "
              f"{len(non_univ)} non-universal")
        if non_univ:
            for r in sorted(non_univ, key=lambda x: x["exponent"]):
                print(f"    n={r['exponent']:.4f}: {r['dimension_sequence']}")


if __name__ == "__main__":
    main()
