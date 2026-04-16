#!/usr/bin/env python3
"""
Extended Level-3 exponent sweep for 1/r^n and r^n potential families.

Densifies L3 coverage beyond the existing fractional_exponent_sweep.json
(22 points for 1/r^n) and rn_exponent_sweep.json (20 points for r^n)
to create a publication-quality continuous plot of dim_L3(n) vs exponent.

Merges existing results and only computes new exponents. Saves after
each computation for resume support.
"""

import sys
import os
import json
from time import time, strftime

sys.setrecursionlimit(100000)
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbody"))
from exact_growth_nbody import NBodyAlgebra

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "l3_exponent_sweep_extended.json")
EXISTING_1RN = os.path.join(RESULTS_DIR, "fractional_exponent_sweep.json")
EXISTING_RN = os.path.join(RESULTS_DIR, "rn_exponent_sweep.json")

N_BODIES = 3
D_SPATIAL = 1
MAX_LEVEL = 3
N_SAMPLES = 2000
SEED = 42
TIMEOUT_PER_EXPONENT = 1800  # 30 min

# New exponents to fill gaps. Existing exponents are loaded and merged.
NEW_1RN_EXPONENTS = [
    # Near-zero gap fill
    0.005, 0.02, 0.05,
    # Intermediate fill (gap between 0.1 and 0.99)
    0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9,
    # Between 1 and 2 (gap between 1.01 and 1.99)
    1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8,
    # Above 2 fill
    2.1, 2.2, 2.3, 2.4, 3.0, 4.0, 4.5, 5.0,
]

NEW_RN_EXPONENTS = [
    # Sub-harmonic fill (gap 0-0.5)
    0.1, 0.2, 0.3,
    # Near r^1 transition
    0.7, 0.8, 0.9,
    1.1, 1.2, 1.3, 1.4,
    # Between 1.5 and 2
    1.6, 1.7, 1.8,
    # Between 2 and 4
    2.1, 2.2, 2.3, 2.7, 3.3,
    # Beyond 5
    5.5, 6.0, 7.0,
]


def load_existing_results():
    """Load existing L3 sweep results and return merged dict keyed by (family, exponent)."""
    existing = {}

    if os.path.exists(EXISTING_1RN):
        with open(EXISTING_1RN) as f:
            data = json.load(f)
        for r in data.get("results", []):
            r.setdefault("family", "1/r^n")
            key = ("1/r^n", r["exponent"])
            existing[key] = r

    if os.path.exists(EXISTING_RN):
        with open(EXISTING_RN) as f:
            data = json.load(f)
        for r in data.get("results", []):
            r.setdefault("family", "r^n")
            key = ("r^n", r["exponent"])
            existing[key] = r

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        for r in data.get("results", []):
            key = (r["family"], r["exponent"])
            existing[key] = r

    return existing


def compute_exponent(family, exponent, n_samples=N_SAMPLES):
    """Compute L3 dimension sequence for a single exponent."""
    if family == "1/r^n":
        potential_params = [(-1, exponent)]
        label = f"1/r^{exponent}"
    else:
        potential_params = [(1, -exponent)]
        label = f"r^{exponent}"

    print(f"\n{'='*60}")
    print(f"  {label}  (composite u^{potential_params[0][1]})")
    print(f"{'='*60}", flush=True)

    t0 = time()
    try:
        alg = NBodyAlgebra(
            n_bodies=N_BODIES,
            d_spatial=D_SPATIAL,
            potential="composite",
            potential_params=potential_params,
        )
        dims = alg.compute_growth(
            max_level=MAX_LEVEL,
            n_samples=n_samples,
            seed=SEED,
            resume=True,
        )
        elapsed = time() - t0
        seq = [dims[lv] for lv in range(MAX_LEVEL + 1)]
        print(f"\n  RESULT: {label} -> {seq}  ({elapsed:.1f}s)")

        return {
            "family": family,
            "exponent": exponent,
            "potential_label": label,
            "N": N_BODIES,
            "d": D_SPATIAL,
            "max_level": MAX_LEVEL,
            "dimension_sequence": seq,
            "computation_time_s": round(elapsed, 1),
            "method": "numerical_svd",
            "n_samples": n_samples,
        }
    except Exception as e:
        elapsed = time() - t0
        print(f"\n  FAILED: {label} -> {e}  ({elapsed:.1f}s)")
        return {
            "family": family,
            "exponent": exponent,
            "potential_label": label,
            "N": N_BODIES,
            "d": D_SPATIAL,
            "max_level": MAX_LEVEL,
            "error": str(e),
            "computation_time_s": round(elapsed, 1),
        }


def save_output(all_results):
    """Save all results (existing + new) to the output file."""
    sorted_results = sorted(all_results, key=lambda r: (r["family"], r["exponent"]))
    output = {
        "title": "Extended Level-3 exponent sweep (1/r^n and r^n families)",
        "description": (
            "Densified L3 coverage for publication-quality dim_L3(n) vs exponent plot. "
            "Merges original fractional_exponent_sweep.json and rn_exponent_sweep.json "
            "with new intermediate exponents."
        ),
        "date": strftime("%Y-%m-%d"),
        "N": N_BODIES,
        "d": D_SPATIAL,
        "max_level": MAX_LEVEL,
        "n_samples": N_SAMPLES,
        "n_results": len(sorted_results),
        "results": sorted_results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {len(sorted_results)} results to {OUTPUT_PATH}")


def main():
    print("=" * 70)
    print("EXTENDED LEVEL-3 EXPONENT SWEEP")
    print("=" * 70)
    print(f"  Started: {strftime('%Y-%m-%d %H:%M:%S')}")

    existing = load_existing_results()
    print(f"  Loaded {len(existing)} existing results")

    # Build task list: only new exponents
    tasks = []
    for exp in NEW_1RN_EXPONENTS:
        if ("1/r^n", exp) not in existing:
            tasks.append(("1/r^n", exp))
    for exp in NEW_RN_EXPONENTS:
        if ("r^n", exp) not in existing:
            tasks.append(("r^n", exp))

    print(f"  New exponents to compute: {len(tasks)}")
    total_tasks = len(tasks)

    if total_tasks == 0:
        print("  Nothing new to compute!")
        # Still save merged output
        all_results = list(existing.values())
        save_output(all_results)
        return

    # Estimate runtime
    avg_time = 5 * 60  # ~5 min average
    est_hours = (total_tasks * avg_time) / 3600
    print(f"  Estimated runtime: ~{est_hours:.1f} hours")

    t_total = time()
    completed = 0
    errors = 0

    for i, (family, exp) in enumerate(tasks):
        elapsed_total = time() - t_total
        if completed > 0:
            avg_per = elapsed_total / completed
            remaining = (total_tasks - i) * avg_per
            eta_min = remaining / 60
            print(f"\n  [{i+1}/{total_tasks}] ETA: {eta_min:.0f} min remaining")
        else:
            print(f"\n  [{i+1}/{total_tasks}]")

        result = compute_exponent(family, exp)
        existing[(family, exp)] = result

        if "error" in result:
            errors += 1
        else:
            completed += 1

        # Save after each computation for resume
        all_results = list(existing.values())
        save_output(all_results)

    total_elapsed = time() - t_total

    # Final summary
    print("\n\n" + "=" * 70)
    print("EXTENDED L3 SWEEP -- SUMMARY")
    print("=" * 70)
    print(f"  New computed: {completed} success, {errors} errors")
    print(f"  Total results: {len(existing)}")
    print(f"  Total time: {total_elapsed/60:.1f} min")

    # Show results by family
    for family in ["1/r^n", "r^n"]:
        fam_results = sorted(
            [r for r in existing.values() if r.get("family") == family],
            key=lambda r: r["exponent"],
        )
        print(f"\n  {family} family ({len(fam_results)} exponents):")
        for r in fam_results:
            if "error" in r:
                print(f"    n={r['exponent']:<10}  ERROR: {r['error'][:50]}")
            else:
                seq = r["dimension_sequence"]
                universal = seq == [3, 6, 17, 116]
                status = "UNIVERSAL" if universal else f"{seq}"
                print(f"    n={r['exponent']:<10}  {status}")

    print(f"\n  Results saved to {OUTPUT_PATH}")
    print(f"  Completed: {strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
