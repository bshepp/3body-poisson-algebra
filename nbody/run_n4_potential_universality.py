#!/usr/bin/env python3
"""
Gap Item 2.2: N=4 Potential Universality Test
=============================================

Tests whether the N=4 dimension sequence [6, 14, 62] is invariant
across potential types, as proven for N=3 (where 1/r, 1/r², 1/r³,
log(r) all yield [3, 6, 17, 116]).

Potentials tested:
  - 1/r²  (Calogero-Moser)
  - 1/r³  (dipole-dipole)
  - log(r) (2D vortex / Coulomb-log)

Uses d=1 (cheapest) since d-independence is already confirmed for N=4.
Reference: N=4, d=1, 1/r → [6, 14, 62]  (established Mar 2026)

This is a CRITICAL falsifiable prediction from Paper 3.
"""

import sys
import os
import json
from time import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra

REFERENCE = {0: 6, 1: 14, 2: 62}
MAX_LEVEL = 2
N_SAMPLES = 500
SEED = 42
D_SPATIAL = 1

POTENTIALS = [
    {"name": "1/r^2", "potential": "1/r^2"},
    {"name": "1/r^3", "potential": "1/r^3"},
    {"name": "log(r)", "potential": "log"},
]


def run_one(pot_spec):
    """Run NBodyAlgebra for one potential and return results dict."""
    name = pot_spec["name"]
    potential = pot_spec["potential"]

    print("\n" + "#" * 70)
    print(f"  N=4, d={D_SPATIAL}, V = {name}")
    print(f"  Reference sequence: {[REFERENCE[k] for k in range(MAX_LEVEL + 1)]}")
    print("#" * 70 + "\n")

    t0 = time()
    alg = NBodyAlgebra(
        n_bodies=4,
        d_spatial=D_SPATIAL,
        potential=potential,
    )
    dims = alg.compute_growth(
        max_level=MAX_LEVEL,
        n_samples=N_SAMPLES,
        seed=SEED,
    )
    elapsed = time() - t0

    seq = [dims[lv] for lv in range(MAX_LEVEL + 1)]
    matches = all(dims[lv] == REFERENCE[lv] for lv in range(MAX_LEVEL + 1))

    result = {
        "potential": name,
        "N": 4,
        "d": D_SPATIAL,
        "max_level": MAX_LEVEL,
        "n_samples": N_SAMPLES,
        "sequence": seq,
        "reference": [REFERENCE[k] for k in range(MAX_LEVEL + 1)],
        "matches_reference": matches,
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    status = "MATCH" if matches else "MISMATCH"
    print(f"\n  {'=' * 50}")
    print(f"  {name}: {seq}  [{status}]  ({elapsed:.1f}s)")
    print(f"  {'=' * 50}")

    return result


def main():
    print("=" * 70)
    print("GAP ITEM 2.2: N=4 POTENTIAL UNIVERSALITY TEST")
    print("=" * 70)
    print(f"  Reference (1/r): {[REFERENCE[k] for k in range(MAX_LEVEL + 1)]}")
    print(f"  Potentials to test: {[p['name'] for p in POTENTIALS]}")
    print(f"  Config: N=4, d={D_SPATIAL}, max_level={MAX_LEVEL}, "
          f"samples={N_SAMPLES}")
    print()

    results = []
    t_total = time()

    for pot_spec in POTENTIALS:
        result = run_one(pot_spec)
        results.append(result)

    total_elapsed = time() - t_total

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: N=4 POTENTIAL UNIVERSALITY")
    print("=" * 70)
    print(f"  {'Potential':<12} {'Sequence':<20} {'Match?':<10} {'Time':<10}")
    print(f"  {'-'*12} {'-'*20} {'-'*10} {'-'*10}")

    all_match = True
    for r in results:
        seq_str = str(r["sequence"])
        match_str = "YES" if r["matches_reference"] else "NO !!!"
        time_str = f"{r['elapsed_s']:.1f}s"
        print(f"  {r['potential']:<12} {seq_str:<20} {match_str:<10} {time_str}")
        if not r["matches_reference"]:
            all_match = False

    print()
    if all_match:
        print("  CONCLUSION: N=4 potential universality CONFIRMED")
        print("  The sequence [6, 14, 62] is invariant across 1/r, 1/r², "
              "1/r³, and log(r).")
    else:
        mismatches = [r for r in results if not r["matches_reference"]]
        print("  WARNING: UNIVERSALITY BROKEN for N=4!")
        for r in mismatches:
            print(f"    {r['potential']}: got {r['sequence']} "
                  f"instead of {r['reference']}")

    print(f"\n  Total wall time: {total_elapsed:.1f}s")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "n4_potential_universality_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "N=4 potential universality (gap item 2.2)",
            "reference_potential": "1/r",
            "reference_sequence": [6, 14, 62],
            "all_match": all_match,
            "total_elapsed_s": round(total_elapsed, 1),
            "results": results,
        }, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
