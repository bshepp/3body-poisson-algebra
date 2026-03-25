#!/usr/bin/env python3
"""
Comprehensive comparison: Poisson algebra dimension sequences across
potentials, spatial dimensions, and mass ratios.

Generates calogero_comparison_table.json with all results.
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nbody'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from time import time
from sympy import Rational, Integer
from exact_growth_nbody import NBodyAlgebra

RESULTS = []


def run_config(label, n_bodies, d_spatial, potential, masses=None,
               max_level=3, samples=500):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N={n_bodies}, d={d_spatial}, V={potential}, masses={masses}")
    print(f"{'='*70}\n")

    t0 = time()
    alg = NBodyAlgebra(
        n_bodies=n_bodies,
        d_spatial=d_spatial,
        potential=potential,
        masses=masses,
    )
    level_dims = alg.compute_growth(
        max_level=max_level,
        n_samples=samples,
        seed=42,
        resume=False,
    )
    elapsed = time() - t0

    seq = [level_dims[lv] for lv in range(max_level + 1)]
    result = {
        "label": label,
        "n_bodies": n_bodies,
        "d_spatial": d_spatial,
        "potential": potential,
        "masses": str(masses) if masses else "equal",
        "dimensions": seq,
        "elapsed_seconds": round(elapsed, 1),
    }
    RESULTS.append(result)

    print(f"\n  >>> {label}: d(k) = {seq}  [{elapsed:.1f}s]\n")
    return result


if __name__ == "__main__":
    # ---- 1D comparison matrix ----
    run_config("1D Newtonian (1/r)",       3, 1, "1/r")
    run_config("1D Calogero-Moser (1/r²)", 3, 1, "1/r^2")
    run_config("1D Cubic (1/r³)",          3, 1, "1/r^3")

    # ---- Galperin superintegrable mass ratios (1D, 1/r²) ----
    # q=3: m/M = tan²(π/3) = 3
    run_config("1D CM, q=3 masses (1,3,1)",
               3, 1, "1/r^2", masses={1: Integer(1), 2: Integer(3), 3: Integer(1)})

    # q=5: m/M = tan²(π/5) ≈ 0.5279
    q5_ratio = Rational(5279, 10000)
    run_config("1D CM, q=5 masses (1,0.528,1)",
               3, 1, "1/r^2", masses={1: Integer(1), 2: q5_ratio, 3: Integer(1)})

    # q=6: m/M = tan²(π/6) = 1/3
    run_config("1D CM, q=6 masses (1,1/3,1)",
               3, 1, "1/r^2", masses={1: Integer(1), 2: Rational(1, 3), 3: Integer(1)})

    # Non-superintegrable generic mass ratio for comparison
    run_config("1D CM, generic masses (1,2.7,0.4)",
               3, 1, "1/r^2",
               masses={1: Integer(1), 2: Rational(27, 10), 3: Rational(2, 5)})

    # ---- Save results ----
    out_path = os.path.join(os.path.dirname(__file__),
                            "calogero_comparison_table.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ---- Summary table ----
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Configuration':<45} {'d(k) sequence':<25} {'Time':>8}")
    print("-"*78)
    for r in RESULTS:
        dims_str = str(r["dimensions"])
        print(f"{r['label']:<45} {dims_str:<25} {r['elapsed_seconds']:>7.1f}s")
