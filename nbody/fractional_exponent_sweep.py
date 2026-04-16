#!/usr/bin/env python3
"""
Fractional exponent sweep: test dimension sequence sensitivity to
infinitesimal variations in the potential exponent 1/r^n.

Tests exponents near integers (n=1, n=2) and near zero to probe whether
the universality of the dimension sequence extends to non-integer exponents
or is strictly an integer phenomenon.
"""

import sys
import os
import json
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra
from sympy import Rational, Float

EXPONENTS = [
    # Near zero (weak singularity)
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    0.5,
    # Near 1 (Newtonian)
    0.99,
    0.999,
    0.99999,
    1.00001,
    1.001,
    1.01,
    # Near 2 (Calogero-Moser)
    1.5,
    1.99,
    1.999,
    1.99999,
    2.00001,
    2.001,
    2.01,
    2.5,
    3.5,
]

N_BODIES = 3
D_SPATIAL = 1
MAX_LEVEL = 3
N_SAMPLES = 2000

results = []

for exp in EXPONENTS:
    print(f"\n{'='*60}")
    print(f"  Exponent n = {exp}  (V ~ u^{exp} = 1/r^{exp})")
    print(f"{'='*60}", flush=True)

    t0 = time()
    try:
        algebra = NBodyAlgebra(
            n_bodies=N_BODIES,
            d_spatial=D_SPATIAL,
            potential="composite",
            potential_params=[(-1, exp)],
        )

        level_dims = algebra.compute_growth(
            max_level=MAX_LEVEL,
            n_samples=N_SAMPLES,
            seed=42,
        )

        elapsed = time() - t0
        seq = [level_dims[lv] for lv in range(MAX_LEVEL + 1)]
        print(f"\n  RESULT: exponent={exp}, sequence={seq}, time={elapsed:.1f}s")

        results.append({
            "exponent": exp,
            "potential_label": f"1/r^{exp}",
            "N": N_BODIES,
            "d": D_SPATIAL,
            "max_level": MAX_LEVEL,
            "dimension_sequence": seq,
            "matches_universal": seq == [3, 6, 17, 116],
            "computation_time_s": round(elapsed, 1),
            "method": "numerical_svd",
            "n_samples": N_SAMPLES,
        })

    except Exception as e:
        elapsed = time() - t0
        print(f"\n  FAILED: exponent={exp}, error={e}, time={elapsed:.1f}s")
        results.append({
            "exponent": exp,
            "potential_label": f"1/r^{exp}",
            "N": N_BODIES,
            "d": D_SPATIAL,
            "error": str(e),
            "computation_time_s": round(elapsed, 1),
        })

out_path = os.path.join(os.path.dirname(__file__), "..",
                        "results", "fractional_exponent_sweep.json")
with open(out_path, "w") as f:
    json.dump({
        "title": "Fractional exponent sweep for 1/r^n potential",
        "description": "Tests whether dimension sequence universality extends to non-integer exponents",
        "N": N_BODIES,
        "d": D_SPATIAL,
        "max_level": MAX_LEVEL,
        "n_samples": N_SAMPLES,
        "results": results,
    }, f, indent=2)

print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for r in results:
    if "error" in r:
        print(f"  n={r['exponent']:12.5f}  ERROR: {r['error']}")
    else:
        seq = r["dimension_sequence"]
        match = "UNIVERSAL" if r["matches_universal"] else f"DIFFERENT: {seq}"
        print(f"  n={r['exponent']:12.5f}  {match}  ({r['computation_time_s']:.1f}s)")

print(f"\nResults saved to {out_path}")
