#!/usr/bin/env python3
"""
r^n exponent sweep around n=2: test dimension sequence for polynomial-type
potentials V(r) = r^n using the composite potential with negative u-powers.

Since u = 1/r, we have V = r^n = u^(-n), implemented as composite term (1, -n).

Tests whether the harmonic oscillator (r^2) is special or whether nearby
fractional exponents r^1.99999, r^2.00001 give the same sequence.
Also compares against the existing exact symbolic r^2 and r^4 results.
"""

import sys
import os
import json
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra

EXPONENTS = [
    # Small r^n (sub-harmonic)
    0.5,
    1.0,
    1.5,
    # Near r^2 (harmonic)
    1.99,
    1.999,
    1.99999,
    2.00001,
    2.001,
    2.01,
    # Between harmonic and quartic
    2.5,
    3.0,
    3.5,
    # Near r^4 (quartic)
    3.99,
    3.999,
    3.99999,
    4.00001,
    4.001,
    4.01,
    4.5,
    5.0,
]

N_BODIES = 3
D_SPATIAL = 1
MAX_LEVEL = 3
N_SAMPLES = 1000

results = []

for exp in EXPONENTS:
    print(f"\n{'='*60}")
    print(f"  V(r) = r^{exp}  =>  composite u^{-exp}")
    print(f"{'='*60}", flush=True)

    t0 = time()
    try:
        algebra = NBodyAlgebra(
            n_bodies=N_BODIES,
            d_spatial=D_SPATIAL,
            potential="composite",
            potential_params=[(1, -exp)],
        )

        level_dims = algebra.compute_growth(
            max_level=MAX_LEVEL,
            n_samples=N_SAMPLES,
            seed=42,
        )

        elapsed = time() - t0
        seq = [level_dims[lv] for lv in range(MAX_LEVEL + 1)]
        print(f"\n  RESULT: r^{exp}, sequence={seq}, time={elapsed:.1f}s")

        results.append({
            "exponent": exp,
            "potential_label": f"r^{exp}",
            "u_power": -exp,
            "N": N_BODIES,
            "d": D_SPATIAL,
            "max_level": MAX_LEVEL,
            "dimension_sequence": seq,
            "computation_time_s": round(elapsed, 1),
            "method": "numerical_svd",
            "n_samples": N_SAMPLES,
        })

    except Exception as e:
        elapsed = time() - t0
        print(f"\n  FAILED: r^{exp}, error={e}, time={elapsed:.1f}s")
        results.append({
            "exponent": exp,
            "potential_label": f"r^{exp}",
            "u_power": -exp,
            "N": N_BODIES,
            "d": D_SPATIAL,
            "error": str(e),
            "computation_time_s": round(elapsed, 1),
        })

out_path = os.path.join(os.path.dirname(__file__), "..",
                        "results", "rn_exponent_sweep.json")
with open(out_path, "w") as f:
    json.dump({
        "title": "r^n exponent sweep (polynomial-type potentials)",
        "description": "Tests dimension sequence for V(r) = r^n potentials around n=2 (harmonic) and n=4 (quartic), using composite u^(-n) representation. Probes whether polynomial potentials also exhibit universality or have distinct behavior at integer exponents.",
        "date": "2026-04-14",
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
        print(f"  r^{r['exponent']:<10}  ERROR: {r['error']}")
    else:
        seq = r["dimension_sequence"]
        print(f"  r^{r['exponent']:<10}  {seq}  ({r['computation_time_s']:.1f}s)")

print(f"\nResults saved to {out_path}")
