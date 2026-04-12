#!/usr/bin/env python3
"""
Phase 4: Level-3 verification at 10 concrete rational p-values.

Intended for Colab (long runtime: ~30-60 min per p-value).
Can also run locally with fewer p-values via --p-values flag.

Sign convention
---------------
V = u^p  where u = 1/r.
All p > 0 here (singular potentials 1/r^p).

p = 1/3, 1/2, 1, 3/2, 2, 3, 4, 5, 7, 10
  p=1: Newton.  p=2: Calogero-Moser.  p=1/2: sub-Coulomb.

Expected: dim = [3, 6, 17, 116] for ALL p > 0.

Usage
-----
    # Full run (all 10 p-values, level 3):
    python nbody/symbolic_n_level3.py

    # Subset (fast test):
    python nbody/symbolic_n_level3.py --p-values "1,2,3" --max-level 2

    # Unequal masses:
    python nbody/symbolic_n_level3.py --masses "1,2,3" --p-values "1,2"

    # Resume from checkpoints:
    python nbody/symbolic_n_level3.py --resume

For Colab: copy this file + exact_growth_nbody.py into your runtime,
install deps (pip install sympy numpy scipy), and run.
"""

import os
import sys
import argparse
import json
from time import time

import sympy as sp
from sympy import Integer, Rational

os.environ["PYTHONUNBUFFERED"] = "1"

# Ensure nbody/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra


# Default p-values: all positive (singular potentials)
DEFAULT_P_VALUES = [
    Rational(1, 3),   # 1/r^(1/3)
    Rational(1, 2),   # 1/r^(1/2)
    Rational(1),       # 1/r (Newton)
    Rational(3, 2),   # 1/r^(3/2)
    Rational(2),       # 1/r^2 (Calogero-Moser)
    Rational(3),       # 1/r^3
    Rational(4),       # 1/r^4
    Rational(5),       # 1/r^5
    Rational(7),       # 1/r^7
    Rational(10),      # 1/r^10
]

EXPECTED_DIMS = {0: 3, 1: 6, 2: 17, 3: 116}


def run_single_p(p_val, max_level, n_samples, seed, masses, resume):
    """Run NBodyAlgebra for a single p-value. Returns dims list."""
    print(f"\n{'='*60}")
    print(f"  p = {p_val}  (V = 1/r^{p_val})")
    print(f"{'='*60}")

    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2,
        potential="composite",
        potential_params=[(Integer(-1), p_val)],
        masses=masses,
    )

    t0 = time()
    level_dims = alg.compute_growth(
        max_level=max_level,
        n_samples=n_samples,
        seed=seed,
        resume=resume,
    )
    elapsed = time() - t0

    # level_dims is a dict {0: rank0, 1: rank1, ...}
    dims = [level_dims[lv] for lv in sorted(level_dims)]

    # Validate
    all_ok = True
    for lv, dim in level_dims.items():
        if lv in EXPECTED_DIMS and dim != EXPECTED_DIMS[lv]:
            all_ok = False

    status = "OK" if all_ok else "ANOMALY"
    print(f"\n  p = {p_val}: dims = {dims}  [{elapsed:.0f}s]  {status}")
    return dims, elapsed, all_ok


def main():
    ap = argparse.ArgumentParser(
        description="Phase 4: Level-3 verification at concrete rational p")
    ap.add_argument("--p-values", type=str, default=None,
                    help="Comma-separated p-values (as rationals), "
                         "e.g. '1,2,3' or '1/2,3/2'")
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--masses", type=str, default=None,
                    help="Comma-separated masses, e.g. '1,2,3'")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoints")
    args = ap.parse_args()

    # Parse p-values
    if args.p_values:
        p_list = []
        for s in args.p_values.split(","):
            s = s.strip()
            if "/" in s:
                num, den = s.split("/")
                p_list.append(Rational(int(num), int(den)))
            else:
                p_list.append(Rational(s))
    else:
        p_list = DEFAULT_P_VALUES

    # Parse masses (NBodyAlgebra expects dict {1: m1, 2: m2, 3: m3})
    if args.masses:
        mass_vals = [Rational(m.strip()) for m in args.masses.split(",")]
        masses = {i + 1: v for i, v in enumerate(mass_vals)}
    else:
        masses = None  # equal masses

    mass_str = str(masses) if masses else "{1: 1, 2: 1, 3: 1}"

    print("=" * 70)
    print("PHASE 4: Level-3 Verification at Concrete Rational p-values")
    print(f"  V = u^p = 1/r^p  (all p > 0, singular potentials)")
    print(f"  p-values: {[str(p) for p in p_list]}")
    print(f"  max_level: {args.max_level}")
    print(f"  masses: {mass_str}")
    print(f"  Expected dims: {[EXPECTED_DIMS[k] for k in range(args.max_level+1)]}")
    print("=" * 70)

    results = []
    for p_val in p_list:
        try:
            dims, elapsed, ok = run_single_p(
                p_val, args.max_level, args.samples,
                args.seed, masses, args.resume)
            results.append({
                "p": str(p_val),
                "p_float": float(p_val),
                "dims": dims,
                "time_s": elapsed,
                "ok": ok,
            })
        except Exception as e:
            print(f"\n  ERROR at p={p_val}: {e}")
            results.append({
                "p": str(p_val),
                "p_float": float(p_val),
                "dims": [],
                "time_s": 0.0,
                "ok": False,
                "error": str(e),
            })

    # Summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'p':>8} | {'dims':<25} | {'time':>8} | status")
    print("-" * 60)
    all_ok = True
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        if not r["ok"]:
            all_ok = False
        dims_str = str(r["dims"]) if r["dims"] else r.get("error", "N/A")
        print(f"{r['p']:>8} | {dims_str:<25} | {r['time_s']:>7.0f}s | {status}")

    print()
    if all_ok:
        print(f"ALL {len(results)} p-values give expected dimensions")
        print(f"  Through level {args.max_level}: "
              f"{[EXPECTED_DIMS[k] for k in range(args.max_level+1)]}")
    else:
        n_fail = sum(1 for r in results if not r["ok"])
        print(f"WARNING: {n_fail}/{len(results)} p-values FAILED")
    print("=" * 70)

    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "n_universality_survey")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "level3_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
