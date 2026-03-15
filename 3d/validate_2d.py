#!/usr/bin/env python3
# Track: 3D Extension | Spatial three-body Poisson algebra
# Parent project: ../preprint.tex (planar/2D results)
# See README.md in this directory for context.
"""
Validate the ND engine by reproducing the known 2D results.

Runs the engine with d=2 and checks the dimension sequence
against the established values [3, 6, 17, 116].

Default: level 2 (fast, ~minutes).
Full validation: python validate_2d.py --max-level 3 (~30-60 min).
"""

import sys
import argparse
from exact_growth_nd import ThreeBodyAlgebra

EXPECTED = {0: 3, 1: 6, 2: 17, 3: 116}


def main():
    ap = argparse.ArgumentParser(
        description="Validate ND engine against known 2D results")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Max bracket level (default: 2, use 3 for full)")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("VALIDATION: ND engine with d=2 vs known results")
    print(f"  Expected: {[EXPECTED[i] for i in range(args.max_level + 1)]}")
    print("=" * 70)
    print()

    alg = ThreeBodyAlgebra(d_spatial=2)
    dims = alg.compute_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    all_pass = True
    for lv in range(args.max_level + 1):
        expected = EXPECTED[lv]
        actual = dims[lv]
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        print(f"  Level {lv}: expected {expected}, got {actual}  [{status}]")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  ALL CHECKS PASSED — engine is validated for d=2")
        print("  Safe to proceed with d=3 and d=1 computations.")
    else:
        print("  *** VALIDATION FAILED ***")
        print("  Do NOT trust d=3 or d=1 results until this is fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
