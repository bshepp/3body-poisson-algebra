#!/usr/bin/env python3
# Track: N-Body Extension | Poisson algebra for arbitrary particle count
# Parent project: ../preprint.tex (planar 3-body results)
"""
Validate the N-body engine by reproducing known N=3, d=2 results.

Runs NBodyAlgebra(n_bodies=3, d_spatial=2) and checks the dimension
sequence against [3, 6, 17, 116].

Default: level 2 (fast).
Full: python validate_n3.py --max-level 3
"""

import sys
import argparse
from exact_growth_nbody import NBodyAlgebra

EXPECTED = {0: 3, 1: 6, 2: 17, 3: 116}


def main():
    ap = argparse.ArgumentParser(
        description="Validate N-body engine against known N=3, d=2 results")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Max bracket level (default: 2, use 3 for full)")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("VALIDATION: N-body engine with N=3, d=2 vs known results")
    print(f"  Expected: {[EXPECTED[i] for i in range(args.max_level + 1)]}")
    print("=" * 70)
    print()

    alg = NBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r")
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
        print("  ALL CHECKS PASSED -- engine validated for N=3, d=2")
        print("  Safe to proceed with N=4 and other configurations.")
    else:
        print("  *** VALIDATION FAILED ***")
        print("  Do NOT trust N=4 or other results until this is fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
