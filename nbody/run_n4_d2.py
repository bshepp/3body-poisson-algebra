#!/usr/bin/env python3
# Track: N-Body Extension | Poisson algebra for arbitrary particle count
# Parent project: ../preprint.tex (planar 3-body results)
"""
Priority 1: N=4, d=2, 1/r through Level 2.

First computation of the four-body Poisson algebra dimension sequence.
Level 0 has C(4,2)=6 generators, Level 1 has C(6,2)=15 candidates.
Level 2 is the first level where interesting structure appears.
"""

import argparse
from exact_growth_nbody import NBodyAlgebra


def main():
    ap = argparse.ArgumentParser(
        description="N=4, d=2 Poisson algebra computation")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Max bracket level (default: 2)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("FOUR-BODY POISSON ALGEBRA  (N=4, d=2, V=1/r)")
    print("First computation of the N=4 dimension sequence")
    print("=" * 70)
    print()

    alg = NBodyAlgebra(n_bodies=4, d_spatial=2, potential="1/r")
    dims = alg.compute_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
        resume=args.resume,
    )

    print("\n" + "=" * 70)
    print("N=4 RESULTS")
    print("=" * 70)
    seq = [dims[lv] for lv in range(args.max_level + 1)]
    print(f"  N=4 dimension sequence: {seq}")
    print(f"  N=3 reference:          [3, 6, 17, 116]")
    print(f"  Level 0: N=4 has {dims[0]} generators "
          f"(vs 3 for N=3)")
    if 1 in dims:
        print(f"  Level 1: N=4 has dim {dims[1]}")
    if 2 in dims:
        print(f"  Level 2: N=4 has dim {dims[2]}")


if __name__ == "__main__":
    main()
