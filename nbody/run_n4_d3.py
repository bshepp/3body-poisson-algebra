#!/usr/bin/env python3
# Track: N-Body Extension | Poisson algebra for arbitrary particle count
# Parent project: ../preprint.tex (planar 3-body results)
"""
Priority 4b: N=4, d=3 through Level 2.

Tests spatial dimension independence for N=4.
If the N=4 sequence is the same at d=3 as at d=2,
d-independence extends beyond N=3.
"""

import argparse
from exact_growth_nbody import NBodyAlgebra


def main():
    ap = argparse.ArgumentParser(
        description="N=4, d=3 Poisson algebra computation")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Max bracket level (default: 2)")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("FOUR-BODY 3D  (N=4, d=3, V=1/r)")
    print("Testing d-independence for N=4")
    print("=" * 70)
    print()

    alg = NBodyAlgebra(n_bodies=4, d_spatial=3, potential="1/r")
    dims = alg.compute_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
        resume=args.resume,
    )

    print("\n" + "=" * 70)
    print("N=4, d=3 RESULTS")
    print("=" * 70)
    seq = [dims[lv] for lv in range(args.max_level + 1)]
    print(f"  N=4, d=3 dimension sequence: {seq}")


if __name__ == "__main__":
    main()
