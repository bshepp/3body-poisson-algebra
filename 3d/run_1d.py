#!/usr/bin/env python3
# Track: 3D Extension | Spatial three-body Poisson algebra
# Parent project: ../preprint.tex (planar/2D results)
# See README.md in this directory for context.
"""
Run the 1D (linear) Poisson algebra computation.

The 1D case is very fast (6D phase space, small expressions)
and provides a lower-dimensional reference point.

In 1D the polynomial trick still works: u_ij = 1/|x_i - x_j|
and du_ij/dx_i = -(x_i - x_j) * u_ij^3, same formula as 2D/3D
because |x_i - x_j| = sqrt((x_i - x_j)^2).
"""

import sys
import argparse
from exact_growth_nd import ThreeBodyAlgebra


def main():
    ap = argparse.ArgumentParser(
        description="1D Poisson algebra computation")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Max bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("1D THREE-BODY POISSON ALGEBRA")
    print("Bonus data point: does the sequence change in 1D?")
    print("2D reference: [3, 6, 17, 116]")
    print("=" * 70)
    print()

    alg = ThreeBodyAlgebra(d_spatial=1)
    dims = alg.compute_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
        resume=args.resume,
    )

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    ref_2d = {0: 3, 1: 6, 2: 17, 3: 116}
    for lv in sorted(dims):
        ref = ref_2d.get(lv)
        if ref is not None:
            if dims[lv] == ref:
                print(f"  Level {lv}: d_1D = {dims[lv]} == d_2D = {ref}")
            elif dims[lv] < ref:
                print(f"  Level {lv}: d_1D = {dims[lv]} < d_2D = {ref}  "
                      f"(lower dimension constrains the algebra)")
            else:
                print(f"  Level {lv}: d_1D = {dims[lv]} > d_2D = {ref}  "
                      f"*** UNEXPECTED — CHECK NUMERICS ***")
        else:
            print(f"  Level {lv}: d_1D = {dims[lv]}  (no 2D reference)")


if __name__ == "__main__":
    main()
