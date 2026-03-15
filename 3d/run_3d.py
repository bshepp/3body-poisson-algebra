#!/usr/bin/env python3
# Track: 3D Extension | Spatial three-body Poisson algebra
# Parent project: ../preprint.tex (planar/2D results)
# See README.md in this directory for context.
"""
Run the 3D (spatial) Poisson algebra computation.

Defaults to level 2 — this is the first bracket level where
d_3D could exceed d_2D.  Level 0 and 1 are guaranteed to
match (same number of generators and brackets).

If level 2 matches the 2D result (d=17), push to level 3:
    python exact_growth_nd.py -d 3 --max-level 3
"""

import sys
import argparse
from exact_growth_nd import ThreeBodyAlgebra


def main():
    ap = argparse.ArgumentParser(
        description="3D Poisson algebra computation")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Max bracket level (default: 2)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("3D THREE-BODY POISSON ALGEBRA")
    print("Testing whether dimension sequence changes in 3D")
    print("2D reference: [3, 6, 17, 116]")
    print("=" * 70)
    print()

    alg = ThreeBodyAlgebra(d_spatial=3)
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
                print(f"  Level {lv}: d_3D = {dims[lv]} == d_2D = {ref}")
            elif dims[lv] > ref:
                print(f"  Level {lv}: d_3D = {dims[lv]} > d_2D = {ref}  "
                      f"*** 3D HAS MORE INDEPENDENT GENERATORS ***")
            else:
                print(f"  Level {lv}: d_3D = {dims[lv]} < d_2D = {ref}  "
                      f"*** UNEXPECTED — CHECK NUMERICS ***")
        else:
            print(f"  Level {lv}: d_3D = {dims[lv]}  (no 2D reference)")


if __name__ == "__main__":
    main()
