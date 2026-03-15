#!/usr/bin/env python3
# Track: N-Body Extension | Poisson algebra for arbitrary particle count
# Parent project: ../preprint.tex (planar 3-body results)
"""
Priority 2: N=3, d=2, 1/r^3 potential through Level 3.

Tests whether the dimension sequence [3, 6, 17, 116] holds for
a stronger singularity.  The 1/r and 1/r^2 potentials both give
this sequence; 1/r^3 tests universality across pole orders.
"""

import argparse
from exact_growth_nbody import NBodyAlgebra


def main():
    ap = argparse.ArgumentParser(
        description="N=3, d=2, 1/r^3 potential test")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Max bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("POTENTIAL TEST: 1/r^3  (N=3, d=2)")
    print("Does the sequence [3, 6, 17, 116] hold for 1/r^3?")
    print("=" * 70)
    print()

    alg = NBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r^3")
    dims = alg.compute_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
        resume=args.resume,
    )

    print("\n" + "=" * 70)
    print("1/r^3 RESULTS")
    print("=" * 70)
    ref = {0: 3, 1: 6, 2: 17, 3: 116}
    seq = [dims[lv] for lv in range(args.max_level + 1)]
    ref_seq = [ref.get(lv, "?") for lv in range(args.max_level + 1)]
    print(f"  1/r^3 sequence: {seq}")
    print(f"  1/r   sequence: {ref_seq}")

    matches = all(dims[lv] == ref.get(lv) for lv in range(args.max_level + 1))
    if matches:
        print("  *** MATCHES 1/r -- universality across pole orders ***")
    else:
        print("  *** DIFFERS from 1/r -- pole order matters ***")
        for lv in range(args.max_level + 1):
            if dims[lv] != ref.get(lv):
                print(f"  First divergence at level {lv}: "
                      f"1/r^3 = {dims[lv]}, 1/r = {ref[lv]}")
                break


if __name__ == "__main__":
    main()
