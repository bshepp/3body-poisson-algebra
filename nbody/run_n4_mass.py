#!/usr/bin/env python3
# Track: N-Body Extension | Poisson algebra for arbitrary particle count
# Parent project: ../preprint.tex (planar 3-body results)
"""
Priority 3: N=4 mass invariance test.

Runs N=4, d=2, 1/r at Level 2 with several mass configurations
to test whether the dimension sequence is mass-independent for N=4,
as it is for N=3.
"""

import argparse
from sympy import Integer, Rational
from exact_growth_nbody import NBodyAlgebra

MASS_CONFIGS = {
    "equal": {1: Integer(1), 2: Integer(1),
              3: Integer(1), 4: Integer(1)},
    "hierarchical": {1: Integer(100), 2: Integer(10),
                     3: Integer(1), 4: Integer(1)},
    "mixed": {1: Integer(3), 2: Integer(7),
              3: Integer(11), 4: Integer(2)},
}


def main():
    ap = argparse.ArgumentParser(
        description="N=4 mass invariance test")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Max bracket level (default: 2)")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("N=4 MASS INVARIANCE TEST  (d=2, V=1/r)")
    print("=" * 70)
    print()

    results = {}

    for config_name, masses in MASS_CONFIGS.items():
        print(f"\n{'='*70}")
        mass_str = ", ".join(f"m{k}={v}" for k, v in masses.items())
        print(f"Config: {config_name}  ({mass_str})")
        print("=" * 70)

        alg = NBodyAlgebra(n_bodies=4, d_spatial=2, potential="1/r",
                           masses=masses)
        dims = alg.compute_growth(
            max_level=args.max_level,
            n_samples=args.samples,
            seed=args.seed,
        )
        seq = [dims[lv] for lv in range(args.max_level + 1)]
        results[config_name] = seq
        print(f"\n  {config_name}: {seq}")

    print("\n" + "=" * 70)
    print("MASS INVARIANCE SUMMARY")
    print("=" * 70)

    ref = None
    all_match = True
    for name, seq in results.items():
        print(f"  {name:20s}: {seq}")
        if ref is None:
            ref = seq
        elif seq != ref:
            all_match = False

    if all_match:
        print("\n  *** ALL CONFIGS MATCH -- mass invariance holds for N=4 ***")
    else:
        print("\n  *** MISMATCH DETECTED -- mass invariance FAILS for N=4 ***")


if __name__ == "__main__":
    main()
