#!/usr/bin/env python3
"""
Composite potential universality test.

Tests whether the dimension sequence [3, 6, 17, 116] is invariant
under linear combinations of singular potential terms.  This extends
the universality conjecture (Paper 3) from single-power potentials
V = c * r^{-p} to composite potentials V = sum_k c_k * r^{-p_k}.

Test cases:
  1. V = -u - u^2          (1/r + 1/r^2 combined)
  2. V = -u - 0.5*u^2 - 0.3*u^3   (three-term composite)
  3. V = -u                (single-term control, should match pure 1/r)
"""

import argparse
from sympy import Rational, Integer
from exact_growth_nbody import NBodyAlgebra


CONFIGS = {
    "control_1r": {
        "label": "Control: single 1/r term",
        "params": [(-Integer(1), 1)],
    },
    "two_term": {
        "label": "Composite: -u - u^2  (1/r + 1/r^2)",
        "params": [(-Integer(1), 1), (-Integer(1), 2)],
    },
    "three_term": {
        "label": "Composite: -u - u^2/2 - 3u^3/10  (three terms)",
        "params": [(-Integer(1), 1), (-Rational(1, 2), 2), (-Rational(3, 10), 3)],
    },
}

EXPECTED = {0: 3, 1: 6, 2: 17, 3: 116}


def main():
    ap = argparse.ArgumentParser(
        description="Composite potential universality test")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Max bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="*", default=None,
                    choices=list(CONFIGS.keys()),
                    help="Which configs to run (default: all)")
    args = ap.parse_args()

    configs_to_run = args.configs or list(CONFIGS.keys())
    results = {}

    for cfg_name in configs_to_run:
        cfg = CONFIGS[cfg_name]
        print("\n" + "#" * 70)
        print(f"# {cfg['label']}")
        print(f"# Config: {cfg_name}")
        print("#" * 70 + "\n")

        alg = NBodyAlgebra(
            n_bodies=3,
            d_spatial=2,
            potential_params=cfg["params"],
        )
        dims = alg.compute_growth(
            max_level=args.max_level,
            n_samples=args.samples,
            seed=args.seed,
        )
        results[cfg_name] = dims

    print("\n" + "=" * 70)
    print("COMPOSITE UNIVERSALITY TEST -- SUMMARY")
    print("=" * 70)

    all_match = True
    for cfg_name, dims in results.items():
        cfg = CONFIGS[cfg_name]
        seq = [dims[lv] for lv in range(args.max_level + 1)]
        expected_seq = [EXPECTED.get(lv, "?") for lv in range(args.max_level + 1)]
        match = all(dims[lv] == EXPECTED[lv]
                    for lv in range(args.max_level + 1) if lv in EXPECTED)
        status = "MATCH" if match else "DIFFERS"
        if not match:
            all_match = False
        print(f"  {cfg['label']}")
        print(f"    Sequence: {seq}  (expected {expected_seq})  [{status}]")

    print()
    if all_match:
        print("  RESULT: Universality holds for composite singular potentials.")
        print("  The dimension sequence [3, 6, 17, 116] is invariant under")
        print("  arbitrary linear combinations of singular 1/r^p terms.")
    else:
        print("  RESULT: *** UNIVERSALITY BROKEN ***")
        print("  Composite potentials produce a DIFFERENT dimension sequence.")
        print("  This is a major discovery -- new paper material.")


if __name__ == "__main__":
    main()
