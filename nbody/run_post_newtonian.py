#!/usr/bin/env python3
"""
Post-Newtonian (1PN) three-body Poisson algebra.

Computes the dimension sequence for the static 1PN approximation
to three gravitating compact objects.  The pairwise potential is:

    V_ij = -G m^2 / r + correction   (equal masses)

In the u = 1/r representation with G = m = 1 and c >> 1, this
becomes:

    V_ij = -u  -  u^2 / (2 c^2)

The leading 1PN correction enters as a u^2 term whose coefficient
is suppressed by 1/c^2 relative to the Newtonian term.  We test
several values of c to probe the weak-field to moderate-field regime.

For the full Einstein-Infeld-Hoffmann Hamiltonian the 1PN correction
also includes momentum-dependent terms.  This script uses the STATIC
approximation (position-dependent terms only), which is the natural
first test of the framework.

Scientific question: does the 1PN correction alter the dimension
sequence [3, 6, 17, 116], or does universality extend to physically
motivated composite potentials?
"""

import argparse
from sympy import Rational, Integer, Symbol
from exact_growth_nbody import NBodyAlgebra


def main():
    ap = argparse.ArgumentParser(
        description="Static 1PN three-body Poisson algebra")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Max bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    EXPECTED = {0: 3, 1: 6, 2: 17, 3: 116}

    c_values = [10, 50, 100]
    results = {}

    for c_val in c_values:
        c2 = c_val * c_val
        pn_coeff = Rational(-1, 2 * c2)

        label = f"1PN with c={c_val} (correction ~ {float(pn_coeff):.2e})"
        print("\n" + "#" * 70)
        print(f"# {label}")
        print(f"# V_ij = -u + ({float(pn_coeff):.6f}) * u^2")
        print("#" * 70 + "\n")

        alg = NBodyAlgebra(
            n_bodies=3,
            d_spatial=2,
            potential_params=[(-Integer(1), 1), (pn_coeff, 2)],
        )
        dims = alg.compute_growth(
            max_level=args.max_level,
            n_samples=args.samples,
            seed=args.seed,
        )
        results[c_val] = dims

    # Also run pure Newtonian as baseline
    print("\n" + "#" * 70)
    print("# Baseline: Pure Newtonian (-u)")
    print("#" * 70 + "\n")

    alg = NBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r")
    dims_newton = alg.compute_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
    )
    results["Newtonian"] = dims_newton

    print("\n" + "=" * 70)
    print("POST-NEWTONIAN THREE-BODY -- SUMMARY")
    print("=" * 70)

    all_match = True
    for key in [*c_values, "Newtonian"]:
        dims = results[key]
        seq = [dims[lv] for lv in range(args.max_level + 1)]
        expected_seq = [EXPECTED.get(lv, "?") for lv in range(args.max_level + 1)]
        match = all(dims[lv] == EXPECTED[lv]
                    for lv in range(args.max_level + 1) if lv in EXPECTED)
        if not match:
            all_match = False
        status = "MATCH" if match else "DIFFERS"

        if key == "Newtonian":
            label = "Pure Newtonian (V = -u)"
        else:
            label = f"1PN c={key} (V = -u - u^2/{2*key*key})"
        print(f"  {label}")
        print(f"    Sequence: {seq}  (expected {expected_seq})  [{status}]")

    print()
    if all_match:
        print("  RESULT: Static 1PN correction preserves the dimension sequence.")
        print("  Universality extends to the post-Newtonian three-body problem.")
    else:
        print("  RESULT: *** 1PN correction changes the algebra ***")
        print("  The post-Newtonian regime has distinct algebraic structure.")


if __name__ == "__main__":
    main()
