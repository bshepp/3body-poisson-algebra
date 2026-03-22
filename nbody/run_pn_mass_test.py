#!/usr/bin/env python3
"""
Post-Newtonian mass invariance test.

In the full 1PN Einstein-Infeld-Hoffmann Hamiltonian, the static
potential for pair (i,j) is:

    V_ij = -G m_i m_j / r_ij  -  G^2 m_i m_j (m_i + m_j) / (2 c^2 r_ij^2)

The 1PN correction coefficient depends on the individual masses,
making it pair-dependent.  This script tests whether the dimension
sequence remains invariant under different mass configurations, even
when the composite potential coefficients vary per pair.

This requires constructing the Hamiltonian directly with explicit
per-pair coefficients (Option B from the plan) rather than the
generic potential_params interface.

Mass configurations tested:
  1. Equal masses:        m = (1, 1, 1)
  2. Hierarchical:        m = (100, 10, 1)
  3. Mixed:               m = (3, 7, 11)
"""

import argparse
from itertools import combinations
from sympy import Rational, Integer, symbols
from exact_growth_nbody import NBodyAlgebra


def build_pn_algebra(masses_dict, c_val, max_level, n_samples, seed):
    """Build 1PN three-body algebra with pair-dependent coefficients.

    The static 1PN potential for pair (i,j) with G=1 is:
        V_ij = -m_i m_j u  -  m_i m_j (m_i + m_j) / (2 c^2) * u^2
    """
    c2 = c_val * c_val
    bodies = sorted(masses_dict.keys())
    pairs = list(combinations(bodies, 2))

    mass_label = "_".join(f"m{b}{masses_dict[b]}" for b in bodies)
    label = f"1PN_c{c_val}_{mass_label}"

    alg = NBodyAlgebra(
        n_bodies=len(bodies),
        d_spatial=2,
        potential="1/r",
        masses=masses_dict,
        checkpoint_dir=f"checkpoints_pn_{label}",
    )

    def kinetic(body):
        m = alg.masses[body]
        return sum(p ** 2 for p in alg.p_by_body[body]) / (2 * m)

    alg.hamiltonians = {}
    alg.hamiltonian_list = []
    alg.hamiltonian_names = []

    for bi, bj in pairs:
        u = alg.u_by_pair[(bi, bj)]
        mi = masses_dict[bi]
        mj = masses_dict[bj]

        newton_coeff = -mi * mj
        pn_coeff = Rational(-mi * mj * (mi + mj), 2 * c2)

        V = newton_coeff * u + pn_coeff * u ** 2
        H = kinetic(bi) + kinetic(bj) + V

        name = f"H{bi}{bj}"
        alg.hamiltonians[name] = H
        alg.hamiltonian_list.append(H)
        alg.hamiltonian_names.append(name)

    alg.potential = "composite"
    alg.potential_params = [("pair-dependent", 1), ("pair-dependent", 2)]

    return alg.compute_growth(
        max_level=max_level,
        n_samples=n_samples,
        seed=seed,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Post-Newtonian mass invariance test")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Max bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c-val", type=int, default=10,
                    help="Speed of light parameter (default: 10)")
    args = ap.parse_args()

    EXPECTED = {0: 3, 1: 6, 2: 17, 3: 116}

    mass_configs = {
        "equal":        {1: 1, 2: 1, 3: 1},
        "hierarchical": {1: 100, 2: 10, 3: 1},
        "mixed":        {1: 3, 2: 7, 3: 11},
    }

    results = {}
    for cfg_name, masses in mass_configs.items():
        mass_str = ", ".join(f"m{k}={v}" for k, v in sorted(masses.items()))
        print("\n" + "#" * 70)
        print(f"# 1PN mass config: {cfg_name}  ({mass_str})")
        print(f"# c = {args.c_val}")
        print("#" * 70 + "\n")

        dims = build_pn_algebra(
            masses, args.c_val,
            args.max_level, args.samples, args.seed,
        )
        results[cfg_name] = dims

    print("\n" + "=" * 70)
    print("POST-NEWTONIAN MASS INVARIANCE -- SUMMARY")
    print("=" * 70)

    all_match = True
    all_same = True
    first_seq = None

    for cfg_name, dims in results.items():
        masses = mass_configs[cfg_name]
        mass_str = ", ".join(f"m{k}={v}" for k, v in sorted(masses.items()))
        seq = [dims[lv] for lv in range(args.max_level + 1)]
        expected_seq = [EXPECTED.get(lv, "?") for lv in range(args.max_level + 1)]
        match = all(dims[lv] == EXPECTED[lv]
                    for lv in range(args.max_level + 1) if lv in EXPECTED)
        if not match:
            all_match = False
        status = "MATCH" if match else "DIFFERS"

        if first_seq is None:
            first_seq = seq
        elif seq != first_seq:
            all_same = False

        print(f"  {cfg_name} ({mass_str})")
        print(f"    Sequence: {seq}  (expected {expected_seq})  [{status}]")

    print()
    if all_match and all_same:
        print("  RESULT: 1PN mass invariance confirmed.")
        print("  The dimension sequence is independent of the mass distribution,")
        print("  even with pair-dependent 1PN correction coefficients.")
    elif all_same:
        print("  RESULT: All mass configs give the same sequence,")
        print("  but it differs from the Newtonian [3, 6, 17, 116].")
    else:
        print("  RESULT: *** MASS INVARIANCE BROKEN at 1PN ***")
        print("  Different mass configurations produce different dimension sequences.")


if __name__ == "__main__":
    main()
