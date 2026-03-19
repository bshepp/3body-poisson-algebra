#!/usr/bin/env python3
# Track: Helium Extension | Coulomb 3-body Poisson algebra with charges
# Parent project: ../preprint.tex (planar 3-body results)
"""
Helium atom Poisson algebra: does the sign of the interaction matter?

Three experiments, all N=3, d=3, V=1/r:

  Exp 1 (control):       All-attractive gravitational, helium mass ratios
                          masses = {1:7344, 2:1, 3:1}, no charges
                          Expected: [3, 6, 17, 116] (mass invariance)

  Exp 2 (full helium):   Nucleus (q=+2) + 2 electrons (q=-1 each)
                          charges = {1:+2, 2:-1, 3:-1}
                          H_12, H_13 attractive; H_23 repulsive

  Exp 3 (all-repulsive): charges = {1:+1, 2:+1, 3:+1}
                          All interactions repulsive
"""

import argparse
from exact_growth_nbody import NBodyAlgebra


def run_experiment(label, max_level, n_samples, seed, resume, **kwargs):
    """Run a single experiment and return its dimension sequence."""
    print("\n" + "#" * 70)
    print(f"# {label}")
    print("#" * 70 + "\n")

    alg = NBodyAlgebra(n_bodies=3, d_spatial=3, potential="1/r", **kwargs)
    dims = alg.compute_growth(
        max_level=max_level,
        n_samples=n_samples,
        seed=seed,
        resume=resume,
    )
    seq = [dims[lv] for lv in range(max_level + 1)]
    print(f"\n  >>> {label}: dimension sequence = {seq}")
    return seq


def main():
    ap = argparse.ArgumentParser(
        description="Helium atom Poisson algebra experiments")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Max bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--experiment", type=int, choices=[1, 2, 3],
                    default=None,
                    help="Run only one experiment (1, 2, or 3)")
    args = ap.parse_args()

    common = dict(max_level=args.max_level, n_samples=args.samples,
                  seed=args.seed, resume=args.resume)

    results = {}

    if args.experiment in (None, 1):
        results["control"] = run_experiment(
            "Exp 1: All-attractive Coulomb (control, helium masses)",
            masses={1: 7344, 2: 1, 3: 1},
            **common,
        )

    if args.experiment in (None, 2):
        results["helium"] = run_experiment(
            "Exp 2: Full helium (nucleus +2, electrons -1)",
            masses={1: 7344, 2: 1, 3: 1},
            charges={1: +2, 2: -1, 3: -1},
            **common,
        )

    if args.experiment in (None, 3):
        results["repulsive"] = run_experiment(
            "Exp 3: All-repulsive Coulomb",
            charges={1: +1, 2: +1, 3: +1},
            **common,
        )

    print("\n" + "=" * 70)
    print("HELIUM ATOM SUMMARY")
    print("=" * 70)
    ref = [3, 6, 17, 116]
    print(f"  Reference (N=3, all-attractive): {ref}")
    for name, seq in results.items():
        match = "MATCH" if seq == ref[:len(seq)] else "DIFFERENT"
        print(f"  {name:20s}: {seq}  [{match}]")

    if len(results) >= 2:
        seqs = list(results.values())
        if all(s == seqs[0] for s in seqs):
            print("\n  *** All experiments give IDENTICAL sequences ***")
            print("  ==> Sign of interaction does NOT affect the algebra")
        else:
            print("\n  *** Sequences DIFFER -- sign of interaction matters! ***")


if __name__ == "__main__":
    main()
