#!/usr/bin/env python3
"""Compute level-4 dimension of the Poisson algebra for 1/r and log potentials.

This is the CORRECT way to check closure: run compute_growth(max_level=4).
If dim(level 4) == dim(level 3) == 116, the algebra closes.
If it grows, we get the exact next term in the sequence.

The previous ad-hoc scripts (diagnose_brackets.py, check_1r_closure.py)
reinvented the closure check numerically when this infrastructure already
existed. This script uses the symbolic pipeline as intended.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nbody'))
from exact_growth_nbody import NBodyAlgebra


def run_closure_check(potential, external_potential=None, n_samples=500,
                      checkpoint_dir=None):
    """Run compute_growth to level 4 for the given potential."""
    label = potential
    if external_potential:
        label += f" + trap(omega={external_potential['omega']})"

    print(f"\n{'#' * 70}")
    print(f"# CLOSURE CHECK: {label}")
    print(f"# If level-4 dim == 116, the algebra closes.")
    print(f"{'#' * 70}\n")

    if checkpoint_dir is None:
        ckpt_dir = os.path.join(
            os.path.dirname(__file__), "results",
            f"closure_{potential.replace('/', '_')}"
            + ("_trap" if external_potential else "")
        )
    else:
        ckpt_dir = checkpoint_dir

    algebra = NBodyAlgebra(
        n_bodies=3,
        d_spatial=1,
        potential=potential,
        checkpoint_dir=ckpt_dir,
        external_potential=external_potential,
    )

    algebra.compute_growth(
        max_level=4,
        n_samples=n_samples,
        resume=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", default="1/r",
                        choices=["1/r", "log", "1/r^2", "1/r^3"],
                        help="Potential to test")
    parser.add_argument("--trap", action="store_true",
                        help="Add harmonic trap (omega=1.0)")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint directory (for resuming from existing checkpoints)")
    args = parser.parse_args()

    ext = {"omega": 1.0} if args.trap else None
    run_closure_check(args.potential, ext, args.n_samples, args.checkpoint_dir)
