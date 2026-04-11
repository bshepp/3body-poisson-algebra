#!/usr/bin/env python3
# Track: Primes | Poisson algebra of the Dyson log-gas (GUE)
# Parent project: ../nbody/exact_growth_nbody.py
# See README.md in this directory for context.
"""
GUE Log-Gas Comparison: Poisson algebra dimensions for the Dyson log-gas
Hamiltonian that governs Riemann zeta zero correlations.

Runs four configurations at N=3, d=1:
  (a) Pure log-gas (no confinement)          — reference
  (b) GUE composite (log + harmonic ω=1)     — the key experiment
  (c) Penning trap (1/r + harmonic ω=1)      — reference
  (d) Harmonic only (quadratic pairwise)      — integrable reference

Usage:
    python primes/run_gue_logas.py                     # default: level 3
    python primes/run_gue_logas.py --max-level 2       # quick check
    python primes/run_gue_logas.py --max-level 3 --resume
"""

import argparse
import json
import os
import sys
import time

# Add parent and nbody directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
NBODY_DIR = os.path.join(PARENT_DIR, "nbody")
sys.path.insert(0, NBODY_DIR)
sys.path.insert(0, PARENT_DIR)

import sympy as sp
from exact_growth_nbody import NBodyAlgebra


CONFIGS = [
    {
        "name": "pure_log_gas",
        "label": "Pure Log-Gas (no confinement)",
        "description": "V_ij = -log|t_i - t_j|, no external potential",
        "physical_model": "2D point vortices / log-Coulomb gas",
        "kwargs": {
            "n_bodies": 3,
            "d_spatial": 1,
            "potential": "log",
        },
        "expected": [3, 6, 17, 116],
    },
    {
        "name": "gue_composite",
        "label": "GUE Composite (log + harmonic)",
        "description": "V_ij = -log|t_i - t_j| + harmonic confinement omega=1",
        "physical_model": "Dyson log-gas / GUE eigenvalue dynamics",
        "kwargs": {
            "n_bodies": 3,
            "d_spatial": 1,
            "potential": "log",
            "external_potential": {"omega": sp.Integer(1)},
        },
        "expected": [3, 6, 17, 116],
    },
    {
        "name": "penning_trap_1d",
        "label": "Penning Trap 1D (1/r + harmonic)",
        "description": "V_ij = 1/r_ij + harmonic confinement omega=1, all-repulsive",
        "physical_model": "Three ions in 1D harmonic trap",
        "kwargs": {
            "n_bodies": 3,
            "d_spatial": 1,
            "potential": "1/r",
            "charges": {1: 1, 2: 1, 3: 1},
            "external_potential": {"omega": sp.Integer(1)},
        },
        "expected": [3, 6, 17, 116],
    },
    {
        "name": "harmonic_only",
        "label": "Harmonic Only (quadratic pairwise)",
        "description": "V_ij = (t_i - t_j)^2, integrable system",
        "physical_model": "Coupled harmonic oscillators",
        "kwargs": {
            "n_bodies": 3,
            "d_spatial": 1,
            "potential": "composite",
            "potential_params": [(sp.Integer(1), -2)],
            # u^{-2} = r^2, giving harmonic pairwise interaction
        },
        "expected_note": "Should close (stabilize) — integrable system",
    },
]


def run_config(config, max_level, n_samples, seed, resume, results_dir):
    """Run a single configuration and return the dimension sequence."""
    name = config["name"]
    label = config["label"]

    print()
    print("=" * 70)
    print(f"CONFIG: {label}")
    print(f"  {config['description']}")
    print(f"  Physical model: {config['physical_model']}")
    print("=" * 70)

    checkpoint_dir = os.path.join(results_dir, f"checkpoints_{name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    kwargs = dict(config["kwargs"])
    kwargs["checkpoint_dir"] = checkpoint_dir

    t0 = time.time()
    alg = NBodyAlgebra(**kwargs)
    dims = alg.compute_growth(
        max_level=max_level,
        n_samples=n_samples,
        seed=seed,
        resume=resume,
    )
    elapsed = time.time() - t0

    seq = [dims[lv] for lv in range(max_level + 1) if lv in dims]

    print(f"\n  Dimension sequence: {seq}")
    print(f"  Elapsed: {elapsed:.1f}s")

    expected = config.get("expected")
    if expected:
        trimmed = expected[:len(seq)]
        if seq == trimmed:
            print(f"  MATCH: agrees with expected {trimmed}")
        else:
            print(f"  *** MISMATCH: expected {trimmed}, got {seq} ***")

    return {
        "name": name,
        "label": label,
        "description": config["description"],
        "physical_model": config["physical_model"],
        "dimensions": seq,
        "max_level": max_level,
        "n_samples": n_samples,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "expected": config.get("expected"),
        "match": seq == (config.get("expected", [])[:len(seq)]) if config.get("expected") else None,
    }


def print_summary(all_results):
    """Print comparison table."""
    print()
    print("=" * 70)
    print("SUMMARY: GUE LOG-GAS COMPARISON")
    print("=" * 70)
    print()

    max_lv = max(len(r["dimensions"]) for r in all_results)
    header = f"{'Configuration':<40s}"
    for lv in range(max_lv):
        header += f"  L{lv:d}"
    header += "   Status"
    print(header)
    print("-" * len(header))

    for r in all_results:
        line = f"{r['label']:<40s}"
        for lv in range(max_lv):
            if lv < len(r["dimensions"]):
                line += f"  {r['dimensions'][lv]:>4d}"
            else:
                line += "     -"
        if r["match"] is True:
            line += "   MATCH"
        elif r["match"] is False:
            line += "   *** MISMATCH ***"
        else:
            line += "   (no prediction)"
        print(line)

    print()

    # Check universality
    singular_results = [r for r in all_results if r["name"] != "harmonic_only"]
    if len(singular_results) >= 2:
        seqs = [tuple(r["dimensions"]) for r in singular_results]
        if len(set(seqs)) == 1:
            print("UNIVERSALITY CONFIRMED: All singular potentials give "
                  f"the same sequence {list(seqs[0])}")
        else:
            print("UNIVERSALITY BROKEN: Different singular potentials "
                  "give different sequences!")
            for r in singular_results:
                print(f"  {r['label']}: {r['dimensions']}")

    # Check GUE specifically
    gue = next((r for r in all_results if r["name"] == "gue_composite"), None)
    if gue and gue["match"]:
        print()
        print("KEY RESULT: The Poisson algebra of the exact GUE Hamiltonian")
        print("(governing zeta zero correlations) gives the SAME dimension")
        print("sequence as Newtonian gravity: " + str(gue["dimensions"]))


def main():
    ap = argparse.ArgumentParser(
        description="GUE log-gas Poisson algebra comparison")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Maximum bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space sample points (default: 500)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
    ap.add_argument("--configs", type=str, default=None,
                    help="Comma-separated config names to run "
                         "(default: all). Options: "
                         "pure_log_gas,gue_composite,penning_trap_1d,"
                         "harmonic_only")
    args = ap.parse_args()

    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Filter configs if requested
    configs = CONFIGS
    if args.configs:
        names = [n.strip() for n in args.configs.split(",")]
        configs = [c for c in CONFIGS if c["name"] in names]
        if not configs:
            print(f"ERROR: No matching configs for {names}")
            print(f"  Available: {[c['name'] for c in CONFIGS]}")
            sys.exit(1)

    print("=" * 70)
    print("GUE LOG-GAS POISSON ALGEBRA COMPARISON")
    print("=" * 70)
    print(f"  Configs:   {[c['name'] for c in configs]}")
    print(f"  Max level: {args.max_level}")
    print(f"  Samples:   {args.samples}")
    print(f"  Seed:      {args.seed}")
    print(f"  Resume:    {args.resume}")
    print(f"  Results:   {results_dir}")
    print()
    print("The GUE composite (log + harmonic) models the statistical")
    print("mechanics of Riemann zeta zeros via the Dyson log-gas.")
    print("Universality predicts [3, 6, 17, 116] for all singular")
    print("potentials including the GUE case.")

    all_results = []
    t_total = time.time()

    for config in configs:
        result = run_config(
            config, args.max_level, args.samples,
            args.seed, args.resume, results_dir,
        )
        all_results.append(result)

        # Save incrementally
        output_path = os.path.join(results_dir, "gue_comparison.json")
        with open(output_path, "w") as f:
            json.dump({
                "experiment": "gue_logas_comparison",
                "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "max_level": args.max_level,
                "n_samples": args.samples,
                "seed": args.seed,
                "results": all_results,
            }, f, indent=2, default=str)

    total_elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    print_summary(all_results)

    # Final save
    output_path = os.path.join(results_dir, "gue_comparison.json")
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "gue_logas_comparison",
            "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "max_level": args.max_level,
            "n_samples": args.samples,
            "seed": args.seed,
            "total_elapsed_seconds": round(total_elapsed, 1),
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
