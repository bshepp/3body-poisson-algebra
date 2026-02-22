#!/usr/bin/env python3
"""
Potential Comparison Study: 1/r vs 1/r² vs r²
================================================

Tests the Poisson algebra growth for three potential types to validate
the framework against known integrability results:

  1/r  (Newton):         Non-integrable -> infinite-dim algebra (known)
  1/r² (Calogero-Moser): Integrable     -> should be finite-dim
  r²   (Harmonic):       Integrable     -> should be finite-dim

Prediction: integrable potentials should produce an algebra that
stabilises at some finite level, while the 1/r potential grows
super-exponentially.

Usage
-----
    python potential_comparison.py                    # all 3 potentials
    python potential_comparison.py --potential newton  # single potential
    python potential_comparison.py --max-level 4      # deeper computation
"""

import os
import sys
import argparse
import numpy as np
from time import time
from sympy import symbols, diff, Integer, cancel, expand, Add
import sympy as sp

os.environ["PYTHONUNBUFFERED"] = "1"

from exact_growth import (
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    CHAIN_RULE, total_deriv,
    poisson_bracket, simplify_generator,
    sample_phase_space, svd_gap_analysis,
    lambdify_generators,
    T1, T2, T3,
)


# =====================================================================
# Potential definitions
# =====================================================================
def build_potentials():
    """Build Hamiltonians for each potential type."""

    # --- 1/r (Newton) ---
    # V_ij = -1/r_ij = -u_ij  (already the default in exact_growth.py)
    newton = {
        "name": "newton",
        "label": "1/r (Newton) — non-integrable",
        "H12": T1 + T2 - u12,
        "H13": T1 + T3 - u13,
        "H23": T2 + T3 - u23,
        "integrable": False,
    }

    # --- 1/r² (Calogero-Moser) ---
    # V_ij = -1/r_ij² = -u_ij²
    # Same u_ij = 1/r_ij, same chain rule table — only V changes
    calogero = {
        "name": "calogero",
        "label": "1/r^2 (Calogero-Moser) — integrable",
        "H12": T1 + T2 - u12**2,
        "H13": T1 + T3 - u13**2,
        "H23": T2 + T3 - u23**2,
        "integrable": True,
    }

    # --- r² (Harmonic) ---
    # V_ij = r_ij² = (x_i-x_j)² + (y_i-y_j)²
    # Already polynomial — no u_ij needed
    r12_sq = (x1 - x2)**2 + (y1 - y2)**2
    r13_sq = (x1 - x3)**2 + (y1 - y3)**2
    r23_sq = (x2 - x3)**2 + (y2 - y3)**2

    harmonic = {
        "name": "harmonic",
        "label": "r^2 (Harmonic) — integrable",
        "H12": T1 + T2 + r12_sq,
        "H13": T1 + T3 + r13_sq,
        "H23": T2 + T3 + r23_sq,
        "integrable": True,
    }

    return {"newton": newton, "calogero": calogero, "harmonic": harmonic}


# =====================================================================
# Growth computation (generic for any potential)
# =====================================================================
def compute_growth(potential, max_level=3, n_samples=500, seed=42):
    """Run Lie algebra growth for a given potential type."""

    H12 = potential["H12"]
    H13 = potential["H13"]
    H23 = potential["H23"]

    print("=" * 70)
    print(f"POTENTIAL: {potential['label']}")
    print(f"  Expected: {'integrable (finite-dim)' if potential['integrable'] else 'non-integrable (infinite-dim)'}")
    print(f"  Max level: {max_level}, Samples: {n_samples}")
    print("=" * 70)

    all_exprs = []
    all_names = []
    all_levels = []
    computed_pairs = set()

    # --- Level 0 ---
    print("\n--- Level 0: Pairwise Hamiltonians ---")
    for name, expr in [("H12", H12), ("H13", H13), ("H23", H23)]:
        all_exprs.append(expr)
        all_names.append(name)
        all_levels.append(0)
        print(f"  {name}: {len(Add.make_args(expr))} terms")

    for i in range(3):
        for j in range(i + 1, 3):
            computed_pairs.add(frozenset({i, j}))

    # --- Level 1 ---
    print("\n--- Level 1: Tidal-competition generators ---")
    pairs_l1 = [
        ("K1", "{H12,H13}", 0, 1),
        ("K2", "{H12,H23}", 0, 2),
        ("K3", "{H13,H23}", 1, 2),
    ]
    for short, full, i, j in pairs_l1:
        print(f"  Computing {full}...", end=" ", flush=True)
        t0 = time()
        expr = poisson_bracket(all_exprs[i], all_exprs[j])
        expr = simplify_generator(expr)
        elapsed = time() - t0
        nterms = len(Add.make_args(expr))
        print(f"{nterms} terms  [{elapsed:.1f}s]")

        if expr == 0:
            print(f"    *** {full} = 0 (Hamiltonians Poisson-commute!) ***")

        all_exprs.append(expr)
        all_names.append(short)
        all_levels.append(1)

    # --- Levels 2+ ---
    for level in range(2, max_level + 1):
        print(f"\n--- Level {level} ---")
        t_level = time()

        frontier_indices = [
            i for i, lv in enumerate(all_levels) if lv == level - 1
        ]
        n_existing = len(all_exprs)

        n_candidates = 0
        new_exprs = []
        new_names = []

        for i in frontier_indices:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)

                # Skip if either expression is zero
                if all_exprs[i] == 0 or all_exprs[j] == 0:
                    continue

                n_candidates += 1
                ni, nj = all_names[i], all_names[j]
                bracket_name = f"{{{ni},{nj}}}"

                print(f"  [{n_candidates:>4d}] {bracket_name}...",
                      end=" ", flush=True)
                t0 = time()
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                t_bracket = time() - t0

                t0s = time()
                expr = simplify_generator(expr)
                t_simp = time() - t0s

                nterms = len(Add.make_args(expr))
                print(f"bracket {t_bracket:.1f}s  "
                      f"simplify {t_simp:.1f}s  -> {nterms} terms")

                new_exprs.append(expr)
                new_names.append(bracket_name)

        for expr, name in zip(new_exprs, new_names):
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(level)

        elapsed_level = time() - t_level
        n_zero = sum(1 for e in new_exprs if e == 0)
        print(f"\n  Level {level}: {len(new_exprs)} candidates "
              f"({n_zero} zero) in {elapsed_level:.1f}s")

    # --- SVD Analysis ---
    # Filter out zero expressions for SVD
    nonzero_mask = [i for i, e in enumerate(all_exprs) if e != 0]
    nonzero_exprs = [all_exprs[i] for i in nonzero_mask]
    nonzero_levels = [all_levels[i] for i in nonzero_mask]

    print(f"\n{'=' * 70}")
    print("SVD ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Total generators: {len(all_exprs)} "
          f"(non-zero: {len(nonzero_exprs)})")

    if len(nonzero_exprs) == 0:
        print("  All expressions are zero — trivial algebra!")
        level_dims = {lv: 0 for lv in range(max_level + 1)}
        level_dims[0] = 3
        return {
            "potential": potential,
            "dims": [level_dims[lv] for lv in range(max_level + 1)],
            "level_dims": level_dims,
            "n_generators": len(all_exprs),
            "stabilised": True,
            "stabilised_at": 0,
        }

    Z_qp, Z_u = sample_phase_space(n_samples, seed)
    evaluate = lambdify_generators(nonzero_exprs)
    eval_matrix = evaluate(Z_qp, Z_u)
    print(f"  Evaluation matrix shape: {eval_matrix.shape}")

    level_dims = {}
    for lv in range(max_level + 1):
        mask = [i for i, l in enumerate(nonzero_levels) if l <= lv]
        if not mask:
            level_dims[lv] = 0
            print(f"  ==> Dimension through level {lv}: 0 (no non-zero generators)")
            continue
        sub = eval_matrix[:, mask]
        rank, svals = svd_gap_analysis(
            sub, label=f"(through level {lv})")
        level_dims[lv] = rank
        print(f"  ==> Dimension through level {lv}: {rank}")

    # Check for stabilisation
    dims = [level_dims[lv] for lv in range(max_level + 1)]
    stabilised = False
    stabilised_at = None
    for i in range(1, len(dims)):
        if dims[i] == dims[i - 1]:
            stabilised = True
            stabilised_at = i - 1
            break

    # Summary
    print(f"\n  Potential: {potential['label']}")
    print(f"  Dimension sequence: {dims}")

    new_per_level = [dims[0]]
    for i in range(1, len(dims)):
        new_per_level.append(dims[i] - dims[i - 1])
    print(f"  New-per-level: {new_per_level}")

    if stabilised:
        print(f"  *** STABILISED at level {stabilised_at} "
              f"(dim = {dims[stabilised_at]}) ***")
        if potential["integrable"]:
            print(f"  --> CONSISTENT with known integrability")
        else:
            print(f"  --> UNEXPECTED for non-integrable system!")
    else:
        print(f"  *** STILL GROWING at level {max_level} ***")
        if not potential["integrable"]:
            print(f"  --> CONSISTENT with known non-integrability")
        else:
            print(f"  --> May need more levels to see stabilisation")

    return {
        "potential": potential,
        "dims": dims,
        "new_per_level": new_per_level,
        "level_dims": level_dims,
        "n_generators": len(all_exprs),
        "n_nonzero": len(nonzero_exprs),
        "stabilised": stabilised,
        "stabilised_at": stabilised_at,
    }


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Potential comparison: 1/r vs 1/r^2 vs r^2"
    )
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--potential", type=str, default="all",
                    choices=["all", "newton", "calogero", "harmonic"])
    args = ap.parse_args()

    potentials = build_potentials()

    if args.potential == "all":
        run = potentials
    else:
        run = {args.potential: potentials[args.potential]}

    all_results = {}
    for name, pot in run.items():
        result = compute_growth(
            pot,
            max_level=args.max_level,
            n_samples=args.samples,
            seed=args.seed,
        )
        all_results[name] = result
        print()

    # ---------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("POTENTIAL COMPARISON TABLE")
        print("=" * 70)

        header = f"  {'Potential':<35s}"
        for lv in range(args.max_level + 1):
            header += f"  L{lv:d}"
        header += "  | Status"
        print(header)
        print("  " + "-" * (len(header) + 10))

        for name, result in all_results.items():
            dims = result["dims"]
            row = f"  {result['potential']['label'][:35]:<35s}"
            for d in dims:
                row += f"  {d:>3d}"

            if result["stabilised"]:
                row += (f"  | FINITE (stabilised at L"
                        f"{result['stabilised_at']}, "
                        f"dim={dims[result['stabilised_at']]})")
            else:
                row += f"  | GROWING"

            print(row)

        # Verdict
        print()
        integrable_finite = all(
            r["stabilised"] for n, r in all_results.items()
            if r["potential"]["integrable"]
        )
        nonintegrable_growing = all(
            not r["stabilised"] for n, r in all_results.items()
            if not r["potential"]["integrable"]
        )

        if integrable_finite and nonintegrable_growing:
            print("  *** FRAMEWORK VALIDATED ***")
            print("  Integrable potentials -> finite-dim algebra")
            print("  Non-integrable potential -> infinite-dim algebra")
            print("  The Poisson algebra growth is a computational")
            print("  integrability test.")
        elif integrable_finite:
            print("  Integrable potentials correctly produce finite algebras.")
            print("  Non-integrable case may need more levels.")
        else:
            print("  Results require further analysis.")

    # Save results
    with open("potential_comparison_results.txt", "w") as f:
        f.write("POTENTIAL COMPARISON RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for name, result in all_results.items():
            f.write(f"Potential: {result['potential']['label']}\n")
            f.write(f"  Dimensions: {result['dims']}\n")
            if "new_per_level" in result:
                f.write(f"  New-per-level: {result['new_per_level']}\n")
            f.write(f"  Stabilised: {result['stabilised']}")
            if result["stabilised"]:
                f.write(f" at level {result['stabilised_at']}")
            f.write("\n")
            f.write(f"  Generators: {result['n_generators']}")
            if "n_nonzero" in result:
                f.write(f" (non-zero: {result['n_nonzero']})")
            f.write("\n\n")

    print(f"\n  Results saved to potential_comparison_results.txt")


if __name__ == "__main__":
    main()
