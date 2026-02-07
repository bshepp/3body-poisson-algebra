#!/usr/bin/env python3
"""
Unequal Mass Study — Dimension Sequence vs Mass Configuration
==============================================================

Tests whether the Lie algebra dimension sequence (3, 6, 17, 116, ...)
depends on the mass ratios.  Runs levels 0-3 for several configurations:

  1. Equal masses:        m1=m2=m3=1          (baseline, S3 symmetry)
  2. Generic unequal:     m1=1, m2=2, m3=3    (no symmetry)
  3. Tsygvintsev case 1:  Σ mi·mj/(Σ mk)² = 1/3
  4. Tsygvintsev case 2:  Σ mi·mj/(Σ mk)² = 23/27
  5. Tsygvintsev case 3:  Σ mi·mj/(Σ mk)² = 2/9

The three Tsygvintsev cases are the exceptional mass ratios where the
linearised variational equations along the parabolic Lagrangian orbit
are partially integrable.  These are the ONLY cases not ruled out by
the first-order Morales-Ramis-Ziglin criterion.

Reference: Tsygvintsev, "On some exceptional cases in the integrability
of the three-body problem", Celest. Mech. Dyn. Astron. 99, 23-47 (2007)

Usage
-----
    python unequal_mass_study.py                    # all configs, levels 0-3
    python unequal_mass_study.py --max-level 2      # quick test
    python unequal_mass_study.py --config generic   # single config
"""

import os
import sys
import argparse
import numpy as np
from time import time
from sympy import (
    symbols, diff, Integer, cancel, expand, Rational,
    sqrt, solve, Add, Symbol
)
import sympy as sp

os.environ["PYTHONUNBUFFERED"] = "1"

# Import core machinery from exact_growth
from exact_growth import (
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    CHAIN_RULE, total_deriv,
    poisson_bracket, simplify_generator,
    sample_phase_space, svd_gap_analysis,
    lambdify_generators,
)


# =====================================================================
# Mass configurations
# =====================================================================
def tsygvintsev_parameter(m1, m2, m3):
    """Compute Σ mi·mj / (Σ mk)² — the Tsygvintsev parameter."""
    S = m1 + m2 + m3
    return (m1*m2 + m1*m3 + m2*m3) / S**2


def find_tsygvintsev_masses(target_param, m1=1, m2=1):
    """
    Find m3 such that Σ mi·mj / (Σ mk)² = target_param.
    Fixes m1 and m2, solves for m3.
    """
    m3_sym = symbols("m3_sym", positive=True)
    S = m1 + m2 + m3_sym
    param = (m1*m2 + m1*m3_sym + m2*m3_sym) / S**2
    eq = param - target_param
    sols = solve(eq, m3_sym)
    # Return positive real solutions
    real_sols = [s for s in sols if s.is_real and s > 0]
    if real_sols:
        return float(m1), float(m2), float(real_sols[0])
    # Try with m1=1, m2=2
    m1, m2 = 1, 2
    S = m1 + m2 + m3_sym
    param = (m1*m2 + m1*m3_sym + m2*m3_sym) / S**2
    eq = param - target_param
    sols = solve(eq, m3_sym)
    real_sols = [s for s in sols if s.is_real and s > 0]
    if real_sols:
        return float(m1), float(m2), float(real_sols[0])
    return None


def build_mass_configs():
    """Build the dictionary of mass configurations to test."""
    configs = {}

    # 1. Equal masses (baseline)
    configs["equal"] = {
        "m1": 1.0, "m2": 1.0, "m3": 1.0,
        "label": "Equal masses (m1=m2=m3=1)",
        "symmetry": "S3",
    }

    # 2. Generic unequal
    configs["generic"] = {
        "m1": 1.0, "m2": 2.0, "m3": 3.0,
        "label": "Generic unequal (m1=1, m2=2, m3=3)",
        "symmetry": "None",
    }

    # 3. Tsygvintsev exceptional cases
    # Case 1: param = 1/3
    # For m1=m2=m3=1: param = 3/(3²) = 1/3 — this IS equal masses!
    # So case 1 is just the equal-mass case. Let's verify and note it.
    configs["tsygvintsev_1"] = {
        "m1": 1.0, "m2": 1.0, "m3": 1.0,
        "label": "Tsygvintsev case 1: param=1/3 (= equal masses)",
        "symmetry": "S3",
        "tsygvintsev_param": Rational(1, 3),
    }

    # Case 2: param = 23/27
    # Solve: (m1*m2 + m1*m3 + m2*m3)/(m1+m2+m3)² = 23/27
    # With m1=1, m2=1: (1 + m3 + m3)/(2+m3)² = 23/27
    #   (1 + 2*m3) / (2+m3)² = 23/27
    #   27(1 + 2m3) = 23(2+m3)²
    #   27 + 54m3 = 23(4 + 4m3 + m3²)
    #   27 + 54m3 = 92 + 92m3 + 23m3²
    #   23m3² + 38m3 + 65 = 0 ... discriminant = 1444 - 5980 < 0
    # No real solution with m1=m2=1. Try m1=1, m2=2:
    #   (2 + m3 + 2m3)/(3+m3)² = 23/27
    #   (2 + 3m3)/(3+m3)² = 23/27
    #   27(2+3m3) = 23(3+m3)²
    #   54+81m3 = 23(9+6m3+m3²) = 207+138m3+23m3²
    #   23m3²+57m3+153 = 0 ... disc = 3249 - 14076 < 0
    # Still no real solution. This param > 1/3 requires very specific ratios.
    # Let me solve numerically with m1=1, m2 free:
    m3s = symbols("m3s", positive=True)
    m2s = symbols("m2s", positive=True)
    # Try m1=1, m2=m2s, m3 very large
    # param → m3*(m1+m2)/m3² = (m1+m2)/m3 → 0 as m3→∞
    # param at m3=0: m1*m2/(m1+m2)² 
    # For param=23/27 ≈ 0.852, need param close to max
    # Max of param is 1/3 for equal masses... wait, 1/3 < 23/27
    # Actually, the maximum of Σmi*mj/(Σmk)² is 1/3 (equal masses)
    # So 23/27 > 1/3 is IMPOSSIBLE for positive masses!
    # 
    # Re-reading Tsygvintsev: the parameter is different.
    # The paper uses: Σ mi*mj / (Σ mk)² 
    # But 23/3³ = 23/27 ≈ 0.852 > 1/3 = max for positive masses
    #
    # Wait — re-checking the paper statement:
    # "∑ m_i m_j/(∑ m_k)² = 1/3, 2³/3³, 2/3²"
    # That's 1/3, 8/27, 2/9
    # NOT 23/27! The "23" was likely "2³" = 8.
    # So: 1/3 ≈ 0.333, 8/27 ≈ 0.296, 2/9 ≈ 0.222

    # Case 2: param = 8/27
    # With m1=1, m2=1: (1 + 2m3)/(2+m3)² = 8/27
    # 27(1 + 2m3) = 8(2+m3)²
    # 27 + 54m3 = 32 + 32m3 + 8m3²
    # 8m3² - 22m3 + 5 = 0
    # m3 = (22 ± √(484-160))/16 = (22 ± √324)/16 = (22 ± 18)/16
    # m3 = 40/16 = 2.5 or m3 = 4/16 = 0.25
    configs["tsygvintsev_2"] = {
        "m1": 1.0, "m2": 1.0, "m3": 2.5,
        "label": "Tsygvintsev case 2: param=8/27 (m1=1,m2=1,m3=2.5)",
        "symmetry": "Z2 (swap 1<->2)",
        "tsygvintsev_param": Rational(8, 27),
    }

    # Case 3: param = 2/9
    # With m1=1, m2=1: (1 + 2m3)/(2+m3)² = 2/9
    # 9(1 + 2m3) = 2(2+m3)²
    # 9 + 18m3 = 8 + 8m3 + 2m3²
    # 2m3² - 10m3 - 1 = 0
    # m3 = (10 ± sqrt(100+8))/4 = (10 ± sqrt(108))/4 = (10 ± 6*sqrt(3))/4
    # m3 = (10 + 10.392)/4 = 5.098 or negative
    m3_case3 = (10 + 6*float(np.sqrt(3))) / 4
    configs["tsygvintsev_3"] = {
        "m1": 1.0, "m2": 1.0, "m3": round(m3_case3, 6),
        "label": f"Tsygvintsev case 3: param=2/9 (m1=1,m2=1,m3={m3_case3:.4f})",
        "symmetry": "Z2 (swap 1<->2)",
        "tsygvintsev_param": Rational(2, 9),
    }

    # Verify parameters
    for key, cfg in configs.items():
        p = tsygvintsev_parameter(cfg["m1"], cfg["m2"], cfg["m3"])
        cfg["computed_param"] = p
        if "tsygvintsev_param" in cfg:
            expected = float(cfg["tsygvintsev_param"])
            assert abs(p - expected) < 1e-6, \
                f"{key}: param={p}, expected={expected}"

    return configs


def build_hamiltonians(m1, m2, m3, G=1):
    """
    Build pairwise Hamiltonians for arbitrary masses.

    H_ij = T_i + T_j - G * m_i * m_j * u_ij

    where T_i = (px_i² + py_i²) / (2 * m_i)  [kinetic energy]
    and u_ij = 1/r_ij  [inverse distance, polynomial representation]
    """
    T1 = (px1**2 + py1**2) / (2 * Rational(m1).limit_denominator(10000))
    T2 = (px2**2 + py2**2) / (2 * Rational(m2).limit_denominator(10000))
    T3 = (px3**2 + py3**2) / (2 * Rational(m3).limit_denominator(10000))

    Gm = Rational(G).limit_denominator(10000)
    M1 = Rational(m1).limit_denominator(10000)
    M2 = Rational(m2).limit_denominator(10000)
    M3 = Rational(m3).limit_denominator(10000)

    H12 = T1 + T2 - Gm * M1 * M2 * u12
    H13 = T1 + T3 - Gm * M1 * M3 * u13
    H23 = T2 + T3 - Gm * M2 * M3 * u23

    return H12, H13, H23


# =====================================================================
# Main computation
# =====================================================================
def compute_growth_unequal(config_name, config, max_level=3,
                           n_samples=500, seed=42):
    """Run Lie algebra growth computation for a given mass configuration."""
    m1, m2, m3 = config["m1"], config["m2"], config["m3"]

    print("=" * 70)
    print(f"MASS CONFIG: {config['label']}")
    print(f"  m1={m1}, m2={m2}, m3={m3}")
    print(f"  Tsygvintsev param: {config['computed_param']:.6f}")
    print(f"  Symmetry: {config['symmetry']}")
    print(f"  Max level: {max_level}, Samples: {n_samples}")
    print("=" * 70)

    H12, H13, H23 = build_hamiltonians(m1, m2, m3)

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
        print(f"\n  Level {level}: {len(new_exprs)} candidates "
              f"in {elapsed_level:.1f}s")

    # --- SVD Analysis ---
    print("\n" + "=" * 70)
    print("SVD ANALYSIS")
    print("=" * 70)

    Z_qp, Z_u = sample_phase_space(n_samples, seed)
    evaluate = lambdify_generators(all_exprs)
    eval_matrix = evaluate(Z_qp, Z_u)
    print(f"  Evaluation matrix shape: {eval_matrix.shape}")

    level_dims = {}
    for lv in range(max_level + 1):
        mask = [i for i, l in enumerate(all_levels) if l <= lv]
        sub = eval_matrix[:, mask]
        rank, svals = svd_gap_analysis(
            sub, label=f"(through level {lv})")
        level_dims[lv] = rank
        print(f"  ==> Dimension through level {lv}: {rank}")

    # Summary
    print(f"\n  Config: {config['label']}")
    dims = [level_dims[lv] for lv in range(max_level + 1)]
    print(f"  Dimension sequence: {dims}")

    new_per_level = [dims[0]]
    for i in range(1, len(dims)):
        new_per_level.append(dims[i] - dims[i - 1])
    print(f"  New-per-level: {new_per_level}")

    return {
        "config_name": config_name,
        "config": config,
        "dims": dims,
        "new_per_level": new_per_level,
        "level_dims": level_dims,
        "n_generators": len(all_exprs),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Unequal mass Lie algebra growth study"
    )
    ap.add_argument("--max-level", type=int, default=3,
                    help="Maximum bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space sample points (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=str, default="all",
                    choices=["all", "equal", "generic",
                             "tsygvintsev_2", "tsygvintsev_3"],
                    help="Which mass config to run (default: all)")
    args = ap.parse_args()

    configs = build_mass_configs()

    # Skip tsygvintsev_1 (= equal masses, redundant)
    run_configs = {k: v for k, v in configs.items()
                   if k != "tsygvintsev_1"}

    if args.config != "all":
        run_configs = {args.config: configs[args.config]}

    all_results = {}
    for name, cfg in run_configs.items():
        result = compute_growth_unequal(
            name, cfg,
            max_level=args.max_level,
            n_samples=args.samples,
            seed=args.seed,
        )
        all_results[name] = result
        print()

    # ---------------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MASS CONFIGURATION COMPARISON")
    print("=" * 70)

    # Equal-mass baseline
    eq_dims = configs["equal"]
    baseline = None
    if "equal" in all_results:
        baseline = all_results["equal"]["dims"]

    header = f"  {'Config':<45s}"
    for lv in range(args.max_level + 1):
        header += f"  L{lv:d}"
    header += "  | Sequence"
    print(header)
    print("  " + "-" * len(header))

    for name, result in all_results.items():
        dims = result["dims"]
        row = f"  {result['config']['label'][:45]:<45s}"
        for d in dims:
            row += f"  {d:>3d}"

        match_str = ""
        if baseline and name != "equal":
            if dims == baseline:
                match_str = "  SAME"
            else:
                match_str = "  DIFFERENT"

        row += f"  | {dims}{match_str}"
        print(row)

    # Key finding
    if len(all_results) > 1 and baseline:
        all_same = all(
            r["dims"] == baseline
            for n, r in all_results.items() if n != "equal"
        )
        print()
        if all_same:
            print("  *** ALL CONFIGS GIVE THE SAME DIMENSION SEQUENCE ***")
            print("  This suggests the sequence is a TOPOLOGICAL INVARIANT")
            print("  independent of mass ratios.")
        else:
            print("  *** DIMENSION SEQUENCE DEPENDS ON MASS CONFIGURATION ***")
            print("  The S3 symmetry of equal masses creates extra")
            print("  dependencies that reduce the count.")


if __name__ == "__main__":
    main()
