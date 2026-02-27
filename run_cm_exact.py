#!/usr/bin/env python3
"""
Run exact CM computation by patching exact_growth.py's Hamiltonians.
Reuses the full infrastructure including _make_flat_func for large expressions.
"""
import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(50000)

# Patch exact_growth to use CM Hamiltonians before importing
import sympy as sp

# Import all infrastructure from exact_growth
from exact_growth import (
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    CHAIN_RULE, total_deriv, poisson_bracket, simplify_generator,
    sample_phase_space, lambdify_generators, svd_gap_analysis,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    Integer,
)
import numpy as np
from time import time

# CM Hamiltonians: V_ij = g^2 * u_ij^2 (g=1)
T1 = (px1**2 + py1**2) / 2
T2 = (px2**2 + py2**2) / 2
T3 = (px3**2 + py3**2) / 2

H12 = T1 + T2 + u12**2
H13 = T1 + T3 + u13**2
H23 = T2 + T3 + u23**2


def compute_cm_exact(max_level=3, n_samples=500, seed=42):
    print("=" * 70)
    print("EXACT BRACKET ALGEBRA -- CALOGERO-MOSER (V = g^2/r^2 = u^2)")
    print("=" * 70)
    print(f"  Max level: {max_level}, Samples: {n_samples}, Seed: {seed}")

    all_exprs = []
    all_names = []
    all_levels = []
    computed_pairs = set()

    # Level 0
    print("\n--- Level 0 ---")
    for name, expr in [("H12", H12), ("H13", H13), ("H23", H23)]:
        all_exprs.append(expr)
        all_names.append(name)
        all_levels.append(0)
        print(f"  {name}: {len(sp.Add.make_args(expr))} terms")
    for i in range(3):
        for j in range(i + 1, 3):
            computed_pairs.add(frozenset({i, j}))

    # Level 1
    print("\n--- Level 1 ---")
    pairs_l1 = [
        ("K1", "{H12,H13}", 0, 1),
        ("K2", "{H12,H23}", 0, 2),
        ("K3", "{H13,H23}", 1, 2),
    ]
    for short, full, i, j in pairs_l1:
        print(f"  {full}...", end=" ", flush=True)
        t0 = time()
        expr = poisson_bracket(all_exprs[i], all_exprs[j])
        expr = simplify_generator(expr)
        nterms = len(sp.Add.make_args(expr))
        print(f"{nterms} terms [{time()-t0:.1f}s]")
        all_exprs.append(expr)
        all_names.append(short)
        all_levels.append(1)

    # Levels 2+
    for level in range(2, max_level + 1):
        print(f"\n--- Level {level} ---")
        t_level = time()
        frontier_indices = [i for i, lv in enumerate(all_levels) if lv == level - 1]
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
                print(f"  [{n_candidates:>4d}] {bracket_name}...", end=" ", flush=True)
                t0 = time()
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                t_b = time() - t0
                print(f"b={t_b:.1f}s...", end=" ", flush=True)
                t0 = time()
                expr = simplify_generator(expr)
                t_s = time() - t0
                nterms = len(sp.Add.make_args(expr))
                print(f"s={t_s:.1f}s -> {nterms} t")
                new_exprs.append(expr)
                new_names.append(bracket_name)

        for expr, name in zip(new_exprs, new_names):
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(level)

        print(f"\n  Level {level}: {len(new_exprs)} candidates [{time()-t_level:.1f}s]")

    # SVD analysis
    print("\n" + "=" * 70)
    print("SVD ANALYSIS")
    print("=" * 70)

    Z_qp, Z_u = sample_phase_space(n_samples, seed)
    print(f"  {Z_qp.shape[0]} sample points")

    evaluate = lambdify_generators(all_exprs)
    print("    Evaluating...", end=" ", flush=True)
    t0 = time()
    eval_matrix = evaluate(Z_qp, Z_u)
    print(f"done [{time()-t0:.1f}s]  shape={eval_matrix.shape}")

    level_dims = {}
    for lv in range(max_level + 1):
        mask = [i for i, l in enumerate(all_levels) if l <= lv]
        sub = eval_matrix[:, mask]
        rank, svals = svd_gap_analysis(sub, label=f"(CM through level {lv})")
        level_dims[lv] = rank
        print(f"  ==> CM d({lv}) = {rank}")

    # Comparison
    grav = {0: 3, 1: 6, 2: 17, 3: 116}
    print("\n" + "=" * 70)
    print("COMPARISON: CM vs GRAVITATIONAL")
    print("=" * 70)
    for lv in range(max_level + 1):
        cm = level_dims[lv]
        g = grav.get(lv, "?")
        m = "MATCH" if cm == g else f"DIFFER (grav={g})"
        print(f"  Level {lv}: CM={cm}, Grav={g}  [{m}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    compute_cm_exact(args.max_level, args.samples, args.seed)
