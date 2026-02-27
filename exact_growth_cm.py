#!/usr/bin/env python3
"""
Exact Bracket Algebra Growth for Calogero-Moser System
=======================================================

Adapts the exact symbolic computation from exact_growth.py for the
Calogero-Moser (rational) 3-body system with V_ij = g^2 / r_ij^2.

In the u_ij = 1/r_ij auxiliary variable representation:
  V_ij = g^2 * u_ij^2

This is POLYNOMIAL in u_ij, making symbolic computation efficient.

The CM system is completely integrable (Moser 1975). Comparing its
bracket algebra dimension sequence against the gravitational system
(d = 3, 6, 17, 116) tests whether algebra growth distinguishes
integrability.

Usage:
    python exact_growth_cm.py                  # levels 0-3
    python exact_growth_cm.py --max-level 2    # quick test
"""

import os
import sys
import argparse
import json
import numpy as np
from time import time

import sympy as sp
from sympy import symbols, diff, Integer, cancel, expand

os.environ["PYTHONUNBUFFERED"] = "1"

# Large CM bracket expressions (15000+ terms) need deeper recursion for compilation
import sys
sys.setrecursionlimit(50000)

# Phase-space variables (same as gravitational)
x1, y1, x2, y2, x3, y3 = symbols("x1 y1 x2 y2 x3 y3", real=True)
px1, py1, px2, py2, px3, py3 = symbols("px1 py1 px2 py2 px3 py3", real=True)
u12, u13, u23 = symbols("u12 u13 u23", positive=True)

Q_VARS = [x1, y1, x2, y2, x3, y3]
P_VARS = [px1, py1, px2, py2, px3, py3]
U_VARS = [u12, u13, u23]
ALL_VARS = Q_VARS + P_VARS + U_VARS

CHECKPOINT_DIR = "checkpoints_cm"


# Chain rule: du_ij/dq_k (same as gravitational - depends only on r_ij)
def _build_chain_rule_table():
    table = {}
    table[(u12, x1)] = -(x1 - x2) * u12**3
    table[(u12, y1)] = -(y1 - y2) * u12**3
    table[(u12, x2)] = -(x2 - x1) * u12**3
    table[(u12, y2)] = -(y2 - y1) * u12**3
    table[(u13, x1)] = -(x1 - x3) * u13**3
    table[(u13, y1)] = -(y1 - y3) * u13**3
    table[(u13, x3)] = -(x3 - x1) * u13**3
    table[(u13, y3)] = -(y3 - y1) * u13**3
    table[(u23, x2)] = -(x2 - x3) * u23**3
    table[(u23, y2)] = -(y2 - y3) * u23**3
    table[(u23, x3)] = -(x3 - x2) * u23**3
    table[(u23, y3)] = -(y3 - y2) * u23**3
    return table


CHAIN_RULE = _build_chain_rule_table()


def total_deriv(expr, var):
    result = diff(expr, var)
    if var in P_VARS:
        return result
    for u_var in U_VARS:
        key = (u_var, var)
        if key in CHAIN_RULE:
            df_du = diff(expr, u_var)
            if df_du != 0:
                result += df_du * CHAIN_RULE[key]
    return result


def poisson_bracket(f, g):
    result = Integer(0)
    for q, p in zip(Q_VARS, P_VARS):
        df_dq = total_deriv(f, q)
        dg_dp = diff(g, p)
        df_dp = diff(f, p)
        dg_dq = total_deriv(g, q)
        result += df_dq * dg_dp - df_dp * dg_dq
    return result


def simplify_generator(expr):
    return cancel(expr)


# =================================================================
# CALOGERO-MOSER HAMILTONIANS
# V_ij = g^2 / r_ij^2 = g^2 * u_ij^2
# =================================================================
T1 = (px1**2 + py1**2) / 2
T2 = (px2**2 + py2**2) / 2
T3 = (px3**2 + py3**2) / 2

# g = 1 (coupling constant)
H12 = T1 + T2 + u12**2      # V12 = g^2 * u12^2 = u12^2
H13 = T1 + T3 + u13**2
H23 = T2 + T3 + u23**2


# Phase-space sampling
def sample_phase_space(n, seed=42, pos_range=3.0, mom_range=1.0, min_sep=0.5):
    rng = np.random.RandomState(seed)
    pts = np.empty((0, 12))
    for _ in range(200):
        bs = max((n - pts.shape[0]) * 5, 256)
        b = np.zeros((bs, 12))
        b[:, :6] = rng.uniform(-pos_range, pos_range, (bs, 6))
        b[:, 6:] = rng.uniform(-mom_range, mom_range, (bs, 6))
        dx12 = b[:, 0] - b[:, 2]; dy12 = b[:, 1] - b[:, 3]
        dx13 = b[:, 0] - b[:, 4]; dy13 = b[:, 1] - b[:, 5]
        dx23 = b[:, 2] - b[:, 4]; dy23 = b[:, 3] - b[:, 5]
        ok = ((dx12**2 + dy12**2 > min_sep**2) &
              (dx13**2 + dy13**2 > min_sep**2) &
              (dx23**2 + dy23**2 > min_sep**2))
        pts = np.vstack([pts, b[ok]])
        if pts.shape[0] >= n:
            break
    pts = pts[:n]
    dx12 = pts[:, 0] - pts[:, 2]; dy12 = pts[:, 1] - pts[:, 3]
    dx13 = pts[:, 0] - pts[:, 4]; dy13 = pts[:, 1] - pts[:, 5]
    dx23 = pts[:, 2] - pts[:, 4]; dy23 = pts[:, 3] - pts[:, 5]
    u12_vals = 1.0 / np.sqrt(dx12**2 + dy12**2)
    u13_vals = 1.0 / np.sqrt(dx13**2 + dy13**2)
    u23_vals = 1.0 / np.sqrt(dx23**2 + dy23**2)
    Z_u = np.column_stack([u12_vals, u13_vals, u23_vals])
    return pts, Z_u


def lambdify_generators(exprs):
    n = len(exprs)
    t0 = time()
    if n <= 50:
        print(f"    Lambdifying {n} expressions (standard)...", end=" ", flush=True)
        func = sp.lambdify(ALL_VARS, exprs, modules="numpy", cse=True)
        print(f"done [{time() - t0:.1f}s]")

        def evaluate(Z_qp, Z_u):
            args = ([Z_qp[:, i] for i in range(12)] +
                    [Z_u[:, i] for i in range(3)])
            vals = func(*args)
            return np.column_stack(vals)
        return evaluate

    print(f"    Lambdifying {n} expressions individually...", flush=True)
    funcs = []
    for idx, expr in enumerate(exprs):
        if (idx + 1) % 20 == 0 or idx == n - 1:
            print(f"      {idx+1}/{n}  [{time()-t0:.1f}s]", flush=True)
        nterms = len(sp.Add.make_args(expr))
        # Large expressions cause RecursionError with CSE=True due to
        # deep AST nesting in the compiler. Use CSE=False for those.
        use_cse = nterms < 5000
        try:
            f = sp.lambdify(ALL_VARS, expr, modules="numpy", cse=use_cse)
        except RecursionError:
            print(f"        (retrying expr {idx} without CSE)", flush=True)
            f = sp.lambdify(ALL_VARS, expr, modules="numpy", cse=False)
        funcs.append(f)
    print(f"    Total lambdify time: {time() - t0:.1f}s")

    def evaluate(Z_qp, Z_u):
        args = ([Z_qp[:, i] for i in range(12)] +
                [Z_u[:, i] for i in range(3)])
        cols = []
        for f in funcs:
            val = f(*args)
            cols.append(np.atleast_1d(val).ravel())
        return np.column_stack(cols)
    return evaluate


def svd_gap_analysis(eval_matrix, label=""):
    norms = np.linalg.norm(eval_matrix, axis=0)
    norms[norms < 1e-15] = 1.0
    M = eval_matrix / norms
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    noise_threshold = 1e-8 * s[0]
    n_meaningful = int(np.sum(s > noise_threshold))

    best_gap_ratio = 1.0
    best_gap_idx = -1
    for i in range(min(n_meaningful, len(s) - 1)):
        if s[i + 1] > noise_threshold:
            gap = s[i] / s[i + 1]
        else:
            gap = s[i] / max(s[i + 1], 1e-300)
        if gap > best_gap_ratio and i >= 2:
            best_gap_ratio = gap
            best_gap_idx = i

    print(f"\n  SVD SPECTRUM  {label}")
    for i in range(len(s)):
        rel = s[i] / s[0] if s[0] > 0 else 0.0
        if i < len(s) - 1 and s[i + 1] > 1e-300:
            gap = s[i] / s[i + 1]
        else:
            gap = float("inf")

        show = (i < 25 or
                (best_gap_idx >= 0 and abs(i - best_gap_idx) < 5) or
                gap > 100.0 or i >= len(s) - 3 or
                (i + 1) in (17, 63, 84, 116))
        if show:
            marker = ""
            if i + 1 in (3, 6, 17): marker = f"  <-- {i+1}"
            if i + 1 == 116: marker = "  <-- 116 (gravitational)"
            if i == best_gap_idx:
                marker += f"  *** GAP ({gap:.1e}) ***"
            elif gap > 100:
                marker += f"  (gap {gap:.1e})"
            print(f"  {i+1:>5} | {s[i]:>18.12f} | {gap:>12.2f} | {rel:>12.2e}{marker}")

    if best_gap_ratio > 1e4:
        rank = best_gap_idx + 1
        print(f"\n  DEFINITIVE GAP at index {rank}: ratio {best_gap_ratio:.2e}x")
    elif best_gap_ratio > 10:
        rank = best_gap_idx + 1
        print(f"\n  Gap at index {rank}: ratio {best_gap_ratio:.1f}x (moderate)")
    else:
        rank = n_meaningful
        print(f"\n  No clear gap (best ratio {best_gap_ratio:.1f}x)")
        print(f"  Using noise-floor threshold: rank = {rank}")

    return rank, s


def compute_exact_growth_cm(max_level=3, n_samples=500, seed=42, resume=False):
    print("=" * 70)
    print("EXACT BRACKET ALGEBRA GROWTH -- CALOGERO-MOSER SYSTEM")
    print("  V_ij = g^2 / r_ij^2 = u_ij^2  (g=1)")
    print("  Polynomial representation: u_ij auxiliary vars")
    print("=" * 70)
    print(f"  Equal masses m1=m2=m3=1, g=1")
    print(f"  Max level: {max_level},  Samples: {n_samples},  Seed: {seed}")
    print()

    start_level = 0
    all_exprs = []
    all_names = []
    all_levels = []
    computed_pairs = set()

    # Level 0
    if start_level <= 0:
        print("--- Level 0: Pairwise Hamiltonians (CM) ---")
        for name, expr in [("H12", H12), ("H13", H13), ("H23", H23)]:
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(0)
            nterms = len(sp.Add.make_args(expr))
            print(f"  {name}: {nterms} terms")

        for i in range(3):
            for j in range(i + 1, 3):
                computed_pairs.add(frozenset({i, j}))

    # Level 1
    if start_level <= 1:
        print("\n--- Level 1: Tidal-competition generators (CM) ---")
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
            nterms = len(sp.Add.make_args(expr))
            print(f"{nterms} terms  [{elapsed:.1f}s]")
            all_exprs.append(expr)
            all_names.append(short)
            all_levels.append(1)

    # Levels 2+
    for level in range(max(2, start_level), max_level + 1):
        print(f"\n--- Level {level} ---")
        t_level = time()
        frontier_indices = [i for i, lv in enumerate(all_levels) if lv == level - 1]
        n_existing = len(all_exprs)
        n_candidates = 0
        new_exprs_this_level = []
        new_names_this_level = []

        for i in frontier_indices:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)
                n_candidates += 1
                ni = all_names[i]
                nj = all_names[j]
                bracket_name = f"{{{ni},{nj}}}"
                print(f"  [{n_candidates:>4d}] {bracket_name}...", end=" ", flush=True)
                t0 = time()
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                t_bracket = time() - t0
                print(f"bracket {t_bracket:.1f}s...", end=" ", flush=True)
                t0s = time()
                expr = simplify_generator(expr)
                t_simp = time() - t0s
                nterms = len(sp.Add.make_args(expr))
                print(f"simplify {t_simp:.1f}s  -> {nterms} terms")
                new_exprs_this_level.append(expr)
                new_names_this_level.append(bracket_name)

        for expr, name in zip(new_exprs_this_level, new_names_this_level):
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(level)

        elapsed_level = time() - t_level
        print(f"\n  Level {level}: {len(new_exprs_this_level)} candidates "
              f"computed in {elapsed_level:.1f}s")

    # Numerical evaluation and SVD
    print("\n" + "=" * 70)
    print("NUMERICAL EVALUATION AND SVD ANALYSIS")
    print("=" * 70)

    Z_qp, Z_u = sample_phase_space(n_samples, seed)
    print(f"  Sample points: {Z_qp.shape[0]}")

    evaluate = lambdify_generators(all_exprs)
    print("    Evaluating at sample points...", end=" ", flush=True)
    t0 = time()
    eval_matrix = evaluate(Z_qp, Z_u)
    print(f"done [{time() - t0:.1f}s]")
    print(f"    Evaluation matrix shape: {eval_matrix.shape}")

    level_dims = {}
    for lv in range(max_level + 1):
        mask = [i for i, l in enumerate(all_levels) if l <= lv]
        sub = eval_matrix[:, mask]
        rank, svals = svd_gap_analysis(sub, label=f"(CM through level {lv})")
        level_dims[lv] = rank
        print(f"  ==> CM Dimension through level {lv}: {rank}")

    # Summary with gravitational comparison
    print("\n" + "=" * 70)
    print("DIMENSION COMPARISON: CM vs GRAVITATIONAL")
    print("=" * 70)

    grav_dims = {0: 3, 1: 6, 2: 17, 3: 116}

    for lv in range(max_level + 1):
        cm_dim = level_dims[lv]
        g_dim = grav_dims.get(lv, "?")
        match = "MATCH" if cm_dim == g_dim else f"DIFFER (grav={g_dim})"
        print(f"  Level {lv}: CM dim = {cm_dim:>5d}    Grav dim = {g_dim}    [{match}]")


def main():
    ap = argparse.ArgumentParser(
        description="Exact CM bracket algebra growth")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Maximum bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space sample points (default: 500)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
    args = ap.parse_args()

    compute_exact_growth_cm(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
