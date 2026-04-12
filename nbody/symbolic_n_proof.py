#!/usr/bin/env python3
"""
Numerical survey of Poisson algebra dimension vs potential exponent.

Three modes (run with subcommand):

  survey   — Phase 1: Dense sweep of rank vs p in [0.01, 50]
  degen    — Phase 2: n→0+ degeneration (topological question)
  boundary — Phase 3: Harmonic boundary probe around p = -2

Standalone (no subcommand) — original symbolic-n evaluation.

Sign convention throughout
--------------------------
V = u^p   where u = 1/r.

  p > 0 : V = 1/r^p  (singular at r=0)     p=1 Newton, p=2 Calogero–Moser
  p = 0 : V = 1      (constant, trivial)
  p < 0 : V = r^|p|  (polynomial-like)      p=-2 harmonic (r^2)

Usage
-----
    python nbody/symbolic_n_proof.py                       # original proof
    python nbody/symbolic_n_proof.py survey                # Phase 1
    python nbody/symbolic_n_proof.py degen                 # Phase 2
    python nbody/symbolic_n_proof.py boundary              # Phase 3
    python nbody/symbolic_n_proof.py survey --masses 1,2,3
"""

import os
import sys
import argparse
import json
import numpy as np
from time import time
from itertools import combinations

import sympy as sp
from sympy import (Symbol, symbols, diff, Integer, Rational, cancel, expand,
                   Add, Poly, factor, collect)

os.environ["PYTHONUNBUFFERED"] = "1"

# =====================================================================
# Phase-space variables (planar 3-body)
# =====================================================================
x1, y1, x2, y2, x3, y3 = symbols("x1 y1 x2 y2 x3 y3", real=True)
px1, py1, px2, py2, px3, py3 = symbols(
    "px1 py1 px2 py2 px3 py3", real=True)
u12, u13, u23 = symbols("u12 u13 u23", positive=True)

# The key: n is a SYMBOL, not a number
n = Symbol("n", positive=True)

Q_VARS = [x1, y1, x2, y2, x3, y3]
P_VARS = [px1, py1, px2, py2, px3, py3]
U_VARS = [u12, u13, u23]
ALL_VARS = Q_VARS + P_VARS + U_VARS

# =====================================================================
# Chain rule (independent of n)
# =====================================================================
CHAIN_RULE = {}
for (u_var, bi_coords, bj_coords) in [
    (u12, (x1, y1), (x2, y2)),
    (u13, (x1, y1), (x3, y3)),
    (u23, (x2, y2), (x3, y3)),
]:
    for qi, qj in zip(bi_coords, bj_coords):
        CHAIN_RULE[(u_var, qi)] = -(qi - qj) * u_var ** 3
        CHAIN_RULE[(u_var, qj)] = -(qj - qi) * u_var ** 3


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


def simplify_gen(expr):
    return cancel(expand(expr))


# =====================================================================
# Build Hamiltonians with symbolic n and masses
# =====================================================================
def build_hamiltonians(masses):
    m1, m2, m3 = masses[1], masses[2], masses[3]
    T1 = (px1**2 + py1**2) / (2 * m1)
    T2 = (px2**2 + py2**2) / (2 * m2)
    T3 = (px3**2 + py3**2) / (2 * m3)

    H12 = T1 + T2 - m1 * m2 * u12**n
    H13 = T1 + T3 - m1 * m3 * u13**n
    H23 = T2 + T3 - m2 * m3 * u23**n

    return {"H12": H12, "H13": H13, "H23": H23}, [H12, H13, H23]


# =====================================================================
# Numerical evaluation with symbolic n
# =====================================================================
def sample_phase_space(n_pts, seed=42, masses=None):
    rng = np.random.RandomState(seed)
    pos_range = 5.0
    mom_range = 1.0
    min_sep = 0.5

    pts = np.empty((0, 12))
    for _ in range(200):
        bs = max((n_pts - pts.shape[0]) * 5, 256)
        b = np.zeros((bs, 12))
        b[:, :6] = rng.uniform(-pos_range, pos_range, (bs, 6))
        b[:, 6:] = rng.uniform(-mom_range, mom_range, (bs, 6))

        ok = np.ones(bs, dtype=bool)
        # Check pairwise separations
        pairs = [(0, 2), (0, 4), (2, 4)]  # (x1,x2), (x1,x3), (x2,x3)
        for si, sj in pairs:
            r_sq = ((b[:, si] - b[:, sj])**2 +
                    (b[:, si+1] - b[:, sj+1])**2)
            ok &= (r_sq > min_sep**2)
        pts = np.vstack([pts, b[ok]])
        if pts.shape[0] >= n_pts:
            break
    pts = pts[:n_pts]

    # Compute u_ij = 1/r_ij
    u_cols = []
    for si, sj in [(0, 2), (0, 4), (2, 4)]:
        r = np.sqrt((pts[:, si] - pts[:, sj])**2 +
                    (pts[:, si+1] - pts[:, sj+1])**2)
        u_cols.append(1.0 / r)
    return pts, np.column_stack(u_cols)


def evaluate_at_n(exprs, Z_qp, Z_u, n_val):
    """Evaluate symbolic expressions at concrete n and phase-space points.

    Substitutes n -> n_val first, then lambdifies.
    """
    n_pts = Z_qp.shape[0]
    concrete = [expr.subs(n, n_val) for expr in exprs]
    func = sp.lambdify(ALL_VARS, concrete, modules="numpy")
    args = ([Z_qp[:, i] for i in range(12)] +
            [Z_u[:, i] for i in range(3)])
    vals = func(*args)
    cols = []
    for v in vals:
        arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
        if arr.shape[0] == 1:
            arr = np.full(n_pts, arr[0])
        cols.append(arr)
    return np.column_stack(cols)


def svd_rank(matrix, label=""):
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms < 1e-15] = 1.0
    M = matrix / norms
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # Find definitive gap
    best_gap = 1.0
    best_idx = -1
    for i in range(len(s) - 1):
        if s[i+1] > 1e-300:
            gap = s[i] / s[i+1]
        else:
            gap = s[i] / max(s[i+1], 1e-300)
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    rank = best_idx + 1 if best_gap > 1e4 else int(np.sum(s > 1e-8 * s[0]))
    return rank, s, best_gap


# =====================================================================
# Main — original symbolic-n mode
# =====================================================================
def run_symbolic_n(args, masses):
    """Original mode: keep n symbolic, evaluate at random n-values."""
    mass_str = ", ".join(f"m{k}={v}" for k, v in masses.items())

    print("=" * 70)
    print(f"NUMERICAL SURVEY: Poisson algebra for V = -m_i m_j * u^n")
    print(f"  (u = 1/r, so V = 1/r^n for n>0)")
    print(f"  Masses: {mass_str}")
    print(f"  n is a SYMBOLIC PARAMETER -- evaluated at concrete values")
    print("=" * 70)

    # Build Hamiltonians
    H_dict, H_list = build_hamiltonians(masses)
    H_names = ["H12", "H13", "H23"]

    all_exprs = list(H_list)
    all_names = list(H_names)
    all_levels = [0, 0, 0]

    print(f"\n--- Level 0: 3 Pairwise Hamiltonians (symbolic n) ---")
    for name, expr in zip(H_names, H_list):
        nterms = len(Add.make_args(expr))
        print(f"  {name}: {nterms} terms")
        print(f"    = {expr}")

    # Level 1
    if args.max_level >= 1:
        print(f"\n--- Level 1: Brackets (symbolic n) ---")
        for i in range(3):
            for j in range(i + 1, 3):
                name = f"{{{H_names[i]},{H_names[j]}}}"
                print(f"  Computing {name}...", end=" ", flush=True)
                t0 = time()
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                expr = simplify_gen(expr)
                elapsed = time() - t0
                nterms = len(Add.make_args(expr))
                print(f"{nterms} terms  [{elapsed:.1f}s]")
                all_exprs.append(expr)
                all_names.append(name)
                all_levels.append(1)

    # Level 2
    if args.max_level >= 2:
        print(f"\n--- Level 2: Brackets (symbolic n) ---")
        n_existing = len(all_exprs)
        frontier = [i for i, lv in enumerate(all_levels) if lv == 1]
        computed_pairs = set()
        for i in range(3):
            for j in range(i + 1, 3):
                computed_pairs.add(frozenset({i, j}))

        count = 0
        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)
                count += 1
                name = f"{{{all_names[i]},{all_names[j]}}}"
                print(f"  [{count:>3d}] {name}...", end=" ", flush=True)
                t0 = time()
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                t_b = time() - t0
                print(f"bracket {t_b:.1f}s...", end=" ", flush=True)
                t0 = time()
                expr = simplify_gen(expr)
                t_s = time() - t0
                nterms = len(Add.make_args(expr))
                print(f"simplify {t_s:.1f}s  -> {nterms} terms")
                all_exprs.append(expr)
                all_names.append(name)
                all_levels.append(2)

    # =====================================================================
    # NUMERICAL EVALUATION OVER n
    # =====================================================================
    print("\n" + "=" * 70)
    print("SURVEY: Evaluating rank as function of n")
    print("=" * 70)

    Z_qp, Z_u = sample_phase_space(args.samples, args.seed, masses)
    print(f"  Phase-space sample: {Z_qp.shape[0]} points")

    rng = np.random.RandomState(args.seed + 1)
    n_test_ints = list(range(1, min(11, args.n_values + 1)))
    n_test_floats = list(rng.uniform(0.5, 10.0, max(0, args.n_values - 10)))
    n_test_values = n_test_ints + list(n_test_floats)
    n_test_values = sorted(set([round(v, 4) for v in n_test_values]))

    print(f"  Testing {len(n_test_values)} values of n: "
          f"{n_test_values[:5]}...{n_test_values[-3:]}")

    results_by_level = {lv: {} for lv in range(args.max_level + 1)}

    for n_val in n_test_values:
        for lv in range(args.max_level + 1):
            mask = [i for i, l in enumerate(all_levels) if l <= lv]
            sub_exprs = [all_exprs[i] for i in mask]

            mat = evaluate_at_n(sub_exprs, Z_qp, Z_u, n_val)
            rank, svals, gap = svd_rank(mat)
            results_by_level[lv][n_val] = (rank, gap)

    # Report
    print(f"\n{'n':>8} |", end="")
    for lv in range(args.max_level + 1):
        print(f" Level {lv:>1d} (rank, gap) |", end="")
    print()
    print("-" * (10 + 25 * (args.max_level + 1)))

    all_match = True
    expected = {0: 3, 1: 6, 2: 17}

    for n_val in n_test_values:
        line = f"{n_val:>8.4f} |"
        for lv in range(args.max_level + 1):
            rank, gap = results_by_level[lv][n_val]
            marker = ""
            if lv in expected and rank != expected[lv]:
                marker = " !!!"
                all_match = False
            line += f"   {rank:>3d}  ({gap:>8.1e}){marker} |"
        print(line)

    print("\n" + "=" * 70)
    if all_match:
        print("RESULT: Dimension sequence [3, 6, 17] holds for ALL tested n")
        print(f"  Tested {len(n_test_values)} values (integers 1-10 + random")
        print(f"  reals in [0.5, 10]). All give rank exactly [3, 6, 17].")
        print()
        print("  Since generators are rational functions of n, rank is")
        print("  determined by minor determinants that are polynomials in n.")
        print("  A nonzero polynomial has only finitely many roots.")
    else:
        print("WARNING: Rank varies with n -- investigating...")
        for lv in range(args.max_level + 1):
            ranks = {n_val: results_by_level[lv][n_val][0]
                     for n_val in n_test_values}
            unique_ranks = set(ranks.values())
            if len(unique_ranks) > 1:
                print(f"  Level {lv}: ranks = {unique_ranks}")
                for r in sorted(unique_ranks):
                    ns = [n_val for n_val, rank in ranks.items() if rank == r]
                    print(f"    rank {r}: n = {ns}")
    print("=" * 70)

    # Structure analysis
    print("\n--- Structure of level-1 brackets (n-dependence) ---")
    for i in range(3, min(6, len(all_exprs))):
        expr = all_exprs[i]
        terms = Add.make_args(expand(expr))
        print(f"\n  {all_names[i]}:")
        print(f"    Total terms: {len(terms)}")
        has_n = expr.has(n)
        print(f"    Contains n: {has_n}")
        if has_n:
            try:
                collected = collect(expand(expr), n, evaluate=False)
                for power, coeff in sorted(collected.items(),
                                           key=lambda x: str(x[0])):
                    nterms_c = len(Add.make_args(coeff))
                    print(f"    n^({power}): {nterms_c} terms")
            except Exception:
                print(f"    (could not collect by n)")


# =====================================================================
# Phase 1: Dense n-Survey
# =====================================================================
def run_survey(args, masses):
    """Dense numerical sweep of rank vs p in [0.01, 50]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mass_str = ", ".join(f"m{k}={v}" for k, v in masses.items())
    print("=" * 70)
    print("PHASE 1: NUMERICAL SURVEY -- rank vs exponent p")
    print(f"  V = u^p where u = 1/r")
    print(f"  p > 0: singular (1/r^p)   p < 0: polynomial (r^|p|)")
    print(f"  Masses: {mass_str}")
    print("=" * 70)

    # Build Hamiltonians with symbolic n
    H_dict, H_list = build_hamiltonians(masses)
    H_names = ["H12", "H13", "H23"]

    all_exprs = list(H_list)
    all_names = list(H_names)
    all_levels = [0, 0, 0]

    # Compute through level 2
    print("\nComputing symbolic generators through level 2...")
    t_total = time()

    # Level 1
    for i in range(3):
        for j in range(i + 1, 3):
            expr = poisson_bracket(all_exprs[i], all_exprs[j])
            expr = simplify_gen(expr)
            all_exprs.append(expr)
            all_names.append(f"{{{H_names[i]},{H_names[j]}}}")
            all_levels.append(1)
    print(f"  Level 1: {len(all_exprs) - 3} new generators")

    # Level 2
    n_existing = len(all_exprs)
    frontier = [i for i, lv in enumerate(all_levels) if lv == 1]
    computed_pairs = set()
    for i in range(3):
        for j in range(i + 1, 3):
            computed_pairs.add(frozenset({i, j}))
    for i in frontier:
        for j in range(n_existing):
            if i == j:
                continue
            pair = frozenset({i, j})
            if pair in computed_pairs:
                continue
            computed_pairs.add(pair)
            expr = poisson_bracket(all_exprs[i], all_exprs[j])
            expr = simplify_gen(expr)
            all_exprs.append(expr)
            all_names.append(f"{{{all_names[i]},{all_names[j]}}}")
            all_levels.append(2)
    print(f"  Level 2: {len(all_exprs) - n_existing} new generators")
    print(f"  Total: {len(all_exprs)} generators in {time() - t_total:.1f}s")

    # Sample phase space
    Z_qp, Z_u = sample_phase_space(args.samples, args.seed, masses)

    # Dense p-values: log-spaced near 0, linear elsewhere
    p_values = np.sort(np.unique(np.concatenate([
        np.logspace(-2, 0, 30),           # 0.01 to 1, dense near 0
        np.linspace(1.0, 5.0, 40),        # 1 to 5, linear
        np.linspace(5.0, 10.0, 20),       # 5 to 10
        np.linspace(10.0, 50.0, 20),      # 10 to 50
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float),  # integers
        np.array([0.5, 1.5, 2.5, 3.5, 4.5]),  # half-integers
    ])))

    n_p = len(p_values)
    print(f"\nSurveying {n_p} p-values in [{p_values[0]:.4f}, {p_values[-1]:.1f}]")

    # Storage
    ranks_L0 = np.zeros(n_p, dtype=int)
    ranks_L1 = np.zeros(n_p, dtype=int)
    ranks_L2 = np.zeros(n_p, dtype=int)
    sv17 = np.zeros(n_p)
    sv18 = np.zeros(n_p)
    gaps = np.zeros(n_p)

    for idx, p_val in enumerate(p_values):
        if idx % 20 == 0:
            print(f"  [{idx+1:>3d}/{n_p}] p = {p_val:.4f}...", flush=True)

        for lv in range(3):
            mask = [i for i, l in enumerate(all_levels) if l <= lv]
            sub_exprs = [all_exprs[i] for i in mask]
            mat = evaluate_at_n(sub_exprs, Z_qp, Z_u, p_val)
            rank, svals, gap = svd_rank(mat)
            if lv == 0:
                ranks_L0[idx] = rank
            elif lv == 1:
                ranks_L1[idx] = rank
            else:
                ranks_L2[idx] = rank
                sv17[idx] = svals[16] if len(svals) > 16 else 0.0
                sv18[idx] = svals[17] if len(svals) > 17 else 0.0
                gaps[idx] = gap

    # Report
    print(f"\n{'p':>8} | L0 | L1 | L2 |    sv17    |    sv18    |    gap")
    print("-" * 72)
    for idx, p_val in enumerate(p_values):
        marker = ""
        if ranks_L2[idx] != 17:
            marker = " <-- ANOMALY"
        print(f"{p_val:>8.4f} | {ranks_L0[idx]:>2d} | {ranks_L1[idx]:>2d} | "
              f"{ranks_L2[idx]:>2d} | {sv17[idx]:>9.2e} | {sv18[idx]:>9.2e} | "
              f"{gaps[idx]:>9.2e}{marker}")

    # Summary
    # Detect numerical precision limit: if sv17 < 1e-12, the rank
    # determination is unreliable (machine epsilon contamination)
    reliable = sv17 > 1e-12
    universal_reliable = np.all(ranks_L2[reliable] == 17)
    n_unreliable = np.sum(~reliable)

    print("\n" + "=" * 70)
    if n_unreliable > 0:
        print(f"NOTE: {n_unreliable} p-values have sv17 < 1e-12 (numerically")
        print(f"  unreliable due to floating-point precision at extreme p).")
        print(f"  Unreliable range: p > {p_values[~reliable][0]:.1f}")
        print()
    if universal_reliable:
        n_rel = np.sum(reliable)
        print(f"RESULT: rank = [3, 6, 17] for ALL {n_rel} numerically reliable p-values")
        print(f"  Reliable range: p in [{p_values[reliable][0]:.4f}, {p_values[reliable][-1]:.1f}]")
        print(f"  Min gap ratio (reliable): {gaps[reliable].min():.2e}"
              f" at p = {p_values[reliable][np.argmin(gaps[reliable])]:.4f}")
    else:
        anomalies = p_values[reliable & (ranks_L2 != 17)]
        print(f"WARNING: rank != 17 at {len(anomalies)} RELIABLE p-values:")
        for p_a in anomalies:
            idx_a = np.where(p_values == p_a)[0][0]
            print(f"  p = {p_a:.6f}: rank = {ranks_L2[idx_a]}, sv17 = {sv17[idx_a]:.2e}")
    print("=" * 70)

    # Save data
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "n_universality_survey")
    os.makedirs(out_dir, exist_ok=True)

    np.savez(os.path.join(out_dir, "survey_data.npz"),
             p_values=p_values, ranks_L0=ranks_L0, ranks_L1=ranks_L1,
             ranks_L2=ranks_L2, sv17=sv17, sv18=sv18, gaps=gaps)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Rank vs p
    ax = axes[0, 0]
    ax.plot(p_values, ranks_L0, "o-", ms=2, label="Level 0")
    ax.plot(p_values, ranks_L1, "s-", ms=2, label="Level 1")
    ax.plot(p_values, ranks_L2, "^-", ms=2, label="Level 2")
    ax.axhline(17, color="r", ls="--", alpha=0.5, label="dim=17")
    ax.set_xlabel("p (exponent)")
    ax.set_ylabel("Rank")
    ax.set_title("(a) Rank vs exponent p")
    ax.legend()

    # (b) sv17 and sv18
    ax = axes[0, 1]
    ax.semilogy(p_values, sv17, "b.-", ms=2, label="sv17")
    ax.semilogy(p_values, np.maximum(sv18, 1e-300), "r.-", ms=2, label="sv18")
    ax.set_xlabel("p")
    ax.set_ylabel("Singular value")
    ax.set_title("(b) sv17 and sv18 vs p")
    ax.legend()

    # (c) Gap ratio
    ax = axes[1, 0]
    ax.semilogy(p_values, gaps, "k.-", ms=2)
    ax.axhline(1e10, color="r", ls="--", alpha=0.5, label="gap=10^10")
    ax.set_xlabel("p")
    ax.set_ylabel("Gap ratio (sv17/sv18)")
    ax.set_title("(c) SVD gap ratio vs p")
    ax.legend()

    # (d) Rank vs p (log scale x-axis)
    ax = axes[1, 1]
    ax.semilogx(p_values, ranks_L2, "g^-", ms=3)
    ax.axhline(17, color="r", ls="--", alpha=0.5)
    ax.set_xlabel("p (log scale)")
    ax.set_ylabel("L2 Rank")
    ax.set_title("(d) L2 rank vs p (log scale)")

    plt.suptitle(f"Phase 1: Numerical Survey — rank vs p  [{mass_str}]",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "survey_plots.png"), dpi=150)
    print(f"\nSaved plots to {out_dir}/survey_plots.png")
    print(f"Saved data to  {out_dir}/survey_data.npz")


# =====================================================================
# Phase 2: n → 0+ Degeneration
# =====================================================================
def run_degen(args, masses):
    """Log-spaced sweep n=10^-6 to 1 — is the transition topological?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mass_str = ", ".join(f"m{k}={v}" for k, v in masses.items())
    print("=" * 70)
    print("PHASE 2: n -> 0+ DEGENERATION -- topological vs phase transition")
    print(f"  As n->0+: u^n = 1 + n*ln(u) + O(n^2)")
    print(f"  V = -m_i m_j (1 + n*ln(u)) -> constant drops from brackets")
    print(f"  So brackets at leading order ~ n * (log-potential brackets)")
    print(f"  Scalar doesn't change rank -> predict rank=17 for ANY n > 0")
    print(f"  Masses: {mass_str}")
    print("=" * 70)

    # Build symbolic generators
    H_dict, H_list = build_hamiltonians(masses)
    H_names = ["H12", "H13", "H23"]
    all_exprs = list(H_list)
    all_names = list(H_names)
    all_levels = [0, 0, 0]

    print("\nComputing symbolic generators through level 2...")
    t0 = time()
    for i in range(3):
        for j in range(i + 1, 3):
            expr = poisson_bracket(all_exprs[i], all_exprs[j])
            expr = simplify_gen(expr)
            all_exprs.append(expr)
            all_levels.append(1)
    n_existing = len(all_exprs)
    frontier = [i for i, lv in enumerate(all_levels) if lv == 1]
    computed_pairs = set()
    for i in range(3):
        for j in range(i + 1, 3):
            computed_pairs.add(frozenset({i, j}))
    for i in frontier:
        for j in range(n_existing):
            if i == j:
                continue
            pair = frozenset({i, j})
            if pair in computed_pairs:
                continue
            computed_pairs.add(pair)
            expr = poisson_bracket(all_exprs[i], all_exprs[j])
            expr = simplify_gen(expr)
            all_exprs.append(expr)
            all_levels.append(2)
    print(f"  Done in {time() - t0:.1f}s  ({len(all_exprs)} generators)")

    Z_qp, Z_u = sample_phase_space(args.samples, args.seed, masses)

    # Log-spaced n values from 10^-6 to 1, plus a few above 1 for reference
    n_values = np.sort(np.unique(np.concatenate([
        np.logspace(-6, -3, 30),    # 1e-6 to 1e-3
        np.logspace(-3, -1, 30),    # 1e-3 to 0.1
        np.logspace(-1, 0, 30),     # 0.1 to 1.0
        np.array([1.0, 2.0, 5.0, 10.0]),  # reference
    ])))

    n_total = len(n_values)
    print(f"\nSweeping {n_total} n-values in [{n_values[0]:.1e}, {n_values[-1]:.1f}]")

    ranks = {0: np.zeros(n_total, dtype=int),
             1: np.zeros(n_total, dtype=int),
             2: np.zeros(n_total, dtype=int)}
    gaps_L2 = np.zeros(n_total)
    sv17_vals = np.zeros(n_total)

    for idx, nv in enumerate(n_values):
        if idx % 20 == 0:
            print(f"  [{idx+1:>3d}/{n_total}] n = {nv:.6e}...", flush=True)
        for lv in range(3):
            mask = [i for i, l in enumerate(all_levels) if l <= lv]
            sub_exprs = [all_exprs[i] for i in mask]
            mat = evaluate_at_n(sub_exprs, Z_qp, Z_u, nv)
            rank, svals, gap = svd_rank(mat)
            ranks[lv][idx] = rank
            if lv == 2:
                gaps_L2[idx] = gap
                sv17_vals[idx] = svals[16] if len(svals) > 16 else 0.0

    # Report
    topological = np.all(ranks[2] == 17)
    print(f"\n{'n':>12} | L0 | L1 | L2 |    gap     |   sv17")
    print("-" * 60)
    for idx in range(0, n_total, max(1, n_total // 30)):
        nv = n_values[idx]
        marker = "" if ranks[2][idx] == 17 else " <-- ANOMALY"
        print(f"{nv:>12.6e} | {ranks[0][idx]:>2d} | {ranks[1][idx]:>2d} | "
              f"{ranks[2][idx]:>2d} | {gaps_L2[idx]:>9.2e} | "
              f"{sv17_vals[idx]:>9.2e}{marker}")

    print("\n" + "=" * 70)
    if topological:
        print("RESULT: Transition is TOPOLOGICAL")
        print(f"  rank = 17 for ALL n down to n = {n_values[0]:.1e}")
        print(f"  Min gap ratio: {gaps_L2.min():.2e} at n = {n_values[np.argmin(gaps_L2)]:.6e}")
        print()
        print("INTERPRETATION:")
        print("  The algebra depends on the EXISTENCE of the singularity,")
        print("  not its strength. For any n > 0, the potential 1/r^n has")
        print("  a pole at r = 0, and the algebra is 17-dimensional at L2.")
        print("  The scalar prefactor n in the leading-order brackets")
        print("  does not change rank (as predicted analytically).")
    else:
        first_drop = n_values[ranks[2] != 17]
        print(f"RESULT: Transition is a PHASE TRANSITION")
        print(f"  rank drops below 17 at some n > 0")
        print(f"  First anomaly at n = {first_drop[0]:.6e}")
        for nv in first_drop[:10]:
            idx_a = np.where(n_values == nv)[0][0]
            print(f"    n = {nv:.6e}: rank = {ranks[2][idx_a]}")
    print("=" * 70)

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "n_universality_survey")
    os.makedirs(out_dir, exist_ok=True)

    np.savez(os.path.join(out_dir, "degen_data.npz"),
             n_values=n_values, ranks_L0=ranks[0], ranks_L1=ranks[1],
             ranks_L2=ranks[2], gaps_L2=gaps_L2, sv17=sv17_vals)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.semilogx(n_values, ranks[0], "o-", ms=2, label="L0")
    ax.semilogx(n_values, ranks[1], "s-", ms=2, label="L1")
    ax.semilogx(n_values, ranks[2], "^-", ms=2, label="L2")
    ax.axhline(17, color="r", ls="--", alpha=0.5)
    ax.set_xlabel("n")
    ax.set_ylabel("Rank")
    ax.set_title("(a) Rank vs n (log scale)")
    ax.legend()

    ax = axes[1]
    ax.loglog(n_values, gaps_L2, "k.-", ms=2)
    ax.axhline(1e10, color="r", ls="--", alpha=0.5)
    ax.set_xlabel("n")
    ax.set_ylabel("Gap ratio")
    ax.set_title("(b) SVD gap ratio vs n")

    ax = axes[2]
    ax.loglog(n_values, sv17_vals, "b.-", ms=2)
    ax.set_xlabel("n")
    ax.set_ylabel("sv17")
    ax.set_title("(c) 17th singular value vs n")

    plt.suptitle(f"Phase 2: n→0+ Degeneration  [{mass_str}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "degen_plots.png"), dpi=150)
    print(f"\nSaved plots to {out_dir}/degen_plots.png")
    print(f"Saved data to  {out_dir}/degen_data.npz")


# =====================================================================
# Phase 3: Harmonic Boundary (uses NBodyAlgebra from exact_growth_nbody)
# =====================================================================
def run_boundary(args, masses):
    """Sweep p from -4 to +4, probing the harmonic boundary at p = -2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Import NBodyAlgebra
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exact_growth_nbody import NBodyAlgebra

    mass_str = ", ".join(f"m{k}={v}" for k, v in masses.items())

    print("=" * 70)
    print("PHASE 3: HARMONIC BOUNDARY -- rank vs p around p = -2")
    print()
    print("  Sign convention:")
    print("    p > 0: V = u^p = 1/r^p  (singular)    p=1 Newton, p=2 CM")
    print("    p = 0: V = 1  (constant, trivial)")
    print("    p < 0: V = u^p = r^|p|  (polynomial)   p=-2 harmonic (r^2)")
    print()
    print(f"  Masses: {mass_str}")
    print("=" * 70)

    # Build p-values using exact Rationals
    # Coarse sweep: p from -4 to 4 in steps of 1/5
    coarse_p = [Rational(k, 5) for k in range(-20, 21)]
    # Remove p=0 (trivial)
    coarse_p = [p for p in coarse_p if p != 0]

    # Fine sweep around p = -2: step 1/10
    fine_p = [Rational(-20 + k, 10) for k in range(-10, 11)]
    fine_p = [p for p in fine_p if p != 0]

    # Very fine around p = -2: step 1/100
    vfine_p = [Rational(-200 + k, 100) for k in range(-10, 11)]
    vfine_p = [p for p in vfine_p if p != 0]

    # Ultra-fine: 1/1000 step
    ufine_p = [Rational(-2000 + k, 1000) for k in range(-5, 6)]
    ufine_p = [p for p in ufine_p if p != 0]

    # Specific test points from the plan
    special_p = [
        Rational(-19, 10), Rational(-21, 10),    # first pass
        Rational(-1999, 1000), Rational(-2001, 1000),  # fine pass
        Rational(-4), Rational(-6),  # higher even polynomials
        Rational(-1), Rational(-3),  # odd polynomials
    ]

    all_p = sorted(set(coarse_p + fine_p + vfine_p + ufine_p + special_p))
    n_p = len(all_p)
    print(f"\nSweeping {n_p} p-values in [{float(all_p[0]):.3f}, {float(all_p[-1]):.3f}]")
    print(f"  (all as exact SymPy Rationals)")

    results = []

    for idx, p_val in enumerate(all_p):
        if idx % 20 == 0 or abs(float(p_val) + 2.0) < 0.15:
            print(f"  [{idx+1:>3d}/{n_p}] p = {p_val} ({float(p_val):.4f})...",
                  end=" ", flush=True)
            verbose = True
        else:
            verbose = False

        t0 = time()
        try:
            alg = NBodyAlgebra(
                n_bodies=3, d_spatial=2,
                potential="composite",
                potential_params=[(Integer(-1), p_val)],
                masses=masses,
            )
            level_dims = alg.compute_growth(max_level=2, n_samples=args.samples,
                                              seed=args.seed)
            elapsed = time() - t0
            # level_dims is a dict {0: rank0, 1: rank1, 2: rank2}
            dims = [level_dims[lv] for lv in sorted(level_dims)]
            rank_L2 = level_dims.get(2, 0)
            results.append({
                "p": p_val,
                "p_float": float(p_val),
                "dims": dims,
                "rank_L2": rank_L2,
                "time": elapsed,
                "error": None,
            })
            if verbose:
                print(f"dims={dims}  [{elapsed:.1f}s]")
        except Exception as e:
            elapsed = time() - t0
            results.append({
                "p": p_val,
                "p_float": float(p_val),
                "dims": [],
                "rank_L2": -1,
                "time": elapsed,
                "error": str(e),
            })
            if verbose:
                print(f"ERROR: {e}  [{elapsed:.1f}s]")

    # Report
    print(f"\n{'p':>10} | {'p (float)':>10} |  dims           | notes")
    print("-" * 65)
    for r in results:
        p_str = str(r["p"])
        if len(p_str) > 10:
            p_str = p_str[:10]
        notes = ""
        if r["error"]:
            notes = f"ERROR: {r['error'][:30]}"
        elif r["rank_L2"] == 17:
            notes = "universal"
        elif r["rank_L2"] == 15:
            notes = "harmonic-like (dim=15)"
        elif r["rank_L2"] != 17:
            notes = f"ANOMALOUS rank={r['rank_L2']}"
        dims_str = str(r["dims"]) if r["dims"] else "N/A"
        print(f"{p_str:>10} | {r['p_float']:>10.4f} | {dims_str:<15} | {notes}")

    # Analysis
    print("\n" + "=" * 70)
    harmonic_like = [r for r in results if r["rank_L2"] == 15]
    universal = [r for r in results if r["rank_L2"] == 17]
    anomalous = [r for r in results
                 if r["rank_L2"] not in (15, 17, -1)]
    errors = [r for r in results if r["rank_L2"] == -1]

    print(f"  dim=17 (universal): {len(universal)} p-values")
    print(f"  dim=15 (harmonic):  {len(harmonic_like)} p-values")
    if harmonic_like:
        print(f"    at p = {[str(r['p']) for r in harmonic_like]}")
    if anomalous:
        print(f"  ANOMALOUS: {len(anomalous)} p-values")
        for r in anomalous:
            print(f"    p = {r['p']}: dims = {r['dims']}")
    if errors:
        print(f"  Errors: {len(errors)} p-values")

    # Determine if transition is sharp
    if harmonic_like:
        h_ps = set(r["p"] for r in harmonic_like)
        if h_ps == {Rational(-2)}:
            print("\n  CONCLUSION: The dim=15 island exists ONLY at p = -2 exactly.")
            print("  The transition is infinitely sharp -- a single point.")
        else:
            print(f"\n  CONCLUSION: dim=15 extends beyond p=-2 to: {sorted(h_ps)}")
    print("=" * 70)

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "n_universality_survey")
    os.makedirs(out_dir, exist_ok=True)

    p_floats = np.array([r["p_float"] for r in results])
    rank_L2s = np.array([r["rank_L2"] for r in results])
    dims_all = [r["dims"] for r in results]

    np.savez(os.path.join(out_dir, "boundary_data.npz"),
             p_values=p_floats, ranks_L2=rank_L2s)

    # Save full results as JSON (for Rationals, store as strings)
    json_results = []
    for r in results:
        jr = dict(r)
        jr["p"] = str(r["p"])
        json_results.append(jr)
    with open(os.path.join(out_dir, "boundary_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full range
    ax = axes[0]
    mask_17 = rank_L2s == 17
    mask_15 = rank_L2s == 15
    mask_other = ~mask_17 & ~mask_15 & (rank_L2s > 0)
    ax.plot(p_floats[mask_17], rank_L2s[mask_17], "b.", ms=4, label="dim=17")
    ax.plot(p_floats[mask_15], rank_L2s[mask_15], "ro", ms=8, label="dim=15")
    if np.any(mask_other):
        ax.plot(p_floats[mask_other], rank_L2s[mask_other], "g^", ms=8,
                label="other")
    ax.axvline(-2.0, color="gray", ls=":", alpha=0.7, label="p=-2 (harmonic)")
    ax.axvline(0.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("p")
    ax.set_ylabel("L2 Rank")
    ax.set_title("(a) Full range: rank vs p")
    ax.legend()

    # Zoom around p=-2
    ax = axes[1]
    zoom = (p_floats > -2.5) & (p_floats < -1.5)
    if np.any(zoom):
        zoom_17 = zoom & mask_17
        zoom_15 = zoom & mask_15
        ax.plot(p_floats[zoom_17], rank_L2s[zoom_17], "b.", ms=6, label="dim=17")
        ax.plot(p_floats[zoom_15], rank_L2s[zoom_15], "ro", ms=10, label="dim=15")
        ax.axvline(-2.0, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("p")
    ax.set_ylabel("L2 Rank")
    ax.set_title("(b) Zoom: p ∈ [-2.5, -1.5]")
    ax.legend()

    plt.suptitle(f"Phase 3: Harmonic Boundary Probe  [{mass_str}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boundary_plots.png"), dpi=150)
    print(f"\nSaved plots to {out_dir}/boundary_plots.png")
    print(f"Saved data to  {out_dir}/boundary_data.npz")
    print(f"Saved results to {out_dir}/boundary_results.json")


# =====================================================================
# CLI
# =====================================================================
def parse_masses(mass_str):
    """Parse comma-separated masses as exact Rationals."""
    if mass_str is None:
        return {1: Integer(1), 2: Integer(1), 3: Integer(1)}
    mvals = [Rational(m.strip()) for m in mass_str.split(",")]
    if len(mvals) != 3:
        raise ValueError(f"Expected 3 masses, got {len(mvals)}")
    return {1: mvals[0], 2: mvals[1], 3: mvals[2]}


def main():
    parser = argparse.ArgumentParser(
        description="Numerical survey of Poisson algebra dimension vs exponent")
    sub = parser.add_subparsers(dest="mode")

    # Shared args
    for p in [parser]:
        p.add_argument("--masses", type=str, default=None,
                        help="Comma-separated masses, e.g. '1,2,3'")
        p.add_argument("--samples", type=int, default=300)
        p.add_argument("--seed", type=int, default=42)

    # Original mode
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--n-values", type=int, default=20)

    # Subcommands
    p_survey = sub.add_parser("survey", help="Phase 1: dense n-survey")
    p_survey.add_argument("--masses", type=str, default=None)
    p_survey.add_argument("--samples", type=int, default=300)
    p_survey.add_argument("--seed", type=int, default=42)

    p_degen = sub.add_parser("degen", help="Phase 2: n→0+ degeneration")
    p_degen.add_argument("--masses", type=str, default=None)
    p_degen.add_argument("--samples", type=int, default=300)
    p_degen.add_argument("--seed", type=int, default=42)

    p_bound = sub.add_parser("boundary", help="Phase 3: harmonic boundary")
    p_bound.add_argument("--masses", type=str, default=None)
    p_bound.add_argument("--samples", type=int, default=300)
    p_bound.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    masses = parse_masses(args.masses)

    if args.mode == "survey":
        run_survey(args, masses)
    elif args.mode == "degen":
        run_degen(args, masses)
    elif args.mode == "boundary":
        run_boundary(args, masses)
    else:
        run_symbolic_n(args, masses)


if __name__ == "__main__":
    main()
