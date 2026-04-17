#!/usr/bin/env python3
"""
Identify the 3 extra generators in the gradient-product neural algebra
that are NOT present in the physical N=3 universal algebra.

Strategy
--------
Both systems live in a 6D phase space (3 positions/weights + 3 momenta).
The physical universal algebra has [3, 6, 17, 116].
The neural gradient-product algebra has [3, 6, 17, 119].

We compute both on the SAME phase-space grid and diagonalize to identify
which linear combinations of neural generators are not reachable from
physical ones. Then we characterize those 3 directions.

Running
-------
    python neural/identify_extra_generators.py
"""
import os
import sys
import json
import numpy as np
from time import time
from itertools import combinations

import sympy as sp
from sympy import symbols, diff, Integer, cancel, expand, Add, Symbol

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_algebra import (
    poisson_bracket_generic, simplify_gen, sample_phase_space_nn,
    lambdify_exprs, svd_gap_analysis,
)


def build_neural_hamiltonians():
    """Build gradient-product pairwise Hamiltonians for 3-layer linear net."""
    w1, w2, w3 = symbols("w1 w2 w3", real=True)
    v1, v2, v3 = symbols("v1 v2 v3", real=True)
    weights = [w1, w2, w3]
    momenta = [v1, v2, v3]
    q_vars = [w1, w2, w3]
    p_vars = [v1, v2, v3]

    product = w1 * w2 * w3
    L = (product - 1)**2 / 2
    grads = [diff(L, w) for w in weights]

    hams = {}
    for i, j in combinations(range(3), 2):
        T = (momenta[i]**2 + momenta[j]**2) / 2
        V = grads[i] * grads[j] / 2
        hams[(i, j)] = expand(T + V)
    return hams, q_vars, p_vars, weights, momenta


def build_physical_hamiltonians_gravity():
    """Build gravitational pairwise Hamiltonians for N=3 in 1D.
    H_ij = (p_i^2 + p_j^2)/2 + 1/(q_i - q_j)
    but polynomial algebra is equivalent to H_ij = (p_i^2+p_j^2)/2 + U_ij
    with U_ij = 1/(q_i - q_j). For an apples-to-apples polynomial comparison
    we multiply by a common denominator and compute the algebra with
    exact rational functions.

    For a simpler polynomial proxy we use V_ij = (q_i - q_j)^{-1} via
    substitution. But since we need polynomial brackets, use v_ij as
    a symbolic variable with known Poisson bracket to positions.

    Simplest: use V_ij = 1/(q_i - q_j) as a rational function and
    let sympy handle it. Poisson brackets will be well-defined.
    """
    q1, q2, q3 = symbols("q1 q2 q3", real=True)
    p1, p2, p3 = symbols("p1 p2 p3", real=True)
    positions = [q1, q2, q3]
    momenta = [p1, p2, p3]

    hams = {}
    for i, j in combinations(range(3), 2):
        T = (momenta[i]**2 + momenta[j]**2) / 2
        V = 1 / (positions[i] - positions[j])
        hams[(i, j)] = T + V
    return hams, positions, momenta


def compute_level3_generators(hams, q_vars, p_vars, label="",
                              max_level=3, verbose=True):
    """Compute all level-0..3 brackets as symbolic expressions."""
    pair_keys = sorted(hams.keys())
    all_exprs = []
    all_names = []
    all_levels = []
    computed_pairs = set()
    n_pairs = len(pair_keys)

    for idx, (i, j) in enumerate(pair_keys):
        expr = hams[(i, j)]
        all_exprs.append(expr)
        all_names.append(f"H{i+1}{j+1}")
        all_levels.append(0)

    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            computed_pairs.add(frozenset({i, j}))

    if verbose:
        print(f"\n[{label}] Level 0: {n_pairs} generators")

    for level in range(1, max_level + 1):
        t0 = time()
        if level == 1:
            frontier = list(range(n_pairs))
        else:
            frontier = [i for i, lv in enumerate(all_levels) if lv == level - 1]
        n_existing = len(all_exprs)
        new_exprs, new_names = [], []

        if level == 1:
            for i in range(n_pairs):
                for j in range(i + 1, n_pairs):
                    expr = simplify_gen(poisson_bracket_generic(
                        all_exprs[i], all_exprs[j], q_vars, p_vars))
                    new_exprs.append(expr)
                    new_names.append(f"{{{all_names[i]},{all_names[j]}}}")
        else:
            for i in frontier:
                for j in range(n_existing):
                    if i == j:
                        continue
                    pair = frozenset({i, j})
                    if pair in computed_pairs:
                        continue
                    computed_pairs.add(pair)
                    if all_exprs[i] == 0 or all_exprs[j] == 0:
                        continue
                    expr = simplify_gen(poisson_bracket_generic(
                        all_exprs[i], all_exprs[j], q_vars, p_vars))
                    new_exprs.append(expr)
                    new_names.append(f"{{{all_names[i]},{all_names[j]}}}")

        for e, n in zip(new_exprs, new_names):
            all_exprs.append(e)
            all_names.append(n)
            all_levels.append(level)

        if verbose:
            n_zero = sum(1 for e in new_exprs if e == 0)
            print(f"[{label}] Level {level}: {len(new_exprs)} candidates "
                  f"({n_zero} zero) [{time()-t0:.1f}s]")

    return all_exprs, all_names, all_levels


def analyze_neural_119():
    """Main analysis: compute 119-class basis and identify extra 3."""
    print("=" * 70)
    print("IDENTIFYING THE 3 EXTRA GENERATORS IN NEURAL [3,6,17,119]")
    print("=" * 70)

    hams, q_vars, p_vars, weights, momenta = build_neural_hamiltonians()

    exprs, names, levels = compute_level3_generators(
        hams, q_vars, p_vars, label="neural-gradient", max_level=3)

    print(f"\nTotal candidates: {len(exprs)}")
    print(f"Non-zero: {sum(1 for e in exprs if e != 0)}")

    nz_idx = [i for i, e in enumerate(exprs) if e != 0]
    nz_exprs = [exprs[i] for i in nz_idx]
    nz_names = [names[i] for i in nz_idx]
    nz_levels = [levels[i] for i in nz_idx]

    all_vars = q_vars + p_vars
    Z = sample_phase_space_nn(6, 800, seed=42)
    evaluate = lambdify_exprs(nz_exprs, all_vars)
    M = evaluate(Z)

    print(f"\nEval matrix shape: {M.shape}")

    norms = np.linalg.norm(M, axis=0)
    norms[norms < 1e-15] = 1.0
    Mn = M / norms

    U, s, Vt = np.linalg.svd(Mn, full_matrices=False)
    print(f"\nTop 20 singular values:")
    for k in range(min(20, len(s))):
        print(f"  s[{k:3d}] = {s[k]:.6e}")

    print(f"\nGap ratios near rank 116:")
    for k in range(110, min(125, len(s)-1)):
        r = s[k] / s[k+1] if s[k+1] > 0 else float('inf')
        print(f"  s[{k:3d}]={s[k]:.4e}  s[{k+1:3d}]={s[k+1]:.4e}  ratio={r:.2f}")

    rank_total, _ = svd_gap_analysis(M, label="full")
    print(f"\nDetected rank: {rank_total}")

    # Compute polynomial degrees for each non-zero expression
    degrees = []
    for expr in nz_exprs:
        if expr == 0:
            degrees.append(0)
            continue
        poly = sp.Poly(expr, *all_vars)
        degrees.append(poly.total_degree())
    unique_deg = sorted(set(degrees))
    print(f"\nPolynomial degrees across {len(nz_exprs)} generators:")
    for d in unique_deg:
        cnt = sum(1 for x in degrees if x == d)
        print(f"  Degree {d:2d}: {cnt} generators")

    # Find the level-3 generators with the maximum degree (most likely
    # the "extras")
    max_deg = max(degrees)
    print(f"\nGenerators at maximum degree {max_deg}:")
    for i in range(len(nz_exprs)):
        if degrees[i] == max_deg:
            nterms = len(Add.make_args(nz_exprs[i]))
            print(f"  [{nz_names[i]}] deg={degrees[i]} "
                  f"level={nz_levels[i]} nterms={nterms}")

    # Rank analysis by maximum degree: how does rank grow as we include
    # generators up to each degree?
    print(f"\nRank growth by degree ceiling:")
    for d in unique_deg:
        mask = [i for i, dd in enumerate(degrees) if dd <= d]
        if not mask:
            continue
        sub = M[:, mask]
        rank_d, _ = svd_gap_analysis(sub, label=f"<=deg{d}")
        print(f"  deg<={d:2d}: {len(mask)} cols, rank={rank_d}")


if __name__ == "__main__":
    analyze_neural_119()
