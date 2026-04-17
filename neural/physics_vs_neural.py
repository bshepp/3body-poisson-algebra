#!/usr/bin/env python3
"""
Direct comparison: physical N=3 algebra vs neural gradient-product algebra
in the SAME 6D phase space.

We evaluate both on the SAME random phase-space points, concatenate the
evaluation matrices column-wise, and compute ranks to identify:

- rank(physical_116)
- rank(neural_119)
- rank([physical | neural])   <-- if = 119, neural strictly contains physics
                                    up to linear recombination
- Which 3 neural generators are NOT in the span of physical 116

This tells us the 3 extra generators in a basis-aware way.
"""
import os
import sys
import numpy as np
from time import time
from itertools import combinations

import sympy as sp
from sympy import symbols, diff, Integer, cancel, expand, Add, Symbol

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_algebra import (
    poisson_bracket_generic, simplify_gen,
    lambdify_exprs, svd_gap_analysis,
)


def build_hamiltonians(potential_fn, q_vars, p_vars, n=3):
    """Build H_ij = (p_i^2 + p_j^2)/2 + V(q_i, q_j)."""
    hams = {}
    for i, j in combinations(range(n), 2):
        T = (p_vars[i]**2 + p_vars[j]**2) / 2
        V = potential_fn(q_vars[i], q_vars[j])
        hams[(i, j)] = T + V
    return hams


def compute_level3(hams, q_vars, p_vars, label="", max_level=3,
                   verbose=True, fast_simplify=False):
    """Compute all brackets up to level max_level. Returns sympy exprs."""
    simp = expand if fast_simplify else simplify_gen
    pair_keys = sorted(hams.keys())
    all_exprs = []
    all_levels = []
    computed_pairs = set()
    n_pairs = len(pair_keys)

    for (i, j) in pair_keys:
        all_exprs.append(hams[(i, j)])
        all_levels.append(0)

    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            computed_pairs.add(frozenset({i, j}))

    if verbose:
        print(f"[{label}] Level 0: {n_pairs}")

    for level in range(1, max_level + 1):
        t0 = time()
        if level == 1:
            frontier = list(range(n_pairs))
        else:
            frontier = [i for i, lv in enumerate(all_levels) if lv == level - 1]
        n_existing = len(all_exprs)
        new_exprs = []

        if level == 1:
            for i in range(n_pairs):
                for j in range(i + 1, n_pairs):
                    e = simp(poisson_bracket_generic(
                        all_exprs[i], all_exprs[j], q_vars, p_vars))
                    new_exprs.append(e)
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
                    e = simp(poisson_bracket_generic(
                        all_exprs[i], all_exprs[j], q_vars, p_vars))
                    new_exprs.append(e)

        for e in new_exprs:
            all_exprs.append(e)
            all_levels.append(level)

        if verbose:
            nz = sum(1 for e in new_exprs if e == 0)
            print(f"[{label}] Level {level}: {len(new_exprs)} "
                  f"candidates ({nz} zero) [{time()-t0:.1f}s]")

    return all_exprs, all_levels


def sample_points(n_dims, n, seed=42, w_range=2.5, v_range=1.5,
                  avoid_singularity=False):
    """Sample. If avoid_singularity, ensure q_i differences are > 0.3."""
    rng = np.random.RandomState(seed)
    n_q = n_dims // 2
    while True:
        Z = np.zeros((n, n_dims))
        Z[:, :n_q] = rng.uniform(-w_range, w_range, (n, n_q))
        Z[:, n_q:] = rng.uniform(-v_range, v_range, (n, n_q))
        if not avoid_singularity:
            return Z
        # Check all q_i - q_j > 0.3
        ok = True
        for i in range(n_q):
            for j in range(i+1, n_q):
                d = np.abs(Z[:, i] - Z[:, j])
                if np.any(d < 0.3):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return Z


def main():
    print("=" * 70)
    print("PHYSICAL vs NEURAL ALGEBRA: DIRECT COMPARISON IN 6D PHASE SPACE")
    print("=" * 70)

    # Shared phase space coordinates
    q1, q2, q3 = symbols("q1 q2 q3", real=True)
    p1, p2, p3 = symbols("p1 p2 p3", real=True)
    q_vars = [q1, q2, q3]
    p_vars = [p1, p2, p3]
    all_vars = q_vars + p_vars

    # --- Physical algebra: polynomial distance potential V = (q_i - q_j)^4 ---
    # r^n with n >= 4 is in the universal [3,6,17,116] class
    print("\n--- PHYSICAL: V = (q_i - q_j)^4 (polynomial distance, universality) ---")

    def Vpoly(a, b):
        return (a - b)**4 / 2

    phys_hams = build_hamiltonians(Vpoly, q_vars, p_vars, n=3)
    phys_exprs, phys_levels = compute_level3(
        phys_hams, q_vars, p_vars, label="phys-poly5", max_level=3)

    # --- Neural gradient-product algebra ---
    print("\n--- NEURAL: gradient-product, 3-layer linear network ---")
    w1, w2, w3 = symbols("w1 w2 w3", real=True)
    v1, v2, v3 = symbols("v1 v2 v3", real=True)
    nn_qs = [w1, w2, w3]
    nn_ps = [v1, v2, v3]
    nn_prod = w1 * w2 * w3
    nn_L = (nn_prod - 1)**2 / 2
    nn_grads = [diff(nn_L, w) for w in nn_qs]

    nn_hams = {}
    for i, j in combinations(range(3), 2):
        T = (nn_ps[i]**2 + nn_ps[j]**2) / 2
        V = nn_grads[i] * nn_grads[j] / 2
        nn_hams[(i, j)] = expand(T + V)

    # Relabel neural variables to same symbols as physics
    subst = {w1: q1, w2: q2, w3: q3, v1: p1, v2: p2, v3: p3}
    nn_hams = {k: v.xreplace(subst) for k, v in nn_hams.items()}

    nn_exprs, nn_levels = compute_level3(
        nn_hams, q_vars, p_vars, label="neural-grad", max_level=3)

    # --- Evaluate both on common phase-space points ---
    print("\n--- SVD analysis on shared phase-space points ---")
    nz_phys = [e for e in phys_exprs if e != 0]
    nz_neural = [e for e in nn_exprs if e != 0]
    print(f"Physical: {len(phys_exprs)} total, {len(nz_phys)} non-zero")
    print(f"Neural:   {len(nn_exprs)} total, {len(nz_neural)} non-zero")

    Z = sample_points(6, 1200, seed=42, w_range=2.5, v_range=1.5,
                      avoid_singularity=False)

    eval_phys = lambdify_exprs(nz_phys, all_vars)
    eval_neural = lambdify_exprs(nz_neural, all_vars)

    M_phys = eval_phys(Z)
    M_neural = eval_neural(Z)

    rp, _ = svd_gap_analysis(M_phys, label="physical")
    rn, _ = svd_gap_analysis(M_neural, label="neural")
    print(f"\nrank(physical)         = {rp}")
    print(f"rank(neural)           = {rn}")

    M_combined = np.hstack([M_phys, M_neural])
    rc, _ = svd_gap_analysis(M_combined, label="combined")
    print(f"rank([phys | neural])  = {rc}")
    print(f"  neural - phys overlap = {rp + rn - rc} dimensions")
    print(f"  pure neural (not in phys) = {rc - rp}")
    print(f"  pure phys (not in neural) = {rc - rn}")

    # --- Now identify which neural generators span the "extra" directions ---
    # Project neural columns onto orthogonal complement of physical column space
    print(f"\n--- Identifying the extra directions ---")

    # Normalize columns first
    def normalize(M):
        norms = np.linalg.norm(M, axis=0)
        norms[norms < 1e-15] = 1.0
        return M / norms, norms

    Mp, _ = normalize(M_phys)
    Mn_n, _ = normalize(M_neural)

    # Get orthonormal basis of physical column space
    Up, sp_vals, _ = np.linalg.svd(Mp, full_matrices=False)
    # Physics rank
    phys_rank = rp
    Up_basis = Up[:, :phys_rank]

    # Project neural columns onto orthogonal complement
    projections = Up_basis.T @ Mn_n
    residuals = Mn_n - Up_basis @ projections
    residual_norms = np.linalg.norm(residuals, axis=0)

    # Sort neural generators by residual (those with largest residual
    # are "most extra")
    order = np.argsort(-residual_norms)
    print(f"\nTop 15 neural generators by orthogonal residual (most 'extra'):")
    for k, idx in enumerate(order[:15]):
        print(f"  [{k:2d}] residual={residual_norms[idx]:.4e}  "
              f"neural_idx={idx}  level={nn_levels[[i for i,e in enumerate(nn_exprs) if e != 0][idx]]}")

    # Which 3-dimensional subspace is "extra"?
    # rank of combined = rp + (rc - rp) extras. The extras = rc - rp.
    n_extra = rc - rp
    print(f"\n{'='*60}")
    print(f"NUMBER OF EXTRA NEURAL DIRECTIONS = {n_extra}")
    print(f"{'='*60}")

    if n_extra > 0:
        Ur, sr, Vr = np.linalg.svd(residuals, full_matrices=False)
        print(f"\nTop {min(10, len(sr))} singular values of residual matrix:")
        for k in range(min(10, len(sr))):
            print(f"  sr[{k}] = {sr[k]:.4e}")

        sig_threshold = 1e-6 * sr[0]
        n_sig = int(np.sum(sr > sig_threshold))
        print(f"Significant residual directions: {n_sig}")

    print(f"\n--- Degree-by-level analysis ---")
    for label, exprs, levels_list in [
        ("PHYSICAL (q_i-q_j)^4", phys_exprs, phys_levels),
        ("NEURAL gradient", nn_exprs, nn_levels),
    ]:
        print(f"\n{label}:")
        by_level_deg = {}
        for e, lv in zip(exprs, levels_list):
            if e == 0:
                continue
            poly = sp.Poly(e, *all_vars)
            d = poly.total_degree()
            key = (lv, d)
            by_level_deg[key] = by_level_deg.get(key, 0) + 1
        for (lv, d), cnt in sorted(by_level_deg.items()):
            print(f"  Level {lv}, Degree {d:3d}: {cnt} generators")

    print(f"\n--- Rank by level+degree ceiling ---")
    for label, exprs, levels_list in [
        ("PHYSICAL", phys_exprs, phys_levels),
        ("NEURAL", nn_exprs, nn_levels),
    ]:
        nz_idx = [i for i, e in enumerate(exprs) if e != 0]
        nz_e = [exprs[i] for i in nz_idx]
        nz_l = [levels_list[i] for i in nz_idx]
        nz_d = [sp.Poly(e, *all_vars).total_degree() for e in nz_e]
        evaluate = lambdify_exprs(nz_e, all_vars)
        M = evaluate(Z)
        all_degs = sorted(set(nz_d))
        print(f"\n{label}:")
        for lv in range(4):
            for d in all_degs:
                mask = [i for i in range(len(nz_e))
                        if nz_l[i] <= lv and nz_d[i] <= d]
                if not mask:
                    continue
                sub = M[:, mask]
                rank_lv, _ = svd_gap_analysis(sub, label=f"{label}-L{lv}d{d}")
                print(f"  L<={lv} deg<={d:3d}: {len(mask)} cols, rank={rank_lv}")


if __name__ == "__main__":
    main()
