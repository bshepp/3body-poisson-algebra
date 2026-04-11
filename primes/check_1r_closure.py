#!/usr/bin/env python3
"""Quick check: does the 1/r potential algebra close at 116?

Computes derivatives and brackets for a small subset of basis elements
from the 1/r N=3 d=1 algebra to test closure.
"""

import sys
import os
import pickle
import numpy as np
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nbody'))
from exact_growth_nbody import NBodyAlgebra

CKPT_DIR = os.path.join(os.path.dirname(__file__), "results", "closure_1r")
N_POINTS = 2000
SEED = 42


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("=" * 70)
    print("CLOSURE CHECK: 1/r potential, N=3, d=1")
    print("=" * 70)

    algebra = NBodyAlgebra(
        n_bodies=3,
        d_spatial=1,
        potential="1/r",
        checkpoint_dir=CKPT_DIR,
    )

    # Build generators through level 3
    ckpt_file = os.path.join(CKPT_DIR, "level_3.pkl")
    if os.path.exists(ckpt_file):
        print(f"\nLoading checkpoint from {ckpt_file}...")
        with open(ckpt_file, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"\nComputing generators through level 3...")
        algebra.compute_growth(max_level=3, n_samples=500, resume=False)
        with open(ckpt_file, 'rb') as f:
            data = pickle.load(f)

    all_exprs = data['exprs']
    all_names = data['names']
    all_levels = data['levels']

    n_total = len(all_exprs)
    for lv in range(4):
        count = sum(1 for l in all_levels if l == lv)
        print(f"  Level {lv}: {count} generators")
    print(f"  Total: {n_total}")

    # Sample phase space
    Z_qp, Z_u = algebra.sample_phase_space(N_POINTS, seed=SEED)

    # Evaluate all generators
    print(f"\nEvaluating all {n_total} generators at {N_POINTS} points...")
    evaluate = algebra.lambdify_generators(all_exprs)
    M_all = evaluate(Z_qp, Z_u)

    # Find rank
    norms = np.linalg.norm(M_all, axis=0)
    norms[norms < 1e-15] = 1.0
    M_norm = M_all / norms
    U, s, Vt = np.linalg.svd(M_norm, full_matrices=False)
    noise = 1e-8 * s[0]
    rank = int(np.sum(s > noise))

    if rank < len(s):
        gap = s[rank - 1] / s[rank]
    else:
        gap = float('inf')

    print(f"  SVD rank: {rank}, gap ratio: {gap:.2e}")

    # Select basis via greedy column selection
    print(f"\nSelecting {rank} basis columns...")
    basis_indices = []
    remaining = list(range(n_total))
    M_basis = np.empty((N_POINTS, 0))

    for _ in range(rank):
        best_idx = -1
        best_norm = -1
        for idx in remaining:
            col = M_all[:, idx:idx + 1]
            if M_basis.shape[1] > 0:
                c, _, _, _ = np.linalg.lstsq(M_basis, col, rcond=None)
                residual = col - M_basis @ c
            else:
                residual = col
            norm = np.linalg.norm(residual)
            if norm > best_norm:
                best_norm = norm
                best_idx = idx
        basis_indices.append(best_idx)
        remaining.remove(best_idx)
        M_basis = np.column_stack([M_basis, M_all[:, best_idx]])

    basis_levels = [all_levels[i] for i in basis_indices]
    for lv in range(4):
        count = sum(1 for l in basis_levels if l == lv)
        print(f"  Basis level {lv}: {count} elements")

    # Test closure with a subset of level-3 basis elements
    n_test = min(20, rank)
    # Try to pick level-3 elements specifically
    l3_basis = [i for i, idx in enumerate(basis_indices)
                if all_levels[idx] == 3]
    if len(l3_basis) >= n_test:
        test_local = l3_basis[:n_test]
    else:
        test_local = list(range(n_test))

    test_exprs = [all_exprs[basis_indices[k]] for k in test_local]
    test_names = [all_names[basis_indices[k]] for k in test_local]

    print(f"\nPrecomputing derivatives for {len(test_local)} test elements "
          f"(levels: {[all_levels[basis_indices[k]] for k in test_local[:5]]}...)...")
    n_q = len(algebra.q_vars)
    test_derivs = algebra.precompute_derivatives(test_exprs, names=test_names)

    # Lambdify derivatives
    deriv_exprs = []
    for k in range(len(test_local)):
        deriv_exprs.extend(test_derivs[k]["dq"])
        deriv_exprs.extend(test_derivs[k]["dp"])

    print(f"Lambdifying {len(deriv_exprs)} derivative expressions...")
    eval_derivs = algebra.lambdify_generators(deriv_exprs)
    M_derivs = eval_derivs(Z_qp, Z_u)

    # Parse
    nt = len(test_local)
    test_dq = []
    test_dp = []
    for k in range(nt):
        dq_k = [M_derivs[:, k * 2 * n_q + i] for i in range(n_q)]
        dp_k = [M_derivs[:, k * 2 * n_q + n_q + i] for i in range(n_q)]
        test_dq.append(dq_k)
        test_dp.append(dp_k)

    # Compute brackets and check closure
    bracket_cols = []
    n_bad = 0
    n_total_brackets = 0
    for a in range(nt):
        for b in range(a + 1, nt):
            bracket_vals = np.zeros(N_POINTS)
            for i in range(n_q):
                bracket_vals += test_dq[a][i] * test_dp[b][i]
                bracket_vals -= test_dp[a][i] * test_dq[b][i]

            c, _, _, _ = np.linalg.lstsq(M_basis, bracket_vals, rcond=None)
            fitted = M_basis @ c
            resid = np.linalg.norm(bracket_vals - fitted) / (
                np.linalg.norm(bracket_vals) + 1e-30)

            n_total_brackets += 1
            if resid > 1e-6:
                n_bad += 1
                la = all_levels[basis_indices[test_local[a]]]
                lb = all_levels[basis_indices[test_local[b]]]
                if n_bad <= 10:
                    print(f"  Bad bracket ({a},{b}): level ({la},{lb}), "
                          f"rel_resid={resid:.2e}")
                bracket_cols.append(bracket_vals)

    print(f"\n  Closure test: {n_bad}/{n_total_brackets} brackets outside span")

    if bracket_cols:
        M_aug = np.column_stack([M_basis] + bracket_cols[:200])
        norms_aug = np.linalg.norm(M_aug, axis=0)
        norms_aug[norms_aug < 1e-15] = 1.0
        M_aug_norm = M_aug / norms_aug
        _, s_aug, _ = np.linalg.svd(M_aug_norm, full_matrices=False)
        noise_aug = 1e-8 * s_aug[0]
        aug_rank = int(np.sum(s_aug > noise_aug))
        print(f"  Augmented rank: {aug_rank} (original: {rank})")
        if aug_rank > rank:
            print(f"  ALGEBRA NOT CLOSED: grew from {rank} to {aug_rank}")
        else:
            print(f"  Rank unchanged -> algebra CLOSED at {rank}")
            print(f"  Condition number: {np.linalg.cond(M_basis):.2e}")
    else:
        print(f"  ALL brackets in span -> algebra CLOSED at {rank}")


if __name__ == "__main__":
    main()
