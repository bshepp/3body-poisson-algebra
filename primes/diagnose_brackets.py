#!/usr/bin/env python3
"""Diagnose bracket computation: compare symbolic vs numerical brackets.

Tests whether the precomputed-derivative approach for numerical Poisson
brackets matches the known symbolic brackets from compute_growth.
"""

import sys
import os
import pickle
import numpy as np
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nbody'))
from exact_growth_nbody import NBodyAlgebra

# Configuration — must match hilbert_polya_search.py
N_BODIES = 3
D_SPATIAL = 1
POTENTIAL = "log"
EXTERNAL_POTENTIAL = {"omega": 1.0}
N_POINTS = 2000
SEED = 42
CKPT_DIR = os.path.join(os.path.dirname(__file__), "results", "hp_search")


def main():
    print("=" * 70)
    print("BRACKET COMPUTATION DIAGNOSTIC")
    print("=" * 70)

    # Create algebra
    algebra = NBodyAlgebra(
        n_bodies=N_BODIES,
        d_spatial=D_SPATIAL,
        potential=POTENTIAL,
        checkpoint_dir=CKPT_DIR,
        external_potential=EXTERNAL_POTENTIAL,
    )

    # Load checkpoint
    ckpt_file = os.path.join(CKPT_DIR, "level_3.pkl")
    print(f"\nLoading checkpoint from {ckpt_file}...")
    with open(ckpt_file, 'rb') as f:
        data = pickle.load(f)

    all_exprs = data['exprs']
    all_names = data['names']
    all_levels = data['levels']

    print(f"  Total generators: {len(all_exprs)}")
    for lv in range(4):
        count = sum(1 for l in all_levels if l == lv)
        print(f"  Level {lv}: {count} generators")

    # Show first few generators
    for i in range(min(12, len(all_exprs))):
        print(f"  [{i}] L{all_levels[i]}: {all_names[i]}")

    # Sample phase space
    print(f"\nSampling {N_POINTS} phase space points...")
    Z_qp, Z_u = algebra.sample_phase_space(N_POINTS, seed=SEED)
    print(f"  Z_qp shape: {Z_qp.shape}, Z_u shape: {Z_u.shape}")

    # -------------------------------------------------------------------
    # Test 1: Verify known symbolic brackets against numerical brackets
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 1: Symbolic vs Numerical bracket comparison")
    print("=" * 70)

    # Pick a few pairs whose brackets are known (level-1 generators)
    # {H12, H13} should be stored as a level-1 generator
    test_pairs = [(0, 1), (0, 2), (1, 2)]  # L0 pairs

    # Evaluate all generators at sample points (for reference)
    print("  Evaluating all generators at sample points...")
    evaluate_all = algebra.lambdify_generators(all_exprs[:12])
    M_all = evaluate_all(Z_qp, Z_u)
    print(f"    First 12 generators evaluated: {M_all.shape}")

    # Precompute derivatives for level-0 generators
    level0_exprs = all_exprs[:3]  # H12, H13, H23
    level0_names = all_names[:3]
    print(f"\n  Precomputing derivatives for {len(level0_exprs)} L0 generators...")
    derivs_L0 = algebra.precompute_derivatives(level0_exprs, names=level0_names)

    # Lambdify and evaluate derivs
    n_q = len(algebra.q_vars)
    all_deriv_exprs = []
    for k in range(len(level0_exprs)):
        all_deriv_exprs.extend(derivs_L0[k]["dq"])
        all_deriv_exprs.extend(derivs_L0[k]["dp"])

    print(f"  Lambdifying {len(all_deriv_exprs)} derivative expressions...")
    eval_derivs = algebra.lambdify_generators(all_deriv_exprs)
    M_derivs = eval_derivs(Z_qp, Z_u)

    # Parse derivative values
    dq_vals = {}
    dp_vals = {}
    for k in range(len(level0_exprs)):
        dq_vals[k] = []
        dp_vals[k] = []
        for i in range(n_q):
            dq_vals[k].append(M_derivs[:, k * 2 * n_q + i])
        for i in range(n_q):
            dp_vals[k].append(M_derivs[:, k * 2 * n_q + n_q + i])

    for a, b in test_pairs:
        name_a, name_b = all_names[a], all_names[b]
        print(f"\n  Bracket {{{name_a}, {name_b}}}:")

        # Numerical bracket from derivatives
        bracket_num = np.zeros(N_POINTS)
        for i in range(n_q):
            bracket_num += dq_vals[a][i] * dp_vals[b][i]
            bracket_num -= dp_vals[a][i] * dq_vals[b][i]

        # Find the corresponding symbolic bracket in the checkpoint
        bracket_name = f"{{{name_a},{name_b}}}"
        bracket_idx = None
        for idx, name in enumerate(all_names):
            if name == bracket_name:
                bracket_idx = idx
                break

        if bracket_idx is not None:
            # Evaluate the symbolic bracket expression
            symbolic_vals = M_all[:, bracket_idx] if bracket_idx < 12 else None
            if symbolic_vals is None:
                eval_single = algebra.lambdify_generators([all_exprs[bracket_idx]])
                symbolic_vals = eval_single(Z_qp, Z_u).ravel()

            # Compare
            diff = np.abs(bracket_num - symbolic_vals)
            rel_diff = diff / (np.abs(symbolic_vals) + 1e-30)

            print(f"    Symbolic bracket idx: {bracket_idx} (L{all_levels[bracket_idx]})")
            print(f"    Numerical bracket range: [{bracket_num.min():.6e}, {bracket_num.max():.6e}]")
            print(f"    Symbolic bracket range:  [{symbolic_vals.min():.6e}, {symbolic_vals.max():.6e}]")
            print(f"    Max abs difference:  {diff.max():.6e}")
            print(f"    Mean abs difference: {diff.mean():.6e}")
            print(f"    Max rel difference:  {rel_diff.max():.6e}")
            print(f"    Mean rel difference: {rel_diff.mean():.6e}")

            if diff.max() < 1e-8:
                print(f"    PASS: Numerical matches symbolic")
            else:
                print(f"    FAIL: Numerical does NOT match symbolic!")
        else:
            print(f"    (Symbolic bracket not found by name '{bracket_name}')")
            print(f"    Numerical bracket range: [{bracket_num.min():.6e}, {bracket_num.max():.6e}]")

    # -------------------------------------------------------------------
    # Test 2: Compute symbolic bracket directly and compare
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 2: Direct symbolic Poisson bracket vs numerical")
    print("=" * 70)

    # Compute {H12, H13} symbolically from scratch
    print("  Computing {H12, H13} symbolically...")
    t0 = time()
    bracket_sym = algebra.poisson_bracket(all_exprs[0], all_exprs[1])
    bracket_sym_simplified = algebra.simplify_generator(bracket_sym)
    print(f"    Symbolic bracket: {len(bracket_sym_simplified.as_ordered_terms())} terms [{time()-t0:.1f}s]")

    # Evaluate symbolic bracket at sample points
    eval_bracket = algebra.lambdify_generators([bracket_sym_simplified])
    bracket_sym_vals = eval_bracket(Z_qp, Z_u).ravel()

    # Also get numerical from derivs
    bracket_num_01 = np.zeros(N_POINTS)
    for i in range(n_q):
        bracket_num_01 += dq_vals[0][i] * dp_vals[1][i]
        bracket_num_01 -= dp_vals[0][i] * dq_vals[1][i]

    diff = np.abs(bracket_num_01 - bracket_sym_vals)
    rel_diff = diff / (np.abs(bracket_sym_vals) + 1e-30)
    print(f"    Numerical range:  [{bracket_num_01.min():.6e}, {bracket_num_01.max():.6e}]")
    print(f"    Symbolic range:   [{bracket_sym_vals.min():.6e}, {bracket_sym_vals.max():.6e}]")
    print(f"    Max abs diff:     {diff.max():.6e}")
    print(f"    Mean abs diff:    {diff.mean():.6e}")
    print(f"    Max rel diff:     {rel_diff.max():.6e}")

    if diff.max() < 1e-8:
        print(f"    PASS")
    else:
        print(f"    FAIL")

    # -------------------------------------------------------------------
    # Test 3: Check individual derivative components
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 3: Individual derivative verification")
    print("=" * 70)

    import sympy as sp
    H12 = all_exprs[0]
    
    print(f"  H12 = {H12}")
    print(f"  q_vars = {algebra.q_vars}")
    print(f"  p_vars = {algebra.p_vars}")
    print(f"  u_vars = {algebra.u_vars}")

    for k, qvar in enumerate(algebra.q_vars):
        sym_deriv = sp.expand(algebra.total_deriv(H12, qvar))
        precomp_deriv = derivs_L0[0]["dq"][k]
        diff_expr = sp.expand(sym_deriv - precomp_deriv)
        
        print(f"\n  dH12/d{qvar}:")
        print(f"    Symbolic:     {sym_deriv}")
        print(f"    Precomputed:  {precomp_deriv}")
        print(f"    Difference:   {diff_expr}")

    for k, pvar in enumerate(algebra.p_vars):
        sym_deriv = sp.diff(H12, pvar)
        precomp_deriv = derivs_L0[0]["dp"][k]
        diff_expr = sp.expand(sym_deriv - precomp_deriv)
        
        print(f"\n  dH12/d{pvar}:")
        print(f"    Symbolic:     {sym_deriv}")
        print(f"    Precomputed:  {precomp_deriv}")
        print(f"    Difference:   {diff_expr}")

    # -------------------------------------------------------------------
    # Test 4: Closure check — SVD of augmented matrix
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 4: Numerical closure check")
    print("=" * 70)

    # Load basis indices
    basis_file = os.path.join(CKPT_DIR, "basis_indices.npy")
    if os.path.exists(basis_file):
        basis_indices = np.load(basis_file)
        r = len(basis_indices)
        print(f"  Loaded {r} basis indices")

        # Evaluate all 156 generators
        print(f"  Evaluating all {len(all_exprs)} generators...")
        eval_full = algebra.lambdify_generators(all_exprs)
        M_full = eval_full(Z_qp, Z_u)

        # Check original rank
        norms = np.linalg.norm(M_full, axis=0)
        norms[norms < 1e-15] = 1.0
        M_norm = M_full / norms
        U, s, Vt = np.linalg.svd(M_norm, full_matrices=False)
        
        noise_floor = 1e-8 * s[0]
        orig_rank = int(np.sum(s > noise_floor))
        
        if orig_rank < len(s):
            gap_ratio = s[orig_rank - 1] / s[orig_rank]
        else:
            gap_ratio = float('inf')
        
        print(f"  Original rank: {orig_rank} (from {M_full.shape[1]} generators)")
        print(f"  Gap ratio at rank {orig_rank}: {gap_ratio:.2e}")

        # Now compute a SAMPLE of numerical brackets and check augmented rank
        M_basis = M_full[:, basis_indices]

        # Precompute derivs for a small subset to test closure
        n_test = min(20, r)  # Test first 20 basis elements
        test_exprs = [all_exprs[basis_indices[k]] for k in range(n_test)]
        test_names = [all_names[basis_indices[k]] for k in range(n_test)]

        print(f"\n  Precomputing derivatives for {n_test} test basis elements...")
        test_derivs = algebra.precompute_derivatives(test_exprs, names=test_names)

        # Lambdify test derivatives
        test_deriv_exprs = []
        for k in range(n_test):
            test_deriv_exprs.extend(test_derivs[k]["dq"])
            test_deriv_exprs.extend(test_derivs[k]["dp"])

        print(f"  Lambdifying {len(test_deriv_exprs)} test derivative expressions...")
        eval_test_derivs = algebra.lambdify_generators(test_deriv_exprs)
        M_test_derivs = eval_test_derivs(Z_qp, Z_u)

        # Parse derivatives
        test_dq = []
        test_dp = []
        for k in range(n_test):
            dq_k = []
            dp_k = []
            for i in range(n_q):
                dq_k.append(M_test_derivs[:, k * 2 * n_q + i])
            for i in range(n_q):
                dp_k.append(M_test_derivs[:, k * 2 * n_q + n_q + i])
            test_dq.append(dq_k)
            test_dp.append(dp_k)

        # Compute brackets and check if they're in the span
        bracket_cols = []
        n_bad = 0
        n_total = 0
        for a in range(n_test):
            for b in range(a + 1, n_test):
                bracket_vals = np.zeros(N_POINTS)
                for i in range(n_q):
                    bracket_vals += test_dq[a][i] * test_dp[b][i]
                    bracket_vals -= test_dp[a][i] * test_dq[b][i]

                # Check residual against full basis
                c, _, _, _ = np.linalg.lstsq(M_basis, bracket_vals, rcond=None)
                fitted = M_basis @ c
                resid_norm = np.linalg.norm(bracket_vals - fitted)
                bracket_norm = np.linalg.norm(bracket_vals)
                rel_resid = resid_norm / (bracket_norm + 1e-30)

                n_total += 1
                if rel_resid > 1e-6:
                    n_bad += 1
                    if n_bad <= 10:
                        print(f"    Bad bracket ({a},{b}): "
                              f"level ({all_levels[basis_indices[a]]},{all_levels[basis_indices[b]]}), "
                              f"rel_resid={rel_resid:.2e}")
                    bracket_cols.append(bracket_vals)

        print(f"\n  Closure test: {n_bad}/{n_total} brackets outside span")

        if bracket_cols:
            # Augmented SVD
            M_aug = np.column_stack([M_basis] + bracket_cols[:100])
            norms_aug = np.linalg.norm(M_aug, axis=0)
            norms_aug[norms_aug < 1e-15] = 1.0
            M_aug_norm = M_aug / norms_aug
            U_aug, s_aug, Vt_aug = np.linalg.svd(M_aug_norm, full_matrices=False)
            
            noise_aug = 1e-8 * s_aug[0]
            aug_rank = int(np.sum(s_aug > noise_aug))
            
            print(f"\n  Augmented matrix: {M_aug.shape}")
            print(f"  Augmented rank: {aug_rank}")
            print(f"  Original rank:  {orig_rank}")
            if aug_rank > orig_rank:
                print(f"  ALGEBRA NOT CLOSED: grew from {orig_rank} to {aug_rank}")
            else:
                print(f"  Rank unchanged -> brackets ARE in span (lstsq issue?)")
                print(f"  Condition number of M_basis: {np.linalg.cond(M_basis):.2e}")
        else:
            print(f"  All brackets in span - algebra appears CLOSED at {orig_rank}")
    else:
        print(f"  No basis_indices.npy found - skipping closure check")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
