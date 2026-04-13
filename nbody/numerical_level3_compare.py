#!/usr/bin/env python3
"""Numerical comparison of level-3 structure constants between potentials.

Much faster than full symbolic computation:
  1. Load level_3.pkl checkpoints
  2. Symbolically compute partial derivatives (for Poisson bracket)
  3. Lambdify everything
  4. Evaluate numerically at many random phase points
  5. SVD for 116-dim basis identification
  6. Compute all 6670 brackets numerically at each point
  7. Express brackets in basis via least-squares
  8. Compare C[i,j,k] tensors

Usage:
  python numerical_level3_compare.py
"""

import os
import sys
import pickle
import numpy as np
from time import time

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sympy as sp
from sympy import diff, expand, Symbol, Integer, cancel, lambdify
from exact_growth_nbody import NBodyAlgebra


def load_checkpoint(path):
    """Load old-format level_N.pkl checkpoint."""
    print(f"  Loading {path}...", end=" ", flush=True)
    t0 = time()
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"[{time()-t0:.1f}s] {len(data['exprs'])} generators")
    return data


def compute_all_derivatives(algebra, exprs, names):
    """Compute all partial derivatives needed for Poisson brackets.

    Returns (dq_list, dp_list) where:
      dq_list[i][k] = total_deriv(exprs[i], q_vars[k])  (includes chain rule)
      dp_list[i][k] = diff(exprs[i], p_vars[k])          (simple partial)

    Note: We do NOT call expand() — lambdify handles un-expanded expressions.
    This is significantly faster for large level-3 expressions.
    """
    n = len(exprs)
    n_q = len(algebra.q_vars)
    dq_list = []
    dp_list = []

    print(f"  Computing derivatives for {n} generators "
          f"({2*n_q} derivs each)...")
    t0 = time()

    for idx, expr in enumerate(exprs):
        dq = []
        for q in algebra.q_vars:
            d = algebra.total_deriv(expr, q)
            dq.append(d)

        dp = []
        for p in algebra.p_vars:
            d = diff(expr, p)
            dp.append(d)

        dq_list.append(dq)
        dp_list.append(dp)

        if (idx + 1) % 10 == 0 or idx == n - 1:
            elapsed = time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n - idx - 1) / rate if rate > 0 else 0
            print(f"    {idx+1}/{n}  [{elapsed:.1f}s, "
                  f"ETA {eta/60:.1f}min]  {names[idx][:40]}",
                  flush=True)

    print(f"  Derivatives done in {time()-t0:.1f}s")
    return dq_list, dp_list


def lambdify_all(algebra, exprs, dq_list, dp_list):
    """Lambdify generators and their derivatives.

    Returns (gen_fns, dq_fns, dp_fns) where each is a list of callables.
    """
    all_vars = list(algebra.all_vars)
    n = len(exprs)

    print(f"  Lambdifying {n} generators + {n*2*len(algebra.q_vars)} "
          f"derivatives...", end=" ", flush=True)
    t0 = time()

    gen_fns = [lambdify(all_vars, expr, 'numpy') for expr in exprs]

    dq_fns = []
    for i in range(n):
        dq_fns.append([lambdify(all_vars, d, 'numpy')
                       for d in dq_list[i]])

    dp_fns = []
    for i in range(n):
        dp_fns.append([lambdify(all_vars, d, 'numpy')
                       for d in dp_list[i]])

    print(f"[{time()-t0:.1f}s]")
    return gen_fns, dq_fns, dp_fns


def evaluate_at_points(algebra, gen_fns, dq_fns, dp_fns, n_points=300,
                       seed=42):
    """Evaluate generators and derivatives at random phase space points.

    Returns:
      V: (n_points, n_gen) - generator values
      DQ: (n_gen, n_q, n_points) - q-derivative values
      DP: (n_gen, n_p, n_points) - p-derivative values
    """
    rng = np.random.default_rng(seed)
    n_gen = len(gen_fns)
    n_vars = len(algebra.all_vars)
    n_q = len(algebra.q_vars)

    # Sample phase space points
    # Use moderate range to avoid numerical issues with u_ij = 1/r_ij
    points = rng.uniform(0.5, 2.0, size=(n_points, n_vars))
    # Pass as individual arrays for vectorized lambdify
    pt_args = [points[:, i] for i in range(n_vars)]

    print(f"  Evaluating at {n_points} points (vectorized)...", flush=True)
    t0 = time()

    V = np.zeros((n_points, n_gen))
    DQ = np.zeros((n_gen, n_q, n_points))
    DP = np.zeros((n_gen, n_q, n_points))

    for gen_idx in range(n_gen):
        try:
            V[:, gen_idx] = gen_fns[gen_idx](*pt_args)
        except Exception:
            # Fallback to point-by-point if vectorized fails
            for pt_idx in range(n_points):
                args = tuple(points[pt_idx])
                V[pt_idx, gen_idx] = gen_fns[gen_idx](*args)

        for k in range(n_q):
            try:
                DQ[gen_idx, k, :] = dq_fns[gen_idx][k](*pt_args)
            except Exception:
                for pt_idx in range(n_points):
                    args = tuple(points[pt_idx])
                    DQ[gen_idx, k, pt_idx] = dq_fns[gen_idx][k](*args)
            try:
                DP[gen_idx, k, :] = dp_fns[gen_idx][k](*pt_args)
            except Exception:
                for pt_idx in range(n_points):
                    args = tuple(points[pt_idx])
                    DP[gen_idx, k, pt_idx] = dp_fns[gen_idx][k](*args)

        if (gen_idx + 1) % 20 == 0 or gen_idx == n_gen - 1:
            print(f"    {gen_idx+1}/{n_gen} generators evaluated "
                  f"[{time()-t0:.1f}s]", flush=True)

    print(f"  Evaluation done in {time()-t0:.1f}s")
    return V, DQ, DP, points


def select_numerical_basis(V, target_rank=116):
    """Select linearly independent generators via column-pivoted QR.

    Returns basis_indices (list of column indices).
    """
    from scipy.linalg import qr
    Q, R, P = qr(V, pivoting=True)

    # Find numerical rank
    diag = np.abs(np.diag(R))
    if len(diag) > target_rank:
        gap = diag[target_rank - 1] / diag[target_rank]
        print(f"  SVD gap at rank {target_rank}: {gap:.2e}")

    basis_indices = sorted(P[:target_rank].tolist())
    print(f"  Selected {len(basis_indices)} basis generators")
    return basis_indices


def compute_numerical_structure_constants(V, DQ, DP, basis_indices):
    """Compute C[i,j,k] numerically.

    {e_a, e_b} = sum_k C[a,b,k] e_k

    where e_a, e_b are basis generators, and the bracket is evaluated
    pointwise using precomputed derivatives.
    """
    r = len(basis_indices)
    n_points = V.shape[0]
    n_q = DQ.shape[1]

    print(f"\n  Computing {r*(r-1)//2} structure constants...")
    t0 = time()

    # Basis matrix: (n_points, r)
    B = V[:, basis_indices]

    # Precompute pseudo-inverse for least squares: (r, n_points)
    # C_coeffs = pinv @ bracket_values
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    # B = U @ diag(s) @ Vt
    # pinv(B) = Vt.T @ diag(1/s) @ U.T
    s_inv = 1.0 / s
    pinv_B = (Vt.T * s_inv[np.newaxis, :]) @ U.T  # (r, n_points)

    print(f"  Condition number of basis matrix: {s[0]/s[-1]:.2e}")

    C = np.zeros((r, r, r))
    max_residual = 0.0

    n_computed = 0
    for a_idx in range(r):
        a = basis_indices[a_idx]
        for b_idx in range(a_idx + 1, r):
            b = basis_indices[b_idx]
            n_computed += 1

            # Compute {gen_a, gen_b} at each point:
            # {f,g} = sum_k (df/dq_k * dg/dp_k - df/dp_k * dg/dq_k)
            bracket_vals = np.zeros(n_points)
            for k in range(n_q):
                bracket_vals += DQ[a, k, :] * DP[b, k, :]
                bracket_vals -= DP[a, k, :] * DQ[b, k, :]

            # Express in basis: bracket = sum_k C[a_idx, b_idx, k] * e_k
            coeffs = pinv_B @ bracket_vals  # (r,)

            # Check residual
            residual = np.max(np.abs(B @ coeffs - bracket_vals))
            max_residual = max(max_residual, residual)

            C[a_idx, b_idx, :] = coeffs
            C[b_idx, a_idx, :] = -coeffs

            if n_computed % 500 == 0:
                print(f"    [{n_computed}/{r*(r-1)//2}] "
                      f"max_residual={max_residual:.2e} [{time()-t0:.1f}s]")

    print(f"  Done: {n_computed} brackets in {time()-t0:.1f}s")
    print(f"  Max residual: {max_residual:.2e}")

    n_nonzero = np.count_nonzero(np.abs(C) > 1e-10)
    print(f"  Non-zero entries (|C|>1e-10): {n_nonzero} / {r**3}")

    return C, max_residual


def compare_numerical_tensors(C1, C2, label1, label2):
    """Compare two numerical structure constant tensors."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {label1} vs {label2}")
    print(f"{'='*70}")

    r = C1.shape[0]
    assert C2.shape[0] == r

    diff = C1 - C2
    abs_diff = np.abs(diff)

    # Normalize by the scale of the entries
    scale = max(np.max(np.abs(C1)), np.max(np.abs(C2)))
    rel_diff = abs_diff / scale if scale > 0 else abs_diff

    print(f"  Tensor shape: {r}x{r}x{r}")
    print(f"  Scale (max |C|): {scale:.6g}")
    print(f"  Max |diff|:      {abs_diff.max():.6e}")
    print(f"  Mean |diff|:     {abs_diff.mean():.6e}")
    print(f"  Max |rel diff|:  {rel_diff.max():.6e}")
    print(f"  Mean |rel diff|: {rel_diff.mean():.6e}")

    # Count entries that match to various precisions
    for tol in [1e-6, 1e-8, 1e-10, 1e-12]:
        n_match = np.sum(abs_diff < tol)
        pct = 100.0 * n_match / diff.size
        print(f"  |diff| < {tol:.0e}: {n_match}/{diff.size} ({pct:.1f}%)")

    # Check if all non-zero entries match
    mask_nz = (np.abs(C1) > 1e-10) | (np.abs(C2) > 1e-10)
    n_nz = np.sum(mask_nz)
    if n_nz > 0:
        nz_diff = abs_diff[mask_nz]
        nz_rel = rel_diff[mask_nz]
        print(f"\n  Non-zero entries: {n_nz}")
        print(f"  Max |diff| (non-zero): {nz_diff.max():.6e}")
        print(f"  Max |rel diff| (non-zero): {nz_rel.max():.6e}")

    # Try to identify the entries as rationals
    print(f"\n  Checking if entries are same rationals...")
    from fractions import Fraction
    n_rational_match = 0
    n_rational_mismatch = 0
    mismatches = []
    for a in range(r):
        for b in range(a+1, r):
            for k in range(r):
                v1 = C1[a, b, k]
                v2 = C2[a, b, k]
                if abs(v1) < 1e-12 and abs(v2) < 1e-12:
                    continue
                # Try to identify as fraction with small denominator
                f1 = Fraction(v1).limit_denominator(10000)
                f2 = Fraction(v2).limit_denominator(10000)
                if f1 == f2:
                    n_rational_match += 1
                else:
                    n_rational_mismatch += 1
                    if len(mismatches) < 5:
                        mismatches.append((a, b, k, v1, v2, f1, f2))

    print(f"  Rational matches: {n_rational_match}")
    print(f"  Rational mismatches: {n_rational_mismatch}")
    if mismatches:
        for a, b, k, v1, v2, f1, f2 in mismatches:
            print(f"    C[{a},{b},{k}]: {v1:.10f} -> {f1} vs "
                  f"{v2:.10f} -> {f2}")

    identical = (abs_diff.max() < 1e-8)
    return identical


def process_potential(potential, ckpt_path, deriv_cache_path):
    """Full numerical pipeline for one potential.

    Returns (V, DQ, DP, basis_indices, gen_fns) or loads from cache.
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {potential}")
    print(f"{'='*70}")
    t_total = time()

    data = load_checkpoint(ckpt_path)
    exprs = data['exprs']
    names = data['names']
    levels = data['levels']

    # Create algebra for bracket computation
    algebra = NBodyAlgebra(
        data['n_bodies'], data['d_spatial'], data['potential'],
        potential_params=data.get('potential_params'),
        charges=data.get('charges'))

    # Compute derivatives (with caching)
    if os.path.exists(deriv_cache_path):
        print(f"  Loading cached derivatives from {deriv_cache_path}...",
              end=" ", flush=True)
        t0 = time()
        with open(deriv_cache_path, 'rb') as f:
            cache = pickle.load(f)
        dq_list = cache['dq_list']
        dp_list = cache['dp_list']
        print(f"[{time()-t0:.1f}s]")
    else:
        dq_list, dp_list = compute_all_derivatives(algebra, exprs, names)
        os.makedirs(os.path.dirname(deriv_cache_path), exist_ok=True)
        with open(deriv_cache_path + '.tmp', 'wb') as f:
            pickle.dump({'dq_list': dq_list, 'dp_list': dp_list},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(deriv_cache_path + '.tmp', deriv_cache_path)
        print(f"  Cached derivatives to {deriv_cache_path}")

    # Lambdify
    gen_fns, dq_fns, dp_fns = lambdify_all(
        algebra, exprs, dq_list, dp_list)

    # Evaluate at random points
    V, DQ, DP, points = evaluate_at_points(
        algebra, gen_fns, dq_fns, dp_fns, n_points=300, seed=42)

    print(f"\n  Total processing time: {time()-t_total:.1f}s")
    return V, DQ, DP, algebra


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_1r = os.path.join(base, "aws_results", "nbody_checkpoints",
                           "checkpoints_N3_d2_1r", "level_3.pkl")
    ckpt_1r2 = os.path.join(base, "nbody", "checkpoints_N3_d2_1r2",
                            "level_3.pkl")
    work_dir = os.path.join(base, "results", "level3_numerical")

    for label, path in [("1/r", ckpt_1r), ("1/r²", ckpt_1r2)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} checkpoint not found: {path}")
            sys.exit(1)
        sz = os.path.getsize(path) / 1024 / 1024
        print(f"  {label}: {path} ({sz:.1f} MB)")

    os.makedirs(work_dir, exist_ok=True)

    # Process both potentials
    V1, DQ1, DP1, alg1 = process_potential(
        "1/r", ckpt_1r,
        os.path.join(work_dir, "derivs_1r.pkl"))

    V2, DQ2, DP2, alg2 = process_potential(
        "1/r²", ckpt_1r2,
        os.path.join(work_dir, "derivs_1r2.pkl"))

    # Select basis using SAME method for both
    # Use 1/r's generator evaluations for basis selection
    print(f"\n{'='*70}")
    print(f"BASIS SELECTION")
    print(f"{'='*70}")

    basis_indices_1 = select_numerical_basis(V1, target_rank=116)
    basis_indices_2 = select_numerical_basis(V2, target_rank=116)

    print(f"  1/r  basis (first 20): {basis_indices_1[:20]}")
    print(f"  1/r² basis (first 20): {basis_indices_2[:20]}")

    if basis_indices_1 == basis_indices_2:
        print(f"  Basis indices: IDENTICAL")
        basis_indices = basis_indices_1
    else:
        print(f"  Basis indices: DIFFERENT")
        diff_count = sum(1 for a, b in zip(basis_indices_1, basis_indices_2)
                         if a != b)
        print(f"  Differ at {diff_count} positions")
        # Use the same basis (from 1/r) for consistent comparison
        basis_indices = basis_indices_1
        print(f"  Using 1/r basis for both potentials")

    # Compute structure constants
    print(f"\n{'='*70}")
    print(f"STRUCTURE CONSTANTS: 1/r")
    print(f"{'='*70}")
    C1, res1 = compute_numerical_structure_constants(
        V1, DQ1, DP1, basis_indices)

    print(f"\n{'='*70}")
    print(f"STRUCTURE CONSTANTS: 1/r²")
    print(f"{'='*70}")
    C2, res2 = compute_numerical_structure_constants(
        V2, DQ2, DP2, basis_indices)

    # Compare
    identical = compare_numerical_tensors(C1, C2, "1/r", "1/r²")

    # Save results
    result = {
        'C_1r': C1,
        'C_1r2': C2,
        'basis_indices': basis_indices,
        'basis_indices_1r': basis_indices_1,
        'basis_indices_1r2': basis_indices_2,
        'max_residual_1r': res1,
        'max_residual_1r2': res2,
    }
    result_path = os.path.join(work_dir, "numerical_comparison.npz")
    np.savez_compressed(result_path, **result)
    print(f"\n  Saved results to {result_path}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    if identical:
        print(f"  Level-3 structure constants are NUMERICALLY IDENTICAL")
        print(f"  (max |diff| < 1e-8 across all {116**3:,} entries)")
        print(f"  Universality extends from level 2 (dim 17) to "
              f"level 3 (dim 116).")
    else:
        print(f"  Level-3 structure constants show DIFFERENCES")
        print(f"  Universality may be limited to level 2.")


if __name__ == "__main__":
    main()
