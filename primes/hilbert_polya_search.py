#!/usr/bin/env python3
"""Hilbert-Pólya operator search via GUE Lie algebra structure.

Computes the full algebraic structure (structure constants, Killing form,
center, derived/lower-central series) of the 116-dimensional GUE log-gas
Lie algebra NUMERICALLY, then analyzes center elements as Hilbert-Pólya
operator candidates.

The log potential produces transcendental expressions (log(u_ij)) that
prevent exact QQ polynomial decomposition, so we use numerical evaluation
at random phase-space points + least-squares for structure constants.
"""

import sys
import os
import json
import pickle
import numpy as np
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nbody'))
from exact_growth_nbody import NBodyAlgebra

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BODIES = 3
D_SPATIAL = 1
POTENTIAL = "log"
EXTERNAL_POTENTIAL = {"omega": 1.0}
MAX_LEVEL = 3
N_SAMPLE_POINTS = 2000  # overdetermined system for structure constants
SEED = 42
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__),
                              "results", "hp_search")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Phase 1: Build generators and select basis
# ---------------------------------------------------------------------------

def build_generators(algebra, max_level, ckpt_dir):
    """Build all generators through max_level with checkpointing.
    
    NBodyAlgebra saves cumulative checkpoints as level_N.pkl.
    The highest-level checkpoint contains ALL generators through that level.
    """
    # Check if final checkpoint already exists
    final_ckpt = os.path.join(ckpt_dir, f"level_{max_level}.pkl")
    if os.path.exists(final_ckpt):
        print(f"Loading generators from {final_ckpt}...")
        with open(final_ckpt, 'rb') as f:
            data = pickle.load(f)
        print(f"  Total generators: {len(data['exprs'])}")
        return data['exprs'], data['names'], data['levels']

    print(f"\nBuilding generators: N={N_BODIES}, d={D_SPATIAL}, "
          f"potential={POTENTIAL}, max_level={max_level}")

    result = algebra.compute_growth(
        max_level=max_level,
        n_samples=500,
        resume=True)

    # Load the final cumulative checkpoint
    with open(final_ckpt, 'rb') as f:
        data = pickle.load(f)

    print(f"  Total generators: {len(data['exprs'])}")
    return data['exprs'], data['names'], data['levels']


def select_numerical_basis(algebra, exprs, n_points, seed):
    """Select linearly independent basis via SVD at random points."""
    print(f"\nSelecting basis via SVD at {n_points} random points...")
    t0 = time()

    Z_qp, Z_u = algebra.sample_phase_space(n_points, seed=seed)
    evaluate = algebra.lambdify_generators(exprs)
    M = evaluate(Z_qp, Z_u)

    # Normalize columns
    norms = np.linalg.norm(M, axis=0)
    norms[norms < 1e-15] = 1.0
    M_norm = M / norms

    U, s, Vt = np.linalg.svd(M_norm, full_matrices=False)

    # Find rank via gap
    noise_threshold = 1e-8 * s[0]
    for i in range(len(s) - 1):
        if s[i+1] < noise_threshold:
            rank = i + 1
            break
    else:
        rank = int(np.sum(s > noise_threshold))

    print(f"  Numerical rank: {rank}")
    print(f"  Gap at rank {rank}: {s[rank-1]:.6e} / {s[rank] if rank < len(s) else 0:.6e}")

    # Greedy column selection for well-conditioned basis
    basis_indices = []
    remaining = list(range(len(exprs)))
    M_basis = np.empty((n_points, 0))

    for _ in range(rank):
        best_idx = -1
        best_norm = -1
        for idx in remaining:
            col = M[:, idx:idx+1]
            if M_basis.shape[1] > 0:
                # Project out existing basis
                coeffs = np.linalg.lstsq(M_basis, col, rcond=None)[0]
                residual = col - M_basis @ coeffs
            else:
                residual = col
            norm = np.linalg.norm(residual)
            if norm > best_norm:
                best_norm = norm
                best_idx = idx
        basis_indices.append(best_idx)
        remaining.remove(best_idx)
        M_basis = np.column_stack([M_basis, M[:, best_idx]])

    print(f"  Selected {len(basis_indices)} basis generators [{time()-t0:.1f}s]")
    return basis_indices, rank, M


def compute_numerical_structure_constants(algebra, exprs, names, levels,
                                          basis_indices, n_points, seed,
                                          ckpt_dir):
    """Compute structure constants numerically via precomputed derivatives.

    Strategy:
    1. Precompute symbolic derivatives of all 116 basis generators (one-time)
    2. Lambdify basis generators AND all their derivatives
    3. Evaluate everything at random sample points
    4. Compute brackets NUMERICALLY: {e_a, e_b} = sum_k(De_a/dq_k * De_b/dp_k - ...)
    5. Fit: bracket_vals = M_basis @ c via least squares

    This avoids per-bracket symbolic computation, making it O(r^2 * n_points)
    instead of O(r^2 * symbolic_cost).
    """
    ckpt_file = os.path.join(ckpt_dir, "structure_constants.pkl")
    if os.path.exists(ckpt_file):
        print(f"\nLoading structure constants from {ckpt_file}...")
        with open(ckpt_file, 'rb') as f:
            data = pickle.load(f)
        return data['C'], data['residuals']

    r = len(basis_indices)
    n_q = len(algebra.q_vars)
    total_brackets = r * (r - 1) // 2

    print(f"\n{'='*70}")
    print(f"NUMERICAL STRUCTURE CONSTANTS (rank {r})")
    print(f"{'='*70}")
    print(f"  Total brackets to compute: {total_brackets}")
    print(f"  Evaluating at {n_points} phase-space points")
    print(f"  Using precomputed-derivative approach (no per-bracket symbolic ops)")
    t_total = time()

    # Sample phase-space points
    Z_qp, Z_u = algebra.sample_phase_space(n_points, seed=seed)

    # Precompute symbolic derivatives for basis generators
    basis_exprs = [exprs[i] for i in basis_indices]
    basis_names = [names[i] for i in basis_indices]

    print(f"\n  Phase 1: Precomputing symbolic derivatives for {r} basis elements...")
    all_derivs = algebra.precompute_derivatives(basis_exprs, names=basis_names)

    # Collect all expressions to lambdify: basis + dq derivatives + dp derivatives
    # Layout: [b_0, ..., b_{r-1}, db0/dq0, db0/dq1, ..., db0/dqK,
    #          db1/dq0, ..., db_{r-1}/dqK, db0/dp0, ..., db_{r-1}/dpK]
    all_exprs_to_eval = list(basis_exprs)
    for k in range(r):
        all_exprs_to_eval.extend(all_derivs[k]["dq"])  # n_q per basis element
    for k in range(r):
        all_exprs_to_eval.extend(all_derivs[k]["dp"])  # n_q per basis element

    n_total_exprs = len(all_exprs_to_eval)
    print(f"\n  Phase 2: Lambdifying {n_total_exprs} expressions "
          f"({r} basis + {r*n_q} dq-derivs + {r*n_q} dp-derivs)...")
    t_lam = time()
    eval_all = algebra.lambdify_generators(all_exprs_to_eval)
    M_all = eval_all(Z_qp, Z_u)  # (n_points, n_total_exprs)
    print(f"    Lambdify + evaluate: {time()-t_lam:.1f}s")

    # Unpack the evaluation matrix
    M_basis = M_all[:, :r]  # (n_points, r)
    dq_start = r
    dp_start = r + r * n_q

    # dq_vals[k][i] = array of shape (n_points,) = ∂b_k/∂q_i evaluated
    dq_vals = []
    for k in range(r):
        derivs_k = []
        for i in range(n_q):
            derivs_k.append(M_all[:, dq_start + k * n_q + i])
        dq_vals.append(derivs_k)

    dp_vals = []
    for k in range(r):
        derivs_k = []
        for i in range(n_q):
            derivs_k.append(M_all[:, dp_start + k * n_q + i])
        dp_vals.append(derivs_k)

    # Phase 3: Compute all brackets numerically
    print(f"\n  Phase 3: Computing {total_brackets} brackets numerically...")
    t_brackets = time()

    C = np.zeros((r, r, r))
    residuals = np.zeros((r, r))

    n_computed = 0
    max_residual = 0.0
    warn_count = 0
    for a in range(r):
        for b in range(a + 1, r):
            # {e_a, e_b} = sum_i (de_a/dq_i * de_b/dp_i - de_a/dp_i * de_b/dq_i)
            bracket_vals = np.zeros(n_points)
            for i in range(n_q):
                bracket_vals += dq_vals[a][i] * dp_vals[b][i]
                bracket_vals -= dp_vals[a][i] * dq_vals[b][i]

            # Solve: bracket_vals ≈ M_basis @ c via least squares
            c, _, _, _ = np.linalg.lstsq(M_basis, bracket_vals, rcond=None)

            # Check residual
            fitted = M_basis @ c
            residual = np.linalg.norm(bracket_vals - fitted) \
                     / (np.linalg.norm(bracket_vals) + 1e-30)
            if residual > 1e-6:
                if warn_count < 20:
                    print(f"    WARNING: bracket ({a},{b}) residual = {residual:.2e}")
                warn_count += 1
            max_residual = max(max_residual, residual)

            C[a, b, :] = c
            C[b, a, :] = -c
            residuals[a, b] = residual
            residuals[b, a] = residual

            n_computed += 1
            if n_computed % 1000 == 0 or n_computed == total_brackets:
                print(f"    [{n_computed}/{total_brackets}] "
                      f"max_res={max_residual:.2e}  [{time()-t_brackets:.1f}s]")

    elapsed = time() - t_total
    print(f"\n  Structure constants computed in {elapsed:.1f}s")
    print(f"  Max residual: {max_residual:.2e}")
    if warn_count > 0:
        print(f"  WARNING: {warn_count} brackets with residual > 1e-6")
    print(f"  Mean residual: {np.mean(residuals[residuals > 0]):.2e}")

    n_nonzero = int(np.count_nonzero(np.abs(C) > 1e-10))
    print(f"  Non-zero entries (|C|>1e-10): {n_nonzero} / {r**3} "
          f"({100*n_nonzero/r**3:.1f}%)")

    # Save final
    with open(ckpt_file, 'wb') as f:
        pickle.dump({'C': C, 'residuals': residuals}, f)

    return C, residuals


# ---------------------------------------------------------------------------
# Phase 2: Algebraic structure analysis
# ---------------------------------------------------------------------------

def verify_antisymmetry(C, tol=1e-8):
    """Verify C[i,j,k] = -C[j,i,k]."""
    err = np.max(np.abs(C + C.transpose(1, 0, 2)))
    ok = err < tol
    print(f"  Antisymmetry: max |C[i,j,k]+C[j,i,k]| = {err:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def verify_jacobi(C, tol=1e-6):
    """Verify Jacobi identity: sum_l (C[i,j,l]*C[l,k,m] + cyclic) = 0."""
    r = C.shape[0]
    max_err = 0.0
    n_checks = 0
    # Sample random triples
    rng = np.random.RandomState(123)
    for _ in range(min(5000, r**3)):
        i, j, k = rng.randint(0, r, size=3)
        if i == j or j == k or i == k:
            continue
        for m in range(r):
            val = (np.dot(C[i, j, :], C[:, k, m]) +
                   np.dot(C[j, k, :], C[:, i, m]) +
                   np.dot(C[k, i, :], C[:, j, m]))
            max_err = max(max_err, abs(val))
        n_checks += 1
    ok = max_err < tol
    print(f"  Jacobi identity ({n_checks} triples): max error = {max_err:.2e} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def killing_form(C):
    """Compute Killing form K[i,j] = trace(ad_i @ ad_j)."""
    r = C.shape[0]
    K = np.zeros((r, r))
    for i in range(r):
        for j in range(i, r):
            val = np.trace(C[i] @ C[j].T)
            K[i, j] = val
            K[j, i] = val
    eigs = np.linalg.eigvalsh(K)
    eigs.sort()
    tol = 1e-10 * np.max(np.abs(eigs)) if len(eigs) > 0 else 1e-10
    n_pos = int(np.sum(eigs > tol))
    n_neg = int(np.sum(eigs < -tol))
    n_zero = int(np.sum(np.abs(eigs) <= tol))
    return K, eigs, (n_pos, n_neg, n_zero)


def derived_series(C, max_depth=10):
    """L^(0) = L, L^(n+1) = [L^(n), L^(n)]."""
    r = C.shape[0]
    current = np.eye(r)
    dims = [r]
    for _ in range(1, max_depth + 1):
        dim = current.shape[0]
        if dim == 0:
            break
        brackets = []
        for a in range(dim):
            for b in range(a + 1, dim):
                br = np.einsum('i,j,ijk->k', current[a], current[b], C)
                if np.linalg.norm(br) > 1e-12:
                    brackets.append(br)
        if not brackets:
            dims.append(0)
            return dims, True, len(dims) - 1
        M = np.array(brackets)
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10 * s[0]
        new_rank = int(np.sum(s > tol))
        if new_rank == 0:
            dims.append(0)
            return dims, True, len(dims) - 1
        current = Vt[:new_rank]
        dims.append(new_rank)
        if new_rank == dims[-2]:
            return dims, False, None
    return dims, False, None


def lower_central_series(C, max_depth=10):
    """L_0 = L, L_{n+1} = [L, L_n]."""
    r = C.shape[0]
    current = np.eye(r)
    dims = [r]
    for _ in range(1, max_depth + 1):
        dim = current.shape[0]
        if dim == 0:
            break
        brackets = []
        for a in range(r):
            e_a = np.zeros(r)
            e_a[a] = 1.0
            for b in range(dim):
                br = np.einsum('i,j,ijk->k', e_a, current[b], C)
                if np.linalg.norm(br) > 1e-12:
                    brackets.append(br)
        if not brackets:
            dims.append(0)
            return dims, True, len(dims) - 1
        M = np.array(brackets)
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10 * s[0]
        new_rank = int(np.sum(s > tol))
        if new_rank == 0:
            dims.append(0)
            return dims, True, len(dims) - 1
        current = Vt[:new_rank]
        dims.append(new_rank)
        if new_rank == dims[-2]:
            return dims, False, None
    return dims, False, None


def compute_center(C):
    """Center Z(L): elements z with [z, x] = 0 for all x."""
    r = C.shape[0]
    A = C.transpose(1, 2, 0).reshape(r * r, r)
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = 1e-10 * s[0] if len(s) > 0 and s[0] > 0 else 1e-10
    null_dim = int(np.sum(s < tol))
    center_basis = Vt[-null_dim:] if null_dim > 0 else np.empty((0, r))
    return center_basis, null_dim


def levi_decomposition(C):
    """Approximate Levi decomposition L = S ⊕ R (semisimple + radical).

    The radical is the maximal solvable ideal, detected via the Killing form
    null space.
    """
    K, eigs, sig = killing_form(C)
    r = C.shape[0]

    # Radical approximation: null space of Killing form
    _, s, Vt = np.linalg.svd(K)
    tol = 1e-8 * s[0] if s[0] > 0 else 1e-10
    null_dim = int(np.sum(s < tol))
    nonnull_dim = r - null_dim

    radical_basis = Vt[-null_dim:] if null_dim > 0 else np.empty((0, r))
    semisimple_basis = Vt[:nonnull_dim]

    return {
        'radical_dim': null_dim,
        'semisimple_dim': nonnull_dim,
        'radical_basis': radical_basis,
        'semisimple_basis': semisimple_basis,
        'killing_signature': sig,
    }


# ---------------------------------------------------------------------------
# Phase 3: Hilbert-Pólya candidate analysis
# ---------------------------------------------------------------------------

def analyze_center_elements(algebra, exprs, basis_indices, center_basis,
                            n_points=500, seed=99):
    """Analyze center elements for Hilbert-Pólya operator properties.

    Tests:
    1. Time-reversal symmetry: H(q, -p) vs H(q, p)
    2. Berry-Keating xp structure
    3. Spectral properties on phase-space grid
    """
    import sympy as sp

    r = len(basis_indices)
    center_dim = center_basis.shape[0]

    if center_dim == 0:
        print("  No center elements found — no HP candidates.")
        return {}

    print(f"\n{'='*70}")
    print(f"HILBERT-PÓLYA CANDIDATE ANALYSIS ({center_dim} center elements)")
    print(f"{'='*70}")

    results = []

    # Reconstruct symbolic center elements
    basis_exprs = [exprs[i] for i in basis_indices]

    for c_idx in range(center_dim):
        coeffs = center_basis[c_idx]
        # Build symbolic expression
        center_expr = sum(float(coeffs[k]) * basis_exprs[k]
                          for k in range(r) if abs(coeffs[k]) > 1e-12)

        if center_expr == 0:
            continue

        print(f"\n  --- Center element {c_idx + 1} ---")
        print(f"  Non-zero basis coefficients: "
              f"{np.sum(np.abs(coeffs) > 1e-10)}/{r}")

        # Which levels contribute?
        level_contrib = {}
        for k in range(r):
            if abs(coeffs[k]) > 1e-10:
                # Need to figure out what level basis_indices[k] is at
                # by looking at the names list
                pass

        # Test 1: Time-reversal symmetry
        # H is time-reversal even if H(q, -p) = H(q, p)
        # H is time-reversal odd if H(q, -p) = -H(q, p)
        p_vars = algebra.p_vars
        p_flip = {p: -p for p in p_vars}
        flipped = center_expr.xreplace(p_flip) if hasattr(center_expr, 'xreplace') else center_expr
        try:
            diff_even = sp.expand(center_expr - flipped)
            diff_odd = sp.expand(center_expr + flipped)
            is_even = diff_even == 0 or sp.simplify(diff_even) == 0
            is_odd = diff_odd == 0 or sp.simplify(diff_odd) == 0

            if is_even:
                tr_sym = "EVEN (T-symmetric)"
            elif is_odd:
                tr_sym = "ODD (T-antisymmetric)"
            else:
                # Numerical check
                Z_qp, Z_u = algebra.sample_phase_space(200, seed=seed + c_idx)
                eval_fn = algebra.lambdify_generators([center_expr])
                vals = eval_fn(Z_qp, Z_u).ravel()

                Z_qp_flip = Z_qp.copy()
                Z_qp_flip[:, algebra.N * algebra.d:] *= -1
                vals_flip = eval_fn(Z_qp_flip, Z_u).ravel()

                ratio_even = np.linalg.norm(vals - vals_flip) / (np.linalg.norm(vals) + 1e-30)
                ratio_odd = np.linalg.norm(vals + vals_flip) / (np.linalg.norm(vals) + 1e-30)

                if ratio_even < 1e-8:
                    tr_sym = f"EVEN (numerical, err={ratio_even:.2e})"
                elif ratio_odd < 1e-8:
                    tr_sym = f"ODD (numerical, err={ratio_odd:.2e})"
                else:
                    tr_sym = f"MIXED (even_err={ratio_even:.2e}, odd_err={ratio_odd:.2e})"
        except Exception as e:
            tr_sym = f"ERROR: {e}"

        print(f"  Time reversal: {tr_sym}")

        # Test 2: Berry-Keating structure
        # The BK operator is xp + px = 2xp - i*hbar
        # Check if center element has dominant xp-like terms
        # We test by checking the momentum-degree structure
        Z_qp, Z_u = algebra.sample_phase_space(500, seed=seed + c_idx + 100)
        eval_fn = algebra.lambdify_generators([center_expr])
        vals = eval_fn(Z_qp, Z_u).ravel()

        # Correlation with xp-type functions
        q_vals = Z_qp[:, :algebra.N * algebra.d]
        p_vals = Z_qp[:, algebra.N * algebra.d:]
        xp_vals = np.sum(q_vals * p_vals, axis=1)  # sum_i q_i * p_i

        if np.std(vals) > 1e-12 and np.std(xp_vals) > 1e-12:
            corr = np.corrcoef(vals, xp_vals)[0, 1]
            print(f"  Correlation with sum(q*p): {corr:.6f}")
        else:
            corr = 0.0
            print(f"  Correlation with sum(q*p): degenerate")

        # Test 3: Spectral analysis — eigenvalues of the matrix
        # representation of center element in adjoint rep
        # (trivial for center: ad_z = 0 by definition, but we can look
        # at the element's value distribution on phase space)
        val_mean = np.mean(vals)
        val_std = np.std(vals)
        val_skew = float(np.mean(((vals - val_mean) / (val_std + 1e-30))**3))
        val_kurt = float(np.mean(((vals - val_mean) / (val_std + 1e-30))**4)) - 3

        print(f"  Phase-space statistics: mean={val_mean:.4f}, std={val_std:.4f}")
        print(f"  Skewness: {val_skew:.4f}, Excess kurtosis: {val_kurt:.4f}")

        # Test 4: Does this look like a Hamiltonian with interesting spectrum?
        # For a self-adjoint HP operator, the "eigenvalues" would be the
        # values of H along classical trajectories. Check if the value
        # distribution has structure (not just Gaussian noise).
        from scipy.stats import normaltest
        if len(vals) >= 20:
            stat, p_value = normaltest(vals)
            print(f"  Normality test: stat={stat:.2f}, p={p_value:.4f} "
                  f"({'Gaussian' if p_value > 0.05 else 'NON-Gaussian'})")

        results.append({
            'index': c_idx,
            'n_nonzero_coeffs': int(np.sum(np.abs(coeffs) > 1e-10)),
            'time_reversal': tr_sym,
            'xp_correlation': float(corr),
            'mean': float(val_mean),
            'std': float(val_std),
            'skewness': float(val_skew),
            'kurtosis': float(val_kurt),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dir(CHECKPOINT_DIR)

    print("=" * 70)
    print("HILBERT-PÓLYA OPERATOR SEARCH")
    print("GUE Log-Gas Lie Algebra Structure Analysis")
    print("=" * 70)
    print(f"N = {N_BODIES}, d = {D_SPATIAL}")
    print(f"Potential: {POTENTIAL}")
    print(f"External potential: {EXTERNAL_POTENTIAL}")
    print(f"Max level: {MAX_LEVEL}")
    t_grand = time()

    # Initialize algebra
    algebra = NBodyAlgebra(
        n_bodies=N_BODIES,
        d_spatial=D_SPATIAL,
        potential=POTENTIAL,
        external_potential=EXTERNAL_POTENTIAL,
        checkpoint_dir=CHECKPOINT_DIR)
    print(f"Phase variables: {len(algebra.all_vars)}")
    print(f"  q: {algebra.q_vars}")
    print(f"  p: {algebra.p_vars}")
    print(f"  u: {algebra.u_vars}")

    # Phase 1: Build generators
    exprs, names, levels = build_generators(algebra, MAX_LEVEL, CHECKPOINT_DIR)
    print(f"\nGenerator summary:")
    for lv in range(MAX_LEVEL + 1):
        n_lv = sum(1 for l in levels if l == lv)
        print(f"  Level {lv}: {n_lv} generators")

    # Select basis
    basis_indices, rank, M = select_numerical_basis(
        algebra, exprs, N_SAMPLE_POINTS, SEED)
    print(f"\n  Dimension sequence: ", end="")
    cum = []
    for lv in range(MAX_LEVEL + 1):
        lv_indices = [i for i in range(len(exprs)) if levels[i] <= lv]
        M_lv = M[:, lv_indices]
        norms = np.linalg.norm(M_lv, axis=0)
        norms[norms < 1e-15] = 1.0
        _, s, _ = np.linalg.svd(M_lv / norms, full_matrices=False)
        r_lv = int(np.sum(s > 1e-8 * s[0]))
        cum.append(r_lv)
    print(cum)

    np.save(os.path.join(CHECKPOINT_DIR, "basis_indices.npy"), basis_indices)

    # Phase 2: Structure constants
    C, residuals = compute_numerical_structure_constants(
        algebra, exprs, names, levels, basis_indices,
        N_SAMPLE_POINTS, SEED, CHECKPOINT_DIR)

    # Verification
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    verify_antisymmetry(C)
    verify_jacobi(C)

    # Killing form
    print(f"\n{'='*70}")
    print("KILLING FORM")
    print(f"{'='*70}")
    K, k_eigs, sig = killing_form(C)
    print(f"  Signature: ({sig[0]}+, {sig[1]}-, {sig[2]} zero)")
    print(f"  Semisimple: {sig[2] == 0}")
    print(f"  Trace: {np.trace(K):.6g}")
    if sig[0] + sig[1] > 0:
        print(f"  Max eigenvalue: {k_eigs[-1]:.6g}")
        nz_idx = sig[2]
        if nz_idx < len(k_eigs):
            print(f"  Min non-zero eigenvalue: {k_eigs[nz_idx]:.6g}")

    # Derived series
    print(f"\n{'='*70}")
    print("DERIVED SERIES")
    print(f"{'='*70}")
    der_dims, is_solv, solv_len = derived_series(C)
    print(f"  Dimensions: {der_dims}")
    print(f"  Solvable: {is_solv}")
    if is_solv:
        print(f"  Solvability length: {solv_len}")

    # Lower central series
    print(f"\n{'='*70}")
    print("LOWER CENTRAL SERIES")
    print(f"{'='*70}")
    lcs_dims, is_nilp, nilp_class = lower_central_series(C)
    print(f"  Dimensions: {lcs_dims}")
    print(f"  Nilpotent: {is_nilp}")
    if is_nilp:
        print(f"  Nilpotency class: {nilp_class}")

    # Center
    print(f"\n{'='*70}")
    print("CENTER")
    print(f"{'='*70}")
    center_basis, center_dim = compute_center(C)
    print(f"  Center dimension: {center_dim}")

    # Levi decomposition
    print(f"\n{'='*70}")
    print("LEVI DECOMPOSITION")
    print(f"{'='*70}")
    levi = levi_decomposition(C)
    print(f"  Semisimple part dimension: {levi['semisimple_dim']}")
    print(f"  Radical dimension: {levi['radical_dim']}")

    # Phase 3: HP candidate analysis
    hp_results = analyze_center_elements(
        algebra, exprs, basis_indices, center_basis)

    # Save all results
    total_time = time() - t_grand
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    np.save(os.path.join(CHECKPOINT_DIR, "structure_constants.npy"), C)
    np.save(os.path.join(CHECKPOINT_DIR, "killing_form.npy"), K)
    np.save(os.path.join(CHECKPOINT_DIR, "killing_eigenvalues.npy"), k_eigs)
    np.save(os.path.join(CHECKPOINT_DIR, "residuals.npy"), residuals)
    if center_dim > 0:
        np.save(os.path.join(CHECKPOINT_DIR, "center_basis.npy"), center_basis)

    output = {
        "N": N_BODIES,
        "d": D_SPATIAL,
        "potential": POTENTIAL,
        "external_potential": EXTERNAL_POTENTIAL,
        "max_level": MAX_LEVEL,
        "n_generators": len(exprs),
        "rank": rank,
        "dimension_sequence": cum,
        "killing_signature": list(sig),
        "killing_trace": float(np.trace(K)),
        "is_semisimple": bool(sig[2] == 0),
        "derived_series": der_dims,
        "is_solvable": is_solv,
        "solvability_length": solv_len,
        "lower_central_series": lcs_dims,
        "is_nilpotent": is_nilp,
        "nilpotency_class": nilp_class,
        "center_dimension": center_dim,
        "levi_semisimple_dim": levi['semisimple_dim'],
        "levi_radical_dim": levi['radical_dim'],
        "max_residual": float(np.max(residuals)),
        "mean_residual": float(np.mean(residuals[residuals > 0])) if np.any(residuals > 0) else 0,
        "hp_candidates": hp_results,
        "computation_time_seconds": round(total_time, 1),
    }

    out_path = os.path.join(CHECKPOINT_DIR, "hp_search_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}")
    print(f"  Total time: {total_time:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Dimension sequence: {cum}")
    print(f"  Killing signature: ({sig[0]}+, {sig[1]}-, {sig[2]} zero)")
    print(f"  Solvable: {is_solv} (length {solv_len})")
    print(f"  Nilpotent: {is_nilp} (class {nilp_class})")
    print(f"  Center dimension: {center_dim}")
    print(f"  Levi: {levi['semisimple_dim']} semisimple + "
          f"{levi['radical_dim']} radical")
    if hp_results:
        print(f"\n  HP Candidates ({len(hp_results)} center elements):")
        for r in hp_results:
            print(f"    #{r['index']+1}: TR={r['time_reversal']}, "
                  f"xp_corr={r['xp_correlation']:.4f}")


if __name__ == "__main__":
    main()
