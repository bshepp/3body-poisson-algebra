#!/usr/bin/env python3
"""
Symbolic level-3 structure constant comparison across potentials.

Computes exact structure constants C_{ab}^k over QQ for the 116-dimensional
level-<=3 Poisson bracket algebra.  Key optimization: reduces each per-bracket
linear solve from RREF of 128,925 x 117 to RREF of 116 x 117 by precomputing
the pivot monomials of the basis matrix (7000x speedup in the LA step).

The symbolic Poisson bracket is the actual bottleneck (~10-60s per bracket).
Total: ~6670 brackets x ~20s avg => ~37 hours per potential.  Designed for
AWS with heavy checkpointing (resumable).

Usage
-----
  # Compute for a single potential (run two instances in parallel on AWS):
  python symbolic_level3_compare.py compute \\
      --potential "1/r" \\
      --checkpoint ../aws_results/nbody_checkpoints/checkpoints_N3_d2_1r/level_3.pkl \\
      --output-dir ../results/sc_1r

  python symbolic_level3_compare.py compute \\
      --potential "1/r^2" \\
      --checkpoint checkpoints_N3_d2_1r2/level_3.pkl \\
      --output-dir ../results/sc_1r2

  # Compare two completed computations:
  python symbolic_level3_compare.py compare ../results/sc_1r ../results/sc_1r2
"""

import argparse
import os
import sys
import pickle
import numpy as np
from time import time
from fractions import Fraction
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    """Load generators from checkpoint (old dict or new tuple format)."""
    print(f"  Loading checkpoint: {path}")
    t0 = time()
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        exprs, names, levels = data['exprs'], data['names'], data['levels']
    elif isinstance(data, (tuple, list)):
        exprs, names, levels = data[0], data[1], data[2]
    else:
        raise ValueError(f"Unknown checkpoint format: {type(data)}")
    print(f"  {len(exprs)} generators loaded in {time()-t0:.1f}s")
    return exprs, names, levels


def save_atomic(path, data):
    """Atomic save via tmp+rename."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Step 1: Monomial extraction (cached)
# ---------------------------------------------------------------------------

def extract_monomials(engine, exprs, cache_path):
    """Extract monomial-coefficient matrix with file caching."""
    if os.path.exists(cache_path):
        print(f"  Loading cached monomials: {cache_path}")
        with open(cache_path, 'rb') as f:
            d = pickle.load(f)
        poly_list = d['poly_list']
        monom_list = d['monom_list']
        monom_to_idx = d['monom_to_idx']
        print(f"  {len(poly_list)} generators, {len(monom_list)} monomials")
        return poly_list, monom_list, monom_to_idx

    poly_list, monom_list, monom_to_idx = engine.extract_monomial_matrix(exprs)
    save_atomic(cache_path, {
        'poly_list': poly_list,
        'monom_list': monom_list,
        'monom_to_idx': monom_to_idx,
    })
    return poly_list, monom_list, monom_to_idx


# ---------------------------------------------------------------------------
# Step 2: Numerical basis & pivot selection (seconds, not hours)
# ---------------------------------------------------------------------------

def _build_float_matrix(poly_list, monom_to_idx, n_mon):
    """Build dense float64 matrix from sparse poly_list."""
    n_gen = len(poly_list)
    M = np.zeros((n_gen, n_mon))
    for i, pdict in enumerate(poly_list):
        for monom, coeff in pdict.items():
            M[i, monom_to_idx[monom]] = float(coeff)
    return M


def select_basis_and_pivots(poly_list, monom_list, monom_to_idx,
                            cache_path, expected_rank=116):
    """Numerical QR to select basis generators AND pivot monomials.

    The SVD gap at rank 116 is >1e10 — numerical selection is exact.
    No DomainMatrix, no RREF on wide matrices. Runs in seconds.

    Returns (basis_indices, pivot_cols, pivot_col_map).
    """
    from scipy.linalg import qr as scipy_qr

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            d = pickle.load(f)
        print(f"  Loaded basis ({len(d['basis_indices'])} gens) and "
              f"pivots ({len(d['pivot_cols'])} cols) from cache")
        return (d['basis_indices'], d['pivot_cols'],
                {c: i for i, c in enumerate(d['pivot_cols'])})

    r = expected_rank
    n_gen = len(poly_list)
    n_mon = len(monom_list)
    print(f"\n  Numerical basis/pivot selection "
          f"({n_gen} x {n_mon})...", flush=True)
    t0 = time()

    M = _build_float_matrix(poly_list, monom_to_idx, n_mon)

    # Step A: Select r independent generators via column-pivoted QR on M^T
    # Column pivoting on M^T (n_mon x n_gen) gives row selection on M.
    _, R_gen, perm_gen = scipy_qr(M.T, pivoting=True)
    diag_gen = np.abs(np.diag(R_gen))
    gap_gen = diag_gen[r-1] / diag_gen[r] if r < len(diag_gen) else float('inf')
    print(f"    Generator QR gap at rank {r}: {gap_gen:.2e}  "
          f"(|R_{r}|={diag_gen[r-1]:.2e}, |R_{r+1}|={diag_gen[r]:.2e})")
    assert gap_gen > 1e6, f"Generator gap too small: {gap_gen}"
    basis_indices = sorted(perm_gen[:r].tolist())

    # Step B: Select r pivot monomials via column-pivoted QR on M_basis^T
    M_basis = M[basis_indices, :]  # (r x n_mon)
    _, R_mon, perm_mon = scipy_qr(M_basis.T, pivoting=True)
    diag_mon = np.abs(np.diag(R_mon))
    gap_mon = diag_mon[r-1] / diag_mon[r] if r < len(diag_mon) else float('inf')
    print(f"    Monomial QR gap at rank {r}: {gap_mon:.2e}")
    assert gap_mon > 1e6, f"Monomial gap too small: {gap_mon}"
    pivot_cols = sorted(perm_mon[:r].tolist())

    # Sanity: the r x r sub-matrix should be well-conditioned
    B_float = M_basis[:, pivot_cols]
    cond = np.linalg.cond(B_float)
    print(f"    Sub-matrix condition number: {cond:.2e}")
    assert cond < 1e14, f"Sub-matrix ill-conditioned: cond={cond}"

    print(f"    Done in {time()-t0:.1f}s")
    print(f"    Basis generators: {basis_indices[:10]}...")
    print(f"    Pivot monomials:  {pivot_cols[:10]}...")

    save_atomic(cache_path, {
        'basis_indices': basis_indices,
        'pivot_cols': pivot_cols,
        'gap_gen': float(gap_gen),
        'gap_mon': float(gap_mon),
        'cond': float(cond),
    })
    pivot_col_map = {c: i for i, c in enumerate(pivot_cols)}
    return basis_indices, pivot_cols, pivot_col_map


# ---------------------------------------------------------------------------
# Step 3: Exact QQ inverse of the r x r sub-matrix (once)
# ---------------------------------------------------------------------------

def build_exact_inverse(poly_list, monom_to_idx, basis_indices,
                        pivot_cols, cache_path):
    """Build exact B^{-1} over QQ for the r x r sub-matrix.

    B[a, j] = basis_generator_a's coefficient at pivot_monomial_j.
    Returns B_inv_rows as list-of-lists of QQ elements (serializable).
    """
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.domains import QQ

    r = len(basis_indices)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            d = pickle.load(f)
        print(f"  Loaded exact {r}x{r} inverse from cache")
        # Convert stored Fractions back to QQ
        B_inv_qq = []
        for row in d['B_inv_rows']:
            B_inv_qq.append([QQ.convert(Fraction(v)) for v in row])
        return B_inv_qq

    pivot_set = {c: j for j, c in enumerate(pivot_cols)}

    print(f"\n  Building exact {r}x{r} sub-matrix over QQ and inverting...",
          end=" ", flush=True)
    t0 = time()

    rows = []
    for idx in basis_indices:
        row = [QQ.zero] * r
        for monom, coeff in poly_list[idx].items():
            col = monom_to_idx[monom]
            if col in pivot_set:
                row[pivot_set[col]] = QQ.convert(coeff)
        rows.append(row)

    B_dm = DomainMatrix(rows, (r, r), QQ)
    B_inv_dm = B_dm.inv()
    print(f"done [{time()-t0:.1f}s]")

    # Extract as Fraction strings for serialisation
    B_inv_rows_str = []
    B_inv_rows_qq = []
    for i in range(r):
        row_str = []
        row_qq = []
        for j in range(r):
            elem = B_inv_dm[i, j].element
            row_str.append(str(Fraction(int(elem.numerator),
                                        int(elem.denominator))))
            row_qq.append(elem)
        B_inv_rows_str.append(row_str)
        B_inv_rows_qq.append(row_qq)

    save_atomic(cache_path, {'B_inv_rows': B_inv_rows_str})
    return B_inv_rows_qq


# ---------------------------------------------------------------------------
# Step 4: Per-bracket solve (exact, O(r^2) multiply — no RREF)
# ---------------------------------------------------------------------------

def solve_one_bracket(bracket_dict, monom_to_idx, pivot_col_map,
                      B_inv_rows, r):
    """Solve for structure constants: c = B^{-1} @ rhs_pivot.

    Extract bracket's coefficients at r pivot monomials, then exact
    matrix-vector multiply by precomputed inverse.  No RREF needed.

    Returns (coeffs_list, False).  Second element kept for API compat.
    """
    from sympy.polys.domains import QQ

    # Extract bracket values at pivot monomials
    rhs = [QQ.zero] * r
    for monom, coeff in bracket_dict.items():
        if monom in monom_to_idx:
            col = monom_to_idx[monom]
            if col in pivot_col_map:
                rhs[pivot_col_map[col]] = QQ.convert(coeff)

    # c = B_inv @ rhs  (exact over QQ, sparse-aware)
    coeffs = [QQ.zero] * r
    # Pre-filter nonzero rhs entries for speed
    nz_rhs = [(j, rhs[j]) for j in range(r) if rhs[j] != QQ.zero]

    for k in range(r):
        s = QQ.zero
        for j, rj in nz_rhs:
            bij = B_inv_rows[k][j]
            if bij != QQ.zero:
                s = s + bij * rj
        coeffs[k] = s

    return coeffs, False


# ---------------------------------------------------------------------------
# Step 6: Verification
# ---------------------------------------------------------------------------

def verify_bracket(bracket_dict, monom_to_idx, poly_list,
                   basis_indices, coeffs):
    """Check reconstruction: sum_k c_k * basis_k == bracket  (over all monomials).

    Returns (exact_match: bool, n_extra_monomials: int).
    n_extra_monomials counts bracket monomials not in the generator set
    (level-4 components).
    """
    from sympy.polys.domains import QQ

    n_extra = sum(1 for m in bracket_dict if m not in monom_to_idx)

    # Reconstruct bracket from coefficients (sparse accumulation)
    reconstructed = {}
    for k, idx in enumerate(basis_indices):
        c_k = coeffs[k]
        if c_k == QQ.zero:
            continue
        for monom, coeff in poly_list[idx].items():
            col = monom_to_idx[monom]
            val = c_k * QQ.convert(coeff)
            if col in reconstructed:
                reconstructed[col] = reconstructed[col] + val
            else:
                reconstructed[col] = val

    # Build bracket vector (only known monomials)
    bracket_vec = {}
    for monom, coeff in bracket_dict.items():
        if monom in monom_to_idx:
            bracket_vec[monom_to_idx[monom]] = QQ.convert(coeff)

    # Compare all entries
    all_cols = set(reconstructed.keys()) | set(bracket_vec.keys())
    for col in all_cols:
        r_val = reconstructed.get(col, QQ.zero)
        b_val = bracket_vec.get(col, QQ.zero)
        if r_val != b_val:
            return False, n_extra

    return True, n_extra


# ---------------------------------------------------------------------------
# Main computation loop
# ---------------------------------------------------------------------------

def compute_structure_constants(engine, exprs, names, levels,
                                poly_list, monom_list, monom_to_idx,
                                basis_indices, pivot_cols, pivot_col_map,
                                B_inv_rows,
                                output_dir, save_every=50, verify_every=200):
    """Compute all structure constants with checkpointing.

    For each pair (a,b) with a < b among basis generators:
      1. Symbolic Poisson bracket  (seconds -- the bottleneck)
      2. Poly expansion over QQ
      3. Extract values at 116 pivot monomials
      4. Exact multiply by precomputed B^{-1} (O(r^2), no RREF)
      5. Store exact rational coefficients

    Checkpoints every save_every brackets; full verification every verify_every.
    """
    from sympy import expand, Poly
    from sympy.polys.domains import QQ

    r = len(basis_indices)
    n_mon = len(monom_list)
    total_brackets = r * (r - 1) // 2

    print(f"\n{'='*70}")
    print(f"STRUCTURE CONSTANTS  (rank {r},  {total_brackets} brackets)")
    print(f"  Monomials: {n_mon},  pivot subset: {r}")
    print(f"  Save every {save_every},  verify every {verify_every}")
    print(f"{'='*70}")

    # Load or initialise results
    sc_path = os.path.join(output_dir, 'structure_constants_partial.pkl')
    vlog_path = os.path.join(output_dir, 'verification_log.pkl')

    C_exact = [[[None] * r for _ in range(r)] for _ in range(r)]
    C_float = np.zeros((r, r, r))
    n_done = 0
    vlog = {}

    if os.path.exists(sc_path):
        with open(sc_path, 'rb') as f:
            ckpt = pickle.load(f)
        C_float = ckpt['C_float']
        C_exact = ckpt['C_exact']
        n_done = ckpt['n_computed']
        print(f"  Resuming from checkpoint: {n_done}/{total_brackets}")

    if os.path.exists(vlog_path):
        with open(vlog_path, 'rb') as f:
            vlog = pickle.load(f)

    t_total = time()
    n_computed = 0
    n_zero = 0
    n_vok = sum(1 for v in vlog.values() if v[0])
    n_vfail = sum(1 for v in vlog.values() if not v[0])

    for a in range(r):
        i = basis_indices[a]
        for b in range(a + 1, r):
            j = basis_indices[b]
            n_computed += 1

            if n_computed <= n_done:
                continue

            t0_bracket = time()

            # ---------- symbolic bracket ----------
            bracket = engine._poisson_bracket(exprs[i], exprs[j])
            bracket = engine._simplify(bracket)

            expanded = expand(bracket)
            if engine._log_subs:
                expanded = expanded.subs(engine._log_subs)

            if expanded == 0:
                for k in range(r):
                    C_exact[a][b][k] = "0"
                    C_exact[b][a][k] = "0"
                n_zero += 1
            else:
                p = Poly(expanded, *engine.phase_vars, domain='QQ')
                bracket_dict = p.as_dict()

                # ---------- solve ----------
                coeffs, _ = solve_one_bracket(
                    bracket_dict, monom_to_idx, pivot_col_map,
                    B_inv_rows, r)

                for k in range(r):
                    val = coeffs[k]
                    if hasattr(val, 'numerator'):
                        frac = Fraction(int(val.numerator),
                                        int(val.denominator))
                    else:
                        frac = Fraction(0)
                    C_exact[a][b][k] = str(frac)
                    C_exact[b][a][k] = str(-frac)
                    C_float[a, b, k] = float(frac)
                    C_float[b, a, k] = -float(frac)

                # ---------- verification ----------
                if verify_every and n_computed % verify_every == 0:
                    ok, n_extra = verify_bracket(
                        bracket_dict, monom_to_idx, poly_list,
                        basis_indices, coeffs)
                    vlog[n_computed] = (ok, n_extra)
                    if ok:
                        n_vok += 1
                    else:
                        n_vfail += 1

            dt = time() - t0_bracket

            # ---------- progress ----------
            if n_computed % 10 == 0 or n_computed == total_brackets:
                elapsed = time() - t_total
                done_since_resume = n_computed - n_done
                rate = done_since_resume / elapsed if elapsed > 0 else 0
                remaining = total_brackets - n_computed
                eta_h = (remaining / rate / 3600) if rate > 0 else float('inf')
                lv_a, lv_b = levels[i], levels[j]
                print(f"  [{n_computed:5d}/{total_brackets}]  "
                      f"lv({lv_a},{lv_b})  "
                      f"{dt:6.1f}s  "
                      f"rate={rate:.2f}/s  "
                      f"ETA={eta_h:.1f}h  "
                      f"zeros={n_zero}  "
                      f"verify={n_vok}/{n_vok+n_vfail}")

            # ---------- checkpoint ----------
            if n_computed % save_every == 0:
                save_atomic(sc_path, {
                    'C_float': C_float,
                    'C_exact': C_exact,
                    'n_computed': n_computed,
                })
                if vlog:
                    save_atomic(vlog_path, vlog)
                print(f"    >> checkpoint saved at {n_computed}")

    # Diagonal = 0
    for a in range(r):
        for k in range(r):
            C_exact[a][a][k] = "0"

    # ---------- final save ----------
    elapsed_total = time() - t_total
    print(f"\n  Completed {total_brackets} brackets in "
          f"{elapsed_total:.1f}s ({elapsed_total/3600:.2f}h)")
    print(f"  Zero brackets: {n_zero}")
    print(f"  Verified: {n_vok} OK, {n_vfail} FAIL")

    n_nonzero = int(np.count_nonzero(C_float))
    print(f"  Non-zero C entries: {n_nonzero}/{r**3} "
          f"({100*n_nonzero/r**3:.1f}%)")

    final_path = os.path.join(output_dir, 'structure_constants_final.pkl')
    save_atomic(final_path, {
        'C_float': C_float,
        'C_exact': C_exact,
        'basis_indices': basis_indices,
        'pivot_cols': pivot_cols,
        'n_computed': total_brackets,
    })
    np.save(os.path.join(output_dir, 'C_float.npy'), C_float)

    if vlog:
        save_atomic(vlog_path, vlog)

    return C_float, C_exact


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def compare_results(dir1, dir2):
    """Compare structure constant tensors from two potentials."""
    print(f"\n{'='*70}")
    print(f"COMPARING STRUCTURE CONSTANTS")
    print(f"  Dir 1: {dir1}")
    print(f"  Dir 2: {dir2}")
    print(f"{'='*70}")

    def load_sc(d):
        for name in ('structure_constants_final.pkl',
                      'structure_constants_partial.pkl'):
            p = os.path.join(d, name)
            if os.path.exists(p):
                if 'partial' in name:
                    print(f"  WARNING: using partial results from {d}")
                with open(p, 'rb') as f:
                    return pickle.load(f)
        raise FileNotFoundError(f"No results in {d}")

    d1 = load_sc(dir1)
    d2 = load_sc(dir2)

    C1_exact, C2_exact = d1['C_exact'], d2['C_exact']
    C1_float, C2_float = d1['C_float'], d2['C_float']
    b1, b2 = d1['basis_indices'], d2['basis_indices']

    r = len(b1)
    print(f"\n  Rank: {r}")
    print(f"  Basis indices match: {b1 == b2}")
    if b1 != b2:
        print(f"    Potential 1 basis: {b1[:10]}...")
        print(f"    Potential 2 basis: {b2[:10]}...")
        print(f"    Differing positions: "
              f"{[i for i in range(r) if b1[i] != b2[i]][:20]}")
        print(f"\n  WARNING: different bases -- direct comparison of "
              f"C[a,b,k] is not valid.")
        print(f"  A change-of-basis transformation is needed.")
        return

    # ---------- exact comparison ----------
    nc1 = d1.get('n_computed', '?')
    nc2 = d2.get('n_computed', '?')
    print(f"\n  Brackets computed: {nc1} / {nc2}")

    n_match = 0
    n_mismatch = 0
    n_both_zero = 0
    n_skip = 0  # entries where one or both are None (not yet computed)
    mismatches = []

    for a in range(r):
        for b in range(r):
            for k in range(r):
                v1 = C1_exact[a][b][k]
                v2 = C2_exact[a][b][k]
                if v1 is None or v2 is None:
                    n_skip += 1
                    continue
                if v1 == v2:
                    n_match += 1
                    if v1 == "0":
                        n_both_zero += 1
                else:
                    n_mismatch += 1
                    if len(mismatches) < 30:
                        mismatches.append((a, b, k, v1, v2))

    total = r ** 3
    print(f"\n  Exact comparison over QQ:")
    print(f"    Total entries:    {total:>12,}")
    print(f"    Compared:         {n_match+n_mismatch:>12,}")
    print(f"    Skipped (None):   {n_skip:>12,}")
    print(f"    Matches:          {n_match:>12,}  "
          f"({100*n_match/max(1,n_match+n_mismatch):.4f}%)")
    print(f"      both zero:      {n_both_zero:>12,}")
    print(f"      non-zero match: {n_match - n_both_zero:>12,}")
    print(f"    MISMATCHES:       {n_mismatch:>12,}")

    if n_mismatch == 0 and n_skip == 0:
        print(f"\n  *** STRUCTURE CONSTANTS ARE BIT-FOR-BIT IDENTICAL "
              f"OVER Q ***")
    elif n_mismatch == 0:
        print(f"\n  All compared entries match (some not yet computed).")
    else:
        print(f"\n  First mismatches:")
        for a, b, k, v1, v2 in mismatches[:20]:
            print(f"    C[{a},{b},{k}]:  {v1}  vs  {v2}")

    # ---------- float comparison ----------
    diff = np.abs(C1_float - C2_float)
    mask = (C1_float != 0) | (C2_float != 0)

    print(f"\n  Float comparison:")
    print(f"    ||C1 - C2||_F  = {np.linalg.norm(diff):.6e}")
    print(f"    max|C1 - C2|   = {np.max(diff):.6e}")
    if mask.any():
        rel = diff[mask] / np.maximum(np.abs(C1_float[mask]),
                                       np.abs(C2_float[mask]))
        print(f"    max relative    = {np.max(rel):.6e}")

    # ---------- spectral / Killing form ----------
    print(f"\n  Killing form K[i,j] = Tr(ad_i @ ad_j):")
    for label, C in [("Potential 1", C1_float), ("Potential 2", C2_float)]:
        K = np.einsum('ikl,jlk->ij', C, C)
        eigs = np.linalg.eigvalsh(K)
        n_nonzero_eig = np.sum(np.abs(eigs) > 1e-10)
        print(f"    {label}:  tr(K)={np.trace(K):.6f}  "
              f"rank={n_nonzero_eig}  "
              f"eigs=[{eigs.min():.4f} .. {eigs.max():.4f}]")

    K1 = np.einsum('ikl,jlk->ij', C1_float, C1_float)
    K2 = np.einsum('ikl,jlk->ij', C2_float, C2_float)
    print(f"    ||K1 - K2||_F  = {np.linalg.norm(K1 - K2):.6e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_compute(args):
    from symbolic_rank_nbody import NBodySymbolicRank

    t_start = time()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"POTENTIAL: {args.potential}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    # 1. Load generators
    exprs, names, levels = load_checkpoint(args.checkpoint)
    lc = Counter(levels)
    print(f"  Levels: {dict(sorted(lc.items()))}")

    # 2. Create engine
    engine = NBodySymbolicRank(
        n_bodies=3, d_spatial=2, potential=args.potential)

    # 3. Monomial extraction
    monom_path = os.path.join(output_dir, 'monomial_matrix.pkl')
    poly_list, monom_list, monom_to_idx = extract_monomials(
        engine, exprs, monom_path)
    print(f"  Monomials: {len(monom_list):,}")

    # 4. Numerical basis & pivot selection (seconds, not hours)
    sel_path = os.path.join(output_dir, 'basis_pivot_selection.pkl')
    basis_indices, pivot_cols, pivot_col_map = select_basis_and_pivots(
        poly_list, monom_list, monom_to_idx, sel_path, expected_rank=116)

    # 5. Exact inverse of r x r sub-matrix (once, fast)
    inv_path = os.path.join(output_dir, 'B_inv.pkl')
    B_inv_rows = build_exact_inverse(
        poly_list, monom_to_idx, basis_indices, pivot_cols, inv_path)

    # 6. Structure constants
    compute_structure_constants(
        engine, exprs, names, levels,
        poly_list, monom_list, monom_to_idx,
        basis_indices, pivot_cols, pivot_col_map,
        B_inv_rows,
        output_dir,
        save_every=args.save_every,
        verify_every=args.verify_every)

    print(f"\nTotal wall time: {time()-t_start:.1f}s "
          f"({(time()-t_start)/3600:.2f}h)")


def cmd_compare(args):
    compare_results(args.dir1, args.dir2)


def main():
    parser = argparse.ArgumentParser(
        description="Exact level-3 structure constants over QQ")
    sub = parser.add_subparsers(dest='command')

    p_c = sub.add_parser('compute',
                         help='Compute structure constants for one potential')
    p_c.add_argument('--potential', required=True)
    p_c.add_argument('--checkpoint', required=True,
                     help='Path to level_3.pkl')
    p_c.add_argument('--output-dir', required=True)
    p_c.add_argument('--save-every', type=int, default=50)
    p_c.add_argument('--verify-every', type=int, default=200)

    p_cmp = sub.add_parser('compare',
                           help='Compare two completed computations')
    p_cmp.add_argument('dir1')
    p_cmp.add_argument('dir2')

    args = parser.parse_args()
    if args.command == 'compute':
        cmd_compute(args)
    elif args.command == 'compare':
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
