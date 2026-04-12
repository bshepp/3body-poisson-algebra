#!/usr/bin/env python3
"""
Identify the 3 Extra Generators: Neural (119) vs Gravitational (116)
====================================================================

Both algebras share the same bracket tree structure (identical pairing order).
We identify which specific bracket positions are linearly independent in the
neural algebra but become dependent in the gravitational algebra.

Methodology (adapted from nbody/identify_117th.py):
1. Build all generators through level 3 for BOTH algebras using identical
   bracket pairing order.
2. Neural: exact rank via DomainMatrix over QQ (6-variable polynomials).
3. Gravitational: numerical rank via Gram-Schmidt on evaluation matrix
   (15 variables including u_ij make exact polynomial method impractical).
4. Compare rank-adding generator indices to find the 3 divergence points.
5. Characterize the extra generators (degree, symmetry, structure).

Expected results:
- Neural (gradient coupling): [3, 6, 17, 119]
- Gravitational (1/r):        [3, 6, 17, 116]
- Difference: 3 generators at level 3

Usage:
    python neural/nn_extra_generators.py
"""

import os
import sys
import numpy as np
from time import time
from itertools import permutations

from sympy import (symbols, diff, Integer, cancel, expand, Add,
                   Rational, Symbol, Poly)
from sympy.polys.matrices import DomainMatrix
from sympy.polys.domains import QQ
import sympy as sp

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exact_growth import (
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    poisson_bracket, simplify_generator,
    lambdify_generators,
    sample_phase_space as grav_sample_phase_space,
)

# =====================================================================
# Variable mapping: neural network -> gravitational engine
# =====================================================================
w1, w2, w3 = x1, x2, x3
v1, v2, v3 = px1, px2, px3

# Active variables for neural algebra (polynomial in 6 vars only)
NN_VARS = [x1, x2, x3, px1, px2, px3]

# =====================================================================
# Hamiltonians
# =====================================================================

# --- Neural: gradient-product coupling ---
T1_nn = v1**2 / 2
T2_nn = v2**2 / 2
T3_nn = v3**2 / 2

product = w1 * w2 * w3
L = (product - 1)**2 / 2
dL_dw1 = diff(L, w1)
dL_dw2 = diff(L, w2)
dL_dw3 = diff(L, w3)

V_12_grad = dL_dw1 * dL_dw2 / 2
V_13_grad = dL_dw1 * dL_dw3 / 2
V_23_grad = dL_dw2 * dL_dw3 / 2

NN_H12 = T1_nn + T2_nn + V_12_grad
NN_H13 = T1_nn + T3_nn + V_13_grad
NN_H23 = T2_nn + T3_nn + V_23_grad

# --- Gravitational: planar 1/r ---
T1_grav = (px1**2 + py1**2) / 2
T2_grav = (px2**2 + py2**2) / 2
T3_grav = (px3**2 + py3**2) / 2

GR_H12 = T1_grav + T2_grav - u12
GR_H13 = T1_grav + T3_grav - u13
GR_H23 = T2_grav + T3_grav - u23


# =====================================================================
# Build bracket tree (same pairing order for both algebras)
# =====================================================================
def build_bracket_tree(H12, H13, H23, max_level=3, label=""):
    """Build all generators through max_level with deterministic pairing.

    Returns: (exprs, names, levels)
    """
    print(f"\n{'='*70}")
    print(f"BUILDING GENERATORS: {label}")
    print(f"{'='*70}")
    t_total = time()

    exprs = [H12, H13, H23]
    names = ["H12", "H13", "H23"]
    levels = [0, 0, 0]
    computed_pairs = set()

    for i in range(3):
        for j in range(i + 1, 3):
            computed_pairs.add(frozenset({i, j}))

    # Level 1
    print("  Level 1...")
    for i, j, sname in [(0, 1, "K1"), (0, 2, "K2"), (1, 2, "K3")]:
        t0 = time()
        expr = poisson_bracket(exprs[i], exprs[j])
        expr = simplify_generator(expr)
        exprs.append(expr)
        names.append(sname)
        levels.append(1)
        nterms = len(Add.make_args(expr))
        print(f"    {sname}={{{names[i]},{names[j]}}}: {nterms} terms [{time()-t0:.1f}s]")

    # Levels 2+
    for level in range(2, max_level + 1):
        print(f"  Level {level}...")
        t_level = time()
        frontier = [i for i, lv in enumerate(levels) if lv == level - 1]
        n_existing = len(exprs)
        new_exprs = []
        new_names = []
        count = 0

        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)

                bname = f"{{{names[i]},{names[j]}}}"

                if exprs[i] == 0 or exprs[j] == 0:
                    new_exprs.append(Integer(0))
                    new_names.append(bname)
                    continue

                count += 1
                t0 = time()
                expr = poisson_bracket(exprs[i], exprs[j])
                expr = simplify_generator(expr)
                elapsed = time() - t0
                nterms = len(Add.make_args(expr))
                new_exprs.append(expr)
                new_names.append(bname)

                if count % 20 == 0 or elapsed > 5:
                    print(f"    [{count:>3d}] {bname[:55]:55s} "
                          f"{nterms:>5d} terms [{elapsed:.1f}s]")

        for expr, name in zip(new_exprs, new_names):
            exprs.append(expr)
            names.append(name)
            levels.append(level)

        n_zero = sum(1 for e in new_exprs if e == 0)
        n_nonzero = len(new_exprs) - n_zero
        print(f"    Level {level}: {len(new_exprs)} candidates "
              f"({n_nonzero} nonzero, {n_zero} zero) [{time()-t_level:.1f}s]")

    print(f"  Total: {len(exprs)} generators [{time()-t_total:.1f}s]")
    return exprs, names, levels


# =====================================================================
# Exact rank progression (DomainMatrix over QQ)
# =====================================================================
def exact_rank_progression(exprs, names, levels, poly_vars, label=""):
    """
    Build exact monomial-coefficient matrix over QQ and track rank
    as each generator is added. Returns which generators add to rank.
    """
    print(f"\n{'='*70}")
    print(f"EXACT RANK PROGRESSION (QQ): {label}")
    print(f"{'='*70}")
    t0 = time()

    # Convert all expressions to polynomial dicts
    poly_dicts = []
    all_monoms = set()
    for i, expr in enumerate(exprs):
        if expr == 0:
            poly_dicts.append({})
            continue
        expanded = expand(expr)
        try:
            p = Poly(expanded, *poly_vars, domain='QQ')
            md = p.as_dict()
        except Exception as e:
            print(f"  WARNING: gen {i} ({names[i]}) Poly failed: {e}")
            poly_dicts.append({})
            continue
        poly_dicts.append(md)
        all_monoms.update(md.keys())

    monom_list = sorted(all_monoms)
    monom_to_idx = {m: idx for idx, m in enumerate(monom_list)}
    n_mon = len(monom_list)
    print(f"  {len(exprs)} generators, {n_mon} distinct monomials [{time()-t0:.1f}s]")

    # Incremental Gaussian elimination over QQ
    # Maintain reduced rows with pivot columns for O(n*m) per new row
    # instead of O(n^2*m) full RREF each time
    pivots = []        # list of (pivot_col, reduced_row) pairs
    rank_adding = []   # indices of generators that add to rank
    current_rank = 0

    for i, md in enumerate(poly_dicts):
        if not md:
            continue

        # Build sparse row as dict for efficiency
        row = {}
        for monom, coeff in md.items():
            row[monom_to_idx[monom]] = QQ.convert(coeff)

        # Reduce against existing pivots
        for pcol, prow in pivots:
            if pcol in row:
                factor = row[pcol]
                for k, v in prow.items():
                    if k in row:
                        row[k] -= factor * v
                    else:
                        row[k] = -factor * v
                del row[pcol]
                # Clean up zeros from floating precision (exact QQ so just check)
                row = {k: v for k, v in row.items() if v != QQ.zero}

        if not row:
            continue  # linearly dependent

        # Find leading (smallest) column index as pivot
        pcol = min(row.keys())
        # Normalize so pivot = 1
        scale = QQ.one / row[pcol]
        row = {k: v * scale for k, v in row.items()}

        # Back-substitute: eliminate this pivot column from existing pivots
        for idx, (existing_pcol, existing_prow) in enumerate(pivots):
            if pcol in existing_prow:
                factor = existing_prow[pcol]
                for k, v in row.items():
                    if k in existing_prow:
                        existing_prow[k] -= factor * v
                    else:
                        existing_prow[k] = -factor * v
                del existing_prow[pcol]
                existing_prow = {k: v for k, v in existing_prow.items()
                                 if v != QQ.zero}
                pivots[idx] = (existing_pcol, existing_prow)

        pivots.append((pcol, row))
        rank_adding.append(i)
        current_rank += 1

        if current_rank % 20 == 0:
            print(f"    rank {current_rank} after gen {i} ({names[i]}) [{time()-t0:.1f}s]")

    # Summary by level
    print()
    for lv in sorted(set(levels)):
        # Count how many rank-adding generators have level <= lv
        rank_at_lv = sum(1 for idx in rank_adding if levels[idx] <= lv)
        print(f"  Through level {lv}: rank {rank_at_lv}")
    print(f"  Final rank: {current_rank} [{time()-t0:.1f}s]")

    return current_rank, rank_adding


# =====================================================================
# Numerical rank progression (SVD with gap-ratio detection)
# =====================================================================
def svd_rank(matrix, gap_threshold=1e6):
    """Determine numerical rank via SVD gap-ratio detection."""
    svs = np.linalg.svd(matrix, compute_uv=False)
    if len(svs) == 0:
        return 0, svs
    # Find position of maximum gap ratio
    max_gap = 0
    max_gap_pos = len(svs) - 1  # default: all are significant
    for r in range(len(svs) - 1):
        if svs[r+1] < 1e-14:
            return r + 1, svs  # everything after is numerically zero
        gap = svs[r] / svs[r+1]
        if gap > max_gap:
            max_gap = gap
            max_gap_pos = r
    if max_gap > gap_threshold:
        return max_gap_pos + 1, svs
    # No clear gap — all singular values are significant
    return len(svs), svs


def numerical_rank_progression(exprs, names, levels,
                               n_samples=2000, seed=42, label=""):
    """
    Track rank via SVD with gap-ratio detection (standard validated method).
    """
    print(f"\n{'='*70}")
    print(f"NUMERICAL RANK PROGRESSION (SVD): {label}")
    print(f"{'='*70}")

    # Filter nonzero
    nonzero_indices = [i for i, e in enumerate(exprs) if e != 0]
    nonzero_exprs = [exprs[i] for i in nonzero_indices]

    print(f"  {len(nonzero_exprs)} nonzero generators, "
          f"sampling {n_samples} points...")
    t0 = time()

    Z_qp, Z_u = grav_sample_phase_space(n_samples, seed)
    evaluate = lambdify_generators(nonzero_exprs)
    M = evaluate(Z_qp, Z_u)
    print(f"  Evaluation matrix: {M.shape} [{time()-t0:.1f}s]")

    # Normalize columns for numerical stability
    col_norms = np.linalg.norm(M, axis=0)
    col_norms[col_norms == 0] = 1.0
    M_normed = M / col_norms[np.newaxis, :]

    # Through-level SVD ranks (validation)
    print(f"\n  Through-level SVD ranks:")
    for lv in sorted(set(levels)):
        cols_thru_lv = [j for j, orig_idx in enumerate(nonzero_indices)
                        if levels[orig_idx] <= lv]
        if not cols_thru_lv:
            continue
        sub = M_normed[:, cols_thru_lv]
        rank_lv, svs = svd_rank(sub)
        if rank_lv < len(svs):
            gap = svs[rank_lv-1] / max(svs[rank_lv], 1e-300)
        else:
            gap = float('inf')
        print(f"    Level {lv}: rank {rank_lv}  "
              f"(gap={gap:.2e})")

    # Incremental rank tracking via SVD
    print(f"\n  Incremental rank tracking via SVD...")
    rank_adding = []
    accumulated_cols = np.zeros((n_samples, 0))
    current_rank = 0

    for nz_j, orig_idx in enumerate(nonzero_indices):
        test_matrix = np.column_stack([accumulated_cols, M_normed[:, nz_j]])
        new_rank, _ = svd_rank(test_matrix)

        if new_rank > current_rank:
            rank_adding.append(orig_idx)
            accumulated_cols = test_matrix
            current_rank = new_rank

        if (nz_j + 1) % 20 == 0:
            print(f"    {nz_j+1}/{len(nonzero_indices)}: "
                  f"rank {current_rank} [{time()-t0:.1f}s]")

    # Summary by level
    print()
    for lv in sorted(set(levels)):
        rank_at_lv = sum(1 for idx in rank_adding if levels[idx] <= lv)
        print(f"  Through level {lv}: rank {rank_at_lv}")
    print(f"  Final rank: {len(rank_adding)} [{time()-t0:.1f}s]")

    return len(rank_adding), rank_adding


# =====================================================================
# Compare and identify extra generators
# =====================================================================
def compare_algebras(nn_rank_adding, nn_names, nn_levels,
                     gr_rank_adding, gr_names, gr_levels):
    """Find generators that are rank-adding in neural but not gravity."""

    print(f"\n{'='*70}")
    print("COMPARISON: NEURAL vs GRAVITATIONAL")
    print(f"{'='*70}")

    nn_set = set(nn_rank_adding)
    gr_set = set(gr_rank_adding)

    # Generators independent in neural but dependent in gravity
    nn_only = sorted(nn_set - gr_set)
    # Generators independent in gravity but dependent in neural
    gr_only = sorted(gr_set - nn_set)
    # Common
    common = sorted(nn_set & gr_set)

    print(f"\n  Common independent generators: {len(common)}")
    print(f"  Neural-only independent:       {len(nn_only)}")
    print(f"  Gravity-only independent:      {len(gr_only)}")

    if nn_only:
        print(f"\n  --- EXTRA NEURAL GENERATORS (the {len(nn_only)} extras) ---")
        for idx in nn_only:
            print(f"    Index {idx:>3d}  level={nn_levels[idx]}  "
                  f"name={nn_names[idx]}")

    if gr_only:
        print(f"\n  --- EXTRA GRAVITATIONAL GENERATORS ---")
        for idx in gr_only:
            print(f"    Index {idx:>3d}  level={gr_levels[idx]}  "
                  f"name={gr_names[idx]}")

    return nn_only, gr_only


# =====================================================================
# Characterize the extra generators
# =====================================================================
def characterize_generators(extra_indices, exprs, names, levels, poly_vars):
    """Detailed analysis of the extra generators."""

    print(f"\n{'='*70}")
    print("CHARACTERIZATION OF EXTRA GENERATORS")
    print(f"{'='*70}")

    # S3 permutation substitutions (on neural variables x1,x2,x3,px1,px2,px3)
    s3_perms = [
        ("identity",  {x1: x1, x2: x2, x3: x3, px1: px1, px2: px2, px3: px3}),
        ("(12)",      {x1: x2, x2: x1, x3: x3, px1: px2, px2: px1, px3: px3}),
        ("(13)",      {x1: x3, x2: x2, x3: x1, px1: px3, px2: px2, px3: px1}),
        ("(23)",      {x1: x1, x2: x3, x3: x2, px1: px1, px2: px3, px3: px2}),
        ("(123)",     {x1: x2, x2: x3, x3: x1, px1: px2, px2: px3, px3: px1}),
        ("(132)",     {x1: x3, x2: x1, x3: x2, px1: px3, px2: px1, px3: px2}),
    ]

    for idx in extra_indices:
        expr = exprs[idx]
        name = names[idx]
        level = levels[idx]

        print(f"\n  {'='*60}")
        print(f"  Generator {idx}: {name}")
        print(f"  Level: {level}")
        print(f"  {'='*60}")

        if expr == 0:
            print("    ZERO expression — this shouldn't happen!")
            continue

        expanded = expand(expr)
        nterms = len(Add.make_args(expanded))
        print(f"  Terms: {nterms}")

        # Polynomial analysis
        try:
            p = Poly(expanded, *poly_vars, domain='QQ')
            md = p.as_dict()
            max_deg = max(sum(m) for m in md.keys())
            min_deg = min(sum(m) for m in md.keys())

            # Separate position and momentum degrees
            n_q = 3  # x1, x2, x3
            q_degs = set()
            p_degs = set()
            for monom in md.keys():
                q_d = sum(monom[:n_q])
                p_d = sum(monom[n_q:])
                q_degs.add(q_d)
                p_degs.add(p_d)

            print(f"  Total degree: {min_deg}..{max_deg}")
            print(f"  Position (w) degrees: {sorted(q_degs)}")
            print(f"  Momentum (v) degrees: {sorted(p_degs)}")
            print(f"  Distinct monomials: {len(md)}")

            # Degree distribution
            deg_counts = {}
            for monom in md.keys():
                d = sum(monom)
                deg_counts[d] = deg_counts.get(d, 0) + 1
            print(f"  Degree distribution: "
                  + ", ".join(f"d={d}:{c}" for d, c in sorted(deg_counts.items())))
        except Exception as e:
            print(f"  Poly analysis failed: {e}")
            md = None

        # S3 symmetry analysis
        print(f"\n  S3 SYMMETRY ANALYSIS:")
        symmetric_count = 0
        anti_count = 0

        for perm_name, sub_dict in s3_perms:
            permuted = expand(expr.subs(sub_dict))
            diff_expr = expand(permuted - expr)
            sum_expr = expand(permuted + expr)

            if diff_expr == 0:
                relation = "INVARIANT"
                symmetric_count += 1
            elif sum_expr == 0:
                relation = "ANTI-SYMMETRIC"
                anti_count += 1
            else:
                # Check if it's a scalar multiple
                ratio = None
                if nterms <= 200:
                    try:
                        r = cancel(permuted / expr)
                        if r.is_number:
                            ratio = r
                    except Exception:
                        pass
                if ratio is not None:
                    relation = f"SCALED by {ratio}"
                else:
                    relation = "MIXED"
            print(f"    {perm_name:>6s}: {relation}")

        if symmetric_count == 6:
            print(f"    => TOTALLY SYMMETRIC (trivial representation)")
        elif anti_count >= 3:
            print(f"    => SIGN REPRESENTATION (anti-symmetric under transpositions)")
        elif symmetric_count == 1:
            print(f"    => STANDARD REPRESENTATION (2-dimensional)")
        else:
            print(f"    => MIXED ({symmetric_count} symmetric, {anti_count} anti)")

        # Show the expression if compact
        if nterms <= 40:
            print(f"\n  Expression:")
            print(f"    {expanded}")
        else:
            terms = Add.make_args(expanded)
            # Group by momentum degree for readability
            if md:
                by_p_deg = {}
                for monom, coeff in md.items():
                    p_d = sum(monom[n_q:])
                    by_p_deg.setdefault(p_d, []).append((monom, coeff))
                print(f"\n  Expression structure by momentum degree:")
                for p_d in sorted(by_p_deg.keys()):
                    items = by_p_deg[p_d]
                    print(f"    v-degree {p_d}: {len(items)} monomials")
                    if len(items) <= 8:
                        for monom, coeff in items:
                            monom_str = " * ".join(
                                f"{v}^{e}" if e > 1 else str(v)
                                for v, e in zip(poly_vars, monom) if e > 0)
                            print(f"      {coeff} * {monom_str}")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("IDENTIFYING THE 3 EXTRA GENERATORS")
    print("Neural algebra (dim 119) vs Gravitational algebra (dim 116)")
    print("=" * 70)
    print(f"SymPy version: {sp.__version__}")

    t_start = time()

    # ---------------------------------------------------------------
    # Step 1: Build generators for both algebras
    # ---------------------------------------------------------------
    nn_exprs, nn_names, nn_levels = build_bracket_tree(
        NN_H12, NN_H13, NN_H23, max_level=3, label="NEURAL (gradient coupling)")

    gr_exprs, gr_names, gr_levels = build_bracket_tree(
        GR_H12, GR_H13, GR_H23, max_level=3, label="GRAVITATIONAL (1/r)")

    # Verify same tree structure
    assert len(nn_exprs) == len(gr_exprs), (
        f"Tree mismatch: {len(nn_exprs)} vs {len(gr_exprs)}")
    print(f"\n  Both algebras: {len(nn_exprs)} generators in the bracket tree")

    # ---------------------------------------------------------------
    # Step 2: Exact rank for neural algebra
    # ---------------------------------------------------------------
    nn_rank, nn_rank_adding = exact_rank_progression(
        nn_exprs, nn_names, nn_levels,
        poly_vars=NN_VARS, label="NEURAL")

    # ---------------------------------------------------------------
    # Step 3: Numerical rank for gravitational algebra
    # ---------------------------------------------------------------
    gr_rank, gr_rank_adding = numerical_rank_progression(
        gr_exprs, gr_names, gr_levels,
        n_samples=2000, seed=42, label="GRAVITATIONAL")

    # ---------------------------------------------------------------
    # Step 4: Compare and identify extras
    # ---------------------------------------------------------------
    nn_only, gr_only = compare_algebras(
        nn_rank_adding, nn_names, nn_levels,
        gr_rank_adding, gr_names, gr_levels)

    # ---------------------------------------------------------------
    # Step 5: Characterize the extra generators
    # ---------------------------------------------------------------
    if nn_only:
        characterize_generators(
            nn_only, nn_exprs, nn_names, nn_levels, NN_VARS)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Neural algebra rank:        {nn_rank}")
    print(f"  Gravitational algebra rank: {gr_rank}")
    print(f"  Extra neural generators:    {len(nn_only)}")
    if nn_only:
        print(f"  Extra generator indices:    {nn_only}")
        print(f"  Extra bracket expressions:")
        for idx in nn_only:
            print(f"    [{idx}] {nn_names[idx]}")
    if gr_only:
        print(f"  Extra gravitational generators: {len(gr_only)}")
        for idx in gr_only:
            print(f"    [{idx}] {gr_names[idx]}")
    print(f"\n  Total time: {time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
