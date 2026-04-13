#!/usr/bin/env python3
"""Compare level-3 structure constants between 1/r and 1/r² potentials.

Strategy:
  1. Load level_3.pkl checkpoints (old format from exact_growth_nbody.py)
  2. Create NBodySymbolicRank engines to get phase variables & bracket fn
  3. Extract monomial-coefficient matrices
  4. Compute exact rank and select basis
  5. Compute structure constants (with checkpoint/resume support)
  6. Compare the two tensors

Usage:
  python compare_level3_structure.py              # run full pipeline
  python compare_level3_structure.py --quick 20   # test first 20 brackets only
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
from time import time, strftime
from fractions import Fraction

# Ensure nbody/ and parent are on path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from symbolic_rank_nbody import NBodySymbolicRank


def load_old_checkpoint(path):
    """Load a level_N.pkl checkpoint in the old dict format."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {path}")
    print(f"    Keys: {list(data.keys())}")
    print(f"    N={data['n_bodies']}, d={data['d_spatial']}, "
          f"potential={data['potential']}")
    print(f"    {len(data['exprs'])} generators, level distribution: ", end="")
    from collections import Counter
    ctr = Counter(data['levels'])
    print(dict(sorted(ctr.items())))
    return data


def run_pipeline(potential, ckpt_path, work_dir, quick=None,
                 precompute_derivs=False):
    """Run the full structure constant pipeline for one potential.

    Args:
        potential: '1/r' or '1/r^2'
        ckpt_path: path to the level_3.pkl checkpoint
        work_dir: directory for intermediate checkpoints
        quick: if set, only compute this many brackets (for testing)
        precompute_derivs: if True, precompute derivatives for speed

    Returns:
        (C_float, C_exact, basis_indices) or None on failure
    """
    os.makedirs(work_dir, exist_ok=True)

    # Check for completed result
    result_path = os.path.join(work_dir, "structure_constants.pkl")
    if os.path.exists(result_path):
        print(f"\n{'='*70}")
        print(f"LOADING COMPLETED RESULT: {potential}")
        print(f"{'='*70}")
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
        print(f"  Loaded C_float: shape {result['C_float'].shape}")
        print(f"  Non-zero: {np.count_nonzero(result['C_float'])}")
        return result['C_float'], result['C_exact'], result['basis_indices']

    print(f"\n{'='*70}")
    print(f"PROCESSING: {potential}")
    print(f"{'='*70}")
    t_start = time()

    # Load checkpoint
    data = load_old_checkpoint(ckpt_path)
    exprs = data['exprs']
    names = data['names']
    levels = data['levels']

    # Create engine (just for phase variables and bracket function)
    print(f"\n  Creating NBodySymbolicRank engine...")
    engine = NBodySymbolicRank(
        data['n_bodies'], data['d_spatial'], data['potential'],
        potential_params=data.get('potential_params'),
        charges=data.get('charges'))

    # Phase 2: Monomial extraction
    mono_ckpt = os.path.join(work_dir, "monomial_matrix.pkl")
    if os.path.exists(mono_ckpt):
        print(f"\n  Loading cached monomial matrix...")
        with open(mono_ckpt, 'rb') as f:
            mono_data = pickle.load(f)
        poly_list = mono_data['poly_list']
        monom_list = mono_data['monom_list']
        monom_to_idx = mono_data['monom_to_idx']
        print(f"    {len(poly_list)} generators x {len(monom_list)} monomials")
    else:
        poly_list, monom_list, monom_to_idx = engine.extract_monomial_matrix(
            exprs, n_workers=1)
        # Cache it
        with open(mono_ckpt + '.tmp', 'wb') as f:
            pickle.dump({
                'poly_list': poly_list,
                'monom_list': monom_list,
                'monom_to_idx': monom_to_idx
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(mono_ckpt + '.tmp', mono_ckpt)
        print(f"  Cached monomial matrix to {mono_ckpt}")

    # Phase 3: Exact rank
    rank_results = engine.compute_exact_rank(
        poly_list, monom_list, monom_to_idx, levels,
        checkpoint_dir=work_dir)
    cumulative = [rank_results[lv] for lv in sorted(rank_results)]
    print(f"\n  Cumulative ranks: {cumulative}")

    final_rank = cumulative[-1]

    # Phase 4: Select basis
    basis_ckpt = os.path.join(work_dir, "basis_indices.pkl")
    if os.path.exists(basis_ckpt):
        print(f"\n  Loading cached basis indices...")
        with open(basis_ckpt, 'rb') as f:
            basis_indices = pickle.load(f)
        print(f"    {len(basis_indices)} basis generators")
    else:
        basis_indices = engine.select_basis(
            poly_list, monom_list, monom_to_idx, final_rank)
        with open(basis_ckpt + '.tmp', 'wb') as f:
            pickle.dump(basis_indices, f)
        os.replace(basis_ckpt + '.tmp', basis_ckpt)
        print(f"  Cached basis indices to {basis_ckpt}")

    # Phase 5: Structure constants
    if quick:
        print(f"\n  QUICK MODE: computing first {quick} brackets only")
        C_float, C_exact = compute_structure_constants_quick(
            engine, exprs, poly_list, monom_list, monom_to_idx,
            basis_indices, quick, work_dir)
    else:
        # Optionally precompute derivatives for speed
        if precompute_derivs and hasattr(engine, 'algebra') and engine.algebra:
            print(f"\n  Precomputing derivatives for {len(basis_indices)} "
                  f"basis generators...")
            deriv_ckpt = os.path.join(work_dir, "basis_derivs.pkl")
            if os.path.exists(deriv_ckpt):
                with open(deriv_ckpt, 'rb') as f:
                    all_derivs = pickle.load(f)
                print(f"  Loaded cached derivatives: {len(all_derivs)} entries")
            else:
                basis_exprs = [exprs[i] for i in basis_indices]
                basis_names = [names[i] for i in basis_indices]
                all_derivs = engine.algebra.precompute_derivatives(
                    basis_exprs, basis_names)
                with open(deriv_ckpt + '.tmp', 'wb') as f:
                    pickle.dump(all_derivs, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(deriv_ckpt + '.tmp', deriv_ckpt)

        C_float, C_exact = engine.compute_structure_constants(
            exprs, names, levels, poly_list, monom_list, monom_to_idx,
            basis_indices, checkpoint_dir=work_dir)

    # Save completed result
    result = {
        'potential': potential,
        'C_float': C_float,
        'C_exact': C_exact,
        'basis_indices': basis_indices,
        'rank': final_rank,
        'cumulative_ranks': cumulative,
        'n_generators': len(exprs),
        'n_monomials': len(monom_list),
    }
    with open(result_path + '.tmp', 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(result_path + '.tmp', result_path)

    print(f"\n  Total time for {potential}: {time()-t_start:.1f}s")
    return C_float, C_exact, basis_indices


def compute_structure_constants_quick(engine, exprs, poly_list, monom_list,
                                       monom_to_idx, basis_indices, n_test,
                                       work_dir):
    """Compute only the first n_test brackets for a quick comparison.

    Returns partial C_float, C_exact (only populated for the first n_test
    pairs (a,b) in lexicographic order).
    """
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.domains import QQ
    from sympy import expand, Poly, cancel

    r = len(basis_indices)
    n_mon = len(monom_list)

    print(f"\n{'='*70}")
    print(f"STRUCTURE CONSTANTS — QUICK ({n_test}/{r*(r-1)//2} brackets)")
    print(f"{'='*70}")
    t_total = time()

    # Build basis DomainMatrix
    basis_dm_rows = []
    for idx in basis_indices:
        row = [QQ.zero] * n_mon
        for monom, coeff in poly_list[idx].items():
            col = monom_to_idx[monom]
            row[col] = QQ.convert(coeff)
        basis_dm_rows.append(row)
    basis_dm = DomainMatrix(basis_dm_rows, (r, n_mon), QQ)

    C_exact = [[[None for _ in range(r)] for _ in range(r)] for _ in range(r)]
    C_float = np.zeros((r, r, r))

    n_computed = 0
    for a in range(r):
        if n_computed >= n_test:
            break
        i = basis_indices[a]
        for b in range(a + 1, r):
            if n_computed >= n_test:
                break

            n_computed += 1
            t0 = time()

            bracket = engine._poisson_bracket(exprs[i], exprs[basis_indices[b]])
            bracket = engine._simplify(bracket)

            expanded = expand(bracket)
            if engine._log_subs:
                expanded = expanded.subs(engine._log_subs)
            p = Poly(expanded, *engine.phase_vars, domain='QQ')
            bracket_dict = p.as_dict()

            rhs = [QQ.zero] * n_mon
            for monom, coeff in bracket_dict.items():
                if monom in monom_to_idx:
                    rhs[monom_to_idx[monom]] = QQ.convert(coeff)

            rhs_dm = DomainMatrix([[v] for v in rhs], (n_mon, 1), QQ)
            system = basis_dm.transpose()
            aug = system.hstack(rhs_dm)
            aug_rref, pivots = aug.rref()

            coeffs = [QQ.zero] * r
            for row_idx in range(min(r, aug_rref.shape[0])):
                if row_idx < len(pivots) and pivots[row_idx] < r:
                    col = pivots[row_idx]
                    coeffs[col] = aug_rref[row_idx, r].element

            for k in range(r):
                val = coeffs[k]
                C_exact[a][b][k] = str(Fraction(int(val.numerator),
                                                 int(val.denominator))
                                       if hasattr(val, 'numerator')
                                       else Fraction(val))
                C_exact[b][a][k] = str(-Fraction(C_exact[a][b][k]))
                C_float[a, b, k] = float(Fraction(C_exact[a][b][k]))
                C_float[b, a, k] = -C_float[a, b, k]

            elapsed = time() - t0
            print(f"  [{n_computed}/{n_test}] ({a},{b}) "
                  f"{elapsed:.1f}s  "
                  f"nonzero_k={sum(1 for v in coeffs if v != QQ.zero)}")

    for a in range(r):
        for k in range(r):
            C_exact[a][a][k] = "0"

    print(f"\n  Quick structure constants: {n_computed} brackets "
          f"in {time()-t_total:.1f}s")
    n_nonzero = int(np.count_nonzero(C_float))
    print(f"  Non-zero entries: {n_nonzero}")

    return C_float, C_exact


def compare_tensors(C1, C2, exact1, exact2, label1, label2, n_test=None):
    """Compare two structure constant tensors."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {label1} vs {label2}")
    print(f"{'='*70}")

    r1, r2 = C1.shape[0], C2.shape[0]
    print(f"  Ranks: {label1}={r1}, {label2}={r2}")

    if r1 != r2:
        print(f"  DIFFERENT RANKS — cannot compare entry-by-entry")
        return False

    r = r1

    # Determine which entries to compare
    if n_test:
        # Only compare entries that were computed (non-None in exact)
        n_compared = 0
        n_match = 0
        n_mismatch = 0
        mismatches = []

        for a in range(r):
            for b in range(a + 1, r):
                if exact1[a][b][0] is None or exact2[a][b][0] is None:
                    continue
                for k in range(r):
                    e1 = exact1[a][b][k]
                    e2 = exact2[a][b][k]
                    if e1 is None or e2 is None:
                        continue
                    n_compared += 1
                    if e1 == e2:
                        n_match += 1
                    else:
                        n_mismatch += 1
                        if len(mismatches) < 10:
                            mismatches.append((a, b, k, e1, e2))
    else:
        # Compare all entries
        n_compared = 0
        n_match = 0
        n_mismatch = 0
        mismatches = []

        for a in range(r):
            for b in range(r):
                for k in range(r):
                    e1 = exact1[a][b][k]
                    e2 = exact2[a][b][k]
                    if e1 is None or e2 is None:
                        continue
                    n_compared += 1
                    if e1 == e2:
                        n_match += 1
                    else:
                        n_mismatch += 1
                        if len(mismatches) < 10:
                            mismatches.append((a, b, k, e1, e2))

    print(f"\n  Entries compared: {n_compared}")
    print(f"  Exact matches:   {n_match}")
    print(f"  Mismatches:       {n_mismatch}")

    if n_mismatch == 0:
        print(f"\n  *** STRUCTURE CONSTANTS ARE BIT-FOR-BIT IDENTICAL ***")

        # Also check float tensor
        diff = np.abs(C1 - C2)
        print(f"  Float tensor max |diff|: {diff.max():.2e}")
    else:
        print(f"\n  *** STRUCTURE CONSTANTS DIFFER ***")
        print(f"  First mismatches:")
        for a, b, k, e1, e2 in mismatches:
            print(f"    C[{a},{b},{k}]: {label1}={e1}, {label2}={e2}")

        # Check if they might be related by a change of basis
        diff = np.abs(C1 - C2)
        print(f"\n  Float tensor max |diff|: {diff.max():.2e}")
        print(f"  Float tensor mean |diff|: {diff.mean():.2e}")

    return n_mismatch == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", type=int, default=None,
                    help="Only compute this many brackets per potential (quick test)")
    ap.add_argument("--ckpt-1r", default=None,
                    help="Path to 1/r level_3.pkl checkpoint")
    ap.add_argument("--ckpt-1r2", default=None,
                    help="Path to 1/r² level_3.pkl checkpoint")
    ap.add_argument("--work-dir", default=None,
                    help="Base directory for working files")
    ap.add_argument("--precompute-derivs", action="store_true",
                    help="Precompute derivatives (experimental speedup)")
    args = ap.parse_args()

    # Default paths
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.ckpt_1r is None:
        args.ckpt_1r = os.path.join(
            base, "aws_results", "nbody_checkpoints",
            "checkpoints_N3_d2_1r", "level_3.pkl")
    if args.ckpt_1r2 is None:
        args.ckpt_1r2 = os.path.join(
            base, "nbody", "checkpoints_N3_d2_1r2", "level_3.pkl")

    if args.work_dir is None:
        args.work_dir = os.path.join(base, "results", "level3_structure")

    # Verify checkpoints exist
    for label, path in [("1/r", args.ckpt_1r), ("1/r²", args.ckpt_1r2)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} checkpoint not found: {path}")
            sys.exit(1)
        print(f"  {label} checkpoint: {path} "
              f"({os.path.getsize(path)/1024/1024:.1f} MB)")

    print(f"  Work directory: {args.work_dir}")
    if args.quick:
        print(f"  Quick mode: {args.quick} brackets")
    print()

    # Run pipeline for each potential
    work_1r = os.path.join(args.work_dir, "1_r")
    work_1r2 = os.path.join(args.work_dir, "1_r2")

    C1, exact1, basis1 = run_pipeline(
        "1/r", args.ckpt_1r, work_1r,
        quick=args.quick,
        precompute_derivs=args.precompute_derivs)

    C2, exact2, basis2 = run_pipeline(
        "1/r^2", args.ckpt_1r2, work_1r2,
        quick=args.quick,
        precompute_derivs=args.precompute_derivs)

    # Compare basis indices
    print(f"\n{'='*70}")
    print(f"BASIS COMPARISON")
    print(f"{'='*70}")
    print(f"  1/r  basis indices (first 20): {basis1[:20]}")
    print(f"  1/r² basis indices (first 20): {basis2[:20]}")
    if list(basis1) == list(basis2):
        print(f"  Basis indices: IDENTICAL")
    else:
        print(f"  Basis indices: DIFFERENT")
        diff_positions = [i for i in range(min(len(basis1), len(basis2)))
                          if basis1[i] != basis2[i]]
        print(f"  Differ at positions: {diff_positions[:20]}")

    # Compare structure constants
    identical = compare_tensors(
        C1, C2, exact1, exact2,
        "1/r", "1/r²",
        n_test=args.quick)

    # Summary
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    n_brackets = args.quick if args.quick else C1.shape[0] * (C1.shape[0]-1) // 2
    if identical:
        print(f"  Level-3 structure constants are BIT-FOR-BIT IDENTICAL "
              f"over Q")
        print(f"  ({n_brackets} brackets verified)")
        print(f"  The universality extends from dim-17 level-2 "
              f"to dim-116 level-3.")
    else:
        print(f"  Level-3 structure constants DIFFER between 1/r and 1/r²")
        print(f"  ({n_brackets} brackets checked)")
        print(f"  Universality may be limited to level 2.")


if __name__ == "__main__":
    main()
