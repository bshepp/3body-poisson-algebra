#!/usr/bin/env python3
"""
Symbolic Rank Over Q — Exact Algebraic Dimension of the Poisson Algebra
========================================================================

Computes the exact rank of the generator set over the rationals (or over
the fraction field Q(m1,m2,m3) for symbolic masses), with NO numerical
approximation: no sampling, no SVD, no thresholds.

The generators are polynomials in 15 phase-space variables.  We extract
the monomial-coefficient matrix and compute its rank via exact Gaussian
elimination.

Usage
-----
    # Equal mass (baseline)
    python symbolic_rank.py

    # Specific rational masses
    python symbolic_rank.py --masses 1 2 3

    # Symbolic masses (proves mass invariance for ALL positive masses)
    python symbolic_rank.py --symbolic

    # Extreme ratio
    python symbolic_rank.py --masses 1 1 1/10000
"""

import os
import sys
import json
import argparse
from time import time
from fractions import Fraction

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)

import sympy as sp
from sympy import symbols, Rational, Add, Poly, Symbol

from exact_growth import (
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    poisson_bracket, simplify_generator,
    build_hamiltonians,
)


def build_generators(masses, potential_type='1/r'):
    """Build all 156 generators through level 3 from scratch.

    Returns (exprs, names, levels).
    """
    print(f"Building generators for masses={masses}, potential={potential_type}")
    t_total = time()

    h12, h13, h23 = build_hamiltonians(potential_type, masses=masses)

    all_exprs = [h12, h13, h23]
    all_names = ["H12", "H13", "H23"]
    all_levels = [0, 0, 0]
    computed_pairs = {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}

    # Level 1
    print("\n--- Level 1 ---")
    pairs_l1 = [
        ("K1", "{H12,H13}", 0, 1),
        ("K2", "{H12,H23}", 0, 2),
        ("K3", "{H13,H23}", 1, 2),
    ]
    for short, full, i, j in pairs_l1:
        t0 = time()
        expr = poisson_bracket(all_exprs[i], all_exprs[j])
        expr = simplify_generator(expr)
        nterms = len(Add.make_args(expr))
        print(f"  {full} -> {nterms} terms [{time()-t0:.1f}s]")
        all_exprs.append(expr)
        all_names.append(short)
        all_levels.append(1)

    # Levels 2-3
    for level in range(2, 4):
        print(f"\n--- Level {level} ---")
        t_level = time()
        frontier = [i for i, lv in enumerate(all_levels) if lv == level - 1]
        n_existing = len(all_exprs)
        new_exprs = []
        new_names = []
        n_cand = 0

        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)
                n_cand += 1

                t0 = time()
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                t_b = time() - t0
                t0 = time()
                expr = simplify_generator(expr)
                t_s = time() - t0

                nterms = len(Add.make_args(expr))
                ni, nj = all_names[i], all_names[j]
                bname = f"{{{ni},{nj}}}"
                print(f"  [{n_cand:>4d}] {bname}  "
                      f"bracket {t_b:.1f}s  simplify {t_s:.1f}s  "
                      f"-> {nterms} terms")

                new_exprs.append(expr)
                new_names.append(bname)

        for expr, name in zip(new_exprs, new_names):
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(level)

        print(f"  Level {level}: {len(new_exprs)} candidates "
              f"in {time()-t_level:.1f}s")

    print(f"\nTotal generators: {len(all_exprs)} "
          f"in {time()-t_total:.1f}s")
    return all_exprs, all_names, all_levels


def extract_monomial_matrix(exprs, phase_vars, domain_spec):
    """Extract the monomial-coefficient matrix from a list of SymPy exprs.

    Each expression is converted to a Poly over phase_vars.  The union of
    all monomials forms the column set.  Returns (matrix_rows, monomial_list)
    where matrix_rows[i] is a dict {monomial_index: coefficient}.
    """
    print("\nExtracting monomial-coefficient matrix...")
    t0 = time()

    all_monoms = set()
    poly_list = []

    for idx, expr in enumerate(exprs):
        expanded = sp.expand(expr)
        p = Poly(expanded, *phase_vars, domain=domain_spec)
        monom_dict = p.as_dict()
        poly_list.append(monom_dict)
        all_monoms.update(monom_dict.keys())
        if (idx + 1) % 20 == 0 or idx == len(exprs) - 1:
            print(f"  Processed {idx+1}/{len(exprs)} generators, "
                  f"{len(all_monoms)} distinct monomials so far")

    monom_list = sorted(all_monoms)
    monom_to_idx = {m: i for i, m in enumerate(monom_list)}

    n_gen = len(exprs)
    n_mon = len(monom_list)
    n_nonzero = sum(len(d) for d in poly_list)
    density = n_nonzero / (n_gen * n_mon) if n_gen * n_mon > 0 else 0

    print(f"  Matrix dimensions: {n_gen} x {n_mon}")
    print(f"  Non-zero entries: {n_nonzero}")
    print(f"  Density: {density:.4f}")
    print(f"  Extraction time: {time()-t0:.1f}s")

    return poly_list, monom_list, monom_to_idx


def compute_exact_rank(poly_list, monom_list, monom_to_idx, levels,
                       domain_ring):
    """Compute exact rank using SymPy DomainMatrix for performance.

    Reports cumulative rank at each level.
    """
    from sympy.polys.matrices import DomainMatrix

    n_mon = len(monom_list)
    results = {}

    for max_lv in range(max(levels) + 1):
        mask = [i for i, lv in enumerate(levels) if lv <= max_lv]
        n_sel = len(mask)

        print(f"\n  Rank through level {max_lv} "
              f"({n_sel} generators x {n_mon} monomials)...",
              end=" ", flush=True)
        t0 = time()

        rows = []
        for i in mask:
            row = [domain_ring.zero] * n_mon
            for monom, coeff in poly_list[i].items():
                col = monom_to_idx[monom]
                row[col] = domain_ring.convert(coeff)
            rows.append(row)

        dm = DomainMatrix(rows, (n_sel, n_mon), domain_ring)
        rank = dm.rank()
        elapsed = time() - t0

        print(f"rank = {rank}  [{elapsed:.1f}s]")
        results[max_lv] = rank

    return results


def parse_mass(s):
    """Parse a mass value: integer, float, or fraction like '1/10000'."""
    if '/' in s:
        num, den = s.split('/')
        return Rational(int(num), int(den))
    try:
        return Rational(int(s))
    except ValueError:
        return Rational(s).limit_denominator(10**12)


def main():
    ap = argparse.ArgumentParser(
        description="Exact algebraic rank of the Poisson algebra")
    ap.add_argument("--masses", nargs=3, default=None,
                    help="Three mass values (e.g. 1 2 3 or 1 1 1/10000)")
    ap.add_argument("--symbolic", action="store_true",
                    help="Use symbolic masses m1,m2,m3 (proves mass invariance)")
    ap.add_argument("--potential", default="1/r",
                    help="Potential type (default: 1/r)")
    ap.add_argument("--output", default=None,
                    help="Output JSON file (default: auto-named)")
    args = ap.parse_args()

    print("=" * 70)
    print("SYMBOLIC RANK OVER Q")
    print("=" * 70)
    print(f"SymPy version: {sp.__version__}")
    print(f"Potential: {args.potential}")

    t_grand = time()

    if args.symbolic:
        m1s, m2s, m3s = symbols("m1 m2 m3", positive=True)
        masses = (m1s, m2s, m3s)
        mass_label = "symbolic"
        phase_vars = list(ALL_VARS)
        from sympy.polys.domains import QQ
        domain_spec = QQ.frac_field(m1s, m2s, m3s)
        domain_ring = domain_spec
        print(f"Masses: SYMBOLIC (m1, m2, m3)")
        print(f"Domain: Q(m1, m2, m3)")
    else:
        if args.masses:
            masses = tuple(parse_mass(m) for m in args.masses)
        else:
            masses = (Rational(1), Rational(1), Rational(1))
        mass_label = [str(m) for m in masses]
        phase_vars = list(ALL_VARS)
        from sympy.polys.domains import QQ
        domain_spec = 'QQ'
        domain_ring = QQ
        print(f"Masses: {masses}")
        print(f"Domain: Q (rationals)")

    # Phase 1: Build generators
    exprs, names, levels = build_generators(masses, args.potential)

    # Phase 2: Extract monomial-coefficient matrix
    poly_list, monom_list, monom_to_idx = extract_monomial_matrix(
        exprs, phase_vars, domain_spec)

    # Phase 3: Exact rank
    print("\n" + "=" * 70)
    print("EXACT RANK COMPUTATION")
    print("=" * 70)

    rank_results = compute_exact_rank(
        poly_list, monom_list, monom_to_idx, levels, domain_ring)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    cumulative = [rank_results[lv] for lv in sorted(rank_results)]
    print(f"  Cumulative rank: {cumulative}")
    new_per_level = [cumulative[0]]
    for i in range(1, len(cumulative)):
        new_per_level.append(cumulative[i] - cumulative[i-1])
    print(f"  New per level:   {new_per_level}")
    print(f"  Total generators: {len(exprs)}")
    print(f"  Total monomials:  {len(monom_list)}")
    total_time = time() - t_grand
    print(f"  Total time: {total_time:.1f}s")

    expected = [3, 6, 17, 116]
    match = cumulative == expected
    print(f"\n  Expected: {expected}")
    print(f"  MATCH: {'YES' if match else 'NO'}")

    # Save results
    output = {
        "masses": mass_label,
        "potential": args.potential,
        "n_generators": len(exprs),
        "n_monomials": len(monom_list),
        "cumulative_rank": cumulative,
        "new_per_level": new_per_level,
        "matrix_density": sum(len(d) for d in poly_list)
            / (len(exprs) * len(monom_list)),
        "computation_time_seconds": round(total_time, 1),
        "sympy_version": sp.__version__,
        "match_expected": match,
    }

    if args.output:
        out_path = args.output
    else:
        if args.symbolic:
            tag = "symbolic"
        else:
            tag = "_".join(str(m).replace("/", "over") for m in masses)
        out_dir = "results/symbolic_rank"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"rank_{tag}.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
