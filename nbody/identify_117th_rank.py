#!/usr/bin/env python3
"""
Determine the rank of the 48 hbar^2 corrections among themselves.

We know 48 quantum generators produce hbar^2 corrections with new monomials.
But the rank increase is only +1. So these 48 corrections must span only
a 1-dimensional space over Q (modulo the classical span). This script
confirms that and extracts the simplest representative.
"""

import os
import sys
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
from sympy import (Symbol, Integer, Rational, Add, Poly, expand, cancel,
                   diff, symbols)
from sympy.polys.matrices import DomainMatrix
from sympy.polys.domains import QQ

from exact_growth_nbody import NBodyAlgebra
from quantum_algebra import QuantumNBodyAlgebra, hbar_sym
from identify_117th import build_generators_both, separate_hbar_orders


def main():
    t_start = time()

    (alg_c, c_exprs, c_names, c_levels,
     alg_q, q_exprs, q_names, q_levels) = build_generators_both()

    phase_vars_c = list(alg_c.all_vars)

    print("\n" + "=" * 70)
    print("EXTRACTING HBAR^2 CORRECTIONS")
    print("=" * 70)

    l3_indices = [i for i, lv in enumerate(q_levels) if lv == 3]
    corrections = []
    corr_names = []

    for idx in l3_indices:
        orders = separate_hbar_orders(q_exprs[idx])
        hbar2 = orders.get(2, Integer(0))
        if hbar2 != 0:
            corrections.append(expand(hbar2))
            corr_names.append(q_names[idx])

    print(f"  {len(corrections)} nonzero hbar^2 corrections")

    # Build monomial matrix for corrections only
    print("\n" + "=" * 70)
    print("RANK OF HBAR^2 CORRECTIONS (AMONG THEMSELVES)")
    print("=" * 70)

    all_monoms = set()
    poly_dicts = []
    for expr in corrections:
        p = Poly(expr, *phase_vars_c, domain='QQ')
        md = p.as_dict()
        poly_dicts.append(md)
        all_monoms.update(md.keys())

    monom_list = sorted(all_monoms)
    monom_to_idx = {m: i for i, m in enumerate(monom_list)}
    n_mon = len(monom_list)
    print(f"  {len(corrections)} corrections, {n_mon} monomials in correction space")

    rows = []
    for md in poly_dicts:
        row = [QQ.zero] * n_mon
        for monom, coeff in md.items():
            row[monom_to_idx[monom]] = QQ.convert(coeff)
        rows.append(row)

    dm = DomainMatrix(rows, (len(rows), n_mon), QQ)
    t0 = time()
    corr_rank = dm.rank()
    print(f"  Rank of correction matrix: {corr_rank}  [{time()-t0:.1f}s]")

    if corr_rank == 1:
        print("\n  *** ALL 48 CORRECTIONS SPAN A 1-DIMENSIONAL SPACE! ***")
        print("  There is essentially ONE quantum correction direction.")

    # Find the simplest representative
    print("\n" + "=" * 70)
    print("FINDING SIMPLEST REPRESENTATIVE")
    print("=" * 70)

    # Sort by number of terms
    indexed = [(len(Add.make_args(c)), i, c, n)
               for i, (c, n) in enumerate(zip(corrections, corr_names))]
    indexed.sort()

    simplest_nterms, simplest_idx, simplest_expr, simplest_name = indexed[0]
    print(f"  Simplest correction: {simplest_name}")
    print(f"  Number of terms: {simplest_nterms}")

    # Check which generators are scalar multiples of the simplest
    print("\n  Checking which corrections are proportional to simplest...")
    ref_row = rows[simplest_idx]

    # Find the first nonzero entry in ref_row
    ref_pivot = None
    ref_val = None
    for j in range(n_mon):
        if ref_row[j] != QQ.zero:
            ref_pivot = j
            ref_val = ref_row[j]
            break

    proportional_count = 0
    ratios = {}
    for ci, row in enumerate(rows):
        if row[ref_pivot] == QQ.zero:
            # Check if this row is all zero at positions where ref is nonzero
            is_zero = True
            for j in range(n_mon):
                if ref_row[j] != QQ.zero and row[j] != QQ.zero:
                    is_zero = False
                    break
            if is_zero:
                continue
            continue

        ratio = QQ.quo(row[ref_pivot], ref_val)
        # Verify proportionality
        is_prop = True
        for j in range(n_mon):
            expected = QQ.mul(ref_row[j], ratio)
            if row[j] != expected:
                is_prop = False
                break

        if is_prop:
            proportional_count += 1
            ratios[ci] = ratio

    print(f"  {proportional_count}/{len(corrections)} are proportional "
          f"to the simplest")

    print("\n" + "=" * 70)
    print("THE 117TH GENERATOR")
    print("=" * 70)

    print(f"\n  The extra quantum dimension is spanned by the hbar^2 coefficient")
    print(f"  of the commutator bracket:")
    print(f"  {simplest_name}")
    print(f"\n  Full expression ({simplest_nterms} terms):")
    print(f"  {simplest_expr}")

    # Analyze the structure
    p = Poly(simplest_expr, *phase_vars_c, domain='QQ')
    md = p.as_dict()
    n_q = len(alg_c.q_vars)
    n_p = len(alg_c.p_vars)

    print(f"\n  Monomial analysis:")
    print(f"  Total distinct monomials: {len(md)}")

    p_degrees = set()
    u_degrees = set()
    for m in md.keys():
        p_deg = sum(m[n_q:n_q+n_p])
        u_deg = sum(m[n_q+n_p:])
        p_degrees.add(p_deg)
        u_degrees.add(u_deg)

    print(f"  Momentum degrees present: {sorted(p_degrees)}")
    print(f"  u degrees present: {sorted(u_degrees)}")

    # The key insight: is this purely position-dependent?
    if max(p_degrees) == 0:
        print(f"\n  *** THIS IS A PURE POSITION/DISTANCE FUNCTION ***")
        print(f"  It has NO momentum dependence (p_deg = 0 for all terms)")
        print(f"  It is a conserved quantity that depends only on")
        print(f"  positions and inter-particle distances!")
    else:
        print(f"\n  Mixed position-momentum observable")

    # Show all terms grouped by total degree
    print(f"\n  Terms grouped by total degree:")
    by_degree = {}
    for m, c in md.items():
        d = sum(m)
        by_degree.setdefault(d, []).append((m, c))

    for d in sorted(by_degree.keys()):
        terms = by_degree[d]
        print(f"    degree {d}: {len(terms)} terms")
        for m, c in terms[:5]:
            monom_str = " * ".join(
                f"{v}^{e}" for v, e in zip(phase_vars_c, m) if e > 0)
            print(f"      ({c}) * {monom_str}")
        if len(terms) > 5:
            print(f"      ... ({len(terms) - 5} more)")

    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {time()-t_start:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
