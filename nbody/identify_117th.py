#!/usr/bin/env python3
"""
Identify the 117th Generator
=============================

The quantum commutator algebra for N=3, d=1, 1/r has rank 117 over QQ[hbar],
while the classical Poisson algebra has rank 116. This script identifies the
extra generator: the specific hbar-dependent combination that is linearly
independent of all classical generators.

Strategy:
1. Build all 156 generators in both classical and quantum mode (d=1 for speed).
2. Extract monomial-coefficient matrices over QQ (classical) and QQ[hbar] (quantum).
3. For each quantum generator, decompose it as:
       g_quantum = g_classical_part + hbar^2 * g_correction
4. Find which level-3 generators contribute the extra dimension by checking
   whether the hbar^2 correction terms are in the span of classical generators.
5. Extract and simplify the 117th generator.
"""

import os
import sys
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
from sympy import (Symbol, Integer, Rational, Add, Poly, expand, cancel,
                   diff, symbols, degree)
from sympy.polys.matrices import DomainMatrix
from sympy.polys.domains import QQ

from exact_growth_nbody import NBodyAlgebra
from quantum_algebra import QuantumNBodyAlgebra, hbar_sym


def build_generators_both(n_bodies=3, d_spatial=1, max_level=3):
    """Build generators in both classical and quantum mode."""

    print("=" * 70)
    print("BUILDING CLASSICAL GENERATORS")
    print("=" * 70)
    t0 = time()

    alg_c = NBodyAlgebra(n_bodies, d_spatial, "1/r")
    c_exprs = list(alg_c.hamiltonian_list)
    c_names = list(alg_c.hamiltonian_names)
    n_l0 = len(c_exprs)
    c_levels = [0] * n_l0
    computed_pairs_c = set()

    for i in range(n_l0):
        for j in range(i + 1, n_l0):
            computed_pairs_c.add(frozenset({i, j}))

    # Level 1
    for i in range(n_l0):
        for j in range(i + 1, n_l0):
            expr = alg_c.poisson_bracket(c_exprs[i], c_exprs[j])
            expr = cancel(expr)
            c_exprs.append(expr)
            c_names.append(f"{{{c_names[i]},{c_names[j]}}}")
            c_levels.append(1)

    # Levels 2+
    for level in range(2, max_level + 1):
        frontier = [i for i, lv in enumerate(c_levels) if lv == level - 1]
        n_existing = len(c_exprs)
        new_c = []
        new_n = []
        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs_c:
                    continue
                computed_pairs_c.add(pair)
                expr = alg_c.poisson_bracket(c_exprs[i], c_exprs[j])
                expr = cancel(expr)
                new_c.append(expr)
                new_n.append(f"{{{c_names[i]},{c_names[j]}}}")
        for expr, name in zip(new_c, new_n):
            c_exprs.append(expr)
            c_names.append(name)
            c_levels.append(level)

    print(f"  Classical: {len(c_exprs)} generators in {time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print("BUILDING QUANTUM GENERATORS")
    print("=" * 70)
    t0 = time()

    alg_q = QuantumNBodyAlgebra(n_bodies, d_spatial, "1/r")
    q_exprs = list(alg_q.hamiltonian_list)
    q_names = list(alg_q.hamiltonian_names)
    q_levels = [0] * n_l0
    computed_pairs_q = set()

    for i in range(n_l0):
        for j in range(i + 1, n_l0):
            computed_pairs_q.add(frozenset({i, j}))

    # Level 1
    for i in range(n_l0):
        for j in range(i + 1, n_l0):
            expr = alg_q.commutator(q_exprs[i], q_exprs[j])
            expr = cancel(expr)
            q_exprs.append(expr)
            q_names.append(f"{{{q_names[i]},{q_names[j]}}}")
            q_levels.append(1)

    # Levels 2+
    for level in range(2, max_level + 1):
        print(f"  Level {level}...")
        frontier = [i for i, lv in enumerate(q_levels) if lv == level - 1]
        n_existing = len(q_exprs)
        new_q = []
        new_n = []
        count = 0
        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs_q:
                    continue
                computed_pairs_q.add(pair)
                count += 1
                t1 = time()
                expr = alg_q.commutator(q_exprs[i], q_exprs[j])
                expr = cancel(expr)
                elapsed = time() - t1
                if count % 20 == 0:
                    nterms = len(Add.make_args(expr))
                    print(f"    [{count}] {elapsed:.1f}s, {nterms} terms")
                new_q.append(expr)
                new_n.append(f"{{{q_names[i]},{q_names[j]}}}")
        for expr, name in zip(new_q, new_n):
            q_exprs.append(expr)
            q_names.append(name)
            q_levels.append(level)
        print(f"    Level {level}: {len(new_q)} generators")

    print(f"  Quantum: {len(q_exprs)} generators in {time()-t0:.1f}s")

    return (alg_c, c_exprs, c_names, c_levels,
            alg_q, q_exprs, q_names, q_levels)


def separate_hbar_orders(expr):
    """Separate an expression into orders of hbar.

    Returns dict: {power_of_hbar: coefficient_expression}
    e.g., {0: classical_part, 2: hbar2_correction, ...}
    """
    expanded = expand(expr)
    terms = Add.make_args(expanded)

    orders = {}
    for term in terms:
        p = 0
        coeff = term
        if term.has(hbar_sym):
            p_obj = Poly(term, hbar_sym)
            monoms = p_obj.as_dict()
            for (power,), c in monoms.items():
                orders[power] = orders.get(power, Integer(0)) + c
        else:
            orders[0] = orders.get(0, Integer(0)) + term

    return orders


def find_117th(alg_c, c_exprs, c_names, c_levels,
               alg_q, q_exprs, q_names, q_levels):
    """Find the 117th generator by comparing classical and quantum spans."""

    phase_vars_c = list(alg_c.all_vars)
    phase_vars_q = list(alg_q.all_vars)  # includes hbar

    print("\n" + "=" * 70)
    print("STEP 1: EXTRACT HBAR CORRECTIONS FROM QUANTUM GENERATORS")
    print("=" * 70)

    # For each quantum generator at level 3, decompose into hbar orders
    l3_indices = [i for i, lv in enumerate(q_levels) if lv == 3]
    print(f"  {len(l3_indices)} level-3 generators")

    hbar_corrections = []
    hbar_correction_names = []
    hbar_correction_indices = []

    for idx in l3_indices:
        expr = q_exprs[idx]
        orders = separate_hbar_orders(expr)

        # The quantum generator is: classical_part + hbar^2 * correction + ...
        # The "correction" part is what's new
        for power, coeff in sorted(orders.items()):
            if power >= 2 and coeff != 0:
                hbar_corrections.append(coeff)
                hbar_correction_names.append(
                    f"hbar^{power}_part_of_{q_names[idx]}")
                hbar_correction_indices.append((idx, power))

    print(f"  Found {len(hbar_corrections)} nonzero hbar corrections")

    # Show which hbar powers appear
    powers_seen = set(p for _, p in hbar_correction_indices)
    print(f"  hbar powers present: {sorted(powers_seen)}")

    # Count by power
    for p in sorted(powers_seen):
        n = sum(1 for _, pp in hbar_correction_indices if pp == p)
        print(f"    hbar^{p}: {n} generators have nonzero correction")

    print("\n" + "=" * 70)
    print("STEP 2: BUILD CLASSICAL MONOMIAL MATRIX")
    print("=" * 70)
    t0 = time()

    # Extract monomial-coefficient matrix for classical generators
    c_monoms_all = set()
    c_poly_list = []
    for expr in c_exprs:
        expanded = expand(expr)
        p = Poly(expanded, *phase_vars_c, domain='QQ')
        md = p.as_dict()
        c_poly_list.append(md)
        c_monoms_all.update(md.keys())

    c_monom_list = sorted(c_monoms_all)
    c_monom_to_idx = {m: i for i, m in enumerate(c_monom_list)}
    n_c_mon = len(c_monom_list)
    print(f"  Classical: {len(c_exprs)} generators, {n_c_mon} monomials [{time()-t0:.1f}s]")

    # Build DomainMatrix for classical generators
    c_rows = []
    for md in c_poly_list:
        row = [QQ.zero] * n_c_mon
        for monom, coeff in md.items():
            row[c_monom_to_idx[monom]] = QQ.convert(coeff)
        c_rows.append(row)

    c_dm = DomainMatrix(c_rows, (len(c_exprs), n_c_mon), QQ)
    c_rank = c_dm.rank()
    print(f"  Classical rank: {c_rank}")

    print("\n" + "=" * 70)
    print("STEP 3: CHECK WHICH HBAR CORRECTIONS ARE OUTSIDE CLASSICAL SPAN")
    print("=" * 70)
    t0 = time()

    # For each hbar^2 correction, check if it's in the span of classical generators
    independent_corrections = []
    augmented_rows = list(c_rows)  # copy

    for ci, (corr_expr, corr_name) in enumerate(
            zip(hbar_corrections, hbar_correction_names)):
        expanded = expand(corr_expr)
        try:
            p = Poly(expanded, *phase_vars_c, domain='QQ')
            md = p.as_dict()
        except Exception as e:
            print(f"  [{ci}] {corr_name}: failed to poly-ify: {e}")
            continue

        # Check if this correction introduces new monomials
        new_monoms = set(md.keys()) - c_monoms_all
        if new_monoms:
            print(f"  [{ci}] {corr_name}: {len(new_monoms)} NEW monomials! "
                  f"(not in classical basis)")
            # This is automatically independent
            independent_corrections.append((ci, corr_name, corr_expr, new_monoms))
            continue

        # Correction uses only classical monomials — check linear independence
        corr_row = [QQ.zero] * n_c_mon
        for monom, coeff in md.items():
            corr_row[c_monom_to_idx[monom]] = QQ.convert(coeff)

        trial_rows = augmented_rows + [corr_row]
        trial_dm = DomainMatrix(trial_rows,
                                (len(trial_rows), n_c_mon), QQ)
        new_rank = trial_dm.rank()

        if new_rank > c_rank + len(independent_corrections):
            print(f"  [{ci}] {corr_name}: INDEPENDENT! "
                  f"(rank {new_rank} > {c_rank + len(independent_corrections)})")
            independent_corrections.append((ci, corr_name, corr_expr, None))
            augmented_rows.append(corr_row)
        else:
            pass  # in the span, not interesting

    print(f"\n  Found {len(independent_corrections)} independent hbar corrections "
          f"[{time()-t0:.1f}s]")

    if not independent_corrections:
        print("\n  WARNING: No independent corrections found via this method.")
        print("  The extra dimension may come from the *full* quantum generators")
        print("  (classical + hbar correction together) being independent in")
        print("  a way that the separated parts are not.")
        print("\n  Trying alternative: direct comparison of full quantum generators...")
        return find_117th_direct(alg_c, c_exprs, c_names, c_levels,
                                 alg_q, q_exprs, q_names, q_levels,
                                 phase_vars_c, c_monom_list, c_monom_to_idx,
                                 c_rows, c_rank)

    print("\n" + "=" * 70)
    print("STEP 4: ANALYZE THE EXTRA GENERATOR(S)")
    print("=" * 70)

    for ci, name, expr, new_monoms in independent_corrections:
        idx, power = hbar_correction_indices[ci]
        print(f"\n  Generator: {name}")
        print(f"  Source: quantum generator #{idx} = {q_names[idx]}")
        print(f"  hbar power: {power}")

        nterms = len(Add.make_args(expand(expr)))
        print(f"  Number of terms: {nterms}")

        if new_monoms:
            print(f"  Contains {len(new_monoms)} new monomials not in classical set:")
            for m in sorted(new_monoms)[:10]:
                monom_str = " * ".join(
                    f"{v}^{e}" for v, e in zip(phase_vars_c, m) if e > 0)
                print(f"    {monom_str}")
            if len(new_monoms) > 10:
                print(f"    ... ({len(new_monoms) - 10} more)")

        # Show the expression if compact enough
        if nterms <= 30:
            print(f"\n  Expression:")
            print(f"    {expand(expr)}")
        else:
            # Show a few terms
            terms = Add.make_args(expand(expr))
            print(f"\n  First 5 terms:")
            for t in terms[:5]:
                print(f"    {t}")
            print(f"  ... ({nterms - 5} more terms)")

        # Analyze monomial structure
        try:
            p = Poly(expand(expr), *phase_vars_c, domain='QQ')
            md = p.as_dict()
            max_deg = max(sum(m) for m in md.keys())
            max_p_deg = 0
            for m in md.keys():
                n_q = len(alg_c.q_vars)
                p_deg = sum(m[n_q:n_q + len(alg_c.p_vars)])
                max_p_deg = max(max_p_deg, p_deg)
            print(f"\n  Max total degree: {max_deg}")
            print(f"  Max momentum degree: {max_p_deg}")
        except Exception:
            pass

    return independent_corrections


def find_117th_direct(alg_c, c_exprs, c_names, c_levels,
                       alg_q, q_exprs, q_names, q_levels,
                       phase_vars_c, c_monom_list, c_monom_to_idx,
                       c_rows, c_rank):
    """Alternative: find 117th by testing full quantum generators (at hbar=1)
    against the classical span."""

    print("\n" + "=" * 70)
    print("ALTERNATIVE: EVALUATE QUANTUM GENERATORS AT hbar=1")
    print("=" * 70)

    # Substitute hbar=1 in quantum generators and check against classical span
    l3_q_indices = [i for i, lv in enumerate(q_levels) if lv == 3]
    n_c_mon = len(c_monom_list)

    augmented_rows = list(c_rows)
    current_rank = c_rank
    found = []

    for qi, idx in enumerate(l3_q_indices):
        q_expr = q_exprs[idx]
        # Substitute hbar -> 1 to get a rational expression
        q_at_1 = expand(q_expr.subs(hbar_sym, 1))

        try:
            p = Poly(q_at_1, *phase_vars_c, domain='QQ')
            md = p.as_dict()
        except Exception:
            continue

        # Check for new monomials
        new_monoms = set(md.keys()) - set(c_monom_to_idx.keys())
        if new_monoms:
            print(f"  [{qi}] {q_names[idx]}: {len(new_monoms)} new monomials "
                  f"(automatically independent)")
            found.append((idx, q_names[idx], q_expr, new_monoms))
            if len(found) >= 3:
                break
            continue

        q_row = [QQ.zero] * n_c_mon
        for monom, coeff in md.items():
            if monom in c_monom_to_idx:
                q_row[c_monom_to_idx[monom]] = QQ.convert(coeff)

        trial = augmented_rows + [q_row]
        trial_dm = DomainMatrix(trial, (len(trial), n_c_mon), QQ)
        new_rank = trial_dm.rank()

        if new_rank > current_rank:
            print(f"  [{qi}] {q_names[idx]}: INDEPENDENT at hbar=1! "
                  f"(rank {new_rank})")
            found.append((idx, q_names[idx], q_expr, None))
            augmented_rows.append(q_row)
            current_rank = new_rank
            if current_rank >= c_rank + 1:
                break

    if found:
        print(f"\n  Found {len(found)} independent quantum generators")
        for idx, name, expr, new_monoms in found:
            analyze_quantum_generator(expr, name, idx,
                                      alg_c, alg_q, phase_vars_c,
                                      q_exprs, q_names, q_levels)
    else:
        print("\n  No independent generators found at hbar=1 either.")
        print("  The extra dimension must arise from the QQ[hbar] domain structure.")
        print("  Trying: check each quantum generator for hbar-dependent ")
        print("  independence (not just hbar=1)...")
        find_117th_over_qhbar(alg_q, q_exprs, q_names, q_levels,
                               alg_c, c_exprs, c_levels, phase_vars_c)

    return found


def find_117th_over_qhbar(alg_q, q_exprs, q_names, q_levels,
                           alg_c, c_exprs, c_levels, phase_vars_c):
    """Find the 117th by working directly over QQ[hbar]."""

    print("\n" + "=" * 70)
    print("DIRECT COMPARISON OVER QQ[hbar]")
    print("=" * 70)

    phase_vars_q = list(alg_q.all_vars)
    domain = QQ[hbar_sym]

    # Build monomial set from all quantum generators
    all_monoms = set()
    q_poly_list = []
    for expr in q_exprs:
        expanded = expand(expr)
        p = Poly(expanded, *phase_vars_q, domain='QQ')
        md = p.as_dict()
        q_poly_list.append(md)
        all_monoms.update(md.keys())

    monom_list = sorted(all_monoms)
    monom_to_idx = {m: i for i, m in enumerate(monom_list)}
    n_mon = len(monom_list)

    print(f"  {len(q_exprs)} generators, {n_mon} monomials")

    # Build the full quantum matrix over QQ[hbar]
    q_rows = []
    for md in q_poly_list:
        row = [domain.zero] * n_mon
        for monom, coeff in md.items():
            row[monom_to_idx[monom]] = domain.convert(coeff)
        q_rows.append(row)

    # Also build the classical generators in the same monomial basis
    # Classical generators are exactly the hbar=0 part of quantum,
    # but we rebuild from the classical expressions for rigor
    c_poly_list = []
    for expr in c_exprs:
        expanded = expand(expr)
        # pad the monomial tuple with 0 for the hbar slot
        p = Poly(expanded, *phase_vars_c, domain='QQ')
        md_orig = p.as_dict()
        # Extend monomials: classical vars are phase_vars_c, quantum has +hbar
        # The hbar index is the last one in phase_vars_q
        md = {}
        for monom, coeff in md_orig.items():
            extended = monom + (0,)  # hbar^0
            md[extended] = coeff
        c_poly_list.append(md)

    c_rows = []
    for md in c_poly_list:
        row = [domain.zero] * n_mon
        for monom, coeff in md.items():
            if monom in monom_to_idx:
                row[monom_to_idx[monom]] = domain.convert(coeff)
        c_rows.append(row)

    # Get classical rank
    c_dm = DomainMatrix(c_rows, (len(c_rows), n_mon), domain)
    c_rank = c_dm.rank()
    print(f"  Classical rank (in quantum monomial basis): {c_rank}")

    # Now check quantum level-3 generators one by one
    l3_q = [i for i, lv in enumerate(q_levels) if lv == 3]
    augmented = list(c_rows)
    cur_rank = c_rank

    for qi, idx in enumerate(l3_q):
        trial = augmented + [q_rows[idx]]
        dm = DomainMatrix(trial, (len(trial), n_mon), domain)
        new_rank = dm.rank()

        if new_rank > cur_rank:
            print(f"\n  FOUND: quantum generator #{idx} = {q_names[idx]}")
            print(f"    rank jumps from {cur_rank} to {new_rank}")
            augmented.append(q_rows[idx])
            cur_rank = new_rank

            # Analyze this generator
            analyze_quantum_generator(q_exprs[idx], q_names[idx], idx,
                                      alg_c, alg_q, phase_vars_c,
                                      q_exprs, q_names, q_levels)

            if cur_rank >= c_rank + 1:
                print(f"\n  Reached rank {cur_rank} = classical + 1. Done.")
                break

    print(f"\n  Final quantum rank: {cur_rank}")


def analyze_quantum_generator(expr, name, idx, alg_c, alg_q, phase_vars_c,
                               q_exprs, q_names, q_levels):
    """Analyze a quantum generator that contributes the extra dimension."""

    print(f"\n  {'='*60}")
    print(f"  ANALYSIS OF THE 117TH GENERATOR")
    print(f"  {'='*60}")
    print(f"  Name: {name}")
    print(f"  Index: {idx}")

    # Separate into hbar orders
    orders = separate_hbar_orders(expr)
    print(f"\n  hbar decomposition:")
    for power in sorted(orders.keys()):
        coeff = orders[power]
        nterms = len(Add.make_args(expand(coeff)))
        print(f"    hbar^{power}: {nterms} terms")

    # The classical part (hbar^0)
    classical_part = orders.get(0, Integer(0))
    print(f"\n  Classical part (hbar^0): {len(Add.make_args(expand(classical_part)))} terms")

    # The quantum corrections
    for power in sorted(orders.keys()):
        if power == 0:
            continue
        correction = orders[power]
        correction = expand(correction)
        nterms = len(Add.make_args(correction))
        print(f"\n  hbar^{power} correction: {nterms} terms")

        if nterms <= 50:
            print(f"    Expression: {correction}")

        # Analyze monomial structure of the correction
        try:
            p = Poly(correction, *phase_vars_c, domain='QQ')
            md = p.as_dict()
            if md:
                n_q = len(alg_c.q_vars)
                n_p = len(alg_c.p_vars)

                max_total = max(sum(m) for m in md.keys())
                max_p_deg = max(sum(m[n_q:n_q+n_p]) for m in md.keys())
                max_u_deg = max(sum(m[n_q+n_p:]) for m in md.keys())

                print(f"    Monomials: {len(md)}")
                print(f"    Max total degree: {max_total}")
                print(f"    Max momentum degree: {max_p_deg}")
                print(f"    Max u degree: {max_u_deg}")

                # Show the monomial structure
                print(f"    Monomial degrees (p_deg, u_deg, coeff):")
                for m, c in sorted(md.items(),
                                   key=lambda x: -abs(float(x[1]))):
                    p_deg = sum(m[n_q:n_q+n_p])
                    u_deg = sum(m[n_q+n_p:])
                    q_deg = sum(m[:n_q])
                    monom_str = " * ".join(
                        f"{v}^{e}" for v, e in zip(phase_vars_c, m)
                        if e > 0)
                    if len(monom_str) < 80:
                        print(f"      ({c}) * {monom_str}  "
                              f"[q={q_deg}, p={p_deg}, u={u_deg}]")
                    if len(md) > 30:
                        # Only show top 15 by magnitude
                        shown = sum(1 for _ in range(15))
                        break
        except Exception as e:
            print(f"    (Could not extract monomial structure: {e})")


if __name__ == "__main__":
    print("=" * 70)
    print("IDENTIFYING THE 117TH GENERATOR")
    print("N=3, d=1, 1/r potential")
    print("Classical rank: 116, Quantum rank: 117")
    print("=" * 70)

    t_start = time()

    (alg_c, c_exprs, c_names, c_levels,
     alg_q, q_exprs, q_names, q_levels) = build_generators_both()

    result = find_117th(alg_c, c_exprs, c_names, c_levels,
                         alg_q, q_exprs, q_names, q_levels)

    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {time()-t_start:.1f}s")
    print(f"{'='*70}")
