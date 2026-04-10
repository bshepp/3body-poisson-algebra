#!/usr/bin/env python3
"""
Analyze the 117th generator's mathematical properties:
- S3 permutation symmetry
- Collision limits (u_ij -> infinity, i.e., r_ij -> 0)
- Sign definiteness
- Special configurations (equilateral, collinear, etc.)
"""

import os
import sys
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
from sympy import (Symbol, Integer, Rational, Add, Poly, expand, cancel,
                   diff, symbols, factor, collect, sqrt, oo, limit,
                   simplify, together, Abs)

from exact_growth_nbody import NBodyAlgebra
from quantum_algebra import QuantumNBodyAlgebra, hbar_sym
from identify_117th import build_generators_both, separate_hbar_orders


def get_117th_expression():
    """Build and extract the simplest 117th generator (hbar^2 correction
    of {{{H12,H13},H12},{H12,H13}})."""

    print("Building quantum generators (N=3, d=1)...")
    t0 = time()

    alg = QuantumNBodyAlgebra(3, 1, "1/r")
    H12, H13, H23 = alg.hamiltonian_list

    # Build the specific bracket {{{H12,H13},H12},{H12,H13}}
    print("  Computing commutator brackets...")
    b1 = alg.commutator(H12, H13)  # {H12,H13}
    b1 = cancel(b1)
    print(f"    {{H12,H13}}: {len(Add.make_args(b1))} terms")

    b2 = alg.commutator(b1, H12)   # {{H12,H13},H12}
    b2 = cancel(b2)
    print(f"    {{{{H12,H13}},H12}}: {len(Add.make_args(b2))} terms")

    b3 = alg.commutator(b2, b1)    # {{{H12,H13},H12},{H12,H13}}
    b3 = cancel(b3)
    print(f"    {{{{{{H12,H13}},H12}},{{H12,H13}}}}: {len(Add.make_args(b3))} terms")

    # Extract hbar^2 part
    orders = separate_hbar_orders(b3)
    g117 = expand(orders.get(2, Integer(0)))
    print(f"  hbar^2 part: {len(Add.make_args(g117))} terms [{time()-t0:.1f}s]")

    return g117, alg


def analyze_permutation_symmetry(g, alg):
    """Test S3 permutation symmetry of the 117th generator."""

    print("\n" + "=" * 70)
    print("1. PERMUTATION SYMMETRY (S3)")
    print("=" * 70)

    x1, x2, x3 = alg.q_vars
    u12, u13, u23 = alg.u_vars

    # S3 has 6 elements. For three particles, the generators are:
    # (12): swap particles 1 and 2
    # (13): swap particles 1 and 3
    # (23): swap particles 2 and 3

    # Under particle swap (i,j), we need:
    #   x_i <-> x_j
    #   u_ik <-> u_jk for all k != i,j
    #   u_ij stays the same

    # Permutation (12): x1<->x2, u13<->u23, u12 stays
    sigma12 = {x1: x2, x2: x1, x3: x3,
               u12: u12, u13: u23, u23: u13}

    # Permutation (13): x1<->x3, u12<->u23, u13 stays
    sigma13 = {x1: x3, x3: x1, x2: x2,
               u12: u23, u23: u12, u13: u13}

    # Permutation (23): x2<->x3, u12<->u13, u23 stays
    sigma23 = {x1: x1, x2: x3, x3: x2,
               u12: u13, u13: u12, u23: u23}

    perms = [("(12)", sigma12), ("(13)", sigma13), ("(23)", sigma23)]

    for name, sigma in perms:
        g_perm = expand(g.subs(sigma))
        diff_expr = expand(g_perm - g)
        sum_expr = expand(g_perm + g)

        if diff_expr == 0:
            print(f"  {name}: g -> g  (SYMMETRIC)")
        elif sum_expr == 0:
            print(f"  {name}: g -> -g  (ANTISYMMETRIC)")
        else:
            # Check ratio
            ratio = cancel(g_perm / g)
            if ratio.is_number:
                print(f"  {name}: g -> {ratio}*g")
            else:
                print(f"  {name}: g -> (non-trivial), diff has "
                      f"{len(Add.make_args(diff_expr))} terms")
                # Check if it's a multiple
                terms_orig = Add.make_args(g)
                terms_perm = Add.make_args(g_perm)
                if len(terms_orig) == len(terms_perm):
                    # Try term-by-term ratio
                    t0 = terms_orig[0]
                    t0p = terms_perm[0]
                    r = cancel(t0p / t0)
                    print(f"    (first term ratio: {r})")

    # Check full S3 invariance: sum over all 6 permutations
    print("\n  Full S3 analysis:")

    # (123): x1->x2->x3->x1, u12->u23->u13->u12
    sigma_cyc = {x1: x2, x2: x3, x3: x1,
                 u12: u23, u23: u13, u13: u12}

    # (132): x1->x3->x2->x1, u12->u13->u23->u12
    sigma_cyc2 = {x1: x3, x2: x1, x3: x2,
                  u12: u13, u13: u23, u23: u12}

    all_perms = [
        ("e", None),
        ("(12)", sigma12),
        ("(13)", sigma13),
        ("(23)", sigma23),
        ("(123)", sigma_cyc),
        ("(132)", sigma_cyc2),
    ]

    g_sym = g  # identity
    results = {"e": g}
    for name, sigma in all_perms[1:]:
        results[name] = expand(g.subs(sigma))

    # Check: is sum = 0? (would mean g is in the alternating rep)
    total_sum = sum(results.values())
    total_sum = expand(total_sum)
    print(f"  Sum over S3: {len(Add.make_args(total_sum))} terms "
          f"({'ZERO' if total_sum == 0 else 'NONZERO'})")

    # Check: is alternating sum = 0? (sum with signs of permutation)
    # Even perms: e, (123), (132). Odd: (12), (13), (23)
    alt_sum = (results["e"] + results["(123)"] + results["(132)"]
               - results["(12)"] - results["(13)"] - results["(23)"])
    alt_sum = expand(alt_sum)
    print(f"  Alternating sum: {len(Add.make_args(alt_sum))} terms "
          f"({'ZERO' if alt_sum == 0 else 'NONZERO'})")

    # Determine the representation
    if total_sum != 0 and alt_sum == 0:
        print("  => g transforms in the TRIVIAL (symmetric) representation")
    elif total_sum == 0 and alt_sum != 0:
        print("  => g transforms in the SIGN (alternating) representation")
    elif total_sum == 0 and alt_sum == 0:
        print("  => g transforms in the STANDARD (2-dim) representation")
    else:
        print("  => g has components in BOTH trivial and standard reps")

    # Extract the symmetric (trivial) component = (1/6) * sum over S3
    g_sym_component = cancel(total_sum / 6)
    n_sym = len(Add.make_args(expand(g_sym_component)))
    print(f"\n  S3-symmetrized component (1/6 * sum): {n_sym} terms")
    if n_sym <= 30 and g_sym_component != 0:
        print(f"    = {expand(g_sym_component)}")

    # Extract the antisymmetric (sign) component = (1/6) * alt sum
    g_alt_component = cancel(alt_sum / 6)
    n_alt = len(Add.make_args(expand(g_alt_component)))
    print(f"  S3-antisymmetric component: {n_alt} terms")

    # The standard rep component is what's left
    g_std = expand(g - g_sym_component - g_alt_component)
    n_std = len(Add.make_args(g_std))
    print(f"  Standard (2-dim) rep component: {n_std} terms")

    # Key question: is the symmetric component nonzero?
    # If so, averaging all 48 corrections over S3 gives a fully symmetric
    # quantum correction.
    if g_sym_component != 0:
        print(f"\n  *** The S3-SYMMETRIC PART is nonzero! ***")
        print(f"  This means there exists a fully permutation-symmetric")
        print(f"  quantum correction, obtainable by S3-averaging.")

    return results


def analyze_collision_limits(g, alg):
    """Analyze behavior as pairwise distances go to zero (u -> infinity)."""

    print("\n" + "=" * 70)
    print("2. COLLISION LIMITS (r_ij -> 0, i.e., u_ij -> infinity)")
    print("=" * 70)

    x1, x2, x3 = alg.q_vars
    u12, u13, u23 = alg.u_vars

    # The 117th generator is a polynomial in x_i and u_ij
    # As u_ij -> infinity (r_ij -> 0), the dominant terms have
    # the highest power of u_ij.

    p = Poly(g, *alg.q_vars, *alg.u_vars, domain='QQ')
    md = p.as_dict()

    # For each u variable, find the leading power
    n_q = len(alg.q_vars)
    u_indices = {u12: n_q, u13: n_q + 1, u23: n_q + 2}

    for u_var, u_idx in u_indices.items():
        u_powers = [m[u_idx] for m in md.keys() if m[u_idx] > 0]
        if not u_powers:
            print(f"\n  {u_var} (= 1/r_{str(u_var)[1:]}): NOT PRESENT in expression")
            continue
        max_power = max(u_powers)
        min_power = min(u_powers)
        print(f"\n  {u_var} (= 1/r_{str(u_var)[1:]}): powers {min_power} to {max_power}")

        # Extract leading-order terms (highest power of this u)
        leading = Integer(0)
        for m, c in md.items():
            if m[u_idx] == max_power:
                monom = c
                for i, v in enumerate(list(alg.q_vars) + list(alg.u_vars)):
                    if i != u_idx:
                        monom *= v ** m[i]
                leading += monom

        leading = expand(leading)
        n_leading = len(Add.make_args(leading))
        print(f"    Leading term ({u_var}^{max_power}): {n_leading} terms")
        if n_leading <= 10:
            print(f"    = {u_var}^{max_power} * ({leading})")
        else:
            print(f"    = {u_var}^{max_power} * ({n_leading}-term polynomial in x,u)")

        # The physical meaning: as r_ij -> 0, u_ij -> infinity,
        # so g ~ u_ij^max_power ~ 1/r_ij^max_power
        print(f"    => As r_{str(u_var)[1:]} -> 0: g ~ 1/r^{max_power}")


def analyze_special_configurations(g, alg):
    """Evaluate g at special geometric configurations."""

    print("\n" + "=" * 70)
    print("3. SPECIAL CONFIGURATIONS")
    print("=" * 70)

    x1, x2, x3 = alg.q_vars
    u12, u13, u23 = alg.u_vars

    # Configuration 1: Equilateral triangle
    # x1=0, x2=1, x3=1/2, r12=r13=r23=1
    # u12=u13=u23=1
    print("\n  (a) Equilateral triangle: x1=0, x2=1, x3=1/2, all r_ij=1")
    g_equil = g.subs({x1: 0, x2: 1, x3: Rational(1, 2),
                      u12: 1, u13: 1, u23: 1})
    g_equil = expand(g_equil)
    print(f"      g = {g_equil}")

    # Configuration 2: Equilateral with general scale
    # x1=0, x2=a, x3=a/2, u_ij = 1/a (all equal)
    a = Symbol('a', positive=True)
    print("\n  (b) Equilateral with scale a: x1=0, x2=a, x3=a/2, u_ij=1/a")
    g_equil_a = g.subs({x1: 0, x2: a, x3: a/2,
                        u12: 1/a, u13: 1/a, u23: 1/a})
    g_equil_a = expand(g_equil_a)
    g_equil_a = cancel(g_equil_a)
    print(f"      g = {g_equil_a}")
    if g_equil_a != 0:
        # Factor out powers of a
        p_a = Poly(g_equil_a * a**20, a, domain='QQ')  # clear denominators
        print(f"      (as polynomial in a: degree {p_a.degree()})")

    # Configuration 3: Collinear, equally spaced
    # x1=0, x2=1, x3=2
    # r12=1, r13=2, r23=1 => u12=1, u13=1/2, u23=1
    print("\n  (c) Collinear equally-spaced: x1=0, x2=1, x3=2")
    g_col = g.subs({x1: 0, x2: 1, x3: 2,
                    u12: 1, u13: Rational(1, 2), u23: 1})
    g_col = expand(g_col)
    print(f"      g = {g_col}")

    # Configuration 4: Two particles coincident (x1=x2), r12->0
    # This should diverge since u12 -> infinity
    print("\n  (d) Near-collision x1 ~ x2: x1=0, x2=eps, x3=1")
    eps = Symbol('epsilon', positive=True)
    g_near = g.subs({x1: 0, x2: eps, x3: 1,
                     u12: 1/eps, u13: 1, u23: 1/(1-eps)})
    g_near_series = sp.series(g_near, eps, 0, n=3)
    print(f"      g ~ {g_near_series}")

    # Configuration 5: Symmetric collinear (center of mass at origin)
    # x1=-a, x2=a, x3=0 => r12=2a, r13=a, r23=a
    print("\n  (e) Symmetric collinear: x1=-a, x2=a, x3=0")
    g_symcol = g.subs({x1: -a, x2: a, x3: 0,
                       u12: 1/(2*a), u13: 1/a, u23: 1/a})
    g_symcol = cancel(g_symcol)
    print(f"      g = {g_symcol}")

    # Configuration 6: One particle at infinity
    # x3 -> infinity, u13, u23 -> 0
    print("\n  (f) Particle 3 at infinity: u13->0, u23->0, x3->inf")
    g_inf = g.subs({u13: 0, u23: 0})
    g_inf = expand(g_inf)
    n_inf = len(Add.make_args(g_inf))
    print(f"      g = {g_inf}")
    if g_inf == 0:
        print(f"      => g VANISHES when any particle goes to infinity")
    else:
        print(f"      => g does NOT vanish ({n_inf} terms survive)")

    return g_equil


def analyze_sign_and_structure(g, alg):
    """Analyze sign definiteness and algebraic structure."""

    print("\n" + "=" * 70)
    print("4. SIGN ANALYSIS AND ALGEBRAIC STRUCTURE")
    print("=" * 70)

    x1, x2, x3 = alg.q_vars
    u12, u13, u23 = alg.u_vars

    # Try to factor the expression
    print("\n  (a) Attempting to factor...")
    g_factored = factor(g)
    if g_factored != g:
        print(f"      Factored form: {g_factored}")
    else:
        print(f"      No overall factorization found")

    # Check if it can be written as a sum/difference of squares
    # or has a definite sign

    # The expression involves differences like (x1-x2), suggesting
    # it might be related to relative coordinates
    # Define relative coords
    r12 = x1 - x2  # = (x1-x2)
    r13 = x1 - x3
    r23 = x2 - x3

    print("\n  (b) Rewriting in terms of relative separations...")
    # Substitute x2 = x1 - r12, x3 = x1 - r13
    # Then r23 = r13 - r12
    R = Symbol('R')  # center of mass ~ x1
    s = Symbol('s')  # r12 = x1-x2
    t = Symbol('t')  # r13 = x1-x3

    # x1 = R + (s+t)/3 (approx, for equal masses CM = (x1+x2+x3)/3)
    # Actually simpler: set x1=0, x2=-s, x3=-t
    g_rel = g.subs({x1: 0, x2: -s, x3: -t})
    g_rel = expand(g_rel)
    print(f"      In relative coords (x1=0, x2=-s, x3=-t): "
          f"{len(Add.make_args(g_rel))} terms")

    # Check translation invariance: x_i -> x_i + c for all i
    c = Symbol('c')
    g_translated = g.subs({x1: x1 + c, x2: x2 + c, x3: x3 + c})
    g_translated = expand(g_translated)
    diff_trans = expand(g_translated - g)
    print(f"\n  (c) Translation invariance (x_i -> x_i + c):")
    if diff_trans == 0:
        print(f"      YES - g is translation invariant")
    else:
        print(f"      NO - {len(Add.make_args(diff_trans))} terms differ")

    # Check scaling behavior: x_i -> lambda*x_i, u_ij -> u_ij/lambda
    lam = Symbol('lambda', positive=True)
    g_scaled = g.subs({x1: lam*x1, x2: lam*x2, x3: lam*x3,
                       u12: u12/lam, u13: u13/lam, u23: u23/lam})
    g_scaled = expand(g_scaled)
    ratio = cancel(g_scaled / g) if g != 0 else None
    print(f"\n  (d) Scaling (x->lambda*x, u->u/lambda):")
    if ratio is not None and ratio.is_number:
        print(f"      g -> {ratio} * g  (homogeneous degree {sp.log(ratio, lam) if ratio != 1 else 0})")
    elif ratio is not None:
        # Check if it's a power of lambda
        try:
            p_lam = Poly(g_scaled, lam)
            degrees = [sum(m) for m in p_lam.as_dict().keys()]
            if len(set(degrees)) == 1:
                print(f"      g -> lambda^{degrees[0]} * g  "
                      f"(homogeneous of degree {degrees[0]})")
            else:
                print(f"      Not homogeneous. Lambda degrees: "
                      f"{sorted(set(degrees))}")
        except Exception:
            # Try ratio approach
            g_s2 = cancel(g_scaled)
            r2 = cancel(g_s2 / g)
            print(f"      Ratio g_scaled/g = {r2}")

    # Check if the expression is a polynomial in (x_i - x_j)^2 * u_ij^k
    print(f"\n  (e) Checking if expressible via gauge-invariant combinations...")
    # Natural gauge-invariant combos: (x_i - x_j) * u_ij = (x_i-x_j)/r_ij
    # These are dimensionless direction cosines
    w12 = (x1 - x2) * u12  # = (x1-x2)/r12, dimensionless
    w13 = (x1 - x3) * u13
    w23 = (x2 - x3) * u23

    # Note: w12^2 = (x1-x2)^2 * u12^2 = (x1-x2)^2/r12^2
    # But u12 = 1/r12 and r12 = |x1-x2| (in 1D, r12 = |x1-x2|)
    # So w12 = (x1-x2)/|x1-x2| = +/- 1 (a sign, not really useful)
    # However in the algebraic framework, x1-x2 and u12 are independent
    # formal variables, so w12 is a genuine variable.
    print(f"      Dimensionless variables: w_ij = (x_i - x_j) * u_ij")
    print(f"      w12 = {w12}, w13 = {w13}, w23 = {w23}")

    # Substitute: x2 = x1 - w12/u12, etc. is complicated.
    # Instead, check specific monomial patterns.
    p = Poly(g, x1, x2, x3, u12, u13, u23, domain='QQ')
    md = p.as_dict()

    # Group by total degree in x and u separately
    x_degrees = set()
    u_total_degrees = set()
    for m in md.keys():
        xd = m[0] + m[1] + m[2]  # total x degree
        ud = m[3] + m[4] + m[5]  # total u degree
        x_degrees.add(xd)
        u_total_degrees.add(ud)

    print(f"      x-degree values: {sorted(x_degrees)}")
    print(f"      u-degree values: {sorted(u_total_degrees)}")
    print(f"      x+u total degrees: {sorted(set(sum(m) for m in md.keys()))}")


def main():
    t_start = time()

    g, alg = get_117th_expression()

    perm_results = analyze_permutation_symmetry(g, alg)
    analyze_collision_limits(g, alg)
    g_equil = analyze_special_configurations(g, alg)
    analyze_sign_and_structure(g, alg)

    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {time()-t_start:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
