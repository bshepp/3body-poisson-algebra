#!/usr/bin/env python3
"""Extract the compact 10-term relative-coordinate form and analyze it."""

import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
from sympy import (Symbol, Integer, Rational, expand, cancel, factor,
                   symbols, Poly, diff, Add, collect, sqrt)
from quantum_algebra import QuantumNBodyAlgebra, hbar_sym
from identify_117th import separate_hbar_orders

def main():
    alg = QuantumNBodyAlgebra(3, 1, "1/r")
    H12, H13, H23 = alg.hamiltonian_list
    x1, x2, x3 = alg.q_vars
    u12, u13, u23 = alg.u_vars

    b1 = cancel(alg.commutator(H12, H13))
    b2 = cancel(alg.commutator(b1, H12))
    b3 = cancel(alg.commutator(b2, b1))
    g = expand(separate_hbar_orders(b3).get(2, Integer(0)))

    # Relative coordinates: set x1=0
    s, t = symbols('s t', real=True)  # s = x1-x2, t = x1-x3
    g_rel = g.subs({x1: 0, x2: -s, x3: -t})
    g_rel = expand(g_rel)

    print("=" * 70)
    print("117TH GENERATOR IN RELATIVE COORDINATES")
    print("(x1=0, x2=-s, x3=-t, so s=x1-x2, t=x1-x3)")
    print("=" * 70)
    print(f"\n  {len(Add.make_args(g_rel))} terms:")
    print(f"\n  g = {g_rel}")

    # Try to factor
    print(f"\n  Factored:")
    g_factored = factor(g_rel)
    print(f"  g = {g_factored}")

    # Now use dimensionless variables w12 = s*u12, w13 = t*u13
    w1 = Symbol('w1')  # = s * u12 = (x1-x2)/r12
    w2 = Symbol('w2')  # = t * u13 = (x1-x3)/r13

    # In the expression, terms have the form s^a * t^b * u12^c * u13^d
    # with a+b = x-degree, c+d = u-degree
    # x+u total is always 12, 16, or 20
    # x-degrees: 2, 4, 6; u-degrees: 10, 12, 14
    # So (x-deg, u-deg) pairs: (2,10), (4,12), (6,14)
    # Each is x+u = 12, 16, 20... no wait those are sums 12, 16, 20
    # But w12 = s*u12 has combined degree 1+1=2
    # w12^2 = s^2 * u12^2
    # If x-deg=2, u-deg=10: we need s^a * t^b * u12^c * u13^d
    # with a+b=2, c+d=10

    # Let me check: the scaling is g -> lambda^(-8) * g
    # under s -> lambda*s, t -> lambda*t, u -> u/lambda
    # So s^a * u12^c has scaling lambda^(a-c)
    # Total scaling: sum_terms lambda^(sum_i a_i - sum_i c_i) = lambda^(x_deg - u_deg)
    # x_deg - u_deg: 2-10=-8, 4-12=-8, 6-14=-8 => all terms scale as lambda^(-8)
    # Good, consistent.

    # Now: w12 = s*u12 scales as lambda^(1-1) = lambda^0 (dimensionless!)
    # So g has scaling dimension -8 in lambda.
    # We can write g = (u12*u13)^? * f(w12, w13) perhaps?
    # Actually let's try g = u12^a * u13^b * f(s*u12, t*u13)
    # With total u-degree varying... hmm.

    # Let's just try the substitution directly
    # s = w1/u12, t = w2/u13
    g_w = g_rel.subs({s: w1/u12, t: w2/u13})
    g_w = cancel(g_w)
    g_w = expand(g_w)

    print(f"\n\n  Substituting s = w1/u12, t = w2/u13:")
    print(f"  g = {g_w}")
    print(f"  ({len(Add.make_args(g_w))} terms)")

    # Try to extract common u factor
    p_g = Poly(g_w * u12**8 * u13**8, w1, w2, u12, u13, domain='QQ')
    # Hmm, let me try a different approach

    # Factor the relative-coord version more aggressively
    print("\n\n  Attempting deeper factorization...")

    # Collect by u-monomial
    p = Poly(g_rel, s, t, u12, u13, domain='QQ')
    md = p.as_dict()
    print(f"\n  Monomial structure (s-deg, t-deg, u12-deg, u13-deg -> coeff):")
    for m in sorted(md.keys()):
        print(f"    s^{m[0]} t^{m[1]} u12^{m[2]} u13^{m[3]}  :  {md[m]}")

    # Check: can we write this as sum_k c_k * (s*u12)^a_k * (t*u13)^b_k * u12^p_k * u13^q_k ?
    # where a+p = u12_power, b+q = u13_power
    # The total x-degree is a+b and total u-degree is (a+p)+(b+q)
    # We want to factor as much as possible into w12 = s*u12, w13 = t*u13

    # Actually, look at the monomial data. Each term has s^a * t^b * u12^c * u13^d
    # with a+b+c+d in {12, 16, 20}
    # and a+b in {2, 4, 6}

    # The key: in each term, is a <= c and b <= d?
    # If so, we can write s^a * u12^a * u12^(c-a) * t^b * u13^b * u13^(d-b)
    #   = w12^a * w13^b * u12^(c-a) * u13^(d-b)

    print(f"\n  Rewriting via w12 = s*u12, w13 = t*u13:")
    for m in sorted(md.keys()):
        a, b, c, d = m
        if a <= c and b <= d:
            extra_u12 = c - a
            extra_u13 = d - b
            print(f"    ({md[m]}) * w12^{a} * w13^{b} * u12^{extra_u12} * u13^{extra_u13}")
        else:
            print(f"    ({md[m]}) * s^{a} * t^{b} * u12^{c} * u13^{d}  "
                  f"[CANNOT express as w12^a w13^b]")

    # Check: extra_u12 + extra_u13 should be constant for this to factor cleanly
    extras = set()
    for m in md.keys():
        a, b, c, d = m
        extras.add((c - a) + (d - b))
    print(f"\n  Residual u-powers (u12^p * u13^q, p+q = ?): {extras}")

    # If constant, then g = u12^p * u13^q * F(w12, w13)
    if len(extras) == 1:
        total_extra = extras.pop()
        print(f"  => g = (1/r12^p * 1/r13^q) * F(w12, w13)")
        print(f"     with p+q = {total_extra}")

    # Also: check the near-collision leading term in factored form
    # Leading as u12 -> inf: u12^14 * (-225/2)(x1-x2)^6
    # = u12^14 * (-225/2) * s^6
    # = (-225/2) * w12^6 * u12^8
    # So this is w12^6 * u12^8 (times coefficient)

    # Check equilateral evaluation of symmetric part
    g_s3 = Integer(0)
    sigma12 = {s: -s, t: t - s, u12: u12, u13: u23, u23: u13}
    sigma13 = {s: t, t: s, u12: u23, u13: u13, u23: u12}
    sigma23 = {s: s, t: s - t, u12: u12, u13: u23, u23: u13}
    # Actually these are wrong for the relative coord form. Let me just
    # evaluate the original expression's S3-symmetric part at equilateral.

    g_original = g  # in x1,x2,x3,u12,u13,u23 coords
    # S3 symmetrize
    perms = [
        {},  # identity
        {x1: x2, x2: x1, u13: u23, u23: u13},  # (12)
        {x1: x3, x3: x1, u12: u23, u23: u12},  # (13)
        {x2: x3, x3: x2, u12: u13, u13: u12},  # (23)
        {x1: x2, x2: x3, x3: x1, u12: u23, u23: u13, u13: u12},  # (123)
        {x1: x3, x2: x1, x3: x2, u12: u13, u13: u23, u23: u12},  # (132)
    ]
    g_sym = Integer(0)
    for sigma in perms:
        g_sym += g_original.subs(sigma)
    g_sym = cancel(g_sym / 6)
    g_sym = expand(g_sym)

    print(f"\n\n{'='*70}")
    print(f"S3-SYMMETRIZED 117TH GENERATOR")
    print(f"{'='*70}")
    print(f"  Terms: {len(Add.make_args(g_sym))}")

    # Evaluate at equilateral
    g_sym_equil = g_sym.subs({x1: 0, x2: 1, x3: Rational(1, 2),
                              u12: 1, u13: 1, u23: 1})
    print(f"  At equilateral (r_ij=1): {cancel(g_sym_equil)}")

    # Evaluate at general equilateral
    a = Symbol('a', positive=True)
    g_sym_equil_a = g_sym.subs({x1: 0, x2: a, x3: a/2,
                                u12: 1/a, u13: 1/a, u23: 1/a})
    g_sym_equil_a = cancel(g_sym_equil_a)
    print(f"  At equilateral (scale a): {g_sym_equil_a}")

    # Relative-coord form of symmetric version
    g_sym_rel = g_sym.subs({x1: 0, x2: -s, x3: -t})
    g_sym_rel = expand(g_sym_rel)
    print(f"\n  In relative coords (x1=0, s=x1-x2, t=x1-x3):")
    print(f"  {len(Add.make_args(g_sym_rel))} terms")

    # Check definite sign of g_sym at equilateral
    print(f"\n  Sign at equilateral: {'NEGATIVE' if g_sym_equil < 0 else 'POSITIVE' if g_sym_equil > 0 else 'ZERO'}")

    # Check: is g_sym negative definite?
    # Evaluate at several configurations
    configs = [
        ("equilateral r=1", {x1:0, x2:1, x3:Rational(1,2), u12:1, u13:1, u23:1}),
        ("equilateral r=2", {x1:0, x2:2, x3:1, u12:Rational(1,2), u13:Rational(1,2), u23:Rational(1,2)}),
        ("collinear 0,1,2", {x1:0, x2:1, x3:2, u12:1, u13:Rational(1,2), u23:1}),
        ("collinear 0,1,3", {x1:0, x2:1, x3:3, u12:1, u13:Rational(1,3), u23:Rational(1,2)}),
        ("right triangle", {x1:0, x2:3, x3:4, u12:Rational(1,3), u13:Rational(1,4), u23:1}),
        ("near collision", {x1:0, x2:Rational(1,10), x3:1, u12:10, u13:1, u23:Rational(10,9)}),
    ]

    print(f"\n  Sign check at various configurations:")
    all_neg = True
    for name, subs in configs:
        val = g_sym.subs(subs)
        val = cancel(val)
        sign = "NEG" if val < 0 else "POS" if val > 0 else "ZERO"
        if val >= 0:
            all_neg = False
        print(f"    {name}: {float(val):.6g}  ({sign})")

    if all_neg:
        print(f"\n  *** g_sym appears to be NEGATIVE DEFINITE ***")


if __name__ == "__main__":
    main()
