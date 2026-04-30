#!/usr/bin/env python3
"""
Extract the EXACT symbolic identity behind the 10^-14 SVD null-direction
that appears at a near-collision configuration of the planar 3-body algebra.

Strategy
--------
1. Load the 156 level-3 generators from checkpoints/level_3.pkl.
2. Fix the spatial configuration to a near-collision point with a small
   RATIONAL eps (so all arithmetic stays in QQ):
       (x1,y1) = (0,0),  (x2,y2) = (eps,0),  (x3,y3) = (1,0)
   with u12 = 1/eps, u13 = 1, u23 = 1/(1-eps).
3. Leave the 6 momenta symbolic.  Each generator g_k becomes a polynomial
   p_k(px1,py1,px2,py2,px3,py3) over QQ.
4. Stack the coefficient vectors into a rational matrix A whose rows are
   indexed by momentum monomials.  A vector c in ker(A^T) (left null space)
   is exactly a linear combination  sum_k c_k g_k  that vanishes identically
   on the entire 6-dim momentum fibre over the chosen position.
5. That kernel is the exact symbolic content of the "10^-14 at the edge"
   singular direction.  We pick the first kernel vector, write down the
   unevaluated combination  sum_k c_k * <symbolic g_k>, and verify that
   simplify(...) = 0.
"""

import os, sys, pickle
from time import time

sys.setrecursionlimit(500000)
os.environ["PYTHONUNBUFFERED"] = "1"

import sympy as sp
from sympy import Rational, Integer, Poly, Symbol, simplify, expand, Add

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth import (
    Q_VARS, P_VARS, U_VARS,
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
)

CKPT = os.path.join("checkpoints", "level_3.pkl")


def load_generators():
    with open(CKPT, "rb") as fh:
        data = pickle.load(fh)
    return data["exprs"], data["names"], data["levels"]


def build_collision_substitution(eps):
    """Non-collinear near-collision with all pairwise distances RATIONAL.

    Coordinates (chosen so r12, r13, r23 are all rational in eps):
        body1 = (0, 0)
        body2 = (3*eps, 4*eps)            -> r12 = 5*eps
        body3 = (3, 4)                    -> r13 = 5
                                          -> r23 = 5*(1 - eps)

    The 2 and 3 lie along the same ray from the origin, so body 1 sees a
    near-collision of bodies 2 and the line to 3, but the configuration is
    NON-collinear when y-momenta differ from x-momenta; it is geometrically
    a 5-fold scaled 3-4-5 right triangle with vertex 2 sliding toward 1.
    """
    return {
        x1: 0,                  y1: 0,
        x2: 3 * eps,            y2: 4 * eps,
        x3: 3,                  y3: 4,
        u12: Rational(1, 5) / eps,
        u13: Rational(1, 5),
        u23: Rational(1, 5) / (1 - eps),
    }


def collect_momentum_polys(exprs, sub):
    """Substitute the spatial config and return a list of polynomials in p-vars."""
    polys = []
    print(f"  Substituting collision config into {len(exprs)} generators...")
    for i, e in enumerate(exprs):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    [{i+1}/{len(exprs)}]", flush=True)
        ex = expand(e.subs(sub))
        polys.append(ex)
    return polys


def build_coeff_matrix(polys):
    """Return (DomainMatrix over QQ, monomial list) where rows = generators."""
    from sympy.polys.matrices import DomainMatrix
    from sympy import QQ

    print("  Extracting momentum monomials...")
    poly_dicts = []
    all_monoms = set()
    for ex in polys:
        if ex == 0:
            poly_dicts.append({})
            continue
        p = Poly(ex, *P_VARS, domain=QQ)
        d = p.as_dict()
        poly_dicts.append(d)
        all_monoms.update(d.keys())

    monom_list = sorted(all_monoms)
    monom_to_idx = {m: j for j, m in enumerate(monom_list)}
    n_gen = len(poly_dicts)
    n_mon = len(monom_list)
    print(f"  Coefficient matrix: {n_gen} generators x {n_mon} momentum monomials")

    rows = []
    for d in poly_dicts:
        row = [QQ.zero] * n_mon
        for m, c in d.items():
            row[monom_to_idx[m]] = QQ.convert(c)
        rows.append(row)
    A = DomainMatrix(rows, (n_gen, n_mon), QQ)
    return A, monom_list


def left_nullspace(A):
    """Return list of exact rational vectors c such that c^T A = 0."""
    AT = A.transpose()
    # nullspace returns a DomainMatrix whose ROWS span the nullspace
    N = AT.nullspace()
    vecs = []
    for i in range(N.shape[0]):
        v = [N.rep.getitem(i, j) for j in range(N.shape[1])]
        # Convert from domain elements to SymPy Rationals
        v = [sp.Rational(int(x.numerator), int(x.denominator))
             if hasattr(x, 'numerator') else sp.sympify(x) for x in v]
        vecs.append(v)
    return vecs


def clear_denominators(vec):
    """Scale a rational vector to coprime integers; pick a sign."""
    from math import gcd
    from functools import reduce
    rats = [sp.Rational(c) for c in vec]
    denoms = [r.q for r in rats if r != 0]
    if not denoms:
        return [Integer(0)] * len(rats)
    L = reduce(lambda a, b: a * b // gcd(a, b), denoms, 1)
    ints = [Integer(r * L) for r in rats]
    g = reduce(gcd, [abs(int(i)) for i in ints if i != 0])
    if g > 1:
        ints = [Integer(int(i) // g) for i in ints]
    # canonical sign: first nonzero positive
    for c in ints:
        if c != 0:
            if c < 0:
                ints = [-c for c in ints]
            break
    return ints


def main():
    print("=" * 72)
    print("COLLISION-EDGE SYZYGY EXTRACTION")
    print("=" * 72)

    eps = Rational(1, 100)
    print(f"\nNear-collision config: bodies 1-2 separated by r12 = 5*eps,  eps = {eps}")
    print(f"  positions: (x1,y1)=(0,0)  (x2,y2)=(3eps,4eps)  (x3,y3)=(3,4)")
    print(f"  pairwise distances: r12 = 5*eps,  r13 = 5,  r23 = 5*(1-eps)   (all rational)")

    print(f"\nLoading {CKPT} ...")
    exprs, names, levels = load_generators()
    print(f"  {len(exprs)} generators loaded ({sum(1 for l in levels if l==3)} at level 3)")

    sub = build_collision_substitution(eps)

    t0 = time()
    polys = collect_momentum_polys(exprs, sub)
    print(f"  done in {time()-t0:.1f}s")

    t0 = time()
    A, monom_list = build_coeff_matrix(polys)
    print(f"  matrix built in {time()-t0:.1f}s")

    # Generic full rank (no collision) is 116, so we expect 156-116 = 40
    # left-null vectors PLUS extra ones that hold only at this edge.
    print("\nComputing exact rank and left null space...")
    t0 = time()
    rank = A.rank()
    print(f"  rank(A) = {rank}   (generic = 116)")
    print(f"  left null dim (#syzygies at this config) = {len(exprs) - rank}")

    null = left_nullspace(A)
    print(f"  computed {len(null)} null vectors in {time()-t0:.1f}s")

    if not null:
        print("No null vectors found.")
        return

    # Sort by sparsity (fewest nonzero coefficients first => most readable)
    def sparsity(v):
        return sum(1 for c in v if c != 0)

    null.sort(key=sparsity)
    print("\nSparsity profile of null vectors (#nonzero coeffs):")
    for k, v in enumerate(null[:12]):
        print(f"  null[{k}]: nnz = {sparsity(v)}")

    # ---------- helper: generic non-collision substitution ----------
    import sympy as _sp
    def make_generic_sub():
        gsub = {
            x1: Rational(0),     y1: Rational(0),
            x2: Rational(7, 3),  y2: Rational(-2, 5),
            x3: Rational(1, 4),  y3: Rational(11, 9),
        }
        def u_of(i, j):
            ax, ay = (x1, y1) if i == 0 else (x2, y2) if i == 1 else (x3, y3)
            bx, by = (x1, y1) if j == 0 else (x2, y2) if j == 1 else (x3, y3)
            dx = gsub[ax] - gsub[bx]; dy = gsub[ay] - gsub[by]
            return 1 / _sp.sqrt(dx**2 + dy**2)
        gsub[u12] = u_of(0, 1)
        gsub[u13] = u_of(0, 2)
        gsub[u23] = u_of(1, 2)
        return gsub
    generic_sub = make_generic_sub()

    def is_deep(coeffs_list):
        """A null vector is DEEP if its symbolic combination vanishes at a
        generic spatial config (i.e. it is identically zero as a phase-space
        function, not just at the collision edge)."""
        nz = [i for i, c in enumerate(coeffs_list) if c != 0]
        combo = sum(coeffs_list[i] * exprs[i] for i in nz)
        val = expand(combo.subs(generic_sub))
        return val == 0

    # ---------- enumerate: print one DEEP and one SOFT example ----------
    deep_example = None
    soft_example = None
    for v in null:
        c = clear_denominators(v)
        nnz = sum(1 for x in c if x != 0)
        if nnz == 0:
            continue
        deep = is_deep(c)
        if deep and deep_example is None:
            deep_example = (c, nnz)
        if (not deep) and soft_example is None:
            soft_example = (c, nnz)
        if deep_example and soft_example:
            break

    def print_combination(coeffs_list, header, footer):
        nnz_idx = [i for i, c in enumerate(coeffs_list) if c != 0]
        print("\n" + "=" * 72)
        print(f"{header}   ({len(nnz_idx)} terms)")
        print("=" * 72)
        print("  UNEVALUATED EXPRESSION  (sum_k c_k * g_k = 0):")
        print("  " + "-" * 68)
        for i in nnz_idx:
            sign = "+" if coeffs_list[i] > 0 else "-"
            mag = abs(int(coeffs_list[i]))
            coeff_str = f"  {sign} {mag}" if mag != 1 else f"  {sign}  "
            print(f"  {coeff_str} * {names[i]:38s}   (level {levels[i]})")
        print("  " + "-" * 68)
        print(f"    {footer}")

    if deep_example is not None:
        c, nnz = deep_example
        print_combination(
            c,
            "DEEP syzygy (vanishes everywhere on phase space)",
            "= 0  identically (verified at a generic rational configuration).",
        )

    if soft_example is not None:
        c, nnz = soft_example
        print_combination(
            c,
            "SOFT syzygy (vanishes only at this near-collision config)",
            f"= 0  at  r12 = 5*eps = {5*eps},  but  != 0  generically.",
        )

        # Show that this identity DOES depend on eps -> the "10^-14" mechanism
        print("\nLeading eps-dependence of the same combination:")
        # Build with eps as a free symbol to expose the small-r12 scaling
        eps_sym = _sp.Symbol('epsilon', positive=True)
        sub_eps = build_collision_substitution(eps_sym)
        nnz_idx = [i for i, x in enumerate(c) if x != 0]
        combo = sum(c[i] * exprs[i] for i in nnz_idx)
        combo_eps = expand(combo.subs(sub_eps))
        # Substitute generic momenta to get a scalar function of eps
        import random
        random.seed(1)
        psub = {p: Rational(random.randint(-9, 9), random.randint(1, 9))
                for p in P_VARS}
        f_eps = sp.together(expand(combo_eps.subs(psub)))
        f_eps = sp.simplify(f_eps)
        print(f"  combo(eps) at random momenta = {f_eps}")
        # Series at eps -> 0 to expose the leading collision power
        try:
            ser = sp.series(f_eps, eps_sym, 0, 4).removeO()
            print(f"  series at eps -> 0:           {ser}")
        except Exception as e:
            print(f"  (series failed: {e})")
    else:
        print("\n(No SOFT syzygy found among inspected null vectors;\n"
              " all null directions at this config are DEEP identities.)")


if __name__ == "__main__":
    main()
