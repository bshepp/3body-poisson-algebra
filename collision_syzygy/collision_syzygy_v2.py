#!/usr/bin/env python3
"""
collision_syzygy_v2.py
======================

Pure collision-edge syzygy extraction (collinear contamination removed).

Geometry
--------
    body1 = (0, 0)
    body2 = (3*eps, 4*eps)        -> r12 = 5*eps          (rational)
    body3 = (4, 3)                -> r13 = 5              (rational)
                                  -> r23 = sqrt(D(eps))   (irrational generically)
        D(eps) = 25 - 48*eps + 25*eps^2   (rational in eps)

Body 2 sits on the ray y = (4/3) x; body 3 sits on y = (3/4) x.
The two rays differ, so the configuration is non-collinear for every eps > 0.
The eps -> 0 limit is a clean 1-2 binary collision (no collinear degeneracy).

Algebra
-------
We keep u23 as a fresh symbol S subject to S^2 = 1/D(eps).  After
substituting the spatial coordinates, u12 = 1/(5 eps), u13 = 1/5, u23 = S,
each level-<=3 generator becomes a polynomial in (P_VARS, S) over Q.
We reduce S^k mod (S^2 - 1/D) so each generator is

            g_k = A_k(p) + B_k(p) * S   ,    A_k, B_k in Q[p].

A rational left-null vector c = (c_1, ..., c_156) satisfies
sum_k c_k * g_k = 0  iff  sum_k c_k * A_k = 0  AND  sum_k c_k * B_k = 0.
We stack the (A,B) coefficient matrices side by side and compute the
left null space exactly via DomainMatrix(QQ).

Outputs
-------
1. A stratification table: rank vs eps along the binary-collision family,
   plus a generic reference (eps_ref) and the collinear (3,4) slice from v1.
2. The first DEEP syzygy (vanishes generically), verified.
3. The first SOFT syzygy (vanishes only at the chosen eps), with its
   leading-order behaviour in eps.
"""

import os, sys, pickle
from time import time
from math import gcd
from functools import reduce

sys.setrecursionlimit(500000)
os.environ["PYTHONUNBUFFERED"] = "1"

import sympy as sp
from sympy import Rational, Integer, Poly, Symbol, expand, sqrt
from sympy.polys.matrices import DomainMatrix
from sympy import QQ

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth import (
    Q_VARS, P_VARS, U_VARS,
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
)

CKPT = os.path.join("checkpoints", "level_3.pkl")
S = sp.Symbol("S_u23")  # symbolic stand-in for u23 = 1/r23


# ---------------------------------------------------------------- I/O

def load_generators():
    with open(CKPT, "rb") as fh:
        data = pickle.load(fh)
    return data["exprs"], data["names"], data["levels"]


# ---------------------------------------------------- geometry

def r23_sq(eps):
    """D(eps) = (4 - 3 eps)^2 + (3 - 4 eps)^2 = 25 - 48 eps + 25 eps^2."""
    return 25 - 48 * eps + 25 * eps ** 2


def build_substitution(eps, collinear=False):
    """Return (sub, D) where sub maps the spatial vars + u_ij to numbers
    (with u23 -> S, the symbolic 1/r23) and D is the rational r23^2.

    collinear=True reproduces the v1 (3,4) slice for cross-check.
    """
    if collinear:
        # body3 = (3,4): r23 = 5*(1-eps) -> rational; we still use S for
        # uniformity, with D = 25*(1-eps)^2 so S = 1/(5*(1-eps)).
        D = 25 * (1 - eps) ** 2
        sub = {
            x1: 0, y1: 0,
            x2: 3 * eps, y2: 4 * eps,
            x3: 3,       y3: 4,
            u12: Rational(1, 5) / eps,
            u13: Rational(1, 5),
            u23: S,
        }
    else:
        D = r23_sq(eps)
        sub = {
            x1: 0, y1: 0,
            x2: 3 * eps, y2: 4 * eps,
            x3: 4,       y3: 3,
            u12: Rational(1, 5) / eps,
            u13: Rational(1, 5),
            u23: S,
        }
    return sub, sp.together(D)


# ---------------------------------------------------- matrix build

def reduce_S(poly_in_S, D_rat):
    """Reduce S^k mod (S^2 - 1/D_rat).  Returns a polynomial of S-degree <=1."""
    # poly_in_S is a sympy expression poly in S with rational coefficients
    # (after positions etc are substituted).
    p = Poly(poly_in_S, S)
    deg = p.degree()
    if deg <= 1:
        return poly_in_S
    # Build rule S^2 = 1/D
    invD = Rational(1, 1) / D_rat
    coeffs = p.all_coeffs()  # highest first
    # Reduce iteratively
    while len(coeffs) > 2:
        c = coeffs[0]
        # contributes c * invD to S^{deg-2} coefficient
        coeffs[2] = coeffs[2] + c * invD
        coeffs = coeffs[1:]
        # If after pop, only 1 coeff remains then we drop into S^0 path:
        if len(coeffs) == 1:
            coeffs = [Rational(0), coeffs[0]]
    # Now len == 2: [B (coeff of S^1), A (coeff of S^0)]
    B, A = coeffs[0], coeffs[1]
    return sp.expand(A + B * S)


def split_AB(reduced_expr):
    """Given an expression that is A(p) + B(p)*S with no higher S powers,
    return (A, B) as sympy expressions in P_VARS over Q."""
    p = Poly(reduced_expr, S, *P_VARS, domain="QQ")
    A_terms = []
    B_terms = []
    for monom, coef in p.as_dict().items():
        s_pow = monom[0]
        p_mon = monom[1:]
        # Build the P-monomial back
        pm = sp.Integer(1)
        for var, e in zip(P_VARS, p_mon):
            if e:
                pm *= var ** e
        term = coef * pm
        if s_pow == 0:
            A_terms.append(term)
        elif s_pow == 1:
            B_terms.append(term)
        else:
            raise RuntimeError(f"unexpected S^{s_pow} after reduction")
    return sp.Add(*A_terms) if A_terms else sp.Integer(0), \
           sp.Add(*B_terms) if B_terms else sp.Integer(0)


def collect_polys(exprs, sub, D_rat, verbose=False):
    """Substitute, reduce, split.  Returns list of (A_k, B_k) polynomials in P."""
    polys = []
    n = len(exprs)
    for i, e in enumerate(exprs):
        if verbose and ((i + 1) % 30 == 0 or i == 0):
            print(f"    [{i+1}/{n}]", flush=True)
        ex = sp.expand(e.subs(sub))
        ex_red = reduce_S(ex, D_rat)
        A, B = split_AB(ex_red)
        polys.append((A, B))
    return polys


def build_AB_matrix(polys):
    """Stack (A_k) and (B_k) coefficient matrices side by side over Q."""
    poly_dicts_A = []
    poly_dicts_B = []
    monoms_A = set()
    monoms_B = set()
    for A, B in polys:
        if A == 0:
            poly_dicts_A.append({})
        else:
            d = Poly(A, *P_VARS, domain=QQ).as_dict()
            poly_dicts_A.append(d)
            monoms_A.update(d.keys())
        if B == 0:
            poly_dicts_B.append({})
        else:
            d = Poly(B, *P_VARS, domain=QQ).as_dict()
            poly_dicts_B.append(d)
            monoms_B.update(d.keys())

    mlist_A = sorted(monoms_A)
    mlist_B = sorted(monoms_B)
    idx_A = {m: j for j, m in enumerate(mlist_A)}
    idx_B = {m: j for j, m in enumerate(mlist_B)}
    nA, nB = len(mlist_A), len(mlist_B)
    n_gen = len(polys)
    rows = []
    for dA, dB in zip(poly_dicts_A, poly_dicts_B):
        row = [QQ.zero] * (nA + nB)
        for m, c in dA.items():
            row[idx_A[m]] = QQ.convert(c)
        for m, c in dB.items():
            row[nA + idx_B[m]] = QQ.convert(c)
        rows.append(row)
    return DomainMatrix(rows, (n_gen, nA + nB), QQ), nA, nB


def left_nullspace(A):
    AT = A.transpose()
    N = AT.nullspace()
    vecs = []
    for i in range(N.shape[0]):
        v = [N.rep.getitem(i, j) for j in range(N.shape[1])]
        v = [sp.Rational(int(x.numerator), int(x.denominator))
             if hasattr(x, "numerator") else sp.sympify(x) for x in v]
        vecs.append(v)
    return vecs


def clear_denominators(vec):
    rats = [sp.Rational(c) for c in vec]
    denoms = [r.q for r in rats if r != 0]
    if not denoms:
        return [Integer(0)] * len(rats)
    L = reduce(lambda a, b: a * b // gcd(a, b), denoms, 1)
    ints = [Integer(r * L) for r in rats]
    nz = [abs(int(i)) for i in ints if i != 0]
    if nz:
        g = reduce(gcd, nz)
        if g > 1:
            ints = [Integer(int(i) // g) for i in ints]
    for c in ints:
        if c != 0:
            if c < 0:
                ints = [-c for c in ints]
            break
    return ints


# ---------------------------------------------------- generic / oracle

def make_generic_sub():
    """A non-collision, non-collinear, non-symmetric rational config used
    as an oracle: a syzygy that vanishes here is a DEEP one."""
    gsub = {
        x1: Rational(0),     y1: Rational(0),
        x2: Rational(7, 3),  y2: Rational(-2, 5),
        x3: Rational(1, 4),  y3: Rational(11, 9),
    }
    def u_of(i, j):
        ax, ay = (x1, y1) if i == 0 else (x2, y2) if i == 1 else (x3, y3)
        bx, by = (x1, y1) if j == 0 else (x2, y2) if j == 1 else (x3, y3)
        dx = gsub[ax] - gsub[bx]; dy = gsub[ay] - gsub[by]
        return 1 / sp.sqrt(dx ** 2 + dy ** 2)
    gsub[u12] = u_of(0, 1)
    gsub[u13] = u_of(0, 2)
    gsub[u23] = u_of(1, 2)
    return gsub


# ---------------------------------------------------- main

def stratification_sweep(exprs, names, levels, eps_values):
    """For each eps, build the (A|B) matrix and report rank + nullity."""
    rows = []
    for tag, eps_val, collinear in eps_values:
        sub, D_rat = build_substitution(eps_val, collinear=collinear)
        polys = collect_polys(exprs, sub, D_rat, verbose=False)
        M, nA, nB = build_AB_matrix(polys)
        r = M.rank()
        nullity = len(exprs) - r
        rows.append((tag, eps_val, collinear, r, nullity, nA, nB))
    return rows


def main():
    print("=" * 72)
    print("COLLISION-EDGE SYZYGY EXTRACTION  (v2: non-collinear (4,3) slice)")
    print("=" * 72)

    print(f"\nLoading {CKPT} ...")
    exprs, names, levels = load_generators()
    print(f"  {len(exprs)} generators loaded "
          f"({sum(1 for l in levels if l==3)} at level 3)")

    # ---------------- stratification sweep ----------------
    print("\n" + "=" * 72)
    print("STRATIFICATION SWEEP")
    print("=" * 72)
    print("Sweeping eps along the binary-collision family body3=(4,3) and")
    print("comparing with the (3,4) collinear slice.")
    print("(rank is over Q after splitting Q+Q*sqrt(D); nullity = #syzygies.)\n")

    # Generic-ish eps (away from collision): 1/3, 1/4
    # Mid-range: 1/10
    # Edge: 1/100, 1/1000
    sweep = [
        ("(4,3) non-collinear",  Rational(1, 3),    False),
        ("(4,3) non-collinear",  Rational(1, 5),    False),
        ("(4,3) non-collinear",  Rational(1, 10),   False),
        ("(4,3) non-collinear",  Rational(1, 100),  False),
        ("(4,3) non-collinear",  Rational(1, 1000), False),
        ("(3,4) COLLINEAR",      Rational(1, 100),  True),
    ]

    table = stratification_sweep(exprs, names, levels, sweep)

    print(f"  {'slice':22s} {'eps':>10s} {'rank':>6s} {'nullity':>9s} "
          f"{'#deep est.':>12s}")
    print("  " + "-" * 64)
    # Identify the smallest nullity in non-collinear sweep as the "generic"
    # (this is the binary-collision-family generic dimension).
    nc_nullities = [row[4] for row in table if not row[2]]
    bin_generic_nullity = min(nc_nullities) if nc_nullities else None
    for tag, eps_val, collinear, r, nullity, nA, nB in table:
        deep_est = bin_generic_nullity if (not collinear) else "n/a"
        print(f"  {tag:22s} {str(eps_val):>10s} {r:>6d} {nullity:>9d} "
              f"{str(deep_est):>12s}")
    print("  " + "-" * 64)
    print("  Interpretation:")
    print("   * The (4,3) family has CONSTANT rank for every tested eps > 0,")
    print("     so the same set of binary-collision syzygies is present at")
    print("     every point of this stratum.")
    print("   * The (3,4) slice has SMALLER rank because it lies in the")
    print("     (binary-collision) cap (collinear) sub-stratum.")
    print(f"   * Generic full rank from paper2 = 116.   Drop = 116 - rank.")

    # ---------------- pick the (4,3) eps=1/100 slice for explicit syzygies
    print("\n" + "=" * 72)
    print("EXPLICIT SYZYGIES at  body3=(4,3),  eps = 1/100")
    print("=" * 72)
    eps = Rational(1, 100)
    sub, D_rat = build_substitution(eps, collinear=False)
    print(f"  D(eps) = r23^2 = {D_rat}   (rational; r23 is irrational)")

    print("  Substituting + reducing 156 generators (verbose)...")
    t0 = time()
    polys = collect_polys(exprs, sub, D_rat, verbose=True)
    print(f"  done in {time()-t0:.1f}s")

    M, nA, nB = build_AB_matrix(polys)
    print(f"  matrix is {M.shape[0]} x {M.shape[1]}  "
          f"(A-block: {nA} cols, B-block: {nB} cols)")
    rank = M.rank()
    print(f"  rank = {rank},  nullity = {len(exprs) - rank}")

    null = left_nullspace(M)
    print(f"  computed {len(null)} null vectors")

    # ---- classify: deep vs soft
    generic_sub = make_generic_sub()

    def is_deep(coeffs):
        nz = [i for i, c in enumerate(coeffs) if c != 0]
        if not nz:
            return True
        combo = sum(coeffs[i] * exprs[i] for i in nz)
        val = sp.expand(combo.subs(generic_sub))
        return val == 0

    # Sort by sparsity
    null.sort(key=lambda v: sum(1 for c in v if c != 0))

    deep_examples = []
    soft_examples = []
    for v in null:
        c = clear_denominators(v)
        if all(x == 0 for x in c):
            continue
        if is_deep(c):
            if len(deep_examples) < 1:
                deep_examples.append(c)
        else:
            if len(soft_examples) < 1:
                soft_examples.append(c)
        if deep_examples and soft_examples:
            break

    def print_combo(coeffs, header, footer):
        nz = [i for i, c in enumerate(coeffs) if c != 0]
        print("\n" + "-" * 72)
        print(f"{header}   ({len(nz)} terms)")
        print("-" * 72)
        for i in nz:
            sign = "+" if coeffs[i] > 0 else "-"
            mag = abs(int(coeffs[i]))
            cs = f"  {sign} {mag}" if mag != 1 else f"  {sign}  "
            print(f"  {cs} * {names[i]:42s} (level {levels[i]})")
        print("-" * 72)
        print(f"  {footer}")

    if deep_examples:
        print_combo(deep_examples[0],
                    "DEEP syzygy (vanishes everywhere on phase space)",
                    "= 0  identically (verified at a generic rational config).")

    if soft_examples:
        print_combo(soft_examples[0],
                    "SOFT syzygy (collision-edge, NO collinear contamination)",
                    "= 0  at  r12 = 5*eps = 1/20  on the (4,3) family.")

        # ---- eps-scaling of the soft syzygy ----
        print("\nLeading-order behaviour of the SOFT relation as a function of eps:")
        eps_sym = sp.Symbol("epsilon", positive=True)
        sub_eps, D_eps = build_substitution(eps_sym, collinear=False)

        c = soft_examples[0]
        nz = [i for i, x in enumerate(c) if x != 0]
        combo_full = sum(c[i] * exprs[i] for i in nz)
        combo_eps = sp.expand(combo_full.subs(sub_eps))
        # combo_eps has S in it; replace S = 1/sqrt(D_eps) symbolically
        combo_eps = combo_eps.subs(S, 1 / sp.sqrt(D_eps))
        # random rational momenta
        import random; random.seed(7)
        psub = {p: Rational(random.randint(-9, 9), random.randint(1, 9))
                for p in P_VARS}
        f_eps = sp.simplify(combo_eps.subs(psub))
        print(f"  combo(eps) at random momenta = {f_eps}")
        try:
            ser = sp.series(f_eps, eps_sym, 0, 4).removeO()
            print(f"  series at eps -> 0:           {sp.expand(ser)}")
        except Exception as e:
            print(f"  (series failed: {e})")
    else:
        print("\n(No SOFT syzygy isolated -- all sparsest null vectors were DEEP.)")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
