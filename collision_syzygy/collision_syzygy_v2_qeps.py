"""
collision_syzygy_v2_qeps.py
===========================

Compute the EXACT left-null kernel of the planar 1/r d=2 level-3
collision matrix on the (4,3) binary stratum over the field Q(eps),
NOT just at sample ε's.

Strategy
--------
1. Make eps a sympy Symbol.
2. body₁=(0,0), body₂=(3ε,4ε), body₃=(4,3); u₁₂=1/(5ε), u₁₃=1/5, u₂₃=S.
3. After expansion + S-reduction (S² = 1/D(ε), D = 25-48ε+25ε²), each
   generator G_k = A_k(p,ε) + B_k(p,ε)·S where A_k, B_k ∈ Q(ε)[p_vars].
4. Collect (A,B) coefficients of each p-monomial → 156×(nA+nB) matrix
   with entries in Q(ε).
5. Build DomainMatrix over QQ.frac_field('eps'); compute rank & nullspace
   exactly.

Output: collision_syzygy_qeps.json  with rank, nullity, pivot/free cols,
        and the canonical Q(ε)-basis vectors as rational functions of ε.

Wall: heavy.  Targeting AWS r6i.4xlarge (~128 GB RAM headroom, 16 vCPU).
"""
from __future__ import annotations

import os, sys, json, pickle
from time import time

sys.setrecursionlimit(500000)
os.environ["PYTHONUNBUFFERED"] = "1"

import sympy as sp
from sympy import Symbol, Rational, Integer, Poly, symbols
from sympy.polys.matrices import DomainMatrix
from sympy.polys.domains import QQ

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collision_syzygy_v2 as v2
from collision_syzygy_v2 import (
    load_generators, S, P_VARS,
    x1, y1, x2, y2, x3, y3, u12, u13, u23,
)

OUT_PATH = "collision_syzygy_qeps.json"

eps_sym = Symbol("eps")


def build_substitution_symbolic():
    """Same geometry as v2.build_substitution(eps_sym, collinear=False)
    but eps is a sympy Symbol instead of a Rational."""
    D = 25 - 48 * eps_sym + 25 * eps_sym ** 2
    sub = {
        x1: 0, y1: 0,
        x2: 3 * eps_sym, y2: 4 * eps_sym,
        x3: 4,           y3: 3,
        u12: Rational(1, 5) / eps_sym,
        u13: Rational(1, 5),
        u23: S,
    }
    return sub, sp.together(D)


def reduce_S_sym(poly_in_S, D_rat):
    """Reduce S^k mod (S^2 - 1/D_rat).  Coefficients may contain eps."""
    p = Poly(poly_in_S, S)
    deg = p.degree()
    if deg <= 1:
        return poly_in_S
    invD = 1 / D_rat
    coeffs = p.all_coeffs()  # highest first
    while len(coeffs) > 2:
        c = coeffs[0]
        coeffs[2] = sp.together(coeffs[2] + c * invD)
        coeffs = coeffs[1:]
        if len(coeffs) == 1:
            coeffs = [Integer(0), coeffs[0]]
    B, A = coeffs[0], coeffs[1]
    return sp.expand(A + B * S)


def split_AB_sym(reduced_expr):
    """Return (A,B) where reduced_expr = A + B·S; coeffs in Q(eps)[p_vars]."""
    p = Poly(reduced_expr, S)
    A = Integer(0)
    B = Integer(0)
    for monom, coef in p.as_dict().items():
        s_pow = monom[0]
        if s_pow == 0:
            A = A + coef
        elif s_pow == 1:
            B = B + coef
        else:
            raise RuntimeError(f"unexpected S^{s_pow}")
    return sp.expand(A), sp.expand(B)


def collect_polys_sym(exprs, sub, D_rat, verbose=True):
    out = []
    n = len(exprs)
    t0 = time()
    for i, e in enumerate(exprs):
        if verbose and (i % 5 == 0 or i == n - 1):
            print(f"    [{i+1}/{n}]  ({time()-t0:.1f}s)", flush=True)
        ex = sp.expand(e.subs(sub))
        ex_red = reduce_S_sym(ex, D_rat)
        A, B = split_AB_sym(ex_red)
        out.append((A, B))
    return out


def build_AB_matrix_qeps(polys):
    """Build DomainMatrix over K = QQ.frac_field('eps').
    Returns (M, nA, nB, monoms_A, monoms_B)."""
    K = QQ.frac_field("eps")
    eps_K = K.gens[0]

    # Convert each (A,B) to Poly over P_VARS with coeffs in K
    poly_dicts_A = []
    poly_dicts_B = []
    monoms_A: set = set()
    monoms_B: set = set()

    def to_K_poly(expr):
        if expr == 0:
            return {}
        # Build sympy Poly over P_VARS with eps as the "extra" symbol
        # then convert each coefficient (which is a rational function of eps)
        # to the K domain.
        p = Poly(expr, *P_VARS)  # coeffs are sympy expressions in eps
        d = {}
        for monom, coef in p.as_dict().items():
            # coef is sympy expression (rational in eps)
            d[monom] = K.from_sympy(sp.together(coef))
        return d

    n = len(polys)
    t0 = time()
    for i, (A, B) in enumerate(polys):
        if i % 10 == 0 or i == n - 1:
            print(f"    convert [{i+1}/{n}]  ({time()-t0:.1f}s)", flush=True)
        dA = to_K_poly(A)
        dB = to_K_poly(B)
        poly_dicts_A.append(dA)
        poly_dicts_B.append(dB)
        monoms_A.update(dA.keys())
        monoms_B.update(dB.keys())

    mlist_A = sorted(monoms_A)
    mlist_B = sorted(monoms_B)
    idx_A = {m: j for j, m in enumerate(mlist_A)}
    idx_B = {m: j for j, m in enumerate(mlist_B)}
    nA, nB = len(mlist_A), len(mlist_B)
    n_gen = len(polys)

    print(f"  matrix shape: {n_gen} x {nA + nB}  (nA={nA}, nB={nB})")
    rows = []
    for dA, dB in zip(poly_dicts_A, poly_dicts_B):
        row = [K.zero] * (nA + nB)
        for m, c in dA.items():
            row[idx_A[m]] = c
        for m, c in dB.items():
            row[nA + idx_B[m]] = c
        rows.append(row)
    M = DomainMatrix(rows, (n_gen, nA + nB), K)
    return M, nA, nB, mlist_A, mlist_B, K


def main():
    print("=" * 72)
    print("FOLLOW-UP A.6: exact Q(eps) left-null kernel on (4,3) stratum")
    print("=" * 72)
    t_start = time()

    print(f"\nLoading {v2.CKPT} ...")
    exprs, names, levels = load_generators()
    print(f"  {len(exprs)} generators")

    print("\n[1/4] Building symbolic substitution (eps = Symbol) ...")
    sub, D_rat = build_substitution_symbolic()
    print(f"  D(eps) = {sp.expand(D_rat)}")

    print("\n[2/4] Substituting + reducing S^2  -> (A_k, B_k) over Q(eps)[p] ...")
    polys = collect_polys_sym(exprs, sub, D_rat, verbose=True)
    print(f"  done in {time()-t_start:.1f}s")

    print("\n[3/4] Building DomainMatrix over Q(eps) ...")
    M, nA, nB, mlist_A, mlist_B, K = build_AB_matrix_qeps(polys)
    print(f"  total elapsed: {time()-t_start:.1f}s")

    print("\n[4/4] Computing rank + canonical nullspace via RREF ...")
    t0 = time()
    MT = M.transpose()
    # Use rref_den to get the row-reduced echelon form
    R, den = MT.rref_den() if hasattr(MT, "rref_den") else (MT.rref()[0], None)
    print(f"  RREF: {time()-t0:.1f}s")

    # Determine rank from nonzero rows of R
    n_rows, n_cols = R.shape
    pivot_cols = []
    pivot_rows = []
    r = 0
    for j in range(n_cols):
        # find first nonzero entry below row r in column j of R
        found = None
        for i in range(r, n_rows):
            if R[i, j].element != K.zero.element:
                found = i
                break
        if found is not None:
            if found != r:
                pass  # already echelon; assume rref returned canonical
            pivot_cols.append(j)
            pivot_rows.append(r)
            r += 1
            if r == n_rows:
                break
    rank = r
    nullity = n_cols - rank
    print(f"  rank(M) = {rank}  nullity(M^T) = {nullity}")
    print(f"  matrix shape was {M.shape}; transpose {n_rows}x{n_cols}")

    free_cols = [j for j in range(n_cols) if j not in set(pivot_cols)]
    print(f"  pivot cols: {len(pivot_cols)}  free cols: {len(free_cols)}")

    # Build canonical basis: for each free col f, set v[f]=1, v[other free]=0,
    # then v[pivot[k]] = -R[pivot_rows[k], f] for k in pivots.
    print("  building canonical basis vectors ...")
    basis = []
    for f in free_cols:
        v = [K.zero] * n_cols
        v[f] = K.one
        for k, pcol in enumerate(pivot_cols):
            v[pcol] = -R[pivot_rows[k], f]
        basis.append(v)

    # Format for JSON: each entry is a sympy expression
    print("  formatting basis as sympy strings ...")
    basis_str = []
    for v in basis:
        row = []
        for c in v:
            row.append(str(K.to_sympy(c)))
        basis_str.append(row)

    print(f"\nTotal wall: {time()-t_start:.1f}s")

    out = {
        "checkpoint": v2.CKPT,
        "stratum": "(4,3) binary-collision  body3=(4,3), body2=(3ε,4ε)",
        "field": "Q(eps)",
        "n_generators": len(exprs),
        "names": list(names),
        "levels": [int(l) for l in levels],
        "matrix_shape": [int(M.shape[0]), int(M.shape[1])],
        "nA": nA, "nB": nB,
        "rank": rank,
        "nullity": nullity,
        "pivot_cols": pivot_cols,
        "free_cols": free_cols,
        "basis": basis_str,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    sz = os.path.getsize(OUT_PATH)
    print(f"\nWrote {OUT_PATH}  ({sz} bytes)")


if __name__ == "__main__":
    main()
