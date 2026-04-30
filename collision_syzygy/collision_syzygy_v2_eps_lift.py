#!/usr/bin/env python3
"""
collision_syzygy_v2_eps_lift.py
================================

Follow-up A: lift the 76-dimensional rank-80 ℚ-nullspace at a single ε
to a family-wide ℚ(ε)-nullspace on the (4,3) binary-collision stratum.

Strategy
--------
The pipeline of `collision_syzygy_v2.py` produces, on the (4,3) slice
body₃ = (4,3), body₂ = (3ε,4ε), an exact ℚ matrix M(ε) of shape
156 × (nA(ε)+nB(ε)) such that left-null vectors c of M(ε) parameterize
syzygies among the 156 generators on that ε-slice.

We sample at K rational ε values, compute the canonical (pivot-set)
nullspace at each, verify the pivot set is independent of ε (necessary
for a single ℚ(ε)-basis to exist), then dump the (rational) coefficient
tables to JSON for interpolation.

For each canonical null vector v_j(ε) (indexed by free column j of the
row-reduced M(ε)ᵀ), every entry v_j[i](ε) is a rational function in ε.
With K ≥ deg(num)+deg(den)+1 distinct sample points, classical rational
function interpolation (Cauchy / Padé) recovers it exactly.

Outputs
-------
  collision_syzygy_eps_lift.json
    {
      "checkpoint":   "checkpoints/level_3.pkl",
      "stratum":      "(4,3) binary-collision",
      "eps_samples":  ["1/3", "1/5", ...],
      "ranks":        [80, 80, ...],
      "nullities":    [76, 76, ...],
      "pivot_set_constant": true,
      "free_column_count": 76,
      "coefficient_tables": {
          "<eps>": [[c00, c01, ...], [c10, ...], ...]   # 76 × 156, rationals as "p/q"
      }
    }

This is a launching pad: actual rational-function interpolation is left
to a downstream sympy script that consumes the JSON.
"""

import os, sys, json, pickle
from time import time
from fractions import Fraction

sys.setrecursionlimit(500000)
os.environ["PYTHONUNBUFFERED"] = "1"

import sympy as sp
from sympy import Rational, QQ
from sympy.polys.matrices import DomainMatrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collision_syzygy_v2 as v2
from collision_syzygy_v2 import (
    load_generators, build_substitution, collect_polys, build_AB_matrix,
)


OUT_PATH = "collision_syzygy_eps_lift.json"

# Default ε samples — kept small to bound CPU.  Each sample costs
# ~400s of substitution + ~few s of nullspace.
DEFAULT_EPS = [
    Rational(1, 2),
    Rational(1, 3),
    Rational(1, 4),
    Rational(1, 5),
    Rational(1, 7),
    Rational(1, 10),
    Rational(1, 25),
    Rational(1, 100),
]


def canonical_nullspace(M):
    """Compute the canonical nullspace of M^T (i.e. the left nullspace
    of M) using the row-reduced echelon form of M^T.  Returns
    (pivot_cols, free_cols, basis_rationals) where:
        - pivot_cols: tuple of column indices of M^T that are pivots
        - free_cols : tuple of column indices of M^T that are free
        - basis_rationals: list of len(free_cols) lists, each of length
              M.shape[0], giving a rational left-null vector of M
              (= right-null of M^T).  In each basis vector, the entry at
              the corresponding free column is +1 and entries at other
              free columns are 0.
    """
    MT = M.transpose()           # (nA+nB, 156)
    R, pivots = MT.rref()        # pivots: tuple of column indices
    nrows, ncols = MT.shape
    pivots = tuple(pivots)
    free = tuple(j for j in range(ncols) if j not in set(pivots))

    # Build pivot row map: pivot_col -> row_index in R
    pivot_row_of = {}
    row = 0
    for p in pivots:
        pivot_row_of[p] = row
        row += 1

    basis = []
    for j in free:
        v = [QQ.zero] * ncols
        v[j] = QQ.one
        for p in pivots:
            r = pivot_row_of[p]
            # v[p] = -R[r, j]
            v[p] = -R.rep.getitem(r, j)
        # Convert to sympy Rationals
        v_sym = []
        for x in v:
            if hasattr(x, "numerator"):
                v_sym.append(Rational(int(x.numerator), int(x.denominator)))
            else:
                v_sym.append(sp.Rational(x))
        basis.append(v_sym)

    return pivots, free, basis


def rational_to_str(r):
    if r == 0:
        return "0"
    return f"{r.p}/{r.q}" if r.q != 1 else str(r.p)


def sample_one(exprs, eps_val):
    sub, D_rat = build_substitution(eps_val, collinear=False)
    polys = collect_polys(exprs, sub, D_rat, verbose=False)
    M, nA, nB = build_AB_matrix(polys)
    rank = M.rank()
    nullity = M.shape[0] - rank
    pivots, free, basis = canonical_nullspace(M)
    return {
        "rank": rank,
        "nullity": nullity,
        "shape": (M.shape[0], M.shape[1]),
        "nA": nA, "nB": nB,
        "pivots": pivots,
        "free": free,
        "basis": basis,
    }


def main():
    print("=" * 72)
    print("FOLLOW-UP A: lift 76-dim nullspace from Q to Q(eps)")
    print("            on the (4,3) binary-collision stratum")
    print("=" * 72)

    print(f"\nLoading {v2.CKPT} ...")
    exprs, names, levels = load_generators()
    print(f"  {len(exprs)} generators loaded")

    eps_values = DEFAULT_EPS
    print(f"\nSampling at K = {len(eps_values)} eps-values: "
          + ", ".join(str(e) for e in eps_values))

    results = []
    pivot_sets = []
    t_start = time()
    for k, eps_val in enumerate(eps_values, 1):
        print(f"\n[{k}/{len(eps_values)}] eps = {eps_val} ...")
        t0 = time()
        r = sample_one(exprs, eps_val)
        dt = time() - t0
        print(f"          rank={r['rank']}  nullity={r['nullity']}  "
              f"matrix={r['shape'][0]}×{r['shape'][1]}  ({dt:.1f}s)")
        print(f"          pivot_count={len(r['pivots'])}  "
              f"free_count={len(r['free'])}")
        results.append((eps_val, r))
        pivot_sets.append(set(r["pivots"]))

        # Live-write partial JSON every iteration
        write_partial_json(eps_values[:k], [x[1] for x in results])

    pivot_set_constant = all(s == pivot_sets[0] for s in pivot_sets[1:])
    print(f"\nPivot-set constancy across all {len(eps_values)} eps's: "
          f"{pivot_set_constant}")
    if not pivot_set_constant:
        print("  ! Pivot sets differ -- cannot use a single Q(eps) basis.")
        print("  ! Differing indices:")
        ref = pivot_sets[0]
        for i, s in enumerate(pivot_sets[1:], 1):
            diff = (s - ref) | (ref - s)
            if diff:
                print(f"     eps={eps_values[i]}: symmetric diff size "
                      f"{len(diff)}")

    # Final JSON dump
    write_partial_json(eps_values, [x[1] for x in results],
                       pivot_set_constant=pivot_set_constant)
    print(f"\nWrote {OUT_PATH}  (total {time()-t_start:.1f}s)")
    print("\n" + "=" * 72)
    print("DONE.  Next step (offline): rational-function interpolation")
    print("of each canonical basis entry coefficient_tables[eps][j][i]")
    print("as a function of eps, using the K >= deg+1 sample points.")
    print("=" * 72)


def write_partial_json(eps_values, results, pivot_set_constant=None):
    obj = {
        "checkpoint": v2.CKPT,
        "stratum": "(4,3) binary-collision  body3=(4,3), body2=(3ε,4ε)",
        "eps_samples": [rational_to_str(e) for e in eps_values],
        "ranks":     [r["rank"]     for r in results],
        "nullities": [r["nullity"]  for r in results],
        "shapes":    [list(r["shape"]) for r in results],
        "nA":        [r["nA"]       for r in results],
        "nB":        [r["nB"]       for r in results],
        "pivot_sets": [sorted(r["pivots"]) for r in results],
        "free_columns": [sorted(r["free"]) for r in results],
        "free_column_count": [len(r["free"]) for r in results],
        "pivot_set_constant": pivot_set_constant,
        "coefficient_tables": {
            rational_to_str(e): [[rational_to_str(c) for c in v]
                                 for v in r["basis"]]
            for e, r in zip(eps_values, results)
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


if __name__ == "__main__":
    main()
