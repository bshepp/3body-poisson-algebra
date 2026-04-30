#!/usr/bin/env python3
"""
collision_syzygy_v2_collinear.py
================================

Follow-up B: classify the 24 extra syzygies that appear on the
(3,4) collinear cap (rank 56, nullity 100) but NOT on the
(4,3) non-collinear binary-collision family (rank 80, nullity 76).

Strategy
--------
1. Build the exact ℚ left-nullspace N_col on the (3,4) ε=1/100 slice
   (100 vectors).
2. Build the exact ℚ left-nullspace N_43 on the (4,3) ε=1/100 slice
   (76 vectors).
3. For each null vector v in N_col, classify on TWO axes:
   - generic test: does Σ vₖ Gₖ vanish at the generic rational config
     `make_generic_sub()`?  →  DEEP (yes) vs SOFT (no).
   - (4,3) test: does v lie in the row-span of N_43?  Equivalently,
     does Σ vₖ Gₖ also vanish on the (4,3) ε=1/100 slice?  →
     "binary-also" (yes) vs "collinear-only" (no).
4. Bucket counts + one explicit example per bucket.

Imports collision_syzygy_v2 directly; does not modify it.
"""

import os, sys, pickle
from time import time

sys.setrecursionlimit(500000)
os.environ["PYTHONUNBUFFERED"] = "1"

import sympy as sp
from sympy import Rational, Integer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collision_syzygy_v2 as v2
from collision_syzygy_v2 import (
    load_generators, build_substitution, collect_polys,
    build_AB_matrix, left_nullspace, clear_denominators,
    make_generic_sub,
)


def vanishes_on(coeffs, exprs, sub_with_S, D_rat):
    """Check whether Σ cₖ Gₖ vanishes on the slice defined by sub_with_S
    after substituting and reducing S²=1/D_rat.

    Implementation: build (A,B) for the linear combination and check
    A == 0 and B == 0.  We do this term-by-term to avoid summing all 156
    enormous expressions.
    """
    nz = [i for i, c in enumerate(coeffs) if c != 0]
    if not nz:
        return True
    # Use already-computed (A_k, B_k) is the efficient path; here we
    # accept the function form for reuse.
    raise NotImplementedError("use vanishes_on_polys() instead")


def vanishes_on_polys(coeffs, polys):
    """polys is the list of (A_k, B_k) on a given slice.  Combination
    vanishes iff Σ cₖ Aₖ = 0 and Σ cₖ Bₖ = 0."""
    A_sum = sp.Integer(0)
    B_sum = sp.Integer(0)
    for c, (A, B) in zip(coeffs, polys):
        if c == 0:
            continue
        if A != 0:
            A_sum += c * A
        if B != 0:
            B_sum += c * B
    return sp.expand(A_sum) == 0 and sp.expand(B_sum) == 0


def vanishes_generic(coeffs, exprs, generic_sub):
    nz = [i for i, c in enumerate(coeffs) if c != 0]
    if not nz:
        return True
    combo = sum(coeffs[i] * exprs[i] for i in nz)
    return sp.expand(combo.subs(generic_sub)) == 0


def main():
    print("=" * 72)
    print("FOLLOW-UP B: classify the 24 extra collinear syzygies")
    print("=" * 72)

    print(f"\nLoading {v2.CKPT} ...")
    exprs, names, levels = load_generators()
    print(f"  {len(exprs)} generators loaded")

    eps = Rational(1, 100)

    # ---------------- (3,4) collinear slice
    print("\n[1/4] Building (3,4) collinear slice at eps=1/100 ...")
    sub_col, D_col = build_substitution(eps, collinear=True)
    t0 = time()
    polys_col = collect_polys(exprs, sub_col, D_col, verbose=False)
    print(f"      collect_polys: {time()-t0:.1f}s")
    M_col, nA, nB = build_AB_matrix(polys_col)
    rank_col = M_col.rank()
    null_col = left_nullspace(M_col)
    print(f"      rank = {rank_col}, nullity = {len(null_col)}")

    # ---------------- (4,3) non-collinear slice
    print("\n[2/4] Building (4,3) non-collinear slice at eps=1/100 ...")
    sub_43, D_43 = build_substitution(eps, collinear=False)
    t0 = time()
    polys_43 = collect_polys(exprs, sub_43, D_43, verbose=False)
    print(f"      collect_polys: {time()-t0:.1f}s")
    M_43, _, _ = build_AB_matrix(polys_43)
    rank_43 = M_43.rank()
    null_43 = left_nullspace(M_43)
    print(f"      rank = {rank_43}, nullity = {len(null_43)}")

    # ---------------- generic config (oracle for DEEP)
    generic_sub = make_generic_sub()

    # ---------------- classify each col null vector
    print("\n[3/4] Classifying 100 collinear null vectors ...")
    print("      axis 1: vanishes at generic config? (DEEP vs SOFT)")
    print("      axis 2: vanishes on (4,3) family at eps=1/100?")
    print("              (binary-also vs collinear-only)")

    buckets = {
        ("DEEP", "binary-also"):     [],
        ("DEEP", "collinear-only"):  [],
        ("SOFT", "binary-also"):     [],
        ("SOFT", "collinear-only"):  [],
    }

    n = len(null_col)
    for i, v in enumerate(null_col):
        c = clear_denominators(v)
        if all(x == 0 for x in c):
            continue
        deep = vanishes_generic(c, exprs, generic_sub)
        binary = vanishes_on_polys(c, polys_43)
        key = ("DEEP" if deep else "SOFT",
               "binary-also" if binary else "collinear-only")
        buckets[key].append((i, c))
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"      [{i+1}/{n}] "
                  f"DEEP/bin={len(buckets[('DEEP','binary-also')])} "
                  f"DEEP/col={len(buckets[('DEEP','collinear-only')])} "
                  f"SOFT/bin={len(buckets[('SOFT','binary-also')])} "
                  f"SOFT/col={len(buckets[('SOFT','collinear-only')])}")

    # ---------------- report
    print("\n[4/4] Bucket counts (should sum to 100):")
    print(f"   DEEP, binary-also     : {len(buckets[('DEEP','binary-also')]):3d}")
    print(f"   DEEP, collinear-only  : {len(buckets[('DEEP','collinear-only')]):3d}")
    print(f"   SOFT, binary-also     : {len(buckets[('SOFT','binary-also')]):3d}")
    print(f"   SOFT, collinear-only  : {len(buckets[('SOFT','collinear-only')]):3d}")
    total = sum(len(v) for v in buckets.values())
    print(f"   TOTAL                 : {total:3d}")
    print()
    print("Sanity: nullity((3,4)) - nullity((4,3)) = "
          f"{len(null_col)} - {len(null_43)} = {len(null_col) - len(null_43)}")
    print("Expectation: SOFT/collinear-only (the new collinear cap)")
    print("             = 24 (modulo any DEEP/collinear-only).")

    # ---------------- one example per non-empty bucket, sparsest
    def print_example(coeffs, header):
        nz = [i for i, c in enumerate(coeffs) if c != 0]
        print("\n" + "-" * 72)
        print(f"{header}   ({len(nz)} terms)")
        print("-" * 72)
        for i in nz:
            sign = "+" if coeffs[i] > 0 else "-"
            mag = abs(int(coeffs[i]))
            cs = f"  {sign} {mag}" if mag != 1 else f"  {sign}  "
            print(f"  {cs} * {names[i]:42s} (level {levels[i]})")

    # ---------------- dump all witnesses to JSON
    import json
    DUMP_PATH = "syzygy_v2_collinear_witnesses.json"
    dump = {
        "checkpoint": v2.CKPT,
        "stratum_collinear": "(3,4) collinear: body3=(0,3), body2=(0,4ε)",
        "stratum_binary":    "(4,3) binary:    body3=(4,3), body2=(3ε,4ε)",
        "eps": str(eps),
        "nullity_collinear": len(null_col),
        "nullity_binary":    len(null_43),
        "n_generators":      len(exprs),
        "names":             list(names),
        "levels":            [int(l) for l in levels],
        "buckets": {
            f"{k[0]}/{k[1]}": [
                {
                    "null_index": int(idx),
                    "n_terms": sum(1 for c in coeffs if c != 0),
                    "coeffs": {
                        str(i): str(c) for i, c in enumerate(coeffs) if c != 0
                    },
                }
                for idx, coeffs in items
            ]
            for k, items in buckets.items()
        },
        "bucket_counts": {
            f"{k[0]}/{k[1]}": len(v) for k, v in buckets.items()
        },
    }
    with open(DUMP_PATH, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, indent=1)
    sz = os.path.getsize(DUMP_PATH)
    print(f"\nWrote {DUMP_PATH}  ({sz} bytes)")

    print("\n" + "=" * 72)
    print("EXAMPLES (sparsest in each non-empty bucket)")
    print("=" * 72)
    for key, items in buckets.items():
        if not items:
            print(f"\n[{key[0]} / {key[1]}]  (empty)")
            continue
        items_sorted = sorted(
            items, key=lambda ic: sum(1 for c in ic[1] if c != 0)
        )
        idx, c = items_sorted[0]
        print_example(c, f"[{key[0]} / {key[1]}]  null-vector index {idx}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
