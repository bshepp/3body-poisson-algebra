"""
Extend `collision_syzygy_eps_lift.json` from K=8 to K=12 ε-samples by
computing 4 additional ε values and merging.  Preserves existing data.

New ε's:  1/6, 1/8, 1/9, 1/13   (distinct from existing 1/{2,3,4,5,7,10,25,100})

Output:   collision_syzygy_eps_lift_k12.json
"""
from __future__ import annotations
import json
from time import time
from sympy import Rational

from collision_syzygy_v2_eps_lift import (
    sample_one, canonical_nullspace, rational_to_str,
)
from collision_syzygy_v2 import load_generators, CKPT

IN_PATH  = "collision_syzygy_eps_lift.json"
OUT_PATH = "collision_syzygy_eps_lift_k12.json"

NEW_EPS = [Rational(1, 6), Rational(1, 8), Rational(1, 9), Rational(1, 13)]


def main() -> None:
    print(f"Loading existing {IN_PATH} ...")
    with open(IN_PATH, "r", encoding="utf-8") as fh:
        D = json.load(fh)

    existing_eps_str = list(D["eps_samples"])
    print(f"  existing eps samples: {existing_eps_str}")
    print(f"  adding new eps:       {[str(e) for e in NEW_EPS]}")

    print("\nLoading generators ...")
    exprs, names, levels = load_generators()
    print(f"  {len(exprs)} generators")

    new_results = []
    t_start = time()
    for k, eps_val in enumerate(NEW_EPS, 1):
        print(f"\n[{k}/{len(NEW_EPS)}] eps = {eps_val} ...")
        t0 = time()
        r = sample_one(exprs, eps_val)
        dt = time() - t0
        print(f"   rank={r['rank']}  nullity={r['nullity']}  "
              f"shape={r['shape']}  ({dt:.1f}s)")
        new_results.append((eps_val, r))

        # Live partial dump
        merged = build_merged_obj(D, new_results[: k])
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
        print(f"   partial JSON written ({k}/{len(NEW_EPS)})")

    # Final consistency check
    all_pivot_sets = [set(s) for s in D["pivot_sets"]] + \
                     [set(r["pivots"]) for _, r in new_results]
    pivot_set_constant = all(s == all_pivot_sets[0] for s in all_pivot_sets[1:])
    print(f"\nPivot-set constancy across all 12 eps's: {pivot_set_constant}")

    merged = build_merged_obj(D, new_results,
                              pivot_set_constant=pivot_set_constant)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"\nWrote {OUT_PATH}  (total {time()-t_start:.1f}s)")


def build_merged_obj(D, new_results, pivot_set_constant=None):
    eps_strs = list(D["eps_samples"]) + [rational_to_str(e) for e, _ in new_results]
    ranks    = list(D["ranks"])     + [r["rank"]    for _, r in new_results]
    nullities= list(D["nullities"]) + [r["nullity"] for _, r in new_results]
    shapes   = list(D["shapes"])    + [list(r["shape"]) for _, r in new_results]
    nA       = list(D["nA"])        + [r["nA"]      for _, r in new_results]
    nB       = list(D["nB"])        + [r["nB"]      for _, r in new_results]
    pivot_sets   = list(D["pivot_sets"])   + [sorted(r["pivots"]) for _, r in new_results]
    free_columns = list(D["free_columns"]) + [sorted(r["free"])   for _, r in new_results]

    coef = dict(D["coefficient_tables"])
    for e, r in new_results:
        coef[rational_to_str(e)] = [
            [rational_to_str(c) for c in v] for v in r["basis"]
        ]

    if pivot_set_constant is None:
        all_sets = [set(s) for s in pivot_sets]
        pivot_set_constant = all(s == all_sets[0] for s in all_sets[1:])

    return {
        "checkpoint": CKPT,
        "stratum": D.get("stratum"),
        "eps_samples": eps_strs,
        "ranks": ranks,
        "nullities": nullities,
        "shapes": shapes,
        "nA": nA, "nB": nB,
        "pivot_sets": pivot_sets,
        "free_columns": free_columns,
        "free_column_count": [len(s) for s in free_columns],
        "pivot_set_constant": pivot_set_constant,
        "coefficient_tables": coef,
    }


if __name__ == "__main__":
    main()
