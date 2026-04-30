"""
K=12 variant of collision_syzygy_v2_pade.py.

Loads collision_syzygy_eps_lift_k12.json (12 eps samples),
allows acceptance bound deg(N) <= K - 2 - a - 2b = 10 - a - 2b
(was 6 - a - 2b at K=8, so the AB grid extends to a + 2b <= 10).

Output: collision_syzygy_pade_k12.json
"""
from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import collision_syzygy_v2_pade as P


IN_PATH  = "collision_syzygy_eps_lift_k12.json"
OUT_PATH = "collision_syzygy_pade_k12.json"

# Extend the (a,b) search grid to cover larger denominator support
AB_GRID_K12 = sorted(
    [(a, b) for a in range(0, 13) for b in range(0, 7)],
    key=lambda ab: (ab[0] + 2 * ab[1], ab[0], ab[1]),
)


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("FOLLOW-UP A.5 (K=12): Padé interpolation of the Q(eps) basis")
    print("=" * 72)

    # Monkey-patch the search grid for the larger K
    P.AB_GRID = AB_GRID_K12

    print(f"\nLoading {IN_PATH} ...")
    with open(IN_PATH, "r") as f:
        D = json.load(f)

    assert D["pivot_set_constant"] is True, "pivot sets differ across eps"
    eps_vals = [P.parse_q(s) for s in D["eps_samples"]]
    K = len(eps_vals)
    print(f"  K = {K} samples: {D['eps_samples']}")
    assert K == 12, f"expected K=12, got K={K}"

    pivot_set = sorted(D["pivot_sets"][0])
    free_cols = D["free_columns"][0]
    print(f"  pivots: {len(pivot_set)} columns")
    print(f"  free columns: {len(free_cols)}")

    tables = D["coefficient_tables"]
    n_free = len(free_cols)
    n_components = len(tables[D["eps_samples"][0]][0])
    print(f"  table shape: {n_free} basis vectors x {n_components} components")

    print("\n[1/3] Parsing tables ...")
    parsed: dict[str, list[list[Fraction]]] = {}
    for k_str, tab in tables.items():
        parsed[k_str] = [[P.parse_q(x) for x in row] for row in tab]

    print(f"\n[2/3] Sanity: free-col entries identity at every eps ...")
    for j in range(n_free):
        own = free_cols[j]
        for k_str in D["eps_samples"]:
            v = parsed[k_str][j]
            assert v[own] == Fraction(1), f"j={j} own={own} eps={k_str}"
            for f in free_cols:
                if f == own:
                    continue
                assert v[f] == 0, f"j={j} other={f} eps={k_str}"
    print("  PASS.")

    print(f"\n[3/3] Interpolating {n_free} x {len(pivot_set)} pivot entries ...")
    results: list[list[dict | None]] = [
        [None] * n_components for _ in range(n_free)
    ]
    ab_histogram: dict[tuple[int, int], int] = {}
    failures = 0
    nonzero = 0
    zero_count = 0
    deg_histogram: dict[int, int] = {}
    loo_failures = 0

    for j in range(n_free):
        for f in free_cols:
            if f == free_cols[j]:
                results[j][f] = {"kind": "const", "a": 0, "b": 0, "num": ["1/1"]}
            else:
                results[j][f] = {"kind": "const", "a": 0, "b": 0, "num": ["0/1"]}

        for p in pivot_set:
            samples = [parsed[k_str][j][p] for k_str in D["eps_samples"]]
            fit = P.fit_pade(eps_vals, samples)
            if fit is None:
                failures += 1
                results[j][p] = {"kind": "FAIL",
                                 "samples": [P.fmt_rational(s) for s in samples]}
                continue
            a, b, num = fit
            if all(c == 0 for c in num):
                zero_count += 1
                results[j][p] = {"kind": "rat", "a": 0, "b": 0, "num": ["0/1"]}
            else:
                if not P.loo_validate(eps_vals, samples, a, b, num):
                    loo_failures += 1
                nonzero += 1
                ab_histogram[(a, b)] = ab_histogram.get((a, b), 0) + 1
                deg = len(num) - 1
                deg_histogram[deg] = deg_histogram.get(deg, 0) + 1
                results[j][p] = {"kind": "rat", "a": a, "b": b,
                                 "num": [P.fmt_rational(c) for c in num]}

        if (j + 1) % 10 == 0 or j == n_free - 1:
            print(f"  [{j + 1}/{n_free}] failures={failures} "
                  f"loo_fail={loo_failures} zeros={zero_count} "
                  f"nonzero={nonzero}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s.")

    print("\n=== (a, b) histogram over nonzero pivot entries ===")
    for ab in sorted(ab_histogram, key=lambda x: (x[0] + 2 * x[1], x[0], x[1])):
        print(f"   eps^{ab[0]} * D^{ab[1]}  : {ab_histogram[ab]:5d}")
    print(f"   total nonzero: {nonzero}")
    print(f"   zero entries (in pivots): {zero_count}")
    print(f"   FAILURES: {failures}")
    print(f"   LOO failures: {loo_failures}")

    print("\n=== Numerator-degree histogram ===")
    for deg in sorted(deg_histogram):
        print(f"   deg(N) = {deg}: {deg_histogram[deg]:5d}")

    # Classify rows: PURE (only kind=const, a=b=0, num all integers) vs MIXED vs FAIL
    print("\n=== Per-row classification ===")
    pure_rows = []
    mixed_rows = []
    fail_rows = []
    for j in range(n_free):
        any_fail = any(results[j][p]["kind"] == "FAIL" for p in pivot_set)
        if any_fail:
            fail_rows.append(j)
            continue
        any_eps = any(
            results[j][p]["kind"] == "rat"
            and (results[j][p]["a"] != 0 or results[j][p]["b"] != 0
                 or len(results[j][p]["num"]) > 1)
            for p in pivot_set
        )
        if any_eps:
            mixed_rows.append(j)
        else:
            pure_rows.append(j)
    print(f"   PURE-constant rows : {len(pure_rows)}   indices: {pure_rows}")
    print(f"   MIXED rows         : {len(mixed_rows)}")
    print(f"   FAIL rows          : {len(fail_rows)}")

    out = {
        "checkpoint": D.get("checkpoint", "checkpoints/level_3.pkl"),
        "stratum": D.get("stratum"),
        "K": K,
        "eps_samples": D["eps_samples"],
        "pivot_set": pivot_set,
        "free_columns": free_cols,
        "denominator_polynomial": "D(eps) = 25 - 48*eps + 25*eps^2",
        "ab_histogram": {f"{a},{b}": v for (a, b), v in ab_histogram.items()},
        "degree_histogram": {str(k): v for k, v in deg_histogram.items()},
        "failures": failures,
        "pure_rows": pure_rows,
        "mixed_rows": mixed_rows,
        "fail_rows": fail_rows,
        "results": results,
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=1))
    print(f"\nWrote {OUT_PATH}  ({Path(OUT_PATH).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
