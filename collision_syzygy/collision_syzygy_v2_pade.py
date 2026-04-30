"""
Pade interpolation of the universal Q(eps) basis for the (4,3) binary-collision
left-null kernel of the planar 1/r d=2 level-3 algebra.

Consumes:  collision_syzygy_eps_lift.json  (8 eps samples, 76x156 Q-tables each)
Produces:  collision_syzygy_pade.json      (76x156 rational functions in eps)

For each (j, i) entry (j in 0..75 free-col index, i in 0..155 generator index)
we fit the 8 sample rationals c_k(eps_k) to

    c_ij(eps)  =  N_ij(eps) / ( eps^{a_ij} * D(eps)^{b_ij} )

with D(eps) = 25 - 48 eps + 25 eps^2 (= 1 / r23(eps)^2, the only irrationality
on the stratum), N_ij an integer polynomial in eps, and (a, b) chosen minimal
in a+2b.

Verification: with K=8 samples we can fit a unique polynomial of degree <= 7.
A valid Pade form with deg(N) <= 7 - a - 2b means the Lagrange interpolant of
(eps^a * D(eps)^b * c) has zero coefficients at degrees a + 2b + 1, ..., 7.
We accept the smallest (a,b) for which this self-consistency holds.

Free-column entries are constants (v[free_cols[j]] = 1 for j-th vector,
v[other free cols] = 0); only pivot-column entries are interpolated.

Wall: ~30 s end-to-end (pure rational arithmetic, no SymPy heavy machinery).
"""

from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path


IN_PATH  = "collision_syzygy_eps_lift.json"
OUT_PATH = "collision_syzygy_pade.json"

# Search order for (a, b): minimize a + 2*b, then a, then b.
AB_GRID = sorted(
    [(a, b) for a in range(0, 9) for b in range(0, 5)],
    key=lambda ab: (ab[0] + 2 * ab[1], ab[0], ab[1]),
)


def parse_q(s: str) -> Fraction:
    return Fraction(s)


def D_eval(eps: Fraction) -> Fraction:
    """D(eps) = 25 - 48 eps + 25 eps^2."""
    return 25 - 48 * eps + 25 * eps * eps


def lagrange_poly(xs: list[Fraction], ys: list[Fraction]) -> list[Fraction]:
    """
    Lagrange-interpolate K points to a polynomial of degree K-1.
    Returns coefficient list [c_0, c_1, ..., c_{K-1}] (low to high).
    All arithmetic exact in Fraction.
    """
    K = len(xs)
    # Start with coeffs = 0, accumulate sum_k y_k * L_k(x)
    coeffs = [Fraction(0)] * K
    for k in range(K):
        # Build L_k(x) = prod_{m != k} (x - xs[m]) / (xs[k] - xs[m])
        num: list[Fraction] = [Fraction(1)]  # polynomial 1
        denom = Fraction(1)
        for m in range(K):
            if m == k:
                continue
            # multiply num by (x - xs[m])
            new = [Fraction(0)] * (len(num) + 1)
            for i, c in enumerate(num):
                new[i] -= c * xs[m]
                new[i + 1] += c
            num = new
            denom *= xs[k] - xs[m]
        scale = ys[k] / denom
        for i, c in enumerate(num):
            coeffs[i] += c * scale
    return coeffs


def trim_zero_tail(coeffs: list[Fraction]) -> list[Fraction]:
    """Strip trailing zero coefficients."""
    out = list(coeffs)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_eval(coeffs: list[Fraction], x: Fraction) -> Fraction:
    """Horner's rule, low-to-high."""
    out = Fraction(0)
    for c in reversed(coeffs):
        out = out * x + c
    return out


def fit_pade(
    eps_vals: list[Fraction],
    samples: list[Fraction],
) -> tuple[int, int, list[Fraction]] | None:
    """
    Find minimal (a, b) such that P(eps) = eps^a * D(eps)^b * c(eps)
    is a polynomial of degree <= K - 2 - a - 2b  (one degree less than
    K-1-a-2b so that we get a real degree-of-freedom check).

    Procedure:  fit Lagrange poly through ALL K modified samples; require
    coefficients in degrees [d_target+1 .. K-1] to vanish, where
    d_target = K - 2 - a - 2b.  This means we are over-determined by
    one constraint per (a,b) -- if the ansatz holds, the leading coeff
    of the K-1-th degree fit must vanish, which is a non-trivial
    rationality test even at (a,b)=(0,0).

    Returns (a, b, numerator_coeffs) or None.
    """
    K = len(eps_vals)
    # Need at least one constraint above d_target, so d_target <= K-2.
    # That's d_target = K - 2 - a - 2b, requiring a + 2b >= 0 (always true).

    # All-zero shortcut
    if all(s == 0 for s in samples):
        return (0, 0, [Fraction(0)])

    for a, b in AB_GRID:
        d_target = K - 2 - a - 2 * b
        if d_target < 0:
            continue
        # Modified samples y_k = eps_k^a * D(eps_k)^b * c_k
        ys = []
        for k in range(K):
            e = eps_vals[k]
            de = D_eval(e)
            ys.append((e ** a) * (de ** b) * samples[k])
        # Lagrange-interpolate to a degree K-1 polynomial
        coeffs = lagrange_poly(eps_vals, ys)
        # Require coeffs in degrees [d_target+1 .. K-1] to vanish
        # (that's at least 1 vanishing constraint).
        ok = True
        for j in range(d_target + 1, K):
            if coeffs[j] != 0:
                ok = False
                break
        if ok:
            return (a, b, trim_zero_tail(coeffs[: d_target + 1]))
    return None


def loo_validate(
    eps_vals: list[Fraction],
    samples: list[Fraction],
    a: int,
    b: int,
    num: list[Fraction],
) -> bool:
    """Leave-one-out: refit (a,b)-form on K-1 samples, check the held-out one."""
    K = len(eps_vals)
    if all(s == 0 for s in samples):
        return all(s == 0 for s in samples)
    for hold in range(K):
        xs = [eps_vals[k] for k in range(K) if k != hold]
        ys = [(eps_vals[k] ** a) * (D_eval(eps_vals[k]) ** b) * samples[k]
              for k in range(K) if k != hold]
        # Fit a polynomial of degree <= K-2 through K-1 points
        # (equivalent to the unique interpolant of K-1 points).
        coeffs = lagrange_poly(xs, ys)
        # Check that the held-out point matches:
        e = eps_vals[hold]
        predicted_y = poly_eval(coeffs, e)
        actual_y = (e ** a) * (D_eval(e) ** b) * samples[hold]
        if predicted_y != actual_y:
            return False
    return True


def fmt_rational(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}"


def fmt_poly(coeffs: list[Fraction]) -> str:
    """Render polynomial low->high in eps as 'c0 + c1*eps + c2*eps^2 + ...'."""
    parts = []
    for j, c in enumerate(coeffs):
        if c == 0:
            continue
        if j == 0:
            parts.append(fmt_rational(c))
        elif j == 1:
            parts.append(f"({fmt_rational(c)})*eps")
        else:
            parts.append(f"({fmt_rational(c)})*eps^{j}")
    return " + ".join(parts) if parts else "0"


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("FOLLOW-UP A.5: Pade interpolation of the Q(eps) basis")
    print("=" * 72)

    print(f"\nLoading {IN_PATH} ...")
    with open(IN_PATH, "r") as f:
        D = json.load(f)

    assert D["pivot_set_constant"] is True, "pivot sets differ across eps -- abort"
    eps_vals = [parse_q(s) for s in D["eps_samples"]]
    K = len(eps_vals)
    print(f"  K = {K} samples: {D['eps_samples']}")

    pivot_set = sorted(D["pivot_sets"][0])
    free_cols = D["free_columns"][0]  # already a list, in some canonical order
    print(f"  pivots: {len(pivot_set)} columns")
    print(f"  free columns (basis indices): {len(free_cols)}")

    # Sanity: tables present
    tables = D["coefficient_tables"]
    assert set(tables.keys()) == set(D["eps_samples"]), \
        "table keys mismatch eps_samples"

    # Build per-(j, i) sample lists. j = 0..75 indexes free columns
    # in the order returned by the script (== D['free_columns'][k]).
    n_free = len(free_cols)
    n_components = len(tables[D["eps_samples"][0]][0])  # 156
    print(f"  table shape: {n_free} basis vectors x {n_components} components")

    # Convert each table into list[list[Fraction]] once
    print("\n[1/3] Parsing tables ...")
    parsed: dict[str, list[list[Fraction]]] = {}
    for k_str, tab in tables.items():
        parsed[k_str] = [[parse_q(x) for x in row] for row in tab]

    # Check the trivial constant entries first as a sanity test:
    # row j has v[free_cols[j]] = 1, v[other free cols] = 0.
    print("\n[2/3] Sanity: checking constant entries on free columns ...")
    for j in range(n_free):
        own = free_cols[j]
        for k_str in D["eps_samples"]:
            v = parsed[k_str][j]
            if v[own] != Fraction(1):
                raise AssertionError(
                    f"row j={j} own-free-col {own} != 1 at eps={k_str}: "
                    f"got {v[own]}"
                )
            for f in free_cols:
                if f == own:
                    continue
                if v[f] != 0:
                    raise AssertionError(
                        f"row j={j} other-free-col {f} != 0 at eps={k_str}: "
                        f"got {v[f]}"
                    )
    print("  PASS: free-col entries are exactly identity at every eps.")

    # Now interpolate the pivot entries.
    print(f"\n[3/3] Interpolating {n_free} x {len(pivot_set)} = "
          f"{n_free * len(pivot_set)} pivot entries ...")
    results: list[list[dict | None]] = [
        [None] * n_components for _ in range(n_free)
    ]
    # Stat counters
    ab_histogram: dict[tuple[int, int], int] = {}
    failures = 0
    nonzero = 0
    zero_count = 0
    deg_histogram: dict[int, int] = {}
    loo_failures = 0

    for j in range(n_free):
        # Constant (free-col) entries: store directly
        for f in free_cols:
            if f == free_cols[j]:
                results[j][f] = {
                    "kind": "const",
                    "a": 0, "b": 0,
                    "num": ["1/1"],  # P(eps) = 1
                }
            else:
                results[j][f] = {
                    "kind": "const",
                    "a": 0, "b": 0,
                    "num": ["0/1"],
                }

        for p in pivot_set:
            samples = [parsed[k_str][j][p] for k_str in D["eps_samples"]]
            fit = fit_pade(eps_vals, samples)
            if fit is None:
                failures += 1
                results[j][p] = {"kind": "FAIL", "samples":
                                 [fmt_rational(s) for s in samples]}
                continue
            a, b, num = fit
            if all(c == 0 for c in num):
                zero_count += 1
                results[j][p] = {"kind": "rat", "a": 0, "b": 0,
                                 "num": ["0/1"]}
            else:
                # Cross-validate with leave-one-out
                if not loo_validate(eps_vals, samples, a, b, num):
                    loo_failures += 1
                nonzero += 1
                ab_histogram[(a, b)] = ab_histogram.get((a, b), 0) + 1
                deg = len(num) - 1
                deg_histogram[deg] = deg_histogram.get(deg, 0) + 1
                results[j][p] = {
                    "kind": "rat", "a": a, "b": b,
                    "num": [fmt_rational(c) for c in num],
                }

        if (j + 1) % 10 == 0 or j == n_free - 1:
            print(f"  [{j + 1}/{n_free}] interpolated; "
                  f"failures={failures} loo_fail={loo_failures} "
                  f"zeros_in_pivots={zero_count} "
                  f"nonzero_pivots={nonzero}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s.")

    # Histograms
    print("\n=== (a, b) histogram over nonzero pivot entries ===")
    for ab in sorted(ab_histogram, key=lambda x: (x[0] + 2 * x[1], x[0], x[1])):
        print(f"   eps^{ab[0]} * D^{ab[1]}  : {ab_histogram[ab]:5d}")
    print(f"   total nonzero entries: {nonzero}")
    print(f"   zero entries (in pivots): {zero_count}")
    print(f"   FAILURES: {failures}")
    print(f"   LOO cross-validation failures: {loo_failures}")

    print("\n=== Numerator-degree histogram ===")
    for deg in sorted(deg_histogram):
        print(f"   deg(N) = {deg}: {deg_histogram[deg]:5d}")

    # Pick a sparsest example to print in human form
    print("\n=== Sparsest non-trivial basis vector (by # nonzero pivots) ===")
    best_j = None
    best_count = None
    for j in range(n_free):
        cnt = sum(
            1 for p in pivot_set
            if results[j][p]["kind"] == "rat"
            and any(c != "0/1" for c in results[j][p]["num"])
        )
        if cnt > 0 and (best_count is None or cnt < best_count):
            best_j = j
            best_count = cnt
    if best_j is not None:
        own = free_cols[best_j]
        print(f"\n  basis vector j={best_j}  (own free column = generator #{own})")
        print(f"  nonzero pivot entries: {best_count}")
        for p in pivot_set:
            r = results[best_j][p]
            if r["kind"] != "rat":
                continue
            if all(c == "0/1" for c in r["num"]):
                continue
            num_coeffs = [Fraction(c) for c in r["num"]]
            denom_str = ""
            if r["a"] > 0:
                denom_str += f" / eps^{r['a']}"
            if r["b"] > 0:
                denom_str += f" / D(eps)^{r['b']}"
            print(f"    v[{p:3d}] = ({fmt_poly(num_coeffs)}){denom_str}")

    # Save full table
    out = {
        "checkpoint": D.get("checkpoint", "checkpoints/level_3.pkl"),
        "stratum": D.get("stratum"),
        "eps_samples": D["eps_samples"],
        "pivot_set": pivot_set,
        "free_columns": free_cols,
        "denominator_polynomial": "D(eps) = 25 - 48*eps + 25*eps^2",
        "ab_histogram": {f"{a},{b}": v for (a, b), v in ab_histogram.items()},
        "degree_histogram": {str(k): v for k, v in deg_histogram.items()},
        "failures": failures,
        "results": results,
        "format": (
            "results[j][i] = {kind: 'const'|'rat'|'FAIL', "
            "a: int, b: int, num: [coeff_low ... coeff_high as 'p/q']}; "
            "v_j(eps)[i] = (sum_k num[k] * eps^k) / (eps^a * D(eps)^b)"
        ),
    }
    Path(OUT_PATH).write_text(json.dumps(out, indent=1))
    print(f"\nWrote {OUT_PATH}  ({Path(OUT_PATH).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
