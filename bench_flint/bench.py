#!/usr/bin/env python3
"""python-flint vs default ground-types benchmark on the actual project workload.

Run with::

    # Default Python ground types (in either venv)
    python bench_flint/bench.py

    # FLINT ground types (in bench_flint/venv with python-flint installed)
    SYMPY_GROUND_TYPES=flint python bench_flint/bench.py

The script does three things:

1. Micro: time a single hard ``cancel()`` on a polynomial expression
   structurally similar to the Schwarzschild L3 brackets that take
   100s+ in the live run.

2. End-to-end (small): run the Schwarzschild composite at L<=2 and time
   the full ``compute_growth(max_level=2)`` call. Repeats N times for
   noise reduction.

3. End-to-end (medium): same at L<=3 with N=3, d=2, low samples. This is
   the most representative single-run benchmark; takes a few minutes
   under stock Python and should be much faster under FLINT.

Output is plain text plus a JSON line at the end suitable for
diffing across runs.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

# Identify backend BEFORE importing sympy heavy bits.
import sympy  # noqa: E402
from sympy.polys.domains import GROUND_TYPES  # noqa: E402

import sympy as sp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nbody"))


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_env() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sympy": sympy.__version__,
        "ground_types": GROUND_TYPES,
        "env_var": os.environ.get("SYMPY_GROUND_TYPES", "<unset>"),
    }
    try:
        import flint  # type: ignore
        env["python_flint"] = flint.__version__
    except ImportError:
        env["python_flint"] = "<not installed>"
    banner("Environment")
    for k, v in env.items():
        print(f"  {k:18s}  {v}")
    return env


# --------------------------------------------------------------------------- #
# Bench 1: hard cancel() on a structurally-realistic polynomial
# --------------------------------------------------------------------------- #

def make_hard_polynomial() -> sp.Expr:
    """Build a multivariate rational polynomial sized like a typical
    Schwarzschild L3 intermediate. 15 variables, lots of cross products,
    nontrivial denominator.
    """
    syms = sp.symbols("x1 y1 x2 y2 x3 y3 px1 py1 px2 py2 px3 py3 u12 u13 u23")
    x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3, u12, u13, u23 = syms

    # Numerator: a deliberately ugly polynomial in 15 variables.
    num = (
        (px1 ** 2 + py1 ** 2) * u12 ** 3 * u13
        + (px2 ** 2 + py2 ** 2) * u12 ** 3 * u23
        + (px3 ** 2 + py3 ** 2) * u13 ** 3 * u23
        + (x1 - x2) ** 2 * (y2 - y3) ** 2 * u12 ** 2 * u23 ** 2
        + (x1 - x3) ** 2 * (y1 - y3) ** 2 * u13 ** 2 * u23 ** 2
        + (px1 * px2 + py1 * py2) * (x1 - x3) * (y2 - y3) * u12 ** 2 * u13 ** 2
        + (px2 * px3 + py2 * py3) * (x2 - x1) * (y3 - y1) * u13 ** 2 * u23 ** 2
        - sp.Rational(7, 4) * u12 ** 4 * u13 ** 2 * u23
        + sp.Rational(11, 8) * u12 ** 2 * u13 ** 4 * u23 ** 2
        - sp.Rational(3, 16) * u12 * u13 * u23 ** 5
    )

    # Denominator: small but nontrivial - exercises GCD path.
    den = u12 ** 2 + u13 ** 2 + u23 ** 2

    return num / den


def bench_micro(n_iter: int = 5) -> dict:
    banner(f"Bench 1: cancel() on a hard rational polynomial  (n_iter={n_iter})")
    expr = make_hard_polynomial()
    print(f"  num/den built: {len(sp.Add.make_args(sp.expand(expr.as_numer_denom()[0])))} terms in numerator")

    times = []
    for i in range(n_iter):
        # Re-create from scratch each iteration to defeat any caches.
        e = make_hard_polynomial()
        t0 = time.perf_counter()
        result = sp.cancel(e)
        elapsed = time.perf_counter() - t0
        nterms = len(sp.Add.make_args(result.as_numer_denom()[0]))
        times.append(elapsed)
        print(f"  iter {i+1}/{n_iter}: {elapsed:8.3f}s   ({nterms} terms in numerator)")

    avg = sum(times) / len(times)
    best = min(times)
    print(f"  -> mean: {avg:.3f}s,  best: {best:.3f}s")
    return {"name": "micro_cancel", "iters": n_iter, "times_s": times,
            "mean_s": avg, "best_s": best}


# --------------------------------------------------------------------------- #
# Bench 2: end-to-end Schwarzschild L2  (small, fast, noise-free)
# --------------------------------------------------------------------------- #

def bench_schwarzschild_l2(n_iter: int = 3) -> dict:
    from exact_growth_nbody import NBodyAlgebra

    banner(f"Bench 2: Schwarzschild composite L<=2 dimseq  (n_iter={n_iter})")
    times = []
    for i in range(n_iter):
        # Use a unique cache dir per iter to avoid loading prior pickle.
        ckpt = REPO_ROOT / "bench_flint" / f"_cache_l2_iter{i}_{GROUND_TYPES}"
        params = [
            (-sp.Integer(1), 1),
            (sp.Rational(1, 2), 2),
            (-sp.Integer(1), 3),
        ]
        alg = NBodyAlgebra(
            n_bodies=3, d_spatial=2, potential="composite",
            potential_params=params, checkpoint_dir=str(ckpt),
        )
        t0 = time.perf_counter()
        dims = alg.compute_growth(max_level=2, n_samples=50, seed=42)
        elapsed = time.perf_counter() - t0
        seq = [int(dims[lv]) for lv in range(3)]
        times.append(elapsed)
        print(f"  iter {i+1}/{n_iter}: {elapsed:8.3f}s   sequence = {seq}")
        # Cleanup
        import shutil
        if ckpt.exists():
            shutil.rmtree(ckpt, ignore_errors=True)

    avg = sum(times) / len(times)
    best = min(times)
    print(f"  -> mean: {avg:.3f}s,  best: {best:.3f}s")
    return {"name": "schwarzschild_l2", "iters": n_iter, "times_s": times,
            "mean_s": avg, "best_s": best}


# --------------------------------------------------------------------------- #
# Bench 3: end-to-end Schwarzschild L3 (slow, single iter)
# --------------------------------------------------------------------------- #

def bench_schwarzschild_l3() -> dict:
    """Time one full Schwarzschild L<=3 run. Single iter (this takes
    minutes-to-hours)."""
    from exact_growth_nbody import NBodyAlgebra

    banner("Bench 3: Schwarzschild composite L<=3 dimseq  (1 iter, can take minutes)")
    ckpt = REPO_ROOT / "bench_flint" / f"_cache_l3_{GROUND_TYPES}"
    params = [
        (-sp.Integer(1), 1),
        (sp.Rational(1, 2), 2),
        (-sp.Integer(1), 3),
    ]
    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2, potential="composite",
        potential_params=params, checkpoint_dir=str(ckpt),
    )
    t0 = time.perf_counter()
    dims = alg.compute_growth(max_level=3, n_samples=100, seed=42)
    elapsed = time.perf_counter() - t0
    seq = [int(dims[lv]) for lv in range(4)]
    print(f"\n  L<=3 elapsed: {elapsed:.1f}s   sequence = {seq}")

    # Don't auto-clean cache - caller may want to inspect it
    return {"name": "schwarzschild_l3", "iters": 1, "times_s": [elapsed],
            "mean_s": elapsed, "best_s": elapsed, "sequence": seq}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-l3", action="store_true",
                    help="Skip the slow L3 benchmark")
    ap.add_argument("--micro-iter", type=int, default=5)
    ap.add_argument("--l2-iter", type=int, default=3)
    ap.add_argument("--out", default=None,
                    help="Append JSON results line to this file")
    args = ap.parse_args()

    env = print_env()
    results = {"env": env, "benches": []}

    results["benches"].append(bench_micro(args.micro_iter))
    results["benches"].append(bench_schwarzschild_l2(args.l2_iter))
    if not args.skip_l3:
        results["benches"].append(bench_schwarzschild_l3())

    banner("SUMMARY")
    print(f"  ground_types: {env['ground_types']}")
    for b in results["benches"]:
        print(f"  {b['name']:30s}  mean={b['mean_s']:8.2f}s   best={b['best_s']:8.2f}s")

    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(results) + "\n")
        print(f"\n  Appended to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
