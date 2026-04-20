#!/usr/bin/env python3
"""Single-strategy compute_growth runner.

Monkey-patches ``NBodyAlgebra.simplify_generator`` to use the requested
strategy (cancel | together | identity), runs ``compute_growth`` for the
requested case, and writes a JSON result file then exits.

This is the worker the orchestrator launches under a watchdog. Keeping it
in its own subprocess means hard timeouts and RAM caps actually work
(can't escape its own process boundary).

Usage::

    python bench_flint/run_one_strategy.py STRATEGY CASE OUT_PATH

  STRATEGY  : cancel | together | identity
  CASE      : schwarz_l2 | 1r_l3 | schwarz_l3
  OUT_PATH  : path to write JSON result

JSON shape::

    {
      "strategy": "together",
      "case":     "schwarz_l3",
      "elapsed_s": 1234.5,
      "sequence":  [3, 6, 17, 116],
      "expected":  [3, 6, 17, 116],
      "match":     true,
      "started_at":  "2026-04-19T22:30:00",
      "completed_at":"2026-04-19T22:50:34"
    }
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nbody"))

import sympy as sp  # noqa: E402
from sympy import Integer, Rational, cancel, together  # noqa: E402

from exact_growth_nbody import NBodyAlgebra  # noqa: E402


# --------------------------------------------------------------------------- #
# Strategies (callable[[Expr], Expr])
# --------------------------------------------------------------------------- #

STRATEGIES = {
    "cancel":   cancel,
    "together": together,
    "identity": (lambda e: e),
}


# --------------------------------------------------------------------------- #
# Cases: each returns (NBodyAlgebra, max_level, n_samples, expected_sequence)
# --------------------------------------------------------------------------- #

def case_schwarz_l2(ckpt_dir: str) -> tuple:
    """Schwarzschild composite V_eff at L<=2. Cheap; sanity gate."""
    params = [
        (-Integer(1), 1),
        (Rational(1, 2), 2),
        (-Integer(1), 3),
    ]
    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2, potential="composite",
        potential_params=params, checkpoint_dir=ckpt_dir,
    )
    return alg, 2, 50, [3, 6, 17]


def case_1r_l3(ckpt_dir: str) -> tuple:
    """N=3 1/r at L<=3. Canonical project reference; expected [3,6,17,116].

    n_samples=50 keeps the SVD-evaluation phase fast (~minutes) so the
    measurement isolates the symbolic / simplify pipeline rather than
    the lambdify+eval phase. For 1/r the SVD gap is huge (>1e10) so 50
    samples is plenty for unambiguous rank determination.
    """
    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2, potential="1/r",
        checkpoint_dir=ckpt_dir,
    )
    return alg, 3, 50, [3, 6, 17, 116]


def case_schwarz_l3(ckpt_dir: str) -> tuple:
    """Schwarzschild composite at L<=3. Production-relevant heavy case.

    n_samples=100 keeps the SVD-eval phase from dominating the measurement;
    the question we want to answer is whether the simplify pipeline is
    faster. Schwarzschild composite has plenty of SVD gap at 100 samples.
    """
    params = [
        (-Integer(1), 1),
        (Rational(1, 2), 2),
        (-Integer(1), 3),
    ]
    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2, potential="composite",
        potential_params=params, checkpoint_dir=ckpt_dir,
    )
    return alg, 3, 100, [3, 6, 17, 116]


CASES = {
    "schwarz_l2": case_schwarz_l2,
    "1r_l3":      case_1r_l3,
    "schwarz_l3": case_schwarz_l3,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    strategy_name = sys.argv[1]
    case_name = sys.argv[2]
    out_path = Path(sys.argv[3])

    if strategy_name not in STRATEGIES:
        print(f"unknown strategy {strategy_name!r}; valid: {list(STRATEGIES)}",
              file=sys.stderr)
        sys.exit(2)
    if case_name not in CASES:
        print(f"unknown case {case_name!r}; valid: {list(CASES)}",
              file=sys.stderr)
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Patch the engine. (Do this BEFORE building NBodyAlgebra; harmless if
    # later but conceptually cleaner here.)
    simplify_fn = STRATEGIES[strategy_name]
    NBodyAlgebra.simplify_generator = (
        lambda self, expr, _fn=simplify_fn: _fn(expr))

    # Fresh per-(case, strategy) checkpoint dir so we always re-derive.
    ckpt_dir = REPO_ROOT / "bench_flint" / f"_run_{case_name}_{strategy_name}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    alg, max_level, n_samples, expected = CASES[case_name](str(ckpt_dir))

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[{started_at}] strategy={strategy_name} case={case_name} "
          f"max_level={max_level} n_samples={n_samples} "
          f"expected={expected}", flush=True)

    result: dict = {
        "strategy": strategy_name,
        "case": case_name,
        "max_level": max_level,
        "n_samples": n_samples,
        "expected": expected,
        "started_at": started_at,
    }

    t0 = time.perf_counter()
    try:
        dims = alg.compute_growth(
            max_level=max_level, n_samples=n_samples, seed=42,
        )
    except KeyboardInterrupt:
        result["elapsed_s"] = time.perf_counter() - t0
        result["completed_at"] = dt.datetime.now().isoformat(timespec="seconds")
        result["status"] = "interrupted"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return 130
    except Exception as exc:
        import traceback
        result["elapsed_s"] = time.perf_counter() - t0
        result["completed_at"] = dt.datetime.now().isoformat(timespec="seconds")
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    seq = [int(dims[lv]) for lv in range(max_level + 1)]
    match = (seq == expected)

    result.update({
        "elapsed_s": elapsed,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "done",
        "sequence": seq,
        "match": match,
    })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    completed_at = result["completed_at"]
    flag = "MATCH" if match else "MISMATCH"
    print(f"[{completed_at}] strategy={strategy_name} case={case_name} "
          f"elapsed={elapsed:.1f}s sequence={seq} {flag}", flush=True)

    return 0 if match else 3


if __name__ == "__main__":
    sys.exit(main())
