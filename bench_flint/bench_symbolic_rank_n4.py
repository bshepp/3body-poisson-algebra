#!/usr/bin/env python3
"""
E3 — N=4 d=1 1/r cancel-vs-together speedup bench.

Two passes:
  Pass 1: head-to-head at L=2 (both cancel and together, same engine
          freshly instantiated; per-level wall time + per-bracket counts)
  Pass 2: together-only at L=3 (compared against the pinned cancel time
          recorded in results/symbolic_rank/rank_N4_d1_1r.json:
          computation_time_seconds = 3137.6s)

Generation phase only (no extract_monomial_matrix, no rank). The patch only
touches generation; the rank phase is identical for both engines.

Output: bench_flint/symrank_n4_timing.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "nbody"))

import sympy as sp  # noqa: E402
from sympy import cancel, expand, together  # noqa: E402

import symbolic_rank_nbody as srn  # noqa: E402
from symbolic_rank_nbody import NBodySymbolicRank  # noqa: E402


def _make_simplify(mode):
    """Return a _simplify implementation that uses the given mode."""
    if mode == "cancel":
        def _simplify(self, expr):
            if self.uses_u:
                return cancel(expr)
            return expand(expr)
    elif mode == "together":
        def _simplify(self, expr):
            if self.uses_u:
                return together(expr)
            return expand(expr)
    else:
        raise ValueError(mode)
    return _simplify


def run_one(mode, n_bodies, d_spatial, potential, max_level):
    print(f"\n{'='*68}")
    print(f"  N={n_bodies} d={d_spatial} potential={potential} "
          f"max_level={max_level}  mode={mode}")
    print(f"{'='*68}")
    # Monkeypatch _simplify on the class, then build engine afresh.
    NBodySymbolicRank._simplify = _make_simplify(mode)
    engine = NBodySymbolicRank(
        n_bodies=n_bodies, d_spatial=d_spatial, potential=potential)

    t0 = time.time()
    exprs, names, levels = engine.build_generators(
        max_level, checkpoint_dir=None, n_workers=1)
    elapsed = time.time() - t0

    # Per-level counts
    by_level = {}
    for lv in levels:
        by_level[lv] = by_level.get(lv, 0) + 1

    # Average term count of generated brackets at the top level
    from sympy import Add
    top_lv = max(levels)
    top_exprs = [e for e, lv in zip(exprs, levels) if lv == top_lv]
    top_nterms = [len(Add.make_args(e)) for e in top_exprs]
    avg_nterms = (sum(top_nterms) / len(top_nterms)) if top_nterms else 0.0

    return {
        "mode": mode,
        "n_bodies": n_bodies,
        "d_spatial": d_spatial,
        "potential": potential,
        "max_level": max_level,
        "elapsed_s": round(elapsed, 2),
        "n_generators": len(exprs),
        "by_level": by_level,
        "top_level_avg_nterms": round(avg_nterms, 1),
        "top_level_max_nterms": max(top_nterms) if top_nterms else 0,
    }


def main():
    print(f"E3 — N=4 d=1 1/r cancel-vs-together (sympy {sp.__version__})")
    t_global = time.time()

    # Pass 1: head-to-head at L=2 (small enough that cancel won't hang)
    pass1 = []
    pass1.append(run_one("cancel",   n_bodies=4, d_spatial=1,
                         potential="1/r", max_level=2))
    pass1.append(run_one("together", n_bodies=4, d_spatial=1,
                         potential="1/r", max_level=2))

    if pass1[0]["elapsed_s"] > 0:
        l2_speedup = pass1[0]["elapsed_s"] / max(pass1[1]["elapsed_s"], 1e-6)
    else:
        l2_speedup = None

    # Pass 2: together-only at L=3, compared against pin
    pin_path = REPO_ROOT / "results" / "symbolic_rank" / "rank_N4_d1_1r.json"
    with open(pin_path, "r", encoding="utf-8") as f:
        pin = json.load(f)
    pin_t_l3 = pin["computation_time_seconds"]

    pass2 = run_one("together", n_bodies=4, d_spatial=1,
                    potential="1/r", max_level=3)
    l3_speedup = pin_t_l3 / max(pass2["elapsed_s"], 1e-6)

    summary = {
        "sympy_version": sp.__version__,
        "pass1_l2_head_to_head": pass1,
        "pass1_l2_speedup_cancel_over_together": round(l2_speedup, 2)
            if l2_speedup is not None else None,
        "pass2_l3_together_only": pass2,
        "pass2_l3_pin_cancel_elapsed_s": pin_t_l3,
        "pass2_l3_speedup_pin_over_together": round(l3_speedup, 2),
        "elapsed_total_s": round(time.time() - t_global, 2),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out = REPO_ROOT / "bench_flint" / "symrank_n4_timing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, out)

    print("\n" + "=" * 68)
    print("  E3 SUMMARY")
    print("=" * 68)
    print(f"  L=2 cancel:    {pass1[0]['elapsed_s']}s")
    print(f"  L=2 together:  {pass1[1]['elapsed_s']}s")
    print(f"  L=2 speedup:   x{summary['pass1_l2_speedup_cancel_over_together']}")
    print(f"  L=3 pin (cancel, 3137.6s reference)")
    print(f"  L=3 together:  {pass2['elapsed_s']}s")
    print(f"  L=3 speedup:   x{summary['pass2_l3_speedup_pin_over_together']}")
    print(f"  Output: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
