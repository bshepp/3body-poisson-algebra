#!/usr/bin/env python3
"""
E2 — Symbolic-rank validation harness for the cancel->together patch.

For each pinned baseline in results/symbolic_rank/, instantiate
`NBodySymbolicRank` with the *patched* engine, recompute the exact Q-rank,
and compare bit-for-bit against the pinned `cumulative_rank` and
`new_per_level`.

Cases (5):
  N=3 d=1 1/r    -> [3, 6, 17, 116]   (max_level=3, ~7.9s baseline)
  N=3 d=1 1/r^2  -> [3, 6, 17, 116]   (max_level=3, ~54.5s baseline)
  N=3 d=1 1/r^3  -> [3, 6, 17, 116]   (max_level=3, ~56.7s baseline)
  N=3 d=1 1/r^4  -> [3, 6, 17, 116]   (max_level=3, ~57.8s baseline)
  N=3 d=2 log    -> [3, 6, 17]        (max_level=2, ~0.8s baseline)

Output: bench_flint/_symrank_results/<case_id>.json  +  summary printed.
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
from symbolic_rank_nbody import NBodySymbolicRank  # noqa: E402


CASES = [
    dict(case_id="r1_n3_d1",     pin="rank_N3_d1_1r.json",
         n_bodies=3, d_spatial=1, potential="1/r",   max_level=3),
    dict(case_id="r1over2_n3_d1", pin="rank_N3_d1_1r2.json",
         n_bodies=3, d_spatial=1, potential="1/r^2", max_level=3),
    dict(case_id="r1over3_n3_d1", pin="rank_N3_d1_1r3.json",
         n_bodies=3, d_spatial=1, potential="1/r^3", max_level=3),
    dict(case_id="r1over4_n3_d1", pin="rank_N3_d1_1r4.json",
         n_bodies=3, d_spatial=1, potential="1/r^4", max_level=3),
    dict(case_id="log_n3_d2",    pin="rank_N3_d2_log.json",
         n_bodies=3, d_spatial=2, potential="log",   max_level=2),
]

PIN_DIR = REPO_ROOT / "results" / "symbolic_rank"
OUT_DIR = REPO_ROOT / "bench_flint" / "_symrank_results"
SUMMARY = REPO_ROOT / "bench_flint" / "validation_results_symbolic_rank.json"


def run_case(case):
    pin_path = PIN_DIR / case["pin"]
    with open(pin_path, "r", encoding="utf-8") as f:
        pin = json.load(f)

    expected_cum = pin["cumulative_rank"]
    expected_new = pin["new_per_level"]
    pin_t = pin.get("computation_time_seconds")

    print(f"\n{'='*68}")
    print(f"  case: {case['case_id']}  (pin: {case['pin']})")
    print(f"  N={case['n_bodies']} d={case['d_spatial']} "
          f"potential={case['potential']} max_level={case['max_level']}")
    print(f"  expected cumulative_rank = {expected_cum}")
    print(f"  expected new_per_level   = {expected_new}")
    print(f"  pin computation_time_seconds = {pin_t}")
    print(f"{'='*68}")

    t0 = time.time()
    engine = NBodySymbolicRank(
        n_bodies=case["n_bodies"],
        d_spatial=case["d_spatial"],
        potential=case["potential"],
    )

    exprs, names, levels = engine.build_generators(
        case["max_level"], checkpoint_dir=None, n_workers=1)
    poly_list, monom_list, monom_to_idx = engine.extract_monomial_matrix(
        exprs, n_workers=1)
    rank_results = engine.compute_exact_rank(
        poly_list, monom_list, monom_to_idx, levels, checkpoint_dir=None)

    cumulative = [rank_results[lv] for lv in sorted(rank_results)]
    new_per_level = [cumulative[0]]
    for i in range(1, len(cumulative)):
        new_per_level.append(cumulative[i] - cumulative[i - 1])

    elapsed = time.time() - t0

    cum_match = (cumulative == expected_cum)
    new_match = (new_per_level == expected_new)
    overall_match = cum_match and new_match

    rec = {
        "case_id": case["case_id"],
        "pin": case["pin"],
        "n_bodies": case["n_bodies"],
        "d_spatial": case["d_spatial"],
        "potential": case["potential"],
        "max_level": case["max_level"],
        "expected_cumulative_rank": expected_cum,
        "expected_new_per_level": expected_new,
        "actual_cumulative_rank": cumulative,
        "actual_new_per_level": new_per_level,
        "n_generators": len(exprs),
        "n_monomials": len(monom_list),
        "elapsed_s": round(elapsed, 2),
        "pin_elapsed_s": pin_t,
        "speedup_vs_pin": (round(pin_t / elapsed, 2)
                            if pin_t and elapsed > 0 else None),
        "cum_match": bool(cum_match),
        "new_match": bool(new_match),
        "match": bool(overall_match),
        "sympy_version": sp.__version__,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{case['case_id']}.json"
    tmp = out_path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, out_path)

    marker = "MATCH" if overall_match else "MISMATCH"
    print(f"\n  --> {marker}: cumulative={cumulative} (pin={expected_cum})")
    print(f"      elapsed = {elapsed:.2f}s  (pin {pin_t}s, "
          f"speedup x{rec['speedup_vs_pin']})")
    return rec


def main():
    print(f"E2 — symbolic-rank validation (sympy {sp.__version__})")
    print(f"Cases: {len(CASES)}")
    t_global = time.time()

    records = []
    for case in CASES:
        try:
            rec = run_case(case)
        except Exception as e:
            rec = {
                "case_id": case["case_id"],
                "pin": case["pin"],
                "match": False,
                "status": "exception",
                "error": str(e),
            }
            print(f"\n  [!] EXCEPTION on {case['case_id']}: {e}")
        records.append(rec)

    n_total = len(records)
    n_match = sum(1 for r in records if r.get("match"))
    overall = (n_match == n_total)

    summary = {
        "sympy_version": sp.__version__,
        "n_total": n_total,
        "n_match": n_match,
        "overall_match": overall,
        "elapsed_total_s": round(time.time() - t_global, 2),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": records,
    }

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, SUMMARY)

    print("\n" + "=" * 68)
    print(f"  E2 SUMMARY: {n_match}/{n_total} match  "
          f"(elapsed {summary['elapsed_total_s']}s)")
    print(f"  Output: {SUMMARY.relative_to(REPO_ROOT)}")
    for r in records:
        spd = r.get("speedup_vs_pin")
        marker = "OK" if r.get("match") else "FAIL"
        print(f"    {marker}  {r['case_id']}: "
              f"{r.get('actual_cumulative_rank', 'n/a')}  "
              f"({r.get('elapsed_s', 'n/a')}s, "
              f"speedup x{spd if spd is not None else 'n/a'})")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
