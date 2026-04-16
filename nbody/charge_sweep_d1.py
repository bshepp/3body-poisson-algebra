#!/usr/bin/env python3
"""
Charge Sweep at d=1 — Exact Symbolic Rank
==========================================

Runs the (+1, +q, -1) charge sweep using exact symbolic rank over QQ
at d=1, which is orders of magnitude faster than numerical SVD at d=2.

Spatial dimension independence has been proven: d=1 and d=2 give the
same dimension sequences for all tested configurations.

Checkpoints after each q value via manifest file.

Usage:
    python charge_sweep_d1.py
    python charge_sweep_d1.py --q-max 30
"""

import sys
import os
import json
import argparse
from time import time, strftime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "results", "charge_sensitivity",
                             "charge_sweep_qqn_d1.json")


def load_manifest():
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, MANIFEST_PATH)


def run_config(q, max_level=3):
    """Run exact symbolic rank for charges (+1, +q, -1) at d=1."""
    from symbolic_rank_nbody import NBodySymbolicRank

    charges = {1: 1, 2: q, 3: -1}
    label = f"(+1, +{q}, -1) unit masses"

    print(f"\n  Running {label}...", flush=True)
    t0 = time()

    engine = NBodySymbolicRank(
        n_bodies=3, d_spatial=1,
        potential="1/r",
        charges=charges,
    )
    exprs, names, levels = engine.build_generators(max_level=max_level)
    poly_list, monom_list, monom_to_idx = engine.extract_monomial_matrix(exprs)
    rank_results = engine.compute_exact_rank(
        poly_list, monom_list, monom_to_idx, levels)

    elapsed = time() - t0

    dims = [rank_results[lv] for lv in sorted(rank_results.keys())]
    ref = [3, 6, 17, 116][:len(dims)]
    matches = dims == ref

    print(f"    Result: {dims}")
    print(f"    Matches universal: {matches}")
    print(f"    Time: {elapsed:.1f}s")

    return {
        "label": label,
        "charges": {str(k): v for k, v in charges.items()},
        "masses": {"1": 1, "2": 1, "3": 1},
        "dims": dims,
        "dim_l3": dims[3] if len(dims) > 3 else None,
        "elapsed_seconds": elapsed,
        "matches_116": dims[3] == 116 if len(dims) > 3 else None,
        "method": "exact_symbolic_QQ",
        "d_spatial": 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Charge sweep (+1,+q,-1) at d=1")
    parser.add_argument("--q-min", type=int, default=1)
    parser.add_argument("--q-max", type=int, default=20)
    parser.add_argument("--max-level", type=int, default=3)
    args = parser.parse_args()

    q_values = list(range(args.q_min, args.q_max + 1))

    manifest = load_manifest()

    print("=" * 60)
    print("CHARGE SWEEP (+1, +q, -1) — EXACT SYMBOLIC (d=1)")
    print("=" * 60)
    print(f"  q range: {args.q_min} to {args.q_max}")
    print(f"  Max level: {args.max_level}")
    print(f"  Already completed: {len(manifest)}")

    t_total = time()
    for q in q_values:
        key = f"sweep_qqn_q{q}_exact"
        if key in manifest and manifest[key].get("status") == "complete":
            dims = manifest[key]["result"]["dims"]
            print(f"\n  q={q}: already complete, dims={dims}")
            continue

        result = run_config(q, max_level=args.max_level)
        manifest[key] = {
            "status": "complete",
            "result": result,
            "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        save_manifest(manifest)

    total_elapsed = time() - t_total

    print(f"\n{'=' * 60}")
    print("CHARGE SWEEP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    print(f"\n  {'q':>4s}  {'dims':>20s}  {'match':>8s}  {'time':>8s}")
    print(f"  {'-'*4}  {'-'*20}  {'-'*8}  {'-'*8}")
    for q in q_values:
        key = f"sweep_qqn_q{q}_exact"
        if key in manifest:
            r = manifest[key]["result"]
            dims = r["dims"]
            match = "YES" if r.get("matches_116") else "NO"
            t = r.get("elapsed_seconds", 0)
            print(f"  {q:4d}  {str(dims):>20s}  {match:>8s}  {t:7.1f}s")

    print(f"\n  Results saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
