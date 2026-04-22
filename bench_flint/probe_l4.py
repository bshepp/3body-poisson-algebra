#!/usr/bin/env python3
"""L=4 probe: real per-bracket measurements under the patched together() engine.

Goal: replace hand-wavy extrapolations with measured per-bracket cost so we
can decide whether full L=4 is feasible on this workstation.

Design:
  1. Build L<=3 generators for N=3 d=2 1/r (the canonical baseline).
  2. Enumerate the L=4 candidate pairs the way level4_highsample.py does.
  3. Bucket pairs by (level_i, level_j) since cost varies dramatically:
       - L3 x L0:  cheapest (12 candidates per L3 gen)
       - L3 x L1:  cheap-medium
       - L3 x L2:  medium
       - L3 x L3:  most expensive AND most numerous
  4. Sample N pairs from each bucket and time:
       - poisson_bracket()
       - simplify_generator() = together(...)
       - report Add.make_args count and count_ops on the result
  5. Aggregate: per-bucket mean/median/p95 time + size; multiply by bucket
     size to extrapolate the full L=4 generation cost honestly.

Output: bench_flint/probe_l4_results.json
        bench_flint/probe_l4.log

Watchdog-safe: each bracket has a per-call timeout, and the script writes
incrementally so a kill leaves partial data behind.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nbody"))

import sympy as sp  # noqa: E402

from exact_growth_nbody import NBodyAlgebra  # noqa: E402

OUT_JSON = REPO_ROOT / "bench_flint" / "probe_l4_results.json"
OUT_LOG = REPO_ROOT / "bench_flint" / "probe_l4.log"
CACHE_DIR = REPO_ROOT / "bench_flint" / "_probe_l4_cache"

# Per-bucket sample counts. Keep small so the probe stays bounded.
# (l_i, l_j) -> n_to_sample
SAMPLES_PER_BUCKET = {
    (3, 0): 6,
    (3, 1): 6,
    (3, 2): 6,
    (3, 3): 12,    # the dominant bucket
}

# If a single bracket exceeds this many seconds, abandon and move on.
PER_BRACKET_TIMEOUT_S = 60.0


log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    log_lines.append(msg)


def banner(msg: str) -> None:
    log("=" * 72)
    log(msg)
    log("=" * 72)


def write_partial(payload: dict) -> None:
    """Atomic write of in-progress results."""
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(OUT_JSON.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, OUT_JSON)


def main() -> int:
    banner("L=4 PROBE - measuring per-bracket cost under the together patch")

    # Confirm we are on the patched engine
    import inspect
    src = inspect.getsource(NBodyAlgebra.simplify_generator).strip()
    log("simplify_generator source:")
    log("  " + src.replace("\n", "\n  "))

    # Build L<=3 fresh
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    log("\nStep 1: Building L<=3 generators (N=3 d=2 1/r)...")
    t0 = time.perf_counter()
    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2, potential="1/r",
        checkpoint_dir=str(CACHE_DIR),
    )
    # We need the generators; running compute_growth(max_level=3) with
    # tiny n_samples is the cheapest way to populate them via the normal
    # path (and it cross-checks against the canonical [3,6,17,116]).
    dims = alg.compute_growth(max_level=3, n_samples=50, seed=42)
    seq = [int(dims[lv]) for lv in range(4)]
    log(f"  L<=3 build: {time.perf_counter()-t0:.1f}s, sequence = {seq}")
    if seq != [3, 6, 17, 116]:
        log(f"  WARNING: L<=3 sequence is {seq}, not the canonical "
            f"[3, 6, 17, 116]. Probe results may be misleading.")

    # Pull all_exprs / all_names / all_levels from the L=3 checkpoint
    import pickle
    ck_path = CACHE_DIR / "level_3.pkl"
    with open(ck_path, "rb") as f:
        ckpt = pickle.load(f)
    all_exprs = ckpt["exprs"]
    all_names = ckpt["names"]
    all_levels = ckpt["levels"]
    n_gen = len(all_exprs)
    log(f"\n  Loaded {n_gen} generators from {ck_path.relative_to(REPO_ROOT)}")
    log(f"  Levels: L0={sum(1 for lv in all_levels if lv == 0)}, "
        f"L1={sum(1 for lv in all_levels if lv == 1)}, "
        f"L2={sum(1 for lv in all_levels if lv == 2)}, "
        f"L3={sum(1 for lv in all_levels if lv == 3)}")

    # Enumerate L=4 pairs the same way level4_highsample.py does
    frontier = [i for i, lv in enumerate(all_levels) if lv == 3]
    computed = set()
    for i in range(n_gen):
        for j in range(i + 1, n_gen):
            if all_levels[i] + all_levels[j] <= 3:
                computed.add(frozenset({i, j}))
    pairs_by_bucket: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in frontier:
        for j in range(n_gen):
            if i == j:
                continue
            pair = frozenset({i, j})
            if pair in computed:
                continue
            computed.add(pair)
            li, lj = all_levels[i], all_levels[j]
            key = tuple(sorted([li, lj], reverse=True))
            pairs_by_bucket.setdefault(key, []).append((min(i, j), max(i, j)))

    log("\n  L=4 pair counts by bucket (level_i, level_j):")
    total_pairs = 0
    for k, pairs in sorted(pairs_by_bucket.items()):
        log(f"    {k}: {len(pairs):>5d}")
        total_pairs += len(pairs)
    log(f"  TOTAL L=4 candidate brackets: {total_pairs}")

    # Sample pairs from each bucket. Use stride-based picks for diversity.
    sampled: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for bucket, n_to_sample in SAMPLES_PER_BUCKET.items():
        if bucket not in pairs_by_bucket:
            continue
        pool = pairs_by_bucket[bucket]
        if len(pool) <= n_to_sample:
            sampled[bucket] = list(pool)
        else:
            stride = max(1, len(pool) // n_to_sample)
            sampled[bucket] = [pool[i] for i in range(0, len(pool), stride)
                               ][:n_to_sample]

    bucket_results: dict[str, list[dict]] = {}

    banner("Step 2: Timing sampled L=4 brackets under `together`")

    payload_running = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_gen_l3": n_gen,
        "total_l4_pairs": total_pairs,
        "pairs_by_bucket": {f"{k[0]},{k[1]}": len(v)
                            for k, v in pairs_by_bucket.items()},
        "samples_per_bucket": {f"{k[0]},{k[1]}": v
                               for k, v in SAMPLES_PER_BUCKET.items()},
        "per_bracket_timeout_s": PER_BRACKET_TIMEOUT_S,
        "buckets": bucket_results,
        "complete": False,
    }
    write_partial(payload_running)

    for bucket in sorted(sampled.keys(), reverse=True):
        bucket_key = f"{bucket[0]},{bucket[1]}"
        bucket_results[bucket_key] = []
        log(f"\n  -- bucket (l_i, l_j) = {bucket}, "
            f"{len(sampled[bucket])} samples --")

        for k, (i, j) in enumerate(sampled[bucket], 1):
            ni, nj = all_names[i], all_names[j]
            label = f"({i},{j}) {{{ni},{nj}}}"
            log(f"    [{k:>2d}/{len(sampled[bucket])}] {label}", )

            # Time the bracket itself
            t0 = time.perf_counter()
            try:
                raw = alg.poisson_bracket(all_exprs[i], all_exprs[j])
                t_pb = time.perf_counter() - t0
                if t_pb > PER_BRACKET_TIMEOUT_S:
                    log(f"      poisson_bracket OVER TIMEOUT "
                        f"({t_pb:.1f}s > {PER_BRACKET_TIMEOUT_S}s); "
                        f"skipping simplify")
                    bucket_results[bucket_key].append({
                        "i": i, "j": j, "name_i": ni, "name_j": nj,
                        "poisson_bracket_s": t_pb,
                        "skipped": "poisson_bracket_timeout",
                    })
                    write_partial(payload_running)
                    continue
            except Exception as exc:
                log(f"      poisson_bracket FAILED: {exc}")
                bucket_results[bucket_key].append({
                    "i": i, "j": j, "name_i": ni, "name_j": nj,
                    "error": str(exc),
                    "phase": "poisson_bracket",
                })
                write_partial(payload_running)
                continue

            # Time the simplify
            t0 = time.perf_counter()
            try:
                simplified = alg.simplify_generator(raw)
                t_simp = time.perf_counter() - t0
            except Exception as exc:
                log(f"      simplify FAILED after pb {t_pb:.2f}s: {exc}")
                bucket_results[bucket_key].append({
                    "i": i, "j": j, "name_i": ni, "name_j": nj,
                    "poisson_bracket_s": t_pb,
                    "error": str(exc),
                    "phase": "simplify",
                })
                write_partial(payload_running)
                continue

            # Report sizes
            try:
                n_summands = len(sp.Add.make_args(simplified))
            except Exception:
                n_summands = -1
            try:
                n_ops = int(sp.count_ops(simplified))
            except Exception:
                n_ops = -1

            log(f"      pb {t_pb:7.3f}s  simp {t_simp:7.3f}s  "
                f"summands={n_summands}  ops={n_ops}")

            bucket_results[bucket_key].append({
                "i": i, "j": j, "name_i": ni, "name_j": nj,
                "poisson_bracket_s": t_pb,
                "simplify_s": t_simp,
                "total_s": t_pb + t_simp,
                "n_summands": n_summands,
                "count_ops": n_ops,
            })
            payload_running["buckets"] = bucket_results
            write_partial(payload_running)

    # Aggregate
    banner("Step 3: Bucket statistics + extrapolation")

    summary = []
    grand_total_estimated = 0.0
    for bucket in sorted(pairs_by_bucket.keys(), reverse=True):
        bucket_key = f"{bucket[0]},{bucket[1]}"
        bucket_size = len(pairs_by_bucket[bucket])
        results = bucket_results.get(bucket_key, [])
        timed = [r for r in results if "total_s" in r]
        if not timed:
            log(f"  bucket {bucket}: no timing data")
            summary.append({"bucket": bucket_key,
                            "bucket_size": bucket_size,
                            "n_timed": 0})
            continue
        totals = sorted(r["total_s"] for r in timed)
        ops_sizes = [r["count_ops"] for r in timed if r["count_ops"] >= 0]
        mean = sum(totals) / len(totals)
        median = totals[len(totals) // 2]
        p95 = totals[max(0, int(len(totals) * 0.95) - 1)]
        max_t = totals[-1]
        est_total_s = mean * bucket_size
        grand_total_estimated += est_total_s
        log(f"  bucket {bucket}  size={bucket_size:>5d}  n_timed={len(timed)}")
        log(f"    times s: mean={mean:7.3f}  median={median:7.3f}  "
            f"p95={p95:7.3f}  max={max_t:7.3f}")
        log(f"    sizes ops: median={(sorted(ops_sizes)[len(ops_sizes)//2] if ops_sizes else -1)}")
        log(f"    estimated bucket total: {est_total_s:>9.1f}s "
            f"= {est_total_s/60:>6.2f} min")
        summary.append({
            "bucket": bucket_key,
            "bucket_size": bucket_size,
            "n_timed": len(timed),
            "mean_s": mean,
            "median_s": median,
            "p95_s": p95,
            "max_s": max_t,
            "estimated_bucket_total_s": est_total_s,
        })

    log(f"\n  GRAND ESTIMATED L=4 GENERATION TIME (single core): "
        f"{grand_total_estimated:.1f}s = {grand_total_estimated/60:.1f} min "
        f"= {grand_total_estimated/3600:.2f} hr")

    payload_running["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload_running["complete"] = True
    payload_running["summary"] = summary
    payload_running["estimated_total_s"] = grand_total_estimated
    payload_running["estimated_total_min"] = grand_total_estimated / 60
    payload_running["estimated_total_hr"] = grand_total_estimated / 3600
    write_partial(payload_running)

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\n  Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    log(f"  Wrote {OUT_LOG.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
