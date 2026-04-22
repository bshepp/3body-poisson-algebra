#!/usr/bin/env python3
"""30-minute timed L=4 generation probe.

Generates L=4 brackets in the same order level4_highsample.py would, but
walking the full pair list (not bucketed sampling). Stops cleanly when a
wall-clock budget runs out or the process is killed. Incremental save
after every BATCH_SIZE brackets so a kill loses at most that batch.

Output:
  bench_flint/probe_l4_30min_results.json  (incremental, atomic)
  bench_flint/probe_l4_30min.log

What we learn:
  - Real (not extrapolated) rate of L=4 bracket completion under together()
  - Whether any pathological brackets blow up vs the 24-bracket sample's prediction
  - A truer extrapolation for the full 11,523-bracket job
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nbody"))

import sympy as sp  # noqa: E402

from exact_growth_nbody import NBodyAlgebra  # noqa: E402

CACHE_DIR = REPO_ROOT / "bench_flint" / "_probe_l4_cache"   # reuse from probe_l4
OUT_JSON = REPO_ROOT / "bench_flint" / "probe_l4_30min_results.json"
OUT_LOG = REPO_ROOT / "bench_flint" / "probe_l4_30min.log"

WALL_BUDGET_S = 28 * 60       # leave 2 min margin under the 30-min cap
PER_BRACKET_TIMEOUT_S = 60.0  # any single bracket > 60s gets recorded and we move on
BATCH_SAVE = 25               # checkpoint every N brackets

log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    log_lines.append(msg)


def banner(msg: str) -> None:
    log("=" * 72)
    log(msg)
    log("=" * 72)


def write_partial(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(OUT_JSON.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, OUT_JSON)


def main() -> int:
    banner("L=4 30-MINUTE TIMED RUN under the patched together() engine")
    log(f"  wall budget: {WALL_BUDGET_S}s ({WALL_BUDGET_S/60:.1f} min)")
    log(f"  per-bracket timeout: {PER_BRACKET_TIMEOUT_S}s")
    log(f"  incremental save every {BATCH_SAVE} brackets")

    # Need the L<=3 cache from the prior probe_l4 run. Build it if missing.
    ck_path = CACHE_DIR / "level_3.pkl"
    if not ck_path.exists():
        log("\nL<=3 cache missing; building (N=3 d=2 1/r, ~30s)...")
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        alg = NBodyAlgebra(
            n_bodies=3, d_spatial=2, potential="1/r",
            checkpoint_dir=str(CACHE_DIR),
        )
        t0 = time.perf_counter()
        dims = alg.compute_growth(max_level=3, n_samples=50, seed=42)
        log(f"  built in {time.perf_counter()-t0:.1f}s, "
            f"sequence = {[int(dims[lv]) for lv in range(4)]}")
    else:
        # Need an alg instance for poisson_bracket and simplify_generator
        log("\nReusing existing L<=3 cache")
        alg = NBodyAlgebra(
            n_bodies=3, d_spatial=2, potential="1/r",
            checkpoint_dir=str(CACHE_DIR),
        )

    with open(ck_path, "rb") as f:
        ckpt = pickle.load(f)
    all_exprs = ckpt["exprs"]
    all_names = ckpt["names"]
    all_levels = ckpt["levels"]
    n_gen = len(all_exprs)
    log(f"  loaded {n_gen} L<=3 generators")

    # Enumerate L=4 pairs (same order as level4_highsample.py)
    frontier = [i for i, lv in enumerate(all_levels) if lv == 3]
    computed = set()
    for i in range(n_gen):
        for j in range(i + 1, n_gen):
            if all_levels[i] + all_levels[j] <= 3:
                computed.add(frozenset({i, j}))
    pairs: list[tuple[int, int]] = []
    for i in frontier:
        for j in range(n_gen):
            if i == j:
                continue
            pair = frozenset({i, j})
            if pair in computed:
                continue
            computed.add(pair)
            pairs.append((min(i, j), max(i, j)))
    n_pairs = len(pairs)
    log(f"  L=4 candidate brackets: {n_pairs}")

    banner(f"Step 2: bracketing in order, budget {WALL_BUDGET_S}s")

    results: list[dict] = []
    payload = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_gen_l3": n_gen,
        "total_l4_pairs": n_pairs,
        "wall_budget_s": WALL_BUDGET_S,
        "per_bracket_timeout_s": PER_BRACKET_TIMEOUT_S,
        "results": results,
        "complete": False,
    }
    write_partial(payload)

    t_start = time.perf_counter()
    last_log = t_start
    completed = 0
    timed_out = 0

    for k, (i, j) in enumerate(pairs):
        elapsed = time.perf_counter() - t_start
        if elapsed >= WALL_BUDGET_S:
            log(f"\n  >>> wall-clock budget reached at bracket {k}/{n_pairs}")
            break

        ni, nj = all_names[i], all_names[j]
        li, lj = all_levels[i], all_levels[j]

        t0 = time.perf_counter()
        try:
            raw = alg.poisson_bracket(all_exprs[i], all_exprs[j])
            t_pb = time.perf_counter() - t0
        except Exception as exc:
            log(f"    [{k:>5d}] {{{ni},{nj}}} FAILED in poisson_bracket: {exc}")
            results.append({
                "k": k, "i": i, "j": j,
                "level_i": li, "level_j": lj,
                "error": str(exc), "phase": "poisson_bracket",
            })
            payload["results"] = results
            write_partial(payload)
            continue

        if t_pb > PER_BRACKET_TIMEOUT_S:
            log(f"    [{k:>5d}] {{{ni},{nj}}} pb {t_pb:.1f}s OVER TIMEOUT; skipping simp")
            timed_out += 1
            results.append({
                "k": k, "i": i, "j": j,
                "level_i": li, "level_j": lj,
                "poisson_bracket_s": t_pb,
                "skipped": "poisson_bracket_timeout",
            })
            payload["results"] = results
            write_partial(payload)
            continue

        t0s = time.perf_counter()
        try:
            simp = alg.simplify_generator(raw)
            t_simp = time.perf_counter() - t0s
        except Exception as exc:
            log(f"    [{k:>5d}] {{{ni},{nj}}} pb {t_pb:.2f}s simp FAILED: {exc}")
            results.append({
                "k": k, "i": i, "j": j,
                "level_i": li, "level_j": lj,
                "poisson_bracket_s": t_pb,
                "error": str(exc), "phase": "simplify",
            })
            payload["results"] = results
            write_partial(payload)
            continue

        try:
            n_sum = len(sp.Add.make_args(simp))
        except Exception:
            n_sum = -1
        try:
            n_ops = int(sp.count_ops(simp))
        except Exception:
            n_ops = -1

        results.append({
            "k": k, "i": i, "j": j,
            "level_i": li, "level_j": lj,
            "poisson_bracket_s": t_pb,
            "simplify_s": t_simp,
            "total_s": t_pb + t_simp,
            "n_summands": n_sum,
            "count_ops": n_ops,
        })
        completed += 1

        # Progress logging every 30s
        now = time.perf_counter()
        if now - last_log >= 30.0 or completed % BATCH_SAVE == 0:
            elapsed_total = now - t_start
            rate = completed / elapsed_total if elapsed_total > 0 else 0.0
            remaining_s = (n_pairs - (k + 1)) / rate if rate > 0 else float('inf')
            log(f"    [{k:>5d}/{n_pairs}] elapsed={elapsed_total:6.0f}s  "
                f"completed={completed}  rate={rate:.2f}/s  "
                f"ETA(full)={remaining_s/3600:.2f}h")
            last_log = now

        if completed % BATCH_SAVE == 0:
            payload["results"] = results
            payload["completed"] = completed
            payload["k_reached"] = k + 1
            write_partial(payload)

    # Final stats
    elapsed_total = time.perf_counter() - t_start
    log(f"\n  Run finished. elapsed={elapsed_total:.1f}s")
    log(f"  brackets attempted: {len(results)}, completed: {completed}, "
        f"per-bracket timeouts: {timed_out}")

    timed = [r for r in results if "total_s" in r]
    if timed:
        totals = sorted(r["total_s"] for r in timed)
        mean = sum(totals) / len(totals)
        median = totals[len(totals) // 2]
        p95 = totals[max(0, int(len(totals) * 0.95) - 1)]
        max_t = totals[-1]
        log(f"\n  per-bracket time (s): mean={mean:.3f} median={median:.3f} "
            f"p95={p95:.3f} max={max_t:.3f}")

        # Per-bucket
        from collections import defaultdict
        buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
        for r in timed:
            key = tuple(sorted([r["level_i"], r["level_j"]], reverse=True))
            buckets[key].append(r["total_s"])
        log("  per-bucket means:")
        for key in sorted(buckets, reverse=True):
            ts = buckets[key]
            log(f"    {key}  n={len(ts):>4d}  mean={sum(ts)/len(ts):.3f}s")

        # Honest extrapolation
        # Use bucket means, weighted by bucket sizes
        bucket_sizes = {(3, 3): 9453, (3, 2): 1656, (3, 1): 414}
        est_total = 0.0
        for key, total in bucket_sizes.items():
            samples = buckets.get(key, [])
            if samples:
                bucket_mean = sum(samples) / len(samples)
                est_total += bucket_mean * total
                log(f"    bucket {key}: size {total} x {bucket_mean:.3f}s = {bucket_mean*total/3600:.2f}h")

        log(f"\n  HONEST FULL L=4 ESTIMATE (single core): "
            f"{est_total:.0f}s = {est_total/3600:.2f}h")
        log(f"  At 16-core perfect parallelism: {est_total/16/3600:.2f}h")

        payload["honest_estimate_total_s"] = est_total
        payload["honest_estimate_total_h"] = est_total / 3600

    payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["complete"] = True
    payload["completed_count"] = completed
    payload["k_reached"] = len(results)
    payload["actual_elapsed_s"] = elapsed_total
    write_partial(payload)

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n  Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    log(f"  Wrote {OUT_LOG.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
