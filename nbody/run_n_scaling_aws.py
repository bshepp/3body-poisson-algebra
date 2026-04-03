#!/usr/bin/env python3
"""
AWS-aware N-body scaling probe.

Wraps NBodyAlgebra to probe N=5..N_MAX at d=1 with 1/r potential,
with S3 sync, heartbeat, SIGTERM handling, and incremental saves.

Usage:
    python run_n_scaling_aws.py                       # default N=5..12
    python run_n_scaling_aws.py --n-start 9 --n-max 12
    python run_n_scaling_aws.py --n-start 5 --n-max 9 --resume
"""

import os
import sys
import json
import signal
import argparse
import subprocess
from math import comb
from time import time, strftime
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra

# =====================================================================
# Configuration
# =====================================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "")
RESULTS_PREFIX = "results/nbody_scaling"

MAX_LEVEL = 2
N_SAMPLES = 2000
SEED = 42
D_SPATIAL = 1
POTENTIAL = "1/r"

# Per-N timeout: 4 hours (generous for large N)
TIMEOUT_PER_N = 4 * 3600

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n  [SIGTERM] Spot reclamation at {strftime('%H:%M:%S')} "
          "-- finishing current N...", flush=True)


signal.signal(signal.SIGTERM, _sigterm_handler)


# =====================================================================
# S3 helpers
# =====================================================================

def s3_sync(local_dir, s3_prefix):
    if not S3_BUCKET:
        return
    dest = f"s3://{S3_BUCKET}/{s3_prefix}"
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, dest],
            capture_output=True, timeout=180,
        )
    except Exception as e:
        print(f"  [S3 sync warning: {e}]", flush=True)


def s3_cp(local_path, s3_key):
    if not S3_BUCKET:
        return
    dest = f"s3://{S3_BUCKET}/{s3_key}"
    try:
        subprocess.run(
            ["aws", "s3", "cp", local_path, dest],
            capture_output=True, timeout=60,
        )
    except Exception as e:
        print(f"  [S3 cp warning: {e}]", flush=True)


def upload_heartbeat(n_bodies, status, extra=None):
    if not S3_BUCKET:
        return
    hb = {
        "job": "nbody_scaling",
        "n_bodies": n_bodies,
        "status": status,
        "time": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        hb.update(extra)
    hb_path = "/tmp/nbody_scaling_heartbeat.json"
    with open(hb_path, "w") as f:
        json.dump(hb, f, indent=2)
    s3_cp(hb_path, f"{RESULTS_PREFIX}/heartbeat.json")


# =====================================================================
# Results management
# =====================================================================

RESULTS_FILE = "n_body_scaling_results_aws.json"


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {
        "experiment": "N-body scaling probe (d=1, 1/r, level 0-2)",
        "known": [
            {"N": 3, "sequence": [3, 6, 17, 116]},
            {"N": 4, "sequence": [6, 14, 62]},
        ],
        "results": [],
    }


def save_results(data):
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    s3_cp(RESULTS_FILE, f"{RESULTS_PREFIX}/{RESULTS_FILE}")


def is_n_complete(data, n_bodies):
    """Check if N already has results."""
    for r in data["results"]:
        if r["N"] == n_bodies and len(r.get("sequence", [])) == MAX_LEVEL + 1:
            return True
    return False


def sync_checkpoints():
    """Sync all checkpoint directories to S3."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for d in os.listdir(script_dir):
        full = os.path.join(script_dir, d)
        if os.path.isdir(full) and d.startswith("checkpoints_N"):
            s3_sync(full, f"{RESULTS_PREFIX}/checkpoints/{d}")


# =====================================================================
# Main
# =====================================================================

def count_candidates(n_bodies):
    n_pairs = comb(n_bodies, 2)
    l1_new = comb(n_pairs, 2)
    total_l1 = n_pairs + l1_new
    l2_candidates = l1_new * total_l1
    return {
        "N": n_bodies,
        "pairs": n_pairs,
        "phase_dim": 2 * n_bodies * D_SPATIAL,
        "level_0": n_pairs,
        "level_1_new": l1_new,
        "total_through_1": total_l1,
        "level_2_candidates": l2_candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-start", type=int, default=5)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--resume", action="store_true",
                        help="Skip N values already complete in results file")
    parser.add_argument("--max-level", type=int, default=MAX_LEVEL)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    print("=" * 70)
    print("N-BODY SCALING PROBE (AWS)  d=1, V=1/r")
    print("=" * 70)
    print(f"  N range:    {args.n_start} .. {args.n_max}")
    print(f"  Max level:  {args.max_level}")
    print(f"  Samples:    {args.n_samples}")
    print(f"  Resume:     {args.resume}")
    print(f"  S3 bucket:  {S3_BUCKET or '(none -- local mode)'}")
    print(f"  Timeout:    {TIMEOUT_PER_N}s per N")
    print(f"  Started:    {strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)

    # Scaling table
    print(f"\n{'N':>3} | {'Pairs':>5} | {'Phase':>5} | {'L0':>4} | "
          f"{'L1 new':>7} | {'Thru L1':>7} | {'L2 cands':>10}")
    print("-" * 65)
    for n in range(3, args.n_max + 1):
        c = count_candidates(n)
        print(f"{c['N']:>3} | {c['pairs']:>5} | {c['phase_dim']:>5} | "
              f"{c['level_0']:>4} | {c['level_1_new']:>7} | "
              f"{c['total_through_1']:>7} | {c['level_2_candidates']:>10}")

    data = load_results()

    for n_bodies in range(args.n_start, args.n_max + 1):
        if _shutdown_requested:
            print(f"\n  [SHUTDOWN] Stopping before N={n_bodies}", flush=True)
            break

        if args.resume and is_n_complete(data, n_bodies):
            existing = next(r for r in data["results"] if r["N"] == n_bodies)
            print(f"\n  N={n_bodies}: already complete -> {existing['sequence']}"
                  f"  (skipping)", flush=True)
            continue

        c = count_candidates(n_bodies)
        print(f"\n{'#' * 70}")
        print(f"  N={n_bodies}: {c['pairs']} pairs, "
              f"{c['phase_dim']}D phase space, "
              f"~{c['level_2_candidates']} L2 candidates")
        print(f"{'#' * 70}", flush=True)

        upload_heartbeat(n_bodies, "running", {
            "pairs": c["pairs"],
            "l2_candidates": c["level_2_candidates"],
        })

        t_start = time()
        alg = NBodyAlgebra(
            n_bodies=n_bodies, d_spatial=D_SPATIAL, potential=POTENTIAL,
        )

        try:
            dims = alg.compute_growth(
                max_level=args.max_level,
                n_samples=args.n_samples,
                seed=SEED,
                resume=True,  # always resume from checkpoint on AWS
            )
        except MemoryError:
            elapsed = time() - t_start
            print(f"\n  MEMORY ERROR at N={n_bodies} after {elapsed:.1f}s",
                  flush=True)
            upload_heartbeat(n_bodies, "memory_error",
                             {"elapsed_s": round(elapsed, 1)})
            break
        except Exception as e:
            elapsed = time() - t_start
            print(f"\n  ERROR at N={n_bodies}: {e} after {elapsed:.1f}s",
                  flush=True)
            upload_heartbeat(n_bodies, "error",
                             {"error": str(e), "elapsed_s": round(elapsed, 1)})
            break

        elapsed = time() - t_start
        seq = [dims[lv] for lv in range(args.max_level + 1)]

        result = {
            "N": n_bodies,
            "d": D_SPATIAL,
            "potential": POTENTIAL,
            "sequence": seq,
            "elapsed_s": round(elapsed, 1),
            "pairs": c["pairs"],
            "n_samples": args.n_samples,
            "timestamp": datetime.now().isoformat(),
        }

        # Remove any prior partial result for this N
        data["results"] = [r for r in data["results"] if r["N"] != n_bodies]
        data["results"].append(result)
        data["results"].sort(key=lambda r: r["N"])

        print(f"\n  N={n_bodies} RESULT: {seq}  ({elapsed:.1f}s)", flush=True)

        # Save and sync after every N
        save_results(data)
        sync_checkpoints()
        upload_heartbeat(n_bodies, "complete", {
            "sequence": seq,
            "elapsed_s": round(elapsed, 1),
        })

        # Check timeout for next N estimate
        if elapsed > TIMEOUT_PER_N:
            print(f"\n  Stopping: N={n_bodies} took {elapsed:.0f}s "
                  f"(> {TIMEOUT_PER_N}s limit)", flush=True)
            break

    # Final summary
    print("\n" + "=" * 70)
    print("SCALING SUMMARY  (d=1, V=1/r)")
    print("=" * 70)
    print(f"  {'N':>3} | {'d(0)':>6} | {'d(1)':>6} | {'d(2)':>6} | {'Time':>10}")
    print(f"  {'---':>3}-+-{'------':>6}-+-{'------':>6}-+-{'------':>6}-+-{'----------':>10}")
    for r in data.get("known", []):
        s = r["sequence"]
        cols = [str(s[i]) if i < len(s) else "?" for i in range(3)]
        print(f"  {r['N']:>3} | {cols[0]:>6} | {cols[1]:>6} | {cols[2]:>6} | "
              f"{'known':>10}")
    for r in data["results"]:
        s = r["sequence"]
        cols = [str(s[i]) if i < len(s) else "?" for i in range(3)]
        print(f"  {r['N']:>3} | {cols[0]:>6} | {cols[1]:>6} | {cols[2]:>6} | "
              f"{r['elapsed_s']:>8.1f}s")

    save_results(data)
    sync_checkpoints()
    upload_heartbeat(0, "finished")

    print(f"\n  Completed: {strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
