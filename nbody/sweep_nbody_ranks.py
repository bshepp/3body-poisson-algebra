#!/usr/bin/env python3
"""
Systematic N-body rank sweep
=============================

Runs symbolic_rank_nbody.py for a configurable list of (N, max_level) jobs.
Skips any job whose result JSON already exists. Saves results and timing
to a summary JSON at the end.

Usage
-----
    # Full sweep (N=3-10 L3, N=11-15 L2)
    python sweep_nbody_ranks.py --workers 15

    # Dry run — show what would be computed
    python sweep_nbody_ranks.py --dry-run

    # Custom job list
    python sweep_nbody_ranks.py --jobs 3:3 4:3 5:2 --workers 4

    # Resume after interruption (skips completed results)
    python sweep_nbody_ranks.py --workers 15
"""

import os
import sys
import json
import subprocess
import argparse
from time import time, strftime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "symbolic_rank"
RANK_SCRIPT = SCRIPT_DIR / "symbolic_rank_nbody.py"

DEFAULT_JOBS = (
    # (N, max_level) — N=3-10 through Level 3, N=11-15 through Level 2
    [(n, 3) for n in range(3, 11)] +
    [(n, 2) for n in range(11, 16)]
)

VALIDATION_EXPECTED = [3, 6, 17, 116]


def result_path(n, d=1, potential="1r"):
    return RESULTS_DIR / f"rank_N{n}_d{d}_{potential}.json"


def load_existing_result(n):
    path = result_path(n)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def job_is_complete(n, max_level):
    """Check if an existing result covers the requested max_level."""
    data = load_existing_result(n)
    if data is None:
        return False
    existing_levels = len(data.get("cumulative_rank", []))
    # cumulative_rank has max_level+1 entries (L0 through Lmax_level)
    return existing_levels >= max_level + 1


def run_job(n, max_level, d, workers, checkpoint_base, dry_run=False):
    """Run a single (N, max_level) job via subprocess."""
    checkpoint_dir = Path(checkpoint_base) / f"checkpoints_n{n}d{d}_level{max_level}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", str(RANK_SCRIPT),
        "-N", str(n),
        "-d", str(d),
        "--max-level", str(max_level),
        "--checkpoint-dir", str(checkpoint_dir),
        "--workers", str(workers),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"JOB: N={n} d={d} max_level={max_level} workers={workers}", flush=True)
    print(f"CMD: {' '.join(cmd)}", flush=True)
    print(f"START: {strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
    print(f"{'='*70}", flush=True)

    if dry_run:
        print("  [DRY RUN] Skipping execution", flush=True)
        return {"N": n, "max_level": max_level, "status": "dry_run", "time": 0}

    t0 = time()
    result = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    elapsed = time() - t0

    status = "success" if result.returncode == 0 else f"failed (exit {result.returncode})"
    print(f"\nFINISHED: N={n} L{max_level} — {status} in {elapsed:.1f}s", flush=True)

    return {
        "N": n,
        "d": d,
        "max_level": max_level,
        "status": status,
        "exit_code": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
    }


def parse_job_spec(spec):
    """Parse 'N:level' string into (N, level) tuple."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid job spec '{spec}', expected N:level")
    return int(parts[0]), int(parts[1])


def main():
    ap = argparse.ArgumentParser(description="Systematic N-body rank sweep")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers for bracket computation")
    ap.add_argument("--d", type=int, default=1, help="Spatial dimension")
    ap.add_argument("--jobs", nargs="+", default=None,
                    help="Custom job list as N:level pairs (e.g. 3:3 4:3 5:2)")
    ap.add_argument("--checkpoint-base", default=None,
                    help="Base directory for checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be computed without running")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if result already exists")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip the N=3 validation run")
    ap.add_argument("--summary", default=None,
                    help="Path to write summary JSON")
    args = ap.parse_args()

    if args.jobs:
        jobs = [parse_job_spec(s) for s in args.jobs]
    else:
        jobs = DEFAULT_JOBS

    checkpoint_base = args.checkpoint_base or str(SCRIPT_DIR)

    print(f"{'#'*70}", flush=True)
    print(f"# N-BODY RANK SWEEP", flush=True)
    print(f"# Started: {strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
    print(f"# Workers: {args.workers}", flush=True)
    print(f"# Dimension: d={args.d}", flush=True)
    print(f"# Jobs: {len(jobs)}", flush=True)
    print(f"{'#'*70}", flush=True)

    # --- Validation: always run N=3 L3 first as a sanity check ---
    if not args.no_validate and not args.dry_run:
        print(f"\n{'='*70}", flush=True)
        print(f"VALIDATION: Running N=3 d=1 L3 (expected {VALIDATION_EXPECTED})",
              flush=True)
        print(f"{'='*70}", flush=True)

        val_result = run_job(3, 3, args.d, args.workers, checkpoint_base)
        val_data = load_existing_result(3)

        if val_data is None:
            print("\n  VALIDATION FAILED: No result file produced!", flush=True)
            print("  ABORTING SWEEP.", flush=True)
            sys.exit(1)

        val_ranks = val_data.get("cumulative_rank", [])
        if val_ranks == VALIDATION_EXPECTED:
            print(f"\n  VALIDATION PASSED: {val_ranks} == {VALIDATION_EXPECTED}",
                  flush=True)
        else:
            print(f"\n  VALIDATION FAILED: got {val_ranks}, "
                  f"expected {VALIDATION_EXPECTED}", flush=True)
            print("  ABORTING SWEEP — something is wrong with the compute "
                  "environment.", flush=True)
            sys.exit(1)

    skip_count = 0
    run_count = 0
    results = []

    for n, max_level in jobs:
        if not args.force and job_is_complete(n, max_level):
            existing = load_existing_result(n)
            ranks = existing.get("cumulative_rank", [])
            print(f"\n  SKIP: N={n} L{max_level} — already have {len(ranks)} "
                  f"levels: {ranks}", flush=True)
            skip_count += 1
            results.append({
                "N": n, "max_level": max_level,
                "status": "skipped (existing)",
                "existing_ranks": ranks,
            })
            continue

        run_count += 1
        job_result = run_job(n, max_level, args.d, args.workers,
                            checkpoint_base, args.dry_run)
        results.append(job_result)

        data = load_existing_result(n)
        if data:
            job_result["ranks"] = data.get("cumulative_rank", [])

    print(f"\n{'#'*70}", flush=True)
    print(f"# SWEEP COMPLETE", flush=True)
    print(f"# Finished: {strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)
    print(f"# Skipped: {skip_count}, Ran: {run_count}", flush=True)
    print(f"{'#'*70}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'N':>4} {'Level':>6} {'Status':>20} {'Time':>10} {'Ranks'}", flush=True)
    print(f"{'-'*4:>4} {'-'*6:>6} {'-'*20:>20} {'-'*10:>10} {'-'*30}", flush=True)
    for r in results:
        n = r["N"]
        lvl = r["max_level"]
        status = r.get("status", "?")
        elapsed = r.get("elapsed_seconds", r.get("time", ""))
        ranks = r.get("ranks", r.get("existing_ranks", ""))
        time_str = f"{elapsed}s" if isinstance(elapsed, (int, float)) and elapsed else ""
        print(f"{n:>4} {lvl:>6} {status:>20} {time_str:>10} {ranks}", flush=True)

    summary_path = args.summary or str(RESULTS_DIR / "sweep_summary.json")
    summary = {
        "started": strftime('%Y-%m-%dT%H:%M:%SZ'),
        "workers": args.workers,
        "d": args.d,
        "jobs_skipped": skip_count,
        "jobs_ran": run_count,
        "results": results,
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
