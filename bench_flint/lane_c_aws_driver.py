#!/usr/bin/env python3
"""
Lane C — N=3, d=2, V=1/r, level 4 mod-p rank computation on AWS.

Thin driver around `NBodyAlgebra.compute_growth_modp`. The engine handles
checkpointing, resume, SIGTERM/SIGINT, and walltime budgets. This script
adds periodic S3 sync of the checkpoints/results directory so that an
interrupted spot instance leaves a recoverable trail.

Usage (on instance):
    python3 -u lane_c_aws_driver.py --max-level 4 --n-samples 120

Environment:
    S3_BUCKET       — bucket for periodic sync (e.g. 3body-compute-290318)
    S3_PREFIX       — prefix under bucket (default: lane_c)
    WORK_DIR        — local checkpoint dir (default: /opt/3body/lane_c)
    MAX_WALLTIME_S  — hard walltime budget in seconds (default: 18000 = 5h)
    BATCH_SAVE      — checkpoint every N brackets (default: 25)
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

# Make the engine importable when running from /opt/3body
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from nbody.exact_growth_nbody import NBodyAlgebra  # noqa: E402

S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "lane_c").strip("/")


def s3_sync(local_dir: str, prefix_suffix: str = ""):
    """Mirror local_dir to s3://$S3_BUCKET/$S3_PREFIX/<prefix_suffix>/."""
    if not S3_BUCKET or not os.path.isdir(local_dir):
        return
    dest = f"s3://{S3_BUCKET}/{S3_PREFIX}/{prefix_suffix}".rstrip("/")
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, dest, "--no-progress"],
            check=True, capture_output=True, text=True, timeout=600,
        )
        print(f"[s3] sync {local_dir} -> {dest}", flush=True)
    except subprocess.SubprocessError as e:
        print(f"[s3] sync warning: {e}", flush=True)


def s3_upload_file(path: str, key_suffix: str):
    if not S3_BUCKET or not os.path.isfile(path):
        return
    dest = f"s3://{S3_BUCKET}/{S3_PREFIX}/{key_suffix}"
    try:
        subprocess.run(
            ["aws", "s3", "cp", path, dest, "--no-progress"],
            check=True, capture_output=True, text=True, timeout=300,
        )
        print(f"[s3] cp {path} -> {dest}", flush=True)
    except subprocess.SubprocessError as e:
        print(f"[s3] cp warning: {e}", flush=True)


def periodic_syncer(work_dir: str, stop_event: threading.Event,
                    interval_s: int = 300):
    """Background thread: mirror work_dir to S3 every `interval_s` seconds."""
    while not stop_event.wait(interval_s):
        s3_sync(work_dir, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20251108)
    ap.add_argument("--prime", type=int, default=2147483647)
    args = ap.parse_args()

    work_dir = os.environ.get("WORK_DIR", "/opt/3body/lane_c")
    os.makedirs(work_dir, exist_ok=True)
    walltime = int(os.environ.get("MAX_WALLTIME_S", "18000"))
    batch_save = int(os.environ.get("BATCH_SAVE", "25"))

    print(f"=== Lane C driver ===", flush=True)
    print(f"  work_dir       = {work_dir}", flush=True)
    print(f"  max_level      = {args.max_level}", flush=True)
    print(f"  n_samples      = {args.n_samples}", flush=True)
    print(f"  prime          = {args.prime}", flush=True)
    print(f"  walltime_s     = {walltime}", flush=True)
    print(f"  batch_save     = {batch_save}", flush=True)
    print(f"  S3 dest        = s3://{S3_BUCKET}/{S3_PREFIX}/", flush=True)
    print(f"  started        = {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Background S3 mirror so spot reclaim leaves recoverable state
    stop_event = threading.Event()
    syncer = threading.Thread(
        target=periodic_syncer, args=(work_dir, stop_event, 300), daemon=True
    )
    syncer.start()

    algebra = NBodyAlgebra(
        n_bodies=3, d_spatial=2, potential="1/r", checkpoint_dir=work_dir,
    )

    t0 = time.time()
    result = None
    is_complete = False
    try:
        result = algebra.compute_growth_modp(
            max_level=args.max_level,
            n_samples=args.n_samples,
            seed=args.seed,
            prime=args.prime,
            batch_save=batch_save,
            max_walltime_s=walltime,
        )
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        elapsed = time.time() - t0
        print(f"[done] elapsed={elapsed:.1f}s", flush=True)
        stop_event.set()
        # Final synchronous sync
        s3_sync(work_dir, "checkpoints")

        summary = {
            "lane": "C",
            "N": 3, "d": 2, "potential": "1/r",
            "max_level": args.max_level,
            "n_samples": args.n_samples,
            "prime": args.prime,
            "seed": args.seed,
            "elapsed_s": elapsed,
            "result": result,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out_path = os.path.join(work_dir, "lane_c_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        s3_upload_file(out_path, "lane_c_summary.json")

        # Status flag for the launcher to poll
        is_complete = bool(result and result.get("rank") is not None
                           and not result.get("partial", True))
        status = "COMPLETE" if is_complete else "PARTIAL"
        flag_path = os.path.join(work_dir, f"status_{status}.txt")
        with open(flag_path, "w") as f:
            f.write(f"{status}\n{summary}\n")
        s3_upload_file(flag_path, f"status_{status}.txt")

    if is_complete:
        print(f"[SUCCESS] L={args.max_level}  rank = {result['rank']}", flush=True)
    else:
        print("[INCOMPLETE] checkpoint persisted; rerun to resume.", flush=True)


if __name__ == "__main__":
    main()
