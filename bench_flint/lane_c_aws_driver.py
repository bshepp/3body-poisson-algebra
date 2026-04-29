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
import signal
import subprocess
import sys
import threading
import time
import urllib.request

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


def periodic_syncer(work_dir: str, run_log_path: str,
                    stop_event: threading.Event, interval_s: int = 60):
    """Background thread: mirror work_dir to S3 every `interval_s` seconds.

    Also uploads the tee'd run log so we have a live tail in S3 instead
    of waiting for the post-driver userdata `aws s3 cp` (which never
    runs if the instance is SIGKILL'd by spot reclaim).
    """
    while not stop_event.wait(interval_s):
        s3_sync(work_dir, "checkpoints")
        if run_log_path and os.path.isfile(run_log_path):
            s3_upload_file(run_log_path, "lane_c_run.log")


# IMDSv2 endpoints for spot interruption notice.
# Returns 200 with JSON ~2 minutes before reclaim; 404 otherwise.
_IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
_IMDS_SPOT_URL = ("http://169.254.169.254/latest/meta-data/"
                  "spot/instance-action")


def _get_imds_token(timeout=2.0):
    try:
        req = urllib.request.Request(
            _IMDS_TOKEN_URL, method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("ascii")
    except Exception:
        return None


def spot_interruption_listener(stop_event: threading.Event,
                               poll_s: int = 5):
    """Watch IMDS for spot reclaim notice; on detection send SIGTERM
    to ourselves so the engine's flush handler runs.

    AWS gives ~2 minutes of warning, plenty for the engine to checkpoint
    the current bracket batch and the driver to do a final S3 sync.
    """
    pid = os.getpid()
    notified = False
    while not stop_event.wait(poll_s):
        token = _get_imds_token()
        if token is None:
            continue  # IMDS unreachable (probably not on EC2); just keep polling
        try:
            req = urllib.request.Request(
                _IMDS_SPOT_URL,
                headers={"X-aws-ec2-metadata-token": token},
            )
            with urllib.request.urlopen(req, timeout=2.0) as r:
                body = r.read().decode("utf-8", errors="replace")
            if not notified:
                print(f"\n[spot] !!! INTERRUPTION NOTICE !!!  {body}",
                      flush=True)
                print("[spot] sending SIGTERM to driver to trigger "
                      "engine flush + S3 sync", flush=True)
                notified = True
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as e:
                    print(f"[spot] SIGTERM failed: {e}", flush=True)
                # Keep polling so we log if the action changes
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue  # normal: no interruption pending
        except Exception:
            continue


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
    sync_interval_s = int(os.environ.get("S3_SYNC_INTERVAL_S", "60"))
    run_log_path = os.environ.get("RUN_LOG", "/opt/3body/lane_c_run.log")

    print(f"=== Lane C driver ===", flush=True)
    print(f"  work_dir       = {work_dir}", flush=True)
    print(f"  max_level      = {args.max_level}", flush=True)
    print(f"  n_samples      = {args.n_samples}", flush=True)
    print(f"  prime          = {args.prime}", flush=True)
    print(f"  walltime_s     = {walltime}", flush=True)
    print(f"  batch_save     = {batch_save}", flush=True)
    print(f"  sync_interval  = {sync_interval_s}s", flush=True)
    print(f"  run_log        = {run_log_path}", flush=True)
    print(f"  S3 dest        = s3://{S3_BUCKET}/{S3_PREFIX}/", flush=True)
    print(f"  started        = {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Background S3 mirror so spot reclaim leaves recoverable state.
    # Also uploads the run log so we get live tailable progress.
    stop_event = threading.Event()
    syncer = threading.Thread(
        target=periodic_syncer,
        args=(work_dir, run_log_path, stop_event, sync_interval_s),
        daemon=True,
    )
    syncer.start()

    # IMDS spot interruption listener — converts the ~2-min reclaim
    # notice into a SIGTERM the engine can flush on cleanly.
    spot_listener = threading.Thread(
        target=spot_interruption_listener,
        args=(stop_event, 5),
        daemon=True,
    )
    spot_listener.start()

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
        # Final synchronous sync (checkpoints + run log)
        s3_sync(work_dir, "checkpoints")
        if run_log_path and os.path.isfile(run_log_path):
            s3_upload_file(run_log_path, "lane_c_run.log")

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
