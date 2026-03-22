#!/usr/bin/env python3
"""
AWS orchestrator for the Multi-System Universality Survey -- stability atlases.

Runs targeted_adaptive_scan.py for each atlas-enabled scenario defined in
expansion_configs.py, with robust S3 sync, SIGTERM handling, and completion
tracking.

Usage:
    python run_expansion_atlas.py                            # all atlas scenarios
    python run_expansion_atlas.py --scenario sun_earth_moon  # one scenario
    python run_expansion_atlas.py --category atomic          # one category
"""

import os
import sys
import json
import signal
import subprocess
from time import time, strftime

os.environ["PYTHONUNBUFFERED"] = "1"

S3_BUCKET = os.environ.get("S3_BUCKET", "")
RESULTS_PREFIX = "results/expansion_atlas"

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  [SIGTERM] Spot reclamation -- finishing current scan...",
          flush=True)


signal.signal(signal.SIGTERM, _sigterm_handler)


def s3_sync(local_dir, s3_prefix):
    if not S3_BUCKET:
        return
    dest = f"s3://{S3_BUCKET}/{s3_prefix}"
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, dest],
            capture_output=True, timeout=300,
        )
    except Exception as e:
        print(f"  [S3 sync warning: {e}]", flush=True)


def s3_cp(local_path, s3_key):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(
            ["aws", "s3", "cp", local_path, f"s3://{S3_BUCKET}/{s3_key}"],
            capture_output=True, timeout=60,
        )
    except Exception as e:
        print(f"  [S3 cp warning: {e}]", flush=True)


def s3_pull(s3_key, local_path):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(
            ["aws", "s3", "cp", f"s3://{S3_BUCKET}/{s3_key}", local_path],
            capture_output=True, timeout=60,
        )
    except Exception:
        pass


MANIFEST_PATH = "expansion_atlas_completion.json"


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    s3_cp(MANIFEST_PATH, f"{RESULTS_PREFIX}/{MANIFEST_PATH}")


def mark_complete(key, exit_code):
    manifest = load_manifest()
    manifest[key] = {
        "status": "complete" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_manifest(manifest)


def is_complete(key):
    manifest = load_manifest()
    entry = manifest.get(key, {})
    return entry.get("status") == "complete"


def run_targeted_scan(scenario_key, cfg):
    """Run targeted_adaptive_scan.py for one scenario (ref + charged if applicable)."""
    potential = cfg["potential"]
    charges = cfg.get("charges")
    masses = cfg.get("masses")

    scan_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "targeted_adaptive_scan.py"
    )

    phases = []

    ref_key = f"{scenario_key}_ref"
    ref_cmd = [
        sys.executable, "-u", scan_script,
        "--potential", potential,
    ]
    if potential == "yukawa" and cfg.get("potential_params"):
        for pname, pval in cfg["potential_params"]:
            if pname == "mu":
                ref_cmd.extend(["--yukawa-mu", str(float(pval))])
    phases.append((ref_key, "reference", ref_cmd))

    if charges:
        chg_key = f"{scenario_key}_chg"
        chg_cmd = list(ref_cmd) + [
            "--charges"] + [str(charges[b]) for b in sorted(charges.keys())]
        phases.append((chg_key, "charged", chg_cmd))

    results = {}

    for phase_key, phase_label, cmd in phases:
        if is_complete(phase_key):
            print(f"    [{phase_label}] Already complete, skipping.")
            results[phase_key] = 0
            continue

        if _shutdown_requested:
            print(f"    [SHUTDOWN] Skipping {phase_label}")
            return results

        print(f"\n    Starting {phase_label} scan: {' '.join(cmd)}")
        t0 = time()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

            while proc.poll() is None:
                if _shutdown_requested:
                    print(f"    [SIGTERM] Forwarding to scan PID {proc.pid}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass

            exit_code = proc.returncode or 0
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            exit_code = 1

        elapsed = time() - t0
        print(f"    {phase_label} completed: exit={exit_code}, "
              f"time={elapsed/60:.1f}min")

        s3_sync("atlas_targeted/", "atlas_targeted/")
        mark_complete(phase_key, exit_code)
        results[phase_key] = exit_code

    # Run analysis
    if not _shutdown_requested:
        for phase_key, phase_label, cmd in phases:
            analyze_cmd = [
                sys.executable, "-u", scan_script,
                "--analyze", "--potential", potential,
            ]
            if phase_label == "charged" and charges:
                analyze_cmd.extend(
                    ["--charges"] + [str(charges[b])
                                     for b in sorted(charges.keys())])
            try:
                subprocess.run(analyze_cmd, timeout=300)
            except Exception as e:
                print(f"    Analysis warning: {e}")

    return results


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Multi-System Universality Survey -- atlas scans")
    ap.add_argument("--scenario", type=str, default=None)
    ap.add_argument("--category", type=str, default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nbody"))
    from nbody.expansion_configs import (
        SCENARIOS, CATEGORIES, get_atlas_scenarios
    )

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            sys.exit(1)
        scenario_keys = [args.scenario]
    elif args.category:
        scenario_keys = [k for k, v in SCENARIOS.items()
                         if v["category"] == args.category
                         and v.get("run_atlas", False)]
    else:
        scenario_keys = get_atlas_scenarios()

    print("=" * 70)
    print("MULTI-SYSTEM UNIVERSALITY SURVEY -- ATLAS SCANS")
    print("=" * 70)
    print(f"  S3 bucket:   {S3_BUCKET or '(none -- local only)'}")
    print(f"  Scenarios:   {len(scenario_keys)}")
    print(f"  Started:     {strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for i, key in enumerate(scenario_keys):
        cfg = SCENARIOS[key]
        print(f"  {i+1:>3}. [{cfg['category']:>14}] {cfg['label']} "
              f"({cfg['potential']})")
    print()

    s3_pull(f"{RESULTS_PREFIX}/{MANIFEST_PATH}", MANIFEST_PATH)
    s3_sync_src = f"s3://{S3_BUCKET}/atlas_targeted/" if S3_BUCKET else ""
    if s3_sync_src:
        try:
            subprocess.run(
                ["aws", "s3", "sync", s3_sync_src, "atlas_targeted/"],
                capture_output=True, timeout=300,
            )
        except Exception:
            pass

    t_total = time()
    completed = 0

    for i, key in enumerate(scenario_keys):
        if _shutdown_requested:
            break

        cfg = SCENARIOS[key]
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(scenario_keys)}] {cfg['label']} "
              f"({cfg['potential']})")
        print(f"  {cfg['description']}")
        print(f"{'='*70}")

        results = run_targeted_scan(key, cfg)
        completed += 1

        s3_sync("atlas_targeted/", "atlas_targeted/")

    total_elapsed = time() - t_total

    print(f"\n{'='*70}")
    if _shutdown_requested:
        print("INTERRUPTED BY SIGTERM -- data saved to S3")
    else:
        print("ALL ATLAS SCANS COMPLETE")
    print(f"  Completed: {completed}/{len(scenario_keys)}")
    print(f"  Total time: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.2f} hours)")
    print("=" * 70)

    for attempt in range(3):
        try:
            s3_sync("atlas_targeted/", "atlas_targeted/")
            break
        except Exception:
            import time as _time
            _time.sleep(10)

    summary = {
        "status": "interrupted" if _shutdown_requested else "complete",
        "scenarios_completed": completed,
        "total_scenarios": len(scenario_keys),
        "total_seconds": total_elapsed,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open("expansion_atlas_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    s3_cp("expansion_atlas_summary.json",
          f"{RESULTS_PREFIX}/expansion_atlas_summary.json")


if __name__ == "__main__":
    main()
