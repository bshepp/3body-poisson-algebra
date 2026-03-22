#!/usr/bin/env python3
"""
AWS orchestrator for the Multi-System Universality Survey -- dimension sequences.

Runs NBodyAlgebra.compute_growth for each scenario defined in expansion_configs.py
with robust S3 sync, checkpointing, SIGTERM handling, and completion manifests.

Usage:
    python run_expansion_dimseq.py                          # run all scenarios
    python run_expansion_dimseq.py --category gravitational # one category
    python run_expansion_dimseq.py --scenario sun_earth_moon # one scenario
    python run_expansion_dimseq.py --tier 1                 # tier 1 only (no engine extensions)
"""

import os
import sys
import json
import signal
import argparse
import subprocess
from time import time, strftime

sys.setrecursionlimit(100000)

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

S3_BUCKET = os.environ.get("S3_BUCKET", "")
RESULTS_PREFIX = "results/expansion_dimseq"

_shutdown_requested = False

TIER1_POTENTIALS = {"1/r", "1/r^2", "1/r^3", "composite"}
TIER2_POTENTIALS = {"log", "yukawa"}


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  [SIGTERM] Spot reclamation -- finishing current scenario...",
          flush=True)


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


def s3_pull(s3_key, local_path):
    if not S3_BUCKET:
        return
    src = f"s3://{S3_BUCKET}/{s3_key}"
    try:
        subprocess.run(
            ["aws", "s3", "cp", src, local_path],
            capture_output=True, timeout=60,
        )
    except Exception:
        pass


MANIFEST_PATH = "expansion_dimseq_completion.json"


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    s3_cp(MANIFEST_PATH, f"{RESULTS_PREFIX}/{MANIFEST_PATH}")


def mark_complete(scenario_key, result):
    manifest = load_manifest()
    manifest[scenario_key] = {
        "status": "complete",
        "result": result,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_manifest(manifest)


def is_complete(scenario_key):
    manifest = load_manifest()
    entry = manifest.get(scenario_key, {})
    return entry.get("status") == "complete"


def sync_all_checkpoints():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for d in os.listdir(script_dir):
        full = os.path.join(script_dir, d)
        if os.path.isdir(full) and d.startswith("checkpoints"):
            s3_sync(full, f"nbody_checkpoints/{d}")


def run_scenario(scenario_key, cfg, max_level, n_samples, seed):
    """Run a single scenario and return its dimension sequence."""
    from exact_growth_nbody import NBodyAlgebra

    print("\n" + "#" * 70)
    print(f"# SCENARIO: {cfg['label']}")
    print(f"# Key: {scenario_key}")
    print(f"# Category: {cfg['category']}")
    print(f"# Potential: {cfg['potential']}")
    if cfg.get("charges"):
        print(f"# Charges: {cfg['charges']}")
    if cfg.get("masses"):
        masses_str = ", ".join(f"m{k}={v}" for k, v in sorted(cfg["masses"].items()))
        print(f"# Masses: {masses_str}")
    print(f"# Description: {cfg['description']}")
    print("#" * 70 + "\n")

    kwargs = {
        "n_bodies": 3,
        "d_spatial": 2,
        "potential": cfg["potential"],
        "masses": cfg.get("masses"),
        "charges": cfg.get("charges"),
        "potential_params": cfg.get("potential_params"),
        "external_potential": cfg.get("external_potential"),
    }

    alg = NBodyAlgebra(**kwargs)

    dims = alg.compute_growth(
        max_level=max_level,
        n_samples=n_samples,
        seed=seed,
        resume=True,
    )
    return [dims[lv] for lv in range(max_level + 1)]


def main():
    ap = argparse.ArgumentParser(
        description="Multi-System Universality Survey -- dimension sequences")
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", type=str, default=None,
                    help="Run a single scenario by key")
    ap.add_argument("--category", type=str, default=None,
                    help="Run all scenarios in a category")
    ap.add_argument("--tier", type=int, default=None, choices=[1, 2],
                    help="Tier 1 = existing potentials, Tier 2 = engine extensions")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    from expansion_configs import SCENARIOS, EXPECTED_SEQUENCE, CATEGORIES

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {list(SCENARIOS.keys())}")
            sys.exit(1)
        scenario_keys = [args.scenario]
    elif args.category:
        scenario_keys = [k for k, v in SCENARIOS.items()
                         if v["category"] == args.category]
        if not scenario_keys:
            print(f"No scenarios in category: {args.category}")
            sys.exit(1)
    elif args.tier == 1:
        scenario_keys = [k for k, v in SCENARIOS.items()
                         if v["potential"] in TIER1_POTENTIALS]
    elif args.tier == 2:
        scenario_keys = [k for k, v in SCENARIOS.items()
                         if v["potential"] in TIER2_POTENTIALS
                         or v.get("external_potential") is not None]
    else:
        scenario_keys = list(SCENARIOS.keys())

    print("=" * 70)
    print("MULTI-SYSTEM UNIVERSALITY SURVEY -- DIMENSION SEQUENCES")
    print("=" * 70)
    print(f"  S3 bucket:   {S3_BUCKET or '(none -- local only)'}")
    print(f"  Max level:   {args.max_level}")
    print(f"  Samples:     {args.samples}")
    print(f"  Seed:        {args.seed}")
    print(f"  Scenarios:   {len(scenario_keys)}")
    print(f"  Started:     {strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for i, key in enumerate(scenario_keys):
        cfg = SCENARIOS[key]
        print(f"  {i+1:>3}. [{cfg['category']:>14}] {cfg['label']}")
    print()

    s3_pull(f"{RESULTS_PREFIX}/{MANIFEST_PATH}", MANIFEST_PATH)

    t_total = time()
    results = {}
    completed = 0
    skipped = 0

    for i, key in enumerate(scenario_keys):
        if _shutdown_requested:
            print(f"\n  [SHUTDOWN] Stopping after {completed} scenarios.",
                  flush=True)
            break

        cfg = SCENARIOS[key]

        if is_complete(key):
            manifest = load_manifest()
            results[key] = manifest[key]["result"]
            skipped += 1
            print(f"\n  [{i+1}/{len(scenario_keys)}] {cfg['label']} "
                  f"-- already complete, skipping.")
            continue

        print(f"\n  [{i+1}/{len(scenario_keys)}] Running: {cfg['label']}")

        try:
            seq = run_scenario(key, cfg, args.max_level, args.samples, args.seed)
            results[key] = seq
            sync_all_checkpoints()
            mark_complete(key, seq)
            completed += 1
        except Exception as e:
            print(f"\n  ERROR in {key}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[key] = {"error": str(e)}
            mark_complete(key, {"error": str(e)})
            completed += 1

    total_elapsed = time() - t_total

    # --- Summary ---
    print("\n" + "=" * 70)
    print("UNIVERSALITY SURVEY -- FINAL SUMMARY")
    print("=" * 70)
    print(f"  Completed: {completed},  Skipped (already done): {skipped}")
    print(f"  Total time: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.2f} hours)\n")

    all_match = True
    for cat_key, cat_label in CATEGORIES:
        cat_scenarios = [k for k in scenario_keys
                         if SCENARIOS[k]["category"] == cat_key
                         and k in results]
        if not cat_scenarios:
            continue

        print(f"\n  {cat_label}:")
        for key in cat_scenarios:
            cfg = SCENARIOS[key]
            seq = results[key]
            if isinstance(seq, dict) and "error" in seq:
                print(f"    {cfg['label']:.<40} ERROR: {seq['error']}")
                all_match = False
                continue

            expected = [EXPECTED_SEQUENCE.get(lv, "?")
                        for lv in range(len(seq))]
            match = all(seq[lv] == EXPECTED_SEQUENCE[lv]
                        for lv in range(len(seq))
                        if lv in EXPECTED_SEQUENCE)
            if not match:
                all_match = False
            status = "MATCH" if match else "DIFFERS"
            print(f"    {cfg['label']:.<40} {seq}  [{status}]")

    verdict = "UNIVERSALITY_HOLDS" if all_match else "UNIVERSALITY_BROKEN"
    print(f"\n  {'='*50}")
    print(f"  VERDICT: {verdict}")
    print(f"  {'='*50}")

    sync_all_checkpoints()

    summary = {
        "status": "interrupted" if _shutdown_requested else "complete",
        "verdict": verdict,
        "scenarios_completed": completed,
        "scenarios_skipped": skipped,
        "results": {k: v for k, v in results.items()},
        "total_seconds": total_elapsed,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open("expansion_dimseq_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    s3_cp("expansion_dimseq_summary.json",
          f"{RESULTS_PREFIX}/expansion_dimseq_summary.json")


if __name__ == "__main__":
    main()
