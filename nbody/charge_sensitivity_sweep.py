#!/usr/bin/env python3
"""
Charge Sensitivity Sweep
=========================

Determines whether the Li+ ([3,6,17,111]) and H2+ ([3,6,17,115]) deviations
from the universal [3,6,17,116] are real physics or SVD conditioning artifacts.

Three phases:
  1. Re-validation: Re-run Li+ and H2+ at multiple sample counts (500–5000)
     with SymPy 1.13.3 to test conditioning sensitivity.
  2. Charge sweep (+q, -1, -1): Vary nuclear charge q = 1..20 with equal
     unit masses to isolate charge effects from mass effects.
  3. Charge sweep (+1, +q, -1): H2+-like family with two positive charges.

Saves per-config: dimension sequence, SVD gap at rank boundary, and full
singular value spectrum at level 3.

Usage:
    python charge_sensitivity_sweep.py                    # all phases
    python charge_sensitivity_sweep.py --phase revalidate # phase 1 only
    python charge_sensitivity_sweep.py --phase sweep_qnn  # phase 2 only
    python charge_sensitivity_sweep.py --phase sweep_qqn  # phase 3 only
"""

import os
import sys
import json
import signal
import argparse
import subprocess
import numpy as np
from time import time, strftime

sys.setrecursionlimit(100000)

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

S3_BUCKET = os.environ.get("S3_BUCKET", "")
RESULTS_PREFIX = "results/charge_sensitivity"

_shutdown_requested = False

REVALIDATION_SAMPLE_COUNTS = [500, 1000, 2000, 5000]
DEFAULT_SAMPLES = 2000
DEFAULT_SEED = 42
MAX_LEVEL = 3

SWEEP_QNN = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
SWEEP_QQN = [1, 2, 3, 5, 10]


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  [SIGTERM] Finishing current config...", flush=True)


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


def s3_sync(local_dir, s3_prefix):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, f"s3://{S3_BUCKET}/{s3_prefix}"],
            capture_output=True, timeout=180,
        )
    except Exception as e:
        print(f"  [S3 sync warning: {e}]", flush=True)


MANIFEST_PATH = "charge_sensitivity_completion.json"


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    s3_cp(MANIFEST_PATH, f"{RESULTS_PREFIX}/{MANIFEST_PATH}")


def mark_complete(key, result):
    manifest = load_manifest()
    manifest[key] = {
        "status": "complete",
        "result": result,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_manifest(manifest)


def is_complete(key):
    manifest = load_manifest()
    return manifest.get(key, {}).get("status") == "complete"


def sync_checkpoints():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for d in os.listdir(script_dir):
        full = os.path.join(script_dir, d)
        if os.path.isdir(full) and d.startswith("checkpoints"):
            s3_sync(full, f"nbody_checkpoints/{d}")


def run_single_config(label, charges, masses, n_samples, seed):
    """Run NBodyAlgebra.compute_growth and return detailed results."""
    from sympy import Integer
    from exact_growth_nbody import NBodyAlgebra

    sym_masses = {}
    for k, v in masses.items():
        sym_masses[k] = Integer(v) if isinstance(v, int) else v

    alg = NBodyAlgebra(
        n_bodies=3,
        d_spatial=2,
        potential="1/r",
        masses=sym_masses,
        charges=charges,
    )

    t0 = time()
    level_dims = alg.compute_growth(
        max_level=MAX_LEVEL,
        n_samples=n_samples,
        seed=seed,
        resume=True,
    )
    elapsed = time() - t0

    dims = [level_dims[lv] for lv in range(MAX_LEVEL + 1)]

    sv_dir = os.path.join(alg.checkpoint_dir, "sv_spectra")
    gap_l3 = None
    sv_spectrum_path = None
    if os.path.isdir(sv_dir):
        for fname in os.listdir(sv_dir):
            if fname.endswith(".npy") and "level_3" in fname:
                sv_spectrum_path = os.path.join(sv_dir, fname)
                svals = np.load(sv_spectrum_path)
                if len(svals) > 116:
                    gap_l3 = float(svals[115] / max(svals[116], 1e-300))

    return {
        "label": label,
        "charges": {str(k): v for k, v in charges.items()},
        "masses": {str(k): int(v) for k, v in masses.items()},
        "n_samples": n_samples,
        "seed": seed,
        "dims": dims,
        "dim_l3": dims[3] if len(dims) > 3 else None,
        "gap_l3": gap_l3,
        "elapsed_seconds": elapsed,
        "matches_116": dims[3] == 116 if len(dims) > 3 else None,
    }


def phase_revalidate(seed):
    """Re-run Li+ and H2+ at multiple sample counts to test conditioning."""
    from sympy import Integer

    configs = {
        "lithium_ion": {
            "label": "Li+ Ion (+3, -1, -1)",
            "charges": {1: 3, 2: -1, 3: -1},
            "masses": {1: 12789, 2: 1, 3: 1},
        },
        "h2_plus_ion": {
            "label": "H2+ Ion (+1, +1, -1)",
            "charges": {1: 1, 2: 1, 3: -1},
            "masses": {1: 1836, 2: 1836, 3: 1},
        },
    }

    print("\n" + "=" * 70)
    print("PHASE 1: RE-VALIDATION (Li+ and H2+ at multiple sample counts)")
    print("=" * 70)

    results = {}
    for cfg_name, cfg in configs.items():
        for ns in REVALIDATION_SAMPLE_COUNTS:
            if _shutdown_requested:
                return results

            key = f"revalidate_{cfg_name}_s{ns}"
            if is_complete(key):
                manifest = load_manifest()
                results[key] = manifest[key]["result"]
                print(f"\n  [{cfg_name} @ {ns} samples] Already complete, "
                      f"skipping.", flush=True)
                continue

            print(f"\n  [{cfg_name} @ {ns} samples] Running...", flush=True)
            try:
                result = run_single_config(
                    label=f"{cfg['label']} ({ns} samples)",
                    charges=cfg["charges"],
                    masses=cfg["masses"],
                    n_samples=ns,
                    seed=seed,
                )
                results[key] = result
                sync_checkpoints()
                mark_complete(key, result)

                print(f"    dims = {result['dims']}, "
                      f"gap_l3 = {result['gap_l3']}, "
                      f"matches_116 = {result['matches_116']}", flush=True)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                results[key] = {"error": str(e)}
                mark_complete(key, {"error": str(e)})

    print("\n" + "-" * 70)
    print("PHASE 1 SUMMARY: Re-validation")
    print("-" * 70)
    for cfg_name in configs:
        print(f"\n  {configs[cfg_name]['label']}:")
        for ns in REVALIDATION_SAMPLE_COUNTS:
            key = f"revalidate_{cfg_name}_s{ns}"
            r = results.get(key, {})
            if isinstance(r, dict) and "dims" in r:
                print(f"    {ns:>5} samples: {r['dims']}  "
                      f"gap_l3={r.get('gap_l3', '?')}")
            elif isinstance(r, dict) and "error" in r:
                print(f"    {ns:>5} samples: ERROR — {r['error']}")
    print()

    return results


def phase_sweep_qnn(n_samples, seed):
    """Sweep (+q, -1, -1) charge family with equal unit masses."""
    print("\n" + "=" * 70)
    print("PHASE 2: CHARGE SWEEP (+q, -1, -1)")
    print(f"  q values: {SWEEP_QNN}")
    print(f"  Samples: {n_samples},  Seed: {seed}")
    print("=" * 70)

    results = {}
    for q in SWEEP_QNN:
        if _shutdown_requested:
            return results

        key = f"sweep_qnn_q{q}_s{n_samples}"
        if is_complete(key):
            manifest = load_manifest()
            results[key] = manifest[key]["result"]
            print(f"\n  [q={q}] Already complete, skipping.", flush=True)
            continue

        print(f"\n  [q={q}] Running (+{q}, -1, -1) with unit masses...",
              flush=True)
        try:
            result = run_single_config(
                label=f"(+{q}, -1, -1) unit masses",
                charges={1: q, 2: -1, 3: -1},
                masses={1: 1, 2: 1, 3: 1},
                n_samples=n_samples,
                seed=seed,
            )
            results[key] = result
            sync_checkpoints()
            mark_complete(key, result)

            match = "MATCH" if result["matches_116"] else "DIFFERS"
            print(f"    dims = {result['dims']}, "
                  f"gap_l3 = {result['gap_l3']}, [{match}]", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[key] = {"error": str(e)}
            mark_complete(key, {"error": str(e)})

    print("\n" + "-" * 70)
    print("PHASE 2 SUMMARY: (+q, -1, -1) sweep")
    print("-" * 70)
    for q in SWEEP_QNN:
        key = f"sweep_qnn_q{q}_s{n_samples}"
        r = results.get(key, {})
        if isinstance(r, dict) and "dims" in r:
            match = "MATCH" if r.get("matches_116") else "DIFFERS"
            print(f"  q={q:>3}: {r['dims']}  "
                  f"gap_l3={r.get('gap_l3', '?'):<12}  [{match}]")
        elif isinstance(r, dict) and "error" in r:
            print(f"  q={q:>3}: ERROR — {r['error']}")
    print()

    return results


def phase_sweep_qqn(n_samples, seed):
    """Sweep (+1, +q, -1) charge family (H2+-like)."""
    print("\n" + "=" * 70)
    print("PHASE 3: CHARGE SWEEP (+1, +q, -1)")
    print(f"  q values: {SWEEP_QQN}")
    print(f"  Samples: {n_samples},  Seed: {seed}")
    print("=" * 70)

    results = {}
    for q in SWEEP_QQN:
        if _shutdown_requested:
            return results

        key = f"sweep_qqn_q{q}_s{n_samples}"
        if is_complete(key):
            manifest = load_manifest()
            results[key] = manifest[key]["result"]
            print(f"\n  [q={q}] Already complete, skipping.", flush=True)
            continue

        print(f"\n  [q={q}] Running (+1, +{q}, -1) with unit masses...",
              flush=True)
        try:
            result = run_single_config(
                label=f"(+1, +{q}, -1) unit masses",
                charges={1: 1, 2: q, 3: -1},
                masses={1: 1, 2: 1, 3: 1},
                n_samples=n_samples,
                seed=seed,
            )
            results[key] = result
            sync_checkpoints()
            mark_complete(key, result)

            match = "MATCH" if result["matches_116"] else "DIFFERS"
            print(f"    dims = {result['dims']}, "
                  f"gap_l3 = {result['gap_l3']}, [{match}]", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[key] = {"error": str(e)}
            mark_complete(key, {"error": str(e)})

    print("\n" + "-" * 70)
    print("PHASE 3 SUMMARY: (+1, +q, -1) sweep")
    print("-" * 70)
    for q in SWEEP_QQN:
        key = f"sweep_qqn_q{q}_s{n_samples}"
        r = results.get(key, {})
        if isinstance(r, dict) and "dims" in r:
            match = "MATCH" if r.get("matches_116") else "DIFFERS"
            print(f"  q={q:>3}: {r['dims']}  "
                  f"gap_l3={r.get('gap_l3', '?'):<12}  [{match}]")
        elif isinstance(r, dict) and "error" in r:
            print(f"  q={q:>3}: ERROR — {r['error']}")
    print()

    return results


def main():
    global MAX_LEVEL

    ap = argparse.ArgumentParser(
        description="Charge sensitivity sweep for Li+/H2+ investigation")
    ap.add_argument("--phase", type=str, default=None,
                    choices=["revalidate", "sweep_qnn", "sweep_qqn"],
                    help="Run a single phase (default: all)")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                    help=f"Sample count for sweep phases (default: {DEFAULT_SAMPLES})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-level", type=int, default=MAX_LEVEL)
    args = ap.parse_args()

    MAX_LEVEL = args.max_level

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("=" * 70)
    print("CHARGE SENSITIVITY SWEEP")
    print("=" * 70)
    print(f"  S3 bucket:   {S3_BUCKET or '(none -- local only)'}")
    print(f"  Max level:   {MAX_LEVEL}")
    print(f"  Sweep samples: {args.samples}")
    print(f"  Seed:        {args.seed}")
    print(f"  Phase:       {args.phase or 'all'}")
    print(f"  Started:     {strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  SymPy:       ", end="")
    import sympy
    print(sympy.__version__)
    print("=" * 70 + "\n")

    # Pull existing manifest from S3
    if S3_BUCKET:
        try:
            subprocess.run(
                ["aws", "s3", "cp",
                 f"s3://{S3_BUCKET}/{RESULTS_PREFIX}/{MANIFEST_PATH}",
                 MANIFEST_PATH],
                capture_output=True, timeout=60,
            )
        except Exception:
            pass

    t_total = time()
    phases = [args.phase] if args.phase else ["revalidate", "sweep_qnn", "sweep_qqn"]
    all_results = {}

    for phase in phases:
        if _shutdown_requested:
            break

        if phase == "revalidate":
            all_results["revalidate"] = phase_revalidate(args.seed)
        elif phase == "sweep_qnn":
            all_results["sweep_qnn"] = phase_sweep_qnn(args.samples, args.seed)
        elif phase == "sweep_qqn":
            all_results["sweep_qqn"] = phase_sweep_qqn(args.samples, args.seed)

    total_elapsed = time() - t_total

    # Final summary
    print("\n" + "=" * 70)
    if _shutdown_requested:
        print("INTERRUPTED BY SIGTERM — data saved to checkpoint + S3")
    else:
        print("ALL PHASES COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.2f} hours)")
    print("=" * 70)

    sync_checkpoints()

    summary = {
        "status": "interrupted" if _shutdown_requested else "complete",
        "phases_completed": list(all_results.keys()),
        "total_seconds": total_elapsed,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open("charge_sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    s3_cp("charge_sensitivity_summary.json",
          f"{RESULTS_PREFIX}/charge_sensitivity_summary.json")


if __name__ == "__main__":
    main()
