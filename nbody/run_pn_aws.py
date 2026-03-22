#!/usr/bin/env python3
"""
AWS orchestrator for composite-potential and post-Newtonian runs.

Runs all three test scripts sequentially with:
  - S3 sync of checkpoints + results after each completed config
  - SIGTERM handling for graceful spot-instance shutdown
  - Heartbeat JSON for remote monitoring
  - Resume from checkpoint on re-launch
  - Completion manifest for skip-if-done logic

Usage:
    python run_pn_aws.py                     # run everything
    python run_pn_aws.py --phase composite   # only composite test
    python run_pn_aws.py --phase pn          # only post-Newtonian
    python run_pn_aws.py --phase pn_mass     # only mass invariance
"""

import os
import sys
import json
import signal
import argparse
import subprocess
from time import time, strftime

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

S3_BUCKET = os.environ.get("S3_BUCKET", "")
RESULTS_PREFIX = "results/composite_pn"

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  [SIGTERM] Spot reclamation -- finishing current config...",
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


def s3_pull(s3_prefix, local_dir):
    if not S3_BUCKET:
        return
    src = f"s3://{S3_BUCKET}/{s3_prefix}"
    try:
        subprocess.run(
            ["aws", "s3", "sync", src, local_dir],
            capture_output=True, timeout=180,
        )
    except Exception as e:
        print(f"  [S3 pull warning: {e}]", flush=True)


MANIFEST_PATH = "pn_completion.json"


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    s3_cp(MANIFEST_PATH, f"{RESULTS_PREFIX}/pn_completion.json")


def mark_complete(phase, result):
    manifest = load_manifest()
    manifest[phase] = {
        "status": "complete",
        "result": result,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_manifest(manifest)


def is_complete(phase):
    manifest = load_manifest()
    return phase in manifest and manifest[phase].get("status") == "complete"


def sync_all_checkpoints():
    """Sync all checkpoint directories to S3."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for d in os.listdir(script_dir):
        full = os.path.join(script_dir, d)
        if os.path.isdir(full) and d.startswith("checkpoints"):
            s3_sync(full, f"nbody_checkpoints/{d}")


def run_phase_composite(max_level, n_samples, seed):
    """Run composite universality test (3 configs)."""
    from sympy import Rational, Integer
    from exact_growth_nbody import NBodyAlgebra

    configs = {
        "control_1r": {
            "label": "Control: single 1/r term",
            "params": [(-Integer(1), 1)],
        },
        "two_term": {
            "label": "Composite: -u - u^2  (1/r + 1/r^2)",
            "params": [(-Integer(1), 1), (-Integer(1), 2)],
        },
        "three_term": {
            "label": "Composite: -u - u^2/2 - 3u^3/10  (three terms)",
            "params": [(-Integer(1), 1), (-Rational(1, 2), 2),
                       (-Rational(3, 10), 3)],
        },
    }

    EXPECTED = {0: 3, 1: 6, 2: 17, 3: 116}
    results = {}

    for cfg_name, cfg in configs.items():
        sub_key = f"composite_{cfg_name}"
        if is_complete(sub_key):
            print(f"\n  [{cfg_name}] Already complete, skipping.", flush=True)
            manifest = load_manifest()
            results[cfg_name] = manifest[sub_key]["result"]
            continue

        if _shutdown_requested:
            print(f"  [SHUTDOWN] Skipping {cfg_name}", flush=True)
            return None

        print("\n" + "#" * 70)
        print(f"# {cfg['label']}")
        print(f"# Config: {cfg_name}")
        print("#" * 70 + "\n")

        alg = NBodyAlgebra(
            n_bodies=3, d_spatial=2,
            potential_params=cfg["params"],
        )
        dims = alg.compute_growth(
            max_level=max_level, n_samples=n_samples,
            seed=seed, resume=True,
        )
        seq = [dims[lv] for lv in range(max_level + 1)]
        results[cfg_name] = seq

        sync_all_checkpoints()
        mark_complete(sub_key, seq)

        if _shutdown_requested:
            return None

    print("\n" + "=" * 70)
    print("COMPOSITE UNIVERSALITY TEST -- SUMMARY")
    print("=" * 70)

    all_match = True
    for cfg_name, seq in results.items():
        expected_seq = [EXPECTED.get(lv, "?") for lv in range(len(seq))]
        match = all(seq[lv] == EXPECTED[lv]
                    for lv in range(len(seq)) if lv in EXPECTED)
        if not match:
            all_match = False
        status = "MATCH" if match else "DIFFERS"
        print(f"  {configs[cfg_name]['label']}")
        print(f"    Sequence: {seq}  (expected {expected_seq})  [{status}]")

    verdict = "UNIVERSALITY_HOLDS" if all_match else "UNIVERSALITY_BROKEN"
    print(f"\n  VERDICT: {verdict}")
    return {"results": results, "verdict": verdict}


def run_phase_pn(max_level, n_samples, seed):
    """Run static 1PN test (3 c-values + Newtonian baseline)."""
    from sympy import Rational, Integer
    from exact_growth_nbody import NBodyAlgebra

    EXPECTED = {0: 3, 1: 6, 2: 17, 3: 116}
    c_values = [10, 50, 100]
    results = {}

    for c_val in c_values:
        sub_key = f"pn_c{c_val}"
        if is_complete(sub_key):
            print(f"\n  [1PN c={c_val}] Already complete, skipping.",
                  flush=True)
            manifest = load_manifest()
            results[c_val] = manifest[sub_key]["result"]
            continue

        if _shutdown_requested:
            return None

        c2 = c_val * c_val
        pn_coeff = Rational(-1, 2 * c2)
        print("\n" + "#" * 70)
        print(f"# 1PN with c={c_val} (correction ~ {float(pn_coeff):.2e})")
        print("#" * 70 + "\n")

        alg = NBodyAlgebra(
            n_bodies=3, d_spatial=2,
            potential_params=[(-Integer(1), 1), (pn_coeff, 2)],
        )
        dims = alg.compute_growth(
            max_level=max_level, n_samples=n_samples,
            seed=seed, resume=True,
        )
        seq = [dims[lv] for lv in range(max_level + 1)]
        results[c_val] = seq

        sync_all_checkpoints()
        mark_complete(sub_key, seq)

        if _shutdown_requested:
            return None

    sub_key = "pn_newtonian"
    if is_complete(sub_key):
        print("\n  [Newtonian] Already complete, skipping.", flush=True)
        manifest = load_manifest()
        results["Newtonian"] = manifest[sub_key]["result"]
    elif not _shutdown_requested:
        print("\n" + "#" * 70)
        print("# Baseline: Pure Newtonian (-u)")
        print("#" * 70 + "\n")

        alg = NBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r")
        dims = alg.compute_growth(
            max_level=max_level, n_samples=n_samples,
            seed=seed, resume=True,
        )
        seq = [dims[lv] for lv in range(max_level + 1)]
        results["Newtonian"] = seq

        sync_all_checkpoints()
        mark_complete(sub_key, seq)

    if _shutdown_requested:
        return None

    print("\n" + "=" * 70)
    print("POST-NEWTONIAN THREE-BODY -- SUMMARY")
    print("=" * 70)

    all_match = True
    for key in [*c_values, "Newtonian"]:
        if key not in results:
            continue
        seq = results[key]
        expected_seq = [EXPECTED.get(lv, "?") for lv in range(len(seq))]
        match = all(seq[lv] == EXPECTED[lv]
                    for lv in range(len(seq)) if lv in EXPECTED)
        if not match:
            all_match = False
        status = "MATCH" if match else "DIFFERS"
        if key == "Newtonian":
            label = "Pure Newtonian (V = -u)"
        else:
            label = f"1PN c={key} (V = -u - u^2/{2*key*key})"
        print(f"  {label}")
        print(f"    Sequence: {seq}  (expected {expected_seq})  [{status}]")

    verdict = "PN_PRESERVES" if all_match else "PN_CHANGES"
    print(f"\n  VERDICT: {verdict}")
    return {"results": {str(k): v for k, v in results.items()},
            "verdict": verdict}


def run_phase_pn_mass(max_level, n_samples, seed, c_val=10):
    """Run 1PN mass-invariance test (3 mass configs)."""
    from itertools import combinations
    from sympy import Rational, Integer
    from exact_growth_nbody import NBodyAlgebra

    EXPECTED = {0: 3, 1: 6, 2: 17, 3: 116}

    mass_configs = {
        "equal":        {1: 1, 2: 1, 3: 1},
        "hierarchical": {1: 100, 2: 10, 3: 1},
        "mixed":        {1: 3, 2: 7, 3: 11},
    }

    results = {}

    for cfg_name, masses in mass_configs.items():
        sub_key = f"pn_mass_{cfg_name}"
        if is_complete(sub_key):
            print(f"\n  [{cfg_name}] Already complete, skipping.", flush=True)
            manifest = load_manifest()
            results[cfg_name] = manifest[sub_key]["result"]
            continue

        if _shutdown_requested:
            return None

        c2 = c_val * c_val
        mass_str = ", ".join(f"m{k}={v}" for k, v in sorted(masses.items()))
        mass_label = "_".join(f"m{b}{masses[b]}" for b in sorted(masses))
        ckpt_label = f"1PN_c{c_val}_{mass_label}"

        print("\n" + "#" * 70)
        print(f"# 1PN mass config: {cfg_name}  ({mass_str})")
        print(f"# c = {c_val}")
        print("#" * 70 + "\n")

        alg = NBodyAlgebra(
            n_bodies=3, d_spatial=2,
            potential="1/r", masses=masses,
            checkpoint_dir=f"checkpoints_pn_{ckpt_label}",
        )

        def kinetic(body):
            m = alg.masses[body]
            return sum(p ** 2 for p in alg.p_by_body[body]) / (2 * m)

        alg.hamiltonians = {}
        alg.hamiltonian_list = []
        alg.hamiltonian_names = []

        for bi, bj in combinations(sorted(masses.keys()), 2):
            u = alg.u_by_pair[(bi, bj)]
            mi, mj = masses[bi], masses[bj]
            newton_coeff = -mi * mj
            pn_coeff = Rational(-mi * mj * (mi + mj), 2 * c2)
            V = newton_coeff * u + pn_coeff * u ** 2
            H = kinetic(bi) + kinetic(bj) + V

            name = f"H{bi}{bj}"
            alg.hamiltonians[name] = H
            alg.hamiltonian_list.append(H)
            alg.hamiltonian_names.append(name)

        alg.potential = "composite"
        alg.potential_params = [("pair-dependent", 1), ("pair-dependent", 2)]

        dims = alg.compute_growth(
            max_level=max_level, n_samples=n_samples,
            seed=seed, resume=True,
        )
        seq = [dims[lv] for lv in range(max_level + 1)]
        results[cfg_name] = seq

        sync_all_checkpoints()
        mark_complete(sub_key, seq)

        if _shutdown_requested:
            return None

    print("\n" + "=" * 70)
    print("POST-NEWTONIAN MASS INVARIANCE -- SUMMARY")
    print("=" * 70)

    all_match = True
    all_same = True
    first_seq = None

    for cfg_name, seq in results.items():
        masses = mass_configs[cfg_name]
        mass_str = ", ".join(f"m{k}={v}" for k, v in sorted(masses.items()))
        expected_seq = [EXPECTED.get(lv, "?") for lv in range(len(seq))]
        match = all(seq[lv] == EXPECTED[lv]
                    for lv in range(len(seq)) if lv in EXPECTED)
        if not match:
            all_match = False
        status = "MATCH" if match else "DIFFERS"

        if first_seq is None:
            first_seq = seq
        elif seq != first_seq:
            all_same = False

        print(f"  {cfg_name} ({mass_str})")
        print(f"    Sequence: {seq}  (expected {expected_seq})  [{status}]")

    verdict = "MASS_INVARIANT" if (all_match and all_same) else "MASS_DEPENDENT"
    print(f"\n  VERDICT: {verdict}")
    return {"results": results, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(
        description="AWS orchestrator for composite/PN runs")
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase", type=str, default=None,
                    choices=["composite", "pn", "pn_mass"],
                    help="Run only one phase (default: all)")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("=" * 70)
    print("COMPOSITE POTENTIALS & POST-NEWTONIAN ORCHESTRATOR")
    print("=" * 70)
    print(f"  S3 bucket:  {S3_BUCKET or '(none -- local only)'}")
    print(f"  Max level:  {args.max_level}")
    print(f"  Samples:    {args.samples}")
    print(f"  Seed:       {args.seed}")
    print(f"  Phase:      {args.phase or 'all'}")
    print(f"  Started:    {strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    t_total = time()
    phases = [args.phase] if args.phase else ["composite", "pn", "pn_mass"]
    all_results = {}

    s3_pull(f"{RESULTS_PREFIX}/pn_completion.json", ".")

    for phase in phases:
        if _shutdown_requested:
            break

        if is_complete(phase):
            print(f"\n  Phase '{phase}' already complete, skipping.\n",
                  flush=True)
            continue

        print(f"\n{'='*70}")
        print(f"  PHASE: {phase}")
        print(f"{'='*70}\n")

        if phase == "composite":
            result = run_phase_composite(
                args.max_level, args.samples, args.seed)
        elif phase == "pn":
            result = run_phase_pn(
                args.max_level, args.samples, args.seed)
        elif phase == "pn_mass":
            result = run_phase_pn_mass(
                args.max_level, args.samples, args.seed)
        else:
            continue

        if result is not None:
            all_results[phase] = result
            mark_complete(phase, result)

    total_elapsed = time() - t_total

    print(f"\n{'='*70}")
    if _shutdown_requested:
        print("INTERRUPTED BY SIGTERM -- data saved to checkpoint + S3")
        print("Re-launch instance to resume from checkpoint.")
    else:
        print("ALL PHASES COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.2f} hours)")
    print("=" * 70)

    sync_all_checkpoints()

    final_summary = {
        "status": "interrupted" if _shutdown_requested else "complete",
        "phases_completed": list(all_results.keys()),
        "results": {k: v.get("verdict", "?") for k, v in all_results.items()},
        "total_seconds": total_elapsed,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open("pn_final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    s3_cp("pn_final_summary.json", f"{RESULTS_PREFIX}/pn_final_summary.json")


if __name__ == "__main__":
    main()
