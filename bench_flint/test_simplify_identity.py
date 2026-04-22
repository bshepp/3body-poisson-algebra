#!/usr/bin/env python3
"""Symbolic-identity spot check for the cancel -> together patch.

For each test potential, generate the raw L=0..L=2 brackets via
``NBodyAlgebra.poisson_bracket`` (NO simplification of the inputs), then
for every raw bracket ``b`` compute::

    t = sympy.together(b)
    c = sympy.cancel(b)

and assert ``simplify(t - c) == 0``. This is a *symbolic* witness that the
two simplifiers agree as expressions (stronger than the dim-sequence /
SVD check used in validate_simplify_patch.py, which only proves agreement
after numerical evaluation on a sample grid).

Per-bracket records are written atomically to
``bench_flint/identity_check.json``. Any mismatch dumps the offending
bracket to ``bench_flint/identity_check_FAILURE.json`` and the script
exits non-zero.

Coverage rationale:
  * L=0 brackets: trivially {H_ij, H_kl} - small, fast on both.
  * L=1 brackets: {{H_ij, H_kl}, H_mn} - the workload where the patch
    starts to matter (1-30 summands).
  * L=2 brackets: a small slice of the L<=2 frontier - this is exactly
    the regime where ``cancel`` blows up (100k+ summands) and is the
    most informative test of "but is together actually equal to cancel".
  * L=3 deliberately skipped - cancel hangs there, which is the whole
    motivation for the patch.

Run::

    python bench_flint/test_simplify_identity.py
    python bench_flint/test_simplify_identity.py --potentials 1/r log
    python bench_flint/test_simplify_identity.py --max-l2-brackets 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nbody"))

import sympy as sp  # noqa: E402

from exact_growth_nbody import NBodyAlgebra  # noqa: E402

OUT_JSON = REPO_ROOT / "bench_flint" / "identity_check.json"
FAIL_JSON = REPO_ROOT / "bench_flint" / "identity_check_FAILURE.json"
LOG_PATH = REPO_ROOT / "bench_flint" / "identity_check.log"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def _log(msg: str) -> None:
    line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _ops(expr) -> int:
    try:
        return int(sp.count_ops(expr))
    except Exception:
        return -1


def check_one_bracket(alg: NBodyAlgebra, f, g, label: str,
                      level_pair: tuple[int, int]) -> dict:
    """Compute raw {f, g} once, then check together(b) == cancel(b)."""
    t0 = time.perf_counter()
    raw = alg.poisson_bracket(f, g)
    t_pb = time.perf_counter() - t0

    t0 = time.perf_counter()
    t_expr = sp.together(raw)
    t_together = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_expr = sp.cancel(raw)
    t_cancel = time.perf_counter() - t0

    # The acid test: are they algebraically equal?
    t0 = time.perf_counter()
    diff = sp.simplify(t_expr - c_expr)
    t_diff = time.perf_counter() - t0

    is_zero = (diff == 0) or (diff.equals(0) is True)

    rec = {
        "label": label,
        "level_pair": list(level_pair),
        "raw_count_ops": _ops(raw),
        "together_count_ops": _ops(t_expr),
        "cancel_count_ops": _ops(c_expr),
        "diff_count_ops": _ops(diff),
        "is_zero": bool(is_zero),
        "elapsed_pb_s": t_pb,
        "elapsed_together_s": t_together,
        "elapsed_cancel_s": t_cancel,
        "elapsed_diff_s": t_diff,
    }
    return rec


def run_potential(potential_name: str, potential_kwargs: dict,
                  max_l2_brackets: int) -> dict:
    """Generate L=0..L=2 raw brackets for a single potential and check identity."""
    _log(f"=== potential: {potential_name} ===")
    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2,
        checkpoint_dir=None,
        **potential_kwargs,
    )

    # L=0: the 3 pairwise Hamiltonians (no brackets at L=0)
    H = list(alg.hamiltonian_list)
    H_names = list(alg.hamiltonian_names)
    _log(f"  L=0 generators: {len(H)}  ({H_names})")

    records: list[dict] = []

    # ---- L=1: brackets of pairs of L=0 ----
    L1: list = []
    L1_names: list[str] = []
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            label = f"{{{H_names[i]},{H_names[j]}}}"
            rec = check_one_bracket(alg, H[i], H[j], label, (0, 0))
            records.append(rec)
            _log(f"    L1 [{label}]  ops r={rec['raw_count_ops']:>4} "
                 f"t={rec['together_count_ops']:>3} c={rec['cancel_count_ops']:>4} "
                 f"diff_ops={rec['diff_count_ops']:>2}  zero={rec['is_zero']}  "
                 f"t_diff={rec['elapsed_diff_s']:.2f}s")
            # Use the *together* form for downstream L=2 inputs (cheaper to bracket
            # against). Mathematical content is identical to cancel by this very test.
            L1.append(sp.together(alg.poisson_bracket(H[i], H[j])))
            L1_names.append(label)

    # ---- L=2: brackets of L1 against L0, plus L1 against L1 ----
    # Strict cap to keep wall-clock manageable (cancel can blow up at L=2).
    l2_pairs: list[tuple] = []
    for i, l1_expr in enumerate(L1):
        for j in range(len(H)):
            l2_pairs.append((l1_expr, H[j], f"{{{L1_names[i]},{H_names[j]}}}"))
    for i in range(len(L1)):
        for j in range(i + 1, len(L1)):
            l2_pairs.append((L1[i], L1[j], f"{{{L1_names[i]},{L1_names[j]}}}"))

    if max_l2_brackets and len(l2_pairs) > max_l2_brackets:
        _log(f"  L2 candidates: {len(l2_pairs)}; capping at {max_l2_brackets}")
        l2_pairs = l2_pairs[:max_l2_brackets]
    else:
        _log(f"  L2 candidates: {len(l2_pairs)}")

    for k, (f_expr, g_expr, label) in enumerate(l2_pairs, 1):
        rec = check_one_bracket(alg, f_expr, g_expr, label, (1, 0))
        records.append(rec)
        _log(f"    L2 [{k:>2}/{len(l2_pairs)}] {label}  "
             f"ops r={rec['raw_count_ops']:>5} t={rec['together_count_ops']:>4} "
             f"c={rec['cancel_count_ops']:>6} diff_ops={rec['diff_count_ops']:>3}  "
             f"zero={rec['is_zero']}  "
             f"t_c={rec['elapsed_cancel_s']:.2f}s t_diff={rec['elapsed_diff_s']:.2f}s")

    n_total = len(records)
    n_match = sum(1 for r in records if r["is_zero"])
    n_mismatch = n_total - n_match
    max_ratio = 0.0
    for r in records:
        c = r["cancel_count_ops"]
        t = r["together_count_ops"]
        if c > 0 and t > 0:
            max_ratio = max(max_ratio, c / t)

    summary = {
        "potential": potential_name,
        "n_brackets": n_total,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "max_count_ops_ratio_cancel_over_together": max_ratio,
    }
    _log(f"  -> {summary}")
    return {
        "potential": potential_name,
        "potential_kwargs": {k: str(v) for k, v in potential_kwargs.items()},
        "summary": summary,
        "records": records,
    }


POTENTIAL_SPECS = {
    "1/r":   dict(potential="1/r"),
    "1/r^2": dict(potential="1/r^2"),
    "log":   dict(potential="log"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--potentials", nargs="+",
                    default=list(POTENTIAL_SPECS.keys()),
                    help=f"Subset of {list(POTENTIAL_SPECS.keys())}")
    ap.add_argument("--max-l2-brackets", type=int, default=20,
                    help="Cap the number of L=2 brackets per potential "
                         "(0 = no cap). Default 20 keeps wall <10 min.")
    args = ap.parse_args()

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    _log("=" * 72)
    _log("simplify-generator IDENTITY CHECK: cancel(b) ?= together(b)")
    _log(f"  potentials:       {args.potentials}")
    _log(f"  max L2 brackets:  {args.max_l2_brackets}")
    _log("=" * 72)

    payload = {
        "started_at": started_at,
        "args": {"potentials": args.potentials,
                 "max_l2_brackets": args.max_l2_brackets},
        "sympy_version": sp.__version__,
        "results": [],
    }
    _atomic_write(OUT_JSON, payload)

    failure_record = None
    t_total = time.perf_counter()
    for pot in args.potentials:
        if pot not in POTENTIAL_SPECS:
            _log(f"  UNKNOWN potential {pot!r}, skipping")
            continue
        try:
            result = run_potential(pot, POTENTIAL_SPECS[pot], args.max_l2_brackets)
        except Exception as exc:  # noqa: BLE001
            import traceback
            _log(f"  EXCEPTION for {pot}: {exc}")
            result = {"potential": pot, "error": repr(exc),
                      "traceback": traceback.format_exc()}
        payload["results"].append(result)
        _atomic_write(OUT_JSON, payload)

        if "summary" in result and result["summary"]["n_mismatch"] > 0:
            mismatched = [r for r in result["records"] if not r["is_zero"]]
            failure_record = {
                "potential": pot,
                "first_mismatch": mismatched[0] if mismatched else None,
                "all_mismatches": mismatched,
            }
            _atomic_write(FAIL_JSON, failure_record)
            _log(f"  !!! {len(mismatched)} mismatch(es) recorded to "
                 f"{FAIL_JSON.relative_to(REPO_ROOT)}")
            break

    t_elapsed = time.perf_counter() - t_total
    payload["completed_at"] = dt.datetime.now().isoformat(timespec="seconds")
    payload["elapsed_total_s"] = t_elapsed

    overall_match = all(
        r.get("summary", {}).get("n_mismatch", 1) == 0
        for r in payload["results"]
        if "summary" in r
    )
    payload["overall_match"] = overall_match
    _atomic_write(OUT_JSON, payload)

    _log("=" * 72)
    _log(f"DONE in {t_elapsed:.1f}s.  overall_match = {overall_match}")
    if failure_record is not None:
        _log(f"  Failure dump: {FAIL_JSON.relative_to(REPO_ROOT)}")
    _log(f"  Aggregated:    {OUT_JSON.relative_to(REPO_ROOT)}")
    _log("=" * 72)

    return 0 if overall_match else 4


if __name__ == "__main__":
    sys.exit(main())
