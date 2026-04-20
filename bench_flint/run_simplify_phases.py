#!/usr/bin/env python3
"""Orchestrator for the smarter-simplify three-phase experiment.

Runs each phase's strategies serially under the watchdog (hard timeout +
RSS cap). Phase 2 is gated on Phase 1 passing (all dim seqs match). Phase
3 is gated on Phase 2 passing.

Outputs (atomic writes after each phase):
  bench_flint/phase1_results.json
  bench_flint/phase2_results.json
  bench_flint/phase3_results.json
  bench_flint/simplify_phases_orchestrator.log

Usage::

    python bench_flint/run_simplify_phases.py
    python bench_flint/run_simplify_phases.py --only-phase 1
    python bench_flint/run_simplify_phases.py --start-phase 2

CTRL-C cleanly kills any running worker via the watchdog.
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
sys.path.insert(0, str(REPO_ROOT / "bench_flint"))

from watchdog import run_with_limits  # noqa: E402

WORKER = REPO_ROOT / "bench_flint" / "run_one_strategy.py"
LOG_PATH = REPO_ROOT / "bench_flint" / "simplify_phases_orchestrator.log"


# --------------------------------------------------------------------------- #
# Phase definitions
# --------------------------------------------------------------------------- #

PHASES = {
    1: {
        "label": "Phase 1 - Schwarzschild composite L<=2 sanity gate",
        "case": "schwarz_l2",
        "strategies": ["cancel", "together", "identity"],
        "expected_seq": [3, 6, 17],
        "per_strategy_timeout_s": 180,    # 3 min each
        "per_strategy_ram_cap_gb": 4.0,
        "phase_budget_s": 10 * 60,        # 10 min total budget
    },
    2: {
        "label": "Phase 2 - N=3 1/r L<=3 controlled baseline",
        "case": "1r_l3",
        "strategies": ["cancel", "together"],
        "expected_seq": [3, 6, 17, 116],
        "per_strategy_timeout_s": 90 * 60,     # 90 min each
        "per_strategy_ram_cap_gb": 16.0,
        "phase_budget_s": 3 * 60 * 60,         # 3 hr total
    },
    3: {
        "label": "Phase 3 - Schwarzschild composite L<=3 production",
        "case": "schwarz_l3",
        "strategies": ["cancel", "together"],
        "expected_seq": [3, 6, 17, 116],
        "per_strategy_timeout_s": 2 * 60 * 60, # 2 hr each
        "per_strategy_ram_cap_gb": 16.0,
        "phase_budget_s": 4 * 60 * 60,         # 4 hr total
    },
}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def banner(msg: str) -> None:
    log("=" * 72)
    log(msg)
    log("=" * 72)


# --------------------------------------------------------------------------- #
# Run a single phase
# --------------------------------------------------------------------------- #

def run_phase(num: int, spec: dict, phase_out: Path) -> dict:
    banner(spec["label"])
    log(f"  case={spec['case']}  strategies={spec['strategies']}  "
        f"expected={spec['expected_seq']}")
    log(f"  per_strategy_timeout={spec['per_strategy_timeout_s']}s  "
        f"ram_cap={spec['per_strategy_ram_cap_gb']}GB  "
        f"phase_budget={spec['phase_budget_s']}s")

    phase_t0 = time.perf_counter()
    runs: list[dict] = []
    aborted = False
    abort_reason: str | None = None

    for strategy in spec["strategies"]:
        elapsed_phase = time.perf_counter() - phase_t0
        remaining_budget = spec["phase_budget_s"] - elapsed_phase
        if remaining_budget <= 0:
            log(f"  >>> phase budget exhausted before {strategy}; aborting "
                f"remaining strategies")
            aborted = True
            abort_reason = "phase_budget_exhausted"
            break

        per_strategy_timeout = min(
            spec["per_strategy_timeout_s"], remaining_budget
        )

        out_json = REPO_ROOT / "bench_flint" / f"p{num}_{strategy}.json"
        worker_log = REPO_ROOT / "bench_flint" / f"p{num}_{strategy}.log"

        log(f"\n  --- {strategy} ---")
        log(f"  worker output -> {out_json.relative_to(REPO_ROOT)}")
        log(f"  worker log    -> {worker_log.relative_to(REPO_ROOT)}")
        log(f"  effective timeout: {per_strategy_timeout:.0f}s, "
            f"RAM cap: {spec['per_strategy_ram_cap_gb']}GB")

        last_tick = [time.perf_counter()]

        def on_tick(elapsed_s: float, rss_gb: float,
                    _strategy=strategy, _last=last_tick) -> None:
            now = time.perf_counter()
            if now - _last[0] >= 60.0:
                log(f"    [tick] {_strategy} elapsed={elapsed_s:6.0f}s  "
                    f"rss={rss_gb:5.2f}GB")
                _last[0] = now

        wd_result = run_with_limits(
            [sys.executable, str(WORKER), strategy, spec["case"], str(out_json)],
            timeout_s=per_strategy_timeout,
            ram_cap_gb=spec["per_strategy_ram_cap_gb"],
            poll_s=5.0,
            cwd=REPO_ROOT,
            log_path=worker_log,
            on_tick=on_tick,
        )

        # Load worker JSON if it managed to write one.
        worker_data: dict | None = None
        if out_json.exists():
            try:
                with open(out_json, "r", encoding="utf-8") as f:
                    worker_data = json.load(f)
            except Exception as exc:
                log(f"  WARN: could not parse {out_json}: {exc}")

        merged = {
            "strategy": strategy,
            "case": spec["case"],
            "watchdog": wd_result,
            "worker": worker_data,
        }
        runs.append(merged)

        wd_status = wd_result.get("status", "?")
        elapsed_s = wd_result.get("elapsed_s", 0.0)
        peak_gb = wd_result.get("peak_rss_gb", 0.0)
        if wd_status == "done":
            seq = (worker_data or {}).get("sequence")
            match = (worker_data or {}).get("match")
            log(f"  >>> done  exit={wd_result.get('exit_code')}  "
                f"elapsed={elapsed_s:.1f}s  peak_rss={peak_gb:.2f}GB  "
                f"seq={seq}  match={match}")
        else:
            log(f"  >>> {wd_status.upper()}  elapsed={elapsed_s:.1f}s  "
                f"peak_rss={peak_gb:.2f}GB  "
                f"rss_at_kill={wd_result.get('rss_gb', 'n/a')}")
            # If RAM cap or timeout fires, do not run further strategies in
            # this phase - probably indicates a real problem.
            if wd_status in {"ram_cap_exceeded", "timeout", "interrupted",
                             "launch_failed"}:
                aborted = True
                abort_reason = wd_status
                # still continue iteration only if interrupted? Stop on all.
                if wd_status == "interrupted":
                    break

        # Atomic write of phase results after each strategy.
        phase_summary = {
            "phase": num,
            "label": spec["label"],
            "case": spec["case"],
            "expected_seq": spec["expected_seq"],
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "runs": runs,
            "aborted": aborted,
            "abort_reason": abort_reason,
        }
        _atomic_write_json(phase_out, phase_summary)

        if aborted:
            break

    # Phase verdict.
    all_done = all(
        r["watchdog"].get("status") == "done"
        and r.get("worker") and r["worker"].get("match") is True
        for r in runs
    ) and len(runs) == len(spec["strategies"])

    elapsed_phase = time.perf_counter() - phase_t0
    phase_summary = {
        "phase": num,
        "label": spec["label"],
        "case": spec["case"],
        "expected_seq": spec["expected_seq"],
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": elapsed_phase,
        "runs": runs,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "passed": all_done and not aborted,
    }
    _atomic_write_json(phase_out, phase_summary)

    log(f"\n  Phase {num} complete in {elapsed_phase:.1f}s. "
        f"passed={phase_summary['passed']}")
    return phase_summary


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-phase", type=int, choices=[1, 2, 3], default=None,
                    help="Run only the specified phase")
    ap.add_argument("--start-phase", type=int, choices=[1, 2, 3], default=1,
                    help="Skip phases below this (default: 1)")
    ap.add_argument("--ignore-gates", action="store_true",
                    help="Run all phases regardless of prior phase pass/fail")
    args = ap.parse_args()

    banner("SMARTER-SIMPLIFY ORCHESTRATOR START")
    log(f"  python: {sys.executable}")
    log(f"  repo:   {REPO_ROOT}")
    log(f"  worker: {WORKER}")

    phases_to_run = (
        [args.only_phase] if args.only_phase
        else [n for n in (1, 2, 3) if n >= args.start_phase]
    )
    log(f"  phases to run: {phases_to_run}")

    summaries: dict[int, dict] = {}
    for n in phases_to_run:
        spec = PHASES[n]
        out = REPO_ROOT / "bench_flint" / f"phase{n}_results.json"
        try:
            s = run_phase(n, spec, out)
        except KeyboardInterrupt:
            log(f"  KeyboardInterrupt during phase {n}; stopping")
            break
        summaries[n] = s
        if not s["passed"] and not args.ignore_gates:
            log(f"\n  Phase {n} did not pass. Stopping (use --ignore-gates "
                f"to override).")
            break

    banner("ORCHESTRATOR DONE")
    for n, s in summaries.items():
        log(f"  Phase {n}: passed={s['passed']}  elapsed={s.get('elapsed_s', 0):.1f}s")

    return 0 if all(s.get("passed") for s in summaries.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
