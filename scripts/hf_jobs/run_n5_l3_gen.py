#!/usr/bin/env python3
"""
E5 — N=5 d=1 1/r generation-only probe (HF Jobs cpu-xl, ~$1).

Runs build_generators(max_level=3) for the patched together-based
NBodySymbolicRank engine. NO rank computation. NO checkpoint writes
of the symbolic generators (memory-only).

Records: per-level wall time, per-level generator count, per-level
top-level term statistics, peak RSS (psutil), final result JSON.

Three documented outcomes (see Phase E5 in plan):
  - completes: write {status: 'complete', ...} to /mnt/results/n5_l3.json
  - OOM: container is killed; nothing to do here, post-mortem is HF logs
  - timeout: caught? hf jobs sets SIGTERM at deadline; we install a
    handler that flushes a partial-results JSON before exiting.

This probe is for the engine, not for any scientific result. The new_per_level
generator counts are the deliverable.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import sympy as sp
from sympy import Add

# nbody/ is on sys.path via bootstrap.py
from symbolic_rank_nbody import NBodySymbolicRank  # noqa: E402

try:
    import psutil
    _proc = psutil.Process()
    def _peak_rss_mb() -> float:
        return _proc.memory_info().rss / (1024 * 1024)
except Exception:
    _proc = None
    def _peak_rss_mb() -> float:
        return -1.0


RESULTS_DIR = Path(os.environ.get("HF_3BODY_MNT", "/mnt")) / "results"
RESULTS_PATH = RESULTS_DIR / "n5_l3.json"
PARTIAL_PATH = RESULTS_DIR / "n5_l3.partial.json"

_state = {
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "sympy_version": sp.__version__,
    "config": {
        "n_bodies": 5,
        "d_spatial": 1,
        "potential": "1/r",
        "max_level": 3,
    },
    "by_level": {},
    "elapsed_s_per_level": {},
    "peak_rss_mb_per_level": {},
    "top_level_avg_nterms_per_level": {},
    "top_level_max_nterms_per_level": {},
    "status": "running",
}


def _atomic_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _flush_partial(reason: str):
    _state["status"] = f"partial:{reason}"
    _state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _state["peak_rss_mb_final"] = _peak_rss_mb()
    try:
        _atomic_write(PARTIAL_PATH, _state)
        print(f"[probe] partial flushed to {PARTIAL_PATH} (reason={reason})",
              flush=True)
    except Exception as e:
        print(f"[probe] partial flush failed: {e}", file=sys.stderr,
              flush=True)


def _sigterm_handler(signum, frame):
    print(f"[probe] caught signal {signum}; flushing partial", flush=True)
    _flush_partial("sigterm")
    sys.exit(143)


def main() -> int:
    signal.signal(signal.SIGTERM, _sigterm_handler)
    print(f"[probe] N=5 d=1 1/r max_level=3 (sympy {sp.__version__})",
          flush=True)
    print(f"[probe] peak_rss_mb at start: {_peak_rss_mb():.1f}", flush=True)

    engine = NBodySymbolicRank(
        n_bodies=5, d_spatial=1, potential="1/r")
    print(f"[probe] phase vars: {len(engine.phase_vars)}, "
          f"H_ij count: {len(engine.algebra.hamiltonian_list)}", flush=True)

    # Build_generators returns three lists; we want per-level breakdowns.
    # We can't easily intercept per-level inside build_generators, so we
    # run it once and post-process.
    t0 = time.time()
    exprs, names, levels = engine.build_generators(
        max_level=3, checkpoint_dir=None, n_workers=1)
    total_elapsed = time.time() - t0

    # Per-level breakdowns
    by_level: dict[int, list[int]] = {}  # level -> indices
    for idx, lv in enumerate(levels):
        by_level.setdefault(lv, []).append(idx)
    for lv in sorted(by_level):
        ixs = by_level[lv]
        nterms = [len(Add.make_args(exprs[i])) for i in ixs]
        _state["by_level"][str(lv)] = len(ixs)
        _state["top_level_avg_nterms_per_level"][str(lv)] = round(
            sum(nterms) / max(len(nterms), 1), 2)
        _state["top_level_max_nterms_per_level"][str(lv)] = max(nterms) if nterms else 0
        print(f"[probe] level {lv}: {len(ixs)} generators, "
              f"avg {sum(nterms)/max(len(nterms),1):.2f} terms, "
              f"max {max(nterms) if nterms else 0} terms", flush=True)

    _state["elapsed_total_s"] = round(total_elapsed, 2)
    _state["n_generators"] = len(exprs)
    _state["peak_rss_mb_final"] = _peak_rss_mb()
    _state["status"] = "complete"
    _state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write(RESULTS_PATH, _state)
    print(f"[probe] DONE  total {len(exprs)} generators in {total_elapsed:.1f}s "
          f"peak {_peak_rss_mb():.1f} MB", flush=True)
    print(f"[probe] results -> {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
