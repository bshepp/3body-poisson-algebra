#!/usr/bin/env python3
"""Phase D - N=3 L=4 symbolic-rank campaign on HF Jobs cpu-xl."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path

import sympy as sp
from symbolic_rank_nbody import NBodySymbolicRank

try:
    import psutil
    _proc = psutil.Process()
    def _peak_rss_mb():
        return _proc.memory_info().rss / (1024 * 1024)
except Exception:
    def _peak_rss_mb():
        return -1.0


MNT = Path(os.environ.get("HF_3BODY_MNT", "/mnt"))
RESULTS_DIR = MNT / "results"
CHECKPOINT_DIR = MNT / "checkpoints"

_state = {
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "sympy_version": sp.__version__,
    "config": {},
    "phase": "init",
    "by_level_n_generators": {},
    "elapsed_s": {"build": None, "extract": None, "rank": None},
    "matrix_shape": None,
    "rank_per_level": {},
    "cumulative_rank": [],
    "new_per_level": [],
    "peak_rss_mb": -1.0,
    "status": "running",
}


def _atomic_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _flush(path, reason=None):
    _state["peak_rss_mb"] = max(_state.get("peak_rss_mb", -1.0), _peak_rss_mb())
    _state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if reason:
        _state["status"] = f"partial:{reason}"
    _atomic_write(path, _state)
    print(f"[d-l4] flushed {path.name} status={_state['status']}", flush=True)


def _make_partial_path(potential, d):
    tag = potential.replace("/", "")
    return RESULTS_DIR / f"l4_{tag}_d{d}.partial.json"


def _make_final_path(potential, d):
    tag = potential.replace("/", "")
    return RESULTS_DIR / f"l4_{tag}_d{d}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--potential", required=True, choices=["1/r", "log"])
    ap.add_argument("--d-spatial", type=int, default=None)
    ap.add_argument("--n-bodies", type=int, default=3)
    ap.add_argument("--max-level", type=int, default=4)
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()

    if args.d_spatial is None:
        args.d_spatial = 2 if args.potential == "log" else 1
    if args.n_workers is None:
        args.n_workers = cpu_count()

    _state["config"] = {
        "n_bodies": args.n_bodies, "d_spatial": args.d_spatial,
        "potential": args.potential, "max_level": args.max_level,
        "n_workers": args.n_workers,
    }
    partial_path = _make_partial_path(args.potential, args.d_spatial)
    final_path = _make_final_path(args.potential, args.d_spatial)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _sigterm(signum, frame):
        print(f"[d-l4] caught signal {signum}; flushing partial", flush=True)
        _flush(partial_path, reason="sigterm")
        sys.exit(143)
    signal.signal(signal.SIGTERM, _sigterm)

    print(f"[d-l4] N={args.n_bodies} d={args.d_spatial} potential={args.potential} max_level={args.max_level} workers={args.n_workers} (sympy {sp.__version__})", flush=True)
    print(f"[d-l4] cpu_count={cpu_count()} peak_rss_mb={_peak_rss_mb():.1f}", flush=True)

    engine = NBodySymbolicRank(n_bodies=args.n_bodies, d_spatial=args.d_spatial, potential=args.potential)

    _state["phase"] = "build"
    t0 = time.time()
    exprs, names, levels = engine.build_generators(max_level=args.max_level, checkpoint_dir=str(CHECKPOINT_DIR), n_workers=args.n_workers)
    build_elapsed = time.time() - t0
    _state["elapsed_s"]["build"] = round(build_elapsed, 2)
    by_level = {}
    for lv in levels:
        by_level[lv] = by_level.get(lv, 0) + 1
    for lv, n in by_level.items():
        _state["by_level_n_generators"][str(lv)] = n
    _state["peak_rss_mb"] = max(_state["peak_rss_mb"], _peak_rss_mb())
    print(f"[d-l4] build done: {len(exprs)} generators in {build_elapsed:.1f}s peak {_peak_rss_mb():.1f} MB", flush=True)
    for lv in sorted(by_level):
        print(f"  level {lv}: {by_level[lv]} generators", flush=True)
    _flush(partial_path)

    _state["phase"] = "extract"
    t0 = time.time()
    poly_list, monom_list, monom_to_idx = engine.extract_monomial_matrix(exprs, n_workers=args.n_workers)
    ext_elapsed = time.time() - t0
    _state["elapsed_s"]["extract"] = round(ext_elapsed, 2)
    _state["matrix_shape"] = [len(exprs), len(monom_list)]
    _state["peak_rss_mb"] = max(_state["peak_rss_mb"], _peak_rss_mb())
    print(f"[d-l4] extract done: matrix {len(exprs)}x{len(monom_list)} in {ext_elapsed:.1f}s peak {_peak_rss_mb():.1f} MB", flush=True)
    _flush(partial_path)

    _state["phase"] = "rank"
    t0 = time.time()
    rank_results = engine.compute_exact_rank(poly_list, monom_list, monom_to_idx, levels, checkpoint_dir=str(CHECKPOINT_DIR))
    rank_elapsed = time.time() - t0
    _state["elapsed_s"]["rank"] = round(rank_elapsed, 2)
    _state["rank_per_level"] = {str(k): int(v) for k, v in rank_results.items()}
    cumulative = [int(rank_results[lv]) for lv in sorted(rank_results)]
    _state["cumulative_rank"] = cumulative
    new_per_level = []
    prev = 0
    for r in cumulative:
        new_per_level.append(int(r - prev))
        prev = r
    _state["new_per_level"] = new_per_level
    _state["peak_rss_mb"] = max(_state["peak_rss_mb"], _peak_rss_mb())
    print(f"[d-l4] rank done in {rank_elapsed:.1f}s peak {_peak_rss_mb():.1f} MB", flush=True)
    print(f"[d-l4] cumulative_rank = {cumulative}", flush=True)
    print(f"[d-l4] new_per_level   = {new_per_level}", flush=True)

    _state["phase"] = "done"
    _state["status"] = "complete"
    _flush(final_path)
    try:
        partial_path.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"[d-l4] DONE  results -> {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())