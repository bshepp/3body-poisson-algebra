#!/usr/bin/env python3
"""Tiny smoke target for HF Jobs bootstrap. N=3 d=1 1/r L=2."""
from __future__ import annotations
import os, json, time
from pathlib import Path
import sympy as sp
from symbolic_rank_nbody import NBodySymbolicRank  # noqa: E402

def main():
    print(f"[smoke] sympy {sp.__version__}", flush=True)
    e = NBodySymbolicRank(n_bodies=3, d_spatial=1, potential="1/r")
    t0 = time.time()
    exprs, names, levels = e.build_generators(2, checkpoint_dir=None, n_workers=1)
    elapsed = time.time() - t0
    by_level = {}
    for lv in levels:
        by_level[lv] = by_level.get(lv, 0) + 1
    out = {
        "status": "complete",
        "config": {"n_bodies": 3, "d_spatial": 1, "potential": "1/r", "max_level": 2},
        "by_level": {str(k): v for k, v in by_level.items()},
        "elapsed_s": round(elapsed, 3),
        "n_generators": len(exprs),
    }
    mnt = Path(os.environ.get("HF_3BODY_MNT", "/mnt"))
    rdir = mnt / "results"; rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "smoke.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[smoke] {out}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
