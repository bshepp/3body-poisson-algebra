#!/usr/bin/env python3
"""Schwarzschild effective-potential dimension-sequence sweep.

Treats the Schwarzschild radial effective potential as a composite in u = 1/r:

    V_eff(r; M, L) = -M/r + L^2/(2 r^2) - M L^2 / r^3
                   = -M u + (L^2/2) u^2 - M L^2 u^3

This fits NBodyAlgebra(potential='composite', potential_params=[(c1, 1), (c2, 2), (c3, 3)])
exactly: each pair experiences the same composite force law in their relative coord.

Scientific question
-------------------
Is the dimension sequence invariant in (M, L)?  That is the GR analogue of the
mass-invariance theorem already established for Newtonian 1/r.  Three outcomes
are interesting:

  1. All (M, L) -> [3, 6, 17, 116].  Universality survives the GR composite.
  2. (M, L) -> something smaller / different.  Schwarzschild is exceptional.
  3. (M, L) -> sequence varies with (M, L).  Phase boundaries inside (r/M, L/M).

This script runs a sweep over a grid of (M, L) values and dumps results to
results/schwarzschild/dimseq_sweep.json (atomic per-point checkpoint).

Usage
-----
    # Smoke test (3 points, L2, ~2 min):
    python nbody/run_schwarzschild.py --smoke

    # Default sweep (~3 hours overnight, level 3, equal masses):
    python nbody/run_schwarzschild.py

    # Resume an interrupted run (skips points already in the JSON):
    python nbody/run_schwarzschild.py --resume

The default grid is 5x6 = 30 (M, L) points; expected runtime per point is
~5-10 min at level 3 with 500 samples on a workstation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nbody"))

from exact_growth_nbody import NBodyAlgebra  # noqa: E402


RESULTS_DIR = ROOT / "results" / "schwarzschild"
RESULTS_FILE = RESULTS_DIR / "dimseq_sweep.json"
LOG_FILE = RESULTS_DIR / "run.log"


def schwarzschild_params(M: sp.Rational, L: sp.Rational):
    """Return the (coeff, power) list for V_eff in u = 1/r.

    V_eff = -M*u + (L^2/2)*u^2 - M*L^2*u^3
    """
    return [
        (-M, 1),
        (L**2 / sp.Integer(2), 2),
        (-M * L**2, 3),
    ]


def make_grid(mode: str) -> list[tuple[sp.Rational, sp.Rational]]:
    """Build the (M, L) sweep grid.

    Both M and L are dimensionless here (geometric units G=c=1). The
    physically interesting feature is that (M, L) determines where the
    photon sphere (3M), ISCO (6M), and horizon (2M) sit in r.
    """
    if mode == "smoke":
        return [
            (sp.Integer(1), sp.Integer(1)),
            (sp.Integer(1), sp.Integer(4)),
            (sp.Rational(1, 2), sp.Integer(2)),
        ]
    if mode == "key":
        # Four physically-meaningful points for the L3 deep dive:
        #   (M=1, L=1)  : tightly bound, well below ISCO
        #   (M=1, L=2)  : near photon sphere (r_ph = 3M)
        #   (M=1, L=4)  : just above ISCO (r_ISCO = 6M); stable bound regime
        #   (M=2, L=1)  : "heavy hole, low ang mom" - sub-ISCO
        return [
            (sp.Integer(1), sp.Integer(1)),
            (sp.Integer(1), sp.Integer(2)),
            (sp.Integer(1), sp.Integer(4)),
            (sp.Integer(2), sp.Integer(1)),
        ]
    # Full grid: 5 mass scales x 6 angular momenta.
    Ms = [sp.Rational(1, 4), sp.Rational(1, 2), sp.Integer(1),
          sp.Integer(2), sp.Integer(4)]
    Ls = [sp.Rational(1, 2), sp.Integer(1), sp.Integer(2),
          sp.Integer(4), sp.Integer(8), sp.Integer(16)]
    return [(M, L) for M in Ms for L in Ls]


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"runs": []}


def save_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def log_line(msg: str) -> None:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def point_id(M: sp.Rational, L: sp.Rational) -> str:
    return f"M{M}_L{L}".replace("/", "over")


def run_one(M: sp.Rational, L: sp.Rational, max_level: int, n_samples: int,
            seed: int, d_spatial: int, resume: bool = False,
            checkpoint_every: int | None = None) -> dict:
    params = schwarzschild_params(M, L)
    label = f"Schwarzschild M={M}, L={L}, d={d_spatial}, level<={max_level}"
    log_line(f"START  {label}")
    t0 = time.time()
    # NBodyAlgebra's default checkpoint dir hashes only powers, not
    # coefficients - all (M,L) Schwarzschild points would otherwise share
    # the same on-disk cache. Use a per-(M,L) dir to avoid races between
    # parallel runs.
    ckpt_dir = str(ROOT / "nbody" / "checkpoints_schwarzschild" /
                   f"d{d_spatial}_{point_id(M, L)}")
    alg = NBodyAlgebra(
        n_bodies=3,
        d_spatial=d_spatial,
        potential="composite",
        potential_params=params,
        checkpoint_dir=ckpt_dir,
    )
    dims = alg.compute_growth(
        max_level=max_level,
        n_samples=n_samples,
        seed=seed,
        resume=resume,
        checkpoint_every=checkpoint_every,
    )
    elapsed = time.time() - t0
    seq = [int(dims[lv]) for lv in range(max_level + 1)]
    log_line(f"DONE   {label}  ->  {seq}   ({elapsed:.1f}s)")
    return {
        "id": point_id(M, L),
        "M": str(M),
        "L": str(L),
        "d_spatial": d_spatial,
        "max_level": max_level,
        "n_samples": n_samples,
        "seed": seed,
        "potential_params": [[str(c), int(p)] for c, p in params],
        "sequence": seq,
        "elapsed_s": elapsed,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="3-point smoke test at L2, 100 samples (~15s)")
    ap.add_argument("--mode", choices=["full", "key", "smoke"], default="full",
                    help="Grid mode: full=30 (M,L) points; key=4 physical points; "
                         "smoke=3-point smoke test (default: full)")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space samples (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-spatial", type=int, default=2,
                    help="Spatial dimension (default: 2; matches paper baseline)")
    ap.add_argument("--out", default=None,
                    help="Override output JSON path (default: results/schwarzschild/dimseq_sweep.json)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (M,L) points already in the output JSON, and "
                         "for the in-progress point, resume from any "
                         "intra-level partial checkpoint on disk.")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="Save a partial checkpoint every N brackets within "
                         "a level so a crash mid-L3 doesn't lose hours of "
                         "work (default: 10; 0 to disable).")
    args = ap.parse_args()

    if args.smoke:
        args.mode = "smoke"
        args.max_level = min(args.max_level, 2)
        args.samples = min(args.samples, 100)

    out_file = Path(args.out).resolve() if args.out else RESULTS_FILE
    grid = make_grid(args.mode)
    try:
        out_display = out_file.relative_to(ROOT)
    except ValueError:
        out_display = out_file

    log_line("=" * 70)
    log_line(f"Schwarzschild dimseq sweep: mode={args.mode}, {len(grid)} points, "
             f"d={args.d_spatial}, level<={args.max_level}, samples={args.samples}, "
             f"seed={args.seed} -> {out_display}")
    log_line(f"Grid = {[(str(M), str(L)) for M, L in grid]}")
    log_line("=" * 70)

    existing = load_existing(out_file)
    done_ids = {r["id"] for r in existing["runs"]} if args.resume else set()
    if args.resume and done_ids:
        log_line(f"Resume mode: {len(done_ids)} points already complete; skipping them.")

    runs = list(existing["runs"]) if args.resume else []
    failures: list[dict] = []
    cfg = {
        "mode": args.mode,
        "max_level": args.max_level,
        "n_samples": args.samples,
        "seed": args.seed,
        "d_spatial": args.d_spatial,
        "grid_size": len(grid),
        "checkpoint_every": args.checkpoint_every,
        "resume": args.resume,
    }

    for i, (M, L) in enumerate(grid, 1):
        pid = point_id(M, L)
        if pid in done_ids:
            log_line(f"SKIP   ({i}/{len(grid)}) {pid}")
            continue
        try:
            ckpt_every = args.checkpoint_every if args.checkpoint_every > 0 else None
            res = run_one(M, L, args.max_level, args.samples,
                          args.seed, args.d_spatial,
                          resume=args.resume, checkpoint_every=ckpt_every)
            runs.append(res)
            save_atomic(out_file, {"runs": runs, "failures": failures, "config": cfg})
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log_line(f"FAIL   {pid}: {exc}")
            log_line(tb)
            failures.append({
                "id": pid, "M": str(M), "L": str(L),
                "error": str(exc), "traceback": tb,
            })
            save_atomic(out_file, {"runs": runs, "failures": failures, "config": cfg})

    log_line("=" * 70)
    log_line(f"Sweep complete: {len(runs)} runs, {len(failures)} failures.")

    if runs:
        seqs = {tuple(r["sequence"]) for r in runs}
        log_line(f"Distinct dimension sequences observed: {len(seqs)}")
        for s in sorted(seqs):
            count = sum(1 for r in runs if tuple(r["sequence"]) == s)
            log_line(f"  {list(s)}  x  {count}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
