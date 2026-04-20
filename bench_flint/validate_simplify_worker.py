#!/usr/bin/env python3
"""Single-case validation worker.

Takes a JSON case spec on the command line, builds an NBodyAlgebra,
runs ``compute_growth``, writes a JSON result file, exits.

Usage::

    python bench_flint/validate_simplify_worker.py CASE_JSON OUT_JSON

CASE_JSON keys:
  case_id          : string, must be unique
  potential        : "1/r" | "1/r^2" | ... | "log" | "composite"
  potential_params : optional, list of (coeff_str, power) tuples
                     when potential=="composite"
  n_bodies         : int (default 3)
  d_spatial        : int (default 2)
  max_level        : int
  n_samples        : int (default 500)
  seed             : int (default 42)
  masses           : optional dict {body_id (str): "Rational(p,q)" or int}
  charges          : optional dict {body_id (str): int}
  expected         : list[int]  (the canonical sequence we want to confirm)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nbody"))

import sympy as sp  # noqa: E402

from exact_growth_nbody import NBodyAlgebra  # noqa: E402


def _parse_rational(s):
    """Accept int, str ('Rational(7344, 1)' or '7344/1' or '7344')."""
    if isinstance(s, int):
        return sp.Integer(s)
    if isinstance(s, str):
        s = s.strip()
        if s.startswith("Rational("):
            inside = s[len("Rational("):-1]
            p, q = (x.strip() for x in inside.split(","))
            return sp.Rational(int(p), int(q))
        if "/" in s:
            p, q = s.split("/")
            return sp.Rational(int(p.strip()), int(q.strip()))
        return sp.Integer(int(s))
    raise ValueError(f"can't parse rational: {s!r}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    case_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    with open(case_path, "r", encoding="utf-8") as f:
        case = json.load(f)

    case_id = case["case_id"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_bodies = int(case.get("n_bodies", 3))
    d_spatial = int(case.get("d_spatial", 2))
    max_level = int(case["max_level"])
    n_samples = int(case.get("n_samples", 500))
    seed = int(case.get("seed", 42))
    expected = list(case["expected"])
    potential = case["potential"]

    # masses + charges
    masses = None
    if case.get("masses"):
        masses = {int(k): _parse_rational(v) for k, v in case["masses"].items()}
    charges = None
    if case.get("charges"):
        charges = {int(k): int(v) for k, v in case["charges"].items()}

    # composite potential params
    potential_params = None
    if potential == "composite":
        params_raw = case["potential_params"]
        potential_params = []
        for coeff, power in params_raw:
            potential_params.append((_parse_rational(coeff), int(power)))

    # external potential (harmonic trap)
    external_potential = None
    if case.get("external_potential"):
        ext = case["external_potential"]
        external_potential = {"omega": _parse_rational(ext["omega"])}

    # Use a per-case checkpoint dir so we always re-derive cleanly.
    ckpt_dir = REPO_ROOT / "bench_flint" / "_validate_cache" / case_id
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[{started_at}] case={case_id} potential={potential} "
          f"N={n_bodies} d={d_spatial} L<={max_level} samples={n_samples} "
          f"masses={masses} charges={charges} expected={expected}", flush=True)

    result = {
        "case_id": case_id,
        "potential": potential,
        "n_bodies": n_bodies,
        "d_spatial": d_spatial,
        "max_level": max_level,
        "n_samples": n_samples,
        "seed": seed,
        "expected": expected,
        "started_at": started_at,
    }

    t0 = time.perf_counter()
    try:
        alg = NBodyAlgebra(
            n_bodies=n_bodies, d_spatial=d_spatial,
            potential=potential,
            masses=masses, charges=charges,
            potential_params=potential_params,
            external_potential=external_potential,
            checkpoint_dir=str(ckpt_dir),
        )
        dims = alg.compute_growth(
            max_level=max_level, n_samples=n_samples, seed=seed,
        )
    except KeyboardInterrupt:
        result.update({
            "elapsed_s": time.perf_counter() - t0,
            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "status": "interrupted",
        })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return 130
    except Exception as exc:
        import traceback
        result.update({
            "elapsed_s": time.perf_counter() - t0,
            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "status": "exception",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    seq = [int(dims[lv]) for lv in range(max_level + 1)]
    match = (seq == expected)

    result.update({
        "elapsed_s": elapsed,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "done",
        "sequence": seq,
        "match": match,
    })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    flag = "MATCH" if match else "MISMATCH"
    print(f"[{result['completed_at']}] case={case_id} elapsed={elapsed:.1f}s "
          f"sequence={seq} expected={expected} {flag}", flush=True)
    return 0 if match else 3


if __name__ == "__main__":
    sys.exit(main())
