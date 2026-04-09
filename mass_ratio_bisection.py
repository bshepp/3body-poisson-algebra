#!/usr/bin/env python3
"""
HISTORICAL ARTIFACT — DO NOT USE FOR NEW ANALYSIS
===================================================
This script and its results in bisection_results/ were produced on
April 8, 2026 using a "term-group factored evaluation" approach that
has since been reverted from exact_growth.py and stability_atlas.py.

The term-group method split each generator's terms by coefficient
magnitude order, lambdified each group separately, and passed an
"expanded" column matrix to SVD (one column per magnitude group rather
than per generator). This produced inflated rank values (up to 200+)
that were artifacts of spurious numerical independence between groups,
NOT genuine algebraic structure. The mpmath "ground truth" labels in
this script are also unreliable — mpmath evaluation at sample points
is still numerical, not algebraic.

The results are preserved for historical reference. For definitive
mass-invariance results, use the symbolic rank over Q approach
(gap_workplan.md §4.4).

Original docstring follows.
---

Mass-Ratio Symmetry-Breaking Bisection
========================================

Systematically locates the mass-ratio threshold(s) where:
  (a) numerical float64 SVD rank drops below 116
  (b) algebraic rank (mpmath ground truth) diverges from 116

Three phases:
  Phase 1 — Logarithmic 1D sweep along controlled mass axes
  Phase 2 — Bisection refinement within the transition decade
  Phase 3 — (manual) Full atlas runs at the transition if warranted

Usage:
    # Phase 1: log sweep (Axis A — sweep m3)
    python mass_ratio_bisection.py sweep --axis A

    # Phase 1: log sweep (Axis B — sweep m2)
    python mass_ratio_bisection.py sweep --axis B

    # Phase 2: bisect within a decade
    python mass_ratio_bisection.py bisect --axis A --lo 0.01 --hi 0.001

    # Custom sweep
    python mass_ratio_bisection.py sweep --m1 1.0 --m2 1.0 \\
        --m3-values "1.0,0.5,0.1,0.01,1e-3,1e-4,1e-5,1e-6,1e-8,1e-10"
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.setrecursionlimit(500000)

OUTPUT_DIR = Path("bisection_results")

AXIS_A = {
    "label": "Axis A: m1=1, m2=1, sweep m3",
    "m1": 1.0,
    "m2": 1.0,
    "sweep_param": "m3",
    "values": [1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10],
}

AXIS_B = {
    "label": "Axis B: m1=1, m3=1e-8, sweep m2",
    "m1": 1.0,
    "m3": 1e-8,
    "sweep_param": "m2",
    "values": [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8],
}

N_SAMPLES = 800
LEVEL = 3
EPSILON = 1e-2


# =====================================================================
# Rank computation: float64 (standard + grouped)
# =====================================================================

def compute_rank_float64(masses, n_samples=N_SAMPLES, level=LEVEL):
    """Build algebra and compute float64 rank at equilateral point.

    Returns dict with rank, n_generators, n_expanded_cols, singular
    values summary, and timing.
    """
    from stability_atlas import AtlasConfig, PoissonAlgebra, ShapeSpace

    config = AtlasConfig(
        masses=tuple(masses),
        potential_type="1/r",
        max_level=level,
        n_phase_samples=n_samples,
        epsilon=EPSILON,
        svd_gap_threshold=1e4,
    )

    t0 = time()
    algebra = PoissonAlgebra(config)
    build_time = time() - t0

    positions = ShapeSpace.shape_to_positions(1.0, np.pi / 3)

    t1 = time()
    rank, S, info = algebra.compute_rank_at_configuration(
        positions, level=level, n_samples=n_samples, epsilon=EPSILON)
    eval_time = time() - t1

    sv_top5 = S[:5].tolist() if len(S) >= 5 else S.tolist()
    sv_bottom5 = S[-5:].tolist() if len(S) >= 5 else []

    return {
        "rank_float64": int(rank),
        "n_generators": info["n_generators"],
        "n_expanded_cols": info.get("n_expanded_cols", info["n_generators"]),
        "max_gap_ratio": float(info["max_gap_ratio"]),
        "gap_location": int(info["gap_location"]),
        "sv_top5": sv_top5,
        "sv_bottom5": sv_bottom5,
        "build_time_s": round(build_time, 1),
        "eval_time_s": round(eval_time, 1),
    }


# =====================================================================
# Rank computation: mpmath (ground truth)
# =====================================================================

def compute_rank_mpmath(masses, n_samples=200, level=LEVEL, dps=50):
    """Compute ground-truth rank using mpmath arbitrary precision.

    Uses sympy.lambdify with mpmath backend for fast vectorised
    evaluation, then determines rank via Gaussian elimination in
    mpmath.
    """
    import mpmath
    from mpmath import mpf, mp
    import sympy as sp
    mp.dps = dps

    from stability_atlas import AtlasConfig, ShapeSpace, Potential
    from exact_growth import (
        poisson_bracket, simplify_generator, ALL_VARS,
    )

    config = AtlasConfig(
        masses=tuple(masses),
        potential_type="1/r",
        max_level=level,
        n_phase_samples=n_samples,
        epsilon=EPSILON,
        svd_gap_threshold=1e4,
    )

    print(f"    [mpmath] Building algebra for masses={masses}...", flush=True)
    t0 = time()

    H12, H13, H23 = Potential.get_symbolic_hamiltonians(
        "1/r", masses=config.masses)

    all_exprs = []
    all_levels = []
    computed_pairs = set()

    for expr in [H12, H13, H23]:
        all_exprs.append(expr)
        all_levels.append(0)
    for i in range(3):
        for j in range(i + 1, 3):
            computed_pairs.add(frozenset({i, j}))

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        expr = simplify_generator(
            poisson_bracket(all_exprs[i], all_exprs[j]))
        all_exprs.append(expr)
        all_levels.append(1)

    for lv in range(2, level + 1):
        frontier = [i for i, l in enumerate(all_levels) if l == lv - 1]
        n_existing = len(all_exprs)
        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)
                if all_exprs[i] == 0 or all_exprs[j] == 0:
                    continue
                expr = simplify_generator(
                    poisson_bracket(all_exprs[i], all_exprs[j]))
                all_exprs.append(expr)
                all_levels.append(lv)

    nonzero = [(e, l) for e, l in zip(all_exprs, all_levels)
               if e != 0 and l <= level]
    exprs = [e for e, _ in nonzero]
    n_gen = len(exprs)
    build_time = time() - t0
    print(f"    [mpmath] {n_gen} generators built in {build_time:.1f}s",
          flush=True)

    # Lambdify each generator with mpmath backend
    print(f"    [mpmath] Compiling {n_gen} evaluators (mpmath backend)...",
          flush=True)
    t_lam = time()
    mp_funcs = []
    for idx, expr in enumerate(exprs):
        try:
            f = sp.lambdify(ALL_VARS, expr, modules="mpmath")
            mp_funcs.append(f)
        except Exception:
            # Fallback to xreplace for problematic expressions
            captured = expr
            var_list = list(ALL_VARS)
            def _xr(ex=captured, vs=var_list):
                def _eval(*args):
                    pt = {v: mpf(str(float(a))) for v, a in zip(vs, args)}
                    return mpf(str(complex(ex.xreplace(pt)).real))
                return _eval
            mp_funcs.append(_xr())
        if (idx + 1) % 40 == 0 or idx == n_gen - 1:
            print(f"      {idx+1}/{n_gen} [{time()-t_lam:.1f}s]", flush=True)
    print(f"    [mpmath] Lambdify done in {time()-t_lam:.1f}s", flush=True)

    # Generate sample points at equilateral config
    positions = ShapeSpace.shape_to_positions(1.0, np.pi / 3)
    rng = np.random.RandomState(123)
    base_q = positions.flatten()

    samples = []
    for _ in range(n_samples * 50):
        if len(samples) >= n_samples:
            break
        q = base_q + rng.randn(6) * EPSILON
        p = rng.randn(6) * 0.5
        dx12 = q[0] - q[2]; dy12 = q[1] - q[3]
        dx13 = q[0] - q[4]; dy13 = q[1] - q[5]
        dx23 = q[2] - q[4]; dy23 = q[3] - q[5]
        r12 = np.sqrt(dx12**2 + dy12**2)
        r13 = np.sqrt(dx13**2 + dy13**2)
        r23 = np.sqrt(dx23**2 + dy23**2)
        if min(r12, r13, r23) < 0.1:
            continue
        u12v = 1.0 / r12; u13v = 1.0 / r13; u23v = 1.0 / r23
        samples.append(list(q) + list(p) + [u12v, u13v, u23v])
    samples = samples[:n_samples]

    print(f"    [mpmath] Evaluating {n_gen} generators at {len(samples)} "
          f"points ({dps}-digit)...", flush=True)
    t1 = time()

    mat = []
    for si, sample in enumerate(samples):
        mp_args = [mpf(str(v)) for v in sample]
        row = []
        for f in mp_funcs:
            try:
                val = f(*mp_args)
                row.append(mpf(val) if not isinstance(val, mpmath.mpf) else val)
            except Exception:
                row.append(mpf(0))
        mat.append(row)
        if (si + 1) % 20 == 0:
            elapsed = time() - t1
            rate = (si + 1) / elapsed
            eta = (len(samples) - si - 1) / rate
            print(f"      {si+1}/{len(samples)} samples "
                  f"[{elapsed:.0f}s, ~{eta:.0f}s remaining]", flush=True)

    eval_time = time() - t1
    print(f"    [mpmath] Evaluation done in {eval_time:.1f}s", flush=True)

    # Gaussian elimination for rank
    print(f"    [mpmath] Computing rank via Gaussian elimination...",
          flush=True)
    t2 = time()
    M = mpmath.matrix(mat)
    n_rows, n_cols = M.rows, M.cols

    pivot_row = 0
    threshold = mpf(10) ** (-dps + 10)
    for col in range(n_cols):
        max_val = mpf(0)
        max_row = -1
        for r in range(pivot_row, n_rows):
            v = abs(M[r, col])
            if v > max_val:
                max_val = v
                max_row = r
        if max_val < threshold:
            continue
        if max_row != pivot_row:
            for c in range(n_cols):
                M[pivot_row, c], M[max_row, c] = M[max_row, c], M[pivot_row, c]
        piv = M[pivot_row, col]
        for r in range(pivot_row + 1, n_rows):
            if abs(M[r, col]) > threshold:
                factor = M[r, col] / piv
                for c in range(col, n_cols):
                    M[r, c] -= factor * M[pivot_row, c]
        pivot_row += 1

    rank_mpmath = pivot_row
    gauss_time = time() - t2
    print(f"    [mpmath] Rank = {rank_mpmath} (elimination took "
          f"{gauss_time:.1f}s)", flush=True)

    return {
        "rank_mpmath": rank_mpmath,
        "n_generators": n_gen,
        "dps": dps,
        "n_samples": len(samples),
        "build_time_s": round(build_time, 1),
        "eval_time_s": round(eval_time, 1),
        "gauss_time_s": round(gauss_time, 1),
    }


# =====================================================================
# Sweep runner
# =====================================================================

def run_sweep(axis_config, include_mpmath=False, mpmath_samples=200,
              output_dir=None):
    """Run a log-spaced sweep along one mass axis."""
    label = axis_config["label"]
    sweep_param = axis_config["sweep_param"]
    values = axis_config["values"]

    if output_dir is None:
        output_dir = OUTPUT_DIR / sweep_param
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "sweep_results.json"
    existing = []
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)

    done_values = {r["mass_value"] for r in existing}

    print(f"\n{'='*70}")
    print(f"  MASS-RATIO BISECTION SWEEP")
    print(f"  {label}")
    print(f"  {len(values)} mass values: {values}")
    print(f"  Samples/point: {N_SAMPLES}, Level: {LEVEL}")
    print(f"  mpmath: {'yes' if include_mpmath else 'no'}")
    print(f"{'='*70}\n")

    for vi, val in enumerate(values):
        if val in done_values:
            print(f"  [{vi+1}/{len(values)}] {sweep_param}={val:.2e} — "
                  f"already computed, skipping")
            continue

        masses = _build_masses(axis_config, val)
        ratio = max(masses) / min(masses)
        print(f"\n  [{vi+1}/{len(values)}] {sweep_param}={val:.2e}  "
              f"masses={masses}  ratio={ratio:.2e}")
        print(f"  {'-'*60}")

        result = {
            "mass_value": val,
            "masses": list(masses),
            "max_ratio": ratio,
            "sweep_param": sweep_param,
        }

        # Float64 rank (always)
        print(f"  Computing float64 rank...", flush=True)
        try:
            f64 = compute_rank_float64(masses)
            result.update(f64)
            print(f"    -> rank_float64 = {f64['rank_float64']}  "
                  f"(gap={f64['max_gap_ratio']:.1f}x at idx {f64['gap_location']})")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            result["rank_float64"] = None
            result["error_float64"] = str(e)

        # mpmath rank (optional)
        if include_mpmath:
            print(f"  Computing mpmath rank ({mpmath_samples} samples, "
                  f"50 digits)...", flush=True)
            try:
                mp_result = compute_rank_mpmath(masses,
                                                n_samples=mpmath_samples)
                result.update(mp_result)
                print(f"    -> rank_mpmath = {mp_result['rank_mpmath']}")
            except Exception as e:
                print(f"    -> FAILED: {e}")
                result["rank_mpmath"] = None
                result["error_mpmath"] = str(e)

        existing.append(result)
        with open(results_file, "w") as f:
            json.dump(existing, f, indent=2, default=_json_default)
        print(f"  Saved to {results_file}")

    _print_summary(existing, label)
    return existing


def run_bisect(axis_config, lo, hi, n_steps=8, include_mpmath=True,
               mpmath_samples=200, output_dir=None):
    """Bisect within a decade to find the precise transition point."""
    sweep_param = axis_config["sweep_param"]

    if output_dir is None:
        output_dir = OUTPUT_DIR / f"{sweep_param}_bisect"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lo_log = np.log10(lo)
    hi_log = np.log10(hi)
    values = np.logspace(lo_log, hi_log, n_steps).tolist()

    bisect_config = dict(axis_config)
    bisect_config["values"] = values
    bisect_config["label"] = (
        f"Bisection: {sweep_param} in [{lo:.2e}, {hi:.2e}]"
    )

    return run_sweep(bisect_config, include_mpmath=include_mpmath,
                     mpmath_samples=mpmath_samples, output_dir=output_dir)


# =====================================================================
# Helpers
# =====================================================================

def _build_masses(axis_config, sweep_value):
    """Construct (m1, m2, m3) tuple from axis config and current value."""
    sp = axis_config["sweep_param"]
    m1 = axis_config.get("m1", 1.0)
    m2 = axis_config.get("m2", 1.0)
    m3 = axis_config.get("m3", 1.0)
    if sp == "m3":
        m3 = sweep_value
    elif sp == "m2":
        m2 = sweep_value
    elif sp == "m1":
        m1 = sweep_value
    return (m1, m2, m3)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _print_summary(results, label):
    """Print a formatted table of sweep results."""
    print(f"\n{'='*70}")
    print(f"  SWEEP SUMMARY: {label}")
    print(f"{'='*70}")
    header = f"  {'mass_value':>12s}  {'ratio':>10s}  {'rank_f64':>8s}"
    has_mp = any(r.get("rank_mpmath") is not None for r in results)
    if has_mp:
        header += f"  {'rank_mp':>8s}"
    header += f"  {'gap':>8s}  {'n_exp':>6s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for r in sorted(results, key=lambda x: -x["mass_value"]):
        line = (f"  {r['mass_value']:>12.2e}  "
                f"{r['max_ratio']:>10.2e}  "
                f"{str(r.get('rank_float64', '?')):>8s}")
        if has_mp:
            line += f"  {str(r.get('rank_mpmath', '—')):>8s}"
        line += (f"  {r.get('max_gap_ratio', 0):>8.1f}  "
                 f"{r.get('n_expanded_cols', '?'):>6}")
        print(line)
    print()


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mass-ratio symmetry-breaking bisection scan")
    sub = parser.add_subparsers(dest="command")

    # --- sweep ---
    p_sweep = sub.add_parser("sweep", help="Run a log-spaced sweep")
    p_sweep.add_argument("--axis", choices=["A", "B"], default="A",
                         help="Predefined axis (A=sweep m3, B=sweep m2)")
    p_sweep.add_argument("--m1", type=float, default=None)
    p_sweep.add_argument("--m2", type=float, default=None)
    p_sweep.add_argument("--m3", type=float, default=None)
    p_sweep.add_argument("--m3-values", type=str, default=None,
                         help="Comma-separated sweep values for m3")
    p_sweep.add_argument("--m2-values", type=str, default=None,
                         help="Comma-separated sweep values for m2")
    p_sweep.add_argument("--mpmath", action="store_true",
                         help="Include mpmath ground-truth evaluation")
    p_sweep.add_argument("--mpmath-samples", type=int, default=200)
    p_sweep.add_argument("--output-dir", type=str, default=None)

    # --- bisect ---
    p_bisect = sub.add_parser("bisect",
                              help="Bisect within a transition decade")
    p_bisect.add_argument("--axis", choices=["A", "B"], default="A")
    p_bisect.add_argument("--lo", type=float, required=True,
                          help="Upper bound of transition decade")
    p_bisect.add_argument("--hi", type=float, required=True,
                          help="Lower bound of transition decade")
    p_bisect.add_argument("--n-steps", type=int, default=8)
    p_bisect.add_argument("--mpmath", action="store_true", default=True)
    p_bisect.add_argument("--no-mpmath", action="store_true")
    p_bisect.add_argument("--mpmath-samples", type=int, default=200)
    p_bisect.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "sweep":
        if args.axis == "A":
            axis_config = dict(AXIS_A)
        else:
            axis_config = dict(AXIS_B)

        if args.m1 is not None:
            axis_config["m1"] = args.m1
        if args.m2 is not None:
            axis_config["m2"] = args.m2
        if args.m3 is not None:
            axis_config["m3"] = args.m3
        if args.m3_values:
            axis_config["values"] = [float(x) for x in
                                     args.m3_values.split(",")]
            axis_config["sweep_param"] = "m3"
        if args.m2_values:
            axis_config["values"] = [float(x) for x in
                                     args.m2_values.split(",")]
            axis_config["sweep_param"] = "m2"

        run_sweep(axis_config, include_mpmath=args.mpmath,
                  mpmath_samples=args.mpmath_samples,
                  output_dir=args.output_dir)

    elif args.command == "bisect":
        axis_config = AXIS_A if args.axis == "A" else AXIS_B
        use_mpmath = args.mpmath and not args.no_mpmath
        run_bisect(dict(axis_config), lo=args.lo, hi=args.hi,
                   n_steps=args.n_steps,
                   include_mpmath=use_mpmath,
                   mpmath_samples=args.mpmath_samples,
                   output_dir=args.output_dir)


if __name__ == "__main__":
    main()
