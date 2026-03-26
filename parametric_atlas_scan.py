#!/usr/bin/env python3
"""
Parametric atlas scanner: build the Poisson algebra ONCE with n as a
SymPy Symbol, then sweep arbitrary 1/r^n exponents at evaluation time.

This avoids rebuilding the algebra for each exponent value, and the
expressions with symbolic n lambdify cleanly (no RecursionError).

Usage:
    python parametric_atlas_scan.py --exponents 2 --resolution 100
    python parametric_atlas_scan.py --exponents 1 2 3 --resolution 50
    python parametric_atlas_scan.py --exponent-range 0.5 5.0 --step 0.5
"""

import argparse
import json
import os
import sys
import signal
import numpy as np
from time import time, strftime

sys.setrecursionlimit(100000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
from exact_growth import (
    Q_VARS, P_VARS, U_VARS,
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    poisson_bracket, simplify_generator,
)
from stability_atlas import AtlasConfig, ShapeSpace

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[SIGTERM] Graceful shutdown requested — finishing current row...")


n_sym = sp.Symbol('n', real=True)

PARA_VARS = Q_VARS + P_VARS + U_VARS + [n_sym]


def build_parametric_algebra(masses=(1.0, 1.0, 1.0), charges=None,
                             max_level=3):
    """Build Poisson algebra with symbolic exponent n."""

    T1 = (px1**2 + py1**2) / 2
    T2 = (px2**2 + py2**2) / 2
    T3 = (px3**2 + py3**2) / 2

    KE1, KE2, KE3 = T1, T2, T3
    if masses is not None and tuple(masses) != (1.0, 1.0, 1.0):
        from sympy import nsimplify as _ns, Integer as _Int
        def _to_sym(v):
            return _Int(v) if isinstance(v, int) else _ns(v, rational=True)
        m1, m2, m3 = _to_sym(masses[0]), _to_sym(masses[1]), _to_sym(masses[2])
        KE1 = (px1**2 + py1**2) / (2 * m1)
        KE2 = (px2**2 + py2**2) / (2 * m2)
        KE3 = (px3**2 + py3**2) / (2 * m3)

    u12_n = u12**n_sym
    u13_n = u13**n_sym
    u23_n = u23**n_sym

    q1 = q2 = q3 = None
    if charges is not None:
        q1, q2, q3 = [sp.Integer(c) if isinstance(c, int) else c
                      for c in charges]

    def _pair(u_n, qi, qj):
        if qi is not None:
            return qi * qj * u_n
        return -u_n

    H12 = KE1 + KE2 + _pair(u12_n, q1, q2)
    H13 = KE1 + KE3 + _pair(u13_n, q1, q3)
    H23 = KE2 + KE3 + _pair(u23_n, q2, q3)

    all_exprs = []
    all_names = []
    all_levels = []
    computed_pairs = set()

    for name, expr in [("H12", H12), ("H13", H13), ("H23", H23)]:
        all_exprs.append(expr)
        all_names.append(name)
        all_levels.append(0)
    for i in range(3):
        for j in range(i + 1, 3):
            computed_pairs.add(frozenset({i, j}))

    for short, full, i, j in [("K1", "{H12,H13}", 0, 1),
                               ("K2", "{H12,H23}", 0, 2),
                               ("K3", "{H13,H23}", 1, 2)]:
        expr = simplify_generator(poisson_bracket(all_exprs[i], all_exprs[j]))
        all_exprs.append(expr)
        all_names.append(short)
        all_levels.append(1)

    for level in range(2, max_level + 1):
        frontier = [i for i, lv in enumerate(all_levels) if lv == level - 1]
        n_existing = len(all_exprs)
        new_exprs = []
        new_names = []

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
                new_exprs.append(expr)
                new_names.append(f"{{{all_names[i]},{all_names[j]}}}")

        for expr, name in zip(new_exprs, new_names):
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(level)

        n_zero = sum(1 for e in new_exprs if e == 0)
        print(f"    Level {level}: {len(new_exprs)} generators ({n_zero} zero)")

    nonzero_mask = [i for i, e in enumerate(all_exprs) if e != 0]
    exprs = [all_exprs[i] for i in nonzero_mask]
    levels = [all_levels[i] for i in nonzero_mask]
    names = [all_names[i] for i in nonzero_mask]

    print(f"    Total: {len(all_exprs)} generators, {len(exprs)} non-zero")
    return exprs, levels, names


def lambdify_parametric(exprs):
    """Lambdify with n as extra (16th) variable. Returns evaluate(Z_qp, Z_u, n_val)."""
    n = len(exprs)
    t0 = time()

    if n <= 50:
        print(f"    Lambdifying {n} parametric expressions (standard)...",
              end=" ", flush=True)
        func = sp.lambdify(PARA_VARS, exprs, modules="numpy", cse=True)
        print(f"done [{time() - t0:.1f}s]")

        def evaluate(Z_qp, Z_u, n_val):
            args = ([Z_qp[:, i] for i in range(12)] +
                    [Z_u[:, i] for i in range(3)] +
                    [np.full(Z_qp.shape[0], n_val)])
            vals = func(*args)
            return np.column_stack(vals)

        return evaluate

    print(f"    Lambdifying {n} parametric expressions individually...",
          flush=True)

    funcs = []
    n_ok = 0
    n_flat = 0
    n_xr = 0

    for idx, expr in enumerate(exprs):
        if (idx + 1) % 20 == 0 or idx == n - 1:
            print(f"      {idx+1}/{n}  [{time()-t0:.1f}s]  "
                  f"(lambdify:{n_ok} flat:{n_flat} xreplace:{n_xr})",
                  flush=True)

        try:
            f = sp.lambdify(PARA_VARS, expr, modules="numpy", cse=False)
            funcs.append(('l', f))
            n_ok += 1
            continue
        except (RecursionError, Exception):
            pass

        try:
            f = _make_flat_parametric(expr, label=f"g{idx}")
            funcs.append(('f', f))
            n_flat += 1
            continue
        except (RecursionError, Exception) as e:
            print(f"      [{idx}] flat-CSE also failed ({e.__class__.__name__}), "
                  f"falling back to xreplace", flush=True)

        var_syms = list(PARA_VARS)
        captured_expr = expr

        def _make_xr(ex, vs):
            def _subs(*args):
                n_pts = len(args[0]) if hasattr(args[0], '__len__') else 1
                result = np.empty(n_pts)
                for i in range(n_pts):
                    pt = {v: (float(a[i]) if hasattr(a, '__len__') else float(a))
                          for v, a in zip(vs, args)}
                    try:
                        result[i] = complex(ex.xreplace(pt)).real
                    except Exception:
                        result[i] = 0.0
                return result
            return _subs
        funcs.append(('x', _make_xr(captured_expr, var_syms)))
        n_xr += 1

    print(f"    Lambdify results: {n_ok} lambdify, {n_flat} flat-CSE, "
          f"{n_xr} xreplace")
    print(f"    Total lambdify time: {time() - t0:.1f}s")

    def evaluate(Z_qp, Z_u, n_val):
        args = ([Z_qp[:, i] for i in range(12)] +
                [Z_u[:, i] for i in range(3)] +
                [np.full(Z_qp.shape[0], n_val)])
        n_pts = Z_qp.shape[0]
        cols = []
        t_eval = time()
        for idx, (layer, f) in enumerate(funcs):
            if layer == 'x':
                print(f"      WARNING: eval {idx+1}/{len(funcs)} using "
                      f"xreplace (slow!)", flush=True)
            val = f(*args)
            arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
            if arr.shape[0] == 1:
                arr = np.full(n_pts, arr[0])
            elif arr.shape[0] < n_pts:
                arr = np.resize(arr, n_pts)
            cols.append(arr[:n_pts])
            if (idx + 1) % 20 == 0:
                print(f"      eval {idx+1}/{len(funcs)}  [{time()-t_eval:.1f}s]",
                      flush=True)
        return np.column_stack(cols)

    return evaluate


def _make_flat_parametric(expr, label="_f"):
    """Flat-CSE fallback with n as extra parameter."""
    import tempfile
    import importlib.util
    from sympy import cse as _cse, pycode

    replacements, (reduced,) = _cse(expr, optimizations='basic')

    var_names = [str(v) for v in PARA_VARS]
    sig = ", ".join(var_names)
    lines = [
        "import numpy as _np",
        "from numpy import exp, log, sqrt, sin, cos, abs, power",
        f"def {label}({sig}):",
    ]
    for sym, sub in replacements:
        lines.extend(_flat_chunk(sub, str(sym)))
    lines.extend(_flat_chunk(reduced, "_result"))
    lines.append("    return _result")
    code = "\n".join(lines)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="flat_para_")
    tmp.write(code)
    tmp.flush()
    tmp.close()

    spec = importlib.util.spec_from_file_location(f"_flat_{label}", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, label)


def _flat_chunk(expr, target_var, indent="    ", max_terms=50):
    from sympy import pycode, Add
    terms = Add.make_args(expr)
    if len(terms) <= max_terms:
        return [f"{indent}{target_var} = {pycode(expr)}"]
    lines = [f"{indent}{target_var} = 0"]
    for i in range(0, len(terms), max_terms):
        chunk = Add(*terms[i:i + max_terms])
        lines.append(f"{indent}{target_var} += {pycode(chunk)}")
    return lines


# =====================================================================
# Sampling (identical to PoissonAlgebra._sample_local)
# =====================================================================

def sample_local(positions, n_samples, epsilon, masses=(1.0, 1.0, 1.0),
                 mom_range=0.5, min_sep=0.1, seed=None):
    rng = np.random.RandomState(seed)
    base_q = positions.flatten()

    dx12 = base_q[0] - base_q[2]; dy12 = base_q[1] - base_q[3]
    dx13 = base_q[0] - base_q[4]; dy13 = base_q[1] - base_q[5]
    dx23 = base_q[2] - base_q[4]; dy23 = base_q[3] - base_q[5]
    base_r_min = min(
        np.sqrt(dx12**2 + dy12**2),
        np.sqrt(dx13**2 + dy13**2),
        np.sqrt(dx23**2 + dy23**2),
    )
    effective_min_sep = min(min_sep, 0.3 * base_r_min)

    mom_scale = np.array([
        np.sqrt(masses[0]), np.sqrt(masses[0]),
        np.sqrt(masses[1]), np.sqrt(masses[1]),
        np.sqrt(masses[2]), np.sqrt(masses[2]),
    ])

    Z_qp = np.zeros((n_samples, 12))
    Z_u = np.zeros((n_samples, 3))
    accepted = 0

    for _ in range(n_samples * 200):
        if accepted >= n_samples:
            break
        q = base_q + rng.randn(6) * epsilon
        p = rng.randn(6) * mom_range * mom_scale

        ddx12 = q[0] - q[2]; ddy12 = q[1] - q[3]
        ddx13 = q[0] - q[4]; ddy13 = q[1] - q[5]
        ddx23 = q[2] - q[4]; ddy23 = q[3] - q[5]

        r12 = np.sqrt(ddx12**2 + ddy12**2)
        r13 = np.sqrt(ddx13**2 + ddy13**2)
        r23 = np.sqrt(ddx23**2 + ddy23**2)

        if min(r12, r13, r23) < effective_min_sep:
            continue

        Z_qp[accepted, :6] = q
        Z_qp[accepted, 6:] = p
        Z_u[accepted] = [1.0 / r12, 1.0 / r13, 1.0 / r23]
        accepted += 1

    return Z_qp[:accepted], Z_u[:accepted]


# =====================================================================
# SVD analysis (from stability_atlas)
# =====================================================================

def rank_from_gap(S):
    if len(S) <= 1:
        return len(S), 1.0
    noise_threshold = 1e-8 * S[0]
    n_meaningful = int(np.sum(S > noise_threshold))
    best_gap = 1.0
    best_idx = -1
    for i in range(min(n_meaningful, len(S) - 1)):
        if S[i + 1] > noise_threshold:
            gap = S[i] / S[i + 1]
        else:
            gap = S[i] / noise_threshold
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    rank = best_idx + 1 if best_idx >= 0 else n_meaningful
    return rank, best_gap


# =====================================================================
# Main scan loop
# =====================================================================

def save_checkpoint_atomic(out_dir, last_row, n_rows):
    cp_file = os.path.join(out_dir, "checkpoint.json")
    tmp_file = cp_file + ".tmp"
    data = {
        "last_completed_row": last_row,
        "total_rows": n_rows,
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(tmp_file, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, cp_file)


def flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map):
    np.save(os.path.join(out_dir, "mu_vals.npy"), mu_vals)
    np.save(os.path.join(out_dir, "phi_vals.npy"), phi_vals)
    np.save(os.path.join(out_dir, "rank_map.npy"), rank_map)
    np.save(os.path.join(out_dir, "gap_map.npy"), gap_map)


def run_parametric_scan(evaluate, levels, n_val, grid_n, n_samples,
                        masses, out_dir, label):
    global _shutdown_requested
    os.makedirs(out_dir, exist_ok=True)

    n_gen = len(levels)
    mu_range = (0.2, 3.0)
    phi_range = (0.1, np.pi - 0.1)
    mu_vals = np.linspace(mu_range[0], mu_range[1], grid_n)
    phi_vals = np.linspace(phi_range[0], phi_range[1], grid_n)

    rank_map = np.full((grid_n, grid_n), -1, dtype=np.int32)
    gap_map = np.zeros((grid_n, grid_n), dtype=np.float64)

    start_row = 0
    cp_file = os.path.join(out_dir, "checkpoint.json")
    if os.path.exists(cp_file):
        with open(cp_file) as f:
            cp = json.load(f)
        if cp.get("total_rows") == grid_n:
            start_row = cp["last_completed_row"] + 1
            if os.path.exists(os.path.join(out_dir, "rank_map.npy")):
                rank_map = np.load(os.path.join(out_dir, "rank_map.npy"))
                gap_map = np.load(os.path.join(out_dir, "gap_map.npy"))
            print(f"  Resuming from row {start_row}/{grid_n}")
        else:
            print(f"  Grid size mismatch in checkpoint, starting fresh")

    cfg = {
        "potential": f"1/r^{n_val}",
        "exponent": n_val,
        "masses": list(masses),
        "charges": None,
        "level": max(levels),
        "grid_n": grid_n,
        "n_phase_samples": n_samples,
        "adaptive": False,
        "mu_range": list(mu_range),
        "phi_range": list(phi_range),
        "label": label,
        "parametric": True,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  PARAMETRIC ATLAS SCAN: {label}")
    print(f"  Grid: {grid_n}x{grid_n}  |  Samples: {n_samples}")
    print(f"  Masses: {masses}  |  Exponent n={n_val}")
    print(f"  Generators: {n_gen}  |  Output: {out_dir}")
    print(f"{'='*70}\n")

    t_total = time()
    shape = ShapeSpace()

    for i in range(start_row, grid_n):
        if _shutdown_requested:
            print(f"  [SIGTERM] Stopping after row {i-1}")
            break

        t_row = time()
        mu = mu_vals[i]

        for j in range(grid_n):
            phi = phi_vals[j]
            positions = shape.shape_to_positions(mu, phi)

            p = positions.reshape(3, 2)
            r12 = np.linalg.norm(p[0] - p[1])
            r13 = np.linalg.norm(p[0] - p[2])
            r23 = np.linalg.norm(p[1] - p[2])
            r_min = min(r12, r13, r23)
            eps = min(1e-2, 0.1 * r_min)

            Z_qp, Z_u = sample_local(positions, n_samples, eps, masses)

            full_matrix = evaluate(Z_qp, Z_u, n_val)

            norms = np.linalg.norm(full_matrix, axis=0)
            norms[norms < 1e-15] = 1.0
            sub = full_matrix / norms

            from numpy.linalg import svd
            U, S, Vt = svd(sub, full_matrices=False)

            rank, gap = rank_from_gap(S)
            rank_map[i, j] = rank
            gap_map[i, j] = gap

        flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map)
        save_checkpoint_atomic(out_dir, i, grid_n)

        elapsed = time() - t_row
        pct = 100.0 * (i + 1) / grid_n
        r116 = np.sum(rank_map[i] == 116) if rank_map[i].max() > 0 else 0
        print(f"  Row {i+1:3d}/{grid_n}  mu={mu:.3f}  "
              f"[{elapsed:.1f}s]  {pct:.0f}%  "
              f"rank-116: {r116}/{grid_n}", flush=True)

    total_time = time() - t_total
    n_valid = np.sum(rank_map >= 0)
    n_116 = np.sum(rank_map == 116)
    summary = {
        "label": label,
        "total_points": int(grid_n * grid_n),
        "valid_points": int(n_valid),
        "rank_116_count": int(n_116),
        "rank_116_fraction": float(n_116 / n_valid) if n_valid > 0 else 0.0,
        "unique_ranks": sorted(set(int(r) for r in rank_map.ravel() if r >= 0)),
        "elapsed_seconds": total_time,
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map)

    print(f"\n  === DONE: {label} ===")
    print(f"  rank=116: {n_116}/{n_valid} "
          f"({100*n_116/n_valid:.1f}%)" if n_valid > 0 else "  No valid points")
    print(f"  Unique ranks: {summary['unique_ranks']}")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f}min)\n")
    return summary


def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)

    parser = argparse.ArgumentParser(
        description="Parametric 1/r^n atlas scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--exponents", nargs="+", type=float, default=None,
                        help="Exponent values to scan (e.g. 1 2 3)")
    parser.add_argument("--exponent-range", nargs=2, type=float, default=None,
                        metavar=("START", "END"),
                        help="Range of exponents (inclusive)")
    parser.add_argument("--step", type=float, default=0.01,
                        help="Step size for exponent range (default: 0.01)")
    parser.add_argument("--resolution", type=int, default=100,
                        help="Grid resolution NxN (default: 100)")
    parser.add_argument("--samples", type=int, default=400,
                        help="Phase-space samples per grid point (default: 400)")
    parser.add_argument("--level", type=int, default=3,
                        help="Max bracket level (default: 3)")
    parser.add_argument("--masses", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                        metavar=("M1", "M2", "M3"))
    parser.add_argument("--output-dir", type=str, default="aws_results/atlas_full",
                        help="Base output directory")

    args = parser.parse_args()

    exponents = []
    if args.exponents:
        exponents = args.exponents
    elif args.exponent_range:
        lo, hi = args.exponent_range
        exponents = list(np.arange(lo, hi + args.step / 2, args.step))
    else:
        parser.error("Specify --exponents or --exponent-range")

    masses = tuple(args.masses)

    print("=" * 70)
    print(f"  PARAMETRIC ATLAS: {len(exponents)} exponent(s)")
    print(f"  n = {exponents[:5]}{'...' if len(exponents) > 5 else ''}")
    print(f"  Grid: {args.resolution}x{args.resolution}  |  "
          f"Samples: {args.samples}  |  Level: {args.level}")
    print("=" * 70)

    print("\n  Building parametric algebra (n = Symbol)...")
    t_build = time()
    exprs, levels, names = build_parametric_algebra(
        masses=masses, max_level=args.level)
    print(f"  Compiling parametric evaluator...")
    evaluate = lambdify_parametric(exprs)
    build_time = time() - t_build
    print(f"  Algebra + evaluator ready ({build_time:.1f}s, "
          f"{len(exprs)} generators)\n")

    for idx, n_val in enumerate(exponents):
        if _shutdown_requested:
            print(f"\n  [SIGTERM] Stopped before exponent n={n_val}")
            break

        tag = f"1r{n_val:g}".replace("-", "-").replace(".", "p")
        out_dir = os.path.join(args.output_dir, tag)
        label = f"1/r^{n_val:g}"

        print(f"\n  [{idx+1}/{len(exponents)}] Scanning n = {n_val}")
        run_parametric_scan(
            evaluate, levels, n_val, args.resolution, args.samples,
            masses, out_dir, label)

    print("\n  All exponents complete.")


if __name__ == "__main__":
    main()
