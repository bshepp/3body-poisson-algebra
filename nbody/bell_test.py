#!/usr/bin/env python3
"""
Bell Test for the Poisson Algebra
=================================

Tests whether the nonlocal algebraic structure of the N=3 Poisson bracket
algebra can produce Bell-inequality-violating correlations when projected
onto local single-body observables.

Three-part investigation:
  A) N=2 baseline — shows the 2-body algebra is trivial
  B) Projection statistics — gradient-based body-locality, conditional
     distributions, mutual information
  C) CHSH computation — three measurement variants with bootstrap CIs

Usage:
    python nbody/bell_test.py                  # run everything
    python nbody/bell_test.py --part A         # just the N=2 baseline
    python nbody/bell_test.py --part B         # just projection statistics
    python nbody/bell_test.py --part C         # just CHSH
    python nbody/bell_test.py --n-samples 50000  # adjust sample count
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
from time import time
from itertools import product

sys.setrecursionlimit(100000)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "bell_test_results")
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "level_3.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# Part A: N=2 Baseline
# =====================================================================

def run_part_a():
    """Run N=2 Poisson algebra to show it closes at dimension 1."""
    print("\n" + "=" * 70)
    print("PART A: N=2 POISSON ALGEBRA BASELINE")
    print("=" * 70)

    sys.path.insert(0, SCRIPT_DIR)
    from exact_growth_nbody import NBodyAlgebra

    alg = NBodyAlgebra(n_bodies=2, d_spatial=2, potential="1/r",
                       checkpoint_dir=os.path.join(RESULTS_DIR, "n2_ckpt"))
    alg.compute_growth(max_level=5, n_samples=100, seed=42)

    summary = (
        "N=2 Poisson Algebra Baseline\n"
        "============================\n"
        "For N=2 (one pairwise interaction H12), the Poisson bracket\n"
        "{H12, H12} = 0 trivially. The algebra closes at dimension 1.\n"
        "All algebraic structure beyond the Hamiltonian is intrinsic to N >= 3.\n"
        "This is expected: the 2-body Kepler problem is integrable.\n"
    )
    with open(os.path.join(RESULTS_DIR, "n2_algebra_summary.txt"), "w") as f:
        f.write(summary)
    print(f"\n  Summary saved to {RESULTS_DIR}/n2_algebra_summary.txt")
    return summary


# =====================================================================
# Stratified Phase-Space Sampling
# =====================================================================

def sample_stratum(n, stratum, seed=42, mom_range=1.0):
    """
    Sample phase-space points with stratum-specific distance constraints.

    Strata:
      'equilateral' — all r_ij in [1.0, 2.0]
      'pair_apparatus' — r12 in [0.5, 1.5], r13/r23 in [4.0, 8.0]
      'separated' — all r_ij in [5.0, 10.0]
    """
    rng = np.random.RandomState(seed)

    if stratum == "equilateral":
        r_bounds = {"r12": (1.0, 2.0), "r13": (1.0, 2.0), "r23": (1.0, 2.0)}
        pos_range = 3.0
    elif stratum == "pair_apparatus":
        r_bounds = {"r12": (0.5, 1.5), "r13": (4.0, 8.0), "r23": (4.0, 8.0)}
        pos_range = 10.0
    elif stratum == "separated":
        r_bounds = {"r12": (5.0, 10.0), "r13": (5.0, 10.0), "r23": (5.0, 10.0)}
        pos_range = 15.0
    else:
        raise ValueError(f"Unknown stratum: {stratum}")

    pts = np.empty((0, 12))
    max_iters = 2000
    for _ in range(max_iters):
        bs = max((n - pts.shape[0]) * 20, 1024)
        b = np.zeros((bs, 12))
        b[:, :6] = rng.uniform(-pos_range, pos_range, (bs, 6))
        b[:, 6:] = rng.uniform(-mom_range, mom_range, (bs, 6))

        dx12 = b[:, 0] - b[:, 2]; dy12 = b[:, 1] - b[:, 3]
        dx13 = b[:, 0] - b[:, 4]; dy13 = b[:, 1] - b[:, 5]
        dx23 = b[:, 2] - b[:, 4]; dy23 = b[:, 3] - b[:, 5]

        r12 = np.sqrt(dx12**2 + dy12**2)
        r13 = np.sqrt(dx13**2 + dy13**2)
        r23 = np.sqrt(dx23**2 + dy23**2)

        ok = (
            (r12 >= r_bounds["r12"][0]) & (r12 <= r_bounds["r12"][1]) &
            (r13 >= r_bounds["r13"][0]) & (r13 <= r_bounds["r13"][1]) &
            (r23 >= r_bounds["r23"][0]) & (r23 <= r_bounds["r23"][1])
        )
        pts = np.vstack([pts, b[ok]])
        if pts.shape[0] >= n:
            break

    pts = pts[:n]
    if pts.shape[0] < n:
        print(f"  WARNING: only got {pts.shape[0]}/{n} points for "
              f"stratum '{stratum}'")

    dx12 = pts[:, 0] - pts[:, 2]; dy12 = pts[:, 1] - pts[:, 3]
    dx13 = pts[:, 0] - pts[:, 4]; dy13 = pts[:, 1] - pts[:, 5]
    dx23 = pts[:, 2] - pts[:, 4]; dy23 = pts[:, 3] - pts[:, 5]
    u12 = 1.0 / np.sqrt(dx12**2 + dy12**2)
    u13 = 1.0 / np.sqrt(dx13**2 + dy13**2)
    u23 = 1.0 / np.sqrt(dx23**2 + dy23**2)
    Z_u = np.column_stack([u12, u13, u23])

    return pts, Z_u


def sample_all_strata(n_per_stratum, seed=42):
    """Sample all three strata, return dict of (Z_qp, Z_u) per stratum."""
    strata = {}
    for i, name in enumerate(["equilateral", "pair_apparatus", "separated"]):
        print(f"  Sampling stratum '{name}' ({n_per_stratum} points)...",
              end=" ", flush=True)
        t0 = time()
        Z_qp, Z_u = sample_stratum(n_per_stratum, name, seed=seed + i * 1000)
        print(f"got {Z_qp.shape[0]} in {time()-t0:.1f}s")
        strata[name] = (Z_qp, Z_u)
    return strata


# =====================================================================
# Generator Evaluation
# =====================================================================

def _expr_to_chunked_lines(expr, target_var, indent="    ",
                           max_terms_per_line=50):
    """Break large additions into chunked += lines to avoid deep AST."""
    import sympy as sp
    terms = sp.Add.make_args(expr)
    if len(terms) <= max_terms_per_line:
        return [f"{indent}{target_var} = {sp.pycode(expr)}"]
    lines = [f"{indent}{target_var} = 0"]
    for i in range(0, len(terms), max_terms_per_line):
        chunk_expr = sp.Add(*terms[i:i + max_terms_per_line])
        lines.append(
            f"{indent}{target_var} += {sp.pycode(chunk_expr)}")
    return lines


def _make_flat_numpy_func(expr, all_vars, label="_f"):
    """
    Layer 2 fallback: CSE-flatten the expression, write chunked Python
    to a temp .py file, and import it.  Bypasses compile() recursion.
    Returns a numpy-vectorised callable, or raises on failure.
    """
    import tempfile
    import importlib.util
    import sympy as sp

    replacements, (reduced,) = sp.cse(expr, optimizations='basic')

    var_names = [str(v) for v in all_vars]
    sig = ", ".join(var_names)
    lines = [
        "import numpy as _np",
        "from numpy import exp, log, sqrt, sin, cos, abs, power",
        f"def {label}({sig}):",
    ]

    for sym, sub in replacements:
        lines.extend(_expr_to_chunked_lines(sub, str(sym)))
    lines.extend(_expr_to_chunked_lines(reduced, "_result"))
    lines.append("    return _result")
    code = "\n".join(lines)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"bell_flat_{label}_")
    tmp.write(code)
    tmp.flush()
    tmp.close()

    spec = importlib.util.spec_from_file_location(f"_flat_{label}", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, label)


def _make_xreplace_func(expr, all_vars):
    """
    Layer 3 fallback: point-by-point symbolic substitution via xreplace.
    Slow but handles arbitrarily nested expressions.
    """
    var_syms = list(all_vars)

    def _subs_eval(*args):
        subs = dict(zip(var_syms, args))
        n_pts = len(args[0]) if hasattr(args[0], '__len__') else 1
        result = np.empty(n_pts)
        for i in range(n_pts):
            pt = {k: (float(v[i]) if hasattr(v, '__len__') else float(v))
                  for k, v in subs.items()}
            try:
                result[i] = complex(expr.xreplace(pt)).real
            except Exception:
                result[i] = 0.0
        return result

    return _subs_eval


LAYER_LAMBDIFY = 0
LAYER_FLAT_CSE = 1
LAYER_XREPLACE = 2


def _robust_lambdify(exprs, all_vars):
    """
    Three-layer lambdification pipeline:
      1) sp.lambdify  (fast numpy vectorised)
      2) CSE + chunked temp-file import  (still vectorised, handles deep AST)
      3) xreplace point-by-point  (slow — used only for small sample sizes)

    Returns (evaluate_func, usable_mask) where usable_mask[i] is True
    for generators that can be evaluated in vectorised mode (layers 1-2).
    """
    import sympy as sp

    n = len(exprs)
    t0 = time()
    funcs = []
    layers = []
    counts = [0, 0, 0]

    for idx, expr in enumerate(exprs):
        if (idx + 1) % 20 == 0 or idx == n - 1:
            print(f"      {idx+1}/{n}  [{time()-t0:.1f}s]  "
                  f"(lambdify:{counts[0]} flat:{counts[1]} "
                  f"xreplace:{counts[2]})", flush=True)

        # Layer 1: standard lambdify
        try:
            f = sp.lambdify(all_vars, expr, modules="numpy", cse=False)
            funcs.append(f)
            layers.append(LAYER_LAMBDIFY)
            counts[0] += 1
            continue
        except (RecursionError, Exception):
            pass

        # Layer 2: CSE + chunked temp-file
        try:
            f = _make_flat_numpy_func(expr, all_vars, label=f"g{idx}")
            funcs.append(f)
            layers.append(LAYER_FLAT_CSE)
            counts[1] += 1
            continue
        except (RecursionError, Exception):
            pass

        # Layer 3: xreplace point-by-point
        f = _make_xreplace_func(expr, all_vars)
        funcs.append(f)
        layers.append(LAYER_XREPLACE)
        counts[2] += 1

    usable = [layer != LAYER_XREPLACE for layer in layers]
    n_usable = sum(usable)

    print(f"    Lambdify results: {counts[0]} lambdify, "
          f"{counts[1]} flat-CSE, {counts[2]} xreplace")
    print(f"    Vectorised (usable for large samples): {n_usable}/{n}")
    print(f"    Total lambdify time: {time() - t0:.1f}s")

    def evaluate(Z_qp, Z_u, skip_xreplace=True):
        args = ([Z_qp[:, i] for i in range(12)] +
                [Z_u[:, i] for i in range(3)])
        n_pts = Z_qp.shape[0]
        cols = []
        t_eval = time()
        n_skipped = 0
        for idx, (f, layer) in enumerate(zip(funcs, layers)):
            if skip_xreplace and layer == LAYER_XREPLACE:
                cols.append(np.full(n_pts, np.nan))
                n_skipped += 1
                continue
            val = f(*args)
            arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
            if arr.shape[0] == 1:
                arr = np.full(n_pts, arr[0])
            elif arr.shape[0] < n_pts:
                arr = np.resize(arr, n_pts)
            cols.append(arr[:n_pts])
            if (idx + 1) % 40 == 0:
                print(f"      eval {idx+1}/{n}  "
                      f"[{time()-t_eval:.1f}s]", flush=True)
        if n_skipped:
            print(f"      ({n_skipped} xreplace generators skipped, "
                  f"filled with NaN)", flush=True)
        return np.column_stack(cols)

    return evaluate, usable


def load_and_lambdify():
    """Load Level 3 checkpoint and lambdify all generators."""
    sys.path.insert(0, PROJECT_DIR)
    from exact_growth import ALL_VARS

    print(f"  Loading checkpoint from {CHECKPOINT_PATH}...")
    with open(CHECKPOINT_PATH, "rb") as f:
        ckpt = pickle.load(f)

    exprs = ckpt["exprs"]
    names = ckpt["names"]
    levels = ckpt["levels"]
    print(f"  Loaded {len(exprs)} generators "
          f"(levels 0-{max(levels)})")

    print("  Lambdifying generators (robust CSE pipeline)...")
    evaluate, usable = _robust_lambdify(exprs, ALL_VARS)

    return exprs, names, levels, evaluate, ALL_VARS, usable


def evaluate_on_strata(evaluate, strata):
    """Evaluate all generators at all sampled points per stratum."""
    evals = {}
    for name, (Z_qp, Z_u) in strata.items():
        print(f"  Evaluating on '{name}' ({Z_qp.shape[0]} points)...",
              end=" ", flush=True)
        t0 = time()
        M = evaluate(Z_qp, Z_u)
        print(f"done [{time()-t0:.1f}s], shape {M.shape}")
        evals[name] = M
    return evals


# =====================================================================
# Part B: Projection Statistics
# =====================================================================

def gradient_locality_scores(evaluate, strata, n_gens, usable,
                             n_vars=15):
    """
    Compute gradient-based body-locality scores via finite differences.

    For each generator, compute |dg/dvar|^2 grouped by body,
    averaged over sampled points.  Generators marked non-usable
    (xreplace fallback) get uniform 1/3 scores.

    Returns (n_gens, 3) array of locality scores per body.
    """
    h = 1e-7
    body_var_indices = {
        1: [0, 1, 6, 7],      # x1, y1, px1, py1
        2: [2, 3, 8, 9],      # x2, y2, px2, py2
        3: [4, 5, 10, 11],    # x3, y3, px3, py3
    }

    scores = np.full((n_gens, 3), 1.0 / 3.0)
    n_total = 0

    for stratum_name, (Z_qp, Z_u) in strata.items():
        n_pts = Z_qp.shape[0]
        n_total += n_pts
        Z_full = np.hstack([Z_qp, Z_u])

        base_vals = evaluate(Z_qp, Z_u)  # (n_pts, n_gens)

        body_grad_sq = np.zeros((n_gens, 3))

        for var_idx in range(12):
            Z_plus = Z_full.copy()
            Z_plus[:, var_idx] += h

            Z_qp_p = Z_plus[:, :12]
            if var_idx < 6:
                dx12 = Z_qp_p[:, 0] - Z_qp_p[:, 2]
                dy12 = Z_qp_p[:, 1] - Z_qp_p[:, 3]
                dx13 = Z_qp_p[:, 0] - Z_qp_p[:, 4]
                dy13 = Z_qp_p[:, 1] - Z_qp_p[:, 5]
                dx23 = Z_qp_p[:, 2] - Z_qp_p[:, 4]
                dy23 = Z_qp_p[:, 3] - Z_qp_p[:, 5]
                Z_u_p = np.column_stack([
                    1.0 / np.sqrt(dx12**2 + dy12**2),
                    1.0 / np.sqrt(dx13**2 + dy13**2),
                    1.0 / np.sqrt(dx23**2 + dy23**2),
                ])
            else:
                Z_u_p = Z_u

            perturbed_vals = evaluate(Z_qp_p, Z_u_p)
            dg_dvar = (perturbed_vals - base_vals) / h  # (n_pts, n_gens)

            # Mask out NaN columns
            valid = ~np.isnan(dg_dvar[0, :])
            grad_sq = np.zeros(n_gens)
            if valid.any():
                grad_sq[valid] = np.nanmean(dg_dvar[:, valid]**2, axis=0)

            for body in [1, 2, 3]:
                if var_idx in body_var_indices[body]:
                    body_grad_sq[:, body - 1] += grad_sq

        # Only update scores for usable generators
        for i in range(n_gens):
            if usable[i]:
                total = body_grad_sq[i].sum()
                if total > 1e-30:
                    scores[i] = body_grad_sq[i] / total

    return scores


def compute_mutual_information(eval_matrix, Z_qp, n_bins=20):
    """
    Compute mutual information between single-body momentum projections
    and generator values.

    Returns dict with MI values for body 1 and body 2 projections.
    """
    n_pts, n_gens = eval_matrix.shape

    p1_angle = np.arctan2(Z_qp[:, 7], Z_qp[:, 6])   # atan2(py1, px1)
    p2_angle = np.arctan2(Z_qp[:, 9], Z_qp[:, 8])   # atan2(py2, px2)

    mi_body1 = np.zeros(n_gens)
    mi_body2 = np.zeros(n_gens)

    p1_bins = np.digitize(p1_angle, np.linspace(-np.pi, np.pi, n_bins + 1)) - 1
    p2_bins = np.digitize(p2_angle, np.linspace(-np.pi, np.pi, n_bins + 1)) - 1
    p1_bins = np.clip(p1_bins, 0, n_bins - 1)
    p2_bins = np.clip(p2_bins, 0, n_bins - 1)

    for k in range(n_gens):
        g_vals = eval_matrix[:, k]
        if np.all(np.isnan(g_vals)):
            continue
        valid = ~np.isnan(g_vals)
        if valid.sum() < 20:
            continue
        g_vals_v = g_vals[valid]
        g_bins_full = np.zeros(n_pts, dtype=int)
        g_bins_full[valid] = np.clip(
            np.digitize(
                g_vals_v,
                np.linspace(np.percentile(g_vals_v, 1),
                            np.percentile(g_vals_v, 99), n_bins + 1)
            ) - 1, 0, n_bins - 1)
        g_bins = g_bins_full

        for p_bins, mi_arr in [(p1_bins, mi_body1), (p2_bins, mi_body2)]:
            joint = np.zeros((n_bins, n_bins))
            for i in range(n_pts):
                joint[p_bins[i], g_bins[i]] += 1
            joint /= n_pts

            p_margin = joint.sum(axis=1)
            g_margin = joint.sum(axis=0)

            mi = 0.0
            for a in range(n_bins):
                for b in range(n_bins):
                    if joint[a, b] > 1e-15 and p_margin[a] > 1e-15 and g_margin[b] > 1e-15:
                        mi += joint[a, b] * np.log(
                            joint[a, b] / (p_margin[a] * g_margin[b]))
            mi_arr[k] = mi

    return {"body1_MI": mi_body1, "body2_MI": mi_body2}


def run_part_b(evaluate, strata, evals, exprs, names, levels, usable):
    """Gradient-based locality scoring, conditional distributions, MI."""
    print("\n" + "=" * 70)
    print("PART B: PROJECTION STATISTICS")
    print("=" * 70)

    n_gens = len(exprs)
    n_usable = sum(usable)
    print(f"  Using {n_usable}/{n_gens} vectorised generators "
          f"(skipping {n_gens - n_usable} xreplace-only)")

    # Gradient-based locality scores
    print("\n  Computing gradient-based body-locality scores...")
    t0 = time()
    locality = gradient_locality_scores(evaluate, strata, n_gens, usable)
    print(f"  Done in {time()-t0:.1f}s")

    print(f"\n  Body-Locality Scores (gradient-based):")
    print(f"  {'Idx':>4} {'Name':>25} {'Lv':>3} "
          f"{'Body1':>8} {'Body2':>8} {'Body3':>8}")
    print(f"  {'-'*4} {'-'*25} {'-'*3} {'-'*8} {'-'*8} {'-'*8}")
    for i in range(min(30, n_gens)):
        print(f"  {i:4d} {names[i]:>25} {levels[i]:3d} "
              f"{locality[i,0]:8.4f} {locality[i,1]:8.4f} "
              f"{locality[i,2]:8.4f}")
    if n_gens > 30:
        print(f"  ... ({n_gens - 30} more)")

    # Mutual information per stratum
    print("\n  Computing mutual information...")
    mi_results = {}
    for sname, (Z_qp, Z_u) in strata.items():
        M = evals[sname]
        mi = compute_mutual_information(M, Z_qp)
        mi_results[sname] = mi
        avg_mi1 = np.mean(mi["body1_MI"])
        avg_mi2 = np.mean(mi["body2_MI"])
        print(f"    {sname}: avg MI(body1) = {avg_mi1:.4f}, "
              f"avg MI(body2) = {avg_mi2:.4f}")

    return locality, mi_results


# =====================================================================
# Part C: CHSH Computation
# =====================================================================

def compute_chsh_sweep(A_vals_dict, B_vals_dict, angles, n_bootstrap=1000,
                       rng_seed=42):
    """
    Sweep over angle quadruples and compute CHSH S with bootstrap CIs.

    A_vals_dict[theta] = array of +/-1 outcomes for setting theta
    B_vals_dict[theta] = array of +/-1 outcomes for setting theta

    Returns dict with max_S, its CI, optimal angles, and full sweep data.
    """
    rng = np.random.RandomState(rng_seed)
    n_angles = len(angles)
    n_pts = len(next(iter(A_vals_dict.values())))

    best_S = 0.0
    best_S_ci = (0.0, 0.0)
    best_angles = (0, 0, 0, 0)
    sweep_data = []

    # Precompute all correlators E(a,b) with bootstrap CIs
    correlators = {}
    for a in angles:
        for b in angles:
            AB = A_vals_dict[a] * B_vals_dict[b]
            E = np.mean(AB)
            boot_Es = np.zeros(n_bootstrap)
            for i in range(n_bootstrap):
                idx = rng.randint(0, n_pts, n_pts)
                boot_Es[i] = np.mean(AB[idx])
            correlators[(a, b)] = {
                "E": E,
                "std": np.std(boot_Es),
                "boot_samples": boot_Es,
            }

    # Sweep angle quadruples
    for ia, a in enumerate(angles):
        for ia2, a_prime in enumerate(angles):
            if ia2 <= ia:
                continue
            for ib, b in enumerate(angles):
                for ib2, b_prime in enumerate(angles):
                    if ib2 <= ib:
                        continue
                    S = (correlators[(a, b)]["E"]
                         - correlators[(a, b_prime)]["E"]
                         + correlators[(a_prime, b)]["E"]
                         + correlators[(a_prime, b_prime)]["E"])

                    if abs(S) > abs(best_S):
                        boot_S = (
                            correlators[(a, b)]["boot_samples"]
                            - correlators[(a, b_prime)]["boot_samples"]
                            + correlators[(a_prime, b)]["boot_samples"]
                            + correlators[(a_prime, b_prime)]["boot_samples"]
                        )
                        ci_lo = np.percentile(boot_S, 2.5)
                        ci_hi = np.percentile(boot_S, 97.5)

                        best_S = S
                        best_S_ci = (ci_lo, ci_hi)
                        best_angles = (a, a_prime, b, b_prime)

    # Recompute bootstrap for the best angles
    a, a_prime, b, b_prime = best_angles
    boot_S = (
        correlators[(a, b)]["boot_samples"]
        - correlators[(a, b_prime)]["boot_samples"]
        + correlators[(a_prime, b)]["boot_samples"]
        + correlators[(a_prime, b_prime)]["boot_samples"]
    )
    best_S_ci = (float(np.percentile(boot_S, 2.5)),
                 float(np.percentile(boot_S, 97.5)))

    # Collect 1D sweep: fix a'=a+pi/4, b=a+pi/8, b'=a+3pi/8 (QM-optimal)
    sweep_1d = []
    for a in angles:
        a_p = a + np.pi / 4
        b_val = a + np.pi / 8
        b_p = a + 3 * np.pi / 8

        def nearest(target):
            return angles[np.argmin(np.abs(angles - target))]

        a_n = nearest(a)
        ap_n = nearest(a_p)
        b_n = nearest(b_val)
        bp_n = nearest(b_p)

        if all(k in correlators for k in
               [(a_n, b_n), (a_n, bp_n), (ap_n, b_n), (ap_n, bp_n)]):
            S_1d = (correlators[(a_n, b_n)]["E"]
                    - correlators[(a_n, bp_n)]["E"]
                    + correlators[(ap_n, b_n)]["E"]
                    + correlators[(ap_n, bp_n)]["E"])
            sweep_1d.append((float(a), float(S_1d)))

    significant = abs(best_S_ci[0]) > 2 or abs(best_S_ci[1]) > 2
    lower_exceeds_2 = min(abs(best_S_ci[0]), abs(best_S_ci[1])) > 2

    return {
        "max_S": float(best_S),
        "ci_95": best_S_ci,
        "optimal_angles_rad": [float(x) for x in best_angles],
        "optimal_angles_deg": [float(np.degrees(x)) for x in best_angles],
        "significant": bool(lower_exceeds_2),
        "n_samples": n_pts,
        "n_bootstrap": n_bootstrap,
        "sweep_1d": sweep_1d,
    }


def variant1_momentum_projections(Z_qp, angles):
    """Variant 1: standard momentum projections, expected |S| <= 2."""
    A_vals = {}
    B_vals = {}
    for theta in angles:
        A_vals[theta] = np.sign(
            np.cos(theta) * Z_qp[:, 6] + np.sin(theta) * Z_qp[:, 7])
        B_vals[theta] = np.sign(
            np.cos(theta) * Z_qp[:, 8] + np.sin(theta) * Z_qp[:, 9])
        A_vals[theta][A_vals[theta] == 0] = 1.0
        B_vals[theta][B_vals[theta] == 0] = 1.0
    return A_vals, B_vals


def variant2_algebra_projections(eval_matrix, locality, angles, usable):
    """
    Variant 2: measurement functions from algebra generators,
    weighted by gradient-based locality scores.
    """
    n_gens = eval_matrix.shape[1]

    # Only use generators with valid evaluations
    valid_cols = np.array([i for i in range(n_gens)
                           if usable[i] and not np.any(np.isnan(eval_matrix[:, i]))])

    body1_weights = np.zeros(n_gens)
    body2_weights = np.zeros(n_gens)
    body1_weights[valid_cols] = locality[valid_cols, 0]
    body2_weights[valid_cols] = locality[valid_cols, 1]

    # Normalize generator columns to unit variance for fair weighting
    col_std = np.nanstd(eval_matrix, axis=0)
    col_std[col_std < 1e-15] = 1.0
    normed = np.nan_to_num(eval_matrix / col_std, nan=0.0)

    top_k = min(20, len(valid_cols))
    body1_idx = valid_cols[np.argsort(locality[valid_cols, 0])[-top_k:]]
    body2_idx = valid_cols[np.argsort(locality[valid_cols, 1])[-top_k:]]

    A_vals = {}
    B_vals = {}
    for theta in angles:
        k = len(body1_idx)
        w1 = np.zeros(n_gens)
        for i, idx in enumerate(body1_idx):
            phase = 2 * np.pi * i / k + theta
            w1[idx] = np.cos(phase) * body1_weights[idx]
        A_raw = normed @ w1
        A_vals[theta] = np.sign(A_raw)
        A_vals[theta][A_vals[theta] == 0] = 1.0

        w2 = np.zeros(n_gens)
        for i, idx in enumerate(body2_idx):
            phase = 2 * np.pi * i / k + theta
            w2[idx] = np.cos(phase) * body2_weights[idx]
        B_raw = normed @ w2
        B_vals[theta] = np.sign(B_raw)
        B_vals[theta][B_vals[theta] == 0] = 1.0

    return A_vals, B_vals


def variant3_mediated_measurements(eval_matrix, names, angles):
    """
    Variant 3: body 3 as measurement apparatus using tidal generators.
    K1 = {H12, H13} mediates body 1, K2 = {H12, H23} mediates body 2,
    K3 = {H13, H23} is the shared apparatus channel.
    """
    # Find the K generator indices
    k_indices = {}
    for i, name in enumerate(names):
        if name == "K1":
            k_indices["K1"] = i
        elif name == "K2":
            k_indices["K2"] = i
        elif name == "K3":
            k_indices["K3"] = i

    K1_vals = eval_matrix[:, k_indices["K1"]]
    K2_vals = eval_matrix[:, k_indices["K2"]]
    K3_vals = eval_matrix[:, k_indices["K3"]]

    # Also find some higher-level generators involving body 3
    # {K1, K3} and {K2, K3} mix body 3 between bodies 1 and 2
    higher_1 = None
    higher_2 = None
    for i, name in enumerate(names):
        if name == "{K1,K3}":
            higher_1 = i
        elif name == "{K2,K3}":
            higher_2 = i

    A_vals = {}
    B_vals = {}
    for theta in angles:
        # Body 1 measurement: K1 (body 1-3 tidal) rotated by K3 (apparatus)
        A_raw = np.cos(theta) * K1_vals + np.sin(theta) * K3_vals
        if higher_1 is not None:
            A_raw += 0.3 * np.cos(2 * theta) * eval_matrix[:, higher_1]
        A_vals[theta] = np.sign(A_raw)
        A_vals[theta][A_vals[theta] == 0] = 1.0

        # Body 2 measurement: K2 (body 2-3 tidal) rotated by K3
        B_raw = np.cos(theta) * K2_vals + np.sin(theta) * K3_vals
        if higher_2 is not None:
            B_raw += 0.3 * np.cos(2 * theta) * eval_matrix[:, higher_2]
        B_vals[theta] = np.sign(B_raw)
        B_vals[theta][B_vals[theta] == 0] = 1.0

    return A_vals, B_vals


def run_part_c(strata, evals, names, locality, usable, n_angles=72,
               n_bootstrap=1000):
    """Run all three CHSH variants with bootstrap CIs."""
    print("\n" + "=" * 70)
    print("PART C: CHSH COMPUTATION")
    print("=" * 70)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    results = {}

    for stratum_name in ["equilateral", "pair_apparatus", "separated"]:
        Z_qp = strata[stratum_name][0]
        M = evals[stratum_name]
        n_pts = Z_qp.shape[0]

        print(f"\n  --- Stratum: {stratum_name} ({n_pts} points) ---")

        # Variant 1
        print(f"\n  Variant 1: Momentum projections...")
        t0 = time()
        A1, B1 = variant1_momentum_projections(Z_qp, angles)
        r1 = compute_chsh_sweep(A1, B1, angles, n_bootstrap)
        print(f"    max |S| = {abs(r1['max_S']):.6f}  "
              f"95% CI: [{r1['ci_95'][0]:.4f}, {r1['ci_95'][1]:.4f}]  "
              f"[{time()-t0:.1f}s]")
        if r1["significant"]:
            print(f"    *** SIGNIFICANT VIOLATION ***")
        else:
            print(f"    No significant violation (as expected)")

        # Variant 2
        print(f"\n  Variant 2: Algebra-projected measurements...")
        t0 = time()
        A2, B2 = variant2_algebra_projections(M, locality, angles, usable)
        r2 = compute_chsh_sweep(A2, B2, angles, n_bootstrap)
        print(f"    max |S| = {abs(r2['max_S']):.6f}  "
              f"95% CI: [{r2['ci_95'][0]:.4f}, {r2['ci_95'][1]:.4f}]  "
              f"[{time()-t0:.1f}s]")
        if r2["significant"]:
            print(f"    *** SIGNIFICANT VIOLATION ***")
        else:
            print(f"    No significant violation")

        # Variant 3
        print(f"\n  Variant 3: Body-3-mediated measurements...")
        t0 = time()
        A3, B3 = variant3_mediated_measurements(M, names, angles)
        r3 = compute_chsh_sweep(A3, B3, angles, n_bootstrap)
        print(f"    max |S| = {abs(r3['max_S']):.6f}  "
              f"95% CI: [{r3['ci_95'][0]:.4f}, {r3['ci_95'][1]:.4f}]  "
              f"[{time()-t0:.1f}s]")
        if r3["significant"]:
            print(f"    *** SIGNIFICANT VIOLATION ***")
        else:
            print(f"    No significant violation")

        results[stratum_name] = {
            "variant1": r1,
            "variant2": r2,
            "variant3": r3,
        }

    return results


# =====================================================================
# Visualization
# =====================================================================

def plot_locality_heatmap(locality, names, levels):
    """Plot gradient-based body-locality as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_gens = len(names)
    n_show = min(50, n_gens)

    fig, ax = plt.subplots(figsize=(8, max(10, n_show * 0.25)))
    im = ax.imshow(locality[:n_show], aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Body 1", "Body 2", "Body 3"])
    ax.set_yticks(range(n_show))
    short_names = []
    for i in range(n_show):
        n = names[i]
        if len(n) > 20:
            n = n[:17] + "..."
        short_names.append(f"[{i}] L{levels[i]} {n}")
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title("Gradient-Based Body-Locality Scores")
    plt.colorbar(im, ax=ax, label="Locality fraction")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "body_locality_gradients.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_projection_distributions(strata, evals, names, levels):
    """Plot conditional distributions of generators given single-body projections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    gen_indices = [3, 4, 5]  # K1, K2, K3
    stratum_names = ["equilateral", "pair_apparatus", "separated"]

    for col, sname in enumerate(stratum_names):
        Z_qp = strata[sname][0]
        M = evals[sname]

        p1_angle = np.arctan2(Z_qp[:, 7], Z_qp[:, 6])

        for row, gi in enumerate(gen_indices):
            ax = axes[row, col]
            g_vals = M[:, gi]

            h, xedges, yedges = np.histogram2d(
                p1_angle, g_vals,
                bins=[30, 30],
                range=[[-np.pi, np.pi],
                       [np.percentile(g_vals, 2), np.percentile(g_vals, 98)]]
            )
            ax.pcolormesh(xedges, yedges, h.T, cmap="viridis")
            ax.set_xlabel("Body 1 momentum angle")
            ax.set_ylabel(f"{names[gi]} value")
            if row == 0:
                ax.set_title(sname)

    plt.suptitle("Conditional Distributions: P(generator | body 1 momentum angle)",
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "projection_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_chsh_sweeps(chsh_results):
    """Plot CHSH S value sweeps for each variant and stratum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for variant_key, variant_label in [("variant1", "Momentum Projections"),
                                        ("variant2", "Algebra-Projected"),
                                        ("variant3", "Body-3 Mediated")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for col, sname in enumerate(["equilateral", "pair_apparatus",
                                      "separated"]):
            ax = axes[col]
            r = chsh_results[sname][variant_key]

            if r["sweep_1d"]:
                sweep_angles = [s[0] for s in r["sweep_1d"]]
                sweep_S = [s[1] for s in r["sweep_1d"]]
                ax.plot(np.degrees(sweep_angles), sweep_S, "o-",
                        markersize=3, label="S(a)")

            ax.axhline(y=2, color="red", linestyle="--", alpha=0.7,
                       label="Classical bound")
            ax.axhline(y=-2, color="red", linestyle="--", alpha=0.7)
            ax.axhline(y=2*np.sqrt(2), color="blue", linestyle=":",
                       alpha=0.5, label="Tsirelson bound")
            ax.axhline(y=-2*np.sqrt(2), color="blue", linestyle=":",
                       alpha=0.5)

            ax.set_xlabel("Base angle (degrees)")
            ax.set_ylabel("CHSH S")
            ax.set_title(f"{sname}\nmax|S|={abs(r['max_S']):.4f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"CHSH Sweep: {variant_label}", fontsize=13)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"chsh_sweep_{variant_key}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def save_summary(chsh_results, locality, mi_results, names, levels):
    """Save JSON summary of all results."""
    summary = {
        "experiment": "Bell Test for the Poisson Algebra",
        "n_generators": len(names),
        "level_counts": {
            str(lv): sum(1 for l in levels if l == lv)
            for lv in range(max(levels) + 1)
        },
        "strata": {},
    }

    for sname in chsh_results:
        stratum_summary = {}
        for vname in ["variant1", "variant2", "variant3"]:
            r = chsh_results[sname][vname]
            stratum_summary[vname] = {
                "max_abs_S": abs(r["max_S"]),
                "max_S": r["max_S"],
                "ci_95": r["ci_95"],
                "optimal_angles_deg": r["optimal_angles_deg"],
                "significant_violation": r["significant"],
                "n_samples": r["n_samples"],
            }
        summary["strata"][sname] = stratum_summary

    # Overall verdict
    any_significant = any(
        chsh_results[s][v]["significant"]
        for s in chsh_results
        for v in ["variant1", "variant2", "variant3"]
    )
    max_S_overall = max(
        abs(chsh_results[s][v]["max_S"])
        for s in chsh_results
        for v in ["variant1", "variant2", "variant3"]
    )
    summary["overall"] = {
        "any_significant_violation": any_significant,
        "max_abs_S_across_all": max_S_overall,
        "classical_bound": 2.0,
        "tsirelson_bound": 2 * np.sqrt(2),
    }

    path = os.path.join(RESULTS_DIR, "chsh_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")

    return summary


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Bell Test for the Poisson Algebra")
    ap.add_argument("--part", type=str, default="all",
                    choices=["all", "A", "B", "C"],
                    help="Which part to run (default: all)")
    ap.add_argument("--n-samples", type=int, default=50000,
                    help="Points per stratum for Part B (default: 50000)")
    ap.add_argument("--n-chsh-samples", type=int, default=200000,
                    help="Points per stratum for Part C (default: 200000)")
    ap.add_argument("--n-angles", type=int, default=72,
                    help="Number of angle steps for CHSH sweep (default: 72)")
    ap.add_argument("--n-bootstrap", type=int, default=1000,
                    help="Bootstrap iterations (default: 1000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_a = args.part in ("all", "A")
    run_b = args.part in ("all", "B")
    run_c = args.part in ("all", "C")

    print("=" * 70)
    print("BELL TEST FOR THE POISSON ALGEBRA")
    print("=" * 70)
    print(f"  Parts: {args.part}")
    print(f"  Samples/stratum (B): {args.n_samples}")
    print(f"  Samples/stratum (C): {args.n_chsh_samples}")
    print(f"  Angle steps: {args.n_angles}")
    print(f"  Bootstrap iterations: {args.n_bootstrap}")
    print(f"  Output: {RESULTS_DIR}")

    # Part A
    if run_a:
        run_part_a()

    if not (run_b or run_c):
        print("\nDone.")
        return

    # Load generators (shared by B and C)
    print("\n" + "=" * 70)
    print("LOADING GENERATORS AND SAMPLING")
    print("=" * 70)

    exprs, names, levels, evaluate, ALL_VARS, usable = load_and_lambdify()

    # Part B sampling
    locality = None
    mi_results = None
    strata_b = None
    evals_b = None

    if run_b:
        print(f"\n  Sampling for Part B ({args.n_samples}/stratum)...")
        strata_b = sample_all_strata(args.n_samples, seed=args.seed)
        evals_b = evaluate_on_strata(evaluate, strata_b)
        locality, mi_results = run_part_b(
            evaluate, strata_b, evals_b, exprs, names, levels, usable)

    # Part C
    chsh_results = None
    if run_c:
        n_c = args.n_chsh_samples
        print(f"\n  Sampling for Part C ({n_c}/stratum)...")
        strata_c = sample_all_strata(n_c, seed=args.seed + 10000)
        evals_c = evaluate_on_strata(evaluate, strata_c)

        if locality is None:
            print("  Computing locality scores for Variant 2...")
            locality = gradient_locality_scores(
                evaluate, strata_c, len(exprs), usable)

        chsh_results = run_part_c(
            strata_c, evals_c, names, locality, usable,
            n_angles=args.n_angles, n_bootstrap=args.n_bootstrap)

    # Visualization
    print("\n" + "=" * 70)
    print("GENERATING PLOTS AND SUMMARY")
    print("=" * 70)

    try:
        if locality is not None:
            plot_locality_heatmap(locality, names, levels)

        if strata_b is not None and evals_b is not None:
            plot_projection_distributions(strata_b, evals_b, names, levels)

        if chsh_results is not None:
            plot_chsh_sweeps(chsh_results)

        if chsh_results is not None:
            summary = save_summary(
                chsh_results, locality, mi_results, names, levels)

            print("\n" + "=" * 70)
            print("FINAL RESULTS")
            print("=" * 70)
            print(f"  Max |S| across all variants and strata: "
                  f"{summary['overall']['max_abs_S_across_all']:.6f}")
            print(f"  Classical bound: {summary['overall']['classical_bound']}")
            print(f"  Tsirelson bound: "
                  f"{summary['overall']['tsirelson_bound']:.4f}")
            if summary["overall"]["any_significant_violation"]:
                print("\n  *** SIGNIFICANT CHSH VIOLATION DETECTED ***")
                print("  This requires careful verification.")
            else:
                print("\n  No significant CHSH violation detected.")
                print("  The Poisson algebra correlations remain within "
                      "classical bounds.")

            for sname in summary["strata"]:
                print(f"\n  {sname}:")
                for vname in ["variant1", "variant2", "variant3"]:
                    v = summary["strata"][sname][vname]
                    print(f"    {vname}: |S| = {v['max_abs_S']:.4f}  "
                          f"CI: [{v['ci_95'][0]:.4f}, {v['ci_95'][1]:.4f}]"
                          f"  {'VIOLATION' if v['significant_violation'] else 'ok'}")
    except Exception as e:
        print(f"\n  Plot/summary error: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
