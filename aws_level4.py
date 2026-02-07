#!/usr/bin/env python3
"""
Level 4 Lie Algebra Dimension — Single-Pass Numerical Pipeline
================================================================

For each of 156 generators:
  1. Compute 12 partial derivatives (symbolic, with expand())
  2. Immediately lambdify each derivative (no CSE)
  3. Evaluate at sample points
  4. Store ONLY the numerical arrays — discard symbolic forms

Then compute all ~11,500 Level 4 brackets as pure NumPy
element-wise operations, and run SVD for dimension analysis.

This approach:
  - Never stores all symbolic derivatives in memory at once
  - No giant pickle files (previous 247 MB cache)
  - Naturally parallelisable across generators
  - Total runtime estimate: 30-60 min on r6i.4xlarge

Usage
-----
    python aws_level4.py compute --workers 15 --samples 3000
"""

import os
import sys
import argparse
import pickle
import numpy as np
from time import time
from multiprocessing import Pool, cpu_count

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(50000)

import sympy as sp
from sympy import diff, expand

from exact_growth import (
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    total_deriv, sample_phase_space, svd_gap_analysis,
    _make_flat_func,
)

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"


def safe_lambdify(expr, name="f"):
    """Lambdify with fallback for deeply nested expressions."""
    try:
        return sp.lambdify(ALL_VARS, expr, modules="numpy")
    except RecursionError:
        return _make_flat_func(expr, f"_{name}")

# =====================================================================
# Module-level globals for workers (inherited via fork)
# =====================================================================
_SAMPLE_ARGS = None  # list of 15 arrays for lambdify evaluation


def _init_eval_worker(sample_args):
    """Each worker stores the sample-point evaluation args."""
    global _SAMPLE_ARGS
    _SAMPLE_ARGS = sample_args


# =====================================================================
# Single-pass worker: derivative → lambdify → evaluate → return numbers
# =====================================================================
def _process_one_generator(args):
    """
    For a single generator expression:
      1. Compute 12 partial derivatives (6 position + 6 momentum)
      2. Lambdify each (no CSE — polynomial, fast)
      3. Evaluate at sample points
      4. Return numerical arrays ONLY

    Returns: (gen_idx, dq_vals, dp_vals, info_str)
      dq_vals: shape (6, n_samples)
      dp_vals: shape (6, n_samples)
    """
    gen_idx, expr, name = args
    t0 = time()
    n_samples = len(_SAMPLE_ARGS[0])

    dq_vals = np.zeros((6, n_samples))
    dp_vals = np.zeros((6, n_samples))

    total_terms = 0

    # Position derivatives (total derivative with chain rule)
    for k, q in enumerate(Q_VARS):
        d = total_deriv(expr, q)
        d = expand(d)
        total_terms += len(sp.Add.make_args(d))
        try:
            f = safe_lambdify(d, f"dq{gen_idx}_{k}")
            val = f(*_SAMPLE_ARGS)
            dq_vals[k, :] = np.atleast_1d(val).ravel()[:n_samples]
        except Exception as e:
            print(f"    WARN: eval failed for d({name})/d{q}: {e}",
                  flush=True)
            dq_vals[k, :] = 0.0

    # Momentum derivatives (simple partial)
    for k, p in enumerate(P_VARS):
        d = diff(expr, p)
        d = expand(d)
        total_terms += len(sp.Add.make_args(d))
        try:
            f = safe_lambdify(d, f"dp{gen_idx}_{k}")
            val = f(*_SAMPLE_ARGS)
            dp_vals[k, :] = np.atleast_1d(val).ravel()[:n_samples]
        except Exception as e:
            print(f"    WARN: eval failed for d({name})/d{p}: {e}",
                  flush=True)
            dp_vals[k, :] = 0.0

    elapsed = time() - t0
    info = f"{name}: {total_terms} deriv terms, {elapsed:.1f}s"

    return gen_idx, dq_vals, dp_vals, info


# =====================================================================
# Checkpoint I/O
# =====================================================================
def load_level3_checkpoint():
    path = os.path.join(CHECKPOINT_DIR, "level_3.pkl")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print("Run  python exact_growth.py --max-level 3  first.")
        sys.exit(1)
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    print(f"Loaded level-3 checkpoint: {len(data['exprs'])} expressions")
    return data


# =====================================================================
# Pair enumeration
# =====================================================================
def enumerate_level4_pairs(all_levels):
    frontier = [i for i, lv in enumerate(all_levels) if lv == 3]
    n_existing = len(all_levels)

    computed = set()
    for i in range(n_existing):
        for j in range(i + 1, n_existing):
            if all_levels[i] + all_levels[j] <= 3:
                computed.add(frozenset({i, j}))

    pairs = []
    for i in frontier:
        for j in range(n_existing):
            if i == j:
                continue
            pair = frozenset({i, j})
            if pair in computed:
                continue
            computed.add(pair)
            pairs.append((min(i, j), max(i, j)))

    return pairs


# =====================================================================
# Main compute pipeline
# =====================================================================
def cmd_compute(args):
    print("=" * 70)
    print("LEVEL 4 — SINGLE-PASS NUMERICAL PIPELINE")
    print("=" * 70)
    t_total = time()

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    data = load_level3_checkpoint()
    all_exprs = data["exprs"]
    all_names = data["names"]
    all_levels = data["levels"]
    n_gen = len(all_exprs)

    for lv in range(4):
        count = sum(1 for l in all_levels if l == lv)
        print(f"  Level {lv}: {count} generators")

    # ------------------------------------------------------------------
    # Sample phase space FIRST (needed by workers)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"STEP 1: Sample phase space ({args.samples} points)")
    print(f"{'='*70}")

    n_samples = args.samples
    Z_qp, Z_u = sample_phase_space(n_samples, seed=42)
    sample_args = ([Z_qp[:, i] for i in range(12)] +
                   [Z_u[:, i] for i in range(3)])
    print(f"  {n_samples} points sampled")

    # Also evaluate base generators at sample points
    print(f"  Evaluating {n_gen} base generators...", flush=True)
    t_base = time()
    base_vals = np.zeros((n_samples, n_gen))
    for idx, expr in enumerate(all_exprs):
        f = safe_lambdify(expr, f"base{idx}")
        val = f(*sample_args)
        base_vals[:, idx] = np.atleast_1d(val).ravel()[:n_samples]
        if (idx + 1) % 50 == 0 or idx == n_gen - 1:
            print(f"    {idx+1}/{n_gen}  [{time()-t_base:.1f}s]",
                  flush=True)
    print(f"  Base evaluation done [{time()-t_base:.1f}s]")

    # ------------------------------------------------------------------
    # Compute derivatives, lambdify, and evaluate (parallel)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"STEP 2: Compute derivatives → lambdify → evaluate")
    print(f"        {n_gen} generators × 12 derivatives, "
          f"{args.workers} workers")
    print(f"{'='*70}")

    t2 = time()

    # Check for cached numerical derivatives
    numcache = os.path.join(CHECKPOINT_DIR, "deriv_numerical.npz")
    if os.path.exists(numcache):
        print(f"  Loading cached numerical derivatives: {numcache}")
        cached = np.load(numcache)
        deriv_dq_vals = cached["dq"]
        deriv_dp_vals = cached["dp"]
        if (deriv_dq_vals.shape == (n_gen, 6, n_samples) and
                deriv_dp_vals.shape == (n_gen, 6, n_samples)):
            print(f"  Cache valid: {deriv_dq_vals.shape}")
        else:
            print(f"  Cache shape mismatch, recomputing...")
            deriv_dq_vals = None
            deriv_dp_vals = None
    else:
        deriv_dq_vals = None
        deriv_dp_vals = None

    if deriv_dq_vals is None:
        deriv_dq_vals = np.zeros((n_gen, 6, n_samples))
        deriv_dp_vals = np.zeros((n_gen, 6, n_samples))

        work = [(idx, all_exprs[idx], all_names[idx])
                for idx in range(n_gen)]

        n_done = 0
        with Pool(
            processes=args.workers,
            initializer=_init_eval_worker,
            initargs=(sample_args,),
        ) as pool:
            for gen_idx, dq, dp, info in pool.imap_unordered(
                    _process_one_generator, work):
                deriv_dq_vals[gen_idx] = dq
                deriv_dp_vals[gen_idx] = dp
                n_done += 1
                if n_done % 10 == 0 or n_done == n_gen:
                    elapsed = time() - t2
                    eta = elapsed / n_done * (n_gen - n_done)
                    print(f"  [{n_done:>4d}/{n_gen}]  "
                          f"[{elapsed:.0f}s, ETA {eta:.0f}s]  {info}",
                          flush=True)

        # Save numerical cache (tiny: ~45 MB)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        np.savez_compressed(numcache,
                            dq=deriv_dq_vals, dp=deriv_dp_vals)
        cache_size = os.path.getsize(numcache) / 1e6
        print(f"  Numerical cache saved: {numcache} ({cache_size:.1f} MB)")

    print(f"  Step 2 time: {time()-t2:.1f}s")

    # Quick sanity: check no NaN in derivatives
    n_nan = (np.sum(~np.isfinite(deriv_dq_vals)) +
             np.sum(~np.isfinite(deriv_dp_vals)))
    if n_nan > 0:
        print(f"  WARNING: {n_nan} non-finite derivative values!")

    # ------------------------------------------------------------------
    # Compute Level 4 brackets (pure NumPy)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 3: Compute Level 4 brackets (NumPy)")
    print(f"{'='*70}")

    t3 = time()
    pairs = enumerate_level4_pairs(all_levels)
    n_pairs = len(pairs)
    print(f"  Bracket pairs: {n_pairs}")

    bracket_vals = np.zeros((n_samples, n_pairs))

    for idx, (i, j) in enumerate(pairs):
        # {f_i, f_j} = Σ_k (dfi/dqk · dfj/dpk - dfi/dpk · dfj/dqk)
        bracket_vals[:, idx] = np.sum(
            deriv_dq_vals[i] * deriv_dp_vals[j]
            - deriv_dp_vals[i] * deriv_dq_vals[j],
            axis=0
        )

    t3_elapsed = time() - t3
    print(f"  {n_pairs} brackets computed in {t3_elapsed:.2f}s")

    n_bad = np.sum(~np.isfinite(bracket_vals))
    if n_bad > 0:
        print(f"  WARNING: {n_bad} non-finite bracket values!")

    # ------------------------------------------------------------------
    # SVD Analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STEP 4: SVD Dimension Analysis")
    print(f"{'='*70}")

    t4 = time()

    all_col_levels = list(all_levels) + [4] * n_pairs
    eval_matrix = np.hstack([base_vals, bracket_vals])
    print(f"  Full evaluation matrix: {eval_matrix.shape}")
    print(f"  Memory: {eval_matrix.nbytes / 1e6:.1f} MB")

    a114491 = [2, 3, 6, 17, 69, 407, 3808, 75165]
    level_dims = {}

    for lv in range(5):
        mask = [i for i, l in enumerate(all_col_levels) if l <= lv]
        if not mask:
            continue
        sub = eval_matrix[:, mask]
        print(f"\n  --- Through level {lv}: {sub.shape[1]} candidates ---")
        rank, svals = svd_gap_analysis(
            sub, label=f"(through level {lv})")
        level_dims[lv] = rank
        print(f"  ==> Dimension through level {lv}: {rank}")

        if rank >= n_samples - 10:
            print(f"  WARNING: rank near n_samples! "
                  f"Increase --samples.")

    print(f"\n  Step 4 time: {time()-t4:.1f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DIMENSION SUMMARY")
    print("=" * 70)

    for lv in sorted(level_dims.keys()):
        dim = level_dims[lv]
        pred = a114491[lv + 1] if lv + 1 < len(a114491) else "?"
        match = "MATCH" if dim == pred else "no match"
        print(f"  Level {lv}: dim = {dim:>5d}    "
              f"A114491({lv+1}) = {str(pred):>5s}    [{match}]")

    dims = [level_dims[lv] for lv in sorted(level_dims.keys())]
    print(f"\n  Dimension sequence: {dims}")

    new_per_level = [dims[0]]
    for i in range(1, len(dims)):
        new_per_level.append(dims[i] - dims[i - 1])
        ratio = dims[i] / dims[i - 1]
        print(f"    d({i})/d({i-1}) = {ratio:.2f},  "
              f"new at level {i}: {new_per_level[-1]}")

    print(f"  New-per-level: {new_per_level}")

    # Growth fit
    if len(dims) >= 3:
        try:
            from scipy.optimize import curve_fit
            def exp_model(x, a, b):
                return a * np.exp(b * x)
            lvls = np.arange(len(dims), dtype=float)
            popt, _ = curve_fit(exp_model, lvls, dims, p0=[1, 1])
            print(f"\n  Exponential fit: dim ~ {popt[0]:.2f} * "
                  f"exp({popt[1]:.3f} * level)")
            pred_5 = exp_model(5, *popt)
            print(f"  Predicted level 5: ~{pred_5:.0f}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "level_dims": level_dims,
        "dims": dims,
        "new_per_level": new_per_level,
        "n_samples": n_samples,
        "n_pairs": n_pairs,
        "eval_matrix_shape": eval_matrix.shape,
    }
    with open(os.path.join(RESULTS_DIR, "level4_results.pkl"), "wb") as fh:
        pickle.dump(results, fh)

    # Save full SVD spectrum
    norms = np.linalg.norm(eval_matrix, axis=0)
    norms[norms < 1e-15] = 1.0
    _, s_full, _ = np.linalg.svd(
        eval_matrix / norms, full_matrices=False)
    np.save(os.path.join(RESULTS_DIR, "svd_spectrum_l4.npy"), s_full)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1: SVD spectrum
        ax = axes[0, 0]
        ax.semilogy(range(1, len(s_full) + 1), s_full / s_full[0],
                     "o-", markersize=1, alpha=0.5)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular value (relative)")
        ax.set_title("SVD Spectrum (through Level 4)")
        ax.grid(True, alpha=0.3)
        for pos, lbl in [(3,"3"),(6,"6"),(17,"17"),(116,"116"),(407,"407")]:
            if pos <= len(s_full):
                ax.axvline(x=pos, color="red", ls="--", alpha=0.4)
                ax.text(pos, 0.5, f" {lbl}", color="red", fontsize=8)
        if 4 in level_dims:
            d4 = level_dims[4]
            ax.axvline(x=d4, color="green", ls="-", lw=2, alpha=0.6)
            ax.text(d4, 0.1, f" {d4}", color="green", fontsize=10,
                    fontweight="bold")

        # 2: Growth curve
        ax = axes[0, 1]
        lvls = sorted(level_dims.keys())
        dv = [level_dims[l] for l in lvls]
        ax.plot(lvls, dv, "o-", lw=2, ms=8, label="Observed")
        a_vals = [a114491[l+1] for l in lvls if l+1 < len(a114491)]
        if a_vals:
            ax.plot(lvls[:len(a_vals)], a_vals, "s--",
                    color="red", alpha=0.5, label="A114491")
        ax.set_xlabel("Bracket Level")
        ax.set_ylabel("Cumulative Dimension")
        ax.set_title("Lie Algebra Growth")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 3: New per level
        ax = axes[1, 0]
        ax.bar(lvls, new_per_level)
        ax.set_xlabel("Bracket Level")
        ax.set_ylabel("New Generators")
        ax.set_title("New Generators per Level")
        ax.grid(True, alpha=0.3, axis="y")
        for x, y in zip(lvls, new_per_level):
            ax.text(x, y, str(y), ha="center", va="bottom", fontsize=10)

        # 4: SVD gap zoom
        ax = axes[1, 1]
        if 4 in level_dims:
            d4 = level_dims[4]
            zs = max(1, d4 - 20)
            ze = min(len(s_full), d4 + 40)
            idx_range = range(zs, ze + 1)
            ax.semilogy(idx_range, s_full[zs-1:ze] / s_full[0],
                        "o-", ms=4)
            ax.axvline(x=d4, color="green", ls="-", lw=2)
            ax.text(d4, s_full[d4-1]/s_full[0],
                    f" dim={d4}", color="green", fontsize=10)
            ax.set_xlabel("Index")
            ax.set_ylabel("Singular value (relative)")
            ax.set_title(f"SVD Gap Detail (around dim={d4})")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "level4_analysis.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Plot: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")

    total_elapsed = time() - t_total
    print(f"\n{'='*70}")
    print(f"TOTAL PIPELINE TIME: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.2f} hours)")
    print(f"{'='*70}")


def main():
    ap = argparse.ArgumentParser(
        description="Level 4 Lie-algebra dimension (numerical)"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compute", help="Run numerical pipeline")
    p.add_argument("--workers", type=int,
                   default=max(1, cpu_count() - 1))
    p.add_argument("--samples", type=int, default=3000)

    args = ap.parse_args()
    if args.command == "compute":
        cmd_compute(args)


if __name__ == "__main__":
    main()
