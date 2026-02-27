#!/usr/bin/env python3
"""
Level 4 Lie Algebra Dimension -- Multi-Configuration Pipeline
================================================================

Computes Level 4 Poisson brackets at multiple triangle configurations
(Lagrange, Euler, scalene, global) with multiple sample counts
(5K, 10K, 20K) to determine:

1. Whether the Level 4 dimension varies with configuration
2. A tighter lower bound on d(4) via higher sample counts
3. Whether the growth rate d(4)/d(3) is configuration-dependent

Architecture (build-once-scan-many per config):
  For each configuration:
    - Sample MAX_SAMPLES phase-space points (local or global)
    - Compute 156x12 derivatives (parallel, symbolic -> lambdify -> evaluate)
    - Compute ~11,500 Level 4 brackets (pure NumPy)
    - Run SVD at each sample count (subsets of the max-samples arrays)
    - Save full results (spectra, gap ratios, bracket norms, metadata)
    - S3 sync after each completed run

Usage:
    python aws_level4.py compute --batch                # all configs, all samples
    python aws_level4.py compute --config lagrange       # single config
    python aws_level4.py compute --config global --samples 20000  # legacy-style
"""

import os
import sys
import json
import argparse
import pickle
import subprocess
import numpy as np
from time import time, strftime
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

S3_BUCKET = os.environ.get("S3_BUCKET", "")

CONFIGS = {
    'lagrange': (1.0, np.pi / 3),
    'euler':    (0.5, np.pi),
    'scalene':  (0.6, 0.785),
    'global':   None,
}

SAMPLE_COUNTS = [5000, 10000, 20000]
MAX_SAMPLES = max(SAMPLE_COUNTS)
LOCAL_EPSILON = 5e-3


# =====================================================================
# S3 sync
# =====================================================================
def s3_sync(local_dir, s3_prefix=""):
    if not S3_BUCKET:
        return
    dest = f"s3://{S3_BUCKET}/{s3_prefix}" if s3_prefix else f"s3://{S3_BUCKET}/{local_dir}"
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, dest],
            capture_output=True, timeout=120
        )
    except Exception as e:
        print(f"  [S3 sync warning: {e}]", flush=True)


# =====================================================================
# Completion manifest
# =====================================================================
def _manifest_path():
    return os.path.join(RESULTS_DIR, "completed.json")


def load_manifest():
    p = _manifest_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def mark_completed(config_name, n_samples, result_dir):
    manifest = load_manifest()
    key = f"{config_name}_{n_samples}"
    manifest[key] = {
        "config": config_name,
        "n_samples": n_samples,
        "result_dir": result_dir,
        "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(_manifest_path(), 'w') as f:
        json.dump(manifest, f, indent=2)


def is_completed(config_name, n_samples):
    manifest = load_manifest()
    return f"{config_name}_{n_samples}" in manifest


# =====================================================================
# Local phase-space sampling
# =====================================================================
def shape_to_positions(mu, phi, scale=1.0):
    """Convert shape parameters (mu, phi) to 3x2 body positions."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([scale, 0.0])
    p3 = np.array([mu * scale * np.cos(phi), mu * scale * np.sin(phi)])
    return np.array([p1, p2, p3])


def sample_local(positions, n, epsilon=5e-3, seed=42,
                 mom_range=1.0, min_sep=0.01):
    """
    Sample phase-space points locally around a specific configuration.
    Positions perturbed by Gaussian noise (std=epsilon), momenta fully random.
    """
    base_q = positions.flatten()
    rng = np.random.RandomState(seed)

    Z_qp = np.zeros((n, 12))
    Z_u = np.zeros((n, 3))
    accepted = 0

    while accepted < n:
        q = base_q + rng.randn(6) * epsilon
        p = rng.randn(6) * mom_range

        dx12 = q[0] - q[2]; dy12 = q[1] - q[3]
        dx13 = q[0] - q[4]; dy13 = q[1] - q[5]
        dx23 = q[2] - q[4]; dy23 = q[3] - q[5]

        r12 = np.sqrt(dx12**2 + dy12**2)
        r13 = np.sqrt(dx13**2 + dy13**2)
        r23 = np.sqrt(dx23**2 + dy23**2)

        if min(r12, r13, r23) < min_sep:
            continue

        Z_qp[accepted, :6] = q
        Z_qp[accepted, 6:] = p
        Z_u[accepted] = [1.0 / r12, 1.0 / r13, 1.0 / r23]
        accepted += 1

    return Z_qp, Z_u


def get_sample_points(config_name, n, seed=42):
    """Generate sample points for a configuration."""
    if config_name == 'global':
        return sample_phase_space(n, seed=seed)
    mu, phi = CONFIGS[config_name]
    positions = shape_to_positions(mu, phi)
    return sample_local(positions, n, epsilon=LOCAL_EPSILON, seed=seed)


# =====================================================================
# Lambdify helper
# =====================================================================
def safe_lambdify(expr, name="f"):
    try:
        return sp.lambdify(ALL_VARS, expr, modules="numpy")
    except RecursionError:
        return _make_flat_func(expr, f"_{name}")


# =====================================================================
# Module-level globals for multiprocessing workers
# =====================================================================
_SAMPLE_ARGS = None


def _init_eval_worker(sample_args):
    global _SAMPLE_ARGS
    _SAMPLE_ARGS = sample_args


def _process_one_generator(args):
    """
    Compute 12 partial derivatives of a generator, lambdify, evaluate.
    Returns numerical arrays only.
    """
    gen_idx, expr, name = args
    t0 = time()
    n_samples = len(_SAMPLE_ARGS[0])

    dq_vals = np.zeros((6, n_samples))
    dp_vals = np.zeros((6, n_samples))
    total_terms = 0

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
# Level 3 checkpoint
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
# SVD analysis for one sample count (subset of full arrays)
# =====================================================================
def run_svd_for_sample_count(config_name, n_samples, base_vals_full,
                             bracket_vals_full, all_levels, n_pairs,
                             mu, phi):
    """Run SVD at a specific sample count and save results."""
    run_key = f"level4_{config_name}_{n_samples}"
    out_dir = os.path.join(RESULTS_DIR, run_key)

    if is_completed(config_name, n_samples):
        print(f"    [{run_key}] Already complete, skipping.", flush=True)
        return

    os.makedirs(out_dir, exist_ok=True)
    t0 = time()

    base_vals = base_vals_full[:n_samples]
    bracket_vals = bracket_vals_full[:n_samples]

    all_col_levels = list(all_levels) + [4] * n_pairs
    eval_matrix = np.hstack([base_vals, bracket_vals])

    print(f"    [{run_key}] eval_matrix: {eval_matrix.shape}, "
          f"{eval_matrix.nbytes / 1e6:.1f} MB", flush=True)

    a114491 = [2, 3, 6, 17, 69, 407, 3808, 75165]
    level_dims = {}

    for lv in range(5):
        mask = [i for i, l in enumerate(all_col_levels) if l <= lv]
        if not mask:
            continue
        sub = eval_matrix[:, mask]
        rank, svals = svd_gap_analysis(sub, label=f"(L{lv})")
        level_dims[lv] = rank

        if rank >= n_samples - 10:
            print(f"    WARNING: rank {rank} near n_samples {n_samples}!",
                  flush=True)

    dims = [level_dims[lv] for lv in sorted(level_dims.keys())]
    new_per_level = [dims[0]]
    for i in range(1, len(dims)):
        new_per_level.append(dims[i] - dims[i - 1])

    # Full SVD spectrum
    norms = np.linalg.norm(eval_matrix, axis=0)
    norms[norms < 1e-15] = 1.0
    _, s_full, _ = np.linalg.svd(
        eval_matrix / norms, full_matrices=False)

    # Gap ratios
    gap_ratios = np.zeros(len(s_full) - 1)
    for i in range(len(s_full) - 1):
        if s_full[i + 1] > 1e-300:
            gap_ratios[i] = s_full[i] / s_full[i + 1]

    # Bracket column norms
    bracket_norms = np.linalg.norm(bracket_vals, axis=0)

    # Save everything
    np.save(os.path.join(out_dir, 'svd_spectrum.npy'), s_full)
    np.save(os.path.join(out_dir, 'gap_ratios.npy'), gap_ratios)
    np.save(os.path.join(out_dir, 'bracket_norms.npy'), bracket_norms)

    elapsed = time() - t0

    results = {
        "config": config_name,
        "mu": mu,
        "phi": phi,
        "epsilon": LOCAL_EPSILON if config_name != 'global' else None,
        "n_samples": n_samples,
        "n_generators": len(all_levels),
        "n_pairs": n_pairs,
        "level_dims": {str(k): v for k, v in level_dims.items()},
        "dims": dims,
        "new_per_level": new_per_level,
        "d4_lower_bound": level_dims.get(4, -1),
        "max_gap_ratio": float(np.max(gap_ratios)) if len(gap_ratios) > 0 else 0,
        "max_gap_index": int(np.argmax(gap_ratios)) + 1 if len(gap_ratios) > 0 else 0,
        "elapsed_seconds": elapsed,
        "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    d4 = level_dims.get(4, -1)
    gmax = results["max_gap_ratio"]
    gidx = results["max_gap_index"]
    print(f"    [{run_key}] dims={dims}  d(4)={d4}  "
          f"max_gap={gmax:.1e} at #{gidx}  [{elapsed:.1f}s]", flush=True)

    for lv in sorted(level_dims.keys()):
        dim = level_dims[lv]
        pred = a114491[lv + 1] if lv + 1 < len(a114491) else "?"
        match_str = "MATCH" if dim == pred else ""
        print(f"      Level {lv}: dim={dim:>5d}  "
              f"(A114491={str(pred):>5s}) {match_str}", flush=True)

    if len(dims) >= 2:
        for i in range(1, len(dims)):
            ratio = dims[i] / dims[i - 1]
            print(f"      d({i})/d({i-1}) = {ratio:.2f}, "
                  f"new = {new_per_level[i]}", flush=True)

    # Plot
    _make_plot(out_dir, s_full, level_dims, dims, new_per_level, a114491,
               config_name, n_samples)

    mark_completed(config_name, n_samples, out_dir)
    s3_sync(out_dir, f"results/{run_key}/")
    return results


# =====================================================================
# Plot generation
# =====================================================================
def _make_plot(out_dir, s_full, level_dims, dims, new_per_level, a114491,
               config_name, n_samples):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Level 4 Analysis: {config_name} (N={n_samples})",
                     fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        ax.semilogy(range(1, len(s_full) + 1), s_full / s_full[0],
                     "o-", markersize=1, alpha=0.5)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular value (relative)")
        ax.set_title("SVD Spectrum (through Level 4)")
        ax.grid(True, alpha=0.3)
        for pos, lbl in [(3, "3"), (6, "6"), (17, "17"),
                         (116, "116"), (407, "407")]:
            if pos <= len(s_full):
                ax.axvline(x=pos, color="red", ls="--", alpha=0.4)
                ax.text(pos, 0.5, f" {lbl}", color="red", fontsize=8)
        if 4 in level_dims:
            d4 = level_dims[4]
            ax.axvline(x=d4, color="green", ls="-", lw=2, alpha=0.6)
            ax.text(d4, 0.1, f" {d4}", color="green", fontsize=10,
                    fontweight="bold")

        ax = axes[0, 1]
        lvls = sorted(level_dims.keys())
        dv = [level_dims[l] for l in lvls]
        ax.plot(lvls, dv, "o-", lw=2, ms=8, label="Observed")
        a_vals = [a114491[l + 1] for l in lvls if l + 1 < len(a114491)]
        if a_vals:
            ax.plot(lvls[:len(a_vals)], a_vals, "s--",
                    color="red", alpha=0.5, label="A114491")
        ax.set_xlabel("Bracket Level")
        ax.set_ylabel("Cumulative Dimension")
        ax.set_title("Lie Algebra Growth")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 0]
        ax.bar(lvls, new_per_level)
        ax.set_xlabel("Bracket Level")
        ax.set_ylabel("New Generators")
        ax.set_title("New Generators per Level")
        ax.grid(True, alpha=0.3, axis="y")
        for x, y in zip(lvls, new_per_level):
            ax.text(x, y, str(y), ha="center", va="bottom", fontsize=10)

        ax = axes[1, 1]
        if 4 in level_dims:
            d4 = level_dims[4]
            zs = max(1, d4 - 20)
            ze = min(len(s_full), d4 + 40)
            idx_range = range(zs, ze + 1)
            ax.semilogy(idx_range, s_full[zs - 1:ze] / s_full[0],
                        "o-", ms=4)
            ax.axvline(x=d4, color="green", ls="-", lw=2)
            ax.text(d4, s_full[d4 - 1] / s_full[0],
                    f" dim={d4}", color="green", fontsize=10)
            ax.set_xlabel("Index")
            ax.set_ylabel("Singular value (relative)")
            ax.set_title(f"SVD Gap Detail (around dim={d4})")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "level4_analysis.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"    (Plot skipped: {e})", flush=True)


# =====================================================================
# Run one configuration at all sample counts
# =====================================================================
def run_config(config_name, all_exprs, all_names, all_levels, workers):
    """
    Full Level 4 pipeline for one configuration.

    Computes derivatives at MAX_SAMPLES points (the expensive step),
    then runs SVD at each sample count using subsets.
    """
    mu = CONFIGS[config_name][0] if CONFIGS[config_name] else None
    phi = CONFIGS[config_name][1] if CONFIGS[config_name] else None
    n_gen = len(all_exprs)

    all_done = all(is_completed(config_name, ns) for ns in SAMPLE_COUNTS)
    if all_done:
        print(f"\n  [{config_name}] All sample counts complete, skipping.",
              flush=True)
        return

    print(f"\n{'='*70}")
    print(f"  CONFIG: {config_name}")
    if mu is not None:
        print(f"  mu={mu:.4f}, phi={phi:.4f} "
              f"(epsilon={LOCAL_EPSILON:.0e})")
    else:
        print(f"  Global random sampling")
    print(f"  Sample counts: {SAMPLE_COUNTS}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Step 1: Sample phase space (MAX_SAMPLES points)
    # ------------------------------------------------------------------
    print(f"  STEP 1: Sampling {MAX_SAMPLES} phase-space points...",
          flush=True)
    t1 = time()
    Z_qp, Z_u = get_sample_points(config_name, MAX_SAMPLES, seed=42)
    sample_args = ([Z_qp[:, i] for i in range(12)] +
                   [Z_u[:, i] for i in range(3)])
    print(f"  Sampled {MAX_SAMPLES} points [{time()-t1:.1f}s]", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Evaluate base generators at all sample points
    # ------------------------------------------------------------------
    print(f"  STEP 2: Evaluating {n_gen} base generators...", flush=True)
    t2 = time()
    base_vals = np.zeros((MAX_SAMPLES, n_gen))
    for idx, expr in enumerate(all_exprs):
        f = safe_lambdify(expr, f"base{idx}")
        val = f(*sample_args)
        base_vals[:, idx] = np.atleast_1d(val).ravel()[:MAX_SAMPLES]
        if (idx + 1) % 50 == 0 or idx == n_gen - 1:
            print(f"    {idx+1}/{n_gen}  [{time()-t2:.1f}s]", flush=True)
    print(f"  Base evaluation done [{time()-t2:.1f}s]", flush=True)

    # ------------------------------------------------------------------
    # Step 3: Compute derivatives (parallel)
    # ------------------------------------------------------------------
    cache_path = os.path.join(CHECKPOINT_DIR,
                              f"deriv_{config_name}_{MAX_SAMPLES}.npz")

    if os.path.exists(cache_path):
        print(f"  STEP 3: Loading cached derivatives: {cache_path}",
              flush=True)
        cached = np.load(cache_path)
        deriv_dq = cached["dq"]
        deriv_dp = cached["dp"]
        if (deriv_dq.shape == (n_gen, 6, MAX_SAMPLES) and
                deriv_dp.shape == (n_gen, 6, MAX_SAMPLES)):
            print(f"  Cache valid: {deriv_dq.shape}", flush=True)
        else:
            print(f"  Cache shape mismatch, recomputing...", flush=True)
            deriv_dq = None
    else:
        deriv_dq = None

    if deriv_dq is None:
        print(f"  STEP 3: Computing derivatives "
              f"({n_gen} generators x 12, {workers} workers)...",
              flush=True)
        t3 = time()

        deriv_dq = np.zeros((n_gen, 6, MAX_SAMPLES))
        deriv_dp = np.zeros((n_gen, 6, MAX_SAMPLES))

        work = [(idx, all_exprs[idx], all_names[idx])
                for idx in range(n_gen)]

        n_done = 0
        with Pool(
            processes=workers,
            initializer=_init_eval_worker,
            initargs=(sample_args,),
        ) as pool:
            for gen_idx, dq, dp, info in pool.imap_unordered(
                    _process_one_generator, work):
                deriv_dq[gen_idx] = dq
                deriv_dp[gen_idx] = dp
                n_done += 1
                if n_done % 10 == 0 or n_done == n_gen:
                    elapsed = time() - t3
                    eta = elapsed / n_done * (n_gen - n_done)
                    print(f"    [{n_done:>4d}/{n_gen}]  "
                          f"[{elapsed:.0f}s, ETA {eta:.0f}s]  {info}",
                          flush=True)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        np.savez_compressed(cache_path, dq=deriv_dq, dp=deriv_dp)
        sz = os.path.getsize(cache_path) / 1e6
        print(f"  Derivatives cached: {cache_path} ({sz:.1f} MB) "
              f"[{time()-t3:.1f}s]", flush=True)

    n_nan = np.sum(~np.isfinite(deriv_dq)) + np.sum(~np.isfinite(deriv_dp))
    if n_nan > 0:
        print(f"  WARNING: {n_nan} non-finite derivative values!", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Compute Level 4 brackets (pure NumPy)
    # ------------------------------------------------------------------
    print(f"  STEP 4: Computing Level 4 brackets...", flush=True)
    t4 = time()
    pairs = enumerate_level4_pairs(all_levels)
    n_pairs = len(pairs)

    bracket_vals = np.zeros((MAX_SAMPLES, n_pairs))
    for idx, (i, j) in enumerate(pairs):
        bracket_vals[:, idx] = np.sum(
            deriv_dq[i] * deriv_dp[j]
            - deriv_dp[i] * deriv_dq[j],
            axis=0
        )

    t4e = time() - t4
    print(f"  {n_pairs} brackets computed in {t4e:.2f}s", flush=True)

    n_bad = np.sum(~np.isfinite(bracket_vals))
    if n_bad > 0:
        print(f"  WARNING: {n_bad} non-finite bracket values!", flush=True)

    # ------------------------------------------------------------------
    # Step 5: SVD at each sample count
    # ------------------------------------------------------------------
    print(f"\n  STEP 5: SVD analysis at {SAMPLE_COUNTS}...", flush=True)

    for ns in SAMPLE_COUNTS:
        run_svd_for_sample_count(
            config_name, ns, base_vals, bracket_vals,
            all_levels, n_pairs, mu, phi
        )

    # Sync the manifest
    s3_sync(RESULTS_DIR, "results/")


# =====================================================================
# Main entry point
# =====================================================================
def cmd_compute(args):
    print("=" * 70)
    print("LEVEL 4 -- MULTI-CONFIGURATION PIPELINE")
    print("=" * 70)
    t_total = time()

    # Load checkpoint
    data = load_level3_checkpoint()
    all_exprs = data["exprs"]
    all_names = data["names"]
    all_levels = data["levels"]
    n_gen = len(all_exprs)

    for lv in range(4):
        count = sum(1 for l in all_levels if l == lv)
        print(f"  Level {lv}: {count} generators")

    pairs = enumerate_level4_pairs(all_levels)
    print(f"  Level 4 candidate brackets: {len(pairs)}")

    # Determine which configs to run
    if args.batch:
        config_list = list(CONFIGS.keys())
    else:
        config_list = [args.config]

    # Show plan
    total_runs = 0
    for cfg in config_list:
        for ns in SAMPLE_COUNTS:
            status = "DONE" if is_completed(cfg, ns) else "pending"
            total_runs += (0 if status == "DONE" else 1)
            print(f"  {cfg:>10s} @ {ns:>6d}: {status}")

    if total_runs == 0:
        print("\n  All runs already complete!")
    else:
        print(f"\n  {total_runs} runs pending")

    # Run each config
    for cfg in config_list:
        run_config(cfg, all_exprs, all_names, all_levels, args.workers)

    total_elapsed = time() - t_total
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.2f} hours)")
    print("=" * 70)

    # Print cross-config comparison
    print(f"\n{'='*70}")
    print("CROSS-CONFIGURATION COMPARISON")
    print("=" * 70)

    manifest = load_manifest()
    for key in sorted(manifest.keys()):
        entry = manifest[key]
        rdir = entry["result_dir"]
        rfile = os.path.join(rdir, "results.json")
        if os.path.exists(rfile):
            with open(rfile) as f:
                r = json.load(f)
            d4 = r.get("d4_lower_bound", "?")
            dims = r.get("dims", [])
            gap = r.get("max_gap_ratio", 0)
            gap_idx = r.get("max_gap_index", 0)
            print(f"  {key:>25s}: d(4)>={d4:>5}  "
                  f"dims={dims}  gap={gap:.1e} @#{gap_idx}")

    # Final S3 sync
    s3_sync(RESULTS_DIR, "results/")


def main():
    ap = argparse.ArgumentParser(
        description="Level 4 Lie-algebra dimension (multi-config)"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compute", help="Run numerical pipeline")
    p.add_argument("--workers", type=int,
                   default=max(1, cpu_count() - 1))
    p.add_argument("--samples", type=int, default=MAX_SAMPLES,
                   help="Max sample count (default: 20000)")
    p.add_argument("--config", type=str, default="global",
                   choices=list(CONFIGS.keys()),
                   help="Configuration to test")
    p.add_argument("--batch", action="store_true",
                   help="Run all configurations")

    args = ap.parse_args()
    if args.command == "compute":
        cmd_compute(args)


if __name__ == "__main__":
    main()
