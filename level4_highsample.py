#!/usr/bin/env python3
"""
Level 4 High-Sample Bound Tightening
=====================================

Incrementally increases sample count to push the d(4) lower bound
higher, saving results to S3 after each step. Stops early if a
definitive SVD gap (ratio > 100) is found.

Uses the global (full phase-space) sampling, which gave the highest
d(4) bound in the multi-config run (3,959 at 20K samples).

Memory budget (c5.4xlarge, 32 GB):
  100K samples x 11,679 columns x 8 bytes = ~9.3 GB for eval matrix
  SVD workspace ~10 GB (U, s, Vt)
  Derivative arrays ~1.5 GB
  Total ~21 GB -- fits with margin.

Usage:
    python level4_highsample.py
"""

import os
import sys
import json
import subprocess
import numpy as np
from time import time, strftime
from multiprocessing import Pool, cpu_count
import pickle

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(50000)

import sympy as sp
from sympy import diff, expand

from exact_growth import (
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    total_deriv, sample_phase_space, svd_gap_analysis,
    _make_flat_func,
)

S3_BUCKET = os.environ.get("S3_BUCKET", "")
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
STATUS_FILE = "results/highsample_status.json"

SAMPLE_COUNTS = [30000, 40000, 50000, 75000, 100000]
MAX_SAMPLES = max(SAMPLE_COUNTS)
GAP_THRESHOLD = 100.0


def s3_sync(local_path, s3_prefix):
    if not S3_BUCKET:
        return
    dest = f"s3://{S3_BUCKET}/{s3_prefix}"
    try:
        subprocess.run(
            ["aws", "s3", "sync" if os.path.isdir(local_path) else "cp",
             local_path, dest],
            capture_output=True, timeout=120
        )
    except Exception as e:
        print(f"  [S3 warning: {e}]", flush=True)


def s3_upload(local_path, s3_key):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(
            ["aws", "s3", "cp", local_path, f"s3://{S3_BUCKET}/{s3_key}"],
            capture_output=True, timeout=60
        )
    except Exception as e:
        print(f"  [S3 upload warning: {e}]", flush=True)


def safe_lambdify(expr, name="f"):
    try:
        return sp.lambdify(ALL_VARS, expr, modules="numpy")
    except RecursionError:
        return _make_flat_func(expr, f"_{name}")


def write_status(status_dict):
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(status_dict, f, indent=2)
    s3_upload(STATUS_FILE, STATUS_FILE)


def rank_from_spectrum(s, label=""):
    """Determine rank from pre-computed singular values (same logic as
    svd_gap_analysis but without re-computing the SVD)."""
    noise_threshold = 1e-8 * s[0]
    n_meaningful = int(np.sum(s > noise_threshold))

    best_gap_ratio = 1.0
    best_gap_idx = -1
    for i in range(min(n_meaningful, len(s) - 1)):
        if s[i + 1] > noise_threshold:
            gap = s[i] / s[i + 1]
        else:
            gap = s[i] / max(s[i + 1], 1e-300)
        if gap > best_gap_ratio and i >= 2:
            best_gap_ratio = gap
            best_gap_idx = i

    if best_gap_ratio > 1e4:
        rank = best_gap_idx + 1
    elif best_gap_ratio > 10:
        rank = best_gap_idx + 1
    else:
        rank = n_meaningful

    print(f"  rank_from_spectrum {label}: rank={rank}, "
          f"gap={best_gap_ratio:.2e}", flush=True)
    return rank


# -- Worker setup for multiprocessing --
_SAMPLE_ARGS = None

def _init_worker(sample_args):
    global _SAMPLE_ARGS
    _SAMPLE_ARGS = sample_args

def _process_generator(args):
    gen_idx, expr, name = args
    t0 = time()
    n_samples = len(_SAMPLE_ARGS[0])
    dq_vals = np.zeros((6, n_samples))
    dp_vals = np.zeros((6, n_samples))

    for k, q in enumerate(Q_VARS):
        d = expand(total_deriv(expr, q))
        try:
            f = safe_lambdify(d, f"dq{gen_idx}_{k}")
            val = f(*_SAMPLE_ARGS)
            dq_vals[k, :] = np.atleast_1d(val).ravel()[:n_samples]
        except Exception as e:
            print(f"    WARN: dq eval fail {name}/{q}: {e}", flush=True)

    for k, p in enumerate(P_VARS):
        d = expand(diff(expr, p))
        try:
            f = safe_lambdify(d, f"dp{gen_idx}_{k}")
            val = f(*_SAMPLE_ARGS)
            dp_vals[k, :] = np.atleast_1d(val).ravel()[:n_samples]
        except Exception as e:
            print(f"    WARN: dp eval fail {name}/{p}: {e}", flush=True)

    return gen_idx, dq_vals, dp_vals, f"{name} [{time()-t0:.1f}s]"


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


def main():
    print("=" * 70)
    print("LEVEL 4 HIGH-SAMPLE BOUND TIGHTENING")
    print(f"Sample counts: {SAMPLE_COUNTS}")
    print(f"Gap threshold for 'definitive': {GAP_THRESHOLD}")
    print("=" * 70, flush=True)

    t_total = time()
    workers = max(1, cpu_count() - 1)

    # Load Level 3 checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "level_3.pkl")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found.", flush=True)
        sys.exit(1)
    with open(ckpt_path, "rb") as fh:
        data = pickle.load(fh)
    all_exprs = data["exprs"]
    all_names = data["names"]
    all_levels = data["levels"]
    n_gen = len(all_exprs)
    print(f"Loaded {n_gen} generators from L3 checkpoint", flush=True)

    pairs = enumerate_level4_pairs(all_levels)
    n_pairs = len(pairs)
    print(f"Level 4 candidate brackets: {n_pairs}", flush=True)

    write_status({
        "phase": "sampling",
        "max_samples": MAX_SAMPLES,
        "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
    })

    # Step 1: Sample phase space
    print(f"\nSTEP 1: Sampling {MAX_SAMPLES} global phase-space points...",
          flush=True)
    t1 = time()
    Z_qp, Z_u = sample_phase_space(MAX_SAMPLES, seed=137)
    sample_args = ([Z_qp[:, i] for i in range(12)] +
                   [Z_u[:, i] for i in range(3)])
    print(f"Sampled in {time()-t1:.1f}s", flush=True)

    # Step 2: Evaluate base generators
    print(f"\nSTEP 2: Evaluating {n_gen} base generators...", flush=True)
    t2 = time()
    base_vals = np.zeros((MAX_SAMPLES, n_gen))
    for idx, expr in enumerate(all_exprs):
        f = safe_lambdify(expr, f"base{idx}")
        val = f(*sample_args)
        base_vals[:, idx] = np.atleast_1d(val).ravel()[:MAX_SAMPLES]
        if (idx + 1) % 50 == 0 or idx == n_gen - 1:
            print(f"  {idx+1}/{n_gen} [{time()-t2:.1f}s]", flush=True)

    # Step 3: Compute derivatives
    cache_path = os.path.join(CHECKPOINT_DIR,
                              f"deriv_global_highsample_{MAX_SAMPLES}.npz")

    if os.path.exists(cache_path):
        print(f"\nSTEP 3: Loading cached derivatives...", flush=True)
        cached = np.load(cache_path)
        deriv_dq = cached["dq"]
        deriv_dp = cached["dp"]
        if deriv_dq.shape != (n_gen, 6, MAX_SAMPLES):
            print("  Cache shape mismatch, recomputing...", flush=True)
            deriv_dq = None
    else:
        deriv_dq = None

    if deriv_dq is None:
        print(f"\nSTEP 3: Computing derivatives ({n_gen} x 12, "
              f"{workers} workers)...", flush=True)
        t3 = time()
        deriv_dq = np.zeros((n_gen, 6, MAX_SAMPLES))
        deriv_dp = np.zeros((n_gen, 6, MAX_SAMPLES))

        write_status({
            "phase": "derivatives",
            "n_generators": n_gen,
            "workers": workers,
            "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
        })

        work = [(idx, all_exprs[idx], all_names[idx])
                for idx in range(n_gen)]
        n_done = 0
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(sample_args,)) as pool:
            for gen_idx, dq, dp, info in pool.imap_unordered(
                    _process_generator, work):
                deriv_dq[gen_idx] = dq
                deriv_dp[gen_idx] = dp
                n_done += 1
                if n_done % 10 == 0 or n_done == n_gen:
                    elapsed = time() - t3
                    eta = elapsed / n_done * (n_gen - n_done)
                    print(f"  [{n_done:>4d}/{n_gen}] "
                          f"[{elapsed:.0f}s, ETA {eta:.0f}s] {info}",
                          flush=True)

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        np.savez_compressed(cache_path, dq=deriv_dq, dp=deriv_dp)
        s3_upload(cache_path,
                  f"checkpoints/deriv_global_highsample_{MAX_SAMPLES}.npz")
        print(f"Derivatives cached + synced to S3 [{time()-t3:.1f}s]",
              flush=True)

    # Step 4: Compute all Level 4 brackets
    print(f"\nSTEP 4: Computing {n_pairs} Level 4 brackets...", flush=True)
    t4 = time()
    bracket_vals = np.zeros((MAX_SAMPLES, n_pairs))
    for idx, (i, j) in enumerate(pairs):
        bracket_vals[:, idx] = np.sum(
            deriv_dq[i] * deriv_dp[j] - deriv_dp[i] * deriv_dq[j],
            axis=0
        )
    print(f"Brackets computed in {time()-t4:.2f}s", flush=True)

    # Free derivative arrays to save memory for SVD
    del deriv_dq, deriv_dp

    all_col_levels = list(all_levels) + [4] * n_pairs

    # Step 5: Incremental SVD at each sample count
    print(f"\n{'='*70}")
    print("STEP 5: INCREMENTAL SVD")
    print(f"{'='*70}", flush=True)

    best_d4 = 0
    found_gap = False

    for ns in SAMPLE_COUNTS:
        print(f"\n--- {ns:,} samples ---", flush=True)
        t5 = time()

        sub_base = base_vals[:ns]
        sub_brackets = bracket_vals[:ns]
        eval_matrix = np.hstack([sub_base, sub_brackets])

        mem_gb = eval_matrix.nbytes / 1e9
        print(f"  Matrix: {eval_matrix.shape}, {mem_gb:.2f} GB", flush=True)

        # Normalize columns
        norms = np.linalg.norm(eval_matrix, axis=0)
        norms[norms < 1e-15] = 1.0
        normed = eval_matrix / norms
        del eval_matrix

        # Full SVD -- singular values only (no U/Vt allocation)
        print(f"  Running SVD...", flush=True)
        t_svd = time()
        s_full = np.linalg.svd(normed, compute_uv=False)
        svd_time = time() - t_svd
        print(f"  SVD complete in {svd_time:.1f}s", flush=True)

        # Compute gap ratios
        gap_ratios = np.zeros(len(s_full) - 1)
        for i in range(len(s_full) - 1):
            if s_full[i + 1] > 1e-300:
                gap_ratios[i] = s_full[i] / s_full[i + 1]

        max_gap = float(np.max(gap_ratios))
        max_gap_idx = int(np.argmax(gap_ratios)) + 1

        # Save spectrum immediately (survives any subsequent crash)
        out_dir = os.path.join(RESULTS_DIR, f"level4_global_{ns}")
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'svd_spectrum.npy'), s_full)
        np.save(os.path.join(out_dir, 'gap_ratios.npy'), gap_ratios)
        s3_sync(out_dir, f"results/level4_global_{ns}/")
        print(f"  Spectrum saved to disk + S3", flush=True)

        # Per-level rank (L0-L3 only; L4 == full matrix, reuse s_full)
        level_dims = {}
        for lv in range(4):
            mask = [i for i, l in enumerate(all_col_levels) if l <= lv]
            if not mask:
                continue
            sub = normed[:, mask]
            rank, _ = svd_gap_analysis(sub, label=f"(L{lv})")
            level_dims[lv] = rank
            del sub

        level_dims[4] = rank_from_spectrum(s_full, label="(L4, from full SVD)")
        del normed

        d4 = level_dims.get(4, 0)
        best_d4 = max(best_d4, d4)

        dims = [level_dims[lv] for lv in sorted(level_dims.keys())]

        elapsed = time() - t5

        print(f"  dims = {dims}")
        print(f"  d(4) = {d4}")
        print(f"  max gap ratio = {max_gap:.2e} at index {max_gap_idx}")
        print(f"  [{elapsed:.1f}s total]", flush=True)

        # Save full results (updates the spectrum already saved above)
        results = {
            "config": "global_highsample",
            "n_samples": ns,
            "n_generators": n_gen,
            "n_pairs": n_pairs,
            "level_dims": {str(k): v for k, v in level_dims.items()},
            "dims": dims,
            "d4_lower_bound": d4,
            "best_d4_so_far": best_d4,
            "max_gap_ratio": max_gap,
            "max_gap_index": max_gap_idx,
            "svd_time_seconds": svd_time,
            "elapsed_seconds": elapsed,
            "definitive_gap": max_gap > GAP_THRESHOLD,
            "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
            "seed": 137,
        }

        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        s3_sync(out_dir, f"results/level4_global_{ns}/")

        write_status({
            "phase": "svd_incremental",
            "completed_sample_count": ns,
            "d4_lower_bound": d4,
            "best_d4_so_far": best_d4,
            "max_gap_ratio": max_gap,
            "definitive": max_gap > GAP_THRESHOLD,
            "remaining_counts": [c for c in SAMPLE_COUNTS if c > ns],
            "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
        })

        if max_gap > GAP_THRESHOLD:
            print(f"\n  *** DEFINITIVE GAP FOUND ***")
            print(f"  d(4) = {d4} (gap ratio {max_gap:.1e})")
            print(f"  This is an exact result, not just a lower bound.",
                  flush=True)
            found_gap = True
            break

        if d4 >= ns - 50:
            print(f"  WARNING: d(4) near sample count, need more samples",
                  flush=True)

    # Final summary
    total_time = time() - t_total
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    if found_gap:
        print(f"  d(4) = {best_d4} (EXACT — definitive SVD gap found)")
    else:
        print(f"  d(4) >= {best_d4} (lower bound — no definitive gap)")
        print(f"  The true d(4) exceeds {best_d4}. We are sample-limited.")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Results saved to S3.", flush=True)

    write_status({
        "phase": "complete",
        "best_d4": best_d4,
        "definitive": found_gap,
        "total_time_minutes": total_time / 60,
        "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
    })


if __name__ == "__main__":
    main()
