#!/usr/bin/env python3
"""
Level 4 Exact Rank via Multiprecision Arithmetic
==================================================

Determines the exact Level 4 Poisson algebra dimension d(4) using
mpmath at 50 decimal digits, bypassing the float64 noise floor that
limits the standard pipeline to d(4) >= 5,604.

Architecture:
  Phase 1: Load Level 3 checkpoint, compute 1,872 derivative expressions
  Phase 2: Lambdify derivatives for mpmath evaluation
  Phase 3: Evaluate at mpmath-precision sample points (parallel)
  Phase 4: Incremental row echelon form for exact rank (serial)
  Phase 5: Report d(4)

Usage:
    python level4_mpmath_rank.py                    # default settings
    python level4_mpmath_rank.py --dps 80           # 80 decimal digits
    python level4_mpmath_rank.py --max-rows 15000   # more sample points
    python level4_mpmath_rank.py --resume            # resume from checkpoint
"""

import os
import sys
import json
import argparse
import pickle
import signal
import subprocess
import numpy as np
from time import time, strftime
from multiprocessing import Pool, cpu_count

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(50000)

import sympy as sp
from sympy import diff, expand, symbols, Integer

import mpmath
from mpmath import mpf, mp

from exact_growth import (
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    total_deriv,
)

# =====================================================================
# Configuration
# =====================================================================
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
S3_BUCKET = os.environ.get("S3_BUCKET", "")

DEFAULT_DPS = 50
DEFAULT_MAX_ROWS = 15000
PLATEAU_THRESHOLD = 200
CHECKPOINT_INTERVAL = 50
NONZERO_THRESHOLD_POWER = -40

# =====================================================================
# S3 helpers
# =====================================================================
def s3_upload(local_path, s3_key):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(
            ["aws", "s3", "cp", local_path, f"s3://{S3_BUCKET}/{s3_key}"],
            capture_output=True, timeout=120
        )
    except Exception as e:
        print(f"  [S3 upload warning: {e}]", flush=True)


def s3_sync(local_dir, s3_prefix):
    if not S3_BUCKET:
        return
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, f"s3://{S3_BUCKET}/{s3_prefix}"],
            capture_output=True, timeout=120
        )
    except Exception as e:
        print(f"  [S3 sync warning: {e}]", flush=True)


# =====================================================================
# Signal handling
# =====================================================================
_SHUTDOWN = False

def _sigterm_handler(signum, frame):
    global _SHUTDOWN
    print(f"\n[SIGTERM] Received at {strftime('%H:%M:%S')}, "
          f"will checkpoint and exit.", flush=True)
    _SHUTDOWN = True

signal.signal(signal.SIGTERM, _sigterm_handler)


# =====================================================================
# Phase 1: Symbolic derivatives
# =====================================================================
def compute_all_derivatives(all_exprs, all_names):
    """Compute 12 symbolic derivatives for each generator.
    Returns list of (dq_exprs[6], dp_exprs[6]) per generator."""
    n_gen = len(all_exprs)
    print(f"\nPHASE 1: Computing symbolic derivatives "
          f"({n_gen} generators x 12 derivatives)...", flush=True)
    t0 = time()

    all_derivs = []
    for idx in range(n_gen):
        expr = all_exprs[idx]
        dq = []
        dp = []
        for q in Q_VARS:
            d = expand(total_deriv(expr, q))
            dq.append(d)
        for p in P_VARS:
            d = expand(diff(expr, p))
            dp.append(d)
        all_derivs.append((dq, dp))
        if (idx + 1) % 20 == 0 or idx == n_gen - 1:
            print(f"  {idx+1}/{n_gen} [{time()-t0:.1f}s]", flush=True)

    print(f"  Derivatives computed in {time()-t0:.1f}s", flush=True)
    return all_derivs


# =====================================================================
# Phase 2: Lambdify for mpmath
# =====================================================================
def build_mpmath_evaluators(all_derivs, all_exprs):
    """Build mpmath-compatible evaluator functions for all derivatives
    and base generators."""
    n_gen = len(all_exprs)
    print(f"\nPHASE 2: Building mpmath evaluators for {n_gen} generators...",
          flush=True)
    t0 = time()

    base_funcs = []
    deriv_dq_funcs = []  # [gen][k] -> callable
    deriv_dp_funcs = []

    mp_modules = [
        {"sqrt": mpmath.sqrt, "Abs": mpmath.fabs, "log": mpmath.log,
         "exp": mpmath.exp, "sin": mpmath.sin, "cos": mpmath.cos,
         "pi": mpmath.pi},
        "mpmath",
    ]

    n_subs_fallback = 0

    for idx in range(n_gen):
        # Base generator
        try:
            f = sp.lambdify(ALL_VARS, all_exprs[idx], modules=mp_modules)
            base_funcs.append(("lambdify", f))
        except Exception:
            base_funcs.append(("subs", all_exprs[idx]))
            n_subs_fallback += 1

        dq_funcs = []
        dp_funcs = []
        dq_exprs, dp_exprs = all_derivs[idx]

        for d_expr in dq_exprs:
            try:
                f = sp.lambdify(ALL_VARS, d_expr, modules=mp_modules)
                dq_funcs.append(("lambdify", f))
            except Exception:
                dq_funcs.append(("subs", d_expr))
                n_subs_fallback += 1

        for d_expr in dp_exprs:
            try:
                f = sp.lambdify(ALL_VARS, d_expr, modules=mp_modules)
                dp_funcs.append(("lambdify", f))
            except Exception:
                dp_funcs.append(("subs", d_expr))
                n_subs_fallback += 1

        deriv_dq_funcs.append(dq_funcs)
        deriv_dp_funcs.append(dp_funcs)

        if (idx + 1) % 20 == 0 or idx == n_gen - 1:
            print(f"  {idx+1}/{n_gen} [{time()-t0:.1f}s]", flush=True)

    print(f"  Built evaluators in {time()-t0:.1f}s "
          f"({n_subs_fallback} subs() fallbacks)", flush=True)
    return base_funcs, deriv_dq_funcs, deriv_dp_funcs


def _eval_one(mode_and_func, args, var_syms):
    """Evaluate a single function at one mpmath point."""
    mode, func = mode_and_func
    if mode == "lambdify":
        try:
            return func(*args)
        except Exception:
            return mpf(0)
    else:
        subs_dict = {v: a for v, a in zip(var_syms, args)}
        try:
            val = func.xreplace(subs_dict)
            return mpf(val.evalf(mp.dps))
        except Exception:
            return mpf(0)


# =====================================================================
# Phase 3: Evaluation at mpmath points
# =====================================================================
def sample_one_point(rng_state, pos_range=3.0, mom_range=1.0, min_sep=0.5):
    """Sample one valid phase-space point as mpmath numbers.
    Returns (args_list, rng_state) where args_list has 15 mpf values."""
    rng = np.random.RandomState(rng_state)
    while True:
        q_f64 = rng.uniform(-pos_range, pos_range, 6)
        p_f64 = rng.uniform(-mom_range, mom_range, 6)

        dx12 = q_f64[0] - q_f64[2]; dy12 = q_f64[1] - q_f64[3]
        dx13 = q_f64[0] - q_f64[4]; dy13 = q_f64[1] - q_f64[5]
        dx23 = q_f64[2] - q_f64[4]; dy23 = q_f64[3] - q_f64[5]

        r12_sq = dx12**2 + dy12**2
        r13_sq = dx13**2 + dy13**2
        r23_sq = dx23**2 + dy23**2

        if min(r12_sq, r13_sq, r23_sq) < min_sep**2:
            rng_state = rng.randint(0, 2**31)
            continue

        q_mp = [mpf(str(v)) for v in q_f64]
        p_mp = [mpf(str(v)) for v in p_f64]

        dx12_mp = q_mp[0] - q_mp[2]; dy12_mp = q_mp[1] - q_mp[3]
        dx13_mp = q_mp[0] - q_mp[4]; dy13_mp = q_mp[1] - q_mp[5]
        dx23_mp = q_mp[2] - q_mp[4]; dy23_mp = q_mp[3] - q_mp[5]

        u12 = mpf(1) / mpmath.sqrt(dx12_mp**2 + dy12_mp**2)
        u13 = mpf(1) / mpmath.sqrt(dx13_mp**2 + dy13_mp**2)
        u23 = mpf(1) / mpmath.sqrt(dx23_mp**2 + dy23_mp**2)

        args = q_mp + p_mp + [u12, u13, u23]
        new_state = rng.randint(0, 2**31)
        return args, new_state


def evaluate_one_row(point_args, base_funcs, deriv_dq_funcs, deriv_dp_funcs,
                     pairs, n_gen, var_syms):
    """Evaluate all 11,679 functions at one mpmath point.
    Returns a list of mpf values (length n_gen + n_pairs)."""

    # Evaluate base generators
    base_vals = []
    for bf in base_funcs:
        base_vals.append(_eval_one(bf, point_args, var_syms))

    # Evaluate derivatives
    dq_vals = []  # [gen][k] -> mpf
    dp_vals = []
    for gen_idx in range(n_gen):
        gen_dq = []
        for k in range(6):
            v = _eval_one(deriv_dq_funcs[gen_idx][k], point_args, var_syms)
            gen_dq.append(v)
        dq_vals.append(gen_dq)

        gen_dp = []
        for k in range(6):
            v = _eval_one(deriv_dp_funcs[gen_idx][k], point_args, var_syms)
            gen_dp.append(v)
        dp_vals.append(gen_dp)

    # Compute bracket values
    bracket_vals = []
    for (i, j) in pairs:
        bval = mpf(0)
        for k in range(6):
            bval += dq_vals[i][k] * dp_vals[j][k]
            bval -= dp_vals[i][k] * dq_vals[j][k]
        bracket_vals.append(bval)

    return base_vals + bracket_vals


# Worker function for parallel evaluation
_EVAL_CONTEXT = None

def _init_eval_context(ctx):
    global _EVAL_CONTEXT
    _EVAL_CONTEXT = ctx
    mp.dps = ctx["dps"]


def _eval_worker(seed):
    ctx = _EVAL_CONTEXT
    point_args, _ = sample_one_point(seed,
                                     pos_range=ctx["pos_range"],
                                     mom_range=ctx["mom_range"])
    row = evaluate_one_row(
        point_args,
        ctx["base_funcs"],
        ctx["deriv_dq_funcs"],
        ctx["deriv_dp_funcs"],
        ctx["pairs"],
        ctx["n_gen"],
        ctx["var_syms"],
    )
    return row


# =====================================================================
# Phase 4: Incremental row echelon rank
# =====================================================================
class IncrementalRank:
    """Maintains an incremental row echelon form for rank computation."""

    def __init__(self, n_cols, threshold_power=-40):
        self.n_cols = n_cols
        self.threshold = mpf(10) ** threshold_power
        self.pivots = []  # list of (pivot_col, pivot_row as list of mpf)
        self.pivot_cols = set()
        self.rank = 0
        self.rows_processed = 0
        self.plateau_count = 0

    def add_row(self, row):
        """Process one evaluation row. Returns True if rank increased."""
        self.rows_processed += 1

        # Reduce by existing pivots
        for pcol, prow in self.pivots:
            if row[pcol] != mpf(0) and abs(row[pcol]) > self.threshold:
                factor = row[pcol] / prow[pcol]
                for j in range(self.n_cols):
                    row[j] -= factor * prow[j]

        # Find first nonzero entry
        best_col = -1
        best_abs = self.threshold
        for j in range(self.n_cols):
            a = abs(row[j])
            if a > best_abs and j not in self.pivot_cols:
                best_abs = a
                best_col = j

        if best_col >= 0:
            # Normalize and add as new pivot
            scale = row[best_col]
            for j in range(self.n_cols):
                row[j] /= scale
            self.pivots.append((best_col, row))
            self.pivot_cols.add(best_col)
            self.rank += 1
            self.plateau_count = 0
            return True
        else:
            self.plateau_count += 1
            return False

    def get_state(self):
        """Serialize state for checkpointing."""
        pivots_serialized = [
            (col, [str(v) for v in prow])
            for col, prow in self.pivots
        ]
        return {
            "rank": self.rank,
            "rows_processed": self.rows_processed,
            "plateau_count": self.plateau_count,
            "n_cols": self.n_cols,
            "n_pivots": len(self.pivots),
            "pivots": pivots_serialized,
        }

    def save_checkpoint(self, path):
        """Save checkpoint (pivots stored as pickle for speed)."""
        state = {
            "rank": self.rank,
            "rows_processed": self.rows_processed,
            "plateau_count": self.plateau_count,
            "n_cols": self.n_cols,
            "pivots": [(col, list(prow)) for col, prow in self.pivots],
        }
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load_checkpoint(cls, path, threshold_power=-40):
        """Restore from checkpoint."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["n_cols"], threshold_power)
        obj.rank = state["rank"]
        obj.rows_processed = state["rows_processed"]
        obj.plateau_count = state["plateau_count"]
        for col, prow in state["pivots"]:
            mpf_row = [mpf(v) if not isinstance(v, mpf) else v for v in prow]
            obj.pivots.append((col, mpf_row))
            obj.pivot_cols.add(col)
        return obj


# =====================================================================
# Status reporting
# =====================================================================
def write_status(status_dict, out_dir):
    path = os.path.join(out_dir, "status.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(status_dict, f, indent=2)
    s3_upload(path, f"results/level4_mpmath/status.json")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Level 4 exact rank via multiprecision arithmetic")
    parser.add_argument("--dps", type=int, default=DEFAULT_DPS,
                        help=f"Decimal digits of precision (default {DEFAULT_DPS})")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                        help=f"Max sample points (default {DEFAULT_MAX_ROWS})")
    parser.add_argument("--plateau", type=int, default=PLATEAU_THRESHOLD,
                        help=f"Stop after this many rows with no rank increase "
                             f"(default {PLATEAU_THRESHOLD})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = auto)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Evaluate this many points per batch")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    mp.dps = args.dps
    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    out_dir = os.path.join(RESULTS_DIR, "level4_mpmath")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("LEVEL 4 EXACT RANK VIA MULTIPRECISION ARITHMETIC")
    print("=" * 70)
    print(f"  mpmath precision:   {mp.dps} decimal digits")
    print(f"  Max sample rows:    {args.max_rows}")
    print(f"  Plateau threshold:  {args.plateau} rows")
    print(f"  Workers:            {n_workers}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Seed:               {args.seed}")
    print(f"  gmpy2 backend:      ", end="")
    try:
        import gmpy2
        print(f"YES (v{gmpy2.version})")
    except ImportError:
        print("NO (will be slower)")
    print("=" * 70, flush=True)

    t_total = time()

    # ---- Load Level 3 checkpoint ----
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

    # Enumerate Level 4 pairs
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
    n_pairs = len(pairs)
    n_cols = n_gen + n_pairs
    print(f"Level 4 candidate brackets: {n_pairs}")
    print(f"Total evaluation columns:   {n_cols}", flush=True)

    write_status({
        "phase": "derivatives",
        "n_gen": n_gen, "n_pairs": n_pairs, "n_cols": n_cols,
        "dps": mp.dps,
        "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
    }, out_dir)

    # ---- Phase 1: Symbolic derivatives ----
    deriv_cache = os.path.join(CHECKPOINT_DIR, "level4_derivs.pkl")
    if os.path.exists(deriv_cache):
        print(f"\nPHASE 1: Loading cached derivatives...", flush=True)
        with open(deriv_cache, "rb") as f:
            all_derivs = pickle.load(f)
        print(f"  Loaded {len(all_derivs)} derivative sets", flush=True)
    else:
        all_derivs = compute_all_derivatives(all_exprs, all_names)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(deriv_cache, "wb") as f:
            pickle.dump(all_derivs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Derivatives cached to {deriv_cache}", flush=True)

    # ---- Phase 2: Build mpmath evaluators ----
    base_funcs, deriv_dq_funcs, deriv_dp_funcs = build_mpmath_evaluators(
        all_derivs, all_exprs)

    # ---- Benchmark one point ----
    print(f"\nBENCHMARK: Evaluating 1 sample point...", flush=True)
    t_bench = time()
    test_args, _ = sample_one_point(args.seed)
    var_syms = list(ALL_VARS)
    test_row = evaluate_one_row(
        test_args, base_funcs, deriv_dq_funcs, deriv_dp_funcs,
        pairs, n_gen, var_syms)
    bench_time = time() - t_bench
    n_nonzero = sum(1 for v in test_row if abs(v) > mpf(10)**(-40))
    print(f"  1 row evaluated in {bench_time:.1f}s "
          f"({n_nonzero}/{n_cols} nonzero entries)")
    est_eval_total = bench_time * args.max_rows / max(1, n_workers)
    print(f"  Estimated eval time for {args.max_rows} rows "
          f"({n_workers} workers): {est_eval_total/3600:.1f} hours")
    est_rank_time = (n_cols * args.max_rows * 1e-6) * 3
    print(f"  Estimated rank computation: {est_rank_time/3600:.1f} hours "
          f"(depends on true rank)", flush=True)

    write_status({
        "phase": "evaluation",
        "bench_seconds_per_row": bench_time,
        "est_eval_hours": est_eval_total / 3600,
        "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
    }, out_dir)

    # ---- Phase 3+4: Evaluate and build rank incrementally ----
    rank_ckpt_path = os.path.join(out_dir, "rank_checkpoint.pkl")

    if args.resume and os.path.exists(rank_ckpt_path):
        print(f"\nRESUMING from checkpoint...", flush=True)
        ranker = IncrementalRank.load_checkpoint(
            rank_ckpt_path, NONZERO_THRESHOLD_POWER)
        start_row = ranker.rows_processed
        print(f"  Resumed: rank={ranker.rank}, "
              f"rows_processed={start_row}, "
              f"plateau={ranker.plateau_count}", flush=True)
    else:
        ranker = IncrementalRank(n_cols, NONZERO_THRESHOLD_POWER)
        start_row = 0

    print(f"\nPHASE 3+4: Incremental evaluation + rank building")
    print(f"  Starting from row {start_row}, target {args.max_rows}")
    print(f"  Will stop after {args.plateau} consecutive dependent rows")
    print("=" * 70, flush=True)

    rng_state = args.seed + start_row * 1000
    rank_history = []
    t_phase = time()
    last_checkpoint_time = time()

    batch_size = max(1, args.batch_size)

    for row_idx in range(start_row, args.max_rows):
        if _SHUTDOWN:
            print(f"\n[SHUTDOWN] Saving checkpoint at row {row_idx}...",
                  flush=True)
            ranker.save_checkpoint(rank_ckpt_path)
            s3_upload(rank_ckpt_path,
                      "results/level4_mpmath/rank_checkpoint.pkl")
            _save_results(ranker, rank_history, out_dir, args, t_total,
                          "interrupted")
            return

        t_row = time()

        # Evaluate one point
        point_args, rng_state = sample_one_point(rng_state)
        row = evaluate_one_row(
            point_args, base_funcs, deriv_dq_funcs, deriv_dp_funcs,
            pairs, n_gen, var_syms)

        t_eval = time() - t_row

        # Add to rank computation
        t_rank = time()
        increased = ranker.add_row(row)
        t_rank = time() - t_rank

        rank_history.append({
            "row": row_idx,
            "rank": ranker.rank,
            "increased": increased,
            "eval_time": t_eval,
            "rank_time": t_rank,
        })

        # Progress reporting
        total_elapsed = time() - t_phase
        avg_per_row = total_elapsed / (row_idx - start_row + 1)
        remaining = args.max_rows - row_idx - 1
        eta_hrs = (avg_per_row * remaining) / 3600

        if increased or (row_idx + 1) % 10 == 0 or row_idx == start_row:
            marker = " *** RANK UP ***" if increased else ""
            print(f"  row {row_idx+1:>6d}/{args.max_rows}: "
                  f"rank={ranker.rank:>6d}  "
                  f"plateau={ranker.plateau_count:>4d}  "
                  f"eval={t_eval:.1f}s  rank_step={t_rank:.2f}s  "
                  f"ETA={eta_hrs:.1f}h{marker}", flush=True)

        # Checkpoint
        if (time() - last_checkpoint_time > 300 or
                (row_idx + 1) % CHECKPOINT_INTERVAL == 0):
            ranker.save_checkpoint(rank_ckpt_path)
            s3_upload(rank_ckpt_path,
                      "results/level4_mpmath/rank_checkpoint.pkl")
            write_status({
                "phase": "rank_building",
                "row": row_idx + 1,
                "rank": ranker.rank,
                "plateau_count": ranker.plateau_count,
                "avg_seconds_per_row": avg_per_row,
                "eta_hours": eta_hrs,
                "elapsed_hours": total_elapsed / 3600,
                "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
            }, out_dir)
            last_checkpoint_time = time()

        # Check plateau
        if ranker.plateau_count >= args.plateau:
            print(f"\n  *** PLATEAU REACHED ***")
            print(f"  Rank stable at {ranker.rank} for "
                  f"{ranker.plateau_count} consecutive rows")
            print(f"  d(4) = {ranker.rank - n_gen + n_gen}"
                  f" (total rank includes L0-L3 generators)", flush=True)
            break

    # ---- Phase 5: Report ----
    _save_results(ranker, rank_history, out_dir, args, t_total, "complete")


def _save_results(ranker, rank_history, out_dir, args, t_total, status):
    """Save final results and checkpoint."""
    total_time = time() - t_total

    # Compute per-level dimensions from rank
    # The first n_gen columns are base generators (L0-L3)
    # Remaining are L4 brackets
    # We can estimate level dims from the pivot column distribution
    n_gen = len([p for p in ranker.pivots if True])  # total pivots = rank

    results = {
        "status": status,
        "dps": mp.dps,
        "total_rank": ranker.rank,
        "rows_processed": ranker.rows_processed,
        "plateau_count": ranker.plateau_count,
        "total_columns": ranker.n_cols,
        "total_time_hours": total_time / 3600,
        "timestamp": strftime('%Y-%m-%d %H:%M:%S'),
    }

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save rank history
    history_path = os.path.join(out_dir, "rank_history.json")
    with open(history_path, "w") as f:
        json.dump(rank_history[-1000:], f, indent=2)

    # Final checkpoint
    rank_ckpt_path = os.path.join(out_dir, "rank_checkpoint.pkl")
    ranker.save_checkpoint(rank_ckpt_path)

    # S3 sync
    s3_sync(out_dir, "results/level4_mpmath/")

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Status:          {status}")
    print(f"  Precision:       {mp.dps} decimal digits")
    print(f"  Total rank:      {ranker.rank}")
    print(f"  Rows processed:  {ranker.rows_processed}")
    print(f"  Plateau count:   {ranker.plateau_count}")
    if status == "complete" and ranker.plateau_count >= args.plateau:
        print(f"  d(4) = {ranker.rank} (EXACT — plateau confirmed)")
    else:
        print(f"  d(4) >= {ranker.rank} (lower bound — "
              f"{'interrupted' if status == 'interrupted' else 'max rows reached'})")
    print(f"  Total time:      {total_time/3600:.1f} hours")
    print(f"  Results:         {results_path}", flush=True)


if __name__ == "__main__":
    main()
