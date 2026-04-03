#!/usr/bin/env python3
"""
AWS Diagnostic: Load level-3 checkpoints from both engines,
verify symbolic identity, run full SVD with parallel subs() evaluation.
"""
import sys
import os
import pickle
import numpy as np
from time import time
from multiprocessing import Pool, cpu_count

sys.setrecursionlimit(200000)
os.environ["PYTHONUNBUFFERED"] = "1"

import sympy as sp
from sympy import cancel, expand, Add, symbols, diff, Integer

# Reconstruct variables (same as exact_growth.py)
x1, y1, x2, y2, x3, y3 = symbols("x1 y1 x2 y2 x3 y3", real=True)
px1, py1, px2, py2, px3, py3 = symbols("px1 py1 px2 py2 px3 py3", real=True)
u12, u13, u23 = symbols("u12 u13 u23", positive=True)
Q_VARS = [x1, y1, x2, y2, x3, y3]
P_VARS = [px1, py1, px2, py2, px3, py3]
U_VARS = [u12, u13, u23]
ALL_VARS = Q_VARS + P_VARS + U_VARS

def sample_phase_space(n, seed=42, pos_range=3.0, mom_range=1.0, min_sep=0.5):
    rng = np.random.RandomState(seed)
    pts = np.empty((0, 12))
    for _ in range(200):
        bs = max((n - pts.shape[0]) * 5, 256)
        b = np.zeros((bs, 12))
        b[:, :6] = rng.uniform(-pos_range, pos_range, (bs, 6))
        b[:, 6:] = rng.uniform(-mom_range, mom_range, (bs, 6))
        dx12 = b[:, 0] - b[:, 2]; dy12 = b[:, 1] - b[:, 3]
        dx13 = b[:, 0] - b[:, 4]; dy13 = b[:, 1] - b[:, 5]
        dx23 = b[:, 2] - b[:, 4]; dy23 = b[:, 3] - b[:, 5]
        ok = ((dx12**2 + dy12**2 > min_sep**2) &
              (dx13**2 + dy13**2 > min_sep**2) &
              (dx23**2 + dy23**2 > min_sep**2))
        pts = np.vstack([pts, b[ok]])
        if pts.shape[0] >= n:
            break
    pts = pts[:n]
    dx12 = pts[:, 0] - pts[:, 2]; dy12 = pts[:, 1] - pts[:, 3]
    dx13 = pts[:, 0] - pts[:, 4]; dy13 = pts[:, 1] - pts[:, 5]
    dx23 = pts[:, 2] - pts[:, 4]; dy23 = pts[:, 3] - pts[:, 5]
    Z_u = np.column_stack([
        1.0 / np.sqrt(dx12**2 + dy12**2),
        1.0 / np.sqrt(dx13**2 + dy13**2),
        1.0 / np.sqrt(dx23**2 + dy23**2),
    ])
    return pts, Z_u


def eval_subs_expr(args):
    """Evaluate a single subs()-expression at all sample points."""
    expr, var_syms_strs, sample_args, idx = args
    var_syms = [sp.Symbol(s) if not s.startswith("u") else sp.Symbol(s, positive=True)
                for s in var_syms_strs]
    n_pts = len(sample_args[0])
    result = np.zeros(n_pts)
    for pt in range(n_pts):
        subs_dict = {var_syms[j]: float(sample_args[j][pt])
                     for j in range(len(var_syms))}
        try:
            result[pt] = float(expr.xreplace(subs_dict))
        except Exception:
            result[pt] = 0.0
    return idx, result


def main():
    workdir = os.environ.get("WORKDIR", ".")
    ckpt_a_dir = os.path.join(workdir, "checkpoints_exact_growth")
    ckpt_b_dir = os.path.join(workdir, "checkpoints_NBodyAlgebra")

    print("=" * 70)
    print("AWS DIAGNOSTIC: NBodyAlgebra vs exact_growth Level-3 Comparison")
    print("=" * 70)
    print(f"  CPUs: {cpu_count()}")
    print(f"  Workdir: {workdir}")
    print(f"  Checkpoint A: {ckpt_a_dir}")
    print(f"  Checkpoint B: {ckpt_b_dir}")

    ckpt_a_path = os.path.join(ckpt_a_dir, "level_3.pkl")
    ckpt_b_path = os.path.join(ckpt_b_dir, "level_3.pkl")

    with open(ckpt_a_path, "rb") as f:
        ckpt_a = pickle.load(f)
    with open(ckpt_b_path, "rb") as f:
        ckpt_b = pickle.load(f)

    exprs_a = ckpt_a["exprs"]
    exprs_b = ckpt_b["exprs"]
    levels_a = ckpt_a["levels"]
    levels_b = ckpt_b["levels"]
    n = len(exprs_a)
    assert n == len(exprs_b), f"Count mismatch: {n} vs {len(exprs_b)}"
    print(f"  Expressions: {n}")

    # Symbolic spot-check
    print("\n--- Symbolic spot-check (all level 0-2 + every 5th level 3) ---")
    check_indices = [i for i in range(n) if levels_a[i] <= 2]
    check_indices += [i for i in range(n) if levels_a[i] == 3 and i % 5 == 0]
    check_indices = sorted(set(check_indices))
    mismatches = 0
    t0 = time()
    for i in check_indices:
        d = cancel(expand(exprs_a[i]) - expand(exprs_b[i]))
        if d != 0:
            mismatches += 1
            print(f"  [{i}] MISMATCH")
    print(f"  Checked {len(check_indices)}/{n}, mismatches: {mismatches}  "
          f"[{time()-t0:.1f}s]")

    # Lambdify phase
    print("\n--- Lambdify phase ---")
    t0 = time()
    funcs_fast = {}
    subs_indices = []
    for i, expr in enumerate(exprs_a):
        try:
            f = sp.lambdify(ALL_VARS, expr, modules="numpy", cse=False)
            funcs_fast[i] = f
        except RecursionError:
            subs_indices.append(i)
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  {i+1}/{n} [{time()-t0:.1f}s]  "
                  f"({len(funcs_fast)} fast, {len(subs_indices)} subs)",
                  flush=True)

    print(f"\n  Fast: {len(funcs_fast)}, Subs: {len(subs_indices)}")

    # Sample points
    n_samples = 500
    seed = 42
    Z_qp, Z_u = sample_phase_space(n_samples, seed)
    args_list = ([Z_qp[:, i] for i in range(12)] +
                 [Z_u[:, i] for i in range(3)])

    # Evaluate fast expressions
    print("\n--- Evaluating fast expressions ---")
    t0 = time()
    cols = {}
    for i, f in funcs_fast.items():
        val = f(*args_list)
        arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
        if arr.shape[0] == 1:
            arr = np.full(n_samples, arr[0])
        cols[i] = arr
    print(f"  {len(cols)} expressions in {time()-t0:.1f}s")

    # Evaluate subs expressions in parallel
    print(f"\n--- Evaluating {len(subs_indices)} subs expressions "
          f"in parallel ({cpu_count()} workers) ---")
    var_strs = [str(v) for v in ALL_VARS]
    sample_args_ser = [a.tolist() for a in args_list]

    tasks = [(exprs_a[i], var_strs, sample_args_ser, i)
             for i in subs_indices]

    t0 = time()
    if not tasks:
        print("  (no subs expressions to evaluate)")
    else:
        n_workers = max(1, min(cpu_count(), len(tasks)))
        with Pool(n_workers) as pool:
            for done, (idx, result) in enumerate(
                    pool.imap_unordered(eval_subs_expr, tasks)):
                cols[idx] = result
                if (done + 1) % 5 == 0 or done + 1 == len(tasks):
                    elapsed = time() - t0
                    print(f"  {done+1}/{len(tasks)} done  [{elapsed:.1f}s]",
                          flush=True)

    print(f"  Subs evaluation phase: {time()-t0:.1f}s")

    # Build full evaluation matrix
    eval_matrix = np.column_stack([cols[i] for i in range(n)])
    print(f"\n  Full evaluation matrix: {eval_matrix.shape}")

    # Per-level SVD
    print("\n" + "=" * 70)
    print("SVD ANALYSIS")
    print("=" * 70)

    level_dims = {}
    for lv in range(4):
        mask = [i for i, l in enumerate(levels_a) if l <= lv]
        sub = eval_matrix[:, mask]
        _, s, _ = np.linalg.svd(sub, full_matrices=False)

        best_gap = 1.0
        best_idx = -1
        for j in range(min(len(s) - 1, sub.shape[1] - 1)):
            if s[j + 1] > 1e-300:
                gap = s[j] / s[j + 1]
            else:
                gap = float("inf")
            if gap > best_gap and j >= 2:
                best_gap = gap
                best_idx = j

        if best_gap > 1e4:
            rank = best_idx + 1
        else:
            rank = sum(1 for sv in s if sv / s[0] > 1e-10)
        level_dims[lv] = rank

        expected = {0: 3, 1: 6, 2: 17, 3: 116}[lv]
        match = "MATCH" if rank == expected else "MISMATCH"
        print(f"\n  Level {lv}: {len(mask)} candidates -> "
              f"rank = {rank}  (expected {expected})  [{match}]")
        print(f"    Gap: {best_gap:.2e}x at index {best_idx + 1 if best_idx >= 0 else 'N/A'}")

        show_range = set(range(min(5, len(s))))
        if best_idx >= 0:
            show_range |= set(range(max(0, best_idx - 2),
                                     min(len(s), best_idx + 4)))
        show_range |= set(range(max(0, len(s) - 3), len(s)))
        for j in sorted(show_range):
            if j < len(s) - 1 and s[j+1] > 1e-300:
                g = s[j] / s[j+1]
            else:
                g = float("inf")
            marker = "  ***GAP***" if j == best_idx else ""
            print(f"    sv[{j+1:>3d}] = {s[j]:.6e}  gap={g:.2e}{marker}")

    seq = [level_dims[lv] for lv in range(4)]
    print(f"\n{'=' * 70}")
    print(f"DIMENSION SEQUENCE: {seq}")
    print(f"EXPECTED:           [3, 6, 17, 116]")
    print(f"MATCH: {seq == [3, 6, 17, 116]}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
