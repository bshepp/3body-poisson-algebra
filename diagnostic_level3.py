#!/usr/bin/env python3
"""
NBodyAlgebra Diagnostic — Level 3 Checkpoint Comparison
Load both engines' level-3 checkpoints and compare ALL expressions.
Then run SVD with reduced samples (200) from ONE engine.
"""
import sys
import os
import pickle
import numpy as np
from time import time

sys.setrecursionlimit(100000)

import sympy as sp
from sympy import cancel, expand, Add

# Load Engine A checkpoint
ckpt_a_path = os.path.join("checkpoints", "level_3.pkl")
ckpt_b_path = os.path.join("nbody", "checkpoints_N3_d2_1r", "level_3.pkl")

print("=" * 70)
print("LOADING LEVEL-3 CHECKPOINTS")
print("=" * 70)

with open(ckpt_a_path, "rb") as f:
    ckpt_a = pickle.load(f)
print(f"  Engine A: {ckpt_a_path}")
print(f"    Level: {ckpt_a['level']}")
print(f"    Expressions: {len(ckpt_a['exprs'])}")
print(f"    Names: {ckpt_a['names'][:6]}...")

with open(ckpt_b_path, "rb") as f:
    ckpt_b = pickle.load(f)
print(f"  Engine B: {ckpt_b_path}")
print(f"    Level: {ckpt_b['level']}")
print(f"    Expressions: {len(ckpt_b['exprs'])}")
print(f"    Names: {ckpt_b['names'][:6]}...")

exprs_a = ckpt_a["exprs"]
exprs_b = ckpt_b["exprs"]
names_a = ckpt_a["names"]
names_b = ckpt_b["names"]
levels_a = ckpt_a["levels"]
levels_b = ckpt_b["levels"]

n = len(exprs_a)
assert n == len(exprs_b), f"Expression count mismatch: {n} vs {len(exprs_b)}"

# =====================================================================
# Compare ALL expressions symbolically
# =====================================================================
print("\n" + "=" * 70)
print(f"SYMBOLIC COMPARISON OF ALL {n} EXPRESSIONS")
print("=" * 70)

mismatches = []
t0 = time()
for i in range(n):
    ea = exprs_a[i]
    eb = exprs_b[i]
    na = names_a[i]
    nb = names_b[i]
    la = levels_a[i]
    lb = levels_b[i]

    diff = cancel(expand(ea) - expand(eb))
    terms_a = len(Add.make_args(ea))
    terms_b = len(Add.make_args(eb))

    if diff != 0:
        mismatches.append(i)
        print(f"  [{i:>3d}] L{la} {na:>30s} ({terms_a} terms)  "
              f"vs  L{lb} {nb:>30s} ({terms_b} terms)  "
              f"MISMATCH ({len(Add.make_args(diff))} diff terms)")
    else:
        if i < 6 or (i + 1) % 20 == 0 or i == n - 1:
            print(f"  [{i:>3d}] L{la} {na:>30s} ({terms_a} terms)  "
                  f"vs  L{lb} {nb:>30s} ({terms_b} terms)  "
                  f"IDENTICAL")

elapsed = time() - t0
print(f"\n  Comparison complete in {elapsed:.1f}s")
print(f"  Total expressions: {n}")
print(f"  Mismatches: {len(mismatches)}")
if mismatches:
    print(f"  Mismatch indices: {mismatches}")
else:
    print(f"  ALL {n} EXPRESSIONS ARE SYMBOLICALLY IDENTICAL")

# =====================================================================
# Level composition
# =====================================================================
print("\n" + "=" * 70)
print("LEVEL COMPOSITION")
print("=" * 70)
for lv in range(4):
    count_a = sum(1 for l in levels_a if l == lv)
    count_b = sum(1 for l in levels_b if l == lv)
    cumul_a = sum(1 for l in levels_a if l <= lv)
    cumul_b = sum(1 for l in levels_b if l <= lv)
    print(f"  Level {lv}: A has {count_a}, B has {count_b}  "
          f"(cumulative: A={cumul_a}, B={cumul_b})")

# =====================================================================
# SVD with reduced samples (200) using Engine A expressions
# =====================================================================
print("\n" + "=" * 70)
print("SVD ANALYSIS (200 samples, from Engine A checkpoint)")
print("=" * 70)

from exact_growth import sample_phase_space, ALL_VARS

n_samples = 200
seed = 42
Z_qp, Z_u = sample_phase_space(n_samples, seed)
print(f"  Sample points: {Z_qp.shape[0]}")

# Only lambdify expressions that DON'T hit RecursionError
# For ones that do, use point-by-point xreplace
t0 = time()
funcs = []
use_subs = []
for idx, expr in enumerate(exprs_a):
    if (idx + 1) % 20 == 0 or idx == n - 1:
        print(f"    Lambdifying {idx+1}/{n}  [{time()-t0:.1f}s]", flush=True)
    try:
        f = sp.lambdify(ALL_VARS, expr, modules="numpy", cse=False)
        funcs.append(f)
        use_subs.append(False)
    except RecursionError:
        funcs.append(expr)
        use_subs.append(True)

n_subs = sum(use_subs)
print(f"  {n_subs}/{n} expressions need subs() evaluator")
print(f"  Lambdify time: {time()-t0:.1f}s")

# Evaluate
print(f"\n  Evaluating {n} expressions at {n_samples} points...")
var_syms = list(ALL_VARS)
args = ([Z_qp[:, i] for i in range(12)] +
        [Z_u[:, i] for i in range(3)])

t0 = time()
cols = []
for idx, (f, is_subs) in enumerate(zip(funcs, use_subs)):
    if is_subs:
        result = np.zeros(n_samples)
        for pt in range(n_samples):
            subs_dict = {var_syms[j]: float(args[j][pt])
                         for j in range(len(var_syms))}
            try:
                result[pt] = float(f.xreplace(subs_dict))
            except Exception:
                result[pt] = 0.0
        cols.append(result)
    else:
        val = f(*args)
        arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
        if arr.shape[0] == 1:
            arr = np.full(n_samples, arr[0])
        cols.append(arr)
    if (idx + 1) % 10 == 0 or idx == n - 1:
        elapsed = time() - t0
        print(f"    eval {idx+1}/{n}  [{elapsed:.1f}s]", flush=True)

eval_matrix = np.column_stack(cols)
print(f"  Evaluation matrix shape: {eval_matrix.shape}")
print(f"  Total eval time: {time()-t0:.1f}s")

# Per-level SVD
print("\n  Per-level SVD:")
for lv in range(4):
    mask = [i for i, l in enumerate(levels_a) if l <= lv]
    sub = eval_matrix[:, mask]
    U, s, Vt = np.linalg.svd(sub, full_matrices=False)

    # Find the best gap
    best_gap = 1.0
    best_idx = -1
    for i in range(min(len(s) - 1, sub.shape[1] - 1)):
        if s[i + 1] > 1e-300:
            gap = s[i] / s[i + 1]
        else:
            gap = float("inf")
        if gap > best_gap and i >= 2:
            best_gap = gap
            best_idx = i

    noise = max(s[-1], 1e-300)
    n_above = sum(1 for sv in s if sv / s[0] > 1e-10)

    if best_gap > 1e4:
        rank = best_idx + 1
    else:
        rank = n_above

    print(f"    Level {lv}: {sub.shape[1]} candidates -> rank = {rank}  "
          f"(gap at {rank}: {best_gap:.2e}x)")

    # Print top SVs around the gap
    for i in range(min(len(s), rank + 3)):
        if i < len(s) - 1 and s[i+1] > 1e-300:
            g = s[i] / s[i+1]
        else:
            g = float("inf")
        marker = " ***" if i == best_idx else ""
        print(f"      sv[{i+1:>3d}] = {s[i]:.6e}  (gap {g:.2e}){marker}")
    if rank + 3 < len(s):
        print(f"      ...")

seq = []
for lv in range(4):
    mask = [i for i, l in enumerate(levels_a) if l <= lv]
    sub = eval_matrix[:, mask]
    _, s, _ = np.linalg.svd(sub, full_matrices=False)
    best_gap = 1.0
    best_idx = -1
    for i in range(min(len(s) - 1, sub.shape[1] - 1)):
        if s[i + 1] > 1e-300:
            gap = s[i] / s[i + 1]
        else:
            gap = float("inf")
        if gap > best_gap and i >= 2:
            best_gap = gap
            best_idx = i
    if best_gap > 1e4:
        rank = best_idx + 1
    else:
        rank = sum(1 for sv in s if sv / s[0] > 1e-10)
    seq.append(rank)

print(f"\n  DIMENSION SEQUENCE: {seq}")
print(f"  Expected:          [3, 6, 17, 116]")
print(f"  Match: {seq == [3, 6, 17, 116]}")
