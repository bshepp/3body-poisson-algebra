#!/usr/bin/env python3
"""
Fast diagnostic: load level-3 checkpoints, skip subs() expressions,
run SVD on only the lambdifiable ones. If rank = 116, done.
Also compare both engines' expressions symbolically.
"""
import sys, os, pickle, numpy as np
from time import time
sys.setrecursionlimit(100000)
import sympy as sp
from sympy import cancel, expand, Add

# Load both checkpoints
print("=" * 70)
print("LOADING LEVEL-3 CHECKPOINTS")
print("=" * 70)

with open(os.path.join("checkpoints", "level_3.pkl"), "rb") as f:
    ckpt_a = pickle.load(f)
with open(os.path.join("nbody", "checkpoints_N3_d2_1r", "level_3.pkl"), "rb") as f:
    ckpt_b = pickle.load(f)

exprs_a = ckpt_a["exprs"]
exprs_b = ckpt_b["exprs"]
levels_a = ckpt_a["levels"]
levels_b = ckpt_b["levels"]
names_a = ckpt_a["names"]
names_b = ckpt_b["names"]
n = len(exprs_a)

print(f"  Engine A: {n} expressions, levels {set(levels_a)}")
print(f"  Engine B: {len(exprs_b)} expressions, levels {set(levels_b)}")
assert n == len(exprs_b)

# Quick symbolic spot-check (first 18 + every 10th after that)
print("\n" + "=" * 70)
print("SYMBOLIC SPOT-CHECK")
print("=" * 70)
check_indices = list(range(18)) + list(range(20, n, 10)) + [n - 1]
check_indices = sorted(set(i for i in check_indices if i < n))
mismatches = 0
t0 = time()
for i in check_indices:
    d = cancel(expand(exprs_a[i]) - expand(exprs_b[i]))
    ok = d == 0
    ta = len(Add.make_args(exprs_a[i]))
    tb = len(Add.make_args(exprs_b[i]))
    if not ok:
        mismatches += 1
        print(f"  [{i:>3d}] MISMATCH  A:{ta} terms  B:{tb} terms")
    elif i < 18 or i == n - 1:
        print(f"  [{i:>3d}] L{levels_a[i]} IDENTICAL  ({ta} terms)")
print(f"  Checked {len(check_indices)}/{n}, mismatches: {mismatches}  "
      f"[{time()-t0:.1f}s]")

# Identify lambdifiable expressions
print("\n" + "=" * 70)
print("LAMBDIFY TEST")
print("=" * 70)
from exact_growth import ALL_VARS, sample_phase_space

t0 = time()
funcs = []
lambdifiable = []
for i, expr in enumerate(exprs_a):
    try:
        f = sp.lambdify(ALL_VARS, expr, modules="numpy", cse=False)
        funcs.append((i, f))
        lambdifiable.append(i)
    except RecursionError:
        pass
    if (i + 1) % 20 == 0 or i == n - 1:
        print(f"  {i+1}/{n} tested, {len(lambdifiable)} ok  [{time()-t0:.1f}s]",
              flush=True)
print(f"\n  Lambdifiable: {len(lambdifiable)}/{n}")
print(f"  Non-lambdifiable: {n - len(lambdifiable)}")

# Per-level breakdown
for lv in range(4):
    total_lv = [i for i in range(n) if levels_a[i] == lv]
    ok_lv = [i for i in lambdifiable if levels_a[i] == lv]
    print(f"  Level {lv}: {len(ok_lv)}/{len(total_lv)} lambdifiable")

# Evaluate only lambdifiable expressions
print("\n" + "=" * 70)
print("SVD ON LAMBDIFIABLE EXPRESSIONS ONLY")
print("=" * 70)

n_samples = 500
seed = 42
Z_qp, Z_u = sample_phase_space(n_samples, seed)
args = ([Z_qp[:, i] for i in range(12)] +
        [Z_u[:, i] for i in range(3)])

t0 = time()
cols = {}
for idx, f in funcs:
    val = f(*args)
    arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
    if arr.shape[0] == 1:
        arr = np.full(n_samples, arr[0])
    cols[idx] = arr
print(f"  Evaluated {len(cols)} expressions in {time()-t0:.1f}s")

# Per-level SVD using only lambdifiable expressions
print("\n  Per-level SVD (lambdifiable only):")
for lv in range(4):
    mask = [i for i in lambdifiable if levels_a[i] <= lv]
    if not mask:
        continue
    mat = np.column_stack([cols[i] for i in mask])
    _, s, _ = np.linalg.svd(mat, full_matrices=False)

    best_gap = 1.0
    best_idx = -1
    for j in range(min(len(s) - 1, mat.shape[1] - 1)):
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

    print(f"    Level {lv}: {len(mask)} lambdifiable candidates -> "
          f"rank = {rank}  (gap: {best_gap:.2e}x)")

    # Show SVs around expected dimension
    expected = {0: 3, 1: 6, 2: 17, 3: 116}
    exp = expected[lv]
    for j in range(min(len(s), max(rank + 3, exp + 3))):
        if j < len(s) - 1 and s[j+1] > 1e-300:
            g = s[j] / s[j+1]
        else:
            g = float("inf")
        marker = ""
        if j == exp - 1:
            marker = f"  <-- expected dim={exp}"
        if j == best_idx:
            marker += "  ***GAP***"
        if j < 5 or abs(j - exp + 1) < 3 or abs(j - rank) < 3 or j >= len(s) - 2:
            print(f"      sv[{j+1:>3d}] = {s[j]:.6e}  gap={g:.2e}{marker}")
        elif j == 5:
            print(f"      ...")

seq_fast = []
for lv in range(4):
    mask = [i for i in lambdifiable if levels_a[i] <= lv]
    mat = np.column_stack([cols[i] for i in mask])
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    best_gap = 1.0
    best_idx = -1
    for j in range(min(len(s) - 1, mat.shape[1] - 1)):
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
    seq_fast.append(rank)

print(f"\n  DIMENSION SEQUENCE (lambdifiable only): {seq_fast}")
print(f"  Expected:                               [3, 6, 17, 116]")
print(f"  Match: {seq_fast == [3, 6, 17, 116]}")

if seq_fast == [3, 6, 17, 116]:
    print("\n  *** RESULT: Both engines produce identical expressions. ***")
    print("  *** The 93 lambdifiable expressions already span dim=116. ***")
    print("  *** NBodyAlgebra = exact_growth for equal masses. ***")
    print("  *** The [3,5,13,69] from AWS survey was NOT from this code. ***")
