#!/usr/bin/env python3
"""
Quick test: validate CSE-optimised bracket computation matches
direct computation, and measure the speedup.
"""
import os
import sys
import pickle
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"

import sympy as sp
from sympy import cancel, expand

from exact_growth import (
    poisson_bracket, simplify_generator,
    poisson_bracket_from_derivs, precompute_derivatives,
    ALL_VARS,
)

# Load checkpoint
with open("checkpoints/level_3.pkl", "rb") as f:
    data = pickle.load(f)

all_exprs = data["exprs"]
all_names = data["names"]
all_levels = data["levels"]

print(f"Loaded {len(all_exprs)} expressions")
print(f"Level distribution: "
      f"{[(lv, sum(1 for l in all_levels if l == lv)) for lv in range(4)]}")

# Quick check: are expressions polynomial (no denominators)?
print("\nExpression type check (first 10):")
for i in range(min(10, len(all_exprs))):
    e = all_exprs[i]
    # cancel() returns p/q; check if q == 1
    p, q = sp.fraction(e)
    print(f"  {all_names[i]:>20s}: {len(sp.Add.make_args(e)):>5d} terms, "
          f"denom={'1' if q == 1 else str(q)[:40]}")

# -------------------------------------------------------
# Phase 1: Pre-compute derivatives for first 25 expressions
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 1: Pre-computing derivatives (first 25 exprs)")
print("=" * 60)

subset_n = 25
subset_exprs = all_exprs[:subset_n]
subset_names = all_names[:subset_n]

t0 = time()
all_derivs = precompute_derivatives(subset_exprs, subset_names, n_workers=1)
t_derivs = time() - t0
print(f"\nDerivative pre-computation: {t_derivs:.1f}s "
      f"({t_derivs/subset_n:.2f}s per expression)")

# -------------------------------------------------------
# Phase 2: Validate correctness
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 2: Correctness validation")
print("=" * 60)

# Test 1: {H12, H13} via pre-computed derivs vs direct
print("  Test 1: {H12, H13} = K1")
t0 = time()
expr_cse = poisson_bracket_from_derivs(all_derivs[0], all_derivs[1])
expr_cse = cancel(expr_cse)
t_cse = time() - t0

t0 = time()
expr_direct = poisson_bracket(all_exprs[0], all_exprs[1])
expr_direct = cancel(expr_direct)
t_direct = time() - t0

diff1 = cancel(expr_cse - expr_direct)
print(f"    CSE: {t_cse:.3f}s, Direct: {t_direct:.3f}s, "
      f"Match: {diff1 == 0}")

# Test 2: bracket with a level-2 generator (indices 6+)
if subset_n > 8:
    i2, j2 = 3, 6  # K1 × first level-2 generator
    print(f"  Test 2: {{{all_names[i2]},{all_names[j2]}}}")
    
    t0 = time()
    expr_cse2 = poisson_bracket_from_derivs(all_derivs[i2], all_derivs[j2])
    expr_cse2 = cancel(expr_cse2)
    t_cse2 = time() - t0

    t0 = time()
    expr_direct2 = poisson_bracket(all_exprs[i2], all_exprs[j2])
    expr_direct2 = cancel(expr_direct2)
    t_direct2 = time() - t0

    diff2 = cancel(expr_cse2 - expr_direct2)
    print(f"    CSE: {t_cse2:.3f}s, Direct: {t_direct2:.3f}s, "
          f"Match: {diff2 == 0}, Speedup: {t_direct2/max(t_cse2,0.001):.1f}x")

# Test 3: bracket of two level-3 generators (the key level-4 test)
if subset_n >= 20:
    i3, j3 = 18, 19  # two level-3 generators
    print(f"  Test 3: {{{all_names[i3]},{all_names[j3]}}} [LEVEL-4-LIKE]")
    
    t0 = time()
    expr_cse3 = poisson_bracket_from_derivs(all_derivs[i3], all_derivs[j3])
    t_mul = time() - t0
    expr_cse3 = cancel(expr_cse3)
    t_cse3 = time() - t0
    n_cse = len(sp.Add.make_args(expr_cse3))
    
    print(f"    CSE: multiply-add {t_mul:.1f}s + cancel {t_cse3-t_mul:.1f}s "
          f"= {t_cse3:.1f}s  ({n_cse} terms)")
    
    # Compare with direct (this is the slow one)
    print(f"    Computing direct (may be slow)...", flush=True)
    t0 = time()
    expr_direct3 = poisson_bracket(all_exprs[i3], all_exprs[j3])
    t_bracket = time() - t0
    expr_direct3 = cancel(expr_direct3)
    t_direct3 = time() - t0
    n_direct = len(sp.Add.make_args(expr_direct3))
    
    diff3 = cancel(expr_cse3 - expr_direct3)
    print(f"    Direct: bracket {t_bracket:.1f}s + cancel "
          f"{t_direct3-t_bracket:.1f}s = {t_direct3:.1f}s  ({n_direct} terms)")
    print(f"    Match: {diff3 == 0}")
    print(f"    TOTAL SPEEDUP: {t_direct3/t_cse3:.1f}x")

# -------------------------------------------------------
# Phase 3: Estimate Level 4 timing
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3: Level 4 ETA estimate")
print("=" * 60)

n_pairs = 11523
n_workers = 15

# Use the CSE bracket time from test 3 as estimate
if subset_n >= 20:
    avg_bracket_time = t_cse3
    total_serial = avg_bracket_time * n_pairs
    total_parallel = total_serial / n_workers
    
    print(f"  Per-bracket (CSE): ~{avg_bracket_time:.0f}s")
    print(f"  Total brackets: {n_pairs}")
    print(f"  Serial time: {total_serial/3600:.1f} hours")
    print(f"  Parallel ({n_workers} workers): {total_parallel/3600:.1f} hours")
    print(f"  + derivative pre-computation: ~{t_derivs/subset_n * 156 / 60:.0f} min")

print("\nDone!")
