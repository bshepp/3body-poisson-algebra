"""Probe what cancel() actually does on a Schwarzschild-like L3 bracket
intermediate. Goal: figure out whether the work happens inside flint or
inside slow Python paths.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import sympy as sp
from sympy import cancel, expand
from sympy.polys.domains import GROUND_TYPES

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "nbody"))

print(f"GROUND_TYPES = {GROUND_TYPES!r}")
print(f"SYMPY_GROUND_TYPES env = {os.environ.get('SYMPY_GROUND_TYPES', '<unset>')!r}")

# Build a Schwarzschild composite L3 intermediate by replaying part of the
# pipeline. Use the small N=3 d=2 setup and force one heavy bracket.
from exact_growth_nbody import NBodyAlgebra

params = [
    (-sp.Integer(1), 1),
    (sp.Rational(1, 2), 2),
    (-sp.Integer(1), 3),
]
alg = NBodyAlgebra(
    n_bodies=3, d_spatial=2, potential="composite",
    potential_params=params, checkpoint_dir=str(REPO_ROOT / "bench_flint" / "_probe_cache"),
)

print("\n--- Building L0/L1/L2 ---")
dims = alg.compute_growth(max_level=2, n_samples=20, seed=42)

# Now grab two L2 generators and bracket them by hand to make a heavy L3
# expression, then time cancel() vs alternatives on the raw bracket.
# Pull from cache.
import pickle
with open(REPO_ROOT / "bench_flint" / "_probe_cache" / "level_2.pkl", "rb") as f:
    d = pickle.load(f)
exprs = d["exprs"]
names = d["names"]
levels = d["levels"]

# Pick two level-2 generators. Look for the ones whose bracket is known
# to be heavy (the {{H12,H13},H12} family).
l2_idxs = [i for i, lv in enumerate(levels) if lv == 2]
print(f"\nL2 generator indices: {l2_idxs[:6]}...  total {len(l2_idxs)} L2 gens")

i, j = l2_idxs[0], l2_idxs[3]
print(f"Bracketing {names[i]} with {names[j]}")
t0 = time.perf_counter()
raw = alg.poisson_bracket(exprs[i], exprs[j])
print(f"  raw bracket built in {time.perf_counter()-t0:.2f}s")
print(f"  raw expression atom count: {len(sp.Add.make_args(raw))} top-level summands")

# Test 1: full cancel()
t0 = time.perf_counter()
res = cancel(raw)
t_cancel = time.perf_counter() - t0
nterms = len(sp.Add.make_args(res))
print(f"\n[1] cancel(raw):       {t_cancel:.2f}s   -> {nterms} terms")

# Test 2: just expand()
t0 = time.perf_counter()
res2 = expand(raw)
t_expand = time.perf_counter() - t0
n2 = len(sp.Add.make_args(res2))
print(f"[2] expand(raw):       {t_expand:.2f}s   -> {n2} terms")

# Test 3: explicit Poly path
t0 = time.perf_counter()
try:
    p = sp.Poly(raw, *alg.q_vars, *alg.p_vars, *alg.u_vars)
    res3 = p.as_expr()
    t_poly = time.perf_counter() - t0
    n3 = len(sp.Add.make_args(res3))
    print(f"[3] Poly().as_expr():  {t_poly:.2f}s   -> {n3} terms")
    print(f"    Poly domain: {p.domain}, type {type(p.domain).__name__}")
    # Check if the rep uses flint
    rep = p.rep
    print(f"    Poly rep type: {type(rep).__name__} from {type(rep).__module__}")
except Exception as e:
    print(f"[3] Poly path failed: {e}")

# Test 4: try together() instead
t0 = time.perf_counter()
res4 = sp.together(raw)
t_tog = time.perf_counter() - t0
n4 = len(sp.Add.make_args(res4))
print(f"[4] together(raw):     {t_tog:.2f}s   -> {n4} terms")

# Test 5: nsimplify (less aggressive)
# Skipped - usually a no-op on rational input

# Test 6: cancel on the EXPANDED form (sometimes faster)
t0 = time.perf_counter()
res6 = cancel(res2)  # res2 is already expanded
t_can_exp = time.perf_counter() - t0
n6 = len(sp.Add.make_args(res6))
print(f"[5] cancel(expand(raw)): {t_can_exp:.2f}s   -> {n6} terms")

print("\n--- Inspecting cancel() internals ---")
# Force a small case to see the dispatch
import sympy.simplify.simplify as ssim
print(f"cancel module: {cancel.__module__}")
print(f"cancel from: {sp.cancel}")
print(f"cancel.__qualname__: {cancel.__qualname__}")
