#!/usr/bin/env python3
"""
Direct evaluation of all generators to identify which are constraints.

Builds the symbolic Poisson algebra, evaluates each generator
independently at special configurations, and identifies:
- Identically zero generators (true constraints)
- Near-zero generators (syzygies / algebraic relations)
- Their bracket-level distribution and null-space structure
"""
import sys, io, os, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.setrecursionlimit(10000)

import numpy as np
from collections import Counter

print("=" * 70, flush=True)
print("DIRAC DIRECT GENERATOR EVALUATION", flush=True)
print("=" * 70, flush=True)

print("\nStep 1: Building PoissonAlgebra...", flush=True)
try:
    from stability_atlas import AtlasConfig, PoissonAlgebra

    config = AtlasConfig(
        potential_type='1/r',
        max_level=3,
        n_phase_samples=500,
        epsilon=1e-3,
    )

    algebra = PoissonAlgebra(config)

    levels = algebra._levels
    names = algebra._names
    n_gen = algebra._n_generators

    print(f"\nAlgebra built: {n_gen} nonzero generators", flush=True)
    for lv in range(max(levels) + 1):
        count = sum(1 for l in levels if l == lv)
        print(f"  Level {lv}: {count} generators", flush=True)

except Exception as e:
    print(f"\nFATAL: Algebra build failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Configurations to test
s = 1.0
configs = {
    'Lagrange': np.array([[0, 0], [s, 0], [s/2, s*np.sqrt(3)/2]]),
    'Isosceles': np.array([[0, 0], [s, 0], [s/2, s*1.5]]),
    'Collinear': np.array([[-s, 0], [0, 0], [s, 0]]),
    'Scalene': np.array([[0, 0], [1.0, 0], [0.3, 0.7]]),
}

# =====================================================================
# TEST A: Column norms — which generators are near-zero?
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST A: GENERATOR RMS VALUES", flush=True)
print("=" * 70, flush=True)

all_norms = {}

for cfg_name, positions in configs.items():
    for eps in [1e-2, 1e-3, 1e-4]:
        print(f"\n  Evaluating {cfg_name}, eps={eps:.0e}...", flush=True)
        try:
            Z_qp, Z_u = algebra._sample_local(positions, 500, eps)
            M = algebra._evaluate(Z_qp, Z_u)

            col_rms = np.sqrt(np.mean(M**2, axis=0))
            key = f"{cfg_name}_eps{eps:.0e}"
            all_norms[key] = col_rms

            n_tiny = np.sum(col_rms < 1e-14)
            n_small = np.sum(col_rms < 1e-10)
            print(f"    RMS < 1e-14: {n_tiny}  |  RMS < 1e-10: {n_small}", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)

# =====================================================================
# TEST B: Which generators vanish at ALL configurations?
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST B: GENERATORS THAT VANISH EVERYWHERE", flush=True)
print("=" * 70, flush=True)

ref_keys = [k for k in all_norms if 'eps1e-03' in k]

gen_max_rms = np.zeros(n_gen)
for k in ref_keys:
    gen_max_rms = np.maximum(gen_max_rms, all_norms[k])

order = np.argsort(gen_max_rms)

print(f"\n  {'idx':>4s} {'Lv':>2s} {'max_RMS':>12s}  Name", flush=True)
print("  " + "-" * 70, flush=True)

vanishing_generators = []
for i, idx in enumerate(order):
    mx = gen_max_rms[idx]
    marker = ""
    if mx < 1e-12:
        marker = " <<< VANISHES"
        vanishing_generators.append(idx)
    if i < 30 or marker:
        print(f"  {idx:>4d} L{levels[idx]:>1d} {mx:>12.4e}  "
              f"{names[idx][:50]}{marker}", flush=True)

print(f"\n  Total vanishing (max RMS < 1e-12): {len(vanishing_generators)}", flush=True)
van_levels = Counter(levels[i] for i in vanishing_generators)
print(f"  By level: {dict(van_levels)}", flush=True)

# =====================================================================
# TEST C: SVD null space analysis
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST C: SVD NULL SPACE ANALYSIS", flush=True)
print("=" * 70, flush=True)

for cfg_name, positions in configs.items():
    print(f"\n  --- {cfg_name} ---", flush=True)
    Z_qp, Z_u = algebra._sample_local(positions, 500, 1e-3)
    M = algebra._evaluate(Z_qp, Z_u)

    norms = np.linalg.norm(M, axis=0)
    n_zero_cols = np.sum(norms < 1e-15)

    norms_safe = norms.copy()
    norms_safe[norms_safe < 1e-15] = 1.0
    M_normed = M / norms_safe

    U, S, Vt = np.linalg.svd(M_normed, full_matrices=False)

    rank_10 = np.sum(S > 1e-10)
    rank_13 = np.sum(S > 1e-13)
    null_dim = n_gen - rank_13

    print(f"    Zero columns: {n_zero_cols}", flush=True)
    print(f"    SVD rank (>1e-10): {rank_10}  (>1e-13): {rank_13}", flush=True)
    print(f"    Null dim: {null_dim}", flush=True)

    if null_dim > 0:
        null_vectors = Vt[rank_13:, :]

        pure_constraint = 0
        syzygies = 0
        for k in range(null_dim):
            vec = null_vectors[k, :]
            n_sig = np.sum(np.abs(vec) > 0.01)
            if n_sig <= 1:
                pure_constraint += 1
            else:
                syzygies += 1

        print(f"    Pure constraints: {pure_constraint}", flush=True)
        print(f"    Syzygies: {syzygies}", flush=True)

        # Show a few syzygies
        n_shown = 0
        for k in range(null_dim):
            vec = null_vectors[k, :]
            sig = np.abs(vec) > 0.01
            n_sig = np.sum(sig)
            if n_sig > 1 and n_shown < 5:
                top = np.argsort(-np.abs(vec))[:5]
                desc = " + ".join(f"{vec[j]:+.3f}*g[{j}](L{levels[j]})"
                                  for j in top if abs(vec[j]) > 0.01)
                print(f"      Syzygy: {desc} = 0", flush=True)
                n_shown += 1

        # Cross-level structure
        cross_level = 0
        for k in range(null_dim):
            vec = null_vectors[k, :]
            sig_levels = set(levels[j] for j in range(n_gen)
                             if abs(null_vectors[k, j]) > 0.01)
            if len(sig_levels) > 1:
                cross_level += 1
        print(f"    Cross-level null vectors: {cross_level}/{null_dim}", flush=True)

        # Level distribution of null-space weight
        print(f"    Null weight by level:", flush=True)
        for lv in range(max(levels) + 1):
            lv_mask = np.array([1.0 if levels[j] == lv else 0.0
                                for j in range(n_gen)])
            weight = np.mean([np.sum(null_vectors[k, :]**2 * lv_mask)
                              for k in range(null_dim)])
            n_lv = sum(1 for l in levels if l == lv)
            print(f"      L{lv} ({n_lv:>3d} gen): {weight:.4f}", flush=True)

# =====================================================================
# TEST D: Compare with 1/r^2
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("TEST D: POTENTIAL COMPARISON (1/r vs 1/r^2)", flush=True)
print("=" * 70, flush=True)

try:
    config2 = AtlasConfig(
        potential_type='1/r2',
        max_level=3,
        n_phase_samples=500,
        epsilon=1e-3,
    )
    print("  Building 1/r^2 algebra...", flush=True)
    algebra2 = PoissonAlgebra(config2)

    positions = configs['Lagrange']
    Z_qp, Z_u = algebra2._sample_local(positions, 500, 1e-3)
    M = algebra2._evaluate(Z_qp, Z_u)

    col_rms = np.sqrt(np.mean(M**2, axis=0))
    n_van_2 = np.sum(col_rms < 1e-12)

    norms = np.linalg.norm(M, axis=0)
    norms[norms < 1e-15] = 1.0
    M_normed = M / norms
    U, S, Vt = np.linalg.svd(M_normed, full_matrices=False)
    rank_2 = np.sum(S > 1e-13)

    print(f"  1/r^2 at Lagrange: {algebra2._n_generators} generators, "
          f"rank={rank_2}, vanishing={n_van_2}", flush=True)

    # Same vanishing generators?
    van_1r = set(vanishing_generators)
    van_1r2 = set(np.where(col_rms < 1e-12)[0].tolist())
    print(f"  Overlap: {len(van_1r & van_1r2)} / "
          f"{len(van_1r)} (1/r) / {len(van_1r2)} (1/r^2)", flush=True)

except Exception as e:
    print(f"  ERROR building 1/r^2: {e}", flush=True)
    traceback.print_exc()

print("\n" + "=" * 70, flush=True)
print("DONE", flush=True)
print("=" * 70, flush=True)
