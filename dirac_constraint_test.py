#!/usr/bin/env python3
"""
Test whether the 40 noise-floor generators are Dirac constraints.

Three tests:
1. EPSILON SCALING: If generator g is identically zero (constraint), its
   contribution to the SVD is zero at ALL epsilons. If it's merely small,
   it should scale with some power of epsilon.

2. CONFIGURATION DEPENDENCE: A true constraint vanishes at ALL configurations.
   Check if the noise-floor generators have zero SVs everywhere in the atlas,
   or only at specific configurations.

3. DIRECT EVALUATION: Build the algebra symbolically and evaluate each
   generator independently at exact special configurations.

4. FIRST-CLASS vs SECOND-CLASS: If constraints exist, compute their mutual
   brackets to classify them.
"""

import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

BASE = 'atlas_output_hires'
OUT = 'potential_comparison_plots'

MU = np.load(os.path.join(BASE, '1_r', 'mu_vals.npy'))
PHI = np.load(os.path.join(BASE, '1_r', 'phi_vals.npy'))
PHI_DEG = np.degrees(PHI)

EPSILONS = ['1e-04', '2e-04', '5e-04', '1e-03', '2e-03']
EPS_FLOAT = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]

LAG_ROW = np.argmin(np.abs(MU - 1.0))
LAG_COL = np.argmin(np.abs(PHI_DEG - 60))
ISO_COL = np.argmin(np.abs(PHI_DEG - 90))

# =========================================================================
# TEST 1: Epsilon scaling of noise-floor singular values
# =========================================================================
print("=" * 70)
print("TEST 1: EPSILON SCALING OF NOISE-FLOOR SINGULAR VALUES")
print("=" * 70)
print("\nIf generator g is a constraint (g = 0 on phase space), its")
print("singular value should be exactly 0 at all epsilons.")
print("If g is small but nonzero, SV should scale as epsilon^alpha.")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Dirac Constraint Test: Epsilon Scaling of Singular Values\n"
             "True constraints are zero at ALL epsilons",
             fontsize=14, fontweight='bold')

configs = [
    ('Lagrange', LAG_ROW, LAG_COL),
    ('Isosceles', LAG_ROW, ISO_COL),
    ('Collinear', LAG_ROW, 0),
    ('Small mu', 5, LAG_COL),
    ('Large mu', -5, LAG_COL),
    ('Generic', 15, 25),
]

constraint_candidates = {}

for cfg_idx, (label, row, col) in enumerate(configs):
    ax = axes[cfg_idx // 3, cfg_idx % 3]

    # Collect SVs across all epsilons for this configuration
    sv_by_eps = {}
    for eps_str, eps_val in zip(EPSILONS, EPS_FLOAT):
        path = os.path.join(BASE, '1_r', f'eps_{eps_str}', 'sv_spectra.npy')
        sv = np.load(path)
        spectrum = sv[row, col, :]
        sv_by_eps[eps_val] = spectrum

    # Stack: shape (n_eps, 156)
    eps_arr = np.array(EPS_FLOAT)
    sv_matrix = np.array([sv_by_eps[e] for e in EPS_FLOAT])

    # For each SV index, fit log(SV) vs log(eps) to get scaling exponent
    # SV ~ eps^alpha means log(SV) = alpha * log(eps) + const
    n_gen = sv_matrix.shape[1]
    alphas = np.zeros(n_gen)
    r_squared = np.zeros(n_gen)
    mean_svs = np.zeros(n_gen)

    for j in range(n_gen):
        svs_j = sv_matrix[:, j]
        mean_svs[j] = np.mean(svs_j)

        # Only fit if all SVs are positive
        if np.all(svs_j > 1e-20):
            log_eps = np.log10(eps_arr)
            log_sv = np.log10(svs_j)
            slope, intercept, r, p, se = linregress(log_eps, log_sv)
            alphas[j] = slope
            r_squared[j] = r ** 2
        else:
            alphas[j] = np.nan
            r_squared[j] = 0

    # Plot: SV index vs mean SV, colored by scaling exponent
    nonzero = mean_svs > 1e-20
    ax.scatter(range(n_gen), np.log10(np.maximum(mean_svs, 1e-20)),
               c=alphas, cmap='coolwarm', s=8, vmin=-2, vmax=4)
    ax.axvline(116, color='red', ls='--', alpha=0.7, label='Rank 116')
    ax.set_xlabel('SV index')
    ax.set_ylabel('log10(mean SV)')
    ax.set_title(f'{label} (mu={MU[row]:.2f}, phi={PHI_DEG[col]:.0f} deg)')
    ax.grid(True, alpha=0.3)
    if cfg_idx == 0:
        ax.legend()

    # Identify constraint candidates: SVs that are near-zero at ALL epsilons
    # or have very high scaling exponent (meaning they grow much faster
    # with epsilon than physical generators)
    max_sv_across_eps = np.max(sv_matrix, axis=0)
    min_sv_across_eps = np.min(sv_matrix, axis=0)

    # True constraints: max SV < threshold at all epsilons
    near_zero_all = max_sv_across_eps < 1e-13
    n_near_zero = np.sum(near_zero_all)

    # Count SVs below machine epsilon at the LARGEST epsilon
    sv_at_largest = sv_matrix[-1, :]  # eps=2e-03
    n_noise_largest = np.sum(sv_at_largest < 1e-13)

    # Count SVs below machine epsilon at the SMALLEST epsilon
    sv_at_smallest = sv_matrix[0, :]  # eps=1e-04
    n_noise_smallest = np.sum(sv_at_smallest < 1e-13)

    print(f"\n  {label}:")
    print(f"    SVs < 1e-13 at ALL epsilons: {n_near_zero}")
    print(f"    SVs < 1e-13 at eps=1e-04: {n_noise_smallest}")
    print(f"    SVs < 1e-13 at eps=2e-03: {n_noise_largest}")

    if n_near_zero > 0:
        idxs = np.where(near_zero_all)[0]
        print(f"    Constraint candidate indices: {idxs.tolist()}")
        constraint_candidates[label] = idxs.tolist()

    # Scaling exponents for noise-floor SVs (index >= 116)
    noise_alphas = alphas[116:]
    noise_alphas = noise_alphas[~np.isnan(noise_alphas)]
    if len(noise_alphas) > 0:
        print(f"    Noise-floor scaling exponents (SV >= 116):")
        print(f"      Mean alpha: {np.mean(noise_alphas):.2f}")
        print(f"      Min alpha: {np.min(noise_alphas):.2f}")
        print(f"      Max alpha: {np.max(noise_alphas):.2f}")

    # Scaling exponents for significant SVs (index < 116)
    sig_alphas = alphas[:116]
    sig_alphas = sig_alphas[~np.isnan(sig_alphas)]
    if len(sig_alphas) > 0:
        print(f"    Significant scaling exponents (SV < 116):")
        print(f"      Mean alpha: {np.mean(sig_alphas):.2f}")

plt.tight_layout()
path = os.path.join(OUT, 'dirac_constraint_epsilon_scaling.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  Saved {path}")

# =========================================================================
# TEST 2: Configuration dependence — do the noise SVs vanish EVERYWHERE?
# =========================================================================
print("\n" + "=" * 70)
print("TEST 2: CONFIGURATION DEPENDENCE OF NOISE-FLOOR SVs")
print("=" * 70)
print("\nA true constraint vanishes at ALL configurations.")
print("Checking SV[116:] across the full 100x100 atlas.")

# Use eps=1e-04 (where the noise floor is most clearly separated)
sv_full = np.load(os.path.join(BASE, '1_r', 'eps_1e-04', 'sv_spectra.npy'))

# For each SV index > 116, compute max value across all configurations
max_across_configs = np.max(sv_full, axis=(0, 1))  # shape (156,)
min_across_configs = np.min(sv_full, axis=(0, 1))

print("\n  SV index | max across atlas | min across atlas | Constraint?")
print("  " + "-" * 65)

n_true_constraints = 0
constraint_indices_global = []

for j in range(156):
    mx = max_across_configs[j]
    mn = min_across_configs[j]
    is_constraint = mx < 1e-13
    marker = " <-- CONSTRAINT" if is_constraint else ""
    if j >= 110 or is_constraint:
        print(f"  {j:>8d} | {mx:>16.6e} | {mn:>16.6e} |{marker}")
    if is_constraint:
        n_true_constraints += 1
        constraint_indices_global.append(j)

print(f"\n  True constraints (max SV < 1e-13 at ALL configs): "
      f"{n_true_constraints}")
print(f"  Indices: {constraint_indices_global}")

# =========================================================================
# TEST 2b: How many SVs are consistently near-zero vs config-dependent?
# =========================================================================
print("\n  --- Noise floor structure ---")
# For indices 116-155 (the 40 noise-floor generators):
noise_maxes = max_across_configs[116:]
noise_mins = min_across_configs[116:]

# Categorize noise-floor SVs
truly_zero = np.sum(noise_maxes < 1e-15)
near_zero = np.sum(noise_maxes < 1e-13) - truly_zero
small_but_nonzero = np.sum((noise_maxes >= 1e-13) & (noise_maxes < 1e-10))
moderate = np.sum(noise_maxes >= 1e-10)

print(f"  Of the 40 noise-floor SVs (indices 116-155):")
print(f"    Truly zero (< 1e-15 everywhere):  {truly_zero}")
print(f"    Near-zero (< 1e-13 everywhere):   {near_zero}")
print(f"    Small but nonzero (1e-13 to 1e-10): {small_but_nonzero}")
print(f"    Moderate (>= 1e-10):              {moderate}")

# =========================================================================
# TEST 3: Rank deficiency analysis
# =========================================================================
print("\n" + "=" * 70)
print("TEST 3: RANK DEFICIENCY vs EPSILON")
print("=" * 70)
print("\nIf constraints exist, the rank should be constant across epsilon.")
print("If the noise floor is numerical, rank should increase with epsilon.")

for pot, pdir in [('1/r', '1_r'), ('1/r^2', '1_r2')]:
    print(f"\n  --- {pot} ---")
    for eps_str in EPSILONS:
        sv = np.load(os.path.join(BASE, pdir, f'eps_{eps_str}', 'sv_spectra.npy'))

        # Compute rank at various thresholds at Lagrange point
        spec = sv[LAG_ROW, LAG_COL, :]

        rank_1e10 = np.sum(spec > 1e-10)
        rank_1e13 = np.sum(spec > 1e-13)
        rank_1e15 = np.sum(spec > 1e-15)

        # Also compute the gap at index 116
        if spec[115] > 0 and spec[116] > 0:
            gap_116 = spec[115] / spec[116]
        else:
            gap_116 = float('inf')

        print(f"    eps={eps_str}: rank(>1e-10)={rank_1e10:>4d}  "
              f"rank(>1e-13)={rank_1e13:>4d}  "
              f"rank(>1e-15)={rank_1e15:>4d}  "
              f"gap@116={gap_116:.1e}")

# =========================================================================
# TEST 4: Check the 4 smallest-tier generators
# =========================================================================
print("\n" + "=" * 70)
print("TEST 4: TIER 4 (4 generators) - NEAR CONSTRAINTS?")
print("=" * 70)
print("\nTier 4 has only 4 generators (indices 112-115). These are the")
print("weakest significant generators. Are they approaching constraint status?")

for pot, pdir in [('1/r', '1_r')]:
    for eps_str in EPSILONS:
        sv = np.load(os.path.join(BASE, pdir, f'eps_{eps_str}', 'sv_spectra.npy'))
        spec = sv[LAG_ROW, LAG_COL, :]

        tier4 = spec[112:116]
        noise_top = spec[116:120]

        print(f"\n  eps={eps_str}:")
        print(f"    Tier 4 SVs [112:116]: {[f'{v:.4e}' for v in tier4]}")
        print(f"    Top noise  [116:120]: {[f'{v:.4e}' for v in noise_top]}")
        if tier4[-1] > 0 and noise_top[0] > 0:
            print(f"    Gap (SV[115]/SV[116]): {tier4[-1]/noise_top[0]:.1f}x")

# =========================================================================
# TEST 5: Build algebra and evaluate generators directly
# =========================================================================
print("\n" + "=" * 70)
print("TEST 5: DIRECT GENERATOR EVALUATION")
print("=" * 70)
print("\nBuilding symbolic algebra to evaluate each generator independently...")
print("This tests whether generators are identically zero on phase space.")

try:
    from stability_atlas import AtlasConfig, PoissonAlgebra

    config = AtlasConfig(
        potential_type='1/r',
        max_level=3,
        n_phase_samples=400,
        epsilon=1e-3,
    )

    print("  Building algebra (this takes ~60s)...")
    algebra = PoissonAlgebra(config)

    levels = algebra._levels
    names = algebra._names
    n_gen = algebra._n_generators

    print(f"  Built: {n_gen} generators")
    print(f"  Level distribution: ", end="")
    for lv in range(max(levels) + 1):
        count = sum(1 for l in levels if l == lv)
        print(f"L{lv}={count} ", end="")
    print()

    # Evaluate at Lagrange equilateral triangle
    # Equal masses at vertices of equilateral triangle with side = 1
    s = 1.0
    positions = np.array([
        [0.0, 0.0],
        [s, 0.0],
        [s / 2, s * np.sqrt(3) / 2],
    ])

    # Sample with various epsilon values
    for eps in [1e-2, 1e-3, 1e-4]:
        Z_qp, Z_u = algebra._sample_local(positions, 500, eps)
        M = algebra._evaluate(Z_qp, Z_u)  # (500, n_gen)

        # Column norms (RMS value of each generator)
        col_norms = np.linalg.norm(M, axis=0) / np.sqrt(M.shape[0])

        # Sort by norm
        order = np.argsort(-col_norms)

        # Count generators with tiny norms
        n_tiny = np.sum(col_norms < 1e-14)
        n_small = np.sum(col_norms < 1e-10)

        print(f"\n  Lagrange, eps={eps:.0e}, 500 samples:")
        print(f"    Generators with RMS < 1e-14: {n_tiny}")
        print(f"    Generators with RMS < 1e-10: {n_small}")

        # Show the smallest generators
        smallest = np.argsort(col_norms)[:20]
        print(f"    20 smallest RMS generators:")
        for idx in smallest:
            print(f"      gen[{idx:>3d}] L{levels[idx]} "
                  f"RMS={col_norms[idx]:.4e}  {names[idx][:50]}")

        # Check: are tiny-norm generators from specific levels?
        if n_tiny > 0:
            tiny_levels = [levels[i] for i in range(n_gen) if col_norms[i] < 1e-14]
            from collections import Counter
            print(f"    Tiny generators by level: {dict(Counter(tiny_levels))}")

    # Now the critical test: column-normalize and SVD
    print("\n  --- SVD after column normalization (matching atlas code) ---")
    Z_qp, Z_u = algebra._sample_local(positions, 400, 1e-4)
    M = algebra._evaluate(Z_qp, Z_u)

    norms = np.linalg.norm(M, axis=0)
    n_zero_cols = np.sum(norms < 1e-15)
    print(f"  Zero columns (norm < 1e-15): {n_zero_cols}")

    norms_safe = norms.copy()
    norms_safe[norms_safe < 1e-15] = 1.0
    M_normed = M / norms_safe

    from numpy.linalg import svd
    U, S, Vt = svd(M_normed, full_matrices=False)

    # Rank
    rank_10 = np.sum(S > 1e-10)
    rank_13 = np.sum(S > 1e-13)
    rank_15 = np.sum(S > 1e-15)

    print(f"  SVD rank (>1e-10): {rank_10}")
    print(f"  SVD rank (>1e-13): {rank_13}")
    print(f"  SVD rank (>1e-15): {rank_15}")

    # The null space of M_normed: right singular vectors with zero SVs
    null_dim = n_gen - rank_13
    print(f"  Null space dimension: {null_dim}")

    if null_dim > 0:
        # The null space vectors (rows of Vt with indices >= rank_13)
        null_vectors = Vt[rank_13:, :]  # shape (null_dim, n_gen)

        print(f"\n  Null space vectors show which generator COMBINATIONS vanish:")
        for k in range(min(10, null_dim)):
            vec = null_vectors[k, :]
            # Find dominant components
            order = np.argsort(-np.abs(vec))
            top = order[:5]
            print(f"    Null vec {k}: ", end="")
            for idx in top:
                if abs(vec[idx]) > 0.01:
                    print(f"{vec[idx]:+.3f}*g[{idx}](L{levels[idx]}) ", end="")
            print()

        # Check: are the null vectors concentrated on specific levels?
        print(f"\n  Null space weight by bracket level:")
        for lv in range(max(levels) + 1):
            lv_mask = np.array([1.0 if levels[j] == lv else 0.0
                                for j in range(n_gen)])
            weight = np.mean([np.sum(null_vectors[k, :] ** 2 * lv_mask)
                              for k in range(null_dim)])
            n_at_lv = sum(1 for l in levels if l == lv)
            print(f"    Level {lv} ({n_at_lv} gen): "
                  f"mean null weight = {weight:.4f}")

    # Check which individual generators are "almost constraints"
    # by looking at their participation in the null space
    print(f"\n  Generator participation in null space:")
    null_participation = np.zeros(n_gen)
    for k in range(null_dim):
        null_participation += null_vectors[k, :] ** 2

    # Normalize
    null_participation /= max(null_dim, 1)

    # Generators with high null participation are constraint-like
    high_null = np.argsort(-null_participation)[:20]
    print(f"  Top 20 generators by null-space participation:")
    for idx in high_null:
        print(f"    gen[{idx:>3d}] L{levels[idx]} "
              f"null_wt={null_participation[idx]:.4f}  "
              f"col_norm={norms[idx]:.4e}  {names[idx][:40]}")

    # FIRST-CLASS vs SECOND-CLASS test
    print("\n" + "=" * 70)
    print("FIRST-CLASS vs SECOND-CLASS CLASSIFICATION")
    print("=" * 70)

    # A constraint is first-class if its Poisson bracket with ALL other
    # constraints vanishes on the constraint surface.
    # For our algebraic setup, the null-space vectors define linear
    # combinations that vanish. Two such combinations C1, C2 are
    # first-class if {C1, C2} also vanishes.

    # We can test this approximately: the bracket of two null-space
    # generators should itself be in the null space.
    # The bracket matrix B[i,j] = {g_i, g_j} can be represented
    # as a structure constants tensor.

    # However, computing {g_i, g_j} for all pairs requires the
    # symbolic bracket machinery. Instead, check numerically:
    # evaluate the null-space combinations at phase-space points
    # and see if their pairwise "brackets" (via finite differences)
    # also vanish.

    # Simpler test: if the null generators are ALL from level 3,
    # they might be syzygies (algebraic identities) rather than
    # dynamical constraints.

    print(f"\n  Null-space generator level distribution:")
    null_gen_levels = []
    for k in range(null_dim):
        vec = null_vectors[k, :]
        dominant = np.argmax(np.abs(vec))
        null_gen_levels.append(levels[dominant])

    from collections import Counter
    print(f"  Dominant level of null vectors: {dict(Counter(null_gen_levels))}")

    # Key question: are null vectors mixtures across levels,
    # or concentrated within a single level?
    cross_level = 0
    single_level = 0
    for k in range(null_dim):
        vec = null_vectors[k, :]
        significant = np.abs(vec) > 0.01
        sig_levels = set(levels[j] for j in range(n_gen) if significant[j])
        if len(sig_levels) > 1:
            cross_level += 1
        else:
            single_level += 1

    print(f"  Null vectors mixing levels: {cross_level}")
    print(f"  Null vectors within single level: {single_level}")

    if cross_level > 0:
        print(f"\n  >>> CROSS-LEVEL null vectors suggest FUNCTIONAL RELATIONS")
        print(f"      (syzygies) between generators at different bracket levels.")
        print(f"      These are NOT Dirac constraints but algebraic identities.")
    if single_level > 0:
        print(f"\n  >>> SINGLE-LEVEL null vectors suggest REDUNDANCY within a")
        print(f"      bracket level. These could be Dirac constraints if they")
        print(f"      correspond to conservation laws.")

except ImportError as e:
    print(f"  Could not import algebra: {e}")
    print("  Skipping direct evaluation test.")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
