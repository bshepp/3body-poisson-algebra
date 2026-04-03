#!/usr/bin/env python3
"""
SV #116 Analytical Prediction — Phase 1 Item 1.6
==================================================

Attempt to predict the 116th singular value landscape from symbolic
generators.  Strategy:

  1. Load the 116 generators from checkpoints/level_3.pkl
  2. Lambdify only the LAST generator (index 115, the newest level-3 bracket)
  3. For each (mu, phi) on the atlas grid, construct a local triangle,
     sample a small cloud of phase-space points, evaluate gen #116,
     and compute its typical magnitude relative to the other generators
  4. Compare this "analytical prediction" to the observed SV #116

This is inherently expensive so we use a coarsened grid (20×20 instead
of 100×100) and a modest sample count.

Output goes to spectral_depth/.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from time import time

# ---- Add project root to path for imports ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HIRES_DIR = 'atlas_output_hires'
OUT_DIR = 'spectral_depth'
os.makedirs(OUT_DIR, exist_ok=True)

SPECIALS = {
    'Lagrange':  (1.0, np.pi / 3),
    'Euler':     (0.5, np.pi),
    'Isos-90':   (1.0, np.pi / 2),
}
STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]

# Coarsened grid for the expensive symbolic evaluation
N_MU_COARSE = 20
N_PHI_COARSE = 20
N_SAMPLES_PER_POINT = 500
EPSILON = 1e-3


def mark_specials(ax):
    for name, (m, p) in SPECIALS.items():
        px = np.degrees(p)
        ax.plot(px, m, '*', color='cyan', markersize=12,
                markeredgecolor='white', markeredgewidth=0.5, zorder=10)
        ax.annotate(name, (px, m), textcoords='offset points',
                    xytext=(6, 4), color='white', fontsize=8,
                    fontweight='bold', path_effects=STROKE)


def shape_to_positions(mu, phi, scale=1.0):
    """Convert (mu, phi) to triangle positions: (3, 2) array."""
    return np.array([
        [0.0, 0.0],
        [scale, 0.0],
        [mu * scale * np.cos(phi), mu * scale * np.sin(phi)],
    ])


def sample_local(positions, n, epsilon, seed=42):
    """
    Sample phase-space points near a triangle configuration.
    Returns (Z_qp, Z_u) matching exact_growth.py conventions.
    """
    rng = np.random.RandomState(seed)
    base_q = positions.flatten()  # (6,)

    Z_qp = np.zeros((n, 12))
    Z_u = np.zeros((n, 3))
    accepted = 0

    for _ in range(n * 100):
        if accepted >= n:
            break
        q = base_q + rng.randn(6) * epsilon
        p = rng.randn(6) * 0.5

        dx12 = q[0] - q[2]; dy12 = q[1] - q[3]
        dx13 = q[0] - q[4]; dy13 = q[1] - q[5]
        dx23 = q[2] - q[4]; dy23 = q[3] - q[5]

        r12 = np.sqrt(dx12**2 + dy12**2)
        r13 = np.sqrt(dx13**2 + dy13**2)
        r23 = np.sqrt(dx23**2 + dy23**2)

        if min(r12, r13, r23) < 0.1:
            continue

        Z_qp[accepted, :6] = q
        Z_qp[accepted, 6:] = p
        Z_u[accepted] = [1.0 / r12, 1.0 / r13, 1.0 / r23]
        accepted += 1

    return Z_qp[:accepted], Z_u[:accepted]


def load_and_lambdify():
    """
    Load checkpoint and lambdify the last few generators.
    Returns (eval_func, n_gens) where eval_func(Z_qp, Z_u) -> (N, n_gens).
    """
    ckpt_path = 'checkpoints/level_3.pkl'
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint {ckpt_path} not found!")
        return None, 0

    print("  Loading checkpoint...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    exprs = ckpt['exprs']
    names = ckpt['names']
    levels = ckpt['levels']
    print(f"  Loaded {len(exprs)} generators (levels 0-{max(levels)})")

    # We need ALL_VARS from exact_growth
    try:
        from exact_growth import lambdify_generators, ALL_VARS
    except ImportError:
        print("  Cannot import exact_growth — aborting lambdification")
        return None, 0

    # Try lambdifying just the last 5 generators (indices 111-115)
    # and the first 3 (level 0) for normalization
    key_indices = [0, 1, 2] + list(range(111, 116))
    key_exprs = [exprs[i] for i in key_indices]
    key_names = [names[i] for i in key_indices]
    print(f"  Lambdifying {len(key_exprs)} key generators: {key_names}")

    # Use individual lambdify to avoid column_stack dimension issues
    # (some level-0 generators may return scalars for certain inputs)
    import sympy as sp
    try:
        individual_funcs = []
        for idx_e, expr in enumerate(key_exprs):
            f = sp.lambdify(ALL_VARS, expr, modules="numpy", cse=False)
            individual_funcs.append(f)

        def evaluate(Z_qp, Z_u):
            n_pts = len(Z_qp)
            args = ([Z_qp[:, k] for k in range(12)] +
                    [Z_u[:, k] for k in range(3)])
            cols = []
            for f in individual_funcs:
                col = f(*args)
                if np.isscalar(col) or (isinstance(col, np.ndarray) and col.ndim == 0):
                    col = np.full(n_pts, float(col))
                cols.append(col)
            return np.column_stack(cols)

        return evaluate, key_indices
    except Exception as e:
        print(f"  Individual lambdification failed: {e}")

    # Fallback: try just the last generator and first generator
    print("  Trying minimal set: gen #1 + gen #116")
    try:
        import sympy as sp
        minimal = [exprs[0], exprs[115]]
        func = sp.lambdify(ALL_VARS, minimal, modules="numpy", cse=True)

        def evaluate_minimal(Z_qp, Z_u):
            args = ([Z_qp[:, i] for i in range(12)] +
                    [Z_u[:, i] for i in range(3)])
            vals = func(*args)
            return np.column_stack(vals)

        return evaluate_minimal, [0, 115]
    except Exception as e:
        print(f"  Minimal lambdification also failed: {e}")
        return None, 0


def evaluate_grid(eval_func, key_indices, mu_vals, phi_vals):
    """Evaluate generators on a coarse grid, return predicted SV116 magnitude."""
    n_mu, n_phi = len(mu_vals), len(phi_vals)
    # We want: |gen_116| / |gen_1| as proxy for normalized SV #116
    magnitude_map = np.full((n_mu, n_phi), np.nan)
    ratio_map = np.full((n_mu, n_phi), np.nan)

    # Find which column in the output corresponds to gen #116 and gen #1
    if 115 in key_indices:
        idx_116 = key_indices.index(115)
        idx_1 = key_indices.index(0) if 0 in key_indices else 0
    else:
        print("  Gen #116 not in key indices, cannot compute")
        return magnitude_map, ratio_map

    t0 = time()
    total = n_mu * n_phi
    done = 0

    for i, mu in enumerate(mu_vals):
        for j, phi in enumerate(phi_vals):
            pos = shape_to_positions(mu, phi)
            Z_qp, Z_u = sample_local(pos, N_SAMPLES_PER_POINT, EPSILON,
                                      seed=42 + i * 1000 + j)
            if len(Z_qp) < 10:
                continue

            try:
                vals = eval_func(Z_qp, Z_u)  # (N, n_key)
            except Exception as e:
                if done == 0:
                    print(f"    First-point error: {type(e).__name__}: {e}")
                continue

            if vals.ndim == 1:
                vals = vals.reshape(-1, 1)
            # RMS of each generator
            rms = np.sqrt(np.mean(vals**2, axis=0))
            magnitude_map[i, j] = rms[idx_116]
            if rms[idx_1] > 0:
                ratio_map[i, j] = rms[idx_116] / rms[idx_1]

            done += 1
            if done % 20 == 0:
                elapsed = time() - t0
                eta = elapsed / done * (total - done)
                print(f"    {done}/{total} points [{elapsed:.0f}s, ETA {eta:.0f}s]")

    return magnitude_map, ratio_map


def fig_comparison(mu_coarse, phi_coarse, ratio_map):
    """Compare analytical prediction to observed SV #116."""
    phi_deg_coarse = np.degrees(phi_coarse)

    # Load observed SV #116 from atlas
    d = os.path.join(HIRES_DIR, '1_r')
    mu_fine = np.load(os.path.join(d, 'mu_vals.npy'))
    phi_fine = np.load(os.path.join(d, 'phi_vals.npy'))
    sv = np.load(os.path.join(d, 'sv_spectra.npy'))
    rank = np.load(os.path.join(d, 'rank_map.npy'))

    sv116_obs = np.log10(sv[:, :, 115] / np.clip(sv[:, :, 0], 1e-30, None))
    sv116_obs[rank < 116] = np.nan

    # Predicted: log ratio
    predicted = np.log10(np.clip(ratio_map, 1e-30, None))

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor='white')

    # Panel 1: Observed SV #116
    ax = axes[0]
    valid_obs = sv116_obs[np.isfinite(sv116_obs)]
    vmin, vmax = np.percentile(valid_obs, 1), np.percentile(valid_obs, 99)
    im = ax.pcolormesh(np.degrees(phi_fine), mu_fine, sv116_obs,
                       cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\mu$')
    ax.set_title('Observed: log₁₀(σ₁₁₆/σ₁) from Atlas', fontsize=12)
    plt.colorbar(im, ax=ax, pad=0.02)

    # Panel 2: Predicted from generators
    ax = axes[1]
    valid_pred = predicted[np.isfinite(predicted)]
    if len(valid_pred) > 0:
        pv_min, pv_max = np.percentile(valid_pred, 1), np.percentile(valid_pred, 99)
        im = ax.pcolormesh(phi_deg_coarse, mu_coarse, predicted,
                           cmap='viridis', vmin=pv_min, vmax=pv_max, shading='auto')
        plt.colorbar(im, ax=ax, pad=0.02)
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\mu$')
    ax.set_title('Predicted: log₁₀(RMS gen₁₁₆ / RMS gen₁)', fontsize=12)

    # Panel 3: Correlation scatter
    ax = axes[2]
    # Interpolate observed to coarse grid for comparison
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((mu_fine, phi_fine), sv116_obs,
                                      method='linear', bounds_error=False,
                                      fill_value=np.nan)
    MU_c, PHI_c = np.meshgrid(mu_coarse, phi_coarse, indexing='ij')
    obs_at_coarse = interp((MU_c, PHI_c))

    both_valid = np.isfinite(obs_at_coarse) & np.isfinite(predicted)
    if both_valid.sum() > 5:
        x = obs_at_coarse[both_valid]
        y = predicted[both_valid]
        ax.scatter(x, y, s=15, alpha=0.6, c='steelblue')
        # Fit line
        coeffs = np.polyfit(x, y, 1)
        r_sq = 1 - np.sum((y - np.polyval(coeffs, x))**2) / np.sum((y - y.mean())**2)
        x_fit = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=2,
                label=f'R² = {r_sq:.3f}')
        ax.legend(fontsize=11)
    ax.set_xlabel('Observed log₁₀(σ₁₁₆/σ₁)', fontsize=11)
    ax.set_ylabel('Predicted log₁₀(RMS ratio)', fontsize=11)
    ax.set_title('Correlation: Predicted vs Observed', fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle('SV #116 Analytical Prediction vs Atlas Observation — 1/r',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = 'sv116_predicted_vs_observed.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")
    return r_sq if both_valid.sum() > 5 else None


if __name__ == '__main__':
    print("\n--- 1.6 SV #116 Analytical Prediction ---")

    # Step 1: Load and lambdify
    eval_func, key_indices = load_and_lambdify()
    if eval_func is None:
        print("\n  DEFERRED: Could not lambdify generators.")
        print("  This item requires working symbolic checkpoints and exact_growth imports.")
        sys.exit(1)

    # Step 2: Build coarse grid
    mu_fine = np.load(os.path.join(HIRES_DIR, '1_r', 'mu_vals.npy'))
    phi_fine = np.load(os.path.join(HIRES_DIR, '1_r', 'phi_vals.npy'))
    mu_coarse = np.linspace(mu_fine.min(), mu_fine.max(), N_MU_COARSE)
    phi_coarse = np.linspace(phi_fine.min(), phi_fine.max(), N_PHI_COARSE)

    # Step 3: Evaluate
    print(f"\n  Evaluating {N_MU_COARSE}x{N_PHI_COARSE} grid "
          f"({N_SAMPLES_PER_POINT} samples/point)...")
    magnitude_map, ratio_map = evaluate_grid(eval_func, key_indices,
                                              mu_coarse, phi_coarse)

    valid = np.isfinite(ratio_map).sum()
    total = ratio_map.size
    print(f"  Valid points: {valid}/{total}")

    # Step 4: Comparison figure
    r_sq = fig_comparison(mu_coarse, phi_coarse, ratio_map)
    if r_sq is not None:
        print(f"  Correlation R² = {r_sq:.3f}")
    else:
        print("  Could not compute correlation (insufficient valid points)")

    print(f"\nAll output in {OUT_DIR}/")
