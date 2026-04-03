#!/usr/bin/env python3
"""
Spectral Depth Mining — Phase 1 Items 1.1 + 1.2 + 1.3
=======================================================

Extracts new structure from the existing high-resolution atlas data:

1.1  Interior SV landscapes at indices 50, 80, 100, 110, 116
1.2  Spectral decay-rate maps and knee-index maps
1.3  Spectral clustering (k-means on normalized SV profiles)

All data from atlas_output_hires/{1_r, 1_r2}/sv_spectra.npy.
Output goes to spectral_depth/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib import cm

# ---------------------------------------------------------------------------
# Shared config (matches sv_landscape_viz.py conventions)
# ---------------------------------------------------------------------------
HIRES_DIR = 'atlas_output_hires'
OUT_DIR = 'spectral_depth'
os.makedirs(OUT_DIR, exist_ok=True)

SPECIALS = {
    'Lagrange':  (1.0, np.pi / 3),
    'Euler':     (0.5, np.pi),
    'Isos-90':   (1.0, np.pi / 2),
}
STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]

# SV indices to visualize (0-based): these are SV #50, 80, 100, 110, 116
SV_INDICES = [49, 79, 99, 109, 115]
SV_LABELS  = ['SV #50', 'SV #80', 'SV #100', 'SV #110', 'SV #116']

POTENTIALS = ['1_r', '1_r2']
POT_DISPLAY = {'1_r': '1/r (Newton)', '1_r2': r'1/r$^2$ (Calogero-Moser)'}


def load(potential_dir):
    d = os.path.join(HIRES_DIR, potential_dir)
    return {
        'mu':   np.load(os.path.join(d, 'mu_vals.npy')),
        'phi':  np.load(os.path.join(d, 'phi_vals.npy')),
        'rank': np.load(os.path.join(d, 'rank_map.npy')),
        'sv':   np.load(os.path.join(d, 'sv_spectra.npy')),
    }


def mark_specials(ax):
    for name, (m, p) in SPECIALS.items():
        px = np.degrees(p)
        ax.plot(px, m, '*', color='cyan', markersize=12,
                markeredgecolor='white', markeredgewidth=0.5, zorder=10)
        ax.annotate(name, (px, m), textcoords='offset points',
                    xytext=(6, 4), color='white', fontsize=8,
                    fontweight='bold', path_effects=STROKE)


def setup_heatmap(ax, phi_deg, mu, data, cmap, vmin, vmax, title, label):
    im = ax.pcolormesh(phi_deg, mu, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    mark_specials(ax)
    ax.set_xlabel(r'$\phi$ (deg)', fontsize=10)
    ax.set_ylabel(r'$\mu$', fontsize=10)
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(label, fontsize=9)
    return im


# ===================================================================
# 1.1  Interior SV Landscapes
# ===================================================================
def item_1_1(D, pot_key):
    print(f"\n--- 1.1 Interior SV Landscapes ({pot_key}) ---")
    sv, mu, phi, rank = D['sv'], D['mu'], D['phi'], D['rank']
    phi_deg = np.degrees(phi)
    mask = rank >= 116

    # Normalized log-ratio to largest SV
    normed = {}
    for idx, lbl in zip(SV_INDICES, SV_LABELS):
        ratio = np.log10(sv[:, :, idx] / np.clip(sv[:, :, 0], 1e-30, None))
        ratio[~mask] = np.nan
        normed[lbl] = ratio

    # Shared colorbar range across first 5 panels
    all_vals = np.concatenate([v[np.isfinite(v)] for v in normed.values()])
    vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)

    # 6th panel: spectral spread = SV#50 - SV#116
    spread = normed['SV #50'] - normed['SV #116']

    fig, axes = plt.subplots(2, 3, figsize=(22, 13), facecolor='white')
    for ax, (lbl, data) in zip(axes.flat[:5], normed.items()):
        setup_heatmap(ax, phi_deg, mu, data, 'viridis', vmin, vmax,
                      lbl, r'$\log_{10}(\sigma_k / \sigma_1)$')

    # Spread panel
    sp_valid = spread[np.isfinite(spread)]
    sp_vmin, sp_vmax = np.percentile(sp_valid, 1), np.percentile(sp_valid, 99)
    setup_heatmap(axes[1, 2], phi_deg, mu, spread, 'inferno', sp_vmin, sp_vmax,
                  'Spectral Spread (SV#50 − SV#116)', r'$\Delta\log_{10}$')

    fig.suptitle(f'Interior Singular Value Landscapes — {POT_DISPLAY[pot_key]}',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f'sv_landscapes_{pot_key}.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


# ===================================================================
# 1.2  Spectral Decay Rate & Knee Index
# ===================================================================
def item_1_2(D, pot_key):
    print(f"\n--- 1.2 Spectral Decay Rate ({pot_key}) ---")
    sv, mu, phi, rank = D['sv'], D['mu'], D['phi'], D['rank']
    phi_deg = np.degrees(phi)
    mask = rank >= 116

    # Mean log-slope between SV #20 and SV #110 (indices 19..109)
    log_sv = np.log10(np.clip(sv[:, :, :116], 1e-30, None))
    decay_rate = (log_sv[:, :, 109] - log_sv[:, :, 19]) / 90.0
    decay_rate[~mask] = np.nan

    # Knee index: steepest consecutive log-drop
    log_diff = -np.diff(log_sv, axis=2)  # shape (..., 115)
    knee_idx = np.argmax(log_diff, axis=2).astype(float)
    knee_idx[~mask] = np.nan

    # Spectral profiles at special points
    def nearest_idx(arr, val):
        return np.argmin(np.abs(arr - val))

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor='white')

    # Panel 1: Decay rate map
    dr_valid = decay_rate[np.isfinite(decay_rate)]
    setup_heatmap(axes[0], phi_deg, mu, decay_rate, 'RdYlBu_r',
                  np.percentile(dr_valid, 1), np.percentile(dr_valid, 99),
                  'Mean Log-Slope (SV#20→SV#110)',
                  r'$(\log\sigma_{110} - \log\sigma_{20})/90$')

    # Panel 2: Knee index map
    ki_valid = knee_idx[np.isfinite(knee_idx)]
    setup_heatmap(axes[1], phi_deg, mu, knee_idx, 'plasma',
                  np.percentile(ki_valid, 1), np.percentile(ki_valid, 99),
                  'Knee Index (steepest SV drop)',
                  'SV index of steepest drop')

    # Panel 3: Spectral profiles at special points
    ax = axes[2]
    colors_list = ['#e41a1c', '#377eb8', '#4daf4a']
    for (name, (m_val, p_val)), c in zip(SPECIALS.items(), colors_list):
        i_mu = nearest_idx(mu, m_val)
        i_phi = nearest_idx(phi, p_val)
        profile = sv[i_mu, i_phi, :116]
        ax.semilogy(np.arange(1, 117), profile, '-', color=c, linewidth=1.5, label=name)
    ax.set_xlabel('SV index', fontsize=11)
    ax.set_ylabel('Singular value', fontsize=11)
    ax.set_title('Spectral Profiles at Special Points', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Spectral Decay Analysis — {POT_DISPLAY[pot_key]}',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = f'spectral_decay_{pot_key}.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


# ===================================================================
# 1.3  Spectral Clustering
# ===================================================================
def item_1_3(D, pot_key):
    print(f"\n--- 1.3 Spectral Clustering ({pot_key}) ---")
    sv, mu, phi, rank = D['sv'], D['mu'], D['phi'], D['rank']
    phi_deg = np.degrees(phi)
    mask = rank >= 116
    n_mu, n_phi = sv.shape[:2]

    # Prepare feature matrix: log-normalize, L2-normalize
    log_sv = np.log10(np.clip(sv[:, :, :116], 1e-30, None))
    log_sv -= log_sv[:, :, 0:1]  # normalize by SV#1

    flat = log_sv.reshape(-1, 116)
    flat_mask = mask.ravel()
    features = flat[flat_mask]

    # L2-normalize rows
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    features = features / norms

    # Try sklearn, fall back to scipy
    try:
        from sklearn.cluster import KMeans
        def do_kmeans(X, k):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            return km.fit_predict(X), km.cluster_centers_
    except ImportError:
        from scipy.cluster.vq import kmeans2
        print("  (sklearn not found, using scipy.cluster.vq)")
        def do_kmeans(X, k):
            centers, labels = kmeans2(X.astype(np.float64), k, minit='++', iter=30)
            return labels, centers

    # Cluster assignment maps for k=3, 5, 7
    ks = [3, 5, 7]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='white')
    cluster_results = {}
    for ax, k in zip(axes, ks):
        labels, centers = do_kmeans(features, k)
        cluster_results[k] = (labels, centers)

        label_map = np.full(n_mu * n_phi, np.nan)
        label_map[flat_mask] = labels
        label_map = label_map.reshape(n_mu, n_phi)

        cmap = plt.get_cmap('Set1', k)
        im = ax.pcolormesh(phi_deg, mu, label_map, cmap=cmap,
                           vmin=-0.5, vmax=k - 0.5, shading='auto')
        mark_specials(ax)
        ax.set_xlabel(r'$\phi$ (deg)', fontsize=10)
        ax.set_ylabel(r'$\mu$', fontsize=10)
        ax.set_title(f'k = {k} clusters', fontsize=12)
        cb = plt.colorbar(im, ax=ax, ticks=range(k), pad=0.02)
        cb.set_label('Cluster', fontsize=9)

    fig.suptitle(f'Spectral Shape Clustering — {POT_DISPLAY[pot_key]}',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f'spectral_clusters_{pot_key}.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")

    # Cluster profile figure for k=5
    labels_5, centers_5 = cluster_results[5]
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    cmap5 = plt.get_cmap('Set1', 5)
    for c_id in range(5):
        member_mask = labels_5 == c_id
        n_members = member_mask.sum()
        median_profile = np.median(features[member_mask], axis=0)
        ax.plot(np.arange(1, 117), median_profile, '-', color=cmap5(c_id),
                linewidth=2, label=f'Cluster {c_id} (n={n_members})')
    ax.set_xlabel('SV index', fontsize=12)
    ax.set_ylabel(r'Normalized $\log_{10}(\sigma_k / \sigma_1)$', fontsize=12)
    ax.set_title(f'Median Spectral Profiles per Cluster (k=5) — {POT_DISPLAY[pot_key]}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'cluster_profiles_{pot_key}.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    for pot in POTENTIALS:
        pot_dir = os.path.join(HIRES_DIR, pot)
        if not os.path.isdir(pot_dir):
            print(f"Skipping {pot}: directory not found")
            continue
        D = load(pot)
        item_1_1(D, pot)
        item_1_2(D, pot)
        item_1_3(D, pot)

    print(f"\nAll output in {OUT_DIR}/")
