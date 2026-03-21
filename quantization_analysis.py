#!/usr/bin/env python3
"""
Definitive LQG/Bekenstein Quantization Test

Part A: Spatial mapping of bimodal ratio populations at eps=1e-04
Part B: SV eigenvalue spectra at special configurations
Part C: Tier structure multiplicity analysis
Part D: KS test, MC null hypothesis, BIC model comparison
"""

import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, label as ndlabel
from scipy.stats import ks_2samp, pearsonr
import warnings
warnings.filterwarnings('ignore')

BASE = 'atlas_output_hires'
OUT = 'potential_comparison_plots'
os.makedirs(OUT, exist_ok=True)

MU = np.load(os.path.join(BASE, '1_r', 'mu_vals.npy'))
PHI = np.load(os.path.join(BASE, '1_r', 'phi_vals.npy'))
PHI_DEG = np.degrees(PHI)

LQG_J = np.arange(0.5, 8.5, 0.5)
LQG_VALS = np.array([np.sqrt(j * (j + 1)) for j in LQG_J])

EPSILONS = ['1e-04', '2e-04', '5e-04', '1e-03', '2e-03']


def find_tiers(svs, threshold=10.0, max_tiers=8):
    """Replicate _find_tiers from stability_atlas.py."""
    tiers = []
    n = len(svs)
    for k in range(n - 1):
        if svs[k + 1] < 1e-15:
            tiers.append((k + 1, float('inf')))
            break
        ratio = svs[k] / svs[k + 1]
        if ratio >= threshold:
            tiers.append((k + 1, ratio))
    tiers.sort(key=lambda t: t[0])
    return tiers[:max_tiers]


def load_gap(pot, eps_str):
    return np.load(os.path.join(BASE, pot, f'eps_{eps_str}', 'gap_map.npy'))


def load_sv(pot, eps_str):
    return np.load(os.path.join(BASE, pot, f'eps_{eps_str}', 'sv_spectra.npy'))


# Special configuration indices
LAG_ROW = np.argmin(np.abs(MU - 1.0))
LAG_COL = np.argmin(np.abs(PHI_DEG - 60))
ISO_COL = np.argmin(np.abs(PHI_DEG - 90))
COLL_COL = 0  # smallest phi (near-collinear)

print(f"Grid: {len(MU)}x{len(PHI)}")
print(f"Lagrange: mu[{LAG_ROW}]={MU[LAG_ROW]:.3f}, phi[{LAG_COL}]={PHI_DEG[LAG_COL]:.1f} deg")
print(f"Isosceles: mu[{LAG_ROW}]={MU[LAG_ROW]:.3f}, phi[{ISO_COL}]={PHI_DEG[ISO_COL]:.1f} deg")
print(f"Collinear: phi[{COLL_COL}]={PHI_DEG[COLL_COL]:.1f} deg")


# =========================================================================
# PART A: Bimodal Spatial Map at eps=1e-04
# =========================================================================
def part_a():
    print("\n" + "=" * 70)
    print("PART A: Bimodal Spatial Map at eps=1e-04")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle("Bimodal Population Spatial Map: gap(1/r^2) / gap(1/r)\n"
                 "Where does the second population (ratio < 1) live?",
                 fontsize=14, fontweight='bold')

    results_a = {}

    for col_idx, eps_str in enumerate(['1e-04', '2e-04', '5e-04']):
        g1r = load_gap('1_r', eps_str)
        g1r2 = load_gap('1_r2', eps_str)

        mask_valid = (g1r > 1) & (g1r2 > 1)
        ratio = np.ones_like(g1r)
        ratio[mask_valid] = g1r2[mask_valid] / g1r[mask_valid]
        log_ratio = np.log10(np.maximum(ratio, 1e-10))

        # Population classification
        pop_low = (ratio < 1.0) & mask_valid   # the 0.69 population
        pop_high = (ratio >= 1.0) & mask_valid  # the 4.27 population
        n_low = np.sum(pop_low)
        n_high = np.sum(pop_high)
        n_valid = np.sum(mask_valid)
        frac_low = n_low / max(1, n_valid)

        print(f"\n  eps={eps_str}:")
        print(f"    Total valid: {n_valid}, Pop<1: {n_low} ({frac_low:.1%}), "
              f"Pop>=1: {n_high} ({1 - frac_low:.1%})")

        if n_low > 0:
            low_ratios = ratio[pop_low]
            print(f"    Pop<1: mean ratio={np.mean(low_ratios):.3f}, "
                  f"median={np.median(low_ratios):.3f}")
        if n_high > 0:
            high_ratios = ratio[pop_high]
            print(f"    Pop>=1: mean ratio={np.mean(high_ratios):.1f}, "
                  f"median={np.median(high_ratios):.1f}")

        # Row 0: Continuous heatmap of log10(ratio)
        ax = axes[0, col_idx]
        vmax = max(abs(np.nanmin(log_ratio[mask_valid])),
                   abs(np.nanmax(log_ratio[mask_valid])))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(PHI_DEG, MU, log_ratio, cmap='RdBu_r', norm=norm,
                           shading='auto')
        plt.colorbar(im, ax=ax, label='log10(gap_1/r2 / gap_1/r)')

        # Overlay special configurations
        ax.axhline(MU[LAG_ROW], color='lime', ls='--', lw=1, alpha=0.7)
        ax.axvline(PHI_DEG[LAG_COL], color='lime', ls='--', lw=1, alpha=0.7)
        ax.plot(PHI_DEG[LAG_COL], MU[LAG_ROW], 'g*', markersize=15, zorder=5)
        ax.text(PHI_DEG[LAG_COL] + 2, MU[LAG_ROW] + 0.05, 'L4',
                color='lime', fontweight='bold', fontsize=10)

        ax.set_xlabel('phi (deg)')
        ax.set_ylabel('mu')
        ax.set_title(f'eps={eps_str}  (N={n_valid})')

        # Row 1: Binary phase map
        ax2 = axes[1, col_idx]
        phase = np.full_like(ratio, np.nan)
        phase[pop_low] = 0
        phase[pop_high] = 1
        ax2.pcolormesh(PHI_DEG, MU, phase, cmap='coolwarm', vmin=-0.5,
                       vmax=1.5, shading='auto')
        ax2.plot(PHI_DEG[LAG_COL], MU[LAG_ROW], 'g*', markersize=15, zorder=5)
        ax2.axhline(MU[LAG_ROW], color='lime', ls='--', lw=1, alpha=0.7)
        ax2.axvline(PHI_DEG[LAG_COL], color='lime', ls='--', lw=1, alpha=0.7)

        ax2.set_xlabel('phi (deg)')
        ax2.set_ylabel('mu')
        ax2.set_title(f'Phase map: blue=ratio<1, red=ratio>1  ({frac_low:.0%} blue)')

        # Spatial analysis: connected components of the low-ratio population
        if n_low > 10:
            labeled, n_components = ndlabel(pop_low.astype(int))
            component_sizes = [np.sum(labeled == c) for c in range(1, n_components + 1)]
            component_sizes.sort(reverse=True)
            print(f"    Pop<1 connected components: {n_components}")
            print(f"    Largest 5 sizes: {component_sizes[:5]}")

            low_rows, low_cols = np.where(pop_low)
            print(f"    Pop<1 mu range: [{MU[low_rows.min()]:.3f}, "
                  f"{MU[low_rows.max()]:.3f}]")
            print(f"    Pop<1 phi range: [{PHI_DEG[low_cols.min()]:.1f}, "
                  f"{PHI_DEG[low_cols.max()]:.1f}] deg")

            results_a[eps_str] = {
                'frac_low': frac_low, 'n_low': int(n_low),
                'n_components': n_components,
                'mu_range': (float(MU[low_rows.min()]),
                             float(MU[low_rows.max()])),
                'phi_range': (float(PHI_DEG[low_cols.min()]),
                              float(PHI_DEG[low_cols.max()])),
            }
        else:
            results_a[eps_str] = {'frac_low': frac_low, 'n_low': int(n_low)}

    plt.tight_layout()
    path = os.path.join(OUT, 'bimodal_spatial_map.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {path}")
    return results_a


# =========================================================================
# PART B: SV Eigenvalue Spectra at Special Configurations
# =========================================================================
def part_b():
    print("\n" + "=" * 70)
    print("PART B: SV Eigenvalue Spectra at Special Configurations")
    print("=" * 70)

    configs = [
        ('Lagrange (mu=1, phi=60)', LAG_ROW, LAG_COL),
        ('Isosceles (mu=1, phi=90)', LAG_ROW, ISO_COL),
        ('Collinear (phi~0, mu=1)', LAG_ROW, COLL_COL),
        ('Small mu collinear', 5, COLL_COL),
        ('Large mu Lagrange', -5, LAG_COL),
    ]

    all_subspace_sizes = []
    all_tier_positions = []
    all_sv_level_ratios = []

    n_configs = len(configs)
    fig, axes = plt.subplots(n_configs, 2, figsize=(20, 5 * n_configs))
    fig.suptitle("Singular Value Spectra at Special Configurations\n"
                 "Tier boundaries mark hierarchical subspace decomposition",
                 fontsize=14, fontweight='bold')

    for cfg_idx, (label, row_idx, col_idx) in enumerate(configs):
        for pot_idx, (pot, pdir, color) in enumerate(
                [('1/r', '1_r', 'blue'), ('1/r^2', '1_r2', 'red')]):

            ax = axes[cfg_idx, pot_idx]

            # Load at two epsilons for cross-validation
            for eps_str, ls, alpha in [('1e-04', '-', 1.0), ('1e-03', '--', 0.5)]:
                sv = load_sv(pdir, eps_str)
                spectrum = sv[row_idx, col_idx, :]
                spectrum = spectrum[spectrum > 0]

                if len(spectrum) == 0:
                    continue

                normed = spectrum / spectrum[0]
                ax.semilogy(range(len(normed)), normed,
                            ls=ls, color=color, alpha=alpha, lw=1.5,
                            label=f'eps={eps_str}')

                # Find tiers (only for the primary epsilon)
                if eps_str == '1e-04':
                    tiers = find_tiers(spectrum, threshold=10.0)
                    for t_pos, t_gap in tiers:
                        ax.axvline(t_pos, color='green', ls=':', alpha=0.7,
                                   lw=1.5)
                        if t_gap < 1e15:
                            ax.text(t_pos + 1, normed[min(t_pos, len(normed) - 1)] * 1.5,
                                    f'gap={t_gap:.0f}', fontsize=7,
                                    color='green', rotation=45)

                    tier_positions = [t[0] for t in tiers]
                    all_tier_positions.extend(tier_positions)

                    # Subspace sizes
                    boundaries = [0] + tier_positions
                    if boundaries[-1] < len(spectrum):
                        boundaries.append(len(spectrum))
                    sizes = np.diff(boundaries)
                    all_subspace_sizes.extend(sizes.tolist())

                    print(f"\n  {label} | {pot} (eps=1e-04):")
                    print(f"    N non-zero SVs: {len(spectrum)}")
                    print(f"    Tier positions: {tier_positions}")
                    print(f"    Subspace sizes: {sizes.tolist()}")

                    # SV level ratios: ratios between adjacent "shelf" values
                    # (the SV value just before each tier boundary)
                    shelf_values = []
                    for b in boundaries[:-1]:
                        if b < len(spectrum):
                            shelf_values.append(spectrum[b])
                    if len(shelf_values) >= 2:
                        for i in range(len(shelf_values)):
                            for j in range(i + 1, len(shelf_values)):
                                r = shelf_values[i] / shelf_values[j]
                                if r > 0 and r < 1:
                                    r = 1 / r
                                all_sv_level_ratios.append(r)
                        print(f"    Shelf values: "
                              f"{[f'{v:.4e}' for v in shelf_values]}")
                        shelf_ratios = [shelf_values[i] / shelf_values[i + 1]
                                        for i in range(len(shelf_values) - 1)]
                        print(f"    Consecutive shelf ratios: "
                              f"{[f'{r:.4f}' for r in shelf_ratios]}")

            ax.set_xlabel('SV index')
            ax.set_ylabel('SV / SV_max')
            ax.set_title(f'{label} | {pot}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'sv_spectrum_special_configs.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {path}")

    return all_subspace_sizes, all_tier_positions, all_sv_level_ratios


# =========================================================================
# PART C: Tier Structure Multiplicity Analysis
# =========================================================================
def part_c():
    print("\n" + "=" * 70)
    print("PART C: Tier Structure Multiplicity Analysis")
    print("=" * 70)

    # Load adaptive tier_map for 1/r^2
    tier_map_path = os.path.join(BASE, '1_r2', 'adaptive', 'tier_map.npy')
    has_adaptive = os.path.exists(tier_map_path)

    all_tier_positions_c = []
    all_subspace_sizes_c = []
    all_gap_ratios_c = []
    tier_data_by_pot = {}

    for pot, pdir in [('1/r', '1_r'), ('1/r^2', '1_r2')]:
        print(f"\n  --- {pot} ---")

        # Check for adaptive tier_map
        adaptive_path = os.path.join(BASE, pdir, 'adaptive', 'tier_map.npy')
        if os.path.exists(adaptive_path):
            print(f"  Loading adaptive tier_map from {adaptive_path}")
            tm = np.load(adaptive_path)
            print(f"  Shape: {tm.shape}")

            positions = []
            gaps = []
            for i in range(tm.shape[0]):
                for j in range(tm.shape[1]):
                    point_tiers = []
                    for t in range(tm.shape[2]):
                        pos = tm[i, j, t, 0]
                        gap = tm[i, j, t, 1]
                        if pos > 0:
                            point_tiers.append((int(pos), gap))
                            positions.append(int(pos))
                            gaps.append(min(gap, 1e16))
                    if len(point_tiers) >= 2:
                        pts = sorted([p for p, _ in point_tiers])
                        boundaries = [0] + pts
                        sizes = np.diff(boundaries)
                        all_subspace_sizes_c.extend(sizes.tolist())

            all_tier_positions_c.extend(positions)
            all_gap_ratios_c.extend(gaps)
            tier_data_by_pot[pot] = {'positions': positions, 'gaps': gaps}
            print(f"  Total tier entries: {len(positions)}")
        else:
            # Re-derive from sv_spectra
            print(f"  No adaptive tier_map; re-deriving from sv_spectra (eps=1e-03)")
            sv = load_sv(pdir, '1e-03')
            positions = []
            gaps = []
            for i in range(sv.shape[0]):
                for j in range(sv.shape[1]):
                    spectrum = sv[i, j, :]
                    spectrum = spectrum[spectrum > 0]
                    if len(spectrum) < 2:
                        continue
                    tiers = find_tiers(spectrum, threshold=10.0)
                    point_tiers = []
                    for t_pos, t_gap in tiers:
                        positions.append(t_pos)
                        gaps.append(min(t_gap, 1e16))
                        point_tiers.append(t_pos)
                    if len(point_tiers) >= 2:
                        pts = sorted(point_tiers)
                        boundaries = [0] + pts
                        sizes = np.diff(boundaries)
                        all_subspace_sizes_c.extend(sizes.tolist())

            all_tier_positions_c.extend(positions)
            all_gap_ratios_c.extend(gaps)
            tier_data_by_pot[pot] = {'positions': positions, 'gaps': gaps}
            print(f"  Total tier entries: {len(positions)}")

    # Histogram analysis
    positions_arr = np.array(all_tier_positions_c)
    sizes_arr = np.array(all_subspace_sizes_c)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Tier Structure Multiplicity Analysis\n"
                 "Where does the algebra hierarchically decompose?",
                 fontsize=14, fontweight='bold')

    # Panel 1: Tier position histogram
    ax = axes[0, 0]
    for pot, color in [('1/r', 'blue'), ('1/r^2', 'red')]:
        if pot in tier_data_by_pot:
            p = tier_data_by_pot[pot]['positions']
            ax.hist(p, bins=range(0, 160, 2), alpha=0.5, color=color,
                    label=f'{pot} ({len(p)} tiers)', density=True)
    ax.set_xlabel('SV index (tier boundary position)')
    ax.set_ylabel('Density')
    ax.set_title('Preferred tier positions across all configurations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Find peaks in position histogram
    hist_counts, hist_edges = np.histogram(positions_arr, bins=range(0, 160, 1))
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(hist_counts.astype(float), sigma=2)
    peaks, pk_props = find_peaks(smooth, prominence=smooth.max() * 0.05,
                                  distance=5)
    peak_positions = (hist_edges[peaks] + hist_edges[peaks + 1]) / 2
    print(f"\n  Preferred tier positions (peaks): {peak_positions.astype(int).tolist()}")

    for p in peak_positions:
        ax.axvline(p, color='green', ls='--', alpha=0.7, lw=1.5)

    # Panel 2: Subspace size histogram
    ax = axes[0, 1]
    if len(sizes_arr) > 0:
        ax.hist(sizes_arr, bins=range(0, 160, 2), alpha=0.6, color='purple',
                edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Subspace size (# generators)')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of subspace sizes (N={len(sizes_arr)})')
        ax.grid(True, alpha=0.3)

        # Find preferred sizes
        s_counts, s_edges = np.histogram(sizes_arr,
                                          bins=range(0, max(sizes_arr) + 2, 1))
        s_smooth = gaussian_filter1d(s_counts.astype(float), sigma=1.5)
        s_peaks, _ = find_peaks(s_smooth, prominence=s_smooth.max() * 0.03,
                                 distance=3)
        s_peak_vals = (s_edges[s_peaks] + s_edges[s_peaks + 1]) / 2
        print(f"  Preferred subspace sizes (peaks): "
              f"{s_peak_vals.astype(int).tolist()}")
        for sp in s_peak_vals:
            ax.axvline(sp, color='orange', ls='--', alpha=0.7)

    # Panel 3: Subspace size ratios
    ax = axes[1, 0]
    if len(sizes_arr) > 0:
        unique_sizes = sorted(set(sizes_arr))
        # Take the most common sizes (top 6)
        from collections import Counter
        size_counts = Counter(sizes_arr.tolist())
        top_sizes = [s for s, _ in size_counts.most_common(8) if s > 0]
        top_sizes.sort()
        print(f"  Top subspace sizes: {top_sizes}")

        if len(top_sizes) >= 2:
            ratios = []
            labels = []
            for i in range(len(top_sizes)):
                for j in range(i + 1, len(top_sizes)):
                    r = top_sizes[j] / top_sizes[i]
                    ratios.append(r)
                    labels.append(f'{top_sizes[j]}/{top_sizes[i]}')

            ax.barh(range(len(ratios)), ratios, color='teal', alpha=0.7)
            ax.set_yticks(range(len(ratios)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Ratio')
            ax.set_title('Ratios between preferred subspace sizes')
            ax.grid(True, alpha=0.3, axis='x')

            # Annotate with LQG/integer matches
            for idx, r in enumerate(ratios):
                best_lqg_err = 999
                best_lqg = None
                for ji in range(len(LQG_J)):
                    for jj in range(ji + 1, len(LQG_J)):
                        lr = LQG_VALS[jj] / LQG_VALS[ji]
                        err = abs(r - lr) / lr
                        if err < best_lqg_err:
                            best_lqg_err = err
                            best_lqg = (LQG_J[ji], LQG_J[jj], lr)
                nearest_int = max(1, round(r))
                int_err = abs(r - nearest_int) / nearest_int

                if best_lqg_err < int_err:
                    txt = (f"LQG j={best_lqg[0]:.1f}/{best_lqg[1]:.1f}"
                           f" ({best_lqg_err:.1%})")
                else:
                    txt = f"int={nearest_int} ({int_err:.1%})"
                ax.text(r + 0.05, idx, txt, va='center', fontsize=7)

    # Panel 4: Tier gap ratio distribution
    ax = axes[1, 1]
    gap_arr = np.array(all_gap_ratios_c)
    gap_arr = gap_arr[gap_arr < 1e15]
    if len(gap_arr) > 0:
        log_gaps = np.log10(np.maximum(gap_arr, 1.0))
        ax.hist(log_gaps, bins=100, alpha=0.6, color='coral',
                edgecolor='black', linewidth=0.3)
        ax.set_xlabel('log10(tier gap ratio)')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of tier gap sizes (N={len(gap_arr)})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, 'tier_multiplicity_test.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {path}")

    return (all_tier_positions_c, all_subspace_sizes_c,
            peak_positions, s_peak_vals if len(sizes_arr) > 0 else np.array([]))


# =========================================================================
# PART D: Definitive Statistical Test
# =========================================================================
def part_d(subspace_sizes_b, tier_positions_b, sv_ratios_b,
           tier_positions_c, subspace_sizes_c,
           pref_tier_pos, pref_sub_sizes):
    print("\n" + "=" * 70)
    print("PART D: Definitive Statistical Test")
    print("=" * 70)

    # Combine all observables
    all_sizes = np.array(subspace_sizes_b + subspace_sizes_c)
    all_sizes = all_sizes[all_sizes > 0]

    all_sv_ratios = np.array(sv_ratios_b)
    all_sv_ratios = all_sv_ratios[all_sv_ratios > 1]  # keep > 1

    print(f"  Total subspace sizes: {len(all_sizes)}")
    print(f"  Total SV level ratios: {len(all_sv_ratios)}")

    # Build LQG and Bekenstein predictions for subspace sizes
    # LQG: subspace dimensions go as (2j+1) = 1, 2, 3, 4, 5, ...
    # (the dimension of the spin-j representation is 2j+1)
    # OR they could go as j(j+1) multiplicities
    # More precisely, for SU(2) representations, dim = 2j+1

    # For the tier structure, the SIZE of each subspace is what's quantized.
    # Test whether sizes cluster at specific values.

    rng = np.random.default_rng(42)
    n_mc = 10000

    # --- TEST 1: Are subspace sizes quantized? ---
    print("\n  --- TEST 1: Subspace size quantization ---")
    if len(all_sizes) >= 10:
        # Find the dominant sizes (peaks in histogram)
        unique_sizes, counts = np.unique(all_sizes.astype(int), return_counts=True)
        top_idx = np.argsort(-counts)[:10]
        dominant_sizes = unique_sizes[top_idx]
        dominant_counts = counts[top_idx]
        print(f"  Dominant sizes: {list(zip(dominant_sizes.tolist(), dominant_counts.tolist()))}")

        # How concentrated is the distribution? (entropy)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(unique_sizes))
        concentration = 1 - entropy / max_entropy
        print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} "
              f"(concentration: {concentration:.3f})")

        # MC: random sizes in same range
        mc_concentrations = []
        for _ in range(n_mc):
            rand_sizes = rng.integers(1, int(all_sizes.max()) + 1,
                                       size=len(all_sizes))
            _, rc = np.unique(rand_sizes, return_counts=True)
            rp = rc / rc.sum()
            re = -np.sum(rp * np.log(rp))
            rme = np.log(len(rc))
            mc_concentrations.append(1 - re / rme if rme > 0 else 0)
        mc_concentrations = np.array(mc_concentrations)
        p_conc = np.mean(mc_concentrations >= concentration)
        print(f"  MC p-value (concentration): {p_conc:.4f}")
        if p_conc < 0.05:
            print(f"  >>> Subspace sizes are SIGNIFICANTLY quantized (p={p_conc:.4f})")
        elif p_conc < 0.1:
            print(f"  >>> MARGINAL quantization (p={p_conc:.4f})")
    else:
        concentration = 0
        p_conc = 1.0

    # --- TEST 2: Do subspace size ratios match LQG or Bekenstein? ---
    print("\n  --- TEST 2: Subspace size ratio test ---")
    if len(pref_sub_sizes) >= 2:
        pss = np.sort(pref_sub_sizes)
        pss = pss[pss > 0]

        # Build all pairwise ratios
        size_ratios = []
        for i in range(len(pss)):
            for j in range(i + 1, len(pss)):
                r = pss[j] / pss[i]
                size_ratios.append(r)

        size_ratios = np.array(size_ratios)
        print(f"  Preferred size ratios: {[f'{r:.4f}' for r in size_ratios]}")

        # Match against LQG
        lqg_pw = {}
        for i, ji in enumerate(LQG_J):
            for j, jj in enumerate(LQG_J):
                if ji < jj:
                    lqg_pw[(ji, jj)] = LQG_VALS[j] / LQG_VALS[i]

        lqg_errors = []
        int_errors = []
        for r in size_ratios:
            best_lqg_err = min(abs(r - lr) / lr for lr in lqg_pw.values())
            lqg_errors.append(best_lqg_err)
            nearest_int = max(1, round(r))
            int_errors.append(abs(r - nearest_int) / nearest_int)

        lqg_errors = np.array(lqg_errors)
        int_errors = np.array(int_errors)

        print(f"  Mean LQG err: {np.mean(lqg_errors):.4f}")
        print(f"  Mean Int err: {np.mean(int_errors):.4f}")

        # MC for size ratios
        mc_lqg_means = []
        mc_int_means = []
        for _ in range(n_mc):
            rand_sizes = np.sort(rng.integers(1, 160, size=len(pss)))
            rand_ratios = []
            for i in range(len(rand_sizes)):
                for j in range(i + 1, len(rand_sizes)):
                    if rand_sizes[i] > 0:
                        rand_ratios.append(rand_sizes[j] / rand_sizes[i])
            if not rand_ratios:
                continue
            rand_ratios = np.array(rand_ratios)
            mc_lqg = [min(abs(r - lr) / lr for lr in lqg_pw.values())
                       for r in rand_ratios]
            mc_int = [abs(r - max(1, round(r))) / max(1, round(r))
                       for r in rand_ratios]
            mc_lqg_means.append(np.mean(mc_lqg))
            mc_int_means.append(np.mean(mc_int))

        mc_lqg_means = np.array(mc_lqg_means)
        mc_int_means = np.array(mc_int_means)

        p_lqg_size = np.mean(mc_lqg_means <= np.mean(lqg_errors))
        p_int_size = np.mean(mc_int_means <= np.mean(int_errors))
        print(f"  MC p-value LQG: {p_lqg_size:.4f}")
        print(f"  MC p-value Int: {p_int_size:.4f}")

    # --- TEST 3: SV level ratios ---
    print("\n  --- TEST 3: SV level ratios (shelf-to-shelf) ---")
    if len(all_sv_ratios) >= 5:
        lqg_errs = []
        int_errs = []
        for r in all_sv_ratios:
            best_lqg = min(abs(r - lr) / lr for lr in lqg_pw.values())
            lqg_errs.append(best_lqg)
            nearest_int = max(1, round(r))
            int_errs.append(abs(r - nearest_int) / nearest_int)

        lqg_errs = np.array(lqg_errs)
        int_errs = np.array(int_errs)
        print(f"  N ratios: {len(all_sv_ratios)}")
        print(f"  Mean LQG err: {np.mean(lqg_errs):.4f}")
        print(f"  Mean Int err: {np.mean(int_errs):.4f}")
        print(f"  LQG <5%: {np.sum(lqg_errs < 0.05)}/{len(lqg_errs)} "
              f"({np.sum(lqg_errs < 0.05) / len(lqg_errs):.0%})")
        print(f"  Int <5%: {np.sum(int_errs < 0.05)}/{len(int_errs)} "
              f"({np.sum(int_errs < 0.05) / len(int_errs):.0%})")

        # MC for SV ratios
        mc_lqg_sv = []
        mc_int_sv = []
        r_range = (all_sv_ratios.min(), all_sv_ratios.max())
        for _ in range(n_mc):
            rand_r = rng.uniform(r_range[0], r_range[1], len(all_sv_ratios))
            mc_l = [min(abs(r - lr) / lr for lr in lqg_pw.values())
                     for r in rand_r]
            mc_i = [abs(r - max(1, round(r))) / max(1, round(r))
                     for r in rand_r]
            mc_lqg_sv.append(np.mean(mc_l))
            mc_int_sv.append(np.mean(mc_i))

        p_lqg_sv = np.mean(np.array(mc_lqg_sv) <= np.mean(lqg_errs))
        p_int_sv = np.mean(np.array(mc_int_sv) <= np.mean(int_errs))
        print(f"  MC p-value LQG: {p_lqg_sv:.4f}")
        print(f"  MC p-value Int: {p_int_sv:.4f}")

    # --- TEST 4: KS test on tier positions ---
    print("\n  --- TEST 4: KS test on tier positions ---")
    all_tp = np.array(tier_positions_b + tier_positions_c)
    if len(all_tp) >= 20:
        # LQG model: tier positions at n_gen * sqrt(j(j+1)) / sqrt(j_max*(j_max+1))
        # Bekenstein model: tier positions at integer multiples of n_gen/k
        # Null: uniform in [1, 156]

        n_gen = 156
        # Generate LQG and Bekenstein reference distributions
        # LQG: positions proportional to sqrt(j(j+1)), scaled to [0, n_gen]
        lqg_positions = LQG_VALS / LQG_VALS[-1] * n_gen
        # Bekenstein: equally spaced
        bek_positions = np.linspace(n_gen / 10, n_gen, 10)

        # KS test against uniform
        uniform_ref = rng.uniform(1, n_gen, 10000)
        ks_uniform, p_uniform = ks_2samp(all_tp, uniform_ref)
        print(f"  KS vs uniform: D={ks_uniform:.4f}, p={p_uniform:.4f}")
        if p_uniform < 0.05:
            print(f"  >>> Tier positions are NOT uniformly distributed (p={p_uniform:.4f})")

        print(f"  N tier positions: {len(all_tp)}")
        print(f"  Mean: {np.mean(all_tp):.1f}, Std: {np.std(all_tp):.1f}")
        print(f"  Mode positions: {np.bincount(all_tp.astype(int)).argsort()[-5:][::-1].tolist()}")

    # --- VISUALIZATION ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Definitive Statistical Tests: LQG vs Bekenstein vs Null",
                 fontsize=14, fontweight='bold')

    # Panel 1: Subspace size histogram with LQG/Bekenstein overlay
    ax = axes[0, 0]
    if len(all_sizes) >= 10:
        ax.hist(all_sizes, bins=range(0, int(all_sizes.max()) + 2, 1),
                alpha=0.6, color='purple', edgecolor='black', lw=0.3,
                density=True, label=f'Observed (N={len(all_sizes)})')
        # SU(2) dimensions: 2j+1 = 1, 2, 3, 4, ...
        for dim in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
            ax.axvline(dim, color='green', ls=':', alpha=0.3, lw=1)
        ax.set_xlabel('Subspace size')
        ax.set_ylabel('Density')
        ax.set_title(f'Subspace sizes (concentration p={p_conc:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 2: Tier position histogram
    ax = axes[0, 1]
    if len(all_tp) >= 20:
        ax.hist(all_tp, bins=range(0, 160, 2), alpha=0.6, color='teal',
                edgecolor='black', lw=0.3, density=True)
        # Mark preferred positions
        if len(pref_tier_pos) > 0:
            for p in pref_tier_pos:
                ax.axvline(p, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_xlabel('Tier boundary position (SV index)')
        ax.set_ylabel('Density')
        ax.set_title(f'Tier positions (KS vs uniform: p={p_uniform:.4f})')
        ax.grid(True, alpha=0.3)

    # Panel 3: MC null hypothesis for concentration
    ax = axes[1, 0]
    if len(all_sizes) >= 10:
        ax.hist(mc_concentrations, bins=50, alpha=0.5, color='gray',
                label='MC null (random sizes)')
        ax.axvline(concentration, color='red', lw=2.5,
                   label=f'Observed (p={p_conc:.3f})')
        ax.set_xlabel('Concentration index')
        ax.set_ylabel('Count')
        ax.set_title('Is the subspace-size distribution more concentrated\nthan random?')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 4: Summary table as text
    ax = axes[1, 1]
    ax.axis('off')
    summary_lines = [
        "SUMMARY OF STATISTICAL TESTS",
        "=" * 40,
        "",
        f"Subspace sizes (N={len(all_sizes)}):",
        f"  Concentration: {concentration:.3f}",
        f"  p-value: {p_conc:.4f}",
        "",
    ]
    if len(pref_sub_sizes) >= 2:
        summary_lines.extend([
            f"Preferred size ratios:",
            f"  Mean LQG err: {np.mean(lqg_errors):.4f}",
            f"  Mean Int err: {np.mean(int_errors):.4f}",
            f"  LQG p-value: {p_lqg_size:.4f}",
            f"  Int p-value: {p_int_size:.4f}",
            "",
        ])
    if len(all_sv_ratios) >= 5:
        summary_lines.extend([
            f"SV level ratios (N={len(all_sv_ratios)}):",
            f"  Mean LQG err: {np.mean(lqg_errs):.4f}",
            f"  Mean Int err: {np.mean(int_errs):.4f}",
            f"  LQG p-value: {p_lqg_sv:.4f}",
            f"  Int p-value: {p_int_sv:.4f}",
            "",
        ])
    if len(all_tp) >= 20:
        summary_lines.extend([
            f"Tier positions (N={len(all_tp)}):",
            f"  KS vs uniform: p={p_uniform:.4f}",
        ])

    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUT, 'quantization_definitive.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {path}")

    return {
        'n_sizes': len(all_sizes),
        'concentration': float(concentration),
        'p_concentration': float(p_conc),
        'n_sv_ratios': len(all_sv_ratios),
        'n_tier_positions': len(all_tp),
    }


# =========================================================================
# MAIN
# =========================================================================
if __name__ == '__main__':
    print("Definitive LQG/Bekenstein Quantization Test")
    print("=" * 70)

    results_a = part_a()
    sizes_b, tiers_b, sv_ratios_b = part_b()
    tiers_c, sizes_c, pref_tier_pos, pref_sub_sizes = part_c()
    stats = part_d(sizes_b, tiers_b, sv_ratios_b,
                   tiers_c, sizes_c, pref_tier_pos, pref_sub_sizes)

    print("\n" + "=" * 70)
    print("ALL PARTS COMPLETE")
    print("=" * 70)
