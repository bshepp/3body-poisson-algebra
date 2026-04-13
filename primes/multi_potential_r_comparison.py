"""
Multi-potential comparison of coadjoint orbit spacing ratio <r>.

Question: Is the <r>=0.639 overage (above GUE(3)=0.602) universal across 
all singular potentials, or does it depend on the potential?

Loads structure constants for all available potentials and computes:
  - Coadjoint orbit frequency spacings
  - Spacing ratio <r> and variance
  - Bootstrap confidence intervals
  - Comparison table
"""

import json
import sys
import numpy as np
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── Reusable functions (from level2_spectral_analysis.py) ───────────────────

def load_structure_constants(path):
    with open(path) as f:
        data = json.load(f)
    dim = len(data)
    C = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                C[i, j, k] = float(Fraction(data[i][j][k]))
    return C


def kirillov_form(C, xi):
    return np.einsum('k,ijk->ij', xi, C)


def orbit_frequencies(C, xi):
    Omega = kirillov_form(C, xi)
    eigs = np.linalg.eigvals(Omega)
    imag_parts = np.sort(np.abs(eigs.imag))[::-1]
    freqs = []
    skip = set()
    for idx, v in enumerate(imag_parts):
        if idx in skip or v < 1e-12:
            continue
        freqs.append(v)
        for idx2 in range(idx + 1, len(imag_parts)):
            if idx2 not in skip and abs(imag_parts[idx2] - v) < 1e-10:
                skip.add(idx2)
                break
    return np.array(sorted(freqs))


def coadjoint_orbit_ensemble(C, n_samples=20000, seed=42):
    rng = np.random.RandomState(seed)
    dim = C.shape[0]
    all_freqs = []
    all_ranks = []
    for _ in range(n_samples):
        xi = rng.randn(dim)
        freqs = orbit_frequencies(C, xi)
        all_freqs.append(freqs)
        Omega = kirillov_form(C, xi)
        rank = np.linalg.matrix_rank(Omega, tol=1e-10)
        all_ranks.append(rank)
    return all_freqs, np.array(all_ranks)


def normalized_spacings(values):
    vals = np.sort(values)
    vals = vals[np.abs(vals) > 1e-12]
    if len(vals) < 3:
        return np.array([])
    spacings = np.diff(vals)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) == 0:
        return np.array([])
    mean_s = np.mean(spacings)
    if mean_s < 1e-15:
        return np.array([])
    return spacings / mean_s


def spacing_ratio(spacings):
    if len(spacings) < 2:
        return np.array([])
    ratios = []
    for i in range(len(spacings) - 1):
        lo = min(spacings[i], spacings[i + 1])
        hi = max(spacings[i], spacings[i + 1])
        if hi > 1e-15:
            ratios.append(lo / hi)
    return np.array(ratios)


def bootstrap_ci(data, stat_fn, n_boot=5000, ci=0.95, seed=123):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    n = len(data)
    stats = []
    for _ in range(n_boot):
        sample = data[rng.randint(0, n, size=n)]
        stats.append(stat_fn(sample))
    stats = np.sort(stats)
    lo = stats[int((1 - ci) / 2 * n_boot)]
    hi = stats[int((1 + ci) / 2 * n_boot)]
    return lo, hi


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    base = Path('results/algebra_structure')
    fig_dir = Path('primes/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # All available potentials — singular ones first, then others
    configs = [
        # Singular (expected: dim 17, <r> ~ 0.639)
        ('1/r',          'N3_d2_1r'),
        ('log(r)',       'N3_d2_log'),
        ('1/r²',        'N3_d2_1r2'),
        ('1/r³',        'N3_d2_1r3'),
        # Non-singular / special
        ('r²',          'N3_d2_r2'),
        ('r⁴',          'N3_d2_r4'),
        # Composites
        ('u+u²',        'N3_d2_composite_u1_2'),
        ('u⁴',          'N3_d2_composite_u4'),
        ('u+u²+u³',     'N3_d2_composite_u1_2_3'),
    ]

    potentials = {}
    for name, subdir in configs:
        path = base / subdir / 'structure_constants_exact.json'
        if path.exists():
            C = load_structure_constants(path)
            potentials[name] = C
            print(f"  Loaded {name:12s}: dim={C.shape[0]}")
        else:
            print(f"  MISSING {name:12s}: {path}")

    if not potentials:
        print("ERROR: No structure constant files found!")
        sys.exit(1)

    print(f"\n{'=' * 78}")
    print("  MULTI-POTENTIAL COADJOINT ORBIT SPACING COMPARISON")
    print(f"  n_samples=20000, seed=42")
    print(f"{'=' * 78}")

    # ── Compute statistics for each potential ──
    results = {}
    for name, C in potentials.items():
        print(f"\n── {name} (dim={C.shape[0]}) ──")
        all_freqs, all_ranks = coadjoint_orbit_ensemble(C, n_samples=20000, seed=42)

        per_sample_spacings = []
        for freqs in all_freqs:
            s = normalized_spacings(freqs)
            if len(s) > 0:
                per_sample_spacings.extend(s)
        per_sample_spacings = np.array(per_sample_spacings)

        if len(per_sample_spacings) > 50:
            var_s = np.var(per_sample_spacings)
            ratios = spacing_ratio(per_sample_spacings)
            mean_r = np.mean(ratios) if len(ratios) > 0 else float('nan')

            # Bootstrap CI on <r>
            if len(ratios) > 100:
                r_lo, r_hi = bootstrap_ci(ratios, np.mean, n_boot=5000)
                var_lo, var_hi = bootstrap_ci(per_sample_spacings, np.var, n_boot=5000)
            else:
                r_lo, r_hi = float('nan'), float('nan')
                var_lo, var_hi = float('nan'), float('nan')

            print(f"  N_spacings = {len(per_sample_spacings)}")
            print(f"  N_ratios   = {len(ratios)}")
            print(f"  var(s)     = {var_s:.6f}  95% CI: [{var_lo:.6f}, {var_hi:.6f}]")
            print(f"  <r>        = {mean_r:.6f}  95% CI: [{r_lo:.6f}, {r_hi:.6f}]")

            results[name] = {
                'dim': C.shape[0],
                'n_spacings': len(per_sample_spacings),
                'n_ratios': len(ratios),
                'var_s': var_s,
                'var_ci': (var_lo, var_hi),
                'mean_r': mean_r,
                'r_ci': (r_lo, r_hi),
                'spacings': per_sample_spacings,
                'ratios': ratios,
                'ranks': all_ranks,
            }
        else:
            print(f"  Too few spacings ({len(per_sample_spacings)}) for statistics")
            results[name] = {
                'dim': C.shape[0],
                'n_spacings': len(per_sample_spacings),
                'n_ratios': 0,
                'var_s': float('nan'),
                'mean_r': float('nan'),
                'spacings': per_sample_spacings,
                'ratios': np.array([]),
                'ranks': all_ranks,
            }

    # ── Summary table ──
    print(f"\n{'=' * 78}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 78}")
    print(f"  {'Potential':12s} {'dim':>4s} {'N_spac':>7s} {'var(s)':>10s} "
          f"{'<r>':>10s} {'95% CI(r)':>20s}")
    print(f"  {'-'*12} {'-'*4} {'-'*7} {'-'*10} {'-'*10} {'-'*20}")

    for name in potentials:
        r = results[name]
        ci_str = ""
        if not np.isnan(r.get('mean_r', float('nan'))):
            r_ci = r.get('r_ci', (float('nan'), float('nan')))
            ci_str = f"[{r_ci[0]:.4f}, {r_ci[1]:.4f}]"
        print(f"  {name:12s} {r['dim']:4d} {r['n_spacings']:7d} "
              f"{r['var_s']:10.6f} {r['mean_r']:10.6f} {ci_str:>20s}")

    # Reference values
    print(f"\n  {'Reference':12s}")
    print(f"  {'Poisson':12s} {'':4s} {'':7s} {'1.000000':>10s} {'0.3863':>10s}")
    print(f"  {'GOE':12s} {'':4s} {'':7s} {'0.2860':>10s} {'0.5359':>10s}")
    print(f"  {'GUE':12s} {'':4s} {'':7s} {'0.1780':>10s} {'0.6027':>10s}")
    print(f"  {'GUE(3)':12s} {'':4s} {'':7s} {'0.1137':>10s} {'0.6018':>10s}")
    print(f"  {'GSE':12s} {'':4s} {'':7s} {'0.1050':>10s} {'0.6762':>10s}")

    # ── Are singular potentials identical? ──
    singular_names = [n for n in ['1/r', 'log(r)', '1/r²', '1/r³'] if n in results]
    if len(singular_names) >= 2:
        print(f"\n{'=' * 78}")
        print("  SINGULAR POTENTIAL UNIVERSALITY TEST")
        print(f"{'=' * 78}")
        r_values = [results[n]['mean_r'] for n in singular_names]
        var_values = [results[n]['var_s'] for n in singular_names]
        print(f"\n  <r> values: {dict(zip(singular_names, [f'{v:.6f}' for v in r_values]))}")
        print(f"  var values: {dict(zip(singular_names, [f'{v:.6f}' for v in var_values]))}")
        r_spread = max(r_values) - min(r_values)
        var_spread = max(var_values) - min(var_values)
        print(f"\n  <r> spread across singular potentials: {r_spread:.6f}")
        print(f"  var spread across singular potentials: {var_spread:.6f}")

        # Check if all CIs overlap
        all_overlap = True
        for i, n1 in enumerate(singular_names):
            for n2 in singular_names[i+1:]:
                ci1 = results[n1].get('r_ci', (float('nan'), float('nan')))
                ci2 = results[n2].get('r_ci', (float('nan'), float('nan')))
                overlap = ci1[0] <= ci2[1] and ci2[0] <= ci1[1]
                if not overlap:
                    all_overlap = False
                    print(f"  WARNING: CIs for {n1} and {n2} do NOT overlap!")
        if all_overlap:
            print(f"  All singular potential CIs overlap — consistent with universality")

    # ── Figures ──
    print(f"\n  Generating figures...")

    # Figure 1: <r> comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names_list = list(results.keys())
    r_vals = [results[n]['mean_r'] for n in names_list]
    r_errs = []
    for n in names_list:
        ci = results[n].get('r_ci', (float('nan'), float('nan')))
        if not np.isnan(ci[0]):
            r_errs.append([results[n]['mean_r'] - ci[0], ci[1] - results[n]['mean_r']])
        else:
            r_errs.append([0, 0])
    r_errs = np.array(r_errs).T

    colors = []
    for n in names_list:
        if n in ['1/r', 'log(r)', '1/r²', '1/r³']:
            colors.append('steelblue')
        elif n in ['r²', 'r⁴']:
            colors.append('coral')
        else:
            colors.append('mediumpurple')

    bars = ax.bar(range(len(names_list)), r_vals, yerr=r_errs, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names_list)))
    ax.set_xticklabels(names_list, rotation=30, ha='right')
    ax.set_ylabel('⟨r⟩ (mean spacing ratio)')
    ax.set_title('Coadjoint orbit spacing ratio across potentials (N=3, d=2)')

    # Reference lines
    ax.axhline(0.6027, color='red', ls='--', lw=1.5, label='GUE (0.603)')
    ax.axhline(0.6018, color='red', ls=':', lw=1.0, label='GUE(3) (0.602)')
    ax.axhline(0.5359, color='green', ls='--', lw=1.5, label='GOE (0.536)')
    ax.axhline(0.6762, color='purple', ls='--', lw=1.5, label='GSE (0.676)')
    ax.axhline(0.3863, color='black', ls='--', lw=1.5, label='Poisson (0.386)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 0.85)

    plt.tight_layout()
    plt.savefig(fig_dir / 'multi_potential_r_comparison.png', dpi=200)
    plt.close()
    print(f"  Saved {fig_dir / 'multi_potential_r_comparison.png'}")

    # Figure 2: Spacing distributions overlay for singular potentials
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    s_grid = np.linspace(0, 4, 300)

    # Left: spacing distributions
    ax = axes[0]
    for name in singular_names:
        if len(results[name]['spacings']) > 50:
            ax.hist(results[name]['spacings'], bins=60, density=True, alpha=0.3,
                    label=name)
    # Theory curves
    from scipy.stats import expon
    ax.plot(s_grid, (32/np.pi**2) * s_grid**2 * np.exp(-4*s_grid**2/np.pi),
            'r-', lw=2, label='GUE')
    ax.plot(s_grid, (np.pi/2) * s_grid * np.exp(-np.pi*s_grid**2/4),
            'g-', lw=2, label='GOE')
    ax.plot(s_grid, np.exp(-s_grid), 'k--', lw=2, label='Poisson')
    ax.set_xlabel('Normalized spacing s')
    ax.set_ylabel('P(s)')
    ax.set_title('Frequency spacing distributions (singular potentials)')
    ax.set_xlim(0, 4)
    ax.legend(fontsize=8)

    # Right: ratio distributions
    ax = axes[1]
    r_grid = np.linspace(0, 1, 200)
    for name in singular_names:
        if len(results[name]['ratios']) > 50:
            ax.hist(results[name]['ratios'], bins=50, density=True, alpha=0.3,
                    label=f"{name} (⟨r⟩={results[name]['mean_r']:.4f})")
    # GUE ratio distribution: P(r) = (81/4π)(r+r²)²/(1+r+r²)^4 (Atas et al. 2013)
    P_gue_r = (81.0 / (4 * np.pi)) * (r_grid + r_grid**2)**2 / (1 + r_grid + r_grid**2)**4
    P_goe_r = (27.0 / 8) * (r_grid + r_grid**2) / (1 + r_grid + r_grid**2)**(5.0/2)
    P_poi_r = 2.0 / (1 + r_grid)**2
    ax.plot(r_grid, P_gue_r, 'r-', lw=2, label='GUE')
    ax.plot(r_grid, P_goe_r, 'g-', lw=2, label='GOE')
    ax.plot(r_grid, P_poi_r, 'k--', lw=2, label='Poisson')
    ax.set_xlabel('Spacing ratio r')
    ax.set_ylabel('P(r)')
    ax.set_title('Spacing ratio distributions (singular potentials)')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / 'singular_potential_spacing_overlay.png', dpi=200)
    plt.close()
    print(f"  Saved {fig_dir / 'singular_potential_spacing_overlay.png'}")

    print("\nDone.")


if __name__ == '__main__':
    main()
