"""
Finite-N GUE comparison for bracket tensor spectral statistics.

Tests whether the level-2 subalgebra's spacing variance (0.117) and mean ratio
(0.639) are finite-size artifacts by comparing against GUE(N) matrices processed
through the exact same statistical pipeline.

Key insight: each 1/r coadjoint orbit produces only 3 eigenvalues → 2 spacings.
Mean-unfolding forces s1 + s2 = 2, a deterministic constraint that suppresses
variance below the asymptotic GUE value of 0.178.

Ensembles compared:
  1. GUE(N) for N = 3, 4, 5, 6, 8, 10, 15, 17, 20, 50, 100
  2. Random 6×6 skew-symmetric (3 frequencies — matches orbit structure)
  3. Random 17×17 skew-symmetric (8 frequencies)
  4. Rank-6 constrained 17×17 skew-symmetric (3 frequencies, structure-matched)
"""

import json
import numpy as np
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── Spacing statistics (copied from level2_spectral_analysis.py) ────────────

def normalized_spacings(values):
    """Unfold and normalize spacings to mean 1."""
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
    """r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})."""
    if len(spacings) < 2:
        return np.array([])
    ratios = []
    for i in range(len(spacings) - 1):
        lo = min(spacings[i], spacings[i + 1])
        hi = max(spacings[i], spacings[i + 1])
        if hi > 1e-15:
            ratios.append(lo / hi)
    return np.array(ratios)

def wigner_gue(s):
    """GUE Wigner surmise: P(s) = (32/pi^2) s^2 exp(-4s^2/pi)."""
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

def wigner_goe(s):
    """GOE Wigner surmise: P(s) = (pi/2) s exp(-pi s^2/4)."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

def wigner_gse(s):
    """GSE Wigner surmise: P(s) = (2^18 / 3^6 pi^3) s^4 exp(-64 s^2 / 9 pi)."""
    return (2**18 / (3**6 * np.pi**3)) * s**4 * np.exp(-64 * s**2 / (9 * np.pi))

def poisson_spacing(s):
    """Poisson: P(s) = exp(-s)."""
    return np.exp(-s)


# ─── Ensemble samplers ───────────────────────────────────────────────────────

def sample_gue(N, rng):
    """Sample one GUE(N) matrix. Returns sorted eigenvalues."""
    G = (rng.randn(N, N) + 1j * rng.randn(N, N)) / np.sqrt(2)
    H = (G + G.conj().T) / (2 * np.sqrt(N))
    return np.sort(np.linalg.eigvalsh(H))

def sample_skew_symmetric(N, rng):
    """Sample random N×N real skew-symmetric matrix.
    Returns positive eigenvalue magnitudes (frequencies)."""
    G = rng.randn(N, N)
    A = (G - G.T) / np.sqrt(2)
    eigs = np.linalg.eigvals(A)
    imag_parts = np.abs(eigs.imag)
    # Take unique positive frequencies (each appears as ±iλ)
    imag_parts = np.sort(imag_parts)[::-1]
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

def sample_rank_constrained_skew(dim, rank, rng):
    """Sample a dim×dim skew-symmetric matrix of given rank.
    Construct as B @ M @ B^T where B is dim×rank, M is rank×rank skew-symmetric."""
    B = rng.randn(dim, rank)
    G = rng.randn(rank, rank)
    M = (G - G.T) / np.sqrt(2)
    A = B @ M @ B.T
    eigs = np.linalg.eigvals(A)
    imag_parts = np.abs(eigs.imag)
    imag_parts = np.sort(imag_parts)[::-1]
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


# ─── Ensemble statistics collection ─────────────────────────────────────────

def collect_spacing_stats(eigenvalue_lists):
    """Given list of eigenvalue arrays (one per sample), compute pooled
    spacing statistics using the same pipeline as level2_spectral_analysis."""
    all_spacings = []
    all_ratios = []
    for eigs in eigenvalue_lists:
        sp = normalized_spacings(eigs)
        if len(sp) > 0:
            all_spacings.extend(sp)
            r = spacing_ratio(sp)
            if len(r) > 0:
                all_ratios.extend(r)
    spacings = np.array(all_spacings)
    ratios = np.array(all_ratios)
    return {
        'var': np.var(spacings) if len(spacings) > 0 else np.nan,
        'mean_ratio': np.mean(ratios) if len(ratios) > 0 else np.nan,
        'n_spacings': len(spacings),
        'n_ratios': len(ratios),
        'spacings': spacings,
        'ratios': ratios,
    }


# ─── Load bracket tensor data ────────────────────────────────────────────────

def load_bracket_tensor_stats(results_dir):
    """Load 1/r structure constants and recompute orbit frequency statistics."""
    path = Path(results_dir) / 'algebra_structure' / '1_r_structure_constants.json'
    if not path.exists():
        print(f"  [!] Structure constants not found at {path}")
        print("      Using saved values: var=0.117, <r>=0.639")
        return None

    with open(path) as f:
        data = json.load(f)
    dim = len(data)
    C = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                C[i, j, k] = float(Fraction(data[i][j][k]))

    # Sample coadjoint orbits
    rng = np.random.RandomState(42)
    all_freqs = []
    for _ in range(20000):
        xi = rng.randn(dim)
        Omega = np.einsum('k,ijk->ij', xi, C)
        eigs = np.linalg.eigvals(Omega)
        imag_parts = np.abs(eigs.imag)
        imag_parts = np.sort(imag_parts)[::-1]
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
        all_freqs.append(np.array(sorted(freqs)))

    return collect_spacing_stats(all_freqs)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    n_samples = 20000
    rng = np.random.RandomState(42)
    fig_dir = Path(__file__).parent / 'figures'
    fig_dir.mkdir(exist_ok=True)
    results_dir = Path(__file__).parent.parent / 'results'

    # ── Phase 0: Bracket tensor (load or use saved values) ───────────────
    print("=" * 70)
    print("FINITE-N GUE COMPARISON")
    print("=" * 70)
    print()
    print("Phase 0: Loading bracket tensor data...")
    bt = load_bracket_tensor_stats(results_dir)
    if bt is not None:
        bt_var = bt['var']
        bt_ratio = bt['mean_ratio']
        print(f"  Bracket tensor (recomputed): var = {bt_var:.4f}, <r> = {bt_ratio:.4f}")
        print(f"  ({bt['n_spacings']} spacings, {bt['n_ratios']} ratios)")
    else:
        bt_var = 0.117
        bt_ratio = 0.639
        bt = None

    # ── Phase 1: GUE(N) scaling ──────────────────────────────────────────
    print()
    print("Phase 1: GUE(N) eigenvalue spacing statistics")
    print("-" * 50)

    gue_Ns = [3, 4, 5, 6, 8, 10, 15, 17, 20, 50, 100]
    gue_results = {}

    for N in gue_Ns:
        eig_lists = [sample_gue(N, rng) for _ in range(n_samples)]
        stats = collect_spacing_stats(eig_lists)
        gue_results[N] = stats
        print(f"  GUE({N:3d}):  var = {stats['var']:.4f},  <r> = {stats['mean_ratio']:.4f}"
              f"  ({stats['n_spacings']} spacings)")

    # ── Phase 2: Skew-symmetric ensembles ────────────────────────────────
    print()
    print("Phase 2: Random skew-symmetric matrix ensembles")
    print("-" * 50)

    skew_results = {}

    # 6×6 skew-symmetric → 3 frequencies (matches 1/r orbit structure)
    eig_lists = [sample_skew_symmetric(6, rng) for _ in range(n_samples)]
    stats = collect_spacing_stats(eig_lists)
    skew_results['skew_6'] = stats
    print(f"  Skew(6×6):      var = {stats['var']:.4f},  <r> = {stats['mean_ratio']:.4f}"
          f"  ({stats['n_spacings']} spacings)")

    # 17×17 skew-symmetric → 8 frequencies
    eig_lists = [sample_skew_symmetric(17, rng) for _ in range(n_samples)]
    stats = collect_spacing_stats(eig_lists)
    skew_results['skew_17'] = stats
    print(f"  Skew(17×17):    var = {stats['var']:.4f},  <r> = {stats['mean_ratio']:.4f}"
          f"  ({stats['n_spacings']} spacings)")

    # ── Phase 3: Rank-constrained skew-symmetric ─────────────────────────
    print()
    print("Phase 3: Rank-constrained skew-symmetric (rank 6 in dim 17)")
    print("-" * 50)

    eig_lists = [sample_rank_constrained_skew(17, 6, rng) for _ in range(n_samples)]
    stats = collect_spacing_stats(eig_lists)
    skew_results['skew_17_rank6'] = stats
    print(f"  Skew(17,rank6): var = {stats['var']:.4f},  <r> = {stats['mean_ratio']:.4f}"
          f"  ({stats['n_spacings']} spacings)")

    # ── Comparison table ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Ensemble':<30s}  {'var(s)':>8s}  {'<r>':>8s}  {'n_spacings':>10s}")
    print("-" * 62)
    print(f"{'Bracket tensor (1/r)':<30s}  {bt_var:8.4f}  {bt_ratio:8.4f}  {'—':>10s}")
    print()
    for N in gue_Ns:
        s = gue_results[N]
        print(f"{'GUE(' + str(N) + ')':<30s}  {s['var']:8.4f}  {s['mean_ratio']:8.4f}  {s['n_spacings']:10d}")
    print()
    for label, key in [('Skew(6×6)', 'skew_6'),
                        ('Skew(17×17)', 'skew_17'),
                        ('Skew(17,rank 6)', 'skew_17_rank6')]:
        s = skew_results[key]
        print(f"{label:<30s}  {s['var']:8.4f}  {s['mean_ratio']:8.4f}  {s['n_spacings']:10d}")
    print()
    print(f"{'Asymptotic GUE (β=2)':<30s}  {'0.1780':>8s}  {'0.6030':>8s}")
    print(f"{'Asymptotic GSE (β=4)':<30s}  {'0.1050':>8s}  {'0.6760':>8s}")
    print(f"{'Asymptotic GOE (β=1)':<30s}  {'0.2860':>8s}  {'0.5360':>8s}")
    print(f"{'Poisson':<30s}  {'1.0000':>8s}  {'0.3860':>8s}")
    print()

    # ── Figure 1: Spacing variance vs N ──────────────────────────────────
    print("Generating Figure 1: Spacing variance vs N...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Ns = np.array(gue_Ns)
    vars_gue = np.array([gue_results[N]['var'] for N in gue_Ns])
    ratios_gue = np.array([gue_results[N]['mean_ratio'] for N in gue_Ns])

    # Left: var(s) vs N
    ax1.plot(Ns, vars_gue, 'o-', color='steelblue', label='GUE(N)', markersize=6, zorder=5)
    ax1.axhline(bt_var, color='red', ls='--', lw=2, label=f'Bracket tensor ({bt_var:.3f})')
    ax1.axhline(0.178, color='steelblue', ls=':', lw=1.5, alpha=0.6, label='GUE ∞ (0.178)')
    ax1.axhline(0.105, color='purple', ls=':', lw=1.5, alpha=0.6, label='GSE ∞ (0.105)')
    ax1.axhline(0.286, color='green', ls=':', lw=1.5, alpha=0.4, label='GOE ∞ (0.286)')

    # Mark GUE(3) and Skew results
    s6 = skew_results['skew_6']
    sr6 = skew_results['skew_17_rank6']
    ax1.plot(3, s6['var'], 's', color='orange', markersize=10, zorder=6,
             label=f'Skew(6×6) ({s6["var"]:.3f})')
    ax1.plot(3, sr6['var'], 'D', color='darkred', markersize=10, zorder=6,
             label=f'Skew(17,rank6) ({sr6["var"]:.3f})')

    ax1.set_xlabel('N (matrix size / number of eigenvalues)', fontsize=12)
    ax1.set_ylabel('Spacing variance var(s)', fontsize=12)
    ax1.set_title('Finite-size effect on spacing variance', fontsize=13)
    ax1.set_xscale('log')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.4)

    # Right: <r> vs N
    ax2.plot(Ns, ratios_gue, 'o-', color='steelblue', label='GUE(N)', markersize=6, zorder=5)
    ax2.axhline(bt_ratio, color='red', ls='--', lw=2, label=f'Bracket tensor ({bt_ratio:.3f})')
    ax2.axhline(0.603, color='steelblue', ls=':', lw=1.5, alpha=0.6, label='GUE ∞ (0.603)')
    ax2.axhline(0.676, color='purple', ls=':', lw=1.5, alpha=0.6, label='GSE ∞ (0.676)')
    ax2.axhline(0.536, color='green', ls=':', lw=1.5, alpha=0.4, label='GOE ∞ (0.536)')

    ax2.plot(3, s6['mean_ratio'], 's', color='orange', markersize=10, zorder=6,
             label=f'Skew(6×6) ({s6["mean_ratio"]:.3f})')
    ax2.plot(3, sr6['mean_ratio'], 'D', color='darkred', markersize=10, zorder=6,
             label=f'Skew(17,rank6) ({sr6["mean_ratio"]:.3f})')

    ax2.set_xlabel('N (matrix size / number of eigenvalues)', fontsize=12)
    ax2.set_ylabel('Mean spacing ratio <r>', fontsize=12)
    ax2.set_title('Finite-size effect on mean ratio', fontsize=13)
    ax2.set_xscale('log')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 0.8)

    plt.tight_layout()
    fig.savefig(fig_dir / 'finite_n_gue_scaling.png', dpi=200, bbox_inches='tight')
    print(f"  → Saved {fig_dir / 'finite_n_gue_scaling.png'}")
    plt.close()

    # ── Figure 2: Spacing distribution overlay ───────────────────────────
    print("Generating Figure 2: Spacing distribution overlay...")
    fig, ax = plt.subplots(figsize=(10, 6))

    s_theory = np.linspace(0, 4, 500)

    # Bracket tensor histogram (if available)
    if bt is not None and len(bt['spacings']) > 0:
        ax.hist(bt['spacings'], bins=80, range=(0, 4), density=True,
                alpha=0.4, color='red', label='Bracket tensor (1/r)')

    # GUE(3) histogram
    g3 = gue_results[3]
    ax.hist(g3['spacings'], bins=80, range=(0, 4), density=True,
            alpha=0.3, color='steelblue', label='GUE(3)')

    # Skew(17, rank 6) histogram
    ax.hist(sr6['spacings'], bins=80, range=(0, 4), density=True,
            alpha=0.3, color='darkred', label='Skew(17, rank 6)')

    # Theory curves
    ax.plot(s_theory, wigner_gue(s_theory), 'b-', lw=2, label='Wigner GUE (N→∞)')
    ax.plot(s_theory, wigner_gse(s_theory), 'm--', lw=1.5, label='Wigner GSE (N→∞)')
    ax.plot(s_theory, wigner_goe(s_theory), 'g:', lw=1.5, label='Wigner GOE (N→∞)')
    ax.plot(s_theory, poisson_spacing(s_theory), 'k:', lw=1.5, alpha=0.5, label='Poisson')

    ax.set_xlabel('Normalized spacing s', fontsize=13)
    ax.set_ylabel('P(s)', fontsize=13)
    ax.set_title('Spacing distributions: bracket tensor vs random matrix ensembles', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / 'finite_n_spacing_overlay.png', dpi=200, bbox_inches='tight')
    print(f"  → Saved {fig_dir / 'finite_n_spacing_overlay.png'}")
    plt.close()

    # ── Figure 3: Ratio distributions ────────────────────────────────────
    print("Generating Figure 3: Ratio distribution overlay...")
    fig, ax = plt.subplots(figsize=(10, 6))

    if bt is not None and len(bt['ratios']) > 0:
        ax.hist(bt['ratios'], bins=60, range=(0, 1), density=True,
                alpha=0.4, color='red', label='Bracket tensor (1/r)')

    ax.hist(g3['ratios'], bins=60, range=(0, 1), density=True,
            alpha=0.3, color='steelblue', label='GUE(3)')

    ax.hist(sr6['ratios'], bins=60, range=(0, 1), density=True,
            alpha=0.3, color='darkred', label='Skew(17, rank 6)')

    # Exact ratio distributions
    r_vals = np.linspace(0.01, 0.99, 200)
    # GUE: P(r) = (81/4π)(r+r²)² / (1+r+r²)^4  [Atas et al. 2013]
    p_gue_r = (81 / (4 * np.pi)) * (r_vals + r_vals**2)**2 / (1 + r_vals + r_vals**2)**4
    p_poisson_r = 2 / (1 + r_vals)**2
    ax.plot(r_vals, p_gue_r, 'b-', lw=2, label='GUE (N→∞)')
    ax.plot(r_vals, p_poisson_r, 'k:', lw=1.5, alpha=0.5, label='Poisson')

    ax.set_xlabel('Spacing ratio r', fontsize=13)
    ax.set_ylabel('P(r)', fontsize=13)
    ax.set_title('Spacing ratio distributions', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / 'finite_n_ratio_overlay.png', dpi=200, bbox_inches='tight')
    print(f"  → Saved {fig_dir / 'finite_n_ratio_overlay.png'}")
    plt.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    g3_var = gue_results[3]['var']
    g3_ratio = gue_results[3]['mean_ratio']
    g100_var = gue_results[100]['var']
    print(f"  Bracket tensor:      var = {bt_var:.4f},  <r> = {bt_ratio:.4f}")
    print(f"  GUE(3):              var = {g3_var:.4f},  <r> = {g3_ratio:.4f}")
    print(f"  GUE(100):            var = {g100_var:.4f}  (sanity → 0.178)")
    print(f"  Skew(6×6):           var = {s6['var']:.4f},  <r> = {s6['mean_ratio']:.4f}")
    print(f"  Skew(17,rank6):      var = {sr6['var']:.4f},  <r> = {sr6['mean_ratio']:.4f}")
    print()
    delta_gue3 = abs(bt_var - g3_var)
    print(f"  |var(bracket) - var(GUE(3))| = {delta_gue3:.4f}")
    if delta_gue3 < 0.02:
        print("  → Bracket tensor variance is CONSISTENT with GUE(3) finite-size effect")
        print("    The 'GSE lean' is a finite-size artifact, not a symmetry class signal.")
    else:
        print(f"  → Bracket tensor variance DIFFERS from GUE(3) by {delta_gue3:.4f}")
        print("    The Lie algebra structure imposes correlations beyond random matrix universality.")
    print()


if __name__ == '__main__':
    main()
