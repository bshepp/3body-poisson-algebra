"""
Spectral analysis of the level-2 Poisson subalgebra for Hilbert-Pólya connection.

The level-2 subalgebra L for the 3-body problem (N=3, d=2) is:
  - 1/r potential: dim 17, nilpotent class 3 (universal for all singular potentials)
  - r^2 harmonic:  dim 15, near-semisimple (structurally opposite)

Key spectral objects computed:
  1. Adjoint representation matrices and their spectra
  2. Killing form eigenvalues
  3. Coadjoint orbit frequencies (Kirillov form) — THE main spectral invariant
     for nilpotent algebras
  4. Level spacing statistics compared to GUE/GOE/Poisson
  5. Invariant bilinear forms
  6. Central extension (cocycle) analysis
"""

import json
import numpy as np
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg


# ─── Data loading ────────────────────────────────────────────────────────────

def load_structure_constants(path):
    """Load C[i,j,k] from JSON.  [e_i, e_j] = sum_k C[i,j,k] e_k.
    Handles rational string entries like '1/2', '-3/2'."""
    with open(path) as f:
        data = json.load(f)
    dim = len(data)
    C = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                C[i, j, k] = float(Fraction(data[i][j][k]))
    return C


# ─── Algebraic verification ─────────────────────────────────────────────────

def verify_antisymmetry(C):
    return np.max(np.abs(C + C.transpose(1, 0, 2)))

def verify_jacobi(C):
    """Jacobi: sum_m (C[i,j,m]*C[m,k,l] + cyclic) = 0 for all i,j,k,l."""
    dim = C.shape[0]
    # Vectorized: J[i,j,k,l] = sum_m C[i,j,m]*C[m,k,l] + C[j,k,m]*C[m,i,l] + C[k,i,m]*C[m,j,l]
    # = (C[i,j,:] @ C[:,:,:])[k,l] etc.
    # C[i,j,:] is a dim-vector; C[:, k, l] contracted with it gives the Jacobi term
    J = np.einsum('ijm,mkl->ijkl', C, C) \
      + np.einsum('jkm,mil->ijkl', C, C) \
      + np.einsum('kim,mjl->ijkl', C, C)
    return np.max(np.abs(J))


# ─── Algebraic structure ────────────────────────────────────────────────────

def killing_form(C):
    """B[i,j] = Tr(ad(e_i) @ ad(e_j))."""
    dim = C.shape[0]
    B = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            B[i, j] = np.trace(C[i] @ C[j])
    return B

def compute_center_dim(C):
    """dim Z(L) where Z(L) = {x : [x, y] = 0 for all y}."""
    dim = C.shape[0]
    # x in center iff sum_j alpha_j C[j,i,k] = 0 for all i,k
    # Reshape C as (dim*dim, dim) matrix acting on alpha
    M = C.transpose(1, 2, 0).reshape(dim * dim, dim)
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    return np.sum(S < 1e-10)

def compute_derived_series(C):
    """D^0=L, D^{k+1}=[D^k, D^k].  Returns list of dimensions."""
    dim = C.shape[0]
    basis = np.eye(dim)
    series = [dim]
    for _ in range(dim):
        n = basis.shape[0]
        if n < 2:
            series.append(0)
            break
        brackets = []
        for i in range(n):
            for j in range(i + 1, n):
                v = np.einsum('a,b,abk->k', basis[i], basis[j], C)
                brackets.append(v)
        if not brackets:
            series.append(0)
            break
        M = np.array(brackets)
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        rank = int(np.sum(S > 1e-10))
        if rank == 0:
            series.append(0)
            break
        basis = Vt[:rank]
        series.append(rank)
        if rank == series[-2]:
            break
    return series

def compute_lower_central_series(C):
    """C^1=L, C^{k+1}=[L, C^k].  Returns list of dimensions."""
    dim = C.shape[0]
    full_basis = np.eye(dim)
    current = np.eye(dim)
    series = [dim]
    for _ in range(dim):
        brackets = []
        for i in range(full_basis.shape[0]):
            for j in range(current.shape[0]):
                v = np.einsum('a,b,abk->k', full_basis[i], current[j], C)
                brackets.append(v)
        if not brackets:
            series.append(0)
            break
        M = np.array(brackets)
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        rank = int(np.sum(S > 1e-10))
        if rank == 0:
            series.append(0)
            break
        current = Vt[:rank]
        series.append(rank)
        if rank == series[-2]:
            break
    return series

def invariant_form_dimension(C):
    """Dimension of space of ad-invariant symmetric bilinear forms."""
    dim = C.shape[0]
    # B([x,y],z) + B(y,[x,z]) = 0 for all x,y,z
    # sum_m C[i,j,m] B[m,k] + C[i,k,m] B[j,m] = 0
    n_vars = dim * (dim + 1) // 2

    def sym_idx(a, b):
        lo, hi = min(a, b), max(a, b)
        return lo * dim - lo * (lo - 1) // 2 + (hi - lo)

    rows = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                row = np.zeros(n_vars)
                for m in range(dim):
                    if C[i, j, m] != 0:
                        row[sym_idx(m, k)] += C[i, j, m]
                    if C[i, k, m] != 0:
                        row[sym_idx(j, m)] += C[i, k, m]
                rows.append(row)
    M = np.array(rows)
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    return int(np.sum(S < 1e-10))


# ─── Coadjoint orbit analysis (Kirillov form) ───────────────────────────────

def kirillov_form(C, xi):
    """Skew matrix Omega_xi[i,j] = sum_k xi[k] C[i,j,k].
    Eigenvalues of i*Omega give the coadjoint orbit frequencies."""
    return np.einsum('k,ijk->ij', xi, C)

def orbit_frequencies(C, xi):
    """Return positive orbit frequencies from the Kirillov form.
    These are the nonzero singular values / 2 of the skew matrix,
    equivalently the positive imaginary eigenvalues."""
    Omega = kirillov_form(C, xi)
    # Eigenvalues of skew-symmetric matrix are purely imaginary: ±i*lambda
    eigs = np.linalg.eigvals(Omega)
    imag_parts = np.sort(np.abs(eigs.imag))[::-1]
    # Each positive frequency appears twice (±)
    freqs = []
    skip = set()
    for idx, v in enumerate(imag_parts):
        if idx in skip or v < 1e-12:
            continue
        freqs.append(v)
        # Find and skip the matching pair
        for idx2 in range(idx + 1, len(imag_parts)):
            if idx2 not in skip and abs(imag_parts[idx2] - v) < 1e-10:
                skip.add(idx2)
                break
    return np.array(sorted(freqs))

def coadjoint_orbit_ensemble(C, n_samples=20000, seed=42):
    """Sample random xi in L* and collect orbit frequency spectra."""
    rng = np.random.RandomState(seed)
    dim = C.shape[0]
    all_freqs = []
    all_ranks = []
    for _ in range(n_samples):
        xi = rng.randn(dim)
        freqs = orbit_frequencies(C, xi)
        all_freqs.append(freqs)
        # Orbit dimension = rank of Omega
        Omega = kirillov_form(C, xi)
        rank = np.linalg.matrix_rank(Omega, tol=1e-10)
        all_ranks.append(rank)
    return all_freqs, np.array(all_ranks)


# ─── Adjoint singular value ensemble ────────────────────────────────────────

def ad_sv_ensemble(C, n_samples=10000, seed=42):
    """Singular values of ad(x) for random x."""
    rng = np.random.RandomState(seed)
    dim = C.shape[0]
    all_svs = []
    for _ in range(n_samples):
        coeffs = rng.randn(dim)
        ad_x = np.einsum('i,ijk->jk', coeffs, C)
        svs = np.linalg.svd(ad_x, compute_uv=False)
        all_svs.append(np.sort(svs)[::-1])
    return np.array(all_svs)


# ─── Level spacing statistics ────────────────────────────────────────────────

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

def poisson_spacing(s):
    """Poisson: P(s) = exp(-s)."""
    return np.exp(-s)


# ─── Central extension analysis ──────────────────────────────────────────────

def cocycle_analysis(C, center_start=6):
    """Analyze the 2-cocycle of the central extension L -> L/Z(L).
    For the 17-dim nilpotent algebra: L/Z = span(e0..e5), Z = span(e6..e16).
    omega(e_i, e_j) = projection of [e_i, e_j] onto Z."""
    n_quotient = center_start
    n_center = C.shape[0] - center_start
    pairs = [(i, j) for i in range(n_quotient) for j in range(i + 1, n_quotient)]
    omega = np.zeros((len(pairs), n_center))
    for idx, (i, j) in enumerate(pairs):
        omega[idx] = C[i, j, center_start:]
    _, S, _ = np.linalg.svd(omega, full_matrices=False)
    return omega, S, pairs


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze_algebra(name, C, fig_dir):
    """Run full spectral analysis on one algebra."""
    dim = C.shape[0]
    print(f"\n{'=' * 70}")
    print(f"  POTENTIAL: V = {name}   (dim = {dim})")
    print(f"{'=' * 70}")

    # --- Verification ---
    print(f"\n[Verification]")
    print(f"  Antisymmetry error:    {verify_antisymmetry(C):.2e}")
    print(f"  Jacobi identity error: {verify_jacobi(C):.2e}")
    nonzero = int(np.sum(np.abs(C) > 1e-10))
    print(f"  Nonzero C[i,j,k]:     {nonzero} / {dim**3}")

    # --- Algebraic structure ---
    print(f"\n[Algebraic structure]")
    derived = compute_derived_series(C)
    lower = compute_lower_central_series(C)
    center = compute_center_dim(C)
    nilpotent = lower[-1] == 0
    nil_class = len(lower) - 1 if nilpotent else None

    print(f"  Derived series:      {derived}")
    print(f"  Lower central series:{lower}")
    print(f"  Solvable:  {'yes (length %d)' % (len(derived)-1) if derived[-1]==0 else 'no'}")
    print(f"  Nilpotent: {'class %d' % nil_class if nil_class else 'no'}")
    print(f"  Center dim: {center}")

    # --- Killing form ---
    print(f"\n[Killing form]")
    B = killing_form(C)
    eig_B = np.sort(np.linalg.eigvalsh(B))[::-1]
    pos = int(np.sum(eig_B > 1e-8))
    neg = int(np.sum(eig_B < -1e-8))
    zero = dim - pos - neg
    print(f"  Signature: ({pos}+, {neg}-, {zero} zero)")
    if pos + neg > 0:
        nz = eig_B[np.abs(eig_B) > 1e-8]
        print(f"  Nonzero eigenvalues: {nz}")

    # --- Invariant bilinear forms ---
    n_inv = invariant_form_dimension(C)
    print(f"  Invariant symmetric bilinear forms: {n_inv}-dimensional space")

    # --- Adjoint matrices ---
    print(f"\n[Adjoint representation]")
    active_gens = 0
    for i in range(dim):
        ad_i = C[i]
        r = np.linalg.matrix_rank(ad_i, tol=1e-10)
        if r > 0:
            active_gens += 1
            eigs = np.linalg.eigvals(ad_i)
            nz_eigs = eigs[np.abs(eigs) > 1e-10]
            svs = np.linalg.svd(ad_i, compute_uv=False)
            nz_svs = svs[svs > 1e-10]
            nilp = "nilpotent" if len(nz_eigs) == 0 else f"eigs={np.sort(nz_eigs.real)}"
            print(f"  ad(e_{i:2d}): rank={r:2d}, {nilp}, SVs={np.round(nz_svs, 4)}")
    print(f"  Active generators (nonzero ad): {active_gens}/{dim}")

    # --- Cocycle analysis (nilpotent case) ---
    if nilpotent and center > 0:
        center_start = dim - center
        print(f"\n[Central extension (cocycle)]")
        print(f"  L/Z(L) dim = {center_start},  Z(L) dim = {center}")
        omega, omega_svs, pairs = cocycle_analysis(C, center_start)
        eff_rank = int(np.sum(omega_svs > 1e-10))
        print(f"  Cocycle matrix: {omega.shape[0]} pairs -> R^{omega.shape[1]}")
        print(f"  Cocycle singular values: {np.round(omega_svs[omega_svs > 1e-10], 6)}")
        print(f"  Effective rank: {eff_rank}")
        print(f"  Nonzero brackets mapping to center:")
        for idx, (i, j) in enumerate(pairs):
            z = omega[idx]
            nz_k = np.where(np.abs(z) > 1e-10)[0]
            if len(nz_k) > 0:
                terms = " + ".join(
                    f"{z[k]:+.0f}*e_{k + center_start}" for k in nz_k
                )
                print(f"    [e_{i}, e_{j}] = {terms}")

    # ================================================================
    # COADJOINT ORBIT ANALYSIS — the main spectral invariant
    # ================================================================
    print(f"\n[Coadjoint orbit frequencies (Kirillov)]")
    print(f"  Sampling 20000 random xi in L*...")
    all_freqs, all_ranks = coadjoint_orbit_ensemble(C, n_samples=20000)

    # Orbit dimension statistics
    unique_ranks, rank_counts = np.unique(all_ranks, return_counts=True)
    print(f"  Orbit dimension distribution:")
    for r, c in zip(unique_ranks, rank_counts):
        print(f"    rank {r:2d}: {c:5d} samples ({100*c/len(all_ranks):.1f}%)")

    # Frequency statistics
    max_n_freq = max(len(f) for f in all_freqs)
    print(f"  Max distinct frequencies per orbit: {max_n_freq}")

    # Collect all frequencies for level spacing analysis
    all_freq_flat = np.concatenate([f for f in all_freqs if len(f) > 0])
    print(f"  Total frequency values collected: {len(all_freq_flat)}")
    if len(all_freq_flat) > 0:
        print(f"  Frequency range: [{np.min(all_freq_flat):.4f}, {np.max(all_freq_flat):.4f}]")
        print(f"  Mean frequency: {np.mean(all_freq_flat):.4f}")

    # Per-sample level spacing of orbit frequencies
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
        print(f"\n  Per-sample frequency spacing statistics:")
        print(f"    N spacings:        {len(per_sample_spacings)}")
        print(f"    Spacing variance:  {var_s:.4f}  (Poisson=1.0, GOE≈0.286, GUE≈0.178)")
        print(f"    Mean ratio <r>:    {mean_r:.4f}  (Poisson≈0.386, GOE≈0.536, GUE≈0.603)")
    else:
        print(f"  (Too few multi-frequency orbits for per-sample spacing analysis)")

    # === Adjoint SV ensemble ===
    print(f"\n[Adjoint singular value ensemble]")
    print(f"  Sampling 10000 random algebra elements...")
    svs_ens = ad_sv_ensemble(C, n_samples=10000)
    for sv_idx in range(min(6, svs_ens.shape[1])):
        col = svs_ens[:, sv_idx]
        col = col[col > 1e-10]
        if len(col) > 0:
            print(f"  SV_{sv_idx+1}: mean={np.mean(col):.4f}, "
                  f"std={np.std(col):.4f}, range=[{np.min(col):.4f}, {np.max(col):.4f}]")

    # Collect all nonzero SVs for level spacing
    all_sv_spacings = []
    for row in svs_ens:
        s = normalized_spacings(row)
        if len(s) > 0:
            all_sv_spacings.extend(s)
    all_sv_spacings = np.array(all_sv_spacings)

    if len(all_sv_spacings) > 100:
        var_sv = np.var(all_sv_spacings)
        ratios_sv = spacing_ratio(all_sv_spacings)
        mean_r_sv = np.mean(ratios_sv) if len(ratios_sv) > 0 else float('nan')
        print(f"\n  SV spacing statistics:")
        print(f"    N spacings:        {len(all_sv_spacings)}")
        print(f"    Spacing variance:  {var_sv:.4f}")
        print(f"    Mean ratio <r>:    {mean_r_sv:.4f}")

    # ================================================================
    # GLOBAL FREQUENCY SPECTRUM — the "Bernoulli polynomial" approach
    # ================================================================
    # Treat ALL orbit frequencies from the ensemble as a single spectrum
    # and analyze its bulk level-spacing statistics
    if len(all_freq_flat) > 100:
        print(f"\n[Global frequency spectrum statistics]")
        global_spacings = normalized_spacings(all_freq_flat)
        if len(global_spacings) > 50:
            var_g = np.var(global_spacings)
            ratios_g = spacing_ratio(global_spacings)
            mean_r_g = np.mean(ratios_g) if len(ratios_g) > 0 else float('nan')
            print(f"  N spacings:        {len(global_spacings)}")
            print(f"  Spacing variance:  {var_g:.4f}")
            print(f"  Mean ratio <r>:    {mean_r_g:.4f}")

    return {
        'dim': dim,
        'nilpotent': nilpotent,
        'nil_class': nil_class,
        'center_dim': center,
        'killing_sig': (pos, neg, zero),
        'derived': derived,
        'lower_central': lower,
        'n_invariant_forms': n_inv,
        'orbit_ranks': all_ranks,
        'orbit_freqs': all_freqs,
        'sv_ensemble': svs_ens,
        'per_sample_spacings': per_sample_spacings if len(per_sample_spacings) > 0 else None,
        'sv_spacings': all_sv_spacings if len(all_sv_spacings) > 0 else None,
    }


def generate_figures(results, fig_dir):
    """Generate comparison figures."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())

    # ── Figure 1: Orbit dimension distribution ──
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ranks = results[name]['orbit_ranks']
        ur, uc = np.unique(ranks, return_counts=True)
        ax.bar(ur, uc / len(ranks), color='steelblue', width=0.8)
        ax.set_xlabel('Orbit dimension (rank of Ω_ξ)')
        ax.set_ylabel('Fraction')
        ax.set_title(f'Coadjoint orbit ranks — V={name}')
        ax.set_xticks(ur)
    plt.tight_layout()
    plt.savefig(fig_dir / 'orbit_rank_distribution.png', dpi=150)
    plt.close()

    # ── Figure 2: Per-sample orbit frequency spacing vs GUE/GOE/Poisson ──
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    s_grid = np.linspace(0, 4, 300)
    for ax, name in zip(axes, names):
        spacings = results[name].get('per_sample_spacings')
        if spacings is not None and len(spacings) > 50:
            ax.hist(spacings, bins=60, density=True, alpha=0.7,
                    color='steelblue', label='Data')
            ax.plot(s_grid, wigner_gue(s_grid), 'r-', lw=2, label='GUE')
            ax.plot(s_grid, wigner_goe(s_grid), 'g-', lw=2, label='GOE')
            ax.plot(s_grid, poisson_spacing(s_grid), 'k--', lw=2, label='Poisson')
            ax.legend()
        ax.set_xlabel('Normalized spacing s')
        ax.set_ylabel('P(s)')
        ax.set_title(f'Orbit frequency spacings — V={name}')
        ax.set_xlim(0, 4)
    plt.tight_layout()
    plt.savefig(fig_dir / 'orbit_frequency_spacings.png', dpi=150)
    plt.close()

    # ── Figure 3: Adjoint SV spacing distribution ──
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        spacings = results[name].get('sv_spacings')
        if spacings is not None and len(spacings) > 50:
            ax.hist(spacings, bins=60, density=True, alpha=0.7,
                    color='darkorange', label='Data')
            ax.plot(s_grid, wigner_gue(s_grid), 'r-', lw=2, label='GUE')
            ax.plot(s_grid, wigner_goe(s_grid), 'g-', lw=2, label='GOE')
            ax.plot(s_grid, poisson_spacing(s_grid), 'k--', lw=2, label='Poisson')
            ax.legend()
        ax.set_xlabel('Normalized spacing s')
        ax.set_ylabel('P(s)')
        ax.set_title(f'Adjoint SV spacings — V={name}')
        ax.set_xlim(0, 4)
    plt.tight_layout()
    plt.savefig(fig_dir / 'adjoint_sv_spacings.png', dpi=150)
    plt.close()

    # ── Figure 4: Frequency density (global) ──
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        all_f = np.concatenate([f for f in results[name]['orbit_freqs'] if len(f) > 0])
        if len(all_f) > 0:
            ax.hist(all_f, bins=100, density=True, alpha=0.7, color='mediumpurple')
        ax.set_xlabel('Frequency λ')
        ax.set_ylabel('Density')
        ax.set_title(f'Orbit frequency density — V={name}')
    plt.tight_layout()
    plt.savefig(fig_dir / 'orbit_frequency_density.png', dpi=150)
    plt.close()

    # ── Figure 5: SV distribution by index ──
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ens = results[name]['sv_ensemble']
        n_sv = min(6, ens.shape[1])
        for sv_idx in range(n_sv):
            col = ens[:, sv_idx]
            col = col[col > 1e-10]
            if len(col) > 0:
                ax.hist(col, bins=80, density=True, alpha=0.5,
                        label=f'SV_{sv_idx+1}')
        ax.set_xlabel('Singular value')
        ax.set_ylabel('Density')
        ax.set_title(f'ad(x) SV distributions — V={name}')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / 'adjoint_sv_by_index.png', dpi=150)
    plt.close()

    # ── Figure 6: Adjoint matrix heatmaps (1/r only) ──
    if '1/r' in results:
        C_1r = load_structure_constants(
            Path('results/algebra_structure/N3_d2_1r/structure_constants_exact.json'))
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle('Adjoint matrices ad(eᵢ) for V = 1/r  (17-dim nilpotent)', fontsize=13)
        for i in range(6):
            ax = axes[i // 3, i % 3]
            im = ax.imshow(C_1r[i], cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='equal')
            ax.set_title(f'ad(e_{i})')
            ax.set_xlabel('k')
            ax.set_ylabel('j')
        plt.colorbar(im, ax=axes, shrink=0.6, label='C[i,j,k]')
        plt.tight_layout()
        plt.savefig(fig_dir / 'adjoint_heatmaps_1r.png', dpi=150)
        plt.close()

    print(f"\nFigures saved to {fig_dir}/")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    base = Path('results/algebra_structure')
    fig_dir = Path('primes/figures')

    # Load available potentials
    configs = [
        ('1/r', 'N3_d2_1r'),
        ('r^2', 'N3_d2_r2'),
    ]
    potentials = {}
    for name, subdir in configs:
        path = base / subdir / 'structure_constants_exact.json'
        if path.exists():
            potentials[name] = load_structure_constants(path)
            print(f"Loaded {name}: {potentials[name].shape}")

    if not potentials:
        print("ERROR: No structure constant files found!")
        return

    print("\n" + "=" * 70)
    print("  LEVEL-2 SUBALGEBRA SPECTRAL ANALYSIS")
    print("  3-body Poisson bracket algebra (N=3, d=2)")
    print("=" * 70)

    # Cross-potential comparison
    if '1/r' in potentials and 'r^2' in potentials:
        print(f"\nNote: 1/r is {potentials['1/r'].shape[0]}-dim, "
              f"r^2 is {potentials['r^2'].shape[0]}-dim (different algebras)")

    results = {}
    for name, C in potentials.items():
        results[name] = analyze_algebra(name, C, fig_dir)

    # ── Summary comparison ──
    print(f"\n{'=' * 70}")
    print("  SUMMARY COMPARISON")
    print(f"{'=' * 70}")
    for name in results:
        r = results[name]
        print(f"\n  {name}:")
        print(f"    dim={r['dim']}, nilpotent={'class %d' % r['nil_class'] if r['nil_class'] else 'no'}, "
              f"center={r['center_dim']}")
        print(f"    Killing signature: {r['killing_sig']}")
        print(f"    Derived series: {r['derived']}")
        print(f"    Invariant forms: {r['n_invariant_forms']}")

        if r.get('per_sample_spacings') is not None and len(r['per_sample_spacings']) > 50:
            var_s = np.var(r['per_sample_spacings'])
            ratios = spacing_ratio(r['per_sample_spacings'])
            mr = np.mean(ratios) if len(ratios) > 0 else float('nan')
            print(f"    Orbit freq spacing: var={var_s:.4f}, <r>={mr:.4f}")

        if r.get('sv_spacings') is not None and len(r['sv_spacings']) > 50:
            var_sv = np.var(r['sv_spacings'])
            ratios_sv = spacing_ratio(r['sv_spacings'])
            mr_sv = np.mean(ratios_sv) if len(ratios_sv) > 0 else float('nan')
            print(f"    Adjoint SV spacing:  var={var_sv:.4f}, <r>={mr_sv:.4f}")

    print(f"\n  Reference values:")
    print(f"    Poisson: var=1.000, <r>=0.386")
    print(f"    GOE:     var≈0.286, <r>≈0.536")
    print(f"    GUE:     var≈0.178, <r>≈0.603")
    print(f"    GSE:     var≈0.105, <r>≈0.676")

    # Generate figures
    generate_figures(results, fig_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
