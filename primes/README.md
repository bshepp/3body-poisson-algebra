# Primes Sub-Project: Poisson Algebra of the Dyson Log-Gas

## Overview

This sub-project investigates the connection between the Poisson algebra
dimension sequence **[3, 6, 17, 116]** and the distribution of prime numbers,
via the Dyson log-gas model of Riemann zeta zeros.

### The Key Insight

The Montgomery-Odlyzko law states that the nontrivial zeros of the Riemann
zeta function have pair correlations matching GUE (Gaussian Unitary Ensemble)
random matrix eigenvalues. The GUE joint eigenvalue density is:

    P(t_1, ..., t_N) ∝ ∏_{i<j} |t_i - t_j|^2 · exp(-½ Σ t_i²)

This is a Boltzmann weight e^{-βE} with energy:

    E = -2 Σ_{i<j} log|t_i - t_j| + ½ Σ_i t_i²

which decomposes into pairwise Hamiltonians:

    H_ij = p_i²/2 + p_j²/2 - 2 log|t_i - t_j| + harmonic confinement

This is exactly the 1D N-body problem with logarithmic potential and harmonic
trap — a system that the `NBodyAlgebra` engine already computes.

### What We Compute

Four configurations for N=3 particles in d=1 spatial dimension:

| Config | Potential | Confinement | Physical Model | Expected Dims |
|--------|-----------|-------------|----------------|---------------|
| (a) Pure log-gas | log(r) | none | 2D point vortices | [3, 6, 17, 116] |
| (b) **GUE composite** | log(r) | harmonic ω=1 | **Zeta zero correlations** | [3, 6, 17, 116] |
| (c) Penning trap | 1/r | harmonic ω=1 | Ion trap (reference) | [3, 6, 17, 116] |
| (d) Harmonic only | — | harmonic r² | Integrable reference | closes at 15 |

The universality conjecture predicts configs (a)-(c) all give **[3, 6, 17, 116]**.
Config (b) is the novel result: the Poisson algebra of the exact GUE Hamiltonian.

### Why This Matters

If confirmed, this establishes that the algebraic structure governing
correlations between Riemann zeta zeros (and hence prime number distributions
via the explicit formula) belongs to the same universality class as Newtonian
gravity.

### Level-2 Spectral Analysis: BGS Conjecture in Algebraic Structure

The level-2 bracket tensor (structure constants C[i,j,k] encoding
[eᵢ, eⱼ] = Σₖ C[i,j,k] eₖ) was extracted from
`results/algebra_structure/` and subjected to spectral analysis via Kirillov
coadjoint orbit theory.

**Method:** For random covectors ξ, form the Kirillov matrix Ω_ξ[i,j] = Σₖ ξₖ C[i,j,k].
The eigenvalues of iΩ yield orbit frequencies; their level spacing statistics
distinguish integrable (Poisson) from chaotic (GUE/GOE/GSE) algebraic structure.

| Algebra | dim | Jacobi | Orbit rank | Freq spacing var | Mean ratio ⟨r⟩ | Class |
|---------|-----|--------|-----------|-----------------|---------------|-------|
| **1/r** (singular) | 17 | FAILS | 6 (3 freq pairs) | **0.117** | **0.639** | GUE–GSE |
| **r²** (harmonic) | 15 | passes | 12 (6 freq pairs) | **1.168** | **0.401** | Poisson |
| *Reference: GUE* | — | — | — | 0.178 | 0.603 | — |
| *Reference: GSE* | — | — | — | 0.105 | 0.676 | — |
| *Reference: Poisson* | — | — | — | 1.000 | 0.386 | — |

**Key results:**

1. **BGS conjecture in algebra:** The non-integrable 1/r bracket tensor shows
   GUE-to-GSE level repulsion (var = 0.117, ⟨r⟩ = 0.639), while the integrable
   r² algebra shows Poisson statistics (var = 1.168, ⟨r⟩ = 0.401). This is the
   Bohigas-Giannoni-Schmit conjecture manifested in the algebraic structure
   itself, not in eigenvalues of a quantum Hamiltonian.

2. **Symplectic lean:** The 1/r statistics sit between GUE and GSE rather than
   exactly at GUE. This likely reflects the inherent symplectic structure of the
   Poisson bracket — the algebra "knows" it comes from symplectic geometry.

3. **Jacobi identity failure (1/r):** The 17-dim structure constants do NOT
   satisfy the Jacobi identity (error = 1.0). This correctly reflects that the
   bracket algebra is infinite-dimensional — the 17 generators at level 2 are not a
   closed Lie subalgebra. The structure constants capture the bracket tensor, not
   a Lie algebra.

4. **Jacobi identity passes (r²):** The 15-dim harmonic algebra IS a genuine
   closed Lie algebra. Killing form signature (6+, 4−, 5 zero) suggests it is
   near-semisimple, likely sp(6) ⊕ center.

5. **Algebraic structure (1/r):** Killing form identically zero. Center dim 11.
   Derived series [17, 14, 3, 0]. Lower central series [17, 14, 11, 0] →
   nilpotent class 3. 13-dim space of invariant bilinear forms. All 6 adjoint
   matrices are nilpotent with rank 5.

### Hilbert-Pólya Search and Infinite-Dimensionality Discovery

An attempt to find a Hilbert-Pólya operator whose eigenvalues encode zeta zeros
led to the discovery that the 1/r Poisson bracket algebra is NOT closed at any
finite level. Specifically:

- Level 2 → level 3 brackets: rank grew from 116 to 216 (not closing at 116)
- Level 3 → level 4 brackets: rank grew from 116 to 306 (even more new directions)
- This means the algebra is **infinite-dimensional** — levels 0–3 giving [3, 6, 17, 116]
  are the first four terms of an unbounded sequence.

The HP search pivot to level-2 subalgebra spectral analysis was the productive
outcome — the coadjoint orbit frequencies themselves provide the "spectral"
information that a hypothetical Hilbert-Pólya operator would encode.

## Files

| File | Purpose |
|------|---------|
| `gue_prime_connection.tex` | LaTeX writeup of the mathematical framework |
| `run_gue_logas.py` | Computation script (runs all four configs) |
| `run_quantum_gue.py` | Quantum Moyal bracket computation (ℏ-deformation) |
| `level2_spectral_analysis.py` | **Spectral analysis of level-2 bracket tensor** — Kirillov orbits, adjoint SVD, level spacing statistics |
| `hilbert_polya_search.py` | Hilbert-Pólya operator search (results superseded by closure discovery) |
| `diagnose_brackets.py` | Diagnostic: 4-test validation of bracket computation |
| `check_1r_closure.py` | Closure test: confirms algebra is infinite-dimensional |
| `closure_check.py` | Wrapper for compute_growth closure verification |
| `launch_gue.py` | AWS launcher (dispatches to EC2 spot instance) |
| `userdata_gue.sh` | EC2 userdata template |
| `figures/` | Spectral analysis figures (6 PNGs) |
| `results/` | Output directory (populated by computation) |

### Figures

| Figure | Content |
|--------|---------|
| `figures/orbit_frequency_spacings.png` | Level spacing histograms vs GUE/GOE/Poisson/GSE reference curves |
| `figures/orbit_rank_distribution.png` | Coadjoint orbit dimension distribution |
| `figures/adjoint_heatmaps_1r.png` | 6 adjoint matrices ad(eᵢ) for 1/r as heatmaps |
| `figures/adjoint_sv_spacings.png` | Singular value spacing distributions |
| `figures/orbit_frequency_density.png` | Frequency density histograms across random covectors |
| `figures/adjoint_sv_by_index.png` | Singular value distributions by generator index |

## Running

### On AWS (recommended if local machine is busy)

```bash
# Dry run first — verifies userdata generation without launching
python primes/launch_gue.py --dry-run

# Launch on a spot instance (default)
python primes/launch_gue.py

# Launch on-demand instead
python primes/launch_gue.py --on-demand

# Monitor progress
aws s3 cp s3://3body-compute-290318/results/primes/live.log -

# Check completion
aws s3 ls s3://3body-compute-290318/results/primes/aws_completion.json

# Pull results
aws s3 sync s3://3body-compute-290318/results/primes/ primes/results/
```

### Locally

```bash
# Quick verification (level 2 only, ~2 min)
python primes/run_gue_logas.py --max-level 2

# Full computation (level 3, ~1 hour)
python primes/run_gue_logas.py --max-level 3

# Resume from checkpoint if interrupted
python primes/run_gue_logas.py --max-level 3 --resume
```

## Further Directions

See `gue_prime_connection.tex` Section 6 for detailed discussion. Summary:

1. **N > 3 GUE zeros**: Run N=4, 5 to test if the graph-theoretic decomposition
   L₂(N) = N(4N²-9N+3)/2 holds for the GUE composite potential.

2. **Level 4**: Would require AWS (days of compute). Tests whether L₃ universality
   extends to the GUE case.

3. **Selberg trace formula**: Geodesic flow on SL(2,Z)\H as a Hamiltonian system
   whose periodic orbits have lengths 2 log p (primes).

4. **Rankin-Selberg L-functions**: Pairwise interactions H_ij between Dirichlet
   L-functions via convolution L(s, χ_i × χ̄_j).
