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

## Files

| File | Purpose |
|------|---------|
| `gue_prime_connection.tex` | LaTeX writeup of the mathematical framework |
| `run_gue_logas.py` | Computation script (runs all four configs) |
| `launch_gue.py` | AWS launcher (dispatches to EC2 spot instance) |
| `userdata_gue.sh` | EC2 userdata template |
| `results/` | Output directory (populated by computation) |

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
