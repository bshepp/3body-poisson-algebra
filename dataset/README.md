---
license: mit
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - mathematics
  - physics
  - lie-algebra
  - poisson-algebra
  - n-body-problem
  - random-matrix-theory
  - symbolic-computation
  - structure-constants
  - bracket-tensor
  - dimension-sequence
  - quantum-mechanics
  - celestial-mechanics
  - quantum-information
  - bell-inequality
  - scaling-laws
  - neural-network
  - sgd-dynamics
  - training-dynamics
language:
  - en
pretty_name: Pairwise Poisson Algebras of the N-Body Problem
size_categories:
  - n<1K
configs:
  - config_name: neural_algebras
    data_files: neural_algebras.parquet
  - config_name: dimension_sequences
    data_files: dimension_sequences.parquet
  - config_name: structure_constants
    data_files: structure_constants.parquet
  - config_name: charge_sensitivity
    data_files: charge_sensitivity.parquet
  - config_name: mass_invariance
    data_files: mass_invariance.parquet
  - config_name: level4_convergence
    data_files: level4_convergence.parquet
  - config_name: spectral_statistics
    data_files: spectral_statistics.parquet
  - config_name: physical_systems
    data_files: physical_systems.parquet
  - config_name: bell_test
    data_files: bell_test.parquet
  - config_name: scaling_formulas
    data_files: scaling_formulas.parquet
  - config_name: tier_decomposition
    data_files: tier_decomposition.parquet
  - config_name: contextuality
    data_files: contextuality.parquet
  - config_name: convergence_trajectories
    data_files: convergence_trajectories.parquet
---

# Pairwise Poisson Algebras: Neural Networks vs Physics

## Dataset Description

This dataset contains the first systematic computation of **pairwise Poisson bracket Lie algebras** for both neural network training dynamics and physical N-body systems. SGD with momentum is a Hamiltonian system; the pairwise interactions between weight layers generate a Lie algebra — and we discover that **neural networks produce richer algebraic structures than any physical system**.

### Neural Network Results

- **21 neural network algebra configurations** sweeping depth (L=2-5), width (k=1-3), coupling type (12 variants), loss function (MSE, cross-entropy, L1), and activation (linear, tanh, ReLU)
- **Seven universality classes discovered** at L=3:
  - `A: [3, 6, 17, 119]` — gradient, fisher, gradient_abs, l1-loss, hessian_plain (5 configs)
  - `B: [3, 6, 17, 115]` — directional (kinetic-gradient) coupling
  - `C: [3, 6, 17, 111]` — gradient_sum (diagonal-only gradient)
  - `D: [3, 6, 17, 104]` — gradient_cubic (cubic gradient potential)
  - `E: [3, 6, 17, 87]` — natural_gradient (Fisher-rescaled gradient)
  - `F: [3, 6, 17, 62]` — cross-entropy loss (nonlinear loss structure)
  - `G: [3, 5, 11, 47]` — hessian, symmetric, hessian_full, loss_power (4 configs) — only class that diverges at level 1
- **Within-class invariances**: The [3, 6, 17, 119] gradient class is invariant under width (k=1,2,3), loss (MSE = L1), and activation (linear = tanh = ReLU, through level 2) — confirmed AT level 3 for width.
- **Depth scaling diverges from physics at L>=4**: Neural L=3 matches physical levels 0-2 then breaks by +3 at level 3. Neural L=4 gives [6, 20, 164] vs physical [6, 14, 62] — breaks by +6 at level 1 already. Pattern: extras at level 1 = C(L,2)*(L-3).

### Physical Benchmark (Universality Reference)

- **685 dimension sequences** for N=3..50, 16+ potentials, quantum/classical
- **576+ exponent sweep values** across the continuous algebraic landscape
- **20 named physical systems** from helium atoms to triple black holes — all producing **[3, 6, 17, 116]**
- **16 exact rational structure constant tensors** (all non-harmonic L2 algebras proven isomorphic)
- **38 charge, 33 mass configurations** confirming invariance across 10 orders of magnitude

**Total: 993 rows across 13 Parquet tables.**

### What Makes This Unique

This dataset bridges **machine learning** and **mathematical physics** by providing exact algebraic invariants computed from first principles. Neural network training dynamics generate Lie algebras that can be directly compared to those of physical systems — revealing **at least seven distinct universality classes** as a function of the coupling structure chosen, with some classes strictly richer than any physical interaction law and others strictly less rich.

## Quick Start

```python
from datasets import load_dataset

# Load neural network algebras and compare coupling types
ds = load_dataset("bshepp/pairwise-poisson-algebras", "neural_algebras")
df = ds["train"].to_pandas()
for _, row in df[df["n_layers"] == 3].iterrows():
    print(f"{row['coupling_type']:12s} {row['loss_function']:15s} "
          f"width={row['width']} -> {row['dimension_sequence']}")
```

```python
# Compare neural vs physical: how many extra generators?
import json
nn = load_dataset("bshepp/pairwise-poisson-algebras", "neural_algebras")["train"].to_pandas()
phys = load_dataset("bshepp/pairwise-poisson-algebras", "physical_systems")["train"].to_pandas()

nn_gradient = nn[(nn["coupling_type"] == "gradient") & (nn["n_layers"] == 3)
                  & (nn["activation"] == "linear") & (nn["loss_function"] == "mse")]
nn_dims = json.loads(nn_gradient.iloc[0]["dimension_sequence"])
phys_dims = json.loads(phys.iloc[0]["dimension_sequence"])
print(f"Neural:  {nn_dims}")
print(f"Physics: {phys_dims}")
print(f"Extra generators at level 3: {nn_dims[3] - phys_dims[3]}")
```

```python
# Browse all dimension sequences
ds = load_dataset("bshepp/pairwise-poisson-algebras", "dimension_sequences")
df = ds["train"].to_pandas()
non_universal = df[df["dim_L3"].notna() & (df["dim_L3"] != 116)]
print(non_universal[["N", "potential", "bracket_type", "dim_L3"]])
```

## Mathematical Framework

Given N bodies with positions q_i in R^d and momenta p_i in R^d, interacting pairwise via potential V(r_ij), define pairwise Hamiltonians:

$$H_{ij} = \frac{|p_i - p_j|^2}{2\mu_{ij}} + q_i q_j \cdot V(|q_i - q_j|)$$

where mu_ij is the reduced mass. The **pairwise Poisson algebra** A_V is the Lie algebra generated by all H_ij under the canonical Poisson bracket:

$$\mathcal{A}_V = \mathrm{Lie}( H_{ij} : i \neq j )$$

This algebra is graded by bracket depth (level): level 0 contains the generators H_ij, level 1 contains {H_ij, H_kl}, etc. The **dimension sequence** [d_0, d_1, d_2, ...] records the cumulative dimension at each level.

### Key Results in This Dataset

1. **Neural networks break physical universality**: Gradient-product coupling on a 3-layer linear network gives [3, 6, 17, **119**] — 3 extra generators vs the physical universal 116. The neural and physical algebras have the same level-0/1/2 structure (3, 6, 17) but diverge at the highest polynomial-degree stratum of level 3. Neural generators peak at degree 34; physical at degree 10. Within each degree stratum, neural has fewer syzygies (linear dependencies), producing 3 extra independent directions.

2. **Seven neural universality classes at L=3** (not three): The choice of pairwise coupling yields distinct algebras. Diagonal gradients (`gradient_sum`) give [3, 6, 17, 111]; kinetic coupling (`directional`) gives 115; cubic gradients give 104; Fisher-rescaled gradients give 87; Hessian/symmetric/loss-squared give [3, 5, 11, 47]; standard gradient and several variants give 119.

3. **Within-class invariances (for the 119 class)**: The algebra is independent of weight dimension (width k=1,2,3 all give [3, 6, 17, 119] at level 3), loss function (MSE = L1 = 119), and activation function (linear, tanh, ReLU all give [3, 6, 17] through level 2) — within the gradient-product universality class.

4. **Depth-scaling asymmetry**: At L=3, neural and physical algebras agree through level 2 (both [3, 6, 17]) and diverge only at level 3. But at L=4 they already diverge at level 1 (neural 20 vs physical 14), and at L=5 even more so (neural 45 vs physical 25). The extra dimensions at level 1 follow the formula `C(L,2)*(L-3)`: 0, 6, 20 for L=3, 4, 5.

5. **Physical potential universality**: For N=3 with 1/r, 1/r^2, 1/r^3, 1/r^4, log, composite, Yukawa, and polynomial r^n (n≥4) potentials, the dimension sequence is universally [3, 6, 17, 116]. Only three exceptional potentials break this: r^1 → [3,4,5,5], r^2 → [3,6,13,15], r^3 → [3,6,17,109].

6. **N-body scaling**: L_0(N) = N(N-1)/2, L_1(N) = N(3N-5)/2, L_2(N) = N(4N^2-9N+3)/2 for N >= 4.

7. **Charge/mass independence**: The sequence [3, 6, 17, 116] is invariant under arbitrary charges and mass ratios spanning 10 orders of magnitude.

8. **Quantum deformation**: Moyal bracket yields [3, 6, 17, 117] for singular potentials — exactly one extra generator. Physics adds 1 generator upon quantization; neural networks add 3 (gradient class) or as many as depth^2 for deeper networks.

9. **GUE universality**: The log-gas potential (Dyson's GUE eigenvalue dynamics) produces the same algebra as Newtonian gravity: [3, 6, 17, 116].

10. **Classical Bell bound and non-commutativity**: All tested algebras have zero commuting pairs and respect the CHSH classical bound (max |S| = 1.77 < 2).

## Dataset Splits

### `neural_algebras` (21 rows)

Poisson bracket algebras arising from neural network training dynamics. SGD with momentum on a linear network is a Hamiltonian system; the pairwise interactions between weight layers generate a Lie algebra. This table sweeps across network depth (L=2--5), width (k=1--3), **twelve coupling types**, loss function (MSE, cross-entropy, L1), and activation function (linear, tanh, ReLU).

| Column | Type | Description |
|--------|------|-------------|
| `n_layers` | int | Number of weight layers (network depth) |
| `width` | int | Weight dimension per layer (1=scalar, >1=vector) |
| `activation` | str | Activation function: "linear", "tanh_taylor", "relu_taylor" |
| `loss_function` | str | Loss function: "mse", "cross_entropy", "l1" |
| `coupling_type` | str | Pairwise coupling: gradient, hessian, symmetric, fisher, gradient_sum, gradient_abs, hessian_plain, hessian_full, directional, natural_gradient, loss_power, gradient_cubic |
| `max_level` | int | Highest bracket depth computed |
| `dimension_sequence` | str | JSON-encoded cumulative dimension list |
| `new_per_level` | str | JSON-encoded new dimensions per level |
| `dim_L0`..`dim_L3` | int/null | Flattened dimensions per level |
| `matches_physical_116` | bool | Whether level-3 dimension equals physical universal 116 |
| `n_generators` | int | Total candidate generators |
| `stabilized` | bool | Whether algebra growth has stabilized |
| `is_exact` | bool | Whether computed with exact arithmetic |
| `computation_method` | str | "numerical_SVD" or "exact_gaussian_elimination_QQ" |
| `computation_time_s` | float | Computation time in seconds |

**Seven distinct universality classes at L=3**:

| Class | Dims | Members |
|-------|------|---------|
| A | [3, 6, 17, **119**] | gradient, fisher, gradient_abs, hessian_plain, l1-loss |
| B | [3, 6, 17, **115**] | directional (kinetic-gradient) |
| C | [3, 6, 17, **111**] | gradient_sum (diagonal only) |
| D | [3, 6, 17, **104**] | gradient_cubic |
| E | [3, 6, 17, **87**] | natural_gradient |
| F | [3, 6, 17, **62**] | gradient + cross-entropy loss |
| G | [3, 5, 11, **47**] | hessian, symmetric, hessian_full, loss_power |

**Within-class invariances** (Class A, gradient coupling with MSE or L1):
- Width invariance: k=1, 2, 3 all give [3, 6, 17, 119] at L=3
- Loss invariance: MSE = L1 = [3, 6, 17, 119]; cross-entropy BREAKS invariance → Class F (62)
- Activation invariance: linear = tanh = ReLU give [3, 6, 17] through level 2

**Depth scaling**: L=2 → [1,1,1,1]; L=3 → [3,6,17,119] (matches physics at L0-L2); L=4 → [6,20,164] (diverges from physics [6,14,62] already at L1 by +6); L=5 → [10,45,210] (diverges at L1 by +20). Level-1 extras follow `C(L,2)*(L-3)` exactly.

### `dimension_sequences` (~685 rows)

The headline table. One row per (N, d, potential, bracket_type) configuration. Includes flattened `dim_L0`..`dim_L4` integer columns for easy filtering in the HF Dataset Viewer, plus the full JSON `dimension_sequence`. Includes 12 quantum (Moyal bracket) rows covering the complete quantum classification.

| Column | Type | Description |
|--------|------|-------------|
| `N` | int | Number of bodies (3--8) |
| `d` | int | Spatial dimensions (1--2) |
| `potential` | str | Interaction potential: "1/r", "1/r^2", "1/r^3", "1/r^4", "r^1"--"r^10", "log", "composite(...)", "neural" |
| `bracket_type` | str | "poisson" (classical) or "moyal" (quantum) |
| `masses` | str/null | JSON-encoded mass list, e.g. '["1","2","3"]' or '"symbolic"' |
| `charges` | str/null | JSON-encoded charge list, e.g. '[2,-1,-1]' |
| `external_potential` | str/null | External confining potential, e.g. "harmonic_omega_1" |
| `max_level` | int | Highest bracket depth computed |
| `dimension_sequence` | str | JSON-encoded cumulative dimension list, e.g. '[3, 6, 17, 116]' |
| `new_per_level` | str/null | JSON-encoded new dimensions per level |
| `dim_L0`..`dim_L4` | int/null | Individual level dimensions (flattened for filtering) |
| `is_exact` | bool | True if computed over QQ (exact arithmetic), False if numerical SVD |
| `computation_method` | str | "symbolic_QQ", "symbolic_QQ_m1m2m3", "numerical_svd", etc. |
| `sympy_version` | str | SymPy version used |
| `computation_time_s` | float | Wall-clock computation time in seconds |
| `physical_system` | str/null | Physical interpretation, e.g. "GUE_quantum_log_gas" |
| `source_file` | str | Path to originating JSON in the repository |

### `structure_constants` (16 rows)

Exact rational structure constants C^k_ij for the bracket algebra (N=3). Most at level 2 (d=1 or d=2), with one level-3 entry (r^3). The tensor satisfies [e_i, e_j] = sum_k C^k_ij e_k. Covers 15 potentials: 1/r through 1/r^4, r^1 through r^10, log, and 3 composites.

| Column | Type | Description |
|--------|------|-------------|
| `potential` | str | Interaction potential |
| `algebra_dim` | int | Algebra dimension (17 for most, 15 for harmonic) |
| `structure_constants` | str | JSON-encoded dim x dim x dim tensor of rational strings |
| `killing_signature` | str | JSON-encoded Killing form signature [pos, zero, neg] |
| `is_semisimple` | bool | Whether the algebra is semisimple |
| `is_solvable` | bool | Whether the algebra is solvable |
| `solvability_length` | int/null | Length of derived series |
| `is_nilpotent` | bool | Whether the algebra is nilpotent |
| `nilpotency_class` | int/null | Nilpotency class |
| `center_dimension` | int | Dimension of the center |
| `derived_series` | str | JSON-encoded derived series dimensions |
| `lower_central_series` | str | JSON-encoded lower central series dimensions |

### `charge_sensitivity` (38 rows)

Tests whether the algebra dimension depends on particle charges q_i. Includes flattened `dim_L0`..`dim_L3` for easy filtering.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_key` | str | Experiment identifier |
| `label` | str | Human-readable label |
| `charges` | str | JSON-encoded charge vector |
| `masses` | str | JSON-encoded mass vector (physical masses in electron-mass units) |
| `n_samples` | int | Number of phase-space samples for numerical SVD |
| `dimension_sequence` | str | JSON-encoded dimension sequence |
| `dim_L0`..`dim_L3` | int/null | Flattened dimension at each level |
| `matches_116` | bool | Whether level-3 dimension equals 116 |
| `physical_system` | str | Physical system label |
| `computation_time_s` | float | Computation time in seconds |

### `mass_invariance` (33 rows)

Sweep of the third mass m_3 from 1 to 10^10 (with m_1 = m_2 = 1), showing dimension invariance through moderate ratios and conditioning degradation at astrophysical extremes.

| Column | Type | Description |
|--------|------|-------------|
| `m3` | float | Third body mass |
| `m3_log10` | float | log10(m_3) |
| `level_0_dim` | int | Dimension at level 0 |
| `level_1_dim` | int | Dimension at level 1 |
| `level_2_dim` | int | Dimension at level 2 |
| `level_2_gap_ratio` | float | Singular value gap ratio at level 2 (rank determinant) |
| `level_2_singular_values` | str | JSON-encoded full singular value spectrum |
| `dims` | str | JSON-encoded [level_0, level_1, level_2] |
| `elapsed_s` | float | Computation time |

### `level4_convergence` (19 rows)

Level-4 lower bounds from numerical SVD at increasing sample sizes.

| Column | Type | Description |
|--------|------|-------------|
| `config` | str | Phase-space configuration: "global", "euler", "lagrange", "scalene" |
| `n_samples` | int | Number of phase-space samples |
| `dimension_sequence` | str | JSON-encoded cumulative dimension sequence through level 4 |
| `new_per_level` | str | JSON-encoded new dimensions per level |
| `d4_lower_bound` | int | Lower bound on level-4 dimension |
| `max_gap_ratio` | float | Maximum singular value gap ratio |
| `max_gap_index` | int | Index of maximum gap |
| `elapsed_seconds` | float | Computation time |
| `mu`, `phi`, `epsilon` | float | Phase-space parametrization (null for global) |

### `spectral_statistics` (14 rows)

Rank distributions across phase space from atlas scans.

| Column | Type | Description |
|--------|------|-------------|
| `config` | str | Configuration identifier |
| `label` | str | LaTeX-formatted label |
| `source_type` | str | "targeted_atlas" or "atlas_full_irrational" |
| `n_regions` | int | Number of phase-space regions sampled |
| `rank_min` | int | Minimum observed rank |
| `rank_max` | int | Maximum observed rank |
| `rank_mode` | int | Modal rank (most frequent) |
| `n_points` | int | Total phase-space points |
| `pct_116` | float | Percentage of points achieving rank 116 |

### `physical_systems` (17 rows)

Named physical systems with their computed dimension sequences, spanning astrophysical (Sun-Earth-Moon, triple black holes) to atomic (helium, lithium ion) to exotic (Penning traps, 2D vortices).

| Column | Type | Description |
|--------|------|-------------|
| `system_name` | str | Machine-readable identifier, e.g. "helium", "triple_bh_lisa" |
| `system_label` | str | Human-readable name, e.g. "Helium Atom", "Triple Black Hole (LISA)" |
| `category` | str | "astrophysical", "atomic", "ion_trap", or "condensed_matter" |
| `dimension_sequence` | str | JSON-encoded dimension sequence |
| `dim_L0`..`dim_L3` | int | Flattened dimensions per level |
| `matches_universal` | bool | Whether the sequence equals the universal [3, 6, 17, 116] |
| `completed_at` | str | ISO timestamp of computation completion |
| `source_file` | str | Path to originating JSON |

### `bell_test` (9 rows)

CHSH Bell inequality tests on the Poisson algebra. Three phase-space strata (equilateral, pair apparatus, separated) with three observable variants each. Tests whether the algebra structure permits violations of the classical CHSH bound (|S| must stay below 2).

| Column | Type | Description |
|--------|------|-------------|
| `stratum` | str | Phase-space stratum: "equilateral", "pair_apparatus", "separated" |
| `variant` | str | Observable variant: "variant1", "variant2", "variant3" |
| `max_abs_S` | float | Maximum abs(S) observed |
| `max_S` | float | Maximum S (signed) |
| `ci_95_lower` | float | 95% confidence interval lower bound |
| `ci_95_upper` | float | 95% confidence interval upper bound |
| `optimal_angles_deg` | str | JSON-encoded optimal measurement angles |
| `significant_violation` | bool | Whether S exceeds classical bound (always False) |
| `n_samples` | int | Number of phase-space samples |
| `classical_bound` | float | CHSH classical bound (2.0) |
| `tsirelson_bound` | float | Tsirelson quantum bound (2*sqrt(2)) |

### `scaling_formulas` (5 rows)

Closed-form scaling formulas for algebra dimension as a function of N bodies. Documents the meta-scientific narrative: original conjecture, falsification at N=7, and corrected formula.

| Column | Type | Description |
|--------|------|-------------|
| `level` | str | Formula identifier: "L0", "L1", "L2_original", "L2_corrected", "L3" |
| `formula_expression` | str | Closed-form expression, e.g. "N(4N^2 - 9N + 3)/2" |
| `formula_status` | str | "verified", "verified_trivial", "falsified", or "unknown" |
| `leading_term` | str | Asymptotic leading term |
| `new_per_level_formula` | str | Formula for new generators per level |
| `verified_N_values` | str | JSON-encoded list of N values where formula is verified |
| `failed_N_values` | str/null | JSON-encoded N values where formula fails |
| `predictions` | str/null | JSON-encoded predictions for untested N |
| `data_points` | str/null | JSON-encoded known data points |
| `notes` | str/null | Context, caveats, and conjectures |

### `tier_decomposition` (40 rows)

Clebsch-Gordan decomposition of the candidate generator spaces under S_3 (N=3) and S_4 (N=4) symmetry. Tracks how each irreducible representation contributes at each bracket level, and compares candidate counts to observed independent generators.

| Column | Type | Description |
|--------|------|-------------|
| `N` | int | Number of bodies |
| `symmetry_group` | str | "S3" or "S4" |
| `level` | int | Bracket level (-1 for total) |
| `total_candidates` | int | Number of candidate generators at this level |
| `observed_rank` | int | Observed independent rank |
| `irrep_name` | str | Irreducible representation name |
| `irrep_dim` | int | Dimension of the irrep |
| `multiplicity` | int | Number of copies of this irrep |
| `contribution` | int | Total generators from this irrep (multiplicity x dim) |

### `contextuality` (16 rows)

Kochen-Specker contextuality tests on all available structure constant algebras. Tests whether the orthogonality graph (edges between Poisson-commuting pairs) permits KS-uncolorable configurations.

| Column | Type | Description |
|--------|------|-------------|
| `N` | int | Number of bodies |
| `d` | int | Spatial dimensions |
| `potential` | str | Interaction potential |
| `algebra_dim` | int | Algebra dimension |
| `n_commuting_pairs` | int | Number of Poisson-commuting generator pairs (always 0) |
| `total_pairs` | int | Total generator pairs tested |
| `commutativity_fraction` | float | Fraction of commuting pairs (always 0.0) |
| `ks_colorable` | bool | Whether KS coloring exists (always True) |
| `contextual` | bool | Whether algebra is contextual (always False) |
| `pm_constructible` | bool | Whether Peres-Mermin square can be built (always False) |

### `convergence_trajectories` (77 rows)

SVD rank convergence as a function of sample count, for multiple (N, d, potential, level) configurations. Shows how many phase-space samples are needed for reliable numerical rank determination.

| Column | Type | Description |
|--------|------|-------------|
| `N` | int | Number of bodies |
| `d` | int | Spatial dimensions |
| `potential` | str | Interaction potential |
| `level` | int | Bracket level |
| `n_samples` | int | Number of phase-space samples |
| `n_candidates` | int | Number of candidate generators |
| `rank` | int | Numerical rank determined by SVD gap |
| `gap_ratio` | float | Best SVD gap ratio |
| `elapsed_s` | float | Computation time in seconds |

## Reproduction

All data was generated from the code at [github.com/bshepp/3body](https://github.com/bshepp/3body).

### Requirements

- Python 3.10+
- SymPy >= 1.12 (for exact symbolic computation)
- NumPy, SciPy (for numerical SVD)
- pandas, pyarrow (for Parquet generation)

### Rebuilding the Dataset

```bash
git clone https://github.com/bshepp/3body.git
cd 3body
pip install pandas pyarrow
python dataset/build_dataset.py
python dataset/validate_dataset.py  # optional: run validation suite
```

The script reads all JSON result files and produces 13 Parquet tables in `dataset/output/`.

### Reproducing Individual Results

Example: compute the N=3, d=1, 1/r dimension sequence:

```python
from nbody.algebra import NBodyAlgebra

alg = NBodyAlgebra(N=3, d=1, potential="1/r")
result = alg.compute_ranks(max_level=3)
print(result["cumulative_rank"])  # [3, 6, 17, 116]
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{sheppeard2026poisson,
  title={Pairwise Poisson Algebras of the N-Body Problem},
  author={Sheppeard, B.},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/bshepp/pairwise-poisson-algebras}
}
```

## License

MIT License. See the [repository](https://github.com/bshepp/3body) for full terms.
