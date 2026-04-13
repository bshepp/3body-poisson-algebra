# Data Acquisition Directions for the Hugging Face Dataset

*April 12, 2026*

This document outlines natural directions for expanding the dataset, organized by which split they feed, how much effort they require, and what scientific value they add. The current dataset has **153 rows across 9 splits**. Each direction below lists the target split, estimated new rows, compute cost, and whether it requires new code or just running existing scripts.

---

## Tier 1: Low-Hanging Fruit (local compute, minutes to hours)

These use existing scripts with new parameters. Rebuild the dataset afterward with `python dataset/build_dataset.py`.

### 1.1 N=9 and N=10 Level-2 Dimension Sequences

- **Feeds:** `dimension_sequences`, `scaling_formulas`
- **New rows:** 2 in `dimension_sequences`
- **Effort:** Minutes. Run `python nbody/symbolic_rank_nbody.py -N 9 -d 1 --max-level 2` and `-N 10`.
- **Value:** Extends the L2 formula verification from N=8 to N=10. Predictions: L2(9) = N(4N²−9N+3)/2 = 9·(324−81+3)/2 = 9·246/2 = **1107**, L2(10) = 10·(400−90+3)/2 = 10·313/2 = **1565**. If these match, the corrected cubic is verified across 7 values. If they don't, the formula needs further revision.
- **ETL:** `build_dataset.py` already picks up `results/symbolic_rank/rank_N*.json` automatically.
- **Priority:** HIGH — nearly free, extends a headline result.

### 1.2 More Structure Constants (Missing Potentials)

- **Feeds:** `structure_constants`
- **New rows:** 3–5 in `structure_constants`
- **Effort:** 5–30 min per potential. Run `python nbody/symbolic_rank_nbody.py -N 3 -d 2 --potential <pot> --structure --max-level 2`.
- **Currently have:** 1/r, 1/r², 1/r³, 1/r⁴, r², r⁴, composite(u+u²), composite(u+u²+u³), composite(u⁴). That's 9 potentials.
- **Missing obvious ones:** log (if structure extraction works for transcendentals), neural (if the engine supports it at d=2).
- **Value:** Completing the structure constant catalog across all tested potentials. Every new tensor is a contribution because nobody else has these objects in machine-readable form. The isomorphism question (are all non-harmonic algebras literally the *same* Lie algebra?) can be answered by comparing Killing signatures and derived series.
- **Priority:** MEDIUM — valuable but the current 9 is already unique.

### 1.3 Named Molecular Systems: H₃⁺ and Ozone

- **Feeds:** `physical_systems`, `dimension_sequences`
- **New rows:** 2 in `physical_systems`, 2 in `dimension_sequences`
- **Effort:** Minutes. These are 1/r Coulomb systems at specific nuclear masses (H₃⁺: three protons at m≈1836; O₃: three oxygen nuclei at m≈29164). Mass invariance guarantees [3, 6, 17, 116], but publishing the result for named molecules has outreach value.
- **Script:** `python nbody/symbolic_rank_nbody.py -N 3 -d 1 --potential 1/r --masses 1836 1836 1836 --max-level 3`
- **Value:** Molecular physicists and chemists will recognize H₃⁺ (simplest polyatomic molecule, important in interstellar chemistry) and O₃. These are "human interest" entries.
- **Priority:** HIGH — essentially free, high recognition factor.

### 1.4 Charge Sweep Phase 3 (+1/+q/−1 Molecular Geometry)

- **Feeds:** `charge_sensitivity`
- **New rows:** ~20 in `charge_sensitivity`
- **Effort:** Hours (local or small AWS). Phase 2 (+q/−1/−1) is complete for q=1–20, all [3, 6, 17, 116]. Phase 3 (+1/+q/−1) crashed and was never completed.
- **Value:** The molecular charge geometry is where the known departures (115 for H₂⁺, 111 for Li⁺) live. A systematic sweep of +1/+q/−1 for q=2..20 would map the landscape of level-3 departures. If 115 persists across many q values, it's a structural feature of molecular geometry; if it varies, the charge-magnitude dependence is richer than currently known.
- **Priority:** HIGH — fills the most interesting gap in the charge sensitivity story.

### 1.5 Quantum Dimension Sequences (Additional Potentials)

- **Feeds:** `dimension_sequences`
- **New rows:** ~5 in `dimension_sequences`
- **Effort:** Minutes to hours. The quantum (Moyal bracket) results currently cover 1/r, 1/r², 1/r³, 1/r⁴, r², r⁴, and the GUE composite. Missing: log (non-GUE), composite potentials.
- **Value:** Fills in the quantum column. The classification "all singular → +1, all polynomial → +0" could be tested at more potentials.
- **Priority:** LOW-MEDIUM — confirms an existing pattern rather than extending it.

---

## Tier 2: Moderate Effort (AWS spot, $10–$50)

### 2.1 N=4 Level-3 with Additional Potentials

- **Feeds:** `dimension_sequences`
- **New rows:** 3–4 in `dimension_sequences`
- **Effort:** ~1 hour per potential on r6i.4xlarge. N=4 Level 3 for 1/r took 52 minutes.
- **Potentials to test:** 1/r², 1/r³, log. Predictions: all should give [6, 14, 62, 1260] if potential universality extends to N=4 Level 3.
- **Value:** VERY HIGH — directly tests whether the 1260 at Level 3 is potential-universal for N=4, just as 116 is for N=3.
- **Priority:** HIGH — falsifiable prediction at moderate cost.

### 2.2 N=4 Quantum Rank

- **Feeds:** `dimension_sequences`
- **New rows:** 1–2 in `dimension_sequences`
- **Effort:** ~1–2 hours on AWS. Run `QuantumNBodyAlgebra(4, 1, "1/r")` through Level 3.
- **Value:** Tests whether the "+1 quantum growth" pattern (116→117 for N=3) generalizes to N=4: does 1260 become 1261? Or something else?
- **Priority:** MEDIUM — unique data point for the quantum deformation story.

### 2.3 Parametric Exponent Level-2 Broad Sweep

- **Feeds:** new split `exponent_sweep` (or extend `dimension_sequences`)
- **New rows:** ~100–1000 in a new split
- **Effort:** Level 2 is trivially fast (seconds per exponent). Running 1/r^n for n = 0.01, 0.02, ..., 5.00 plus n = −5.00, ..., −0.01 (1015 values) at Level 2 costs almost nothing — it's just polynomial evaluation at each n.
- **Script:** Already exists (`parametric_atlas_scan.py`). Run at `--max-level 2` only.
- **Value:** Maps the transition between finite (n=−2, harmonic) and infinite (all other n) algebras as a function of exponent. The sharp boundary at n=0 and the special point at n=−2 are the key features. This would be the first continuous map of algebraic dimension vs. interaction law in any dataset.
- **New split schema:** `exponent`, `n_value`, `dimension_sequence`, `dim_L0`, `dim_L1`, `dim_L2`, `matches_universal`, `is_singular`, `computation_time_s`
- **Priority:** VERY HIGH — unique result with nearly zero cost; natural paper figure.

### 2.4 Level-3 Structure Constants (Rank 116 Tensor)

- **Feeds:** `structure_constants` (dramatically)
- **New rows:** 1–3 in `structure_constants` (but each tensor is 116³ = 1,560,896 entries)
- **Effort:** 6–24 hours per potential on r6i.4xlarge. Currently in progress for 1/r.
- **Value:** The 17×17×17 tensors at Level 2 are already unique. A 116×116×116 tensor at Level 3 would be unprecedented — nobody has structure constants for a 116-dimensional Poisson algebra. Even one such tensor would be a significant contribution.
- **Priority:** MEDIUM-HIGH — expensive but unique. The in-progress 1/r computation may already provide this.

### 2.5 Mass Ratio Sweep Extension

- **Feeds:** `mass_invariance`
- **New rows:** ~10–20 in `mass_invariance`
- **Effort:** Hours. Extend the sweep to include extreme ratios (1:1:10⁸, 1:1:10¹⁰) at Level 2 to map where SVD conditioning breaks down and to capture the Born-Oppenheimer transition.
- **Value:** The current sweep covers 1 to 10⁶. Extending to astrophysical ratios (10⁸–10¹⁰, comparable to Sun-Earth-Moon) with full SVD spectra would provide a complete conditioning landscape. This feeds the noise-plateau mapping analysis (gap_workplan 4.6).
- **Priority:** MEDIUM — methodological contribution.

---

## Tier 3: Substantial Effort (AWS, $50–$200+)

### 3.1 N=5 Level-3 (Algorithmic Challenge)

- **Feeds:** `dimension_sequences`, `scaling_formulas`
- **New rows:** 1 in `dimension_sequences`, updates to `scaling_formulas`
- **Effort:** The N=5 L3 computation OOM-killed at 256 GB. The 1.1M×760K matrix over QQ exceeds available memory. Options:
  1. **Modular rank:** Compute rank over GF(p) for several primes p and use CRT. This avoids the QQ coefficient explosion.
  2. **Larger instance:** r6i.16xlarge (512 GB) or x2idn.xlarge (768 GB).
  3. **Out-of-core:** Stream matrix rows from disk.
- **Value:** CRITICAL for the graph-theoretic conjecture. The prediction is new_L3(5) = 5990 if a=1198 holds. This is the first non-trivial test of the C(N,4) pattern at Level 3.
- **Priority:** VERY HIGH — but requires algorithmic innovation, not just more compute.

### 3.2 Yukawa Potential (Exponential Screening)

- **Feeds:** `dimension_sequences`, `physical_systems`
- **New rows:** 3+ in `dimension_sequences`, 3+ in `physical_systems`
- **Effort:** Requires debugging the Yukawa lambdification pipeline. The exponential damping e^{−μr}/r creates deeply nested expressions that cause OOM during symbolic compilation.
- **Systems:** Dusty plasma, tritium/He-3, proton-neutron-neutron scattering.
- **Value:** Yukawa is the key non-power-law singular potential. It governs nuclear forces and screened Coulomb interactions. Confirming universality for Yukawa would extend the result beyond pure power-law and logarithmic potentials to the full class of singular interactions relevant to particle physics.
- **Priority:** HIGH — blocked on engineering, not science.

### 3.3 Parametric Exponent Level-3 (Selected Values)

- **Feeds:** new split `exponent_sweep` or `dimension_sequences`
- **New rows:** ~20–50
- **Effort:** $50–100 on AWS. After the Level-2 broad sweep (Tier 2.3) identifies interesting transition regions, run Level 3 at ~20–50 selected n-values around transitions.
- **Value:** Combined with the Level-2 broad sweep, this creates a publication-quality figure showing algebra dimension as a continuous function of the potential exponent. The transition at n=0 (singular→regular) and the special point n=−2 (harmonic) would be clearly visible.
- **Priority:** HIGH — but depends on 2.3 being done first.

### 3.4 N=4 Atlas Scans

- **Feeds:** `spectral_statistics` (or new split)
- **New rows:** ~10+ in `spectral_statistics`
- **Effort:** The shape space for N=4 is higher-dimensional (5D vs 2D for N=3), so atlas scans are more expensive. Start with 1D slices through interesting configurations.
- **Value:** First atlas data for N=4. Tests whether the critical locus conjecture (rank drops at S₄ fixed points) holds at the next value of N.
- **Priority:** MEDIUM — exploratory.

---

## Tier 4: New Research Directions (New Code Required)

### 4.1 S₄ Representation Decomposition (N=4 Tier Structure)

- **Feeds:** new split `tier_decomposition` or extend `structure_constants`
- **New rows:** ~5–10
- **Effort:** Implement S₄ representation theory. Decompose the 62-dim N=4 level-2 algebra and the 1260-dim level-3 algebra into S₄ irreps.
- **Value:** Tests whether the beautiful S₃ tier structure (52+44+16+4=116 with integer-quantized scaling exponents) generalizes to S₄. If it does, this is strong evidence for a universal structure theory of pairwise algebras.
- **Priority:** MEDIUM-HIGH — significant theoretical value.

### 4.2 Cross-Potential Isomorphism Testing

- **Feeds:** `structure_constants` (metadata columns)
- **New rows:** 0 (augments existing rows)
- **Effort:** Write a script that takes two structure constant tensors and tests whether they define isomorphic Lie algebras (via invariants: Killing form, derived series, Casimir polynomials, or direct Lie algebra isomorphism testing).
- **Value:** The current data shows all non-harmonic potentials produce the same Killing signature, solvability, and nilpotency. But are they literally the *same* Lie algebra up to basis change? This is a stronger statement than dimensional universality.
- **Priority:** MEDIUM — theoretical depth.

### 4.3 Contextuality / Kochen-Specker Tests

- **Feeds:** extend `bell_test` or new split
- **New rows:** ~10–50
- **Effort:** Implement Kochen-Specker or Peres-Mermin type contextuality tests on the algebra. The 39 syzygies at Level 3 create algebraic dependencies that might force contextual value assignments.
- **Value:** Even though CHSH shows no Bell violation, contextuality is a separate quantum-information phenomenon. A positive result (contextuality in a classical Poisson algebra) would be remarkable.
- **Priority:** LOW-MEDIUM — speculative but interesting.

### 4.4 Time-Series: Convergence Trajectories

- **Feeds:** new split `convergence_trajectories`
- **New rows:** ~100–500
- **Effort:** For each (N, potential) configuration, record the rank at each level as a function of the number of samples used in the SVD. Currently `level4_convergence` does this at Level 4. Extending to all levels for all configurations would produce convergence curves.
- **Schema:** `N`, `d`, `potential`, `level`, `n_samples`, `rank`, `gap_ratio`, `elapsed_s`
- **Value:** Methodological: shows how many samples are needed for reliable rank determination at each level. Useful for anyone wanting to reproduce or extend the results.
- **Priority:** LOW — methodological completeness.

---

## What Would Make the Biggest Splash on Hugging Face

Ranked by likely community interest:

1. **Exponent sweep (2.3 + 3.3)** — A continuous map of algebra dimension vs. interaction law. No dataset anywhere has this. The figure alone would draw attention. The Level-2 version is nearly free.

2. **N=5 Level 3 (3.1)** — Resolving the graph-theoretic conjecture would be a headline result. The prediction new_L3(5) = 5990 is a clean, falsifiable number.

3. **Named molecular systems (1.3)** — H₃⁺ and O₃ add human interest and cross-disciplinary appeal. Chemists and spectroscopists would notice.

4. **N=9, N=10 scaling (1.1)** — Nearly free confirmation of the L2 formula. Extends a clean mathematical result.

5. **Charge sweep phase 3 (1.4)** — Maps the landscape of charge-dependent departures from universality. This is where the most scientifically interesting open question lives.

6. **Level-3 structure constants (2.4)** — A 116³ tensor of exact rational structure constants for a Poisson algebra. Nothing like this exists anywhere.

---

## How Results Feed the Pipeline

After computing new results, the pipeline automatically picks them up:

| Result type | Save to | Picked up by |
|------------|---------|-------------|
| Symbolic rank (N, d, pot) | `results/symbolic_rank/rank_N{N}_d{d}_{pot}.json` | `build_dimension_sequences()` |
| Structure constants | `results/algebra_structure/N{N}_d{d}_{pot}/structure_constants_exact.json` | `build_structure_constants()` |
| Charge sensitivity | `results/charge_sensitivity/charge_sensitivity_completion.json` | `build_charge_sensitivity()` |
| Mass sweep | `data/mass_ratio_sweep.json` | `build_mass_invariance()` |
| Level-4 bounds | `results/level4_*/results.json` | `build_level4_convergence()` |
| Atlas summaries | `atlas_figures/atlas_summary.json`, `results/atlas_full/*/summary.json` | `build_spectral_statistics()` |
| Physical systems | `results/expansion_dimseq/expansion_dimseq_completion.json` | `build_physical_systems()` |
| Bell tests | `nbody/bell_test_results/chsh_summary.json` | `build_bell_test()` |
| Scaling formulas | `results/analysis/nbody_scaling_formulas.json` | `build_scaling_formulas()` |

For new result types (exponent sweep, tier decomposition), add a new builder function to `build_dataset.py`, a new validation function to `validate_dataset.py`, and a new config entry in `dataset/README.md`.

Rebuild command after any new results:

```bash
python dataset/build_dataset.py
python dataset/validate_dataset.py
cp dataset/README.md dataset/output/README.md
```
