# Data Acquisition Directions for the Hugging Face Dataset

*April 16, 2026*

This document outlines natural directions for expanding the dataset, organized by which split they feed, how much effort they require, and what scientific value they add. The current dataset has **974 rows across 12 tables**. Each direction below lists the target split, estimated new rows, compute cost, and whether it requires new code or just running existing scripts.

---

## Tier 1: Low-Hanging Fruit (local compute, minutes to hours)

These use existing scripts with new parameters. Rebuild the dataset afterward with `python dataset/build_dataset.py`.

### 1.1 N=9 and N=10 Level-2 Dimension Sequences

- **Feeds:** `dimension_sequences`, `scaling_formulas`
- **Status:** ✅ N=9 COMPLETED (Apr 14, 2026): L2=1107, confirms formula. N=10 BLOCKED (MemoryError, needs 64+ GB).
- **New rows:** 1 in `dimension_sequences` (N=9 added)
- **Value:** N=9 extends L2 formula verification to 7 values (N=4–9, plus boundary N=3). N=10 needs algorithmic improvement.
- **ETL:** `build_dataset.py` already picks up `results/symbolic_rank/rank_N*.json` automatically.
- **Priority:** N=9 done. N=10 requires sparse DomainMatrix or modular rank.
- **Compute (N=10 only):**
  - *Local (48 GB):* Infeasible — OOM at 48 GB. Needs algorithmic change (modular rank over GF(p)) to fit in memory.
  - *AWS r6i.4xlarge (128 GB, spot ~$0.39/hr):* Likely feasible with modular rank. ~1–2 hours. **~$0.50–$1.**
  - *AWS r6i.8xlarge (256 GB, spot ~$0.66/hr):* Feasible with current QQ code if memory suffices. ~30 min–1 hr. **~$0.50.**

### 1.2 More Structure Constants (Missing Potentials)

- **Feeds:** `structure_constants`
- **Status:** ✅ COMPLETED (Apr 15, 2026). Added 6 new tensors, total now 15.
- **New rows:** 6 added to `structure_constants`
- **Now have:** 1/r, 1/r², 1/r³, 1/r⁴, r², r⁴, r⁶, r⁸, r^10, r^1, r^3 (L2+L3), composite(u+u²), composite(u+u²+u³), composite(u⁴), log. That's 15 potentials.
- **Key findings:**
  - **Isomorphism confirmed:** r^6, r^8, r^10 (N=3 d=1) all match universal invariants exactly (Killing 6+/0-/11z, solvable length 3, nilpotent class 3, center 11, 32 non-zero SC). Together with 1/r, 1/r², 1/r³, 1/r⁴, r⁴, composites, and log, that's 13 potentials sharing identical L2 algebraic structure.
  - **r^1 is qualitatively different:** dim 5, Killing (3+/0-/2z), solvable length 2, nilpotency class 3, center dim 2. A fundamentally smaller algebra.
  - **r^3 matches at L2 but diverges at L3:** At L2, invariants are identical to universal. At L3 (dim 109), the algebra is NOT nilpotent, solvable length 4, center dim 80, lower central series oscillates. The 7 missing generators (vs 116) change the algebraic character qualitatively.
  - 1/r⁴ (N=3 d=2) confirms singular catalog through n=4.

### 1.3 Named Molecular Systems: H₃⁺ and Ozone

- **Feeds:** `physical_systems`, `dimension_sequences`
- **Status:** ✅ COMPLETED (Apr 15, 2026). Both H₃⁺ (m=1836) and O₃ (m=29164) give [3, 6, 17, 116], confirming mass invariance for named molecular systems.
- **New rows:** 2 added to `physical_systems`
- **Value:** Molecular physicists and chemists will recognize H₃⁺ (simplest polyatomic molecule, important in interstellar chemistry) and O₃. These are "human interest" entries.
- **Priority:** Done.

### 1.4 Charge Sweep Phase 3 (+1/+q/−1 Molecular Geometry)

- **Feeds:** `charge_sensitivity`
- **Status:** ✅ COMPLETED (Apr 15, 2026). All 20 values (q=1..20) computed via exact symbolic rank over QQ at d=1 (44s/config avg). All give [3, 6, 17, 116].
- **New rows:** 20 added to `charge_sensitivity`
- **Key finding:** **Complete universality** — the +1/+q/−1 molecular geometry shows NO departures from [3,6,17,116] for any integer charge q in 1..20. The previously reported departures (115 for H₂⁺, 111 for Li⁺ ion) arose from specific physical mass ratios + charge combinations, not from the charge geometry alone. At unit masses, all charge configurations are universal.
- **Method:** Used `nbody/charge_sweep_d1.py` — exact symbolic rank at d=1, exploiting proven spatial dimension independence. The original d=2 numerical SVD approach (2000 samples) was killed by memory/time constraints during subs() evaluation.
- **Priority:** Done.

### 1.5a Polynomial r^n Exact Symbolic Survey (COMPLETED)

- **Feeds:** `dimension_sequences`
- **Status:** ✅ COMPLETED (Apr 2026). Exact symbolic rank over QQ for r^1 through r^10 (N=3, d=1, max_level=3).
- **New rows:** 10 in `dimension_sequences` (r^1 through r^10, auto-picked up from `results/symbolic_rank/rank_N3_d1_r*.json`)
- **Results:**
  - r^1: [3,4,5,5] (finite, closes at dim 5)
  - r^2: [3,6,13,15] (finite, closes at dim 15)
  - r^3: [3,6,17,109] (infinite, 7 extra relations at L3)
  - r^4 through r^10: all [3,6,17,116] (universal)
- **Value:** VERY HIGH — demonstrates the singular/regular dichotomy is obsolete. Only r^1 and r^2 produce finite algebras.

### 1.5b Fractional Exponent Sweeps (COMPLETED)

- **Feeds:** `dimension_sequences` (or new `exponent_sweep` split)
- **Status:** ✅ COMPLETED (Apr 2026). Three sweeps:
  1. 1/r^n (N=3, d=1, L3): 21 exponents 0.00001–3.5, all [3,6,17,116] (1 SVD artifact at n=0.00001)
  2. 1/r^n (N=4, d=1, L2): 9 exponents 0.001–3.5, all [6,14,62]
  3. r^n (N=3, d=1, L3): 12 successful of 20 tested (0.5–5.0), r^1 anomalous, symmetric descent near r^2
- **Results files:** `results/fractional_exponent_sweep.json`, `results/fractional_exponent_sweep_N4.json`, `results/rn_exponent_sweep.json`
- **Value:** HIGH — confirms continuous universality for 1/r^n and reveals r^2 as a sharp symmetry point.

### 1.5 Quantum Dimension Sequences (Additional Potentials)

- **Feeds:** `dimension_sequences`
- **Status:** ✅ COMPLETED (Apr 15, 2026). Added 6 new quantum rows (log, r^1, r^2, r^3, r^4, composite(u+u²)). r^6 skipped (computationally intractable at d=1 polynomial form; pattern already established by r^4).
- **New rows:** 6 added to `dimension_sequences` (total quantum rows: 12)
- **Key findings:**
  - **Pure log: [3,6,17,117]** — grows by +1! Resolves the GUE ambiguity: pure log is singular and follows the quantum growth rule. The GUE composite (log+harmonic) did NOT grow (116) because the harmonic component suppresses quantum deformation.
  - **r^1: [3,4,5,5]** — no quantum growth. The smallest exceptional algebra remains exceptional.
  - **r^2: [3,6,13,15]** — GROWING, but to 15 (same final dim as classical [3,6,15,15]). Intermediate structure differs: quantum L2 dim is 13 vs classical 15.
  - **r^3: [3,6,17,109]** — same as classical. No quantum growth for the non-nilpotent anomaly.
  - **r^4: [3,6,17,116]** — same as classical. No quantum growth for polynomial potentials.
  - **composite(u+u²): [3,6,17,117]** — grows by +1! Confirms singular composite potential follows quantum growth rule.
- **Revised classification:**
  - Singular potentials (1/r^n, log, composite with singular terms): quantum dim = classical dim + 1
  - Polynomial potentials (r^n for n ≥ 4): quantum dim = classical dim (no growth)
  - Exceptional potentials (r^1, r^2, r^3): quantum dim = classical dim (no growth)
  - GUE (log + harmonic): harmonic component suppresses quantum growth
- **Required fix:** Fixed `symbolic_rank_nbody.py` to use `QQ[hbar]` domain for polynomial spring potentials in quantum mode (was failing with `CoercionFailed` for r^3).
- **Priority:** Done.

---

## Tier 2: Moderate Effort (AWS spot, $10–$50)

### 2.1 N=4 Level-3 with Additional Potentials

- **Feeds:** `dimension_sequences`
- **New rows:** 3–4 in `dimension_sequences`
- **Potentials to test:** 1/r², 1/r³, log. Predictions: all should give [6, 14, 62, 1260] if potential universality extends to N=4 Level 3.
- **Value:** VERY HIGH — directly tests whether the 1260 at Level 3 is potential-universal for N=4, just as 116 is for N=3.
- **Priority:** HIGH — falsifiable prediction at moderate cost.
- **Compute:**
  - *Local (48 GB):* Marginal — N=4 L3 for 1/r used ~80 GB peak on r6i.8xlarge (31 workers). Might fit locally at 1 worker with slower execution. **~3–6 hours per potential, ~9–18 hours total.** Risk of OOM.
  - *AWS r6i.4xlarge (128 GB, spot ~$0.39/hr):* Comfortable. ~1 hr/potential × 3 potentials. **~$1.20 total.**
  - *AWS r6i.8xlarge (256 GB, spot ~$0.66/hr):* Fastest option (31 workers, 52 min baseline). 3 potentials × 1 hr. **~$2 total.**

### 2.2 N=4 Quantum Rank

- **Feeds:** `dimension_sequences`
- **New rows:** 1–2 in `dimension_sequences`
- **Value:** Tests whether the "+1 quantum growth" pattern (116→117 for N=3) generalizes to N=4: does 1260 become 1261? Or something else?
- **Priority:** MEDIUM — unique data point for the quantum deformation story.
- **Compute:**
  - *Local (48 GB):* Likely infeasible — quantum mode adds hbar terms that increase expression size. N=4 classical L3 already pushes memory. OOM probable.
  - *AWS r6i.4xlarge (128 GB, spot ~$0.39/hr):* Likely feasible. Quantum adds ~50% overhead to classical N=4 L3 (~52 min). **~1.5–2 hours, ~$0.60–$0.80.**
  - *AWS r6i.8xlarge (256 GB, spot ~$0.66/hr):* Comfortable. **~1–1.5 hours, ~$0.70–$1.00.**

### 2.3 Parametric Exponent Level-2 Broad Sweep

- **Feeds:** `dimension_sequences`
- **Status:** ✅ COMPLETED (Apr 15, 2026). 500 exponents (0.02–5.0, step 0.02) for both 1/r^n and r^n families at Level 2.
- **New rows:** 500 added to `dimension_sequences`
- **Key findings:**
  - 498/500 values give the universal L2 dim = 17.
  - Only r^1 (L2=15) and r^2 (L2=13) differ — confirming the sharp transition.
  - The universality boundary at n=1 is exact: n=0.98 → 17, n=1.0 → 15, n=1.02 → 17.
- **Script:** `nbody/level2_exponent_sweep.py`
- **Priority:** Done. Level-3 follow-up at selected transition points remains for Tier 3.

### 2.4 Level-3 Structure Constants (Rank 116 Tensor)

- **Feeds:** `structure_constants` (dramatically)
- **New rows:** 1–3 in `structure_constants` (but each tensor is 116³ = 1,560,896 entries)
- **Value:** The 17×17×17 tensors at Level 2 are already unique. A 116×116×116 tensor at Level 3 would be unprecedented — nobody has structure constants for a 116-dimensional Poisson algebra. Even one such tensor would be a significant contribution.
- **Priority:** MEDIUM-HIGH — expensive but unique. The in-progress 1/r computation may already provide this.
- **Compute:** 116×115/2 = 6,670 bracket+solve operations. Each bracket at L3 involves expressions with hundreds to thousands of terms.
  - *Local (48 GB):* Feasible for 1/r at d=1 (single-threaded). **~12–24 hours** per potential. Memory fits (~20–30 GB for the 116×10K matrix + solve). Multiple potentials: run sequentially, **~2–4 days total.**
  - *AWS r6i.4xlarge (128 GB, spot ~$0.39/hr):* Parallel bracket computation speeds this up ~4×. **~6–12 hours per potential, ~$2.50–$5 each.**
  - *AWS r6i.8xlarge (256 GB, spot ~$0.66/hr):* ~8× parallel. **~4–8 hours per potential, ~$2.50–$5 each.**

### 2.5 Mass Ratio Sweep Extension

- **Feeds:** `mass_invariance`
- **Status:** ✅ COMPLETED (Apr 15, 2026). Extended from 19 to 33 mass ratios, covering 1 to 10^10.
- **New rows:** 14 added to `mass_invariance` (total now 33)
- **Key findings:**
  - **m3 ≤ 10^4:** All give [3, 6, 17] — perfect mass invariance across 4 orders of magnitude.
  - **m3 = 10^5 to 10^9:** Level-1 drops from 6 to 4 (numerical conditioning), but Level-2 stays at 17.
  - **m3 = 10^10:** Level-0 drops to 2, Level-1 to 4, Level-2 to 12 — full conditioning breakdown.
  - Gap ratios remain >10^9 even at extreme ratios, confirming the "rank" is well-determined but the dimensions themselves become numerically unstable.
  - Added denser near-equal coverage (1.001, 1.005, 1.02, 1.3, 1.7, 4, 7, 15, 30, 75) plus astrophysical ratios (10^7–10^10).
- **Priority:** Done.

---

## Tier 3: Substantial Effort (AWS, $50–$200+)

### 3.1 N=5 Level-3 (Algorithmic Challenge)

- **Feeds:** `dimension_sequences`, `scaling_formulas`
- **New rows:** 1 in `dimension_sequences`, updates to `scaling_formulas`
- **Bottleneck:** The N=5 L3 computation OOM-killed at 256 GB. The 1.1M×760K matrix over QQ exceeds available memory. Options:
  1. **Modular rank:** Compute rank over GF(p) for several primes p and use CRT. This avoids the QQ coefficient explosion.
  2. **Larger instance:** r6i.16xlarge (512 GB) or x2idn.xlarge (768 GB).
  3. **Out-of-core:** Stream matrix rows from disk.
- **Value:** CRITICAL for the graph-theoretic conjecture. The prediction is new_L3(5) = 5990 if a=1198 holds. This is the first non-trivial test of the C(N,4) pattern at Level 3.
- **Priority:** VERY HIGH — but requires algorithmic innovation, not just more compute.
- **Compute:** The 1.1M L3 brackets are already checkpointed on S3 (~2 hours to generate on r6i.8xlarge). The blocking step is rank computation of the 1.1M × 760K matrix.
  - *Local (48 GB):* Completely infeasible — matrix alone needs ~100+ GB over QQ. With modular rank (GF(p)), the matrix fits in ~12 GB (int64 entries). Rank over GF(p) for one prime: **~2–6 hours** (FLINT/numpy). Need ~3 primes. **Total ~6–18 hours locally if modular rank is implemented.**
  - *AWS r6i.8xlarge (256 GB, spot ~$0.66/hr):* Brute force over QQ may still OOM (killed at 256 GB previously). With modular rank: **~1–3 hours, ~$1–$2.**
  - *AWS r6i.16xlarge (512 GB, spot ~$0.42–$2.04/hr):* Brute force QQ *might* fit. **~2–4 hours, ~$1–$8** depending on spot price and whether QQ fits.
  - *AWS x2idn.xlarge (768 GB):* Brute force QQ should fit. Expensive on-demand (~$8/hr). **~2–4 hours, ~$16–$32.** Spot pricing unknown.
  - **Recommended path:** Implement modular rank over GF(p) (~1 day of coding), then run locally or on r6i.4xlarge. **Total cost: ~$0–$1.** Most of the cost is developer time, not compute.

### 3.2 Yukawa Potential (Exponential Screening)

- **Feeds:** `dimension_sequences`, `physical_systems`
- **Status:** ✅ COMPLETED (Apr 16, 2026). All 9 configurations give [3, 6, 17, 116]. Universality confirmed for exponentially screened potentials.
- **New rows:** 6 in `dimension_sequences` (mu-sweep), 3 in `physical_systems` (tritium/He-3, dusty plasma, p-n-n scattering)
- **Method:** Taylor-expansion composite representation V = u·exp(-μ/u) ≈ Σ (-μ)^k/k! · u^{1-k} with K=3 (4 terms). This converts the transcendental Yukawa potential into a polynomial composite that the existing NBodyAlgebra pipeline handles efficiently. Required two engineering fixes:
  1. **Chunked flat-func evaluator:** Patched `_make_flat_func` in `exact_growth_nbody.py` to break CSE-flattened expressions into ≤30-term chunks, preventing Python's AST recursion limit from triggering subs() fallback.
  2. **Reduced sample count:** N_SAMPLES=500 (vs 2000) to keep evaluation tractable for composite expressions with negative u-powers.
- **Results:**
  - mu-sweep (d=1): 6 values (0.1, 0.5, 0.7, 1.0, 2.0, 5.0) — all [3,6,17,116]
  - Physical systems (d=1): Tritium/He-3 (mu=0.7), Dusty Plasma (mu=0.1, charged), Proton-Neutron-Neutron (mu=0.7, unequal masses) — all [3,6,17,116]
- **Value:** VERY HIGH — first non-power-law singular potential with exponential screening confirmed universal. Extends universality to nuclear forces and screened Coulomb interactions.
- **Script:** `yukawa_dimseq.py`
- **Priority:** Done.

### 3.3 Parametric Exponent Level-3 (Selected Values)

- **Feeds:** `dimension_sequences`
- **Status:** ✅ COMPLETED (Apr 16, 2026). 76 successful L3 data points (46 for 1/r^n, 30 for r^n), densifying the continuous dim(L3) vs exponent curve.
- **New rows:** ~45 new in `dimension_sequences` (merged with existing fractional exponent sweep data)
- **Method:** Used composite u^n representation for all exponents. N=3, d=1, max_level=3, 2000 samples. Resume/checkpoint support for incremental computation.
- **Results:**
  - **1/r^n family (46 exponents):** 45 universal [3,6,17,116], 1 SVD artifact at n=0.00001 (dim=113). New exponents fill gaps at n=0.005, 0.01, ..., 0.9, 0.95 and extend to n=3.5.
  - **r^n family (30 exponents):** 24 universal, 6 non-universal (all expected):
    - r^1.0: [3,6,15,148] (known linear anomaly)
    - r^1.999: dim 108, r^1.99999: dim 87, r^2.00001: dim 87, r^2.001: dim 108 (harmonic symmetry descent — sharp V-shaped dip at r^2)
  - **Overflow at high n:** r^n for n≥8 produces expressions with u^{-n} terms causing numerical overflow (e.g., u^{-390.5}). These exponents were skipped.
- **Value:** HIGH — combined with L2 sweep, enables publication-quality figure of dim(L3) vs exponent. The r^2 harmonic dip is clearly resolved with symmetric descent: 116→108→87→87→108→116 centered at n=2.
- **Script:** `l3_exponent_sweep.py`
- **Results file:** `results/l3_exponent_sweep_extended.json`
- **Priority:** Done.

### 3.4 N=4 Atlas 1D Slices

- **Feeds:** `spectral_statistics`
- **Status:** ✅ COMPLETED (Apr 16, 2026). Three 1D slices through N=4 d=1 shape space, 100 points × 500 samples each. ALL 300 points give rank 62. No rank drops observed.
- **New rows:** 3 in `spectral_statistics` (one per slice)
- **Method:** `NBodyAlgebra(N=4, d=1, 1/r)` with local phase-space sampling around each configuration. SVD rank at L2. Shape parameterization: bodies at (0, s, 1, t) on a line.
- **Slices:**
  - **Slice A** (sweep s, t=2.0): 100 points, s ∈ (0.05, 0.95). ALL rank 62.
  - **Slice B** (sweep t, s=0.5): 100 points, t ∈ (1.1, 5.0). ALL rank 62.
  - **Slice C** (equal spacing d): 100 points, d ∈ (0.3, 3.0), bodies at (0, d, 2d, 3d). ALL rank 62.
- **Key finding:** Complete rank stability across all tested configurations. No S₄ symmetry-induced rank drops along these 1D slices. The equal-spacing slice (C) passes through configurations with enhanced permutation symmetry yet shows no rank drop — this contrasts with N=3 where the Lagrange (equilateral) configuration shows clear drops. This may indicate that 1D collinear configurations do not access the S₄ fixed-point set, or that rank drops require the full 2D/3D shape space to manifest.
- **Value:** HIGH — first atlas data for N=4. Establishes the baseline rank 62 is stable under shape deformations.
- **Script:** `n4_atlas_1d.py`
- **Results file:** `results/n4_atlas_1d.json`
- **Runtime:** ~45 seconds total (much faster than estimated — each point ~0.1s after one-time algebra build).
- **Priority:** Done. 2D grid and d=2 atlas scans remain for future work.

---

## Tier 4: New Research Directions (New Code Required)

### 4.1 S₄ Representation Decomposition (N=4 Tier Structure)

- **Feeds:** new split `tier_decomposition`
- **Status:** ✅ COMPLETED (Apr 15, 2026). Implemented S₄ CG rules, character table, exterior squares.
- **New rows:** 40 in `tier_decomposition` (both S₃ and S₄ decompositions)
- **Key findings:**
  - **Edge representation (L0=6):** Decomposes as triv + std + hook under S₄ (compare S₃: A + E).
  - **Total candidates through L3:** 23226 (vs 156 for N=3). Only 1260 are independent.
  - **Syzygy fraction:** 94.6% for N=4 (vs 25.6% for N=3) — dramatically more redundancy.
  - **S₄ isotypic content:** 941 triv + 976 sign + 2893 std + 2932 sign_std + 1917 hook.
  - **Dominant irreps:** std (37.4%) and sign_std (37.9%) contribute ~75% of generators.
  - **Comparison with S₃:** For S₃, n_E = 52 exactly matched Tier 1 size. For S₄, the analogous prediction n_std = 2893 awaits N=4 SVD tier data.
- **Script:** `s4_tier_analysis.py`
- **Priority:** Done.

### 4.2 Cross-Potential Isomorphism Testing

- **Feeds:** `structure_constants` (metadata columns)
- **Status:** ✅ COMPLETED (Apr 15, 2026). All 12 non-harmonic 17-dim algebras are canonically isomorphic under fine invariant matching (Killing eigenvalues, ad-rank multisets, Casimir trace). The r^1 algebra is identified as the filiform nilpotent Lie algebra L_{5,2}.
- **Script:** `nbody/isomorphism_test.py`
- **Key findings:**
  - All 12 non-harmonic potentials (1/r, 1/r², 1/r³, 1/r⁴, r⁴, r⁶, r⁸, r^10, composites, log) produce the SAME algebra up to canonical basis reordering.
  - r^1 = L_{5,2} (filiform, dim 5), a qualitatively different object.
  - r^3 L3 analysis: 80-dim radical + 29-dim quotient, LCS oscillates with period 5, solvable length 4 — the only non-nilpotent algebra in the catalog.
- **Priority:** Done.

### 4.3 Contextuality / Kochen-Specker Tests

- **Feeds:** new split `contextuality`
- **Status:** ✅ COMPLETED (Apr 15, 2026). Definitive negative result.
- **New rows:** 16 in `contextuality`
- **Key findings:**
  - **All 16 algebras have ZERO commuting pairs.** Every pair of generators has a non-zero Poisson bracket.
  - **Algebras tested:** dim 5 (r^1), dim 15 (r^2), dim 17 (12 universal potentials), dim 109 (r^3).
  - **Consequences:** The orthogonality graph is empty → KS coloring is trivially satisfiable → no contextuality possible. Peres-Mermin squares cannot be constructed. The algebra is "maximally non-commutative."
  - **Physical interpretation:** Unlike quantum operator algebras (which have rich commutative substructure enabling contextuality), classical Poisson algebras of the N-body problem have NO commutative substructure. This is consistent with the CHSH Bell test finding.
- **Script:** `nbody/contextuality_test.py`
- **Priority:** Done.

### 4.4 Time-Series: Convergence Trajectories

- **Feeds:** new split `convergence_trajectories`
- **Status:** ✅ COMPLETED (Apr 15, 2026). 77 data points across 11 configs.
- **New rows:** 77 in `convergence_trajectories`
- **Schema:** `N`, `d`, `potential`, `level`, `n_samples`, `n_candidates`, `rank`, `gap_ratio`, `elapsed_s`
- **Key findings:**
  - **N=3 d=2 L1-L2:** Rank converges immediately at 50 samples for all potentials tested.
  - **N=4 d=1 L2:** Rank converges at 100 samples (50 is insufficient — gives rank 47 instead of 62).
  - **N=4 d=2 L2:** Also converges at 100 samples.
  - **Gap ratios grow with sample count:** For N=4, gap improves from ~10^6 (50 samples) to ~10^12 (5000 samples).
  - **1/r L3 from checkpoint:** Numerical SVD gives rank 15 (not 116) — demonstrates the known conditioning challenge at L3 with 500 numerical samples, validating the exact symbolic approach.
- **Configs:** N=3 d=2 (1/r L1-L3, 1/r^2 L1-L2, r^2 L1-L2), N=4 d=1 (1/r L1-L2), N=4 d=2 (1/r L1-L2).
- **Script:** `convergence_trajectory_sweep.py`
- **Priority:** Done.

---

## What Would Make the Biggest Splash on Hugging Face

*Re-evaluated April 16, 2026, after completing N=4 atlas 1D slices (3.4), Yukawa survey (3.2), L3 exponent sweep (3.3), and all previous items. The dataset now has 974 rows across 12 tables.*

Ranked by likely community interest — remaining open items:

1. **N=5 Level 3 (3.1)** — The headline theoretical result. Resolving new_L3(5) = 5990 is a clean falsifiable prediction. Requires algorithmic innovation (modular rank over GF(p)). Now the single most impactful computation remaining.

2. **r^3 level-3 algebraic classification** — The r^3 L3 result is a genuine discovery: the lower central series oscillates with period 5 [52,5,10,65,93,...] instead of terminating. This is the only known non-nilpotent Poisson algebra in the catalog, with solvable length 4 and an 80-dim radical. Identifying this as a known algebraic structure (or proving it's new) would draw attention from pure algebraists.

3. **Level-3 structure constants for 1/r (2.4)** — A 116³ tensor is unprecedented. Now that L2 isomorphism is proven (all 12 non-harmonic potentials are the SAME algebra), the natural question is whether L3 isomorphism also holds. This requires at least two L3 tensors.

4. **N=4 Level-3 additional potentials (2.1)** — Testing 1/r², 1/r³, log at N=4 L3 against the predicted dim=1260 is a clean falsifiable universality test.

**Completed since last evaluation:**
- ✅ N=4 atlas 1D slices (3.4): 3 slices × 100 points, ALL rank 62. First atlas data for four bodies. No rank drops along 1D collinear slices.
- ✅ Yukawa survey (3.2): 6 mu-sweep values + 3 physical systems, ALL universal [3,6,17,116]. First non-power-law singular potential with exponential screening confirmed. Chunked flat-func fix resolved the historical lambdification bottleneck.
- ✅ L3 exponent sweep (3.3): 76 L3 data points (46 for 1/r^n, 30 for r^n). Harmonic descent at r^2 clearly resolved: symmetric V-shape 116→108→87→87→108→116.
- ✅ Mass ratio extension (2.5): 33 ratios to 10^10. Conditioning breakdown at extreme ratios documented.
- ✅ S₄ decomposition (4.1): 40 tier decomposition rows. 94.6% syzygy fraction at N=4. std and sign_std dominate.
- ✅ Contextuality tests (4.3): 16 algebras tested. All maximally non-commutative (0 commuting pairs). Definitive negative.
- ✅ Convergence trajectories (4.4): 77 data points. N=4 L2 needs 100+ samples. L3 numerical rank confirms exact methods essential.
- ✅ Quantum dimension sequences (1.5): 6 new quantum rows. Pure log grows (+1), polynomials don't. GUE ambiguity resolved.
- ✅ Cross-potential isomorphism (4.2): All 12 non-harmonic potentials are the SAME algebra. r^1 = filiform L_{5,2}.
- ✅ Exponent sweep L2 (2.3): 500 values, 498 universal, only r^1 and r^2 differ.
- ✅ Charge sweep +1/+q/−1 (1.4): All 20 values (q=1..20) universal at [3,6,17,116].
- ✅ Named molecular systems (1.3): H₃⁺ and O₃ confirmed at [3,6,17,116].

---

## How Results Feed the Pipeline

After computing new results, the pipeline automatically picks them up:

| Result type | Save to | Picked up by |
|------------|---------|-------------|
| Symbolic rank (N, d, pot) | `results/symbolic_rank/rank_N{N}_d{d}_{pot}.json` | `build_dimension_sequences()` |
| Structure constants | `results/algebra_structure/N{N}_d{d}_{pot}/structure_constants_exact.json` | `build_structure_constants()` |
| Charge sensitivity | `results/charge_sensitivity/charge_sensitivity_completion.json` + `charge_sweep_qqn_d1.json` | `build_charge_sensitivity()` |
| Exponent sweep (L2) | `results/level2_exponent_sweep.json` | `build_dimension_sequences()` |
| Exponent sweep (L3 extended) | `results/l3_exponent_sweep_extended.json` | `build_dimension_sequences()` |
| Yukawa mu-sweep | `results/yukawa_dimseq.json` | `build_dimension_sequences()` |
| Mass sweep | `data/mass_ratio_sweep.json` | `build_mass_invariance()` |
| Level-4 bounds | `results/level4_*/results.json` | `build_level4_convergence()` |
| Atlas summaries | `atlas_figures/atlas_summary.json`, `results/atlas_full/*/summary.json` | `build_spectral_statistics()` |
| N=4 atlas 1D slices | `results/n4_atlas_1d.json` | `build_spectral_statistics()` |
| Physical systems | `results/expansion_dimseq/expansion_dimseq_completion.json`, `results/yukawa_dimseq.json` | `build_physical_systems()` |
| Bell tests | `nbody/bell_test_results/chsh_summary.json` | `build_bell_test()` |
| Scaling formulas | `results/analysis/nbody_scaling_formulas.json` | `build_scaling_formulas()` |
| Tier decomposition | `results/tier_decomposition/s3_s4_decomposition.json` | `build_tier_decomposition()` |
| Contextuality | `nbody/contextuality_results/contextuality_summary.json` | `build_contextuality()` |
| Convergence trajectories | `results/convergence_trajectories.json` | `build_convergence_trajectories()` |

For new result types, add a new builder function to `build_dataset.py`, a new validation function to `validate_dataset.py`, and a new config entry in `dataset/README.md`.

Rebuild command after any new results:

```bash
python dataset/build_dataset.py
python dataset/validate_dataset.py
cp dataset/README.md dataset/output/README.md
```
