# Gap Work Plan — Systematic Analysis Campaign

*Created: April 1, 2026*
*Tracks 21 identified gaps in the project, ordered by effort and impact.*

---

## Phase 1: Free Post-Processing (existing data, zero compute)

### 1.1 Spectral Depth Mining — Interior SV Landscapes
- **Status:** ✅ COMPLETED — `spectral_depth_mining.py` generates SV landscapes at indices [49,79,99,109,115] for 1/r and 1/r²
- **Data:** `atlas_output_hires/*/sv_spectra.npy` (156 SVs × 10,000 grid points per atlas)
- **Task:** Plot SV #50, #80, #100, #110 as individual shape-sphere heatmaps for 1/r and 1/r². Compare to the known SV #116 landscape.
- **Question answered:** Does the interior of the spectrum vary smoothly, or are there hidden phase boundaries at intermediate SV indices?
- **Script to create:** `spectral_depth_mining.py`
- **Output:** `spectral_depth/sv_{50,80,100,110}_landscape_{1r,1r2}.png`

### 1.2 Spectral Decay Rate Map
- **Status:** ✅ COMPLETED — decay rate and knee-index maps in `spectral_depth_mining.py` item_1_2()
- **Data:** Same `sv_spectra.npy`
- **Task:** At each grid point, fit (or compute) the decay rate of the SV spectrum (e.g., log-slope between SV #80 and #115). Plot as a shape-sphere heatmap.
- **Question answered:** Are there configurations where the spectrum drops steeply (implying near-degenerate generators) vs. gradually?
- **Script to create:** Same `spectral_depth_mining.py` or separate `spectral_decay_map.py`
- **Output:** `spectral_depth/decay_rate_landscape_{1r,1r2}.png`

### 1.3 Spectral Clustering on Shape Sphere
- **Status:** ✅ COMPLETED — k-means clustering (k=3,5,7) with cluster profiles in `spectral_depth_mining.py` item_1_3()
- **Data:** Same `sv_spectra.npy`
- **Task:** Normalize each 156-dim SV vector, run k-means or hierarchical clustering (k=3,4,5), plot cluster assignments on the shape sphere. Do natural "spectral regions" emerge?
- **Question answered:** Does the shape sphere partition into algebraically distinct zones beyond the S₃ symmetry structure?
- **Script to create:** `spectral_clustering.py`
- **Output:** `spectral_depth/spectral_clusters_k{3,4,5}_{1r,1r2}.png`

### 1.4 Clebsch-Gordan Predictions vs. Full Atlas
- **Status:** ✅ COMPLETED — Lagrange=33 doublets (E-frac=0.57), Euler=14 (0.24), Isos-90=24 (0.41). See `cg_atlas_comparison.py`
- **Data:** `clebsch_gordan_analysis.py` predictions + `atlas_targeted/*/sv_spectra.npy`
- **Task:** The CG analysis predicts doublet counts and tier sizes from S₃ representation theory. Currently compared only at the Lagrange point. Extend to the full shape sphere: at each grid point, count near-degenerate SV doublets and compare to CG prediction. Plot predicted vs. observed as a scatter plot + residual heatmap.
- **Question answered:** Does the CG decomposition predict spectral structure everywhere, or only at the S₃ fixed point?
- **Script to create:** `cg_atlas_comparison.py`
- **Output:** `spectral_depth/cg_predicted_vs_observed.png`, `spectral_depth/cg_residual_map.png`

### 1.5 Level-4 Comparison Chart
- **Status:** ✅ COMPLETED — 18 unique records. Global 200K→5604, Scalene 20K→3218, Lagrange 20K→3112, Euler 20K→2194. See `level4_comparison.py`
- **Data:** `results/level4_{global,lagrange,euler,scalene}_*/results.json`
- **Task:** Extract d(4) lower bounds and gap ratios for each configuration type at each sample count. Plot (a) bar chart of d(4) at max samples per config, and (b) convergence curves d(4) vs. sample count.
- **Question answered:** How does the level-4 rank vary across configuration types? Is the Lagrange rank drop persistent?
- **Script to create:** `level4_comparison.py`
- **Output:** `level4_comparison_chart.png`, `level4_convergence_curves.png`

### 1.6 Analytical Prediction of SV #116
- **Status:** ✅ COMPLETED — R²=0.630 correlation between analytical prediction and observed SV #116. See `sv116_analytical.py`
- **Data:** `checkpoints/level_3.pkl` (symbolic generators) + `atlas_output_hires/1_r/sv_spectra.npy`
- **Task:** Load the 116th generator from the checkpoint. Compute its symbolic gradient norm (or evaluation magnitude) as a function of (μ, φ). Compare to the observed SV #116 landscape. If they correlate, we have analytical control over the rank boundary.
- **Question answered:** Can the weakest algebraic constraint be predicted from the symbolic structure alone?
- **Script to create:** `sv116_analytical.py`
- **Output:** `spectral_depth/sv116_predicted_vs_observed.png`
- **Note:** This is the most complex of the free items — may require careful symbolic → numerical pipeline.

---

## Phase 2: Light Compute (hours, local or small AWS)

### 2.1 N=5 Level 1–2
- **Status:** ✅ COMPLETED — Exact symbolic rank [10, 25, 145] for both d=1 and d=2 (14s and 45s respectively). d-independence confirmed for N=5. Also computed N=6: [15, 39, 279] for both d=1 and d=2 (54s and 170s). N=4 [6, 14, 62] confirmed algebraically over Q. See `nbody/symbolic_rank_nbody.py`, results in `results/symbolic_rank/`.
- **Estimated time:** Seconds to minutes (local)
- **Command:** `python nbody/symbolic_rank_nbody.py -N 5 -d 1 --max-level 2`
- **Question answered:** Third AND fourth data points for d_N(k) vs. N. d(0)=C(N,2), d(1) and d(2) computed. d-independence confirmed for all N tested.
- **Impact:** HIGH — headline result for universality across N.

### 2.2 N=4 with 1/r², 1/r³, log(r)
- **Status:** ✅ COMPLETED — All three potentials give [6, 14, 62] with definitive SVD gaps (7.3×10¹²–1.8×10¹³). See `nbody/run_n4_potential_universality.py`, results in `nbody/n4_potential_universality_results.json`
- **Estimated time:** ~10s total (d=1, ~3s per potential)
- **Task:** Run `NBodyAlgebra(4, 1, "1/r2")`, `NBodyAlgebra(4, 1, "1/r3")`, `NBodyAlgebra(4, 1, "log")` through level 2 and compare to [6, 14, 62].
- **Question answered:** YES — the N=4 sequence is potential-universal. [6, 14, 62] holds for 1/r, 1/r², 1/r³, and log(r).
- **Impact:** CRITICAL — directly falsifiable prediction CONFIRMED.

### 2.3 r⁴ Potential (Regular) and 1/r⁴ (Singular)
- **Status:** ✅ COMPLETED — Both r⁴ and 1/r⁴ give exact rank [3, 6, 17, 116] through level 3 for N=3, d=2. See `nbody/symbolic_rank_nbody.py`, results in `results/symbolic_rank/`.
- **Estimated time:** ~5 min (r⁴) / ~10 min (1/r⁴)
- **Task:** Run N=3, d=2 with V ~ r⁴ (quartic spring) and V ~ 1/r⁴ (inverse quartic).
- **Result:** BOTH potentials produce [3, 6, 17, 116] — identical to 1/r, 1/r², 1/r³, log(r), and composite. The prediction that r⁴ would be finite-dimensional (like r² → dim 15) was FALSIFIED. The r⁴ quartic spring generates the same infinite-dimensional algebra as singular potentials. The harmonic r² appears to be the unique exception, not a representative of a "regular" class.
- **Question answered:** The singular/regular dichotomy does NOT predict algebra dimension. r⁴ is regular (polynomial) yet generates the same dimension sequence as 1/r. The harmonic potential r² is special because of its enhanced symmetry (oscillator algebra), not because of regularity.

### 2.4 Charge Sweep Phase 3 (+1/+q/−1)
- **Status:** CRASHED — phase 3 never completed
- **Data so far:** Phase 2 (+q/−1/−1) complete for q=1–20, all give [3, 6, 17, 116].
- **Task:** Fix the crash in the AWS sweep script and run phase 3 for the mixed-sign (+1/+q/−1) geometry.
- **Question answered:** Does the universality hold for the "molecular" charge configuration across magnitudes?

### 2.5 S₄ Tier Decomposition (N=4)
- **Status:** NOT STARTED
- **Estimated time:** Hours (algebraic computation)
- **Task:** Implement S₄ representation theory analogous to `clebsch_gordan_analysis.py`. Decompose the 62-dim N=4 level-2 algebra into S₄ irreps (trivial, sign, standard, standard⊗sign, 2D). Count tiers.
- **Question answered:** Does the S₃ tier structure generalize to S₄? Are the scaling exponents still integer-quantized?
- **Script to create:** `s4_tier_analysis.py`

### 2.6 Harmonic Dimension 15 — Representation-Theoretic Derivation
- **Status:** NOT STARTED (question posed in conjectures.md)
- **Task:** The 3-body harmonic oscillator has the Lie algebra of coupled oscillators (sp(4,ℝ) or similar). The 12D phase space should close at a predictable dimension from representation theory. Derive the number 15 from the isotropic oscillator algebra.
- **Question answered:** Is dim=15 a known Lie-algebraic quantity, or anomalous?
- **Approach:** Compute the centralizer of the harmonic Hamiltonian analytically; or simply identify the Lie algebra generators symbolically from the checkpoints.

### 2.7 H₃⁺ and Ozone Molecular Systems
- **Status:** NOT STARTED (mentioned in project_status.md as high-impact)
- **Task:** Configure and run:
  - H₃⁺: three protons with effective 1/r interaction, equal mass ≈ 1836
  - O₃ (ozone): three oxygen nuclei, mass ≈ 29,164 (16 × 1836)
- **Question answered:** Does the universality extend to molecular triatomic systems?
- **Note:** These are really just reconfirmations of mass invariance at specific physical masses, but publishing the result for named molecules has outreach value.

---

## Phase 3: Medium Compute (AWS, $10–$200)

### 3.1 Parametric Exponent Sweep (1/r^n for 1,015 values of n)
- **Status:** Script ready (`parametric_atlas_scan.py`), only n={2,−2} attempted
- **Estimated cost:** ~$13–50 (Tier 3 pragmatic hybrid)
- **Task:** Run the full sweep: n from −5 to +5 in 0.01 steps, plus special values (π, e, φ, √2). 50×50 grid at 200 samples for the coarse pass, 100×100 for ~20 interesting values.
- **Output:** Gap ratio at Lagrange vs. n (continuous curve), Jahn-Teller ring radius vs. n, phase boundary at n=0.
- **Impact:** VERY HIGH — unique result, probably a paper figure.

### 3.2 Yukawa Potential (3 Scenarios)
- **Status:** BROKEN — lambdification/OOM issues, recursion fix deployed but not confirmed
- **Scenarios:** Dusty plasma, tritium/He-3, p-n-n scattering
- **Task:** Debug the Yukawa lambda compilation pipeline. The exponential damping e^{−μr}/r creates deeply nested expressions. May need CSE-based compilation or a numerical-only evaluation path.
- **Impact:** HIGH — Yukawa is the key non-power-law singular potential.

### 3.3 Re-run 7 Retracted Gravitational Configs
- **Status:** 2 of 7 directly re-validated with SymPy ≥1.13.3; 5 inferred from mass sweep
- **Task:** Directly re-run all 7 on AWS with SymPy 1.13.3: Sun-Earth-Moon, Sun-Jupiter-Asteroid, Three Cluster Stars, Binary Star + Planet, Three Merging Galaxies, Triple BH (LISA), Binary BH + Neutron Star.
- **Question answered:** Closes the "inferred" gap in the survey — direct confirmation for all configs.
- **Impact:** MEDIUM — mostly for completeness and paper credibility.

### 3.4 Complete Interrupted Atlases
- **Status:** ✅ COMPLETED — Sun-Earth-Moon and Sun-Jupiter-Asteroid atlases finished April 7–8, 2026 on AWS spot instances. Results synced to `aws_results/`. Both show rank 91–100 at float64, consistent with conditioning expectations at extreme mass ratios (10²⁴–10³⁰ coefficient dynamic range).
- **Impact:** LOW-MEDIUM — extreme mass ratios, confirms universality at edge cases (pending symbolic rank verification).

### 3.5 Lagrange Hires 1000×1000 Scan
- **Status:** Plan complete, script designed, never deployed
- **Estimated cost:** ~$3–12
- **Task:** Deploy focused 1000×1000 scan of μ=[0.3,2.0], φ=[15°,105°] at ε=2e-4 for 1/r² (Calogero-Moser).
- **Question answered:** Resolve the concentric ring features and discrete isosceles beads at high resolution.
- **Impact:** MEDIUM — specific structural question.

---

## Phase 4: Verification & Theory

### 4.1 SageMath Independent Verification
- **Status:** NOT STARTED (listed as "ESSENTIAL" in adversarial_analysis.md)
- **Task:** Install SageMath (WSL or standalone). Independently compute [3, 6, 17, 116] for N=3, 1/r using SageMath's Poisson bracket facilities. Compare symbol-by-symbol against SymPy output.
- **Impact:** HIGH — addresses the "single CAS" criticism head-on.

### 4.2 Growth Rate Formula / Generating Function
- **Status:** NOT STARTED (but new data available)
- **Task:** With data points:
  - N=3: d(k) = [3, 6, 17, 116, ≥5604]
  - N=4: d(k) = [6, 14, 62, 1260] (exact over Q). new_L3(4) = 1198.
  - N=5: d(k) = [10, 25, 145] (exact over Q, d-independent). L3 OOM-killed.
  - N=6: d(k) = [15, 39, 279] (exact over Q, d-independent)
  - N=7: d(k) = [21, 56, 476] (exact over Q)
  - N=8: d(k) = [28, 76, 748] (exact over Q, cross-verified)
  - Search OEIS for subsequences and related sequences
  - Test recurrence relations (e.g., d(k+1) = a·d(k)² + b·d(k) + c)
  - Test exponential/super-exponential fits
  - Compare growth to known Lie algebra dimension formulas
  - Scaling formulas: L0 = C(N,2), L1 = N(3N-5)/2, L2 = N(4N²-9N+3)/2 (N≥4)
  - New-per-level: new_L0 = C(N,2), new_L1 = N(N-2), new_L2 = 12·C(N,3) (N≥4), new_L3 = 1198·C(N,4)? (only N=4 data, boundary case)
  - Graph-theoretic conjecture: new_L_k ~ f(k)·C(N,k+1) for large N.
- **Question answered:** Is there a pattern, or is the sequence "wild"?

### 4.3 Level-4 Bound Improvement
- **Status:** Current best: d(4) ≥ 5,604 at 200K samples, gap NOT definitive
- **Task:** Continue pushing sample count (300K? 500K?) or switch to mpmath high-precision rank computation to resolve the true d(4).
- **Note:** The mpmath computation was at 4.4% (667/15,000 rows) when spot-reclaimed. Instance terminated. Checkpoint on S3. Needs relaunch on new instance.

### 4.4 Symbolic Rank Over Q (Exact Algebraic Dimension)
- **Status:** ✅ COMPLETED — Rank [3, 6, 17, 116] confirmed at 5 specific mass points (exact over Q) and with symbolic masses (exact over Q(m1,m2,m3)). Mass invariance is now an algebraic theorem for all positive masses. See `symbolic_rank.py`, results in `results/symbolic_rank/`.
- **Motivation:** All rank determinations in this project use numerical SVD on float64 evaluations. While the SVD gap is large (6–13 orders of magnitude) at generic configurations, at extreme mass ratios the float64 rank drops to 91–100 due to coefficient dynamic range exceeding float64 precision. An attempted term-group factoring approach (splitting generators by magnitude order and evaluating groups independently) produced inflated ranks (up to 200) that were artifacts of spurious column independence, not genuine algebraic structure. The mpmath "ground truth" approach also proved unreliable for this purpose. The only definitive answer comes from exact linear algebra over Q.
- **Approach:** The 156 generators are built from scratch via `build_hamiltonians()` and `poisson_bracket()` — polynomials in 15 variables `(x1,y1,...,u12,u13,u23)` with rational coefficients (parameterized by masses). The monomial-coefficient matrix (156 × 128,925) is extracted via `sympy.Poly` and its rank computed by exact Gaussian elimination (`DomainMatrix.rank()`) over Q or Q(m1,m2,m3), immune to numerical noise.
- **Result:** Rank [3, 6, 17, 116] confirmed at 5 specific rational mass points (over Q, locally) and with symbolic masses (over Q(m1,m2,m3), AWS r6i.8xlarge, 3.2 hours). Mass invariance is an algebraic theorem for all positive masses.
- **Impact:** VERY HIGH — converted the central conjecture from "supported by numerical evidence" to "proven by exact computation."
- **Reference:** See supplemental memo `poisson_numerical_robustness_memo.md` for discussion of the numerical robustness landscape. Note: the mpmath claims in that memo should be treated with caution; the symbolic approach supersedes them.

### 4.5 Algebra Structure Extraction (Structure Constants, Killing Form, Derived Series)
- **Status:** ✅ COMPLETED (level 2, rank 17) — Structure constants computed exactly over Q for 1/r, 1/r⁴, r⁴, and r² (harmonic). Killing form, derived/lower central series, center all computed. See `nbody/symbolic_rank_nbody.py --structure`, results in `results/algebra_structure/`.
- **Key result:** All non-harmonic potentials (1/r, 1/r⁴, r⁴) produce **identical** algebraic structure at level 2: Killing signature (6+, 0-, 11 zero), solvable (length 3), nilpotent (class 3), center dim 11, derived series [17, 14, 3, 0]. The harmonic r² is structurally opposite: Killing (14+, 0-, 1 zero), not solvable, not nilpotent, center dim 1, perfect algebra [L,L]=L.
- **Next steps:** Scale to level 3 (rank 116) on AWS; compare structure constants between potentials to test isomorphism.
- **Impact:** VERY HIGH — first classification of the Poisson algebra beyond dimension counting. Proves structure universality, not just dimensional universality.

### 4.6 Noise Plateau Mapping
- **Status:** NOT STARTED
- **Motivation:** The SVD-gap rank determination depends on a threshold choice (`1e-8 × σ_max` currently). Understanding how the reported rank varies as a function of this threshold — across mass configurations, spatial positions, and potential types — is valuable both for validating the robustness of the 116 result and for characterizing the conditioning structure of the algebra.
- **Approach:** At each configuration, sweep the SVD threshold from 10⁻¹ down to 10⁻¹⁵ and plot the reported dimension. Three possible outcomes: (a) a clean plateau at 116 (strong robustness), (b) continuous variation (threshold artifact), (c) irregular steps (hierarchical scale structure). Run at equal mass, moderate ratio (10:1), and extreme ratio (10⁶:1 and beyond).
- **Expected value:** Produces a single figure showing plateau width vs. mass ratio — a reviewer-accessible demonstration of robustness that complements the symbolic rank result.
- **Impact:** MEDIUM-HIGH — unique methodological contribution; explains *mechanistically* why float64 undercounts at extreme mass ratios (Born-Oppenheimer decoupling analogy).
- **Reference:** `poisson_numerical_robustness_memo.md` (Sections 3, 4, 7). Note: some claims in Section 2 regarding mpmath verification are not reliable; this research path would provide the actual data.

---

## Tracking

Mark items with status as work proceeds:
- ⬜ Not started
- 🔄 In progress
- ✅ Complete
- ❌ Blocked

| ID | Item | Status |
|----|------|--------|
| 1.1 | Spectral depth mining — SV landscapes | ✅ |
| 1.2 | Spectral decay rate map | ✅ |
| 1.3 | Spectral clustering | ✅ |
| 1.4 | CG predictions vs. full atlas | ✅ |
| 1.5 | Level-4 comparison chart | ✅ |
| 1.6 | Analytical SV #116 prediction | ✅ |
| 2.1 | N=5 Level 1–2 (+ N=6) | ✅ |
| 2.2 | N=4 with 1/r², 1/r³, log(r) | ✅ |
| 2.3 | r⁴ and 1/r⁴ potentials | ✅ |
| 2.4 | Charge sweep phase 3 | ⬜ |
| 2.5 | S₄ tier decomposition | ⬜ |
| 2.6 | Harmonic dim=15 derivation | ⬜ |
| 2.7 | H₃⁺ and ozone | ⬜ |
| 3.1 | Parametric exponent sweep | ⬜ |
| 3.2 | Yukawa debugging + run | ❌ |
| 3.3 | Re-run 7 retracted gravitational configs | ⬜ |
| 3.4 | Complete interrupted atlases | ✅ |
| 3.5 | Lagrange hires 1000×1000 | ⬜ |
| 4.1 | SageMath verification | ⬜ |
| 4.2 | Growth rate formula | ⬜ |
| 4.3 | Level-4 bound improvement | ⬜ |
| 4.4 | Symbolic rank over Q | ✅ |
| 4.5 | Algebra structure extraction | ✅ |
| 4.6 | Noise plateau mapping | ⬜ |
| 4.7 | 1D structure cross-section (singularity detection) | ✅ |
| 4.8 | Level-3 structure extraction (rank 116) | 🔄 |
| 4.9 | Symbolic Gram determinant sweep (rationalized Bareiss) | ✅ |
