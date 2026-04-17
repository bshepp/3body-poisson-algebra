# Three-Body Poisson Algebra — Project Status & Roadmap

*Last updated: April 16, 2026*

---

## 1. Atlas Campaign Status

### Completed Configurations (18)

| # | Configuration | Potential | Masses | Charges | Notes |
|---|---|---|---|---|---|
| 1 | Gravitational baseline | 1/r | 1:1:1 | (1,1,1) | Reference atlas |
| 2 | Coulomb mixed | 1/r | 1:1:1 | (1,+1,-1) | |
| 3 | Positronium- (Ps-) | 1/r | 1:1:1 | (1,-1,-1) | |
| 4 | Coulomb asymmetric | 1/r | 1:1:1 | (2,-1,-1) | Strongest gap ratio |
| 5 | Coulomb strong | 1/r | 1:1:1 | (3,-1,-1) | |
| 6 | Calogero-Moser charged | 1/r^2 | 1:1:1 | (2,-1,-1) | |
| 7 | Cubic potential | 1/r^3 | 1:1:1 | (1,1,1) | Jahn-Teller ring |
| 8 | Cubic charged | 1/r^3 | 1:1:1 | (2,-1,-1) | |
| 9 | Logarithmic (vortex) | log(r) | 1:1:1 | — | Transcendental singularity |
| 10 | H- ion | 1/r | 1836:1:1 | (1,-1,-1) | Born-Oppenheimer decoupling |
| 11 | H2+ molecular ion | 1/r | 1836:1836:1 | (1,+1,-1) | Two-heavy + one-light |
| 12 | Positronium- ion | 1/r | 1:1:1 | (1,-1,-1) | |
| 13 | Lithium ion Li+ | 1/r | 12789:1:1 | (3,-1,-1) | |
| 14 | Muonic helium | 1/r | 7294:1:207 | (2,-1,-1) | 3-scale mass hierarchy |
| 15 | Binary star + planet | 1/r | 1:1:0.001 | — | |
| 16 | Triple BH (LISA) | 1/r | 1:0.01:1e-5 | — | |
| 17 | Sun-Earth-Moon | 1/r | 1:3e-6:3.7e-8 | — | Extreme mass ratio; ranks 102–108 due to SVD conditioning |
| 18 | Sun-Jupiter-Asteroid | 1/r | 1:9.5e-4:1e-10 | — | Most extreme mass ratio; ranks 91–100 due to SVD conditioning |
| 19 | Irrational exponent (π) | 1/r^π | 1:1:1 | (1,1,1) | 50×50 atlas, 200 samples, level 3. 60% rank 116, unique ranks {90–116}. |
| 20 | Irrational exponent (e) | 1/r^e | 1:1:1 | (1,1,1) | 50×50 atlas, 200 samples, level 3. 63.2% rank 116, unique ranks {89–116}. |
| 21 | Irrational exponent (φ) | 1/r^φ | 1:1:1 | (1,1,1) | 50×50 atlas, 200 samples, level 3. 63.4% rank 116, unique ranks {106–116}. |

Additional completed work:
- Harmonic oscillator 1/r^-2 atlas (100x100) — finite algebra, rank 15
- 1/r^2 (Calogero-Moser, equal mass) atlas (100x100) — rank 116 at 87.9%, completed in 3.6h on c6i.4xlarge with 16 workers. Instance terminated.
- 1/r^2 vs 1/r^-2 triptych rendered (`aws_results/atlas_figures/triptych_1r2_vs_1r-2.png`)
- All atlas figures rendered (singles + triptychs) in `aws_results/atlas_figures/`
- Bell test completed locally (results in `nbody/bell_test_results/`)
- S3 data fully synced locally (9.04 GB), verified by `audit_atlas_data.py`
- Data integrity audit: 17/42 configs clean, 3 fixed from S3 stale sync
- Sun-Earth-Moon atlas (100x100, 800 samples) — completed April 7, 2026 in 6h28m on r6i.4xlarge (on-demand, 16 workers). 10,000/10,000 valid points, unique ranks 102–108. Dynamic range 10²⁰–10²⁶ limits SVD rank detection.
- Sun-Jupiter-Asteroid atlas (100x100, 800 samples) — completed April 8, 2026 in 9h33m on r6i.4xlarge (on-demand, 16 workers). 10,000/10,000 valid points, unique ranks 91–100. Dynamic range 10²⁵–10³² (most extreme system computed).

### Completed — Energy Bound Search (April 10, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **g_q symmetry verification** | Commutes with P_total and X_cm (confirmed). |
| 2 | **Quantum commutant of H_total** | rank(ad_H) = 116 over QQ[hbar], commutant dim = **40** (156 − 116). |
| 3 | **Classical commutant of H_total** | rank(ad_H) = 115 over QQ, commutant dim = **41** (156 − 115). |
| 4 | **Comparison** | **SMALLER** — quantum commutant (40) < classical (41). Quantization *removes* one conservation law. |
| 5 | **Kernel vector analysis** | All 40 quantum kernel vectors have rational (hbar-independent) coefficients. Mostly level-3 generators. |
| 6 | **Casimir construction** | Level-2 Casimir: rank 17, Killing rank 0, center dim 17. |
| 7 | **Interpretation** | The 117th generator does NOT participate in any conserved combination with H_total. Energy bound via this approach is not possible. The quantum deformation strictly reduces symmetry (Case C). |

### Completed — N-Body Rank Sweep (April 11, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **N=4 d=1 Level 3 exact rank** | **[6, 14, 62, 1260]**. new_L3(4) = 1198. Computed in 3140s (52 min) on r6i.8xlarge (31 workers). |
| 2 | **N=3 d=1 Level 3 validation** | [3, 6, 17, 116]. Matches previous result. 8.2s. |
| 3 | **N=5 d=1 Level 3** | **OOM killed** (exit -9). Generated 1,115,280 L3 brackets (checkpointed to S3). Rank of 1.1M × 760K matrix over QQ exceeds 256 GB RAM. |
| 4 | **N=6 d=1 Level 3** | **OOM killed**. Died during L2 bracket generation. Infeasible on current hardware. |
| 5 | **Graph-theoretic test** | new_L3(4) = 1198 with C(4,4) = 1. Consistent with a = 1198 but N=4 is a boundary case. Critical test is N=5 (prediction: new_L3 = 5990 if a = 1198). |
| 6 | **Instance cleanup** | All 5 EC2 instances terminated. All data synced to S3. ~$23 compute cost. |

### Blocked — N=5 d=1 Level 3

The N=5 L3 computation is **memory-blocked**: the 1.1M × 760K matrix
over QQ cannot be rank-computed in 256 GB. Each worker consumes ~27 GB.
The 1.1M L3 brackets are checkpointed on S3. Options to unblock:
1. Algorithmic: modular rank (compute over GF(p) for several primes)
2. Hardware: r6i.16xlarge (512 GB) or x2idn.xlarge (768+ GB)
3. Out-of-core: stream matrix rows from disk

### Completed — N=7 d=1 Level 2 (April 11, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **N=7 d=1 Level 2 exact rank** | [21, 56, 476]. Computed in 50.9s on r6i.4xlarge (15 workers). |
| 2 | **L1 formula verification** | L1(7) = 7·16/2 = 56. **Matches** formula N(3N−5)/2. Now verified for N=3–8. |
| 3 | **Old L2 cubic check** | Predicted 477, observed 476. Discrepancy of 1 (at N=8 it was 4). |
| 4 | **L2 formula RESOLVED** | The old cubic (13N³−42N²+83N−120)/6, fitted from N=3-6, was polluted by a boundary effect at N=3. The correct formula is **L2(N) = N(4N²−9N+3)/2** for N≥4, arising from new_L2 = 12·C(N,3). At N=3, new_L2 = 11 (boundary: 12−1). |
| 5 | **Graph-theoretic structure** | new_L0 = C(N,2) [edges], new_L1 = N(N−2) [wedges], new_L2 = 12·C(N,3) [triangles]. The algebra sees K_N subgraph structure at each bracket depth. |

### Completed — N=8 d=1 Level 2 (April 11, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **N=8 d=1 Level 2 exact rank** | [28, 76, 748]. Computed in 190.8s on r6i.4xlarge (15 workers). |
| 2 | **L1 formula verification** | L1(8) = 8(3·8−5)/2 = 76. **Matches** formula N(3N−5)/2. |
| 3 | **L2 formula falsification** | L2(8) predicted 752 by cubic formula, observed **748**. Discrepancy of 4. |
| 4 | **Implication** | The cubic L2(N) = (13N³−42N²+83N−120)/6, fitted from N=3,4,5,6, is **not the correct formula**. A higher-degree polynomial or different functional form is needed. N=7 data would help disambiguate. |
| 5 | **Cross-version verification** | Independently reproduced locally on SymPy 1.12 (887.9s, single-threaded Windows). Exact match with AWS result (SymPy 1.14.0). Both give [28, 76, 748]. |

### Stalled (1)

| # | Task | Progress | Status |
|---|---|---|---|
| 1 | **Level-4 mpmath rank computation** | 667/15,000 rows (4.4%) | Spot reclaimed, instance terminated. Rank=667, plateau=0. Checkpoint on S3. Needs relaunch on new instance. |

### Completed — GUE Log-Gas / Prime Distribution (April 11, 2026)

**Result: UNIVERSALITY CONFIRMED.** All singular-potential configs produce **[3, 6, 17, 116]**, including the GUE composite Hamiltonian. The Poisson algebra of the Dyson log-gas belongs to the same universality class as Newtonian gravity.

| # | Config | Potential | Confinement | Result | Status |
|---|--------|-----------|-------------|--------|--------|
| 1 | Pure log-gas | log(r) | none | **[3, 6, 17, 116]** | Complete |
| 2 | **GUE composite** | log(r) | harmonic ω=1 | **[3, 6, 17, 116]** | Complete |
| 3 | Penning trap (1D) | 1/r | harmonic ω=1 | **[3, 6, 17, 116]** | Complete |
| 4 | Harmonic only | r² (pairwise) | — | **[3, 6, 13, 15]** | Complete |

- **Instance:** i-0255259da7fdfb045 (r6i.2xlarge spot, ~$0.15/hr)
- **Parameters:** N=3, d=1, level 3, 500 samples, seed 42
- **Total compute:** 178.5 seconds
- **Code:** `primes/run_gue_logas.py`, launched via `primes/launch_gue.py`
- **Results:** `primes/results/gue_comparison.json`
- **Significance:** The algebraic structure governing zeta zero correlations (and hence prime distribution via the explicit formula) is in the same universality class as gravity.

**Additional results:**
- **Quantum GUE (Moyal bracket):** log potential with harmonic confinement gives **[3, 6, 17, 116]** under quantum deformation — no +1 growth for the GUE composite (unlike 1/r^n which grows to 117). See `primes/results/quantum_gue.json`.
- **Hilbert-Pólya search:** HP operator search on the log-gas algebra yields a semisimple algebra (Killing signature [116+, 0−, 0 zero]) — a genuine closed Lie algebra at level 3, unlike the nilpotent 1/r algebra. See `primes/results/hp_search/hp_search_results.json`.
- **Finite-N GUE comparison:** Level-2 spacing statistics (var=0.117, ⟨r⟩=0.639) analyzed against finite-N GUE reference ensembles. The variance suppression (0.117 < GUE asymptotic 0.178) is a genuine finite-size effect from the 3-eigenvalue constraint. See `primes/finite_n_gue_comparison.py`.
- **Multi-potential ⟨r⟩ comparison:** Coadjoint orbit spacing ratio computed for all available potentials — confirms ⟨r⟩ ≈ 0.64 is universal across singular potentials. See `primes/multi_potential_r_comparison.py`.

### Completed — Polynomial r^n Exact Symbolic Survey (April 2026)

| # | Potential | Sequence | Notes |
|---|----------|----------|-------|
| 1 | **r^1** (linear) | **[3, 4, 5, 5]** | Finite algebra, closes at dim 5. Unique: L1=4 (not 6). |
| 2 | **r^2** (harmonic) | **[3, 6, 13, 15]** | Finite algebra, closes at dim 15. Enhanced SU(d) symmetry. |
| 3 | **r^3** (cubic) | **[3, 6, 17, 109]** | Infinite but 7 extra relations at L3. |
| 4 | **r^4** (quartic) | **[3, 6, 17, 116]** | Universal — matches singular potentials. |
| 5 | **r^5** | **[3, 6, 17, 116]** | Universal. |
| 6 | **r^6** | **[3, 6, 17, 116]** | Universal. |
| 7 | **r^7** | **[3, 6, 17, 116]** | Universal. |
| 8 | **r^8** | **[3, 6, 17, 116]** | Universal. |
| 9 | **r^9** | **[3, 6, 17, 116]** | Universal. |
| 10 | **r^10** | **[3, 6, 17, 116]** | Universal. |

All computed with exact symbolic rank over QQ using `symbolic_rank_nbody.py` (N=3, d=1, max_level=3).
Results in `results/symbolic_rank/rank_N3_d1_r*.json`.

**Key finding:** The harmonic oscillator r^2 is not representative of "regular" potentials.
Universality [3,6,17,116] holds for ALL r^n with n ≥ 4. The singular/regular dichotomy
is replaced by: the dimension sequence distinguishes only the exceptional potentials r^1
(linear), r^2 (harmonic), and r^3 (cubic) from the universal class.

### Completed — Fractional Exponent Sweeps (April 2026)

| # | Sweep | N | d | max_level | n_samples | Exponents tested | Key result |
|---|-------|---|---|-----------|-----------|-----------------|------------|
| 1 | 1/r^n (N=3) | 3 | 1 | 3 | 500–2000 | 21 (0.00001–3.5) | All [3,6,17,116] except SVD artifact at n=0.00001→113 |
| 2 | 1/r^n (N=4) | 4 | 1 | 2 | 1000 | 9 (0.001–3.5) | All [6,14,62] — universal at N=4 |
| 3 | r^n (N=3) | 3 | 1 | 3 | 1000 | 20 (0.5–5.0, 8 SVD failures) | 12 successful: r^1 anomalous [3,6,15,148], symmetric descent near r^2, generic [3,6,17,116] for n≥2.01 |

Results in `results/fractional_exponent_sweep.json`, `results/fractional_exponent_sweep_N4.json`,
`results/rn_exponent_sweep.json`.

### Completed — Structure Constants Expansion Campaign (April 15, 2026)

Expanded exact structure constant catalog from 9 to 15 potentials. All computed at N=3.

| # | Potential | N | d | Level | Dim | Killing | Solvable | Nilpotent | Center | Notes |
|---|----------|---|---|-------|-----|---------|----------|-----------|--------|-------|
| 1 | r^6 | 3 | 1 | 2 | 17 | (6+,0-,11z) | len 3 | class 3 | 11 | Universal — matches 1/r |
| 2 | r^8 | 3 | 1 | 2 | 17 | (6+,0-,11z) | len 3 | class 3 | 11 | Universal — matches 1/r |
| 3 | r^10 | 3 | 1 | 2 | 17 | (6+,0-,11z) | len 3 | class 3 | 11 | Universal — matches 1/r |
| 4 | r^1 | 3 | 1 | 2 | 5 | (3+,0-,2z) | len 2 | class 3 | 2 | Qualitatively different |
| 5 | r^3 | 3 | 1 | 2 | 17 | (6+,0-,11z) | len 3 | class 3 | 11 | Matches universal at L2 |
| 6 | r^3 | 3 | 1 | 3 | 109 | (29+,0-,80z) | len 4 | **NO** | 80 | Diverges at L3: not nilpotent, LCS oscillates |
| 7 | 1/r^4 | 3 | 2 | 2 | 17 | (6+,0-,11z) | len 3 | class 3 | 11 | Completes singular catalog through n=4 |

**Key findings:**
- **Isomorphism conjecture strongly supported:** 13 non-harmonic potentials share identical L2 algebraic invariants (Killing 6+/0-/11z, 32 non-zero SC, solvable length 3, nilpotent class 3, center 11). Includes both singular (1/r through 1/r^4) and polynomial (r^4 through r^10) families.
- **r^1 is qualitatively different:** 5-dimensional algebra with distinct Killing signature. Only potential with L1 rank 4 instead of 6.
- **r^3 L3 is qualitatively different:** Despite matching universal pattern at L2, the L3 algebra is NOT nilpotent (lower central series oscillates: [109, 106, 103, 95, 52, 5, 10, 65, 93, 52, 5]), solvable length increases to 4, and center is massive (80/109 dimensions).

Results in `results/algebra_structure/N3_d1_r{1,3,6,8,10}/` and `results/algebra_structure/N3_d2_1r4/`.
Structure constant tensors in `results/symbolic_rank/rank_N3_d1_r{1,3,6,8,10}.json` and `rank_N3_d2_1r4.json`.

### Completed — Cross-Potential Isomorphism Test (April 15, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **Isomorphism test (L2 algebras)** | All 12 non-harmonic 17-dim algebras are canonically isomorphic under fine invariant matching (Killing eigenvalues, ad-rank multisets, Casimir trace). |
| 2 | **r^1 identification** | Identified as the filiform nilpotent Lie algebra L_{5,2}. |
| 3 | **r^3 L3 deep analysis** | 80-dim radical + 29-dim quotient. LCS oscillates with period 5: [52,5,10,65,93,...]. Solvable length 4. Only non-nilpotent algebra in the catalog. |

Script: `nbody/isomorphism_test.py`.

### Completed — Level-2 Exponent Sweep (April 15, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **500 exponents at L2** | 1/r^n and r^n for n=0.02..5.0 (step 0.02). 498/500 give universal L2 dim=17. |
| 2 | **Only exceptions** | r^1 (L2=15) and r^2 (L2=13). |
| 3 | **Transition boundary** | Sharp: n=0.98→17, n=1.0→15, n=1.02→17. |

Script: `nbody/level2_exponent_sweep.py`. Results: `results/level2_exponent_sweep.json`.

### Completed — Charge Sweep Phase 3 (April 15, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **(+1,+q,−1) q=1..20** | All 20 give [3,6,17,116]. Complete universality — no departures at unit masses. |
| 2 | **Method** | Exact symbolic rank over QQ at d=1 (44s/config avg). |
| 3 | **Implication** | Previously reported departures (115 for H₂⁺, 111 for Li⁺) arose from specific mass ratios + charge combinations, not charge geometry alone. |

Script: `nbody/charge_sweep_d1.py`. Results: `results/charge_sensitivity/charge_sweep_qqn_d1.json`.

### Completed — Named Molecular Systems (April 15, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **H₃⁺ (m=1836)** | [3, 6, 17, 116]. Mass invariance confirmed. |
| 2 | **O₃ (m=29164)** | [3, 6, 17, 116]. Mass invariance confirmed. |

Script: `nbody/named_molecular_systems.py`.

### Completed — Yukawa Potential Survey (April 16, 2026)

**UNIVERSALITY CONFIRMED for all Yukawa configurations.**

Uses Taylor-expansion composite representation: `V = u·exp(-μ/u) ≈ Σ (-μ)^k/k! · u^{1-k}` (K=3, 4 terms). This converts the transcendental Yukawa potential into a composite polynomial that the existing pipeline handles efficiently.

| # | Configuration | mu | d | Result | Time |
|---|---|---|---|---|---|
| 1 | mu=0.1 (long Debye) | 1/10 | 1 | **[3, 6, 17, 116]** | 835s |
| 2 | mu=0.5 | 1/2 | 1 | **[3, 6, 17, 116]** | 610s |
| 3 | mu=0.7 (nuclear) | 7/10 | 1 | **[3, 6, 17, 116]** | 590s |
| 4 | mu=1.0 | 1 | 1 | **[3, 6, 17, 116]** | 526s |
| 5 | mu=2.0 (short-range) | 2 | 1 | **[3, 6, 17, 116]** | 507s |
| 6 | mu=5.0 (very short) | 5 | 1 | **[3, 6, 17, 116]** | 525s |

Physical systems:

| # | System | Category | mu | Result | Time |
|---|---|---|---|---|---|
| 1 | Tritium / He-3 (3 nucleons) | nuclear | 7/10 | **[3, 6, 17, 116]** | 836s |
| 2 | Dusty Plasma (screened Coulomb) | plasma | 1/10 | **[3, 6, 17, 116]** | 1281s |
| 3 | Proton-Neutron-Neutron | nuclear | 7/10 | **[3, 6, 17, 116]** | 969s |

**Significance:** Yukawa (screened Coulomb) is the first non-power-law singular potential with exponential screening confirmed to produce the universal dimension sequence. This extends universality beyond homogeneous potentials to the physically important class of exponentially screened interactions.

Script: `yukawa_dimseq.py`. Results: `results/yukawa_dimseq.json`.

### Completed — Extended L3 Exponent Sweep (April 16, 2026)

Densified the Level-3 exponent sweep from ~41 to 76 successful data points across 1/r^n and r^n families. Fills gaps for a continuous dim(L3) vs exponent curve.

| Family | Successful | Universal [3,6,17,116] | Non-universal | Key findings |
|---|---|---|---|---|
| 1/r^n | 46 | 45 | 1 (n=0.00001: dim=113, SVD artifact) | Continuous universality across all tested n |
| r^n | 30 | 24 | 6 (r^1, r^{near 2}) | Sharp transitions at r^1 and r^2 |

Non-universal results (all expected):
- `1/r^0.00001`: 113 (SVD artifact at near-zero exponent)
- `r^1.0`: [3,6,15,148] (known linear anomaly)
- `r^1.999`, `r^1.99999`, `r^2.00001`, `r^2.001`: dim 87–108 (harmonic symmetry descent)

Script: `l3_exponent_sweep.py`. Results: `results/l3_exponent_sweep_extended.json`.

### Completed — N=4 Atlas 1D Slices (April 16, 2026)

First atlas data for four bodies (N=4, d=1, 1/r). Three 1D slices through the 2D shape space of collinear 4-body configurations, 100 points × 500 samples each.

| # | Slice | Parameterization | Points | Unique Ranks |
|---|---|---|---|---|
| 1 | Sweep s (t=2.0) | x=(0, s, 1, 2.0), s∈(0.05, 0.95) | 100 | {62} |
| 2 | Sweep t (s=0.5) | x=(0, 0.5, 1, t), t∈(1.1, 5.0) | 100 | {62} |
| 3 | Equal spacing | x=(0, d, 2d, 3d), d∈(0.3, 3.0) | 100 | {62} |

**Key finding:** Complete rank stability — ALL 300 points show rank 62 (the generic N=4 d=1 L2 dimension). No rank drops detected at any configuration, including the equal-spacing slice which passes through configurations with enhanced permutation symmetry. This contrasts with N=3 where the Lagrange (equilateral) point shows clear rank drops. The absence of drops in 1D may indicate that the S₄ fixed-point set requires the full 2D or 3D shape space to manifest rank reductions.

Script: `n4_atlas_1d.py`. Results: `results/n4_atlas_1d.json`. Runtime: ~45s total.

### Not Yet Started (4)

| # | Task | Notes |
|---|---|---|
| 1 | ~~Quantum rank for other potentials~~ | **DONE** — r², r⁴, 1/r², 1/r³, 1/r⁴ all tested. See Universality Classification above. |
| 3 | ~~**Parametric exponent sweep**~~ | **DONE** — L2 broad sweep (500 values) and L3 extended sweep (76 values). |
| 4 | ~~**Dusty plasma Yukawa**~~ | **DONE** — [3, 6, 17, 116]. See Yukawa Survey above. |
| 5 | ~~**Tritium/He-3 Yukawa**~~ | **DONE** — [3, 6, 17, 116]. See Yukawa Survey above. |
| 6 | **SageMath verification** | Independent verification of dimension sequence. SageMath not yet installed. |
| 8 | **Structure extraction at level 3 (rank 116)** | Level-2 structure computed for 6 potentials. Level 3 requires AWS. |

### Completed — Neural Network Algebra (April 2026)

| # | Task | Result |
|---|---|---|
| 1 | **Neural algebra computation** | Gradient-product coupling V_ij = (∂L/∂w_i)(∂L/∂w_j)/2 for 3-layer linear network. Exact rank over ℚ: **[3, 6, 17, 119]**. |
| 2 | **Gravitational comparison** | SVD gap-ratio 1.44×10¹⁴ confirms [3, 6, 17, 116]. Neural has 3 extra generators at level 3. |
| 3 | **Extra generator identification** | Indices 125, 128, 152 — all S₃ standard rep (E), all involve H₂₃. Bracket structures: {{K₂,H₂₃},{K₃,H₂₃}}, {{K₂,K₃},H₂₃}, {{K₃,H₂₃},H₂₃}. |
| 4 | **Universality implication** | Level-2 universality ([3, 6, 17]) extends to non-physical couplings. Level-3 dimension (116) is specific to singular pairwise potentials. |

### Completed — Algebra Structure (April 9, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **N-body exact rank scaling** | N=3: [3,6,17,116], N=4: [6,14,62,1260], N=5: [10,25,145], N=6: [15,39,279], N=7: [21,56,476], N=8: [28,76,748]. L3 available for N=3,4. d-independence confirmed for N=3–6. |
| 2 | **New potentials (r⁴, 1/r⁴)** | Both give [3, 6, 17, 116]. The r² (harmonic) finite algebra is special, not representative of all regular potentials. |
| 3 | **Structure constants (exact/Q)** | Computed for 1/r, 1/r⁴, r⁴, r² at level 2 (rank 17). |
| 4 | **Killing form & signature** | Non-harmonic: (6+, 0-, 11 zero). Harmonic: (14+, 0-, 1 zero). |
| 5 | **Derived/lower central series** | Non-harmonic: solvable (length 3), nilpotent (class 3). Harmonic: neither. |
| 6 | **Center dimension** | Non-harmonic: 11/17. Harmonic: 1/15. |
| 7 | **SVD component saving** | `--save-svd` flag added to `exact_growth.py` and `nbody/exact_growth_nbody.py`. |

### Completed — Quantum Commutator Algebra (April 10, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **Quantum engine** | `QuantumNBodyAlgebra` with Moyal bracket, validated (Jacobi, hbar→0 limit). |
| 2 | **N=3 d=1 quantum rank** | [3, 6, 17, **117**] — one extra dimension vs classical 116. |
| 3 | **N=3 d=2 quantum rank (L2)** | [3, 6, 17] — matches classical through level 2. |
| 4 | **N=3 d=2 quantum rank (L3)** | [3, 6, 17, **117**] — confirmed +1 in 2D (AWS r6i.4xlarge, 3562s). |
| 5 | **Post-Newtonian 1PN** | -1/r - 1/r² gives [3, 6, 17, 116] — GR correction does not change algebra. |
| 6 | **Post-Newtonian 2PN** | -1/r - 1/r² - 1/r³ — level 3 in progress locally. |
| 7 | **117th generator identified** | Sum-of-squares proof: g = −(9/4)[(A−B)² + A²], negative semi-definite. Legendre P₃ structure. |

### Completed — Quantum Universality Classification (updated April 15, 2026)

| # | Potential | Quantum dims | Classical dims | Growth? |
|---|---|---|---|---|
| 1 | **1/r** (Coulomb) | [3, 6, 17, **117**] | [3, 6, 17, 116] | **+1** |
| 2 | **1/r^2** (Calogero-Moser) | [3, 6, 17, **117**] | [3, 6, 17, 116] | **+1** |
| 3 | **1/r^3** (dipole) | [3, 6, 17, **117**] | [3, 6, 17, 116] | **+1** |
| 4 | **1/r^4** | [3, 6, 17, **117**] | [3, 6, 17, 116] | **+1** |
| 5 | **log** (pure) | [3, 6, 17, **117**] | [3, 6, 17, 116] | **+1** |
| 6 | **composite(u+u²)** | [3, 6, 17, **117**] | [3, 6, 17, 116] | **+1** |
| 7 | **GUE (log+harmonic)** | [3, 6, 17, 116] | [3, 6, 17, 116] | No (harmonic suppresses) |
| 8 | **r^1** (linear) | [3, 4, 5, 5] | [3, 4, 5, 5] | No (exceptional) |
| 9 | **r^2** (harmonic) | [3, 6, 13, 15] | [3, 6, 15, 15] | Different intermediate (13 vs 15 at L2), same final dim |
| 10 | **r^3** (cubic) | [3, 6, 17, 109] | [3, 6, 17, 109] | No (exceptional) |
| 11 | **r^4** (quartic) | [3, 6, 17, 116] | [3, 6, 17, 116] | No |

**Classification (revised):**
- All singular potentials (1/r^n, log, composite with singular terms): **quantum dim = classical + 1**
- Polynomial potentials (r^n, n ≥ 4): **quantum dim = classical** (no growth)
- Three exceptional potentials (r^1, r^2, r^3): **quantum dim = classical** (no growth)
- GUE composite (log + harmonic): harmonic component suppresses quantum growth despite log being singular
- **Key insight from pure log:** The GUE composite's lack of growth is NOT because log is non-singular — pure log DOES grow. The harmonic (r^2) term is what suppresses quantum deformation.

### Completed — N-body Scaling Formulas (updated April 11, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **L1 formula** | L1(N) = N(3N-5)/2. Verified for N=3,4,5,6,7,8. New-per-level: N(N-2). |
| 2 | **L2 formula (original)** | L2(N) = (13N^3 - 42N^2 + 83N - 120)/6 **FALSIFIED at N=7,8**: predicts 477/752, observed 476/748. Cubic was fitted from N=3,4,5,6 (4 points determine a unique cubic) but does not extrapolate. This was a legitimate inference that failed at the next data points. |
| 2b | **L2 formula (resolved)** | With N=7 filling the gap, the falsification is resolved. new_L2 = 12·C(N,3) for N≥4 (boundary: 11 at N=3). Cumulative: **L2(N) = N(4N²−9N+3)/2** for N≥4. The true formula IS cubic — a different cubic. The original was polluted by the N=3 boundary effect. Verified for N=4,5,6,7,8. |
| 3 | **L3 formula** | Two data points: N=3 L3=116 (new_L3=99), N=4 L3=1260 (new_L3=1198). Both are boundary cases for C(N,4): C(3,4)=0, C(4,4)=1. If a=1198, prediction: new_L3(5)=5990. N=5 L3 OOM-killed on 256 GB; needs algorithmic improvement or larger instance. |
| 4 | **Degree pattern** | Leading polynomial degree in N: L0~N^2, L1~N^2, L2~N^3. Confirmed cubic growth at L2. |

### Completed — Mass Invariance Statement (April 10, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **Symbolic rank** | Rank over Q(m_1,m_2,m_3) = [3,6,17,116] — proves generic mass invariance. |
| 2 | **Specific verifications** | Tested at masses (1,1,1), (1,2,3), (1,1,5/2), (1,1,1/100), (1,1,1/10000) — all [3,6,17,116]. |
| 3 | **Formal statement** | Algebra dimension is determined solely by potential type, N, and d. Masses play no role. |

### Completed — Non-Integrability Analysis Framework (April 10, 2026)

| # | Task | Result |
|---|---|---|
| 1 | **Evidence assessment** | Strong computational evidence for non-integrability, not yet a complete proof. |
| 2 | **Key gap** | Need to show algebra is infinite-dimensional (level 4 data) or connect to Morales-Ramis theory. |
| 3 | **Integrable comparison** | Harmonic (r^2) is the ONLY tested potential producing a finite algebra (dim 15). All others grow to 116+. |

### Completed — 117th Generator Analysis (April 10, 2026)

The extra quantum dimension was fully characterized via `analyze_117th.py`:

| Property | Result |
|---|---|
| **Compact form** | 10 terms in relative coords (down from 66) |
| **Sum-of-squares** | g = −(9/4)·[(A−B)² + A²] where A = u₁₂⁴·Φ₁, B = u₁₃⁴·Φ₂ |
| **Sign** | **Negative semi-definite** (algebraic proof, verified 100K samples) |
| **Angular structure** | Φᵢ = wᵢ(5wᵢ²−3) = 2P₃(wᵢ) — third Legendre polynomial (octupole) |
| **Translation invariance** | Yes — depends only on relative positions |
| **Scaling** | Homogeneous degree −8 (g → λ⁻⁸g) |
| **Collision divergence** | 1/r¹⁴ (vs classical max ~1/r⁸) |
| **Equilateral triangle** | −7065/256 ≈ −27.6 (does NOT vanish) |
| **Permutation symmetry** | Not S₃-symmetric (bracket-specific), but S₃-average is nonzero and also negative |
| **Conserved?** | NO — [G, H_total] ≠ 0 (919 terms). Element of pair-Hamiltonian Lie algebra, not a constant of motion. |
| **Physical meaning** | Operator inequality Ĝ ≤ Ĝ_classical; quantum deformation opens asymmetrically (bounded above). |
| **Collision comparison** | 1/r¹⁴ divergence vs Bohm potential 1/r² and Darwin term 1/r³ — probes short-distance structure at higher order. |

---

## 2. Physical Systems Catalog

A comprehensive catalog of three-body systems that can be (or have been)
investigated with the pairwise Poisson algebra framework. Systems marked
with a check have completed atlas computations.

### Gravitational

| System | Masses | Status | Notes |
|--------|--------|--------|-------|
| Equal-mass three-body | 1:1:1 | Done | Baseline atlas |
| Sun-Earth-Moon | 1 : 3e-6 : 3.7e-8 | Done | Ranks 102–108; 10K/10K valid. Dynamic range limits SVD. |
| Sun-Jupiter-Asteroid | 1 : 9.5e-4 : 1e-10 | Done | Ranks 91–100; 10K/10K valid. Most extreme mass ratio. |
| Three stars (globular cluster) | comparable | — | Equal-mass baseline covers this |
| Binary star + planet | 1:1:0.001 | Done | |
| Three galaxies merging | comparable | — | Equal-mass baseline covers this |
| Hierarchical triple BH (LISA) | 1:0.01:1e-5 | Done | |
| Binary BH + neutron star | — | — | Dense cluster dynamics |

### Atomic / Coulomb

| System | Potential | Masses | Charges | Status |
|--------|----------|--------|---------|--------|
| Helium atom | 1/r | 7294:1:1 | (2,-1,-1) | Done (muonic He) |
| Lithium ion Li+ | 1/r | 12789:1:1 | (3,-1,-1) | Done |
| H- ion | 1/r | 1836:1:1 | (1,-1,-1) | Done |
| Positronium- ion | 1/r | 1:1:1 | (1,-1,-1) | Done |
| Muonic helium | 1/r | 7294:1:207 | (2,-1,-1) | Done |
| H2+ molecular ion | 1/r | 1836:1836:1 | (1,+1,-1) | Done |

### Molecular

| System | Notes |
|--------|-------|
| H3+ triatomic ion | Simplest polyatomic molecule, astrophysically important. Three nuclei with effective potentials. Whether classical nuclear dynamics is integrable matters for reaction rate theory and molecular spectroscopy. |
| Ozone (O3) | Three oxygen nuclei. Similar framework applies. |

### Nuclear

| System | Potential | Status | Notes |
|--------|----------|--------|-------|
| Tritium / He-3 (3 nucleons) | Yukawa (mu=0.7) | **Done** | [3, 6, 17, 116] via Taylor composite K=3 |
| Proton-neutron-neutron scattering | Yukawa (mu=0.7) | **Done** | [3, 6, 17, 116] via Taylor composite K=3 |
| Three-quark bound states | — | — | QCD, not Coulomb. Beyond current framework. |

### Plasma / Charged Particles

| System | Potential | Status |
|--------|----------|--------|
| Three ions in a Penning trap | 1/r (+1,+1,+1) + harmonic | Done (Penning trap config) |
| Three dust grains in dusty plasma | Yukawa (mu=0.1) | **Done** — [3, 6, 17, 116] |
| Three vortices in 2D fluid | log(r) | Done |

### Post-Newtonian / GR

| System | Notes |
|--------|-------|
| Three compact objects with 1PN | Design doc exists (`schwarzschild_scope.md`). Composite potential 1/r + 1/r^2 + ... |
| Kozai-Lidov oscillation | Inner binary + distant perturber with GR precession. |

### Exotic / Theoretical

| System | Potential | Notes |
|--------|----------|-------|
| Three magnetic monopoles | 1/r^2 | Hypothetical. Calogero-Moser potential. Atlas complete (87.9% rank 116). |
| Three anyons in 2D | — | Not yet modeled. |
| Dark matter halo scattering | — | Not yet modeled. |

### High-Impact Applications

Several communities would find specific results immediately relevant:

- **Calogero-Moser community**: The planar CM Poisson algebra is
  infinite-dimensional despite 1D integrability. This is a surprise
  the integrable systems / mathematical physics community would want
  to understand. They know the 1D system closes; showing the 2D
  system doesn't, with the same algebraic signature as gravity, is
  a novel result.

- **Restricted three-body / astrodynamics**: The mass invariance
  conjecture predicts the algebra shouldn't care when one mass goes
  to zero. The Sun-Earth-Moon (3.7×10⁻⁸) and Sun-Jupiter-Asteroid
  (10⁻¹⁰) atlases are now complete, confirming the algebra produces
  non-trivial rank across the full shape sphere even at extreme mass
  ratios. Detected ranks (102–108 and 91–100 respectively) are below
  the canonical 116 due to SVD conditioning at dynamic ranges of
  10²⁰–10³², not due to algebraic closure. The restricted problem
  appears to preserve the infinite-dimensional algebra structure.

- **Vortex dynamics / fluid mechanics**: Three point vortices with
  log(r) interaction confirmed [3, 6, 17, 116], establishing
  universality for a non-power-law singularity. This result would
  interest the atmospheric science and oceanography communities.

- **Molecular physics**: For triatomic molecules like H3+, the
  question "is the classical nuclear dynamics integrable" matters
  for reaction rate theory and molecular spectroscopy. The framework
  gives a direct answer.

---

## 3. Parametric Exponent Sweep Plan

### Motivation

The completed atlas campaign revealed a Jahn-Teller-like ring in the
1/r^3 gap landscape: the Lagrange equilateral point is a local gap
minimum surrounded by an annulus of gap maxima. This ring is barely
visible for 1/r and absent for regular potentials. How does the
algebraic landscape evolve continuously as a function of the potential
exponent n in V ~ 1/r^n?

### The Parametric n Trick

For 1/r^n with n as a SymPy Symbol, the entire Poisson algebra can be
built ONCE and swept at runtime. The bracket structure is identical for
all n — only exponents and certain coefficients change. One symbolic
build (~10 min), one lambdify (~2 min), then sweep n as a runtime
parameter. This eliminates 1,014 redundant symbolic builds.

Confirmed working: `lambdify` with n as the 16th argument produces
correct numerical output for n = 1, 2, 3, pi, -1. Both NumPy and
C code generation confirmed.

### Sweep Regions

| Region | Range of n | Step | Count | Physics |
|--------|-----------|------|-------|---------|
| Sub-Coulomb | 0.01 to 0.99 | 0.01 | 99 | Weak singularity to near-constant |
| Coulomb to strong | 1.00 to 5.00 | 0.01 | 401 | Standard to dipole to extreme |
| Negative (confining) | -5.00 to -0.01 | 0.01 | 500 | V ~ r^\|n\|, confinement |
| Special values | pi, e, sqrt(2), phi, etc. | — | ~15 | Irrational / transcendental |

**Total: ~1,015 atlas computations.**

### Special Values of Interest

| Value | n | Why |
|-------|---|-----|
| Coulomb | 1 | Baseline (have it) |
| Calogero-Moser | 2 | Known integrable system |
| Dipole-dipole | 3 | Jahn-Teller ring observed |
| Golden ratio | 1.618... | Irrational, tests number-theoretic effects |
| pi | 3.14159... | Transcendental |
| e | 2.71828... | Transcendental |
| sqrt(2) | 1.41421... | Algebraic irrational |
| 1/2 | 0.5 | Square-root potential — soft singularity |
| -1 | -1 | Linear confining (V ~ r), QCD-inspired |
| -2 | -2 | Harmonic (V ~ r^2), integrable calibration |
| -4 | -4 | Quartic confinement |
| 0+ (limit) | 0+ | Logarithmic (already tested separately) |

### Implementation Tiers

**Tier 1: Parametric Python (~$200-400)**

Modify `stability_atlas.py` to accept n as a Symbol. 50x50 grid, 200
samples for the coarse sweep; 100x100 for interesting values. One core
per job on c6a.xlarge spot ($0.04/hr).

**Tier 2: C Code Generation (~$50-100)**

Use SymPy `ccode()` to generate C evaluators with n as a double
parameter, link against LAPACK for SVD. 10-50x faster evaluation.
Produces a standalone `atlas_sweep` binary.

**Tier 3: Pragmatic Hybrid (~$13-50) — Recommended**

- 50x50 grid at 200 samples = 1/8 the cost of 100x100 at 400
- Batch 50 values of n per spot instance = amortize the 10-min boot
- One c6a.xlarge per batch at $0.04/hr spot
- ~1,015 values / 50 per instance = 21 instances, ~15h each = **~$13 total**
- Re-run at 100x100 for the ~20 most interesting n values (~$30 more)
- **Total under $50** for the full sweep

### Negative Exponents (Confining Potentials)

Physically very different: V ~ r^\|n\| means the potential grows with
distance (confinement, no escape). Key differences:

- No singularity at r=0 (collision is smooth)
- Potential barrier at large r
- n = -2 (harmonic oscillator) is integrable with special structure

The `u_ij = 1/r_ij` substitution means `u_ij^(-|n|) = r_ij^|n|`,
which SymPy handles natively.

### Expected Science

- Gap ratio at Lagrange vs n: continuous curve, possible cusps
- Jahn-Teller ring radius vs n: should tighten for steeper potentials
- Phase boundary at n=0 between singular (infinite) and regular (finite) algebra
- Behavior at integrable points (n=2, n=-2): any distinctive signature?
- Confining regime (n<0): fundamentally different — no collision singularity

---

## 4. Infrastructure Notes

### Multiprocessing (implemented March 26, 2026)

`full_atlas_scan.py` now supports `--workers N` for parallel column
evaluation via `multiprocessing.Pool`. On a 16-vCPU instance, this
gives ~8x speedup over single-threaded execution. Combined with the
lambdify fix (eliminating xreplace fallback), the 1/r^2 atlas dropped
from ~28h/$28 to ~3.5h/$2.50.

### Lambdify Fallback Chain (fixed March 26, 2026)

`exact_growth.py` uses a 4-layer fallback for compiling generators:

1. Standard `sp.lambdify(cse=False)` — fast when it works
2. Flat-file no-CSE — writes expression via `pycode()` to temp .py,
   bypasses `compile()` recursion limit
3. Flat-file with CSE — same but with CSE pre-flattening
4. Point-by-point `xreplace` — last resort, ~1000x slower

For 1/r^2: 92 generators via layer 1, 63 via layer 2, zero xreplace.

### AWS Cost Summary

Previous campaign ran 19 instances single-threaded for ~48 hours,
costing roughly $800-1000. With multiprocessing, equivalent work
would cost ~$100-125. Future runs should always use `--workers`.

### S3 Sync Caveat (discovered March 27, 2026)

`aws s3 sync` compares files by size and timestamp. Numpy arrays with
fixed dimensions produce identical file sizes regardless of content
(e.g., a 100×100 float64 gap_map.npy is always 80128 bytes whether
filled or zeros). This caused `s3 sync` to skip downloading updated
arrays, leaving stale local copies from interrupted runs.

**Solution**: Use `aws s3 cp --recursive` for numpy data, or add
`--exact-timestamps` to `s3 sync`. Created `audit_atlas_data.py` to
detect and fix stale syncs across all atlas configs.

### Structure Cross-Section: Singularity Detection (April 9, 2026)

New capability: numerical structure constant sweeps across parameter space.
Script `nbody/structure_cross_section.py` evaluates generators at localized
phase-space samples and solves for C_ijk by least-squares.

Key findings from 1D sweep (mu=0.05..5.0, phi=pi/3, 200 points):
- Singular potentials (1/r, 1/r^4) produce SC norm variation of 16-23 OOM
- Smooth potential (r^4) varies only 5 OOM
- Condition numbers: 1/r^4 reaches 10^13, r^4 stays under 10^3
- Center dim instability: 1/r^4 varies {0..10}, r^4 stable at 0
- Confirms that singularities are detectable through algebra conditioning

### Symbolic Gram Determinant Sweep (completed, April 9, 2026)

Exact symbolic Gram determinants computed for all three potentials
using rationalized Bareiss (DomainMatrix over QQ[mu,s]). All
potentials give det(G) = 0 on the 1D mu-manifold, confirming the
17 generators are linearly dependent when restricted from the full
phase space to a 1D configuration family.

LCM denominators: 1/r has mu^10*(mu^2-mu+1)^6, r^4 has 1, 1/r^4
has mu^22*(mu^2-mu+1)^11. The mu^k factor encodes collision
singularity strength; mu^2-mu+1 has no real roots.

### Hugging Face Dataset Pipeline (April 17, 2026)

The project maintains a structured Hugging Face dataset built from computation results. The pipeline (`dataset/build_dataset.py`) reads JSON result files and produces **13 Parquet tables (993 rows total)**. The builder deduplicates symbolic rank entries, keeping only the highest `max_level` for each (N, d, potential, bracket_type). The `neural_algebras` split (added April 17) covers Poisson algebras of linear neural network training dynamics across 12 coupling types and 7 universality classes at L=3.

| Split | Source file(s) | Rows | Description |
|-------|---------------|------|-------------|
| `neural_algebras` | `results/neural_algebras/*.json` | 21 | Poisson algebras of linear neural networks across 12 coupling types, depth L=2..5, width k=1..3, 3 losses, 2 activations; includes `universality_class_L3` label (7 classes: A_119_gradient, B_115_directional, C_111_gradient_sum, D_104_gradient_cubic, E_87_natural_gradient, F_62_cross_entropy, G_47_hessian) |
| `dimension_sequences` | `results/symbolic_rank/rank_N*.json`, `primes/results/gue_comparison.json`, `results/yukawa_dimseq.json`, `results/l3_exponent_sweep_extended.json`, etc. | ~685 | Cumulative rank at each bracket level per (N, d, potential); includes 12 quantum (Moyal) rows, 6 Yukawa mu-sweep, ~76 extended L3 exponent sweep |
| `structure_constants` | `results/algebra_structure/*/structure_constants_exact.json` | 16 | Exact rational C^k_ij tensors at level 2 (and level 3 for r^3) |
| `charge_sensitivity` | `results/charge_sensitivity/*.json` | ~38 | Charge-independence tests |
| `mass_invariance` | `data/mass_ratio_sweep.json` | 33 | Mass ratio sweep (1 to 10^10) with SVD spectra |
| `level4_convergence` | `results/level4_*/results.json` | ~19 | Level-4 lower bounds |
| `spectral_statistics` | `atlas_figures/atlas_summary.json`, `results/atlas_full/*/summary.json`, `results/n4_atlas_1d.json` | ~17 | Rank distributions across phase space (incl. 3 N=4 1D slices) |
| `physical_systems` | `results/expansion_dimseq/expansion_dimseq_completion.json`, `results/yukawa_dimseq.json` | 17 | Named physical systems (helium, Sun-Earth-Moon, H₃⁺, O₃, Yukawa nuclear/plasma, etc.) |
| `bell_test` | `nbody/bell_test_results/chsh_summary.json` | 9 | CHSH Bell inequality tests |
| `scaling_formulas` | `results/analysis/nbody_scaling_formulas.json` | 5 | Closed-form L_k(N) formulas with verification status |
| `tier_decomposition` | `results/tier_decomposition/s3_s4_decomposition.json` | 40 | S₃ and S₄ CG representation decompositions |
| `contextuality` | `nbody/contextuality_results/contextuality_summary.json` | 16 | KS/PM contextuality tests (all non-contextual) |
| `convergence_trajectories` | `results/convergence_trajectories.json` | 77 | SVD rank convergence vs sample count |

**After any computation campaign**, rebuild the dataset:

```bash
python dataset/build_dataset.py      # reads JSON → writes Parquet
python dataset/validate_dataset.py   # validates structure + content
cp dataset/README.md dataset/output/README.md  # update dataset card
```

The validation script checks row counts, schema integrity, flattened dimension column consistency, and YAML frontmatter.

### Extended Dimension Sequence Campaign (April 14, 2026)

Computed exact symbolic rank (over QQ) for the 1/r potential, d=1, across a wide range of N values:

| N range | Levels computed | File pattern | Time per run |
|---------|----------------|-------------|--------------|
| 3–9 | L0, L1, L2 | `rank_N{N}_d1_1r.json` | seconds–106 min |
| 11–26 | L0, L1 | `rank_N{N}_d1_1r_L1.json` | 11s–28 min |
| 27–50 | L0 | `rank_N{N}_d1_1r_L0.json` | <1s–8 min |

**N=10 L2 failed** with `MemoryError` — the DomainMatrix construction requires ~56 GB (535K generators x 13K monomials, dense over QQ). Would need 64+ GB free RAM or a sparse rank algorithm.

Key L1 sequence (new_per_level at L1 = dim(L1) - dim(L0)):

| N | L0 | L1 | new@L1 |
|---|----|----|--------|
| 3 | 3 | 6 | 3 |
| 4 | 6 | 14 | 8 |
| 5 | 10 | 25 | 15 |
| 6 | 15 | 39 | 24 |
| 7 | 21 | 56 | 35 |
| 8 | 28 | 76 | 48 |
| 9 | 36 | 99 | 63 |
| 11 | 55 | 154 | 99 |
| 15 | 105 | 300 | 195 |
| 20 | 190 | 550 | 360 |
| 26 | 325 | 949 | 624 |

**Verified scaling formulas:**
- L0: dim(L0) = N(N-1)/2 — confirmed for all N=3..50
- L1: dim(L1) = N(3N-5)/2 — **newly discovered**, verified for all N=3..26 (23 data points, exact match)
- L1 new: new@L1 = N(N-2)

N=10 L2 failed with `MemoryError` during DomainMatrix construction (535K generators x 13K monomials, dense over QQ, requires ~56 GB).

### Level-3 Structure Extraction (in progress, April 9–12, 2026)

Running on AWS (i-003c53042d76de01b, r6i.4xlarge): exact symbolic
structure constants for the rank-116 algebra at N=3 d=2 1/r.
6,670 Poisson bracket pairs. Expected 6-24 hours.

Four comparison scripts developed for cross-potential level-3 structure analysis:
- `nbody/symbolic_level3_compare.py` — exact QQ structure constants (37h/potential on AWS)
- `nbody/numerical_level3_compare.py` — numerical SVD-based comparison
- `nbody/fast_level3_compare.py` — finite-difference fast comparison
- `nbody/compare_level3_structure.py` — checkpoint-based comparison pipeline
