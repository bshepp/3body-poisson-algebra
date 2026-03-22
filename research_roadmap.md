# Research Roadmap: Poisson Algebra of the Three-Body Problem

This document catalogs all identified future analysis directions, ordered
by expected impact and computational cost.  Items marked [DONE] have been
completed; [IN PROGRESS] are currently running; the rest are prioritized
by estimated value per compute-hour.

---

## Completed Work

| Milestone | Key result |
|-----------|-----------|
| Exact symbolic engine | Dimension sequence [3, 6, 17, 116] via SymPy + polynomial u_ij representation |
| Mass invariance | Sequence independent of mass ratios (tested 20+ configs including Tsygvintsev) |
| Potential comparison | Harmonic (r^2) closes at dim 15; Calogero-Moser (1/r^2) matches Newton (1/r) at [3,6,17,116] |
| Level 4 lower bound | d(4) >= 4,501 at generic config (30K samples, definitive SVD gap); d(4) >= 3,112 at Lagrange (20K) |
| High-res atlas (eps=5e-3) | 100x100 grid, all 3 potentials, full SV spectra |
| S3 symmetry discovery | Gap ratio valleys trace the three isosceles curves; Lagrange point is S3 fixed point |
| SV landscape analysis | SV #116 varies 4000x across shape sphere; spectral fingerprints differ between 1/r and 1/r^2 |
| Multi-epsilon atlas | COMPLETE: 2 potentials (1/r, 1/r^2) x 6 epsilon values, 100x100 each, full SV spectra |
| Isosceles bead discovery | Discrete rank-drop beads along symmetry curves in 1/r^2; concentric ring features at Lagrange |
| Clean panel renders | Overlay-free multi-epsilon panels revealing hidden structure |
| 1000x1000 atlas | 880/1000 rows complete (1/r, eps=5e-3); failure cause identified: near-collision timeouts |
| Spatial dimension independence | d=1,2,3 all give [3,6,17,116]; sequence is d-independent (Mar 2026) |
| ND engine (`exact_growth_nd.py`) | Parameterized by d_spatial ∈ {1,2,3}; validated against known 2D results |
| N-body engine (`exact_growth_nbody.py`) | Parameterized by N, d, potential; validated against N=3 results (Mar 2026) |
| N=4 dimension sequence | **[6, 14, 62]** through L2, definitive SVD gap 3.4×10¹¹ (Mar 2026) |
| N=4 mass invariance | Confirmed: 3 configs (equal, hierarchical, mixed) all give [6, 14, 62] |
| N=4 d-independence | Confirmed: d=1,2,3 all give [6, 14, 62] |
| 1/r³ potential (N=3) | [3, 6, 17, 116] — matches 1/r and 1/r², universality across pole orders |
| Charge-sign invariance (helium) | **All-attractive, all-repulsive, mixed Coulomb (+2,-1,-1) all give [3,6,17,116]** (Mar 2026) |
| Charges engine extension | `NBodyAlgebra` now supports `charges` parameter for mixed-sign Coulomb interactions |
| **Paper 1 (preprint.tex)** | Dimension sequence, mass invariance, potential comparison (Mar 2026) |
| **Paper 2 (paper2_s3_filtration.tex)** | S₃ tier decomposition, jet filtration, syzygies (Mar 2026) |
| **Paper 3 (paper3_universality.tex)** | N=4 sequence, d-independence, 1/r³, charge-sign, universality conjecture (Mar 2026) |

---

## 1. Multi-Epsilon Atlas [DONE]

**Goal**: Map how local Poisson algebra rank evolves with sampling scale
across the full shape sphere.

**Epsilon values**: [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]

**Derived quantities** (all computed):
- Rank-drop onset map: largest epsilon where rank first drops below 116
- Differential rank-drop: extra ranks lost beyond generic numerical floor
- Gap sensitivity: d(log gap)/d(log eps) as function on shape sphere
- Rank-vs-epsilon curves at notable configurations
- Clean panels (no overlaid symmetry lines) revealing hidden structure
- Isosceles bead analysis: discrete extra rank-drop spots along symmetry curves
- Zoomed Lagrange-center renders at eps=2e-4

**Status**: COMPLETE. All 12 scans (2 potentials x 6 epsilon values)
finished Feb 26, 2026. Derived analysis and clean renders completed
Feb 28, 2026. Data in S3 under `atlas_output_hires/`.

**Key discovery**: Removing the overlaid symmetry lines revealed
**discrete "beaded" rank-drop spots** along isosceles curves in the
1/r^2 potential, and **concentric ring features** centered on the
equilateral Lagrange point.  These are configurations where extra
dynamical symmetries create additional Poisson bracket dependencies.

**Script**: `multi_epsilon_atlas.py`, `render_zoomed.py`

**Diminishing returns**: Below eps=1e-4, numerical noise dominates.
Above eps=5e-3, rank is uniformly 116 everywhere.  Best structure
visible at eps=2e-4.

---

## 2. Spatial Dimension Independence [DONE]

**Goal**: Determine whether the dimension sequence depends on spatial
dimension d.

**Method**: Created a parameterized engine (`3d/exact_growth_nd.py`)
that takes d_spatial ∈ {1,2,3} and computes exact symbolic Poisson
brackets.  Separate `3d/` directory with no imports from parent project.

**Results**:

| d | Phase space | d(0) | d(1) | d(2) | d(3) | Time (L3) |
|---|-------------|------|------|------|------|-----------|
| 1 | 6D | 3 | 6 | 17 | 116 | 44s |
| 2 | 12D | 3 | 6 | 17 | 116 | ~10 min |
| 3 | 18D | 3 | 6 | 17 | 116 | ~3.2 hr |

**Conclusion**: The dimension sequence is completely independent of
spatial dimension.  This was NOT predicted — the original conjecture
expected d-dependence, and falsifiable prediction #3 specifically
anticipated a different 3D result.

**Impact on conjecture**: The universal signature conjecture is
STRONGER than originally stated.  The sequence depends only on N
and the singularity class (singular vs regular) — not on d, masses,
or pole order.

**Bug note**: A resume-logic bug in the checkpoint system initially
produced d_3D(3) = 102.  Fixed by correcting the condition for
reconstructing computed pairs from `sum(levels) < start_level` to
`max(levels) < start_level - 1`.  See session_log.md for details.

**Status**: COMPLETE (Mar 15, 2026).  Data in `3d/checkpoints_d{1,2,3}/`.

**Priority**: N/A (completed).

---

## 3. Spectral Depth Mining

**Goal**: Exploit the full 156-dimensional singular value spectrum saved
at each of the 10,000 grid points, rather than just SVs #115-116.

**Proposed analyses**:
- Plot SVs #50, #80, #100, #110 as individual landscapes on the shape sphere
- Compute spectral decay rate (how fast SVs fall off) as a continuous
  function -- configurations where the spectrum drops steeply may have
  qualitatively different algebraic structure
- Identify which SV index shows the maximum spatial variation across the
  grid -- this is the most "configuration-sensitive" generator
- Spectral clustering: group configurations by spectral shape similarity
  to discover natural regions on the shape sphere

**Compute cost**: Negligible -- all data already exists in `sv_spectra.npy`.
Pure post-processing in NumPy + matplotlib.

**Priority**: HIGH.  Free analysis on existing data.

---

## 4. High-Resolution Lagrange Region Scan [IN PROGRESS]

**Goal**: Resolve the concentric ring features and isosceles bead
structure discovered in the multi-epsilon atlas at much higher resolution.

**Method**: 1000x1000 focused scan of the Lagrange region using the
proven `atlas_1000.py` distributed infrastructure on AWS.

**Parameters**:
- Region: mu=[0.3, 2.0], phi=[15°, 105°] (centered on equilateral point)
- Potential: 1/r² (Calogero-Moser) — shows richest structure
- Epsilon: 2e-4 — best feature visibility
- 10 blocks of 100 rows, parallelized across instances

**Timing estimate** (r6i.4xlarge, 15 workers):
- Per block: ~2.25 hours (including 15 min algebra build)
- 10 parallel instances: ~2.5 hours total, ~$8-12
- Single instance: ~22 hours, ~$3-4

**Safety**: The focused phi range [15°, 105°] has minimum inter-body
distance r_min=0.259, avoiding the near-collision timeout issue that
affected the original 1000x1000 scan (see below).

**Script**: `hires_lagrange_aws.py` (to be created from `atlas_1000.py`)

**Status**: Plan complete, awaiting deployment.  Local timing test
confirmed 1.05s per point.  See `.cursor/plans/` for full plan.

**Supersedes**: The earlier "targeted high-resolution strips" idea.
A full 2D scan is more informative than 1D strips now that we know
there is 2D ring structure to resolve.

**Priority**: MEDIUM.  Quick, cheap, answers specific structural questions.

---

## 5. Level 4 at Selected Configurations

**Goal**: Check whether the *growth rate* of the algebra (not just the
current dimension) varies with configuration.

**Current state**: Level 4 computed at multiple configurations:
- **Generic (global)**: d(4) >= 4,501 (30K samples, gap ratio 2,225 in
  full SVD spectrum). Dimension sequence: [3, 6, 17, 116, >=4501].
  New generators per level: [3, 3, 11, 99, >=4385].
- **Lagrange**: d(4) >= 3,112 (20K samples), d(3)=110 (local rank drop).
  Growth ratio d(4)/d(3) = 28.3x.
- **Euler, scalene**: computed at 5K, 10K, 20K sample counts.

**Key finding**: The Level 3 local rank drop at Lagrange (110 vs 116)
persists into Level 4. The gap at d(4) boundary is not as clean as
the d(3)=116 gap (ratio ~10^8), suggesting d(4) may be higher with
more samples — 4,501 is best understood as a lower bound.

**Note on "definitive" claim**: The 30K-sample script declared d(4)=4501
"definitive" based on the max gap ratio (2,225 at SVD index 11,673),
but the gap *at the rank boundary itself* is only 1.2x. Compare to
d(3)=116 which has a gap of 2.17×10^8. The full SVD spectrum is saved
for further analysis.

**Compute cost**: ~$5-15 per configuration on AWS (c5.9xlarge, 128 GB,
~2-4 hours each).

**Priority**: HIGH.  Small cost, potentially transformative result.

---

## 6. Additional Potential Types [MOSTLY DONE]

**Goal**: Extend the classification table beyond {1/r, 1/r^2, 1/r^3, r^2}.

**Completed**:
- **1/r³** [DONE, Mar 2026]: gives [3, 6, 17, 116] — matches 1/r and 1/r².
  Universality across pole orders confirmed.
- **log(r)** [DONE, Mar 2026]: gives **[3, 6, 17, 116]** — logarithmic
  potential (2D vortex dynamics) preserves the universal sequence.
  Transcendental singularity produces the same algebra as algebraic poles.
- **Composite (1/r + 1/r²)** [DONE, Mar 2026]: gives **[3, 6, 17, 116]** —
  two-term composite potential with different pole orders still universal.
- **Yukawa (e^{-μr}/r)** [IN PROGRESS, Mar 2026]: computation requires
  deeply nested symbolic expressions; implemented CSE-based fallback
  compiler for expression trees that exceed Python's recursion limit.

**Remaining**:

| Potential | Type | Status | Notes |
|-----------|------|--------|-------|
| Yukawa: e^(-μr)/r | Singular + exponential | In progress | Recursion limit fix deployed |
| r^4 | Regular | Not started | Prediction: finite algebra |

**Outcome** (from completed tests):

| Result | Interpretation |
|--------|---------------|
| 1/r, 1/r², 1/r³, log(r), composite all → [3,6,17,116] | **Universality across singularity types confirmed** |
| log(r) matches polynomial singularities | Algebra depends on singularity *existence*, not strength |

**Priority**: IN PROGRESS (awaiting Yukawa results from AWS).

---

## 7. Analytical Prediction of SV #116

**Goal**: Explain theoretically why the 116th singular value varies by
4000x across the shape sphere.

**Observation**: SV #116 (normalized) ranges from ~10^-10 at large mu
to ~10^-7 at small mu near phi~0.3.  This smooth, monotonic variation
suggests it should be predictable from the symbolic structure of the
generators.

**Approach**:
- Examine the symbolic expression of the 116th generator (the last
  non-trivially-independent bracket at Level 3)
- Evaluate its gradient norm as a function of (mu, phi)
- Compare to the observed SV #116 landscape

If the landscape can be predicted analytically, it validates the entire
numerical framework and provides a formula for "algebraic constraint
strength" at arbitrary configurations.

**Compute cost**: Moderate.  Requires careful symbolic analysis.

**Priority**: MEDIUM-HIGH.  Would significantly strengthen the theoretical
foundation.

---

## 8. OEIS Submission

**Goal**: Submit the dimension sequence [3, 6, 17, 116] (and growth
sequence [3, 3, 11, 99]) to the On-Line Encyclopedia of Integer Sequences.

**Requirements for submission**:
- At least 4 terms (we have exactly 4, with a lower bound on the 5th)
- A clear mathematical definition
- References or a preprint

**Status**: Sequence not found in OEIS as of initial search.  Need to
finalize the preprint. Level 4 lower bound now at d(4) >= 4,501.

**Next steps**:
1. Finalize preprint.tex with multi-epsilon atlas results
2. Submit preprint to arXiv
3. Submit sequence to OEIS with arXiv reference

**Priority**: LOW (dependent on publication).

---

## 9. Collaborator Outreach

**Identified researchers at the intersection of Morales-Ramis theory and
computational celestial mechanics**:
- Juan Morales-Ruiz (Madrid) -- originator of the differential Galois
  approach to integrability
- Andrzej Maciejewski & Maria Przybylska (Zielona Gora) -- computational
  non-integrability proofs
- Thierry Combot (Dijon) -- algorithmic Poisson algebra methods
- Alexei Tsygvintsev (Lyon) -- exceptional mass ratios and first-order
  obstructions

**Draft outreach emails**: See `outreach_emails.md`.

**Priority**: MEDIUM.  Best done after the multi-epsilon atlas is complete
and results are documented.

---

## Quick-Reference: Priority x Cost Matrix

### Completed
| Analysis | Status |
|----------|--------|
| Multi-epsilon analysis | **DONE** — 12 scans in S3 |
| Spatial dimension independence (N=3) | **DONE** (d=1,2,3 all [3,6,17,116]) |
| N=4 dimension sequence | **DONE** [6, 14, 62] through L2 |
| N=4 mass invariance | **DONE** (3 configs identical) |
| N=4 d-independence | **DONE** (d=1,2,3 identical) |
| 1/r³ potential (N=3) | **DONE** [3,6,17,116] matches 1/r |
| Charge-sign invariance (helium) | **DONE** all-attract/mixed/all-repulsive identical |
| Paper 1 (preprint.tex) | **DONE** |
| Paper 2 (paper2_s3_filtration.tex) | **DONE** |
| Paper 3 (paper3_universality.tex) | **DONE** |

### In progress
| Analysis | Status | Key results |
|----------|--------|-------------|
| Multi-System Universality Survey | **Running on AWS** | 15/21 dimseq complete; 15/21+ atlas complete |
| Composite/PN Pipeline | **Running on AWS** | 2/6 tasks complete (control, two-term) |
| 1/r³ Targeted Scan | **DONE** | Reference + charged scans complete |

### Completed (Multi-System Survey, Mar 2026)
| Milestone | Result |
|-----------|--------|
| Gravitational mass-dependence | All 7 configs → [3, 5, 13, 69] with unequal masses |
| Charge-class mass-invariance | He, H⁻, Ps⁻, muonic He all → [3, 6, 17, 116] (mass range 1 to 7294) |
| Charge magnitude sensitivity | Li⁺ (+3,−1,−1) → [3, 6, 17, 111]; H₂⁺ (+1,+1,−1) → [3, 6, 17, 115] |
| Penning trap ions (+1,+1,+1) | [3, 6, 17, 116] — all-repulsive with external harmonic trap |
| Logarithmic potential (2D vortices) | [3, 6, 17, 116] — transcendental singularity, universality holds |
| Composite 1/r + 1/r² | [3, 6, 17, 116] — multi-pole composite, universality holds |

### Next priorities (post-trilogy + survey)
| Analysis | Priority | Compute | Notes |
|----------|----------|---------|-------|
| arXiv submission (Papers 1-3) | HIGHEST | ~0h | Establish priority |
| Complete Yukawa dimseq (3 scenarios) | HIGH | ~hours (AWS) | Recursion fix deployed, awaiting results |
| Remaining atlas scans (Yukawa, log, 1/r²) | HIGH | ~hours (AWS) | Running with mu-parameter fix |
| N=5 Level 1-2 | HIGH | ~hours | Extends universality to 3rd N value |
| N=4 Level 3 | HIGH | ~hours (AWS) | Extends [6,14,62,...] sequence |
| N=4 with 1/r² potential | HIGH | ~hours | Paper 3 falsifiable prediction #2 |
| Survey analysis + comparative plots | MED-HIGH | ~0h | Post-processing once compute completes |
| S₄ tier structure (N=4) | MED-HIGH | ~hours | Paper 2 → Paper 3 bridge |
| OEIS submission | MEDIUM | ~0h | [3,6,17,116], [6,14,62], [3,5,13,69] |
| Collaborator outreach | MEDIUM | ~0h | Full trilogy + survey available |
| Spectral depth mining | MEDIUM | ~0h (post-proc) | Existing data |
| Lagrange hires 1000x1000 | MEDIUM | ~$3-12 AWS | Resolve ring/bead structure |
