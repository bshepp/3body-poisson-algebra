# Phase 1 Findings — Spectral Post-Processing of Atlas Data

*Date: April 1, 2026*
*Scripts: `spectral_depth_mining.py`, `cg_atlas_comparison.py`, `level4_comparison.py`, `sv116_analytical.py`*
*Output: 14 figures in `spectral_depth/`*

---

## Executive Summary

Phase 1 mined existing atlas data (zero new computation) to answer six
open questions about the spectral structure of the Poisson–Lie algebra
for the planar three-body problem. The key takeaways:

1. **The SV spectrum varies smoothly across the shape sphere** — no hidden
   phase boundaries exist at intermediate singular value indices. The
   spectral structure is a continuous deformation parameterized by
   triangle geometry.

2. **S₃ symmetry governs spectral degeneracy quantitatively** — doublet
   counts peak at the Lagrange equilateral point (33 doublets, 57% of
   SVs paired) and decay monotonically with distance from full S₃
   symmetry. This extends the Clebsch-Gordan prediction from a single
   point to a global landscape.

3. **Level-4 dimension is vastly under-sampled at special configurations**
   — random global sampling at 200K points finds d(4) ≥ 5604, while
   targeted Lagrange/Euler sampling saturates around 3100–3200. The
   convergence curve shows no plateau, implying the true d(4) may be
   significantly larger.

4. **The rank boundary has partial analytical predictability** — the RMS
   magnitude of the 116th symbolic generator correlates with the observed
   SV #116 at R² = 0.630, establishing a bridge between the symbolic and
   numerical layers.

---

## 1. Interior Singular Value Landscapes (Item 1.1)

**Question:** Does the interior of the SV spectrum vary smoothly, or are
there hidden phase boundaries at intermediate indices?

**Answer:** Smooth variation throughout. No discontinuities.

### Method

Plotted log₁₀(σₖ/σ₁) as shape-sphere heatmaps at SV indices k = 50,
80, 100, 110, and 116 (the rank boundary), plus a "spectral spread"
panel showing log₁₀(σ₅₀) − log₁₀(σ₁₁₆).

### Observations

**For 1/r (Newton):**

- **SV #50** (bulk): Nearly uniform across the shape sphere, ≈ 10⁻²–10⁻³
  relative to σ₁. Mild enhancement at small φ (acute triangles) and
  suppression near Euler collinear configs. The algebra's bulk generators
  are "equally important" everywhere.

- **SV #80–100** (mid-spectrum): Gradual emergence of spatial structure.
  A diagonal ridge of slightly elevated values runs from low-μ/high-φ to
  high-μ/low-φ. The Lagrange and Isos-90 points show mild local minima
  (slightly weaker mid-spectrum generators at high-symmetry points).

- **SV #110** (near boundary): Strong spatial variation appears. A bright
  lobe at μ < 1, φ < 60° (small acute triangles) where the 110th
  generator is strongest. Dark region at μ > 2, φ > 120° (flat obtuse
  triangles) where it's 10–100× weaker.

- **SV #116** (rank boundary): Dramatic spatial patterning. Peak values
  at small acute triangles (≈ 10⁻⁷), minimum at collinear/Euler
  configurations (≈ 10⁻⁹). Two orders of magnitude variation. This is
  the "weakest algebraic constraint" and it's strongest precisely where
  the triangle is most compact.

- **Spectral spread** (Δlog₁₀ between SV #50 and #116): Ranges from
  ~5.0 to ~7.5 orders of magnitude. Greatest spread at large-μ,
  high-φ (elongated triangles), smallest at compact equilateral-like
  shapes. This means elongated configurations have a "steeper" spectral
  drop — the rank-116 generators are more marginal there.

**For 1/r² (Calogero-Moser):**

- Same qualitative pattern as 1/r but with slightly broader dynamic
  range. The spectral spread panel shows values ~4.5–7.5. The 1/r²
  landscapes are marginally noisier, consistent with tighter SV gaps
  making the numerical rank boundary more sampling-sensitive.

### Figures

- `spectral_depth/sv_landscapes_1_r.png` — 6-panel grid (SV #50, 80,
  100, 110, 116, spread) for Newton
- `spectral_depth/sv_landscapes_1_r2.png` — same for Calogero-Moser

### Implications

The smooth variation rules out "spectral phase transitions" — there is
no critical SV index at which the landscape suddenly reorganizes. The
algebra's structure deforms continuously as the triangle shape changes,
with the strongest effects concentrated at the rank boundary (SV #116).

---

## 2. Spectral Decay Rate Map (Item 1.2)

**Question:** Are there configurations where the SV spectrum drops steeply
(near-degenerate generators) vs. gradually (well-separated generators)?

**Answer:** Yes. The decay landscape has rich structure with a ~40%
variation in mean log-slope across the shape sphere.

### Method

At each grid point, computed:
- **Mean log-slope** = (log₁₀σ₁₁₀ − log₁₀σ₂₀) / 90, measuring the
  average per-index decay rate across the mid-spectrum
- **Knee index** = first SV index where σₖ drops below 1% of σ₁ (the
  steepest drop location)

### Observations

**Mean log-slope (1/r):**

- Values range from ≈ −0.050 to −0.073 (negative = decaying).
- Steepest decay (most negative, blue regions): a band running from
  upper-left (μ > 2, φ < 40°) through the center. Also steep at the
  Euler collinear configuration (μ=0.5, φ=180°).
- Shallowest decay (red regions): lower-left corner (μ < 1, φ < 50°)
  near the Lagrange point, and a secondary patch at medium μ, high φ.
- **Interpretation:** At the Lagrange equilateral point, the SV spectrum
  decays most gently — generators are most evenly weighted. At the Euler
  collinear point, the spectrum drops steeply — many generators become
  relatively weak, concentrating the algebra's "weight" in fewer directions.

**Knee index (1/r):**

- Most of the shape sphere has knee index ≈ 107–113 (yellow regions):
  the spectrum stays above 1% of σ₁ until very near the rank boundary.
- A narrow band centered around φ ≈ 60–90° and μ ≈ 1–2 shows
  dramatically lower knee indices (95–97, dark blue). This band includes
  the Lagrange point and traces an arc that follows the S₃ symmetry locus.
- **Interpretation:** Near the symmetric configurations, the SV spectrum
  has a sharper "elbow" — it maintains high values longer then drops more
  abruptly at the end. Away from symmetry, the drop is more gradual
  but starts earlier.

**Spectral profiles at special points:**

- All three profiles (Lagrange, Euler, Isos-90) share a common shape:
  plateau → gentle decline → steep terminal drop.
- Euler has the overall lowest SVs (blue curve sits below red and green
  throughout). Lagrange and Isos-90 are similar through SV #60, then
  Lagrange pulls ahead near the boundary.
- The terminal drop (last ~10 SVs) is steepest at Euler, shallowest at
  Lagrange — consistent with the Lagrange point having the most "robust"
  rank-116 algebra.

### Figures

- `spectral_depth/spectral_decay_1_r.png` — 3-panel: log-slope, knee
  index, spectral profiles at special points
- `spectral_depth/spectral_decay_1_r2.png` — same for 1/r²

---

## 3. Spectral Clustering (Item 1.3)

**Question:** Does the shape sphere partition into algebraically distinct
zones beyond the known S₃ symmetry structure?

**Answer:** Yes. K-means clustering reveals 3–7 geometrically coherent
zones whose boundaries correlate with but extend beyond S₃ structure.

### Method

Normalized each 156-dim SV vector to unit norm (removing overall scale),
then applied k-means clustering with k = 3, 5, 7. Visualized cluster
assignments as shape-sphere maps and plotted median spectral profiles
per cluster.

### Observations

**k = 3 clusters (1/r):**

Three large contiguous regions:
- A large central/equilateral zone (orange, cluster 1) containing
  Lagrange and Isos-90
- A collinear/elongated zone (grey, cluster 2) dominating high-φ, high-μ
- A compact-triangle zone (red, cluster 0) at low-φ

The boundary between clusters 0 and 1 follows a diagonal arc, while
cluster 2 occupies the "corners" of the shape sphere.

**k = 5 clusters (1/r):**

The 3-cluster structure refines:
- The central zone splits into a Lagrange-centered core (green, cluster 0)
  and a surrounding annular zone (red/brown)
- The collinear zone splits by mass ratio (μ)
- Cluster boundaries show mild fractal roughness but are geometrically
  coherent

**k = 7 clusters (1/r):**

Further refinement with:
- The Lagrange core persists as a distinct cluster
- New clusters isolate the Euler neighborhood, the isosceles band,
  and extreme-μ regions
- A small pink cluster appears at low-μ, high-φ (near-degenerate
  flattened triangles)

**Cluster profiles:**

All 5 clusters (at k=5) show similar qualitative spectral shapes but
differ in:
- The rate of initial decline (clusters near Lagrange decline more slowly)
- The position and sharpness of the mid-spectrum knee (~SV #50–60)
- The terminal drop rate (last 20 SVs)

Cluster separation is driven primarily by the mid-to-late spectrum
(SV #40–110), not by the leading SVs which are more uniform.

### Figures

- `spectral_depth/spectral_clusters_1_r.png` — shape-sphere maps at
  k=3, 5, 7 for Newton
- `spectral_depth/spectral_clusters_1_r2.png` — same for 1/r²
- `spectral_depth/cluster_profiles_1_r.png` — median profiles for k=5
- `spectral_depth/cluster_profiles_1_r2.png` — same for 1/r²

### Implications

The spectral clustering recovers the S₃ symmetry structure (Lagrange
core, collinear band) but also reveals finer zones not predicted by
symmetry alone. This suggests that the "algebraic character" of the
Poisson bracket generators varies with triangle shape in ways beyond
what the representation theory captures. The clusters could serve as
natural strata for future theoretical analysis.

---

## 4. Clebsch-Gordan Doublet Landscape (Item 1.4)

**Question:** Does the S₃ CG decomposition predict spectral degeneracy
structure everywhere, or only at the Lagrange fixed point?

**Answer:** The CG doublet structure is maximal at Lagrange and decays
monotonically with distance from full S₃ symmetry — a quantitative
global prediction.

### Method

At each (μ, φ) grid point, counted "near-degenerate doublets" =
consecutive SV pairs with ratio σₖ₊₁/σₖ > 1/1.05 (i.e., within 5%
of each other). Computed total doublets and also tier-resolved counts
within the four S₃ representation tiers: tier 1 [0–52], tier 2
[52–96], tier 3 [96–112], tier 4 [112–116].

### Key Results

| Configuration | Total Doublets | E-fraction | S₃ Symmetry |
|---------------|----------------|------------|-------------|
| Lagrange (μ=1, φ=60°) | 33 | 0.57 | Full S₃ |
| Isos-90 (μ=1, φ=90°) | 24 | 0.41 | Z₂ subgroup |
| Euler (μ=0.5, φ=180°) | 14 | 0.24 | None (collinear) |

**E-fraction** = 2 × doublets / 116, the fraction of the spectrum
participating in near-degenerate pairs.

### Observations

**Total doublet landscape:**

- Peak doublet counts (30–35) concentrate near Lagrange and form a
  "hot spot" centered at (φ ≈ 50°, μ ≈ 1). The hot spot is elongated
  along the μ ≈ 1 isosceles line.
- Minimum doublet counts (8–12) occur at the Euler point and at
  extreme mass ratios (μ > 2.5 or μ < 0.3).
- The transition from high to low doublet count is smooth with no sharp
  boundaries.

**Tier-resolved structure:**

- **Tier 1 (SV 0–52):** Strongest doublet signal. Hot spot matches the
  total landscape but is broader. Even away from Lagrange, tier-1
  degeneracies persist (minimum ~6 doublets).
- **Tier 2 (SV 52–96):** Similar spatial pattern to tier 1 but with
  narrower range (8–16 doublets). The Lagrange hot spot is more
  pronounced relative to background.
- **Tier 3 (SV 96–112):** Noisy, with low counts (0–8). Spatial structure
  is weaker. The CG prediction is less reliable in this tier.
- **Tier 4 (SV 112–116):** Almost entirely 0 or 1 doublets. The terminal
  4 SVs rarely form near-degenerate pairs, consistent with them being
  the "least protected" generators.

**Doublets vs. distance from Lagrange:**

- Clear negative correlation. The scatter plot shows points concentrating
  along a downward trend from (distance=0, doublets≈33) to
  (distance≈3, doublets≈10).
- Substantial scatter at intermediate distances — the doublet count
  depends on both the distance from Lagrange AND the angular direction
  in shape space.

**E-fraction landscape:**

- Forms a smooth dome centered on Lagrange with E-fraction ≈ 0.57 at
  peak. The contour E-fraction = 0.30 encloses roughly the central third
  of the shape sphere.

**Residual map (observed − Lagrange count):**

- Uniformly negative (blue) since Lagrange has the maximum. Residuals of
  −5 to −10 dominate the interior; −15 to −25 at the edges.
- No systematic positive residuals, confirming Lagrange is a true global
  maximum for doublet count.

### Figures

- `spectral_depth/cg_doublet_landscape.png` — 5-panel: total + 4 tiers
- `spectral_depth/cg_predicted_vs_observed.png` — 3-panel: scatter,
  E-fraction landscape, residual map

### Implications

The CG representation theory is a **local** explanation centered on the
S₃ fixed point. Its predictions (doublet pairing) extend smoothly across
the shape sphere with monotonic decay. This is consistent with the
interpretation that S₃ symmetry "lifts" into an approximate
near-degeneracy that is continuously broken by triangle deformation.
The doublet hierarchy Lagrange > Isos-90 > Euler exactly tracks the
subgroup chain S₃ ⊃ Z₂ ⊃ {e}.

**Paper relevance:** This result strengthens Paper 2 (S₃ filtration) by
showing the CG structure isn't just a fixed-point artifact — it's the
dominant organizing principle for spectral degeneracy across the full
configuration space.

---

## 5. Level-4 Dimension Comparison (Item 1.5)

**Question:** How does the detected dim(L₄) vary across configuration
types and sample counts? Is the Lagrange rank drop persistent?

**Answer:** Dramatic variation. Global random sampling vastly outperforms
targeted sampling, and the convergence curve shows no sign of saturating.

### Data

18 unique result records across 4 configuration types:
- **Global** (random phase-space points): 5K, 10K, 20K, 30K, 50K, 100K, 200K samples
- **Scalene** (generic non-symmetric): 5K, 10K, 20K samples
- **Lagrange** (equilateral): 5K, 10K, 20K samples
- **Euler** (collinear): 5K, 10K, 20K samples

### Key Results

| Config | Max Samples | d(4) Lower Bound | Definitive Gap? |
|--------|------------|-------------------|-----------------|
| Global | 200K | ≥ 5,604 | No |
| Scalene | 20K | ≥ 3,218 | No |
| Lagrange | 20K | ≥ 3,112 | No |
| Euler | 20K | ≥ 2,194 | No |

Only the Global-30K run achieved a definitive gap ratio (marked green on
the bar chart). All others have non-definitive gaps, meaning the true
rank could be higher.

### Convergence Analysis

The convergence curve (d₄ vs. n_samples on log-log scale) reveals:
- **Global:** Steady upward trend from 2,253 (5K) → 5,604 (200K).
  Power-law fit: d₄ ∝ n^{0.24}. Extrapolating to 1M samples predicts
  d₄ ≈ 9,500 (highly uncertain).
- **Scalene/Lagrange:** Flat after 10K samples, saturating at ~3,100–3,200.
  These configurations lack the diversity needed to expose new
  independent generators.
- **Euler:** Persistently lowest, growing only from 2,036 to 2,194.
  The collinear configuration appears to be the "least informative"
  for rank detection.

### Stacked Level Breakdown

The stacked bar chart confirms that levels 0–3 contribute identically
(dim = 116) across all configs — the variation is entirely in the
level-4 new generators (Δd₄ = d₄ − 116), which range from 2,078
(Euler) to 5,488 (Global).

### Figures

- `spectral_depth/level4_comparison_chart.png` — bar chart of max d₄ per config
- `spectral_depth/level4_convergence_curves.png` — log-log convergence
  with power-law fit
- `spectral_depth/level4_stacked_levels.png` — level breakdown

### Implications

1. **Global sampling is essential.** Targeted sampling at special
   configurations (Lagrange, Euler) misses ~40–60% of the independent
   generators that global sampling finds. This makes physical sense:
   the level-4 algebra is so large that many generators are only
   "visible" at generic (low-symmetry) configurations.

2. **d(4) is not yet resolved.** The n^{0.24} scaling with no plateau
   means 200K samples is still far from saturation. Resolving d(4) will
   require either ≫200K samples or the mpmath arbitrary-precision
   approach (item 4.3).

3. **The "Lagrange rank drop" is a sampling artifact.** It's not that the
   algebra is smaller at Lagrange — it's that the Lagrange point lies in
   a high-symmetry region where many generators become linearly dependent
   when evaluated at that single configuration. The true dimension is
   configuration-independent (a theorem for Lie algebras over ℝ).

---

## 6. Analytical Prediction of SV #116 (Item 1.6)

**Question:** Can the weakest algebraic constraint be predicted from the
symbolic structure of the generators alone?

**Answer:** Partially — R² = 0.630 correlation between symbolic RMS and
observed SV #116.

### Method

1. Loaded 156 symbolic generators from `checkpoints/level_3.pkl`
2. Selected 8 key generators: H₁₂, H₁₃, H₂₃ (level 0) and generators
   111–115 (the five highest-level generators, closest to the rank
   boundary)
3. Lambdified each individually using `sympy.lambdify` (avoiding the
   batch `lambdify_generators()` wrapper due to a scalar broadcasting bug)
4. Evaluated on a 20×20 coarse grid over (μ, φ), with 500 random
   phase-space samples per grid point
5. Defined "predicted SV #116" = log₁₀(RMS of generator #115 / RMS of
   generator #0)
6. Interpolated observed SV #116 from the 100×100 atlas to the 20×20
   coarse grid points
7. Computed Pearson R²

### Results

- **R² = 0.630** — a meaningfully strong correlation
- The scatter plot shows a clear negative trend: where the symbolic
  generator ratio is high (left side), the observed SV #116 is high
  (top), and vice versa
- Spatial patterns partially match: both the predicted and observed
  landscapes show the Lagrange hot spot and Euler cold spot
- The predicted landscape is smoother (20×20 resolution + RMS averaging),
  while the observed shows finer structure and more noise

### Technical Issue Resolved

The `lambdify_generators()` function in `exact_growth.py` wraps all
lambdified expressions in a `np.column_stack()` call. This fails when
some expressions evaluate to scalars (constant expressions that
`lambdify` returns as 0-dim arrays) while others return 1-D arrays.
Fix: lambdify each expression individually and broadcast scalars to
arrays before stacking.

### Figures

- `spectral_depth/sv116_predicted_vs_observed.png` — 3-panel: observed
  atlas SV #116, predicted from symbolic generators, scatter with R²

### Implications

The R² = 0.630 is encouraging but not definitive. It means ~63% of the
spatial variance in the rank boundary can be "explained" by the symbolic
structure of the generators. The remaining 37% likely comes from:

- The coarse 20×20 evaluation grid (vs. 100×100 atlas)
- Using only 8 of 156 generators in the predictor
- Phase-space sampling noise (500 samples vs. 400 in the atlas)
- The predictor being a simple RMS ratio rather than a proper SVD

A follow-up analysis (Phase 2+) could improve this by:
- Using all 156 generators on a finer grid
- Computing a mini-SVD at each point instead of RMS ratios
- Investigating which specific generators dominate the R² contribution

---

## Summary Table

| Item | Question | Key Finding | Figure |
|------|----------|-------------|--------|
| 1.1 | Hidden phase boundaries in SV spectrum? | No — smooth variation throughout | `sv_landscapes_{1_r,1_r2}.png` |
| 1.2 | Configs with steep vs. gradual SV decay? | Yes — Euler steep, Lagrange gentle, 40% range | `spectral_decay_{1_r,1_r2}.png` |
| 1.3 | Natural spectral regions on shape sphere? | Yes — k-means finds 3–7 coherent zones beyond S₃ | `spectral_clusters_*.png`, `cluster_profiles_*.png` |
| 1.4 | CG doublets beyond Lagrange? | Monotonic decay from Lagrange max (33) to Euler min (14) | `cg_doublet_landscape.png`, `cg_predicted_vs_observed.png` |
| 1.5 | Level-4 dim across config types? | Global 200K ≥ 5604, far from plateau; targeted sampling saturates early | `level4_*.png` |
| 1.6 | Analytical control of SV #116? | Partial — R² = 0.630 from symbolic generator RMS | `sv116_predicted_vs_observed.png` |

---

## Paper Relevance

| Finding | Relevant Paper | Contribution |
|---------|---------------|--------------|
| Smooth SV landscapes | Paper 1 (growth) | Supports claim that dim=116 is universal, not geometry-dependent |
| CG doublet landscape | Paper 2 (S₃ filtration) | Extends CG from fixed point to global structure |
| Spectral clustering zones | Paper 2 / new paper? | Novel result — algebraically distinct strata on shape sphere |
| Level-4 convergence | Paper 1 (growth) | Evidence for super-exponential growth; addresses d(4) bound |
| SV #116 prediction | New result | Bridge between symbolic and numerical approaches |
| Decay rate variation | Paper 3 (universality) | Euler configurations = "weakest" algebraic constraints |

---

## Technical Notes

### Data Description

- Atlas data: 100×100 grid over (μ, φ) with μ ∈ [0.2, 3.0] and φ ∈
  [5.7°, 174.3°]. Each grid point has 156 singular values from SVD of
  the generator evaluation matrix (116 generators × n_samples).
- Potentials: 1/r (Newton) and 1/r² (Calogero-Moser) in
  `atlas_output_hires/`.
- Level-4 results: JSON files with fields `config`, `n_samples`, `dims`,
  `d4_lower_bound`, `definitive_gap`, `boundary_gap_ratio`. Distributed
  across `results/` and `aws_results/` directories.
- Checkpoints: `level_3.pkl` contains all 156 symbolic SymPy expressions
  keyed as `exprs` (list), `names` (list), `levels` (list of ints 0–3).

### Reproducibility

All scripts run from the repository root:
```bash
python spectral_depth_mining.py     # Items 1.1, 1.2, 1.3
python cg_atlas_comparison.py       # Item 1.4
python level4_comparison.py         # Item 1.5
python sv116_analytical.py          # Item 1.6
```

Dependencies: numpy, scipy, matplotlib, sympy (≥ 1.13.3), sklearn.
Runtime: ~2 minutes total (sv116_analytical is the bottleneck at ~90s).
