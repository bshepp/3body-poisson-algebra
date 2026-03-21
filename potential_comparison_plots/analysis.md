# Potential Comparison: 1/r (Newton) vs 1/r² (Calogero-Moser)

## Overview

Gap ratio landscape comparison between the Newtonian gravitational
potential (1/r) and the Calogero-Moser potential (1/r²) across five
epsilon values, using the 100×100 hires atlas data.

Both potentials produce the identical dimension sequence **[3, 6, 17, 116]**
for the Poisson algebra. This comparison examines whether the *gap structure*
— which measures the sharpness of algebraic separation — also agrees.

## Data

- **Source**: `atlas_output_hires/1_r/eps_*/` and `atlas_output_hires/1_r2/eps_*/`
- **Grid**: 100×100 (μ × φ), 10,000 points
- **μ range**: [0.2, 3.0], **φ range**: [3°, 177°]
- **Level**: 3 (156 generators)
- **Plots**: `potential_comparison_plots/potential_comparison_eps_*.png`

## Summary Table

| ε | Pearson r | Rank agreement | Δ mean | Δ std | Δ range |
|---|-----------|----------------|--------|-------|---------|
| 2e-03 | 0.9180 | 9171/10000 (91.7%) | +0.366 | 0.412 | [-1.46, +1.56] |
| 1e-03 | 0.9159 | 8842/10000 (88.4%) | +0.375 | 0.411 | [-1.64, +1.45] |
| 5e-04 | 0.9133 | 6135/10000 (61.4%) | +0.373 | 0.407 | [-1.43, +1.64] |
| 2e-04 | 0.9110 | 4813/10000 (48.1%) | +0.352 | 0.394 | [-0.96, +1.46] |
| 1e-04 | 0.8763 | 6723/10000 (67.2%) | +0.100 | 0.415 | [-1.05, +1.24] |

## Gap Statistics

| ε | 1/r mean ± std | 1/r² mean ± std | 1/r range | 1/r² range |
|---|----------------|-----------------|-----------|------------|
| 2e-03 | 5.64 ± 0.94 | 6.00 ± 1.04 | [0.82, 7.64] | [0.72, 7.83] |
| 1e-03 | 4.74 ± 0.92 | 5.11 ± 1.02 | [0.76, 6.77] | [0.63, 6.99] |
| 5e-04 | 3.85 ± 0.89 | 4.22 ± 1.00 | [0.68, 5.82] | [0.62, 6.06] |
| 2e-04 | 2.71 ± 0.79 | 3.06 ± 0.94 | [0.60, 4.62] | [0.50, 4.87] |
| 1e-04 | 2.19 ± 0.49 | 2.29 ± 0.77 | [0.63, 3.69] | [0.52, 3.94] |

All values are log₁₀(gap ratio).

## Regional Analysis

Mean Δlog₁₀(gap) = 1/r² − 1/r by region:

| ε | Lagrange (μ≈1, φ≈60°) | Collinear (φ<15°) | Large μ (μ>2.5) | Global |
|---|------------------------|-------------------|-----------------|--------|
| 2e-03 | +0.136 | +0.370 | +0.152 | +0.366 |
| 1e-03 | +0.139 | +0.365 | +0.156 | +0.375 |
| 5e-04 | +0.174 | +0.358 | +0.159 | +0.373 |
| 2e-04 | +0.165 | +0.299 | +0.145 | +0.352 |
| 1e-04 | −0.108 | −0.068 | −0.312 | +0.100 |

## Interpretation

### 1. High correlation confirms potential invariance

Pearson r > 0.87 at all epsilon values. The two potentials share the
same broad gap landscape structure: the Lagrange equilateral region,
the isosceles symmetry curves, the near-collinear dip, and the
large-μ asymptotic region are all present in both.

### 2. Systematic positive bias: 1/r² has slightly sharper gaps

The mean delta is consistently positive (+0.35 to +0.37 at ε ≥ 2e-4),
meaning 1/r² produces cleaner algebraic separation than 1/r. This is
consistent with the quadratic potential having simpler near-singularity
structure — the 1/r² potential's derivatives are algebraically simpler
(rational functions of lower degree in the u_ij variables).

### 3. Correlation tightens at larger epsilon

r increases from 0.876 at ε=1e-4 to 0.918 at ε=2e-3. At finer scales,
the sampling ball resolves differences in the singularity structure
that are smoothed over at coarser scales. This is expected: the
potentials differ in their near-collision behavior (1/r vs 1/r²),
and smaller epsilon probes closer to the singularities.

### 4. Rank agreement is non-monotonic

Rank agreement peaks at large ε (91.7% at 2e-3) where both potentials
cleanly resolve rank 116, and dips at intermediate ε (48.1% at 2e-4)
where the gap is weakening but not yet collapsed. At the smallest ε
(1e-4), agreement partially recovers (67.2%) because both potentials
are struggling equally with numerical conditioning.

### 5. The ε=1e-4 regime is qualitatively different

At ε=1e-4, the mean delta drops to +0.100 and reverses sign in all
three subregions (Lagrange, collinear, large μ). This suggests that at
the finest scale, the 1/r potential actually resolves structure slightly
better than 1/r², possibly because the 1/r singularity is "simpler"
(lower-order pole) and the sampling ball hasn't fully collapsed.

### 6. The collinear region is most potential-sensitive

The near-collinear strip (φ < 15°) consistently shows the largest
positive delta at ε ≥ 2e-4, meaning 1/r² has the biggest advantage
over 1/r in precisely the region where the potential singularity is
most directly probed (two bodies approaching collision).

## Connection to Charge-Sign Invariance

For reference, the charged (helium +2,−1,−1) vs uncharged comparison
within the *same* potential type gives:

| Comparison | Pearson r (ε=1e-3) |
|------------|-------------------|
| 1/r vs 1/r² (uncharged) | 0.916 |
| 1/r: charged vs uncharged | 0.911 |
| 1/r²: charged vs uncharged | 0.774 |

The cross-potential uncharged comparison (0.916) is actually *more*
correlated than the within-potential charged comparison for 1/r²
(0.774). This hierarchy suggests:

- **Potential type** (singularity order) matters less than expected
- **Charge signs** matter more for 1/r² than for 1/r
- The gap landscape is primarily determined by the **geometry of shape
  space**, not the specifics of the interaction
