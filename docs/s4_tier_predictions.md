# S₄ tier decomposition — predictions and what's missing

The S₃ tier story for N=3 is settled: the 116-dimensional level-3
algebra decomposes as

```
116 = 52 (Tier 1, ε⁰) + 44 (Tier 2, ε¹) + 16 (Tier 3, ε²) + 4 (Tier 4, ε³)
```

with **integer-quantized ε-scaling exponents (0, 1, 2, 3)** and the
striking exact identity `n_E = 52 = Tier 1 size`. That is, the standard
representation E of S₃ exhausts Tier 1 of the jet filtration.

This document records what S₄ representation theory predicts for the
N=4 algebra and what concrete experiment would test it.

## S₄ Clebsch-Gordan content of the N=4 candidates

The S₄ permutation action on the 6 edges of K₄ (the pair Hamiltonians
H_ij) and the iterated Lambda² / tensor products generate the
candidate generators at each bracket level. Multiplicities computed by
[`s4_tier_analysis.py`](../s4_tier_analysis.py); written to
[`results/tier_decomposition/s3_s4_decomposition.json`](../results/tier_decomposition/s3_s4_decomposition.json):

| Level | dim | triv | sign | std | sign·std | hook |
|-------|----:|----:|----:|----:|--------:|----:|
| L0    |   6 |   1 |   0 |   1 |   0 |   0 |
| L1    |  15 |   1 |   2 |   2 |   1 |   1 |
| L2    | 285 |  12 |  13 |  35 |  37 |  23 |
| L3    | ?   |  927|  961|2855|2894|1893 |
| **Total** | **23,226** | **941** | **976** | **2,893** | **2,932** | **1,917** |

(L3 row computed from `total` minus L0+L1+L2 — confirms the script.)

Independent count after Lie-bracket relations: **1,260** at L3, so the
**syzygy fraction is 94.6 %** — twenty-three out of every twenty-four
candidates collapse. (Compare S₃: 156 candidates, 116 independent,
**syzygy fraction 25.6 %**.)

## Tier-1 prediction

For S₃ the standard rep E exhausts Tier 1: `n_E = 52` = Tier 1 size. If
the same mechanism holds for S₄ — the standard rep `std` (dim 3) being
the geometry-sensing irrep — the naive analogy gives

```
Tier 1 (N=4) ≟ n_std generators? = 2,893
```

**But this is too large**: the entire N=4 algebra at L3 has only 1,260
independent generators, fewer than the 2,893 candidate `std` copies. So
the analogy must break in one of three structural ways:

1. **Tier 1 ⊂ L0–L2.** The lower bracket levels (cumulative 62
   generators) plus the upper part of L3 might constitute Tier 1, and
   higher tiers are pure-L3 quotients. This is consistent with the
   step-by-step S₃ story where Tiers correspond to ε-orders and L3
   generators distribute across all four tiers.

2. **`std` does not dominate a single tier under S₄.** The
   ε-quantization story for S₃ used the 2-dim rep E because of its
   special role as the *only* faithful, non-trivial rep of S₃. S₄ has
   five irreps and at least two faithful ones (`std` and `sign·std`,
   both dim 3, plus the dim-2 hook). The "single irrep = single
   tier" map of S₃ may fragment.

3. **The syzygy structure is much tighter for S₄.** With 94.6 %
   syzygy rate vs. 25.6 % for S₃, most of the candidate `std` copies
   become linearly dependent through Jacobi-type relations. The
   *surviving* `std` copies — call it `n_std^surv` — may match Tier 1
   directly. This number is currently unknown.

The current best statement is that `n_std = 2,893` is an upper bound
on the std content of Tier 1, and the actual Tier 1 size is bounded
above by 1,260 (the full L3 dimension).

## What concrete experiment would test the prediction

The S₃ tier identification was made by sweeping the SVD threshold ε
across a 100×100 atlas at d=2, watching tier boundaries form (separate
ε⁰, ε¹, ε², ε³ plateaus) at *individual configurations* and counting
generators in each plateau. The same approach for N=4 needs SVD spectra
at d≥2 for the L3 1260×N_monomials matrix. **This data does not yet
exist.** The 1D-slice atlas in `n4_atlas_1d.py` is too low-dimensional;
collinear N=4 shows no rank drops on any of its three slices.

The minimum next step: a **single-configuration probe** at the N=4
equilateral (tetrahedron at d=3, or 4-fold symmetric 4-body
configuration at d=2 — the "square" or "trapezoid" configurations).
Sweep ε at one (μ, φ, χ) point and count tier sizes. This is
local-only work, no atlas required, and would give the first hard
data on whether Tier 1 has 2,893 or something smaller.

If the single-point probe is informative, the natural followup is a
focused atlas covering the S₄ fixed-point set on the 6D shape space —
the analog of the (μ, φ) shape sphere for N=3.

## Where this lives

- Script: [`s4_tier_analysis.py`](../s4_tier_analysis.py) — pure
  representation theory, no compute. Status: **complete** in the
  registry.
- JSON: [`results/tier_decomposition/s3_s4_decomposition.json`](../results/tier_decomposition/s3_s4_decomposition.json) — combined S₃ + S₄ record consumed by `dataset/build_dataset.py`.
- Hugging Face dataset: `tier_decomposition` split (40 rows).
- N=3 jet filtration narrative:
  [`potential_comparison_plots/quantization_analysis.md`](../potential_comparison_plots/quantization_analysis.md).
- Original prediction in [`docs/conjectures.md`](conjectures.md)
  Section 4: "by analogy with S₃ (where n_E = 52 = Tier 1 size), the
  N=4 Tier 1 prediction is n_std, but this awaits N=4 SVD tier data."
  The analysis above sharpens that one-liner.
