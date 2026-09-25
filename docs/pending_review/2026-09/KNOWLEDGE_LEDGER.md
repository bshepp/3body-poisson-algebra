# What this project knows: one-page ledger (draft, UNREVIEWED)

> **Status: DRAFT for Brian's review. Not accepted into the project.**
> The aim is one place to answer "do we actually know X, and where is it?"
> after time away. Each line is a claim, its status, and where the
> evidence lives.
>
> **Scope of the papers (per Brian, 2026-09-25).** Only paper 1
> (`papers/preprint.tex`, the Zenodo deposit) has been through the
> author's own review. Papers 2–4 are **exploration notes in paper form**:
> a record of the walk, not claims the author stands behind. Items citing
> them are informational, not "needs fixing".
>
> **How it was built (2026-09-25).** Four read-only passes over the whole
> repo (papers, status/conjecture docs plus the full session log, every
> results file, OEIS/public/side projects). They were then assembled and
> cross-checked. Items marked ✔ were re-verified by hand while drafting.
> Items marked ⚠ are *new findings from the Sept review*: they are
> unreviewed and should be checked before anyone acts on them. Where a
> reader and the files disagreed, the files won.

## Status labels

| label | means |
|---|---|
| **EXACT-Q** | Exact rank over ℚ (or ℚ(m), ℚ[ħ]) of the monomial-coefficient matrix, with u_ij as free symbols. This is an upper bound on the true function-space dimension, and exact for the formal algebra. |
| **EXACT-p** | Exact rank mod a prime (sampling or elimination). This is a certified *lower* bound, and equal to the true value with high probability. |
| **CERTIFIED** | Lower bound = upper bound, so settled. |
| **NUMERICAL** | float64 SVD rank. Not a bound of any kind. |
| **CONJ** | Conjectured. |
| **REFUTED / RETRACTED / SUPERSEDED** | Tested and moved on from (see §9). |
| **STALLED / PLANNED** | Not done. |

---

## 1. The headline: N = 3, V = 1/r (A395423: 3, 6, 17, 116)

| claim | status | evidence |
|---|---|---|
| d(0..3) = 3, 6, 17, 116 in 1D | EXACT-Q | `results/symbolic_rank/rank_N3_d1_1r.json` |
| same, planar | EXACT-Q (SymPy, Mathematica); Sage leg is mod-p | `results/symbolic_rank/rank_1_1_1.json`, `mathematica/results/n3_d2_dimseq.json`, `sage/results/n3_d2_dimseq.json` |
| same, true functions (u = 1/r enforced), 1D and planar | ⚠ CERTIFIED (with the EXACT-Q upper bound) | `pending_review/2026-09/scripts/formal_vs_physical.py` |
| same, 3D | NUMERICAL only; no results file | `bench_flint/stage7_crosscheck.json` (bench artifact) |
| new generators per level 3, 3, 11, 99; candidates 3, 3, 12, 138 | EXACT-Q | same files |
| levels 0–2 equal the free-Lie-algebra count; one non-free relation at bracket length 4 explains 119 → 116 | ⚠ proven (unreviewed) | `REVIEW_PENDING.md` B2, `scripts/relation4.py` |
| A395423 approved and published 2026-07-29 | fact | `docs/session_log.md` (2026-07-29 entry); entry source `OEIS/candidates/A395423.md` is gitignored and not in the repo |
| early match to A114491 (…, 69) | REFUTED (finite-difference artifact) | session log, early March |

## 2. Level 4 (N = 3, V = 1/r)

| claim | status | evidence |
|---|---|---|
| d(4) ≥ 4,501 (30k samples, "definitive") | SUPERSEDED; "definitive" retracted (the gap sat in the zero tail) | `results/level4_global_30000/results.json` |
| d(4) ≥ 5,604 (200k samples, 11,523 of 11,937 brackets) | NUMERICAL; boundary gap 1.01 | `results/level4_global_200000/`, `results/highsample_status.json` |
| d(4) ≥ 5,625 (full census, jaga) | NUMERICAL; boundary gap 1.01 | `results/level4_full_census/` |
| **d(4) ≥ 5,914 planar** | ⚠ EXACT-p lower bound, 1 run | `REVIEW_PENDING.md` B4, `scripts/d4.py` |
| **d(4) ≥ 5,042 in 1D** | ⚠ EXACT-p lower bound, 2 runs agree | same |
| d(4) exactly 5,914; 1D ≠ planar | **not proven**; needs an exact upper bound | B4 |
| mpmath high-precision rank | STALLED at 665 of 15,000 rows; eval matrix lacks the 414 skipped brackets; treat as void | `results/level4_mpmath/` |
| Lane C (mod-p streaming, April) | ABANDONED; 26 of 11,937 brackets; the sample count capped the rank at 120 | session log; "never resurrect as configured" |

## 3. N-body (V = 1/r, 1D unless noted)

| claim | status | evidence |
|---|---|---|
| L0 = C(N,2), N = 3–50 | EXACT-Q | `rank_N{11..50}_d1_1r_L0.json` |
| L1 = N(3N−5)/2 (new = N(N−2)), N = 3–26 | EXACT-Q | `rank_N*_d1_1r_L1.json` |
| L2 = N(4N²−9N+3)/2 for N ≥ 4 (new = 12·C(N,3)) | EXACT-Q for N = 4–10; N = 10 was a registered prediction, confirmed 2026-07-19 | `rank_N{4..10}_d1_1r.json` ✔ (N = 9 file exists) |
| N = 4: 6, 14, 62, 1260 | EXACT-Q (L3 for 1/r only) | `rank_N4_d1_1r.json` |
| N = 5: 10, 25, 145, **5,965** | L≤2 EXACT-Q; L3 EXACT-p (2 primes, 2 seeds, plus mod-p elimination) | `rank_N5_d1_1r_L3_modp.json` and raw files |
| new_L3 = 1198·C(N,4) (predicted 6,135) | REFUTED 2026-07-29 | same |
| new_L3 = 1294·C(N,4) − 147·C(N,3) + 82·C(N,2), so L3(6) = 17,979 | CONJ (zero degrees of freedom) | `results/analysis/nbody_scaling_formulas.json` |
| old cubic L2 = (13N³−42N²+83N−120)/6; quartic fit | REFUTED (N = 7, 8) | session log 2026-04-11 |
| N = 4 mass invariance (3 configs); N = 4 d-independence | NUMERICAL | paper 3; `results/n4_atlas_1d.json` |
| N = 6 L3 | PLANNED (next test of the fit) | |

## 4. Potentials (N = 3)

| claim | status | evidence |
|---|---|---|
| 1/r², 1/r³, 1/r⁴ give 116 (1D) | EXACT-Q | `rank_N3_d1_1r{2,3,4}.json` |
| same, planar, level 3 | EXACT-Q only for 1/r² (Mathematica, Sage); the planar SymPy files hold level 2 only | `rank_N3_d2_1r{2,3,4}.json` |
| 1D Calogero–Moser and Galperin mass ratios give 116 | exact files exist; the paper describes the method as float SVD | paper 4; registry |
| log r gives 116 | exact only through L2; L3 NUMERICAL | `rank_N3_d2_log.json` |
| Yukawa e^{−μr}/r gives 116 | NUMERICAL, **4-term Taylor truncation**, so not actually Yukawa | `results/yukawa_dimseq.json` |
| e^{−r} gives 116 (1D, one ordering, 3 mass triples) | ⚠ CERTIFIED | `REVIEW_PENDING.md` B3, `scripts/exp_exact.py` |
| 1PN composite 1/r + 1/r² gives 116 | EXACT-Q | `rank_N3_d2_composite_u1_2.json` |
| 2PN three-term composite | records disagree: registry says 116; file holds L2 only | `rank_N3_d2_composite_u1_2_3.json` |
| r¹: 3, 4, 5, 5 (closes at 5) | EXACT-Q through L2; L3 = 5 appears only in a quantum file | `rank_N3_d1_r1.json` |
| r²: 3, 6, 13, 15, 15 (closes at 15) | EXACT-Q (three CAS) | `rank_N3_d{1,2}_r2.json` |
| r³: 3, 6, 17, 109 | EXACT-Q | `rank_N3_d1_r3.json` |
| r⁴ … r¹⁰ give 116 | EXACT-Q for r⁴, r⁵, r⁷, r⁹. **r⁶, r⁸, r¹⁰ files hold L2 only** ✔, so the L3 claim for those three is unbacked in the current tree | `rank_N3_d1_r*.json` |
| exceptional set {r¹, r², r³} replaces the singular/regular dichotomy | the dichotomy is REFUTED (by r⁴); the set is observed through L3 only | session log 2026-04-09/11 |
| r² closure = Jacobi algebra sp(4,ℝ) ⋉ h₂ | identified by matching invariants only; the stored tensor fails the Jacobi check | `results/algebra_structure/harmonic_n3_d2_identification.json` |
| fractional / irrational exponents | NUMERICAL | `fractional_exponent_sweep*.json`, `rn_exponent_sweep.json` |
| L2 sweep says r¹ gives L2 = 15 | NUMERICAL, contradicts exact L2 = 5, **unflagged** | `level2_exponent_sweep.json` |
| Schwarzschild V_eff (M=1, L=4) gives 115 | NUMERICAL, not followed up | `results/schwarzschild/dimseq_l3_key.json` |
| neural gradient-product coupling gives 3, 6, 17, 119 | EXACT-Q; ⚠ this is exactly the free-Lie bound | `rank_N3_d1_neural*.json` |

## 5. Masses, charges, spatial dimension

| claim | status | evidence |
|---|---|---|
| rank over ℚ(m1,m2,m3) = 116, planar, generic masses | EXACT-Q (a proper exceptional subvariety is not excluded) | `results/symbolic_rank/rank_symbolic.json` |
| 5 specific rational mass triples give 116 | EXACT-Q | `rank_1_*.json` |
| "all positive masses" | RETRACTED to "generic" (July audit C1) | |
| unequal-mass survey gave 3, 5, 13, 69 | RETRACTED (SymPy 1.10 lambdify artifact); 5 of 7 configs inferred, not rerun | flagged rows in the dataset |
| charges (+q,−1,−1), (+1,+q,−1), q = 1..20 | EXACT-Q | `results/charge_sensitivity/charge_sweep_qqn_d1.json` |
| Li⁺ gives 111, H₂⁺ gives 115 | RETRACTED (undersampling) | `charge_sensitivity_completion.json` |
| d-independence | EXACT-Q d = 1, 2 through L3; d = 3 NUMERICAL; ⚠ level 4 unverified (lower bounds 5,042 in 1D vs 5,914 planar) | B4 |

## 6. Quantum, commutant, 117th generator

| claim | status | evidence |
|---|---|---|
| Moyal bracket gives 3, 6, 17, 117 (1D; 1/r, 1/r²–1/r⁴, log, composite) | EXACT over ℚ[ħ]; planar holds L2 only | `rank_N3_d1_quantum_*.json` |
| Moyal for r⁴, r⁶ gives 116; r³ gives 109 | EXACT; r³ is omitted from the summary file | `quantum_r*.json` |
| "quantum commutant 40 < classical 41, so quantization removes a conservation law" | ⚠✔ **counting artifact.** Computed as 156 − rank over the raw 156 generators, so it includes the 40 (classical) / 39 (quantum) linear relations among them. Intrinsic commutant is 1 in both (just H) | `nbody/energy_bound_search.py`, `results/energy_bound/energy_bound_results.json` |
| "center 17/17" (dim − Killing-rank shortcut) | CORRECTED to 11/17 | |
| 117th generator g = −(9/4)[(A−B)² + A²] | recorded as algebraic | `nbody/analyze_117th.py` |

## 7. Structure constants and "isomorphism"

| claim | status | evidence |
|---|---|---|
| 12–13 non-harmonic potentials share identical L2 invariants (nilpotent class 3, center 11, …) | CONJ (isomorphism) | `results/isomorphism_test.json` |
| ⚠✔ why they match | A₂ (dim 17) is **not closed** under the bracket. The "structure constants" come from a truncated algebra (out-of-span parts dropped). The free Lie algebra truncated there has center 8 + 3 = 11 and class 3, so identical invariants are expected for any generic potential | `nbody/symbolic_rank_nbody.py` (out-of-span warning) |
| r³ L3 "lower central series oscillates" | ⚠ impossible for a genuine Lie algebra; same truncation artifact | README, `hf_article_physics.md` |
| "Killing signature (6+,0−,11z)" | CORRECTED to ad-Gram/Frobenius (true Killing form is 0 for a nilpotent algebra); the `killing_form.npy` files were not regenerated; `rank_N3_d2_1r.json` still pairs "nilpotent" with a nonzero Killing form | |
| r¹ = filiform L_{5,2} | recorded | |
| 116-dim L3 structure constants | STALLED | |

## 8. Numerics-based programs

| claim | status | evidence |
|---|---|---|
| 21 completed 100×100 atlases; "critical locus" at S₃ fixed points | NUMERICAL, CONJ. ⚠ Local rank below the global rank is conditioning, not algebra (analytic functions stay independent on every open ball) | `docs/project_status.md` §1; S3 data not in repo |
| ranks > 116 near collinear; "soft syzygies wake up" | RETRACTED in `collision_syzygy/COLLISION_SYZYGY_REPORT.md` (paper 2, an exploration note, still has the old text) | |
| collision stratum: exact ℚ rank 80 (76 syzygies) on the (4,3) ε-family | EXACT-Q | `collision_syzygy/` |
| four-tier SV structure 52+44+16+4; ε^α exponents 0, 1, 2, 3 | NUMERICAL (paper 2); tier 4 fitted as 2.82 ± 0.21 | paper 2 |
| S₃ isotypic 24A + 28A′ + 52E of the 156 candidates | exact counting (on candidates, not on the 116-dim span) | `results/tier_decomposition/` |
| S₄ "Tier 1 = 2,893" | REFUTED in its own doc (exceeds 1,260) | `docs/s4_tier_predictions.md` |
| 1000×1000 atlas | STALLED at 880 of 1,000 rows | |
| parametric 1,015-exponent sweep | ABORTED (cost) | |
| extreme-mass atlases (ranks 91–108) | NUMERICAL, explained as conditioning | |

## 9. Tested and moved on from (the "we already did that" list)

- Non-integrability certificate. Dead since Feb 2026: Calogero–Moser gives the same sequence (`docs/peer_review_analysis.md`).
- Singular/regular dichotomy. Refuted by r⁴ (Apr 2026). "Harmonic is the unique exception" was superseded by r¹ and r³.
- A114491 match. Refuted.
- Unequal-mass 3, 5, 13, 69. SymPy artifact.
- Li⁺ 111 and H₂⁺ 115. Undersampling.
- Old cubic and quartic L2 fits. Refuted at N = 7, 8.
- new_L3 = 1198·C(N,4). Refuted at N = 5.
- "Needs 512 GB" for N = 5. Wrong framing (mod p runs in 0.5 GB).
- Numerical remedies for extreme masses (term-group factoring, post-scaling, mpmath as ground truth, bisection). Abandoned Apr 6–8.
- Lane C configuration. Do not resurrect as-is.
- Hilbert–Pólya "[116+,0,0] semisimple". Superseded and corrected as impossible.
- "Quantum enlarges the algebra with new conserved quantities". Superseded. ⚠ And the 40 < 41 commutant replacement is itself an artifact (§6).
- A027376 shortfall pattern (3^(L−3) − 1)/2. ⚠ Refuted at L = 8 (572, not 689).
- d(4) ≥ 4,501 "definitive". Retracted.

## 10. Known bugs (fixed) and what they touched

| bug | affected | status |
|---|---|---|
| resume sum-vs-max | 3D L3 = 102 (Mar); latent in `exact_growth.py` until 2026-07-19 | fixed + regression test |
| L4 enumeration (sum ≤ 3 instead of max ≤ 2) | skipped 414 brackets in all float L4 runs and the mpmath matrix | fixed; 5,604 stays a (numerical) lower bound |
| stale harmonic pickle | early soft-syzygy claims | fixed; checkpoint rebuilt |
| Killing `.T` bug | "Killing" signatures | relabelled ad-Gram; npy not regenerated |
| SymPy 1.10.1 lambdify drops | unequal-mass survey | retracted |
| `--save-svd` clobbering | Yukawa dataset overwritten | restored |
| vacuous Padé test; s3 stale .npy; column_stack; argparse 1/r³ | various | fixed |
| **evidence overwritten** (docs cite L3 = 116, files hold L2 only) | `rank_N3_d1_r{1,6,8,10}.json`, `rank_N3_d2_{1r,1r2,1r3,1r4,r4}.json` | ⚠ **open**: regenerate or recover from S3 |

## 11. Housekeeping (stale, inconsistent, or wrong text)

**Paper 1 (Zenodo; author-reviewed; these are the ones that matter)**
- Calls Thm 2 both a "theorem" and "a proof sketch".
- Names μ = 2/9 but never tests it.
- Says the sequence is "not in OEIS".
- Methods section says ranks come from a float SVD gap test.
- Title and abstract say "super-exponential"; the discussion says "unchanged by spatial dimension"; Yukawa is cited from truncated runs; the atlas paragraph describes "rank drops" (see REVIEW_PENDING A1–A5).
- Cites no script or results file.

**Papers 2–4 (exploration notes, not author-reviewed; informational only)**
- Paper 4 defines the filtration as ℓ({f,g}) = ℓ(f)+ℓ(g)+1 ✔. That is a different filtration from the one used everywhere else, yet it reports 116.
- Paper 3's N = 4 level-1 arithmetic ✔ says "one additional dependence" where four are needed.
- Paper 3's proof sketch for charge invariance assumes {H_ij,H_kl} ∝ c_ij c_kl, which the kinetic terms break.
- Paper 2's abstract says "156 at level 3"; the level-3 count is 138.
- Paper 2's soft syzygies conflict with its own "exactly zero" statement and with the collision report.
- Paper 4 says "functionally independent at each level".
- The three papers cite each other inconsistently.

**OEIS**
- **L2 closed-form draft:** the generating function is **wrong** ✔ (it gives 62, 145, 257…; should be 62, 145, 279…). The claim of invariance for N ≥ 5 masses and potentials is unbacked.
- **Harmonic draft:** H_ij has no kinetic terms (the Zabolotskii error again); says "sp(6,R)" where the repo says sp(4,ℝ) ⋉ h₂; checklist unchecked.
- **Stale text:** `OEIS/README.md` says "Submitted"; `checklist.md:234` says a(4) = 116; the readable says "17 is genuinely irregular".

**Public text** (README, website, articles, explainer)
- "Super-exponential" and "infinite" are stated flatly.
- 116 → 306 is offered as proof of infinite dimension (a partial sample).
- "Jacobi identity FAILS" is offered as evidence (it only shows the truncation is not closed).
- The explainer states the filtration as L_n + {L_0, L_n} and gives H_ij without kinetic terms.
- `hf_article_physics.md` says r¹ gives 15 and "N = 3 through 50".
- The GUE "CONFIRMED" wording.
- Contextuality "zero commuting pairs" despite nontrivial centers.

**Synthesis docs**
- `project_status.md`, `conjectures.md`, `research_roadmap.md` and `gap_workplan.md` predate July. Only the session log records N = 5 L3 = 5,965, N = 10 L2 = 1,565, d(4) ≥ 5,625 and the OEIS approval. `conjectures.md:397` contradicts the paragraph below it.
- `nbody_scaling_formulas.json` still says N = 5 is "PROVISIONAL".

**Missing data**
- Gitignored or S3-only: `checkpoints/`, `aws_results/`, `*.npy`, the dataset parquet, `3d/checkpoints_*`, `atlas_output_hires/`.
- `results/n_universality_survey/boundary_results.json` is empty.

## 12. Open questions, ranked by value per effort

1. **Exact upper bound for d(4)**, 1D then planar (per bracket-length block, over ℚ, on jaga). This settles a(4) and d-independence at level 4.
2. **Regenerate the overwritten L3 files** (§10) so every "116" has a backing file.
3. **N = 6 L3 mod p**, to test the 17,979 prediction.
4. **Prove L1 and L2 for all N by hand.** This is the most likely real theorem.
5. **Understand the single length-4 relation** (generic masses, planar).
6. **Untruncated Yukawa and log at L3**, exact.
7. **Editorial pass** over the paper 1 and public-text items in §11 before any new submission. Papers 2–4 only if they are ever promoted.
