# Jet evaluation: bracket-tree ranks without symbolic generation

*2026-07-30 (overnight session). Status: prototype implemented and
validated; see `jet_eval/`.*

## The wall this removes

Every frontier of the level filtration has been blocked by **symbolic
bracket generation**, never by rank:

- Lane C (April): N=3 L4 generation at ~2,872 s/bracket → 396-day ETA.
- N=5 L3 (April): generation succeeded (28 h AWS) but only after
  heroic effort; the QQ rank then OOM'd. The 2026-07-29 campaign
  showed the *rank* is nearly free mod p (475 s / 0.47 GB for the
  full 1.1M-generator census) once the generators can be *evaluated*.
- N=6 L3: generation OOM'd during L2. Never attempted further.

The observation: the rank only needs each generator's **values at
random F_p sample points** — and a generator is a *bracket tree* over
the base pair-Hamiltonians. Values of bracket trees can be computed
numerically, mod p, with no symbolic expression ever built.

## Method

A Poisson bracket consumes one derivative order. So to get the *value*
(order 0) of a level-L tree at a point z, it suffices to carry
**truncated Taylor jets**: each function is represented by its Taylor
coefficients at z up to total order m, and

    order-m jet of {f, g}  needs  order-(m+1) jets of f and g.

Working back from the target: level-L trees need order 0, level-(L−1)
trees order 1, …, the base Hamiltonians order L. In `nv` variables an
order-m jet has C(nv+m, m) coefficients — a few hundred to a few
thousand numbers, independent of how large the symbolic expression
would have been (the N=5 dense-tail generators that blew a worker to
34 GB under `expand()` are, as jets, the same few thousand residues
as everyone else).

Key properties:

- **Memoization.** Lower-level jets are shared by *all* higher trees.
  The per-tree marginal cost at the target level is one bracket at
  order 0: a few dozen scalar mults per phase-space q-variable.
- **Vectorization.** Every jet coefficient is stored as a length-S
  vector (one residue per sample point); all arithmetic is numpy
  uint64 vector ops mod p, batched over points.
- **u-chain rule.** u_ij = 1/r_ij is an independent jet variable and
  ∂/∂q goes through Dq with chain(u_ij, q_ik) = −(q_ik − q_jk)·u_ij³,
  exactly the symbolic engine's convention (independent-u
  Schwartz-Zippel, same as every prior mod-p leg).
- **Census fidelity.** Tree enumeration mirrors
  `symbolic_rank_nbody.build_generators` (level 1 = all H-pairs;
  level ≥ 2 = frontier × all existing, frozenset-deduplicated), so
  the evaluated census is the same family of candidates the symbolic
  path spans. Trees that simplify to zero contribute zero rows and
  cannot change any rank.
- **Rank.** The evaluation matrix goes through the same exact mod-p
  block-RREF (16-bit-limb float64 GEMM) kernel validated against
  FLINT during the N=5 campaign, wrapped in an incremental basis so
  cumulative per-level ranks come out of one pass.

Epistemic status of any number produced this way: **certified lower
bound** on the true QQ rank (rank of an evaluation matrix ≤ rank_p ≤
rank_QQ), equal to it up to Schwartz-Zippel probability ~rank·deg/p
per run — the same rung as the N=5 L3 result, strengthenable by
rerunning at a second (prime, seed).

## Validation (known-value gauntlet)

| Case | Trees per level | Expected | Result |
|------|-----------------|----------|--------|
| N=3 d=2 L≤3 | 3/3/12/138 (matches engine) | [3, 6, 17, 116] | **PASS, 1 s** |
| N=5 d=1 L≤2 | 10/45/1440 (matches engine) | [10, 25, 145] | **PASS, 1 s** |
| N=4 d=1 L≤3 | 6/15/195/23010 (matches engine) | [6, 14, 62, 1260] | **PASS, 92 s** — the symbolic original took 52 min on a 31-worker r6i.8xlarge |

3/3 exact, S=2048 for the 1260 case (headroom 788). All tree censuses
match the symbolic engine's kept-generator histograms exactly.

## What this opens

1. **Exact a(4) for A395423** (`run_l4.py`): the full corrected
   11,937-tree L4 census at S=8192 points. Local-desktop scale
   (minutes, not days). A definitive a(4) extends the *published*
   sequence (keyword `more`). The float64-SVD bound to beat/confirm:
   d(4) ≥ 5,625, sample-limited.
2. **N=6 L3** (the L3(6) = 17,979 vs 18,249 hypothesis test): ~25.6M
   trees over 27 variables. The L2-jet stage dominates; estimated
   jaga-hours to low-days, vs "OOM during generation" before. Needs
   S ≈ 20K–24K samples (rank headroom over 17,979).
3. **N=3 L5** (a(5)): base jets to order 5 in 15 vars = C(20,5) =
   15,504 coefficients; L5 census ≈ C(≈12K, 2) ≈ 7×10⁷ trees; values
   need S > a(5) which is unknown (≥ tens of thousands). Feasible on
   jaga if a(5) ≲ 10⁵; the S-scaling is the real constraint.
4. **Any potential/exponent atlas at higher levels** — the same
   machinery with different base-H jets.

## Files

- `jet_eval/jets.py` — jet arithmetic over F_p (mult tables,
  derivative maps, vectorized over points).
- `jet_eval/nbody_jets.py` — base-H and chain jets, bracket-of-jets,
  census enumeration, per-level value extraction.
- `jet_eval/modp_linalg.py` — exact mod-p GEMM/RREF (same kernel as
  the N=5 campaign) + incremental cumulative-rank basis.
- `jet_eval/validate.py` — the known-value gauntlet above.
- `jet_eval/run_l4.py` — the a(4) census run.
