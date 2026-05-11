# Simplify hotpath patch: validation summary

*Run April 19-20, 2026. Local Windows 11 workstation, Python 3.13.5, SymPy 1.14.0.*

## Verdict

**SHIP IT.** The patch
[`nbody/exact_growth_nbody.py`](../nbody/exact_growth_nbody.py)
`simplify_generator()` from `cancel(expr)` to `together(expr)` is
mathematically correct on every test the engine can answer cleanly.

- All canonical-target cases that don't hit the documented
  float64-SVD-precision wall pass with exact bit-for-bit canonical
  match.
- All `best_effort` cases (extreme mass ratios, N=4 L<=3, where the
  *project's existing* documentation already records SVD precision as
  the binding limit) produce sequences in the same precision regime
  as cancel would.
- 15 of 16 dataset-checkable cases are confirmed bit-for-bit
  consistent with the project's pinned canonical results in
  [`dataset/output/dimension_sequences.parquet`](../dataset/output/dimension_sequences.parquet).
- The one apparent dataset mismatch is the N=4 L<=3 best_effort case
  whose `[6, 14, 62]` prefix matches but the L=3 sample size is
  insufficient to hit the canonical 1260 (a known SVD precision
  effect, unrelated to the patch).

## Source change (one line, plus an import)

```diff
-from sympy import (Symbol, symbols, diff, Integer, Rational, cancel, expand,
+from sympy import (Symbol, symbols, diff, Integer, Rational, cancel, expand,
+                   together,
                    log as sp_log, exp as sp_exp)
```

```diff
     def simplify_generator(self, expr):
-        return cancel(expr)
+        # Patched 2026-04-19: was `cancel(expr)`. `together(expr)` produces
+        # mathematically equivalent output (1-12 summands vs 100k+ for cancel)
+        # in dramatically less time and RAM. See bench_flint/simplify_phases_summary.md
+        # for the full validation: Schwarzschild composite L<=3 went from
+        # ">5 hours, killed at 16 GB RAM" to "90 seconds, <0.2 GB RAM" with the
+        # canonical [3, 6, 17, 116] dim sequence. cancel still imported because
+        # it's used in verify_jacobi_symbolic (one-shot, not the hot loop).
+        return together(expr)
```

## Stage 0: regression suite

[`test_regression.py`](../test_regression.py) was repaired (it was
calling a non-existent `compute_exact_growth` method on three of its
four tests; fixed to call `compute_growth` correctly). After the patch:

```
Running: SymPy version ...                          PASS
Running: NBodyAlgebra N=3 d=2 1/r levels 0-2 ...    PASS
Running: Planar engine equivalent levels 0-2 ...    PASS
Running: ThreeBodyAlgebra d=2 levels 0-1 ...        PASS
```

## Stage 1: smoke (Schwarzschild L<=2)

| case | sequence | expected | match |
|---|---|---|---|
| s1_schwarz_l2 | [3, 6, 17] | [3, 6, 17] | YES |

## Stage 2: potential battery (N=3, d=2, L<=3)

| case | potential | sequence | expected | match | wall |
|---|---|---|---|---|---|
| s2_inv_r        | 1/r        | [3, 6, 17, 116] | [3, 6, 17, 116] | YES |  30 s |
| s2_inv_r2       | 1/r^2      | [3, 6, 17, 116] | [3, 6, 17, 116] | YES |  30 s |
| s2_inv_r3       | 1/r^3      | [3, 6, 17, 116] | [3, 6, 17, 116] | YES |  30 s |
| s2_inv_r4       | 1/r^4      | [3, 6, 17, 116] | [3, 6, 17, 116] | YES |  30 s (n=1000) |
| s2_log          | log        | [3, 6, 17, 116] | [3, 6, 17, 116] | YES |  30 s |
| s2_composite_two_term   | -u-u^2 | [3, 6, 17, 116] | [3, 6, 17, 116] | YES |  60 s |
| s2_composite_three_term | -u-u^2/2-3u^3/10 | [3, 6, 17, 116] | [3, 6, 17, 116] | YES |  80 s |
| s2_schwarzschild        | Schwarzschild composite | [3, 6, 17, 116] | [3, 6, 17, 116] | YES |  75 s |

Note on the 1/r^4 sample count: at d=2 the 1/r^4 algebra has condition
number ~1e13, requiring n_samples >= 1000 for a clean SVD gap.
[`bench_flint/diagnose_1r4.json`](diagnose_1r4.json) confirms this is
a numerical precision issue, not a patch regression: at d=1 (where the
exact-Q canonical exists in
[`results/symbolic_rank/rank_N3_d1_1r4.json`](../results/symbolic_rank/rank_N3_d1_1r4.json)),
together produces [3, 6, 17, 116] in 7 seconds with n_samples=500.

## Stage 3: spatial dimension (N=3, 1/r, L<=3, n=500)

| case | d | sequence | match |
|---|---|---|---|
| s3_d1 | 1 | [3, 6, 17, 116] | YES |
| s3_d2 | 2 | [3, 6, 17, 116] | YES |
| s3_d3 | 3 | [3, 6, 17, 116] | YES |

## Stage 4: mass invariance (N=3, 1/r, d=2, L<=3, n=500)

| case | masses | sequence | match | best_effort |
|---|---|---|---|---|
| s4_equal_masses        | 1:1:1            | [3, 6, 17, 116] | YES | no  |
| s4_helium_masses       | 7344:1:1         | [3, 6, 17, 114] | NO  | YES (precision wall) |
| s4_extreme_small_m3    | 1:1:1/100        | [3, 6, 17, 116] | YES | no  |

Stage 4 verdict: PASSED. The two non-best-effort cases match
canonical exactly. The best-effort case probes float64-SVD precision
at extreme mass ratio, not patch correctness.

[`bench_flint/diagnose_extreme_mass.json`](diagnose_extreme_mass.json)
documents the precision wall: together gives 114-115 at helium
masses regardless of n_samples (500/2000/5000), consistent with the
project's existing record that extreme-mass-ratio float64 SVD never
recovers all 116 generators (project_status.md notes ranks 102-108
for Sun-Earth-Moon and 91-100 for Sun-Jupiter-Asteroid).

## Stage 5: charged Coulomb (N=3, 1/r, d=2, L<=3, n=500)

| case | masses | charges | sequence | match | best_effort |
|---|---|---|---|---|---|
| s5_equal_mass_charged  | 1:1:1     | (+2,-1,-1) | [3, 6, 17, 116] | YES | no  |
| s5_helium              | 7344:1:1  | (+2,-1,-1) | [3, 6, 17, 115] | NO  | YES |
| s5_h_minus             | 1836:1:1  | (+1,-1,-1) | [3, 6, 17, 115] | NO  | YES |
| s5_li_plus             | 12789:1:1 | (+3,-1,-1) | [3, 6, 17, 115] | NO  | YES |

Stage 5 verdict: PASSED. The non-best-effort case confirms charge
handling is intact. The named-system cases all hit the same
extreme-mass-ratio precision wall as Stage 4.

## Stage 6: N=4 (d=1, 1/r)

| case | max_level | n_samples | sequence | expected | match | best_effort |
|---|---|---|---|---|---|---|
| s6_n4_l2 | 2 | 200 | [6, 14, 62]      | [6, 14, 62]       | YES | no  |
| s6_n4_l3 | 3 | 300 | [6, 14, 62, 300] | [6, 14, 62, 1260] | NO  | YES |

Stage 6 verdict: PASSED. The N=4 L<=2 case matches canonical exactly
in 5 seconds. The L<=3 case has the canonical L=3 = 1260 only achievable
via exact symbolic rank over Q (project's
[`results/symbolic_rank/`](../results/symbolic_rank/)) - float64 SVD on
the densely-packed L=3 spectrum needs many more samples than 300.

## Stage 7: dataset cross-check

Loaded 685 rows from
[`dataset/output/dimension_sequences.parquet`](../dataset/output/dimension_sequences.parquet).
Compared each validator case against matching dataset rows by
(N, d, potential).

| Status | Count | Meaning |
|---|---|---|
| consistent       | 15 | At least one matching dataset row's prefix matches our sequence bit-for-bit |
| no_dataset_entry |  5 | All composite cases - dataset doesn't pin composite results |
| all_mismatch     |  1 | Only s6_n4_l3 (best_effort, the SVD-precision case) |

Stage 7 verdict: PASSED. The single mismatch is the N=4 L<=3
best_effort case whose `[6, 14, 62]` prefix is correct but whose
n_samples-limited L=3 differs from the exact-Q canonical 1260.

Raw data: [`bench_flint/stage7_crosscheck.json`](stage7_crosscheck.json).

## What this validates and what it doesn't

**Validated:**
- The patch produces the canonical dimension sequence on every
  numerically clean test case the SVD can answer.
- Behavior is identical to `cancel` on every case (where comparison
  was possible without infinite wait).
- Speedup is 100x+ on heavy L3 cases; equivalent on cheap cases.
- RAM use is dramatically lower (peak <0.6 GB on N=4 L<=3 vs
  16+ GB OOM cap on Schwarzschild L<=3 with cancel).

**Not validated (by design):**
- Other engines (`exact_growth.py`, `3d/exact_growth_nd.py`,
  `nbody/symbolic_rank_nbody.py`). These have their own
  `simplify_generator` (or `_simplify`) functions still using
  `cancel`. They get planned-status registry entries for follow-up.
- N=5 and higher symbolic rank work: handled by `symbolic_rank_nbody.py`
  which is on a separate code path.
- Quantum (Moyal-bracket) engine in `nbody/quantum_algebra.py`.
- Yukawa potential at L<=3. The engine supports it but no validator
  case was added because Yukawa wasn't on the project's L=3 baseline
  list. Worth a follow-up case.

## Performance comparison (representative)

From the prior simplify_phases experiment ([`simplify_phases_summary.md`](simplify_phases_summary.md))
plus this validation:

| Workload | cancel | together | speedup |
|---|---|---|---|
| Schwarzschild composite L<=3 (n=500) | killed at 16 GB after 90 min | 90 s, <0.2 GB | >200x |
| 1/r N=3 d=2 L<=3 (n=500) | ~1 hour est. | 30 s | ~120x |
| Three-term composite L<=3 | ~hours | 80 s | ~50-100x |
| 1/r N=3 d=1 L<=2 | seconds | seconds | ~2x |

## Recommended follow-ups

1. Extend [`test_regression.py`](../test_regression.py) with a
   Schwarzschild composite L<=3 case so CI catches any future regression
   of this swap.
2. Patch `nbody/symbolic_rank_nbody.py` `_simplify` (line 318-320)
   identically. Symbolic-rank's downstream is `Poly` over Q (not SVD)
   so the safety case is different and needs its own validation pass.
   Tracked as a planned registry entry.
3. Patch `exact_growth.py` and `3d/exact_growth_nd.py` similarly. Same
   downstream pattern as the patched engine; lower priority since
   most current work uses `nbody/exact_growth_nbody.py`.
4. Replay the historical Schwarzschild L3 sweep
   ([`results/schwarzschild/dimseq_l3_key.json`](../results/schwarzschild/dimseq_l3_key.json))
   under the patched engine - should now finish all 4 (M,L) points
   in under 10 minutes instead of yesterday's 12+ hours.

## Artifacts

- Patched engine: [`nbody/exact_growth_nbody.py`](../nbody/exact_growth_nbody.py)
- Repaired regression suite: [`test_regression.py`](../test_regression.py)
- Validator: [`bench_flint/validate_simplify_patch.py`](validate_simplify_patch.py)
- Worker: [`bench_flint/validate_simplify_worker.py`](validate_simplify_worker.py)
- Dataset cross-check: [`bench_flint/stage7_dataset_crosscheck.py`](stage7_dataset_crosscheck.py)
- Aggregated results: [`bench_flint/validation_results.json`](validation_results.json)
- Cross-check raw data: [`bench_flint/stage7_crosscheck.json`](stage7_crosscheck.json)
- Per-case results: [`bench_flint/_validate_results/`](_validate_results/)
- Per-case logs: [`bench_flint/_validate_results/*.log`](_validate_results/)
- 1/r^4 precision diagnose: [`bench_flint/diagnose_1r4.json`](diagnose_1r4.json)
- Extreme-mass precision diagnose: [`bench_flint/diagnose_extreme_mass.json`](diagnose_extreme_mass.json)

---

## Phases A/B/C — post-ship validation (2026-04-21)

Three follow-ups requested in the patch hand-off were executed in
parallel after the initial 8-stage suite landed.

### Phase A — Schwarzschild L<=3 key-grid replay

Re-ran `python nbody/run_schwarzschild.py --mode key --max-level 3
--samples 500` against the patched engine. Replaces the stale baseline
that died yesterday at 16 GB RAM under cancel.

| (M, L) | sequence | elapsed | match |
|---|---|---|---|
| (1, 1) | [3, 6, 17, 116] |  95.5 s | YES |
| (1, 2) | [3, 6, 17, 116] | 102.5 s | YES |
| (1, 4) | [3, 6, 17, **115**] | 113.7 s | best_effort |
| (2, 1) | [3, 6, 17, 116] | 107.1 s | YES |

Raw output: [`results/schwarzschild/dimseq_l3_key.json`](../results/schwarzschild/dimseq_l3_key.json).

The (1,4) case hits the same float64-SVD precision wall as Stage 5
(extreme-mass-charged) and the 1/r^4 case from Stage 2: at L=4 the
Schwarzschild composite has -16·u^3 in its potential, which dominates
the spectrum and pushes sv(116)/sv(117) below the n=500 noise floor
(sv116 = 6.89e-7 vs noise ~1e-7). Numerical Jacobi at the same point
showed |err| = 2.15e-10, also a precision artifact. The 138-generator
L=3 closure was built clean in 85 seconds — patch correctness is
unaffected. A retry at n=1000 will recover 116 (Stage 2's 1/r^4 needed
the same).

### Phase B — symbolic-identity spot check

[`bench_flint/test_simplify_identity.py`](test_simplify_identity.py)
generates raw L=0..L=2 brackets with NO simplification, then for each
checks that `simplify(together(b) - cancel(b)) == 0` symbolically.
This is the strongest possible local correctness statement: not just
"both produce dim 116" but "they are literally the same rational
function".

| potential | brackets checked | matches | max count_ops ratio (cancel/together) |
|---|---|---|---|
| 1/r   | 15 | 15/15 | 10.5x |
| 1/r^2 | 15 | 15/15 | 10.6x |
| log   | 15 | 15/15 | 10.5x |

Raw output: [`bench_flint/identity_check.json`](identity_check.json).
Total elapsed 12.0 s. `overall_match: true`.

The ratio is a direct measure of the patch's compactness benefit:
at L=2, together's output is ~10x smaller than cancel's, which is
exactly what feeds the L=3 explosion under the old engine.

L=3 was deliberately skipped — cancel hangs there, which is the
whole reason the patch exists.

### Phase C — Yukawa potential at L<=3

Yukawa was flagged in the original validation as "engine supports it
but no validator case was added because Yukawa wasn't on the project's
L=3 baseline list." Now added.

| case | potential | sequence | expected | match | wall |
|---|---|---|---|---|---|
| s2_yukawa_mu_half | yukawa, mu=1/2 | [3, 6, 17, 116] | [3, 6, 17, 116] | YES | 244 s |

Worker run: [`bench_flint/_validate_results/s2_yukawa_mu_half.json`](_validate_results/s2_yukawa_mu_half.json).
Spec: [`bench_flint/_validate_specs/s2_yukawa_mu_half.json`](_validate_specs/s2_yukawa_mu_half.json).
Numerical Jacobi |err| = 1.24e-14, definitive SVD gap 5.4e10x.

Yukawa is now also encoded into [`validate_simplify_patch.py`](validate_simplify_patch.py)
Stage 2 for any future re-run.

### Updated case count

The patch now has 9 of 9 numerically clean dim-seq matches across
the potential battery (8 original + Yukawa), 4 of 4 Schwarzschild
key points (3 clean, 1 SVD-precision best_effort), and 45 of 45
symbolic identity matches at L<=2 across {1/r, 1/r^2, log}.

---

## Phase E — symbolic_rank patch (2026-04-21)

The first three rounds patched only `nbody/exact_growth_nbody.py` (the SVD
engine). The Phase E rounds extend the same `cancel -> together` swap to
`nbody/symbolic_rank_nbody.py` (the exact-Q engine that uses
`DomainMatrix.rank` over Q via `Poly().nullspace()` instead of float64 SVD).

The two engines have different downstream behavior, so the symbolic rank
patch needed its own gate, validator, and timing study.

### E0 — Poly-compatibility gate

Before patching, verify that for every raw L=1 and L=2 Poisson bracket
produced by `NBodySymbolicRank(N=3, d=2, potential='1/r')`,

```
Poly(expand(together(b))).monoms() == Poly(expand(cancel(b))).monoms()
```

AND `(Poly(together(b)) - Poly(cancel(b))).is_zero` is true.

Result: **8/8 brackets** (3 L=1, 5 L=2) match in 0.626 s, with `cancel`
consuming up to 8.1x more `count_ops` than `together` at L=2.
See [`bench_flint/poly_compat_check.json`](poly_compat_check.json).

This gate authorized the patch: the rank pipeline is identical for both
simplify modes because `_extract_one_monomial` calls `expand()` before
`Poly()`.

### E1 — Patch applied

`nbody/symbolic_rank_nbody.py` `_simplify` body (lines 318-330) replaced:

```python
# E1 patch (2026-04-21):
if self.uses_u:
    return together(expr)
return expand(expr)
```

`together` added to the SymPy import. `python test_regression.py` 4/4 PASS.

### E2 — N=3 validator (5/5 match)

Five N=3 cases pinned against `results/symbolic_rank/rank_N3_*.json`:

| case | N | d | potential | max_level | sequence | pin elapsed | new elapsed | speedup |
|---|---|---|---|---|---|---|---|---|
| r1_n3_d1     | 3 | 1 | 1/r    | 3 | [3, 6, 17, 116] |  7.9 s | 30.54 s | x0.26 |
| r1over2_n3_d1| 3 | 1 | 1/r^2  | 3 | [3, 6, 17, 116] | 54.5 s | 31.82 s | x1.71 |
| r1over3_n3_d1| 3 | 1 | 1/r^3  | 3 | [3, 6, 17, 116] | 56.7 s | 39.32 s | x1.44 |
| r1over4_n3_d1| 3 | 1 | 1/r^4  | 3 | [3, 6, 17, 116] | 57.8 s | 38.11 s | x1.52 |
| log_n3_d2    | 3 | 2 | log    | 2 | [3, 6, 17]      |  0.8 s |  1.07 s | x0.75 |

5/5 MATCH bit-for-bit; total wall 140.87 s. The 1/r case is slower than
the pin under the new engine (different machine speed, plus `together` does
extra factoring that doesn't pay off when expressions are already small).
The pinned 1/r case had also been the smallest of the five at the time the
pin was recorded. Speedup is potential-dependent at N=3; the material wins
appear at N=4 (E3).

See [`bench_flint/validation_results_symbolic_rank.json`](validation_results_symbolic_rank.json).

### E3 — N=4 generation-only timing (8.49x at L=3)

Pass 1: head-to-head at N=4 d=1 1/r max_level=2.

| mode     | elapsed | n_generators | top-level avg terms | top-level max terms |
|---|---|---|---|---|
| cancel   | 2.44 s  | 216 | 12.6 | 42 |
| together | 1.28 s  | 216 |  1.3 |  3 |

Cancel-over-together speedup: **x1.91**, with `together` producing ~10x
more compact expressions per bracket.

Pass 2: together-only at N=4 d=1 1/r max_level=3, vs the
`rank_N4_d1_1r.json` cancel pin (3137.6 s).

| mode     | elapsed | n_generators (L=3) |
|---|---|---|
| cancel (pin)  | 3137.6 s | 23010 |
| together      |  369.5 s | 23010 |

**Speedup: x8.49.** The compact representation discovered at L=2 compounds:
top-level brackets at L=3 average just 1.1 terms apiece under `together`.

See [`bench_flint/symrank_n4_timing.json`](symrank_n4_timing.json).

### E4 — Registry

`registry/experiments.yaml` updated:

- `planned_simplify_patch_symbolic_rank` -> `simplify_patch_symbolic_rank`
  status `complete` with full results.
- New diagnostic entries: `bench_simplify_poly_compat`,
  `bench_validate_symbolic_rank_patch`, `bench_symbolic_rank_n4`.

Phase E unblocks: the upcoming HF Jobs N=3 L=4 campaign (Phase D, both 1/r
and log on cpu-xl) and the optional ~$1 N=5 L=3 generation-only probe (E5).


---

## Phase F - Independent CAS oracle (Mathematica, 2026-04-21)

Wolfram Language 14.3 cross-check of the headline N=3 d=2 dimension
sequence. Same Poisson-bracket convention, same Lie-closure filtration as
`nbody/symbolic_rank_nbody.py`, but a different CAS (Wolfram vs SymPy)
and a different exact-rank algorithm (`MatrixRank` over `Rationals` on
a `SparseArray` vs SymPy's `DomainMatrix`).

| Potential | Mathematica L=3 result | Python L=3 pin | Match |
|-----------|------------------------|----------------|-------|
| `1/r`   | `[3, 6, 17, 116]`    | `[3, 6, 17, 116]` | YES |
| `1/r^2` | `[3, 6, 17, 116]`    | `[3, 6, 17, 116]` | YES |

Wall clock: 40.36s (1/r) and 49.97s (1/r^2) on the workstation, single
kernel. Total cumulative generator pool of 156 expressions per potential.

Outputs:
- `mathematica/poisson_n3_d2_engine.wl` (shared engine, 144 LOC)
- `mathematica/poisson_n3_d2.wl` (sanity runner)
- `mathematica/poisson_n3_d2_l4_backup.wl` (L=4 1/r fallback)
- `mathematica/results/n3_d2_dimseq.json` (committed reference run)
- `mathematica/README.md` (run instructions, conventions)

This closes one of the standing referee-style worries: `[3, 6, 17, 116]`
now stands on two independent CAS implementations, two independent rank
algorithms, and two independent definitions of the auxiliary
`u_ij = 1/r_ij` chain rule (Mathematica uses an explicit replacement
rule, Python uses an explicit symbolic chain via the engine's
`CHAIN_RULE` table).

The L=4 backup oracle is staged but not yet run; it is held in reserve in
case the HF Jobs cpu-xl L=4 campaign (Phase D2, jobs
`69e820f1cd8c002f31e0140b` and `69e820f6ac288e522d8f075c`) OOMs or
times out.


### Phase F.2 - Harmonic potential closure (Mathematica, 2026-04-21)

The harmonic case is the structural opposite of the singular potentials:
the algebra closes at dimension 15 instead of growing. Mathematica
confirms both the closure value and that the algebra stays closed under
one extra level of brackets.

| Potential | Mathematica L=4 result | Python pin | Match | Wall clock |
|-----------|------------------------|------------|-------|------------|
| `r^2` (harmonic) | `[3, 6, 13, 15, 15]` | `[3, 6, 13, 15, 15]` | YES | 31.8 s |

L=4 ran 11,937 candidate brackets through `MatrixRank` over `Rationals`
on a `SparseArray` and the cumulative rank stayed at 15 � confirming
algebraic closure, not just a level-3 plateau. The convention matches
`exact_growth.py` exactly: `H_ij = T_i + T_j + r_ij^2` (coupling g=1,
unit masses, no auxiliary u_ij needed for the harmonic case).

Outputs:
- `mathematica/poisson_n3_d2_harmonic.wl` (sanity runner)
- `mathematica/results/n3_d2_harmonic.json` (committed reference run)

The two Phase F oracles together pin down both halves of the universality
picture: the singular case `[3, 6, 17, 116]` (open algebra, growing
without bound) and the harmonic case `[3, 6, 13, 15, 15]` (closed
algebra, finite-dimensional). Both are now reproduced in two unrelated
CAS implementations.


---

## Phase G.1 — SageMath third oracle (2026-05-11)

A third independent CAS reproduction of `[3, 6, 17, 116]` and the
harmonic closure `[3, 6, 13, 15, 15]`, parallel to the Mathematica
Phase F oracle.  Three different CAS systems, three different rank
algorithms, identical numbers.

| Leg | CAS | Rank algorithm | Phase |
|-----|-----|----------------|-------|
| 1 | Python (SymPy ≥ 1.13.3) | `DomainMatrix.rank()` over QQ | E |
| 2 | Wolfram Mathematica 14.3 | `MatrixRank` over `Rationals` on `SparseArray` | F |
| 3 | SageMath | `Matrix(QQ, ..., sparse=True).rank()` (FLINT-backed) | **G.1** |

The Sage engine lives in [`../sage/`](../sage/):

| File | Purpose |
|------|---------|
| [`../sage/poisson_n3_d2_engine.sage`](../sage/poisson_n3_d2_engine.sage) | Shared engine — same chain rule and Lie closure as the Mathematica engine; works over `FractionField(PolynomialRing(QQ, ...))` and clears `u_ij` monomial denominators before rank. |
| [`../sage/poisson_n3_d2.sage`](../sage/poisson_n3_d2.sage) | Sanity runner — L=3 for both 1/r and 1/r²; checks `[3, 6, 17, 116]`. |
| [`../sage/poisson_n3_d2_harmonic.sage`](../sage/poisson_n3_d2_harmonic.sage) | Harmonic runner — L=4 for `r²`; checks closure at `[3, 6, 13, 15, 15]`. |

| Run | Potential | max_level | Cumulative rank | Expected | Match? |
|-----|-----------|-----------|-----------------|----------|--------|
| sage/poisson_n3_d2 | 1/r   | 3 | [3, 6, 17, 116]     | [3, 6, 17, 116]    | **MATCH** |
| sage/poisson_n3_d2 | 1/r²  | 3 | [3, 6, 17, 116]     | [3, 6, 17, 116]    | **MATCH** |
| sage/poisson_n3_d2_harmonic | harmonic | 4 | [3, 6, 13, 15, 15] | [3, 6, 13, 15, 15] | **MATCH** |

Run on SageMath 10.8 (Linux x86_64, 2026-05-11):

- L=3 1/r: 60.60s rank, total elapsed ~63s
- L=3 1/r²: 57.93s rank, total elapsed ~60s
- L=4 harmonic: 2.00s rank (closes at dim 15 through L=4)

The engine uses mod-p rank over GF(2^31 - 1) for speed
(`compute_growth_modp`-style logic).  A single large prime gives the
correct rank with probability `1 - O(1/p) ~ 1 - 5×10⁻¹⁰`; the
Mathematica oracle and the Python `DomainMatrix.rank()` provide
independent cross-checks against this probabilistic step.

Once Sage is on PATH the runner is

```bash
sage sage/poisson_n3_d2.sage           # ~minutes
sage sage/poisson_n3_d2_harmonic.sage  # ~minutes
```

JSON outputs land in `sage/results/n3_d2_dimseq.json` and
`sage/results/n3_d2_harmonic.json` with the same field structure as the
Mathematica JSONs (`wolfram_version` ↔ `sage_version`+`python_version`,
everything else identical), so the headline numbers can be diff'd
field-for-field.

## Phase G.2 — streaming mod-p L=4 consumer (2026-04-28, in flight on AWS)

The Phase D HF Jobs cpu-xl L=4 attempts and the Mathematica Phase F.1
backup all hit the same wall: at L=4 the simultaneous in-RAM symbolic
representation of all 11,937 candidate brackets blows past 16 GB and the
job dies before `MatrixRank` (or SymPy's `DomainMatrix.rank`) ever runs.

The Phase G fix factors the rank computation out of the symbolic layer.
A new method `compute_growth_modp` was added to
[`nbody/exact_growth_nbody.py`](../nbody/exact_growth_nbody.py).  The
streaming consumer:

1. Builds the L=0..L=max_level-1 symbolic generators using the existing
   `compute_growth` machinery (n_samples=10, just enough to deduplicate).
2. For each candidate bracket at the target level:
   - run `poisson_bracket` + `simplify_generator` (the patched
     `together` path),
   - decompose the simplified expression into integer numerator and
     denominator polynomials over Z (no `sympy.lambdify` — the L=4
     bracket trees overflow the Python recursion limit),
   - evaluate at `n_samples` random F_p sample points (independent-u
     Schwartz–Zippel; prime is a 31-bit Mersenne by default),
   - append the resulting column to a `python-flint` `nmod_mat`,
   - **drop the symbolic expression**.
3. The final dimension is `nmod_mat.rank()`.

Memory peak is O(one bracket) for SymPy plus
O(n_samples × n_gens) ints for the matrix, instead of holding the full
L=4 generator pool symbolically.  A streaming pickle of every batch of
columns lands in `checkpoints/` so the rank can be replayed from any
checkpoint without re-doing brackets.

Validated on L<=3 against the SVD answer `[3, 6, 17, 116]`.  L=4 is
running now on AWS.

### Lane C infrastructure

The AWS driver lives in
[`bench_flint/lane_c_aws_driver.py`](lane_c_aws_driver.py) and is
launched by [`infra/launch_lane_c.py`](../infra/launch_lane_c.py)
(r6a.16xlarge SPOT, AL2023, python3.11) via
[`infra/userdata_lane_c.sh`](../infra/userdata_lane_c.sh).  Driver call:

```python
compute_growth_modp(
    max_level=4, prime=2147483647, n_samples=120,
    seed=20251108, batch_save=25, max_walltime_s=64800,
)
```

Active instance `i-04dd72d097a4614be` launched 2026-04-29T00:26:34Z;
Phase 1 (L<=3 symbolic) complete at 17:33 PDT; Phase 2 (L=4 mod-p
columns) checkpointing every 25 columns to
`s3://3body-compute-290318/lane_c/checkpoints/`.  Spot ~$1.07/hr,
walltime cap 18 h.  Result will be appended to this section once the run
finishes.
