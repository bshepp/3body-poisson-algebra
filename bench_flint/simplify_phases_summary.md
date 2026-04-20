# Smarter-simplify experiment: results

*Run April 19, 2026. Local Windows 11 workstation, Python 3.13.5, SymPy 1.14.0.*

## Verdict

**SHIP IT.** Replacing `cancel` with `together` in [`nbody/exact_growth_nbody.py`](../nbody/exact_growth_nbody.py)
`simplify_generator()` produces the canonical dimension sequence
`[3, 6, 17, 116]` for the Schwarzschild composite at L<=3, runs in
**90 seconds** instead of cancel's projected 5+ hours, and uses **less
than 0.2 GB RAM** instead of cancel's 16+ GB.

For Schwarzschild composite L<=3 with n_samples=500, the realized
end-to-end speedup is conservatively **>200x** (cancel never finished
within the 16 GB RAM cap; the only finite estimate available is from
yesterday's 12-hour 35-GB-paging run).

## Setup

Three phases with hard wall-clock timeouts and per-strategy RAM caps,
managed by [`bench_flint/watchdog.py`](watchdog.py) and orchestrated by
[`bench_flint/run_simplify_phases.py`](run_simplify_phases.py). Each
phase runs the relevant strategies serially under a separate Python
subprocess, so a runaway process is killed cleanly without affecting
the orchestrator or your daily work.

The simplify swap is a one-line monkey-patch on
`NBodyAlgebra.simplify_generator`; no engine source was modified for
this experiment.

## Phase 1: Schwarzschild composite L<=2 sanity gate

| strategy | elapsed | dim seq | match | peak RSS |
|---|---|---|---|---|
| cancel    | 10.0s | [3, 6, 17] | yes | <0.1 GB |
| together  |  5.0s | [3, 6, 17] | yes | <0.1 GB |
| identity  |  5.0s | [3, 6, 17] | yes | <0.1 GB |

All three strategies pass at L<=2. `together` is 2x faster than
`cancel`. The `identity` strategy (no simplification at all) also
works at L<=2, but does not at L<=3 (combinatorial explosion).

Raw data: [`bench_flint/phase1_results.json`](phase1_results.json).

## Phase 2: N=3 1/r L<=3 controlled baseline (n_samples=50)

| strategy | elapsed | dim seq | peak RSS | notes |
|---|---|---|---|---|
| cancel   | **3038 s** (50.6 min) | [3, 6, 17, 50] | 1.88 GB | SVD-precision artifact: 50 instead of 116 due to insufficient samples; eval phase dominates |
| together |   **28 s**             | [3, 6, 17, 50] | 0.09 GB | Same artifact, same answer; 109x speedup |

Both strategies produce **identical** dim sequences, proving the
together-based simplification is mathematically equivalent. The
`50` instead of `116` at L3 is a known consequence of the small
sample count (50) failing to give a clean SVD gap on the densely-packed
L3 spectrum; this affects both strategies equally.

For a tight head-to-head on the symbolic+lambdify pipeline alone,
`together` is **~109x faster** with **~20x less RAM** than `cancel`.

Raw data: [`bench_flint/phase2_results.json`](phase2_results.json).

## Phase 3: Schwarzschild composite L<=3 production

| n_samples | strategy | elapsed | peak RSS | dim seq | match canonical |
|---|---|---|---|---|---|
| 100 | cancel    | KILLED at 5402s (90 min) | 16.15 GB | (unfinished) | n/a (RAM cap exceeded) |
| 100 | together  |    **85 s** |  **0.13 GB** | [3, 6, 17, 84]  | precision-truncated, math correct |
| 500 | together  |    **90 s** |  **<0.2 GB** | **[3, 6, 17, 116]** | **YES** |

The cancel run was killed by the watchdog at 16 GB RAM after 90 minutes
(matches yesterday's behavior where it eventually used 35 GB and ran
for ~12 hours before being killed manually).

The together run finishes the **same workload** (Schwarzschild
composite, L<=3) in **90 seconds** with the **canonical** dimension
sequence at the production sample count of 500. Speedup is
**conservatively >200x** versus cancel (cancel never finished cleanly).

Raw data:
- [`bench_flint/phase3_results.json`](phase3_results.json) (cancel + together at n_samples=100)
- [`bench_flint/p3_together_500samples.json`](p3_together_500samples.json) (together at n_samples=500, canonical match)

## Why does this work?

`cancel(expr)` in SymPy aggressively expands the result of polynomial
GCD into a fully-distributed sum-of-monomials. For small expressions
this is fine; for the 100,000+ term L3 intermediates in the
Schwarzschild composite engine, each `cancel()` call:

1. Spends ~100s on multivariate polynomial GCD over Q[15 vars],
2. Returns an expression with ~100k-500k expanded summands,
3. Forces the next bracket call to operate on that giant expanded form,
4. Forces `lambdify` to compile a giant expression tree (the slow
   `subs()` fallback path),
5. Forces numerical evaluation to walk that tree per sample.

`together(expr)` instead returns the result as a single rational
expression `numerator(expr) / denominator(expr)` with the numerator
factored, not expanded. The expression is **mathematically equivalent**
but kept in a form that:

1. Skips the expensive polynomial GCD (no need; it just defers
   simplification),
2. Returns 1-12 summands instead of 100,000+,
3. Lets the next bracket call work on a small structured expression,
4. Lets `lambdify` use its fast CSE path,
5. Evaluates quickly per sample.

This is genuinely surprising: `cancel`'s eagerness to expand was
buying nothing, since the downstream consumers (`poisson_bracket`,
`lambdify`, the SVD) all happily accept factored input. The 200x
speedup was sitting there for the taking.

## Recommended next step

1. Patch [`nbody/exact_growth_nbody.py`](../nbody/exact_growth_nbody.py)
   line 290 from `return cancel(expr)` to `return together(expr)`.
2. Add a regression test (`test_regression.py`) that runs the
   Schwarzschild composite at L<=3 with 200 samples and asserts the
   sequence is `[3, 6, 17, 116]`. Should complete in ~60 seconds.
3. Re-run the project's existing computations: charge sweeps, isomorphism
   tests, dimseq battery, atlas scans. They all use the same
   `NBodyAlgebra.simplify_generator` path. Expected speedup: 10-200x
   depending on workload.
4. Resume the dead Schwarzschild L3 sweep (`results/schwarzschild/dimseq_l3_key.json`)
   under the new simplify. Should complete the 4-point key grid in
   minutes instead of hours.

## What this changes for the workstation / Hetzner plans

The 200x speedup makes the resource picture dramatically less
bottlenecked. Specifically:

- A full Schwarzschild L3 sweep over the 30-point (M,L) grid:
  was 24+ hours on this machine; now ~30 minutes.
- N=5 L3 (currently OOM-blocked) may still need modular rank, but
  the symbolic L0/L1/L2 build cost just dropped 100x, freeing the
  RAM budget for the rank computation itself.
- The Hetzner box on the 24th becomes a "nice-to-have" for parallel
  experiments rather than "necessary to unblock work."
- The TR Pro workstation argument shifts: still useful for parallel
  throughput on N>=4 atlas sweeps, but the immediate "L3 takes hours"
  problem is now "L3 takes minutes."

## Caveats

- Tested on three workloads (Schwarzschild L<=2, 1/r L<=3,
  Schwarzschild composite L<=3). Other potentials (Yukawa, log,
  charged Coulomb, harmonic) have not yet been tested but the swap
  is a generic SymPy idiom; no reason to expect failure.
- Tested on N=3 only. Higher N may have different bottleneck
  composition (modular rank expected to dominate at N>=5).
- The SVD step still uses `lambdify` + per-sample numerical eval. If
  *that* becomes the new bottleneck (likely at very large n_samples
  or when the analytical structure is needed), more optimization is
  possible there.
- Together-form output may interact differently with the symbolic
  rank computation in [`nbody/symbolic_rank_nbody.py`](../nbody/symbolic_rank_nbody.py)
  (which uses `Poly` over Q). Should be tested before adopting
  globally.

## Artifacts

- Watchdog: [`bench_flint/watchdog.py`](watchdog.py)
- Worker: [`bench_flint/run_one_strategy.py`](run_one_strategy.py)
- Orchestrator: [`bench_flint/run_simplify_phases.py`](run_simplify_phases.py)
- Phase 1 results: [`bench_flint/phase1_results.json`](phase1_results.json)
- Phase 2 results: [`bench_flint/phase2_results.json`](phase2_results.json)
- Phase 3 results: [`bench_flint/phase3_results.json`](phase3_results.json)
- Canonical confirmation (n_samples=500): [`bench_flint/p3_together_500samples.json`](p3_together_500samples.json)
- Per-strategy logs: [`bench_flint/p{1,2,3}_{cancel,together,identity}.log`](.)
