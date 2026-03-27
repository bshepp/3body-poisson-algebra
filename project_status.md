# Three-Body Poisson Algebra — Project Status & Roadmap

*Last updated: March 27, 2026*

---

## 1. Atlas Campaign Status

### Completed Configurations (16)

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

Additional completed work:
- Harmonic oscillator 1/r^-2 atlas (100x100) — finite algebra, rank 15
- 1/r^2 (Calogero-Moser, equal mass) atlas (100x100) — rank 116 at 87.9%, completed in 3.6h on c6i.4xlarge with 16 workers. Instance terminated.
- 1/r^2 vs 1/r^-2 triptych rendered (`aws_results/atlas_figures/triptych_1r2_vs_1r-2.png`)
- All atlas figures rendered (singles + triptychs) in `aws_results/atlas_figures/`
- Bell test completed locally (results in `nbody/bell_test_results/`)
- S3 data fully synced locally (9.04 GB), verified by `audit_atlas_data.py`
- Data integrity audit: 17/42 configs clean, 3 fixed from S3 stale sync

### In Progress (3)

| # | Task | Progress | Status |
|---|---|---|---|
| 1 | **Sun-Earth-Moon atlas** (1 : 3e-6 : 3.7e-8) | 11/100 rows | Spot reclaimed. Checkpoint safe on S3. Needs relaunch. |
| 2 | **Sun-Jupiter-Asteroid atlas** (1 : 9.5e-4 : 1e-10) | 7/100 rows | Spot reclaimed. Checkpoint safe on S3. Needs relaunch. |
| 3 | **Level-4 mpmath rank computation** | 667/15,000 rows (4.4%) | Spot reclaimed. Rank=667, plateau=0. ETA ~574h. Checkpoint safe. |

### Not Yet Started (6)

| # | Task | Notes |
|---|---|---|
| 1 | **Parametric exponent sweep** (1,015 values of n) | Script written (`parametric_atlas_scan.py`), not yet run at scale. See Section 3 below. |
| 2 | **Dusty plasma Yukawa atlas** | Prior run failed (exit code 1). Yukawa lambdification issue. |
| 3 | **Tritium/He-3 Yukawa atlas** | Instance terminated before producing data. |
| 4 | **SageMath verification** | Independent verification of dimension sequence. SageMath not yet installed. |
| 5 | **N=4 body Level-3** | Sequence [6, 14, 62] through L2; L3 not computed. |
| 6 | **Paper 3 (universality) finalization** | Depends on atlas campaign + parametric sweep. |

---

## 2. Physical Systems Catalog

A comprehensive catalog of three-body systems that can be (or have been)
investigated with the pairwise Poisson algebra framework. Systems marked
with a check have completed atlas computations.

### Gravitational

| System | Masses | Status | Notes |
|--------|--------|--------|-------|
| Equal-mass three-body | 1:1:1 | Done | Baseline atlas |
| Sun-Earth-Moon | 1 : 3e-6 : 3.7e-8 | 11% | Extreme mass ratio, slow eval. Spot reclaimed. |
| Sun-Jupiter-Asteroid | 1 : 9.5e-4 : 1e-10 | 7% | Restricted three-body regime. Spot reclaimed. |
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

| System | Potential | Notes |
|--------|----------|-------|
| Tritium / He-3 (3 nucleons) | Yukawa | Prior run terminated early. Needs relaunch. |
| Proton-neutron-neutron scattering | Yukawa | Not attempted. |
| Three-quark bound states | — | QCD, not Coulomb. Beyond current framework. |

### Plasma / Charged Particles

| System | Potential | Status |
|--------|----------|--------|
| Three ions in a Penning trap | 1/r (+1,+1,+1) + harmonic | Done (Penning trap config) |
| Three dust grains in dusty plasma | Yukawa | Failed (exit code 1). Needs fix. |
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
  to zero. But the restricted problem has a different Hamiltonian
  structure (the small body doesn't influence the primaries). Testing
  whether the restricted case produces the same sequence would be
  immediately relevant to mission design.

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
