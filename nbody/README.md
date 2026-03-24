# Track: N-Body Extension -- Poisson Algebra for Arbitrary Particle Count

**Parent project:** `../preprint.tex` (Paper 1: planar 3-body results)

**Paper:** All results in this directory are presented in
[`../paper3_universality.tex`](../paper3_universality.tex) (Paper 3:
universality conjecture).

## Scientific Questions

1. What is the N=4 dimension sequence?
2. Is the dimension sequence mass-invariant for N=4?
3. Is the dimension sequence d-independent for N=4?
4. Does the 1/r^3 potential give the same sequence as 1/r and 1/r^2 for N=3?

## Results (March 15, 2026)

### N=4 Dimension Sequence (NEW -- first ever computed)

| Level | dim (N=4) | dim (N=3 ref) |
|-------|-----------|---------------|
| 0 | 6 | 3 |
| 1 | 14 | 6 |
| 2 | 62 | 17 |

SVD gap at index 62: ratio 3.4 x 10^11 (definitive).

Notable: Level 1 has 15 candidates (C(6,2)) but only 14 are independent.
Three pairs with disjoint bodies ({H12,H34}, {H13,H24}, {H14,H23}) bracket
to zero, but there is one additional linear dependence among the 12 non-zero
brackets.

### N=4 Mass Invariance

| Config | Masses | Sequence |
|--------|--------|----------|
| Equal | 1:1:1:1 | [6, 14, 62] |
| Hierarchical | 100:10:1:1 | [6, 14, 62] |
| Mixed | 3:7:11:2 | [6, 14, 62] |

**Mass invariance holds for N=4.**

### N=4 Spatial Dimension Independence

| d | Phase space | Sequence |
|---|-------------|----------|
| 1 | 8D + 6 aux | [6, 14, 62] |
| 2 | 16D + 6 aux | [6, 14, 62] |
| 3 | 24D + 6 aux | [6, 14, 62] |

**d-independence holds for N=4.**

### 1/r^3 Potential (N=3, d=2)

| Potential | Sequence |
|-----------|----------|
| 1/r | [3, 6, 17, 116] |
| 1/r^2 | [3, 6, 17, 116] |
| 1/r^3 | [3, 6, 17, 116] |

**Universality across pole orders confirmed with third potential type.**

### Charge-Sign Invariance (Helium Atom)

Flipping interaction signs (attractive ↔ repulsive) does not change the
dimension sequence. Verified with:
- All-attractive gravity (control)
- All-repulsive (+1, +1, +1)
- Mixed helium (+2, −1, −1): nucleus-electron attractive, electron-electron repulsive

All produce [3, 6, 17, 116]. The multi-epsilon atlas comparison tool
(`helium_atlas.py`) provides detailed gap-ratio and singular-value
spectrum comparisons across configuration space, supporting both
1/r and 1/r² potentials.

### Multi-System Universality Survey (March 2026)

Extended the charge and mass tests to 21 physical three-body systems.
Key findings:

**Gravitational systems (1/r, no charges):**
The original survey reported [3, 5, 13, 69] for all 7 unequal-mass configs.
**This was a SymPy version artifact** (SymPy 1.10.1 on the original AWS AMI
failed to lambdify 63/156 level-3 expressions, causing dimension undercounting).
Re-validation with SymPy 1.13.3 confirms **[3, 6, 17, 116]** for both equal
and unequal masses, including Three Galaxies (1:2:3) and Binary Star + Planet
(1:1:0.001). A mass ratio sweep across 25 points from m₃=0.001 to 10⁶
confirmed [3, 6, 17] at level 2 for all ratios.
(Verified March 23, 2026 — see mass ratio sweep in session_log.md.)

**Charge-coupled systems (1/r with charges):**
| System | Charges | Masses | Sequence |
|--------|---------|--------|----------|
| Helium | +2,−1,−1 | 7294:1:1 | [3, 6, 17, 116] |
| H⁻ Ion | +1,−1,−1 | 1836:1:1 | [3, 6, 17, 116] |
| Positronium Ps⁻ | +1,−1,−1 | 1:1:1 | [3, 6, 17, 116] |
| Muonic Helium | +2,−1,−1 | 7294:1:207 | [3, 6, 17, 116] |
| Li⁺ Ion | +3,−1,−1 | 12789:1:1 | [3, 6, 17, 111] |
| H₂⁺ Ion | +1,+1,−1 | 1836:1836:1 | [3, 6, 17, 115] |
| Penning Trap | +1,+1,+1 | 1:1:1 | [3, 6, 17, 116] |

**New potential types:**
| System | Potential | Sequence |
|--------|-----------|----------|
| 2D Vortices | log(r) | [3, 6, 17, 116] |
| Composite | 1/r + 1/r² | [3, 6, 17, 116] |
| Dusty Plasma | Yukawa (e^{-μr}/r) | In progress |

**Engine extensions for the survey:**
- Logarithmic potential V ~ log(r) for 2D vortex dynamics
- Yukawa potential V ~ exp(-μr)/r for nuclear and screened Coulomb
- External harmonic potential V ~ ½mω²r² for Penning trap confinement
- Composite potentials (sums of u^p terms)
- CSE-based fallback compiler for deeply nested expression trees

## Files

| File | Purpose |
|------|---------|
| `exact_growth_nbody.py` | Core engine: NBodyAlgebra(n_bodies, d_spatial, potential, charges, masses) |
| `expansion_configs.py` | Multi-System Survey scenario definitions (21 systems) |
| `run_expansion_dimseq.py` | AWS orchestrator: dimension sequences for all survey scenarios |
| `expansion_analysis.py` | Post-processing: comparative plots and summary from survey data |
| `run_pn_aws.py` | AWS orchestrator: composite/PN potential tests |
| `run_composite_test.py` | Composite potential universality test (local) |
| `run_post_newtonian.py` | Static 1PN three-body computation (local) |
| `run_pn_mass_test.py` | 1PN mass invariance test (local) |
| `validate_n3.py` | Validates engine reproduces N=3 results |
| `run_n4_d2.py` | N=4, d=2, 1/r computation |
| `run_n4_d1.py` | N=4, d=1, 1/r computation |
| `run_n4_d3.py` | N=4, d=3, 1/r computation |
| `run_n4_mass.py` | N=4 mass invariance test (3 configs) |
| `run_potential_1r3.py` | N=3, 1/r^3 potential test |
| `run_helium.py` | Helium Coulomb algebra experiments (3 charge configs) |
| `helium_atlas.py` | Charge-sign invariance atlas comparison tool |
| `README.md` | This file |

## Usage

```bash
cd nbody/

# Validate engine against known N=3 results
python validate_n3.py

# N=4 computation (default: level 2)
python run_n4_d2.py
python run_n4_d2.py --max-level 3    # push to level 3 (expensive)

# Potential test
python run_potential_1r3.py

# Mass invariance
python run_n4_mass.py

# d-independence
python run_n4_d1.py
python run_n4_d3.py

# Helium charge-sign invariance
python run_helium.py

# Compare charged vs all-attractive atlas data
python helium_atlas.py compare               # 1/r (default)
python helium_atlas.py compare --potential 1/r2   # 1/r² comparison
python helium_atlas.py orbits --potential 1/r2

# Generic usage
python exact_growth_nbody.py -N 4 -d 2 --max-level 2
python exact_growth_nbody.py -N 5 -d 2 --max-level 1    # N=5 (expensive)
```

## Implications for the Conjecture

The conjecture (formally stated in [`../paper3_universality.tex`](../paper3_universality.tex),
Conjecture 7) has been **refined** by the Multi-System Survey:

- **N=3 gravitational (all mass ratios)**: sequence [3, 6, 17, 116] —
  universal across 1/r, 1/r², 1/r³, log(r), composite, all spatial
  dimensions, and all mass ratios tested (0.001 to 10⁶)
- **N=3 charge-coupled**: sequence [3, 6, 17, 116] for most charge
  configs (He, H⁻, Ps⁻, muonic He, Penning trap)
- **N=3 high-charge deviations**: sequences [3, 6, 17, 111] (Li⁺) and
  [3, 6, 17, 115] (H₂⁺) — possible SVD conditioning artifacts at
  level 3 (investigation pending)
- **N=4**: sequence [6, 14, 62, ...] — mass-invariant, d-independent

**Note (March 23, 2026):** The original survey report of [3, 5, 13, 69] for
unequal-mass gravitational systems was a SymPy version artifact, not a
real second universality class. See session_log.md for full diagnostic.

## Isolation from Parent Project

- No imports from the parent project. All code is self-contained.
- Separate checkpoints in `checkpoints_N{n}_d{d}_{potential}/` subdirectories.
- No modifications to any parent project files.
- Dependencies: numpy, sympy, scipy, matplotlib.
