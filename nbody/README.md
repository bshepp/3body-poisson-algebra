# Track: N-Body Extension -- Poisson Algebra for Arbitrary Particle Count

**Parent project:** `../preprint.tex` (planar 3-body results: dimension sequence [3, 6, 17, 116])

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

## Files

| File | Purpose |
|------|---------|
| `exact_growth_nbody.py` | Core engine: NBodyAlgebra(n_bodies, d_spatial, potential) |
| `validate_n3.py` | Validates engine reproduces N=3 results |
| `run_n4_d2.py` | N=4, d=2, 1/r computation |
| `run_n4_d1.py` | N=4, d=1, 1/r computation |
| `run_n4_d3.py` | N=4, d=3, 1/r computation |
| `run_n4_mass.py` | N=4 mass invariance test (3 configs) |
| `run_potential_1r3.py` | N=3, 1/r^3 potential test |
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

# Generic usage
python exact_growth_nbody.py -N 4 -d 2 --max-level 2
python exact_growth_nbody.py -N 5 -d 2 --max-level 1    # N=5 (expensive)
```

## Implications for the Conjecture

The conjecture (from `../conjectures.md`) now has evidence at TWO values of N:

- **N=3**: sequence [3, 6, 17, 116] -- mass-invariant, d-independent, potential-type-independent
- **N=4**: sequence [6, 14, 62, ...] -- mass-invariant, d-independent (through L2)

The sequence depends only on N and the singularity class (singular vs regular).
All other parameters (masses, spatial dimension, pole order) wash out.

## Isolation from Parent Project

- No imports from the parent project. All code is self-contained.
- Separate checkpoints in `checkpoints_N{n}_d{d}_{potential}/` subdirectories.
- No modifications to any parent project files.
- Dependencies: numpy, sympy, scipy, matplotlib.
