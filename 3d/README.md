# Track: 3D Extension — Spatial Three-Body Poisson Algebra

**Parent project:** `../preprint.tex` (Paper 1: planar/2D results)

**Paper:** Results in this directory are presented in
[`../paper3_universality.tex`](../paper3_universality.tex) (Paper 3:
universality conjecture, Section 5: Spatial Dimension Independence).

## Scientific Question

Does the Poisson algebra dimension sequence change when moving from the planar (2D) to spatial (3D) three-body problem?

The parent project proves that the sequence [3, 6, 17, 116] is invariant under:
- Mass ratios (including Tsygvintsev exceptional cases)
- Singular potential type (Newton 1/r and Calogero-Moser 1/r² give identical sequences)

The dependence on spatial dimension d is explicitly listed as an open question (preprint Section 5 item 4; `../conjectures.md` prediction 3).

## Key Predictions

- d(0) = 3 and d(1) = 6 are guaranteed in all dimensions (same generator/bracket count).
- d\_3D(n) ≥ d\_2D(n) at every level (the 2D algebra embeds into 3D via z=0, p\_z=0).
- Level 2 is the first level where d\_3D could exceed d\_2D = 17.
- The 1D case may show d\_1D(n) ≤ d\_2D(n) (fewer degrees of freedom to generate independence).

## Files

| File | Purpose |
|------|---------|
| `exact_growth_nd.py` | Core engine parameterized by spatial dimension d |
| `run_3d.py` | Run the 3D computation (default: level 2) |
| `run_1d.py` | Run the 1D computation (default: level 3, very fast) |
| `validate_2d.py` | Validate engine reproduces 2D results [3, 6, 17, 116] |
| `README.md` | This file |

## Usage

```bash
cd 3d/

# Step 1: Validate the engine reproduces known 2D results
python validate_2d.py                    # quick (level 2)
python validate_2d.py --max-level 3      # full  (level 3, ~30-60 min)

# Step 2: Run 3D computation
python run_3d.py                         # level 2 (first divergence test)
python run_3d.py --max-level 3           # level 3 (if level 2 looks good)

# Step 3: Run 1D for comparison
python run_1d.py                         # level 3 (fast in 1D)

# Custom runs via the engine directly
python exact_growth_nd.py -d 3 --max-level 2 --samples 1000
python exact_growth_nd.py -d 1 --max-level 3
python exact_growth_nd.py -d 2 --max-level 3 --resume
```

## Interpreting Results

| Outcome | Meaning |
|---------|---------|
| d\_3D(2) = 17, d\_3D(3) = 116 | Sequence is dimension-independent. Major surprise. Strong follow-up paper. |
| d\_3D(2) > 17 | Spatial dimension matters. Expected per conjecture. Determines nature of d-dependence. |
| d\_1D(n) < d\_2D(n) | Lower dimension constrains the algebra. Expected. |
| d\_1D(n) = d\_2D(n) | Sequence is dimension-independent even in 1D. Very surprising. |

## Isolation from Parent Project

- **No imports** from the parent project. All code is self-contained.
- **Separate checkpoints** stored in `checkpoints_dN/` subdirectories.
- **No modifications** to any parent project files.
- Dependencies are the same as the parent (numpy, sympy, scipy, matplotlib, mpmath).

## Mathematical Notes

The polynomial trick u\_ij = 1/r\_ij works identically in all spatial dimensions.
The chain rule formula du\_ij/dx\_i = -(x\_i - x\_j) · u\_ij³ has the same structure
regardless of whether r\_ij = |x\_i - x\_j| (1D), √(Δx² + Δy²) (2D), or
√(Δx² + Δy² + Δz²) (3D), because in every case r\_ij = √(sum of squares)
and the derivative follows from the same calculus.

What changes with d:
- Phase space grows: 6d variables (3d positions + 3d momenta) + 3 auxiliary u\_ij
- The Poisson bracket sum has 3d terms instead of 6 (2D) or 3 (1D)
- Symbolic expressions are larger, so computation is slower
- The space of functions is larger, allowing potentially more independence
