# What is the 3-body harmonic Poisson algebra?

*Closes the open question from `docs/conjectures.md` §3 line 309:
"the number 15 should be computable from the Lie algebra of SO(2) × …
acting on coupled oscillators."*

## Setup

N = 3 bodies in the plane (d = 2), each with unit mass, interacting
through the harmonic pairwise potential `V(r) = r²`. The pairwise
Hamiltonians are

```
H_ij = ½(p_i·p_i + p_j·p_j) + (x_i − x_j)² + (y_i − y_j)²
     = T_i + T_j + r²_ij
```

with phase space `ℝ¹²` coordinates `(x_i, y_i, p_{xi}, p_{yi})` for
`i = 1, 2, 3`. The canonical Poisson bracket is the standard one;
`u_ij = 1/r_ij` does **not** appear because the harmonic case lives
entirely in polynomial coordinates.

Repeated bracketing closes the algebra at dimension 15:

| Level | Cumulative dim | Source |
|-------|---------------:|--------|
| L₀ | 3   | the three pair Hamiltonians |
| L₁ | 6   | + 3 brackets |
| L₂ | 13  | + 7 new |
| L₃ | 15  | + 2 new |
| L₄ | 15  | **closure** |

(Cross-verified in three CAS systems: SymPy via `exact_growth.py`,
Wolfram via `mathematica/poisson_n3_d2_harmonic.wl`, and SageMath via
`sage/poisson_n3_d2_harmonic.sage`. All three reproduce
`[3, 6, 13, 15, 15]`.)

The question this document answers is: **which 15-dimensional Lie
algebra is this**?

## Candidate algebras of dimension 15

The naive search for simple Lie algebras of dimension 15:

| Algebra | Dim | Notes |
|---------|----:|-------|
| `so(6, ℝ)` (compact)  | 15 | Compact real form of D₃; equivalently `su(4, ℝ)` |
| `so*(6)`              | 15 | One of three non-compact real forms; isomorphic to `su(3, 1)`'s double cover quotient |
| `so(5, 1)`            | 15 | Lorentz algebra in 5+1 D |
| `so(4, 2)`            | 15 | Conformal algebra of Minkowski-3, isomorphic to `su(2, 2)` |
| `so(3, 3)`            | 15 | Split form |
| `sp(4, ℝ)` ⊕ trivial center? | no | `sp(4, ℝ)` is 10, not 15 |
| `u(3) ⊕ u(2) ⊕ ℝ`?    | 9 + 4 + 1 = 14 | wrong dimension |

So the dimension 15 alone narrows the answer to the family
`D₃ = A₃` — i.e. one of the real forms of the complex Lie algebra of
type D₃ ≅ A₃. The accidental isomorphism `so(6, ℂ) ≅ sl(4, ℂ)`
gives us the equivalent names `so(2p, q, ℝ) ↔ su(p, q, ℝ)`:

```
so(6)   ↔ su(4)         (compact)
so(5,1) ↔ sl(2, ℍ) = su*(4)
so(4,2) ↔ su(2,2)
so(3,3) ↔ sl(4, ℝ)
so*(6)  ↔ su(3,1)
```

Five real forms total. Each has a distinctive **Killing-form
signature**. That signature is what our numerical script reads off
the structure constants.

| Algebra      | Killing signature (p+, q−, z) | Type |
|--------------|------------------------------:|------|
| `so(6)`      | (0, 15, 0)                    | compact |
| `so(5, 1)`   | (5, 10, 0)                    | non-compact |
| `so(4, 2)`   | (8, 7, 0)                     | non-compact (conformal of M³) |
| `so(3, 3)`   | (9, 6, 0)                     | split |
| `so*(6)`     | (7, 8, 0)                     | non-compact (twisted) |

The signature for the compact form is *all negative*; sign conventions
vary, so the script normalizes by the dimension of the maximal
negative-definite subspace and reports both `p_minus` and `p_plus`.

## Why "Fradkin / SU(d)" doesn't apply directly

The Fradkin tensor argument identifies the dynamical symmetry of a
*single* isotropic harmonic oscillator in d spatial dimensions. The
single-particle, 2d-D phase-space algebra is `u(d)` (dim `d²`), not
some bigger group. For d = 2, `u(2)` has dimension 4 (1 Hamiltonian +
1 angular momentum + 2 traceless Fradkin components).

Our 3-body system is *not* a single isotropic oscillator. It's three
coupled oscillators (the pair potentials `r²_ij` *couple* the bodies).
After eliminating the centre of mass (which is a translational symmetry,
3 dim) the relative-motion algebra acts on a smaller phase space and
its dimension does not have to be `d²` of anything.

The right reference frame is **coupled-oscillator Bogoliubov
diagonalization**: write the 3-body harmonic Hamiltonian as a
quadratic form in `(x, p)`, diagonalize the spring-constant matrix
(eigenvalues from the K₃ Laplacian: 0, 3, 3), and reduce to one
zero-mode (centre of mass) plus two oscillators at frequency `√3`.

The Lie algebra of quadratic forms in (`x, p`) on the reduced
4-dimensional symplectic phase space (2 oscillator modes after CM
removal) is `sp(4, ℝ)` of dimension 10. The additional 5 generators
to reach 15 are the **CM phase-space coordinates**: the centre of
mass position, the total momentum, and certain quadratic-in-position
or quadratic-in-momentum invariants that survive only at zero
frequency. Their inclusion brings the count to 15 = 10 + 5.

This decomposition strongly suggests the algebra is **non-compact**
(it contains time-translation, which is a hyperbolic generator).
Without the numerical signature, two candidates remain plausible:
`sp(4, ℝ) ⋊ heisenberg(2)` doesn't quite fit (dimensions don't match
either way), but a homogeneous non-compact `so(p, q)` with q > p is
the natural ansatz.

## The numerical answer — the **Jacobi algebra** `sp(4, ℝ) ⋉ h₂`

Running [`harmonic_lie_algebra_id.py`](../harmonic_lie_algebra_id.py)
on the exact rational structure constants of
`results/algebra_structure/N3_d2_r2/` produces:

| Quantity | Value |
|---|---|
| Dimension | 15 |
| Antisymmetry of structure constants | ✓ |
| Killing form signature | **(6+, 4−, 5z)** |
| Center dim | 1 |
| Radical (kernel of Killing form) dim | 5 |
| `[rad, rad]` dim | **1** (i.e. radical is NOT abelian) |
| Derived series | [15, 15] (perfect: `g = [g, g]`) |
| Lower central series | [15, 15] (not nilpotent) |
| Solvable | False |
| Nilpotent | False |

The signature `(6+, 4−, 5z)` decomposes as a 10-dim semisimple quotient
with Killing signature `(6+, 4−)` plus a 5-dim radical. The
10-dim simple Lie algebras with Killing signature `(6+, 4−)` form a
unique isomorphism class:

```
sp(4, ℝ) ≅ so(3, 2)
```

(the 3D de Sitter algebra, also known as the anti-conformal algebra of
2D Minkowski). The 5-dim radical has a 1-dim center *and* a 1-dim
commutator subspace `[rad, rad]` — exactly the structure of the
**5-dim Heisenberg algebra `h₂`** with three position-like generators,
two momentum-like generators (or equivalently a 4-dim symplectic
representation), and one central element where `[p_i, q_j] = δ_{ij} c`.

So the identification is

> ```
> g_harmonic(N=3, d=2) ≅ sp(4, ℝ) ⋉ h₂
> ```

This is the **Jacobi algebra**, the Lie algebra of the *Jacobi group*
`Sp(4, ℝ) ⋉ H₂`. It is a classical object in mathematical physics:

- The Jacobi group is the natural symmetry group of the harmonic
  oscillator's coherent / squeezed-state coadjoint orbits.
- It acts on the bosonic Fock space and contains both the metaplectic
  representation of `Sp(4, ℝ)` and the Schrödinger representation of
  the Heisenberg group.
- Its representation theory is fully classified (Jacobi forms;
  Berndt–Schmidt; Eichler–Zagier).

Saved data:
[`results/algebra_structure/harmonic_n3_d2_identification.json`](../results/algebra_structure/harmonic_n3_d2_identification.json).

### Why the structure makes physical sense

1. **The `sp(4, ℝ)` factor** is the dynamical symmetry of the
   *internal* (CM-removed) 4-dim phase space of two coupled
   oscillators. After Bogoliubov-diagonalizing the K₃ spring matrix
   (eigenvalues 0, 3, 3), the two non-zero modes form a 4-dim
   symplectic space with `sp(4, ℝ)` as its full linear symplectic
   group's Lie algebra.

2. **The Heisenberg radical `h₂`** is the algebra of position +
   momentum *for the CM mode* (the zero-frequency mode that the
   spring matrix's kernel produced) plus the total Hamiltonian as
   the central element. `[X_{cm}, P_{cm}] = `(some scalar × `H`)` is
   the only non-trivial bracket among radical elements, giving the
   1-dim `[rad, rad]` that the script measured.

3. **Perfect (`g = [g, g]`)** because `sp(4, ℝ)` is simple and the
   semidirect-product action carries radical elements into the
   image. The 1-dim center being inside `[g, g]` is consistent — it
   arises from Heisenberg-style brackets within the radical.

4. **Signature `(6+, 4−)`** is exactly the de Sitter signature of
   `so(3, 2)`. The harmonic 3-body system's internal phase space is
   thus a de Sitter-like geometry under its natural symmetry group.

### Comparison with the conjecture phrasing

`docs/conjectures.md` §3 line 309 asked: *"15 should be computable
from the Lie algebra of SO(2) × … acting on coupled oscillators."*

That phrasing was on the right track but slightly conservative — the
answer is *not* SO(2) × something compact, but `Sp(4, ℝ) ⋉ H₂`. The
non-compact `sp(4, ℝ)` part contains rotational `so(2)` as a maximal
compact subalgebra (along with two more compact dimensions, totaling 4
compact directions inside the 10-dim `sp(4, ℝ)`). The Heisenberg part
adds 5 more directions including the central total Hamiltonian.

In a more detailed phrasing: the *maximal compact subgroup* of the
Jacobi algebra is `u(2) ⊕ u(1)` (dim 4 + 1 = 5), and the conjecture's
"SO(2) × …" gestures at `u(2) ⊃ so(2)`. The full algebra grows the
non-compact directions on top.

## Why this matters

The harmonic potential is the *only* pairwise potential tested that
produces a finite Poisson algebra. Every other potential — singular
(`1/rⁿ`, log, Yukawa, composite) and most regular ones (`r^n` for
n ≥ 4) — generates the universal `[3, 6, 17, 116]` sequence with
infinite Gelfand-Kirillov dimension. Identifying the harmonic algebra
as the **Jacobi algebra `sp(4, ℝ) ⋉ h₂`** completes the classification
of finite-dimensional pairwise Poisson algebras among the exceptional
potentials:

| Potential | Algebra | Dimension | Identification |
|-----------|---------|----------:|----------------|
| `r¹` (linear) | `L_{5,2}` filiform nilpotent | 5  | known Lie algebra |
| `r²` (harmonic) | **`sp(4, ℝ) ⋉ h₂` Jacobi algebra** | **15** | **this work** |
| `r³` (cubic)  | infinite-dim, 109 at L₃, non-nilpotent | (infinite) | structure characterized |
| `r^n` (n ≥ 4), 1/r^n, log, Yukawa, composite | universal [3, 6, 17, 116] | (infinite) | universality |

The Jacobi algebra is the *physicist's* dynamical symmetry algebra of
the simple harmonic oscillator (lifted to N=3 coupled oscillators);
its appearance here ties the harmonic exception cleanly to the
coherent-state / metaplectic representation theory of `Sp(4, ℝ)`.

## See also

- The full universality classification is summarized in
  `docs/conjectures.md` §1.
- Cross-CAS reproduction:
  - `exact_growth.py --potential harmonic`
  - `mathematica/poisson_n3_d2_harmonic.wl`
  - `sage/poisson_n3_d2_harmonic.sage`
- Numerical Lie-algebra identification: `harmonic_lie_algebra_id.py`
  and `results/algebra_structure/harmonic_n3_d2_identification.json`.
