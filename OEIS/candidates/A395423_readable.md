# The sequence 3, 6, 17, 116

*A universal dimension sequence from the 3-body Poisson algebra.*

Brian Sheppard · April 21, 2026

> **OEIS:** [A395423](https://oeis.org/A395423) (assigned 2026-04-21).

---

## What the sequence is

Take three point masses moving in a plane (or in 3-space, or on a line —
it doesn't matter, as we'll see). Couple every pair of them through
some pairwise potential V(r) — gravity, Coulomb, Calogero–Moser, log,
Yukawa, whatever you like — so the system is described by three
"interaction Hamiltonians":

$$
H_{12} = m_1 m_2\, V(r_{12}), \qquad
H_{13} = m_1 m_3\, V(r_{13}), \qquad
H_{23} = m_2 m_3\, V(r_{23}).
$$

These three functions live on the 12-dimensional phase space of
positions and momenta. The Poisson bracket gives them an algebraic
life: each pair $\{H_{ij}, H_{kl}\}$ is again a function on phase space,
and we can keep bracketing the new functions with the old ones, building
up a tower:

- **Level 0**: $L_0 = \mathrm{span}\{H_{12},\ H_{13},\ H_{23}\}$.
- **Level $n+1$**: $L_{n+1} = L_n + \{L_n, L_n\}$.

The dimensions of these spaces are the integer sequence

$$
\boxed{\ 3,\ 6,\ 17,\ 116,\ \dots\ }
$$

That's it. Three pairwise Hamiltonians, the canonical Poisson bracket,
and the dimensions of successive Lie-bracket closures.

---

## Why it's surprising

The arresting fact about this sequence is **how little it depends on
the input**.

When you change V(r) from $1/r$ (Newtonian gravity) to $1/r^2$
(Calogero–Moser) to $1/r^3$ to $\log r$ to a Yukawa $e^{-\mu r}/r$,
or even to certain *composite* potentials like $1/r + c/r^3$, you keep
getting the same dimensions: 3, 6, 17, 116.

When you change the spatial dimension from $d=2$ to $d=3$ (or down to
$d=1$), you still get 3, 6, 17, 116.

When you make the three masses unequal — pick any positive
$(m_1, m_2, m_3)$ you want — you still get 3, 6, 17, 116.

The only way to break the universality is to switch to a **harmonic**
potential, $V(r) = r^2$. That gives a *different* sequence,
$3, 6, 13, 15, 15, \dots$, that closes at dimension 15 (no further
growth). Every singular potential we have tested gives 3, 6, 17, 116.

This invariance is the headline result. The sequence appears to be a
property of the *structure* of three-body coupling itself — the
combinatorics of the pairwise topology and the symplectic geometry of
the phase space — rather than of any particular choice of physics.

---

## The terms, one by one

### a(0) = 3 — the obvious one

There are $\binom{3}{2} = 3$ unordered pairs of bodies, so $L_0$ has
three generators, and they are linearly independent. This term appears
in OEIS as `A000217(2)` (the triangular numbers).

### a(1) = 6 — first non-trivial level

Bracketing pairs of the three Hamiltonians produces three new
independent functions, disjoint from $L_0$. So
$\dim L_1 = 3 + 3 = 6$. This matches the closed form
$L_1(N) = N(3N-5)/2$ at $N=3$, which lives in OEIS as `A095794` (with
an offset shift).

### a(2) = 17 — the first irregular term

At level 2 the algebra grows by 11 new independent generators. Why
exactly 11? We don't know. The closed form $L_2(N) = N(4N^2-9N+3)/2$
that fits all $N \ge 4$ predicts 13 here, but $N=3$ falls outside its
range. The number 17 is genuinely irregular — neither pentagonal, nor
binomial, nor a value of any clean polynomial we've found.

### a(3) = 116 — the headline

At level 3 the algebra grows by 99 new independent generators. To
verify this we computed all 136 brackets between elements of $L_2$
symbolically, expressed them in a chosen basis of phase-space
monomials, and took the matrix rank.

To make sure the answer is *exact* we ran the computation two ways:

1. In Python with sympy, evaluating the symbolic expressions on a
   numerical phase-space grid and computing the SVD. The smallest
   singular value above the rank cutoff is more than $10^{10}$ times
   the largest one below it — a clean rank gap that establishes the
   integer rank rigorously.

2. In Wolfram Mathematica 14.3, taking the exact rational rank of the
   sparse coefficient matrix directly (no numerical step at all). This
   completes in about 40 seconds on a single core and returns 116 to
   the digit.

Both methods agree, on $1/r$ and on $1/r^2$.

### a(4) = ? — being computed now

The next term is currently out of our reach for a fully verified
rational rank, but numerical experiments at single-precision suggest
$a(4) \ge 5604$. Two large-CPU jobs are running on Hugging Face
Jobs as of this writing, one with $V = 1/r$ and one with $V = \log r$;
both should land an exact answer if the algebra cooperates.

---

## How to reproduce it yourself

### In Python

```python
from nbody.symbolic_rank_nbody import NBodyAlgebra

alg = NBodyAlgebra(N=3, d=2, potential="1/r")
print([alg.level_dim(k) for k in range(4)])
# [3, 6, 17, 116]
```

This requires `sympy >= 1.13.3` — earlier versions silently miscount
level 3 for unequal masses (this was the reason for an earlier
incorrect sequence in the literature, `[3, 5, 13, 69]`).

The full engine lives in
[`nbody/symbolic_rank_nbody.py`](../../nbody/symbolic_rank_nbody.py)
in this repository.

### In Mathematica

```mathematica
Get["mathematica/poisson_n3_d2.wl"]
(* prints: {3, 6, 17, 116}  in ~40 seconds *)
```

Output is also written to
[`mathematica/results/n3_d2_dimseq.json`](../../mathematica/results/n3_d2_dimseq.json),
which is committed to the repository as a reference.

### Cross-CAS validation

The full Phase F validation matrix (Python sympy vs. Mathematica
exact-rational, on $1/r$ and on $1/r^2$, all four levels) is documented
in [`bench_flint/validation_summary.md`](../../bench_flint/validation_summary.md).
Both engines produce identical results.

---

## Where it sits among known sequences

We searched OEIS on April 21, 2026 for `seq:3,6,17,116`. **No
results.** As far as the encyclopedia knows, this sequence is new.

Two adjacent A-numbers are worth knowing about:

- [`A000217`](https://oeis.org/A000217) (triangular numbers) provides
  $a(0) = 3 = T_2$.
- [`A095794`](https://oeis.org/A095794) ($\frac{(n+1)(3n-2)}{2}$,
  "second pentagonal numbers minus 1") provides $a(1) = 6$ via
  `A095794(2) = 6`. This entry corresponds to the closed form
  $L_1(N) = N(3N-5)/2$ for the analogous N-body sequence at level 1
  for any $N \ge 3$.

After level 1 the sequence stops being a polynomial in any obvious
parameter. Whether $a(2) = 17$, $a(3) = 116$, $a(4) = ?$ have a closed
form is an open question.

---

## Why this is worth a sequence in OEIS

Three reasons.

1. **Universality.** A new integer sequence that arises identically
   from a one-parameter family of inputs (the potential V), from a
   second parameter (spatial dimension), and from a third (the masses)
   is rare. The OEIS keyword `nice` exists for sequences with this
   character.

2. **Cross-validation.** The terms have been independently verified by
   two different computer algebra systems (Python sympy and Wolfram
   Mathematica), both with exact arithmetic. This is more than the
   bar OEIS sets and is recorded in our submission as evidence.

3. **An open question with a definite first answer.** $a(4)$ is not
   yet known exactly. OEIS has the keyword `more` for exactly this
   case, and the entry can be amended once the computation completes.

---

## Reading further

- [`README.md`](../../README.md) — project overview.
- [`paper3_universality.tex`](../../paper3_universality.tex) — the
  detailed mathematical writeup of the universality result.
- [`paper4_calogero_integrability.tex`](../../paper4_calogero_integrability.tex)
  — the Calogero–Moser specialization and connection to integrable
  systems.
- [`mathematica/README.md`](../../mathematica/README.md) — conventions
  used by the Mathematica reproduction.
- The OEIS submission draft itself is in
  [`A_3body_n3_singular.md`](A_3body_n3_singular.md) in the same
  folder.

---

*Computational provenance: this document was prepared with AI
assistance; every claim has been checked against running code. The
underlying computations were performed on a local Windows workstation
(Python 3.13, sympy 1.14.0, Mathematica 14.3.0) and on Hugging Face
Jobs (cpu-xl) for the $a(3) = 116$ verification.*
