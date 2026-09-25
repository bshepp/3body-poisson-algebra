# PENDING REVIEW: external evaluation, September 2026

> **Status: UNREVIEWED. Nothing in this folder has been accepted into the
> project.** No paper, doc, result file, or OEIS text was changed. Each item
> below is a proposal for Brian to accept, reject, or send back.
>
> An earlier version of this review was committed as `review_2026_09/`.
> It was **struck (reverted)** because it mixed proven and unproven claims
> and did not check prior repo work first. This file replaces it.

## How to read the labels

| label | meaning |
|---|---|
| **PROVEN** | A mathematical argument, or an exact computation with both a lower and an upper bound. |
| **LOWER BOUND** | An exact rank mod p of an evaluation matrix. It can undercount but never overcount, so it certifies "≥". It does **not** certify "=". |
| **EVIDENCE** | Consistent, reproducible, but not a proof. |
| **PREVIOUSLY KNOWN** | Already found in the repo; listed so it is not re-credited. |

Scope checked: `papers/preprint.tex`, which is the Zenodo deposit
(`3body_poisson_algebra.pdf` is byte-identical to `preprint.pdf`), read in
full. For papers 2–4: abstracts plus paper 2's syzygy section. Docs
searched for overlap: `project_status.md`, `session_log.md` (tail),
`peer_review_analysis.md`, `adversarial_analysis.md`,
`related_literature_2026-07-19.md`,
`collision_syzygy/COLLISION_SYZYGY_REPORT.md`, README. The overlap check
was done *after* the computations, not before.

---

## Part A: What needs fixing (in existing project material)

### A1. "Super-exponential growth" and "infinite GK dimension" (Zenodo paper: title, abstract, Θ(n²) remark)
**PROVEN (argument; pending your check).**
* A_n = A_{n−1} + {A_{n−1}, A_{n−1}} contains Lie words of length up to
  2^n. Any algebra with polynomial growth L^k in word length therefore
  grows like 2^{kn}, exponentially in n. The per-level ratios cannot tell
  exponential from super-exponential.
* Degree bound: give u total degree 1. A bracket raises total degree by at
  most 2: D_x can raise it by 3 via ∂u/∂x = −(x_i−x_j)u³, and ∂_p lowers
  it by 1. So a length-L word has degree ≤ 4L − 2. Every element is a
  translation-invariant function (of coordinate differences, momenta, and
  u, with u²r² = 1), so dim(words of length ≤ L) = O(L^k) with k ≤ 5 in
  1D and k ≤ 10 in the plane. Hence **GKdim is finite**, and d(n+1)/d(n)
  is eventually bounded by about 2^k.
* The bound does not bind at computed levels; growth through level 4 is
  genuinely fast.
* Repo status: infinite GK is conjectured throughout. Not previously
  addressed.

### A2. "Unchanged by the spatial dimension" (Zenodo paper: Discussion; Future directions item 4; paper 3; OEIS comments)
**Verified only through level 3.** At level 4 there is EVIDENCE of
divergence, not proof (see B4). Suggested wording until B4 is resolved:
"verified for d = 1, 2, 3 through level 3".

### A3. Yukawa listed as confirming universality (Zenodo paper: Future directions item 2)
**Factual (from `docs/project_status.md`, Yukawa survey).** The runs used a
4-term Taylor truncation, V ≈ Σ (−μ)^k/k! · u^{1−k}, which is a composite
power potential, not e^{−μr}/r. The sentence should say so, or the result
should be rerun with an exact exponential representation (B3 shows how).

### A4. d(4) ≥ 5,604 (Zenodo paper: abstract, theorem, remark)
This is a float64 SVD estimate with boundary gap ≈ 1.2, so it is not a
certified bound. If B4 is accepted, replace it with the certified
**d(4) ≥ 5,914**.

### A5. Stability-atlas paragraph (Zenodo paper, Future directions item 6: "rank drops at Lagrange/Euler ... encoding stability boundaries")
**PROVEN (standard fact).** Real-analytic functions that are linearly
independent on phase space stay linearly independent on every open ball.
Local rank below 116 in an ε-ball is therefore numerical conditioning,
not algebra. This is consistent with the repo's own
collision-syzygy finding (A6).

### A6. Paper 2, §"Deep and soft syzygies" (116 → 124, "wake up")
**PREVIOUSLY KNOWN.** `collision_syzygy/COLLISION_SYZYGY_REPORT.md` and the
README already identify ranks > 116 as a numerical-rank bug. Paper 2's
text was never updated. The fix is editorial.

### A7. Free-Lie bound in `docs/related_literature_2026-07-19.md`
**PROVEN (computation; pending your check).** The note gives the free bound
under the depth filtration as 3, 6, 32, 1318. That sums *all* Lie words of
length ≤ 2^n, but the depth filtration reaches only some of them. The
correct free-Lie depth-filtration dimensions are **3, 6, 17, 119**
(`scripts/free_lie_depth.py`). Consequence: level 2 saturates the free
bound, and the OEIS readable's "17 is genuinely irregular" is not
accurate: 17 = 3 + 3 + 8 + 3 is the free count.

### A8. Speculative material (primes/GUE, cosmology, Bell)
Opinion, not a finding. None of it is in the Zenodo paper. Recommend
keeping it out of the paper track.

### Withdrawn from the earlier version
* "Paper 4 asks the wrong question": **PREVIOUSLY KNOWN** and already
  stated in paper 4 and in the February 2026 adversarial review. Withdrawn.
* "Spatial-dimension independence breaks at level 4": **withdrawn as
  stated.** Two lower bounds cannot show a difference; see B4.

---

## Part B: Things that are new

### B1. 3, 6, 17, 116 certified for the actual phase-space functions (1D and planar)
**PROVEN, given the repo's exact ℚ rank.** The repo's exact rank treats
u_ij as free symbols, which gives an upper bound on the true function
dimension. The matching lower bound previously came only from float64 SVD
(this gap was flagged in `peer_review_analysis.md` §5 but not closed).
`scripts/formal_vs_physical.py` evaluates every generator exactly mod p
at points satisfying u = 1/(x_i−x_j) in 1D, or u²r² = 1 in the plane.
Result: [3, 6, 17, 116] in both. Lower bound = upper bound.

### B2. Everything through level 3 is free except one relation
**PROVEN.**
* Free Lie algebra, same filtration: 3, 6, 17, 119 (A7).
* By word length (the 1/r algebra is graded by word length: weights x:2,
  p:−1, u:−2 give a length-L word weight 1 − 3L):

  | length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|---|
  | free, inside A_3 | 3 | 3 | 8 | 18 | 24 | 36 | 24 | 3 |
  | physical, inside A_3 | 3 | 3 | 8 | 17 | 23 | 35 | 24 | 3 |

* There is exactly one non-free relation, at word length 4. It was found
  by exact nullspace over ℚ (`scripts/relation4.py`, 1D, equal masses) and
  has small integer coefficients. The other two missing dimensions at
  level 3 are its consequences.
* Not yet checked: the relation's form for generic masses and in 2D.

**Interpretation (opinion).** This largely explains the potential-universality
at levels ≤ 3: for a generic potential, the algebra is nearly free at low
depth.

### B3. Pure exponential potential V = e^{−r} gives 3, 6, 17, 116
**PROVEN for 1D, one ordering (x1 > x2 > x3), at masses (1,1,1), (1,2,3),
(3,5,11).** Not tested: generic symbolic masses, all orderings, planar.

* Representation: on the ordering x1 > x2 > x3, r_ij = x_i − x_j and
  w_ij = e^{−(x_i−x_j)} with ∂w_ij/∂x_i = −w_ij and ∂w_ij/∂x_j = +w_ij.
  Everything is an exact polynomial in (p, w), with no Taylor truncation.
* Proof: substituting w13 = w12·w23 (exact on this ordering) leaves
  polynomials in (p1, p2, p3, w12, w23). These five are algebraically
  independent functions on phase space, so the ℚ rank of the
  monomial-coefficient matrix *is* the function-space dimension.
  `scripts/exp_exact.py`: exact ℚ ranks [3, 6, 17, 116] at all three
  mass triples; also [3, 6, 17, 116] with w free.
* Why it was run: to test the hypothesis that potentials whose derivatives
  collapse (V′ = −V) give a *smaller* algebra. That hypothesis is
  **refuted**. The result instead fits B2: at depth ≤ 3 the algebra is
  nearly free for essentially any non-polynomial potential.
* Relevance to A3: the same exact-variable trick (w = e^{−μr} alongside
  u = 1/r) would allow an untruncated Yukawa run.
* Process note: `scripts/exp_potential.py` (the first, sampling version)
  had two bugs: rational coefficients truncated to integers, and sample
  points redrawn per row. Its buggy output (18 at level 2) violated the
  Jacobi identity, which is how both bugs were caught. The exact script
  supersedes it.

### B4. d(4) lower bounds
**LOWER BOUND.** `scripts/d4.py` uses the repo's derivative pipeline: level
0–3 generators built symbolically as exact polynomials (coefficients mod
p), first derivatives symbolic, then brackets evaluated exactly in GF(p)
at points on real phase space. There is no floating point anywhere. The
rank is split by word-length block.

| case | result | runs | largest block rank / samples |
|---|---|---|---|
| planar | **d(4) ≥ 5,914** | 1 (seed 17) | 1,213 / 2,500 |
| 1D (all orderings) | **d(4) ≥ 5,042** | 2 (seeds 17, 99), identical | 960 / 2,500 |

* The planar lower bound improves on the repo's float bound of ≥ 5,625.
* **Not proven:** that either number is the exact value, and that 1D and
  planar differ. Both need an exact **upper** bound: the ℚ rank of the
  formal monomial-coefficient matrix of the level-4 brackets, computed per
  word-length block. The 1D case is cheaper and would settle A2. The
  planar case would settle a(4) for A395423.
* EVIDENCE for divergence: 1D and planar block ranks agree at every word
  length ≤ 8 and split at lengths 9–12 (e.g. 550 vs 609 at length 9).
* The planar result needs a second prime and seed before anyone relies
  on it.

### B5. 1D definition subtlety
**LOWER BOUND = formal count.** The word-length-7 piece has dimension 272
on the full line (all orderings; equals the formal count), but 215 if the
bodies are confined to one ordering x1 > x2 > x3. Levels ≤ 3 are
unaffected. The 1D definition should say which is meant. (An earlier
message read the 215 as "formal overcounts"; that was wrong and was
retracted.)

### B6. Word-length-graded dimensions (1D, full line) vs A027376
**LOWER BOUND** (exact mod-p evaluation on real phase space; formal count
agrees through L = 7). g_L = 3, 3, 8, 17, 44, 103, 272, **572** for
L = 1..8 (`scripts/graded.py` to L = 7; `scripts/graded_fast.py` to
L = 8). The planar run matched 1D through L = 6.

A027376 (Witt numbers, free Lie algebra on 3 generators) is the exact
per-length upper bound for any 3-generated Lie algebra:

| L | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| A027376 | 3 | 3 | 8 | 18 | 48 | 116 | 312 | 810 |
| physical g_L | 3 | 3 | 8 | 17 | 44 | 103 | 272 | 572 |
| shortfall | 0 | 0 | 0 | 1 | 4 | 13 | 40 | 238 |

* The shortfall 1, 4, 13, 40 looked like (3^(L−3) − 1)/2, which predicted
  689 at L = 8. **Refuted:** the value is 572. Record this so the pattern
  is not rediscovered.
* The growth ratio g_{L+1}/g_L runs 2.6, 2.3, 2.6, 2.1 and falls well
  below A027376's (about 2.6 at L = 8). This is EVIDENCE consistent with
  A1 (finite GK dimension), not proof.
* d(3) = 116 = A027376(6) is a coincidence (d(3) mixes lengths 1–8), as
  `related_literature_2026-07-19.md` already concluded. "Cf. A027376" in
  A395423 is appropriate.

### B7. Where the single length-4 relation comes from
**Mixed labels, per line below. 1D, N = 3, bracket length 4 only.**

* **The same relation for different potentials (EXACT over ℚ).** The
  relation has identical coefficients for V = 1/r, 1/r², 1/r³ (equal
  masses). Its coefficients change with the masses (e.g. ½, ⅔, ⅓ at masses
  1, 2, 3). So it is not a property of the potential
  (`scripts/relation4_compare.py`).
* **What it needs (`scripts/relation_origin.py`, `logs/relation_origin.log`).**
  A rank of 18 is certified (it reaches the free maximum, so the relation
  is absent). A rank of 17 is a lower bound (the relation is present with
  high probability).

  | variant of H_ij = T_i + T_j + V_ij | rank | relation |
  |---|---|---|
  | V = 1/(x_i−x_j); + (x_i−x_j)³; V = 1/(x_i−2x_j) (no translation invariance); + x_i² trap; + x_i x_j² | 17 | holds |
  | curved local kinetic T_k = (1+x_k²)p_k²/2; T_k = p_k²/2 + a_k p_k | 17 | holds |
  | three random flat quadratic kinetic forms + random potentials (no body structure) | 18 | **gone** |
  | V_12 also depends on body 3 (non-local) | 18 | **gone** |
  | potential containing momentum; kinetic term of body j with a coefficient depending on body i | 18 | **gone** |
  | T_k with p³ or p⁴ terms (non-quadratic in momentum) | 18 | **gone** |

* **Reading (EVIDENCE, not proof).** The relation is present exactly when
  three conditions hold:
  1. **Locality.** H_ij involves only bodies i and j, and body i's own term
     T_i is the same in every Hamiltonian containing i.
  2. **Kinetic terms are quadratic in momentum.** They may be curved or
     shifted.
  3. **Potentials are momentum-free.** They do not need translation
     invariance (so it is *not* Newton's third law, which was tested and
     ruled out).

  Conditions 2 and 3 are exactly what makes {V_a, {V_b, T}} a function of
  position only, so that {V_c, {V_a, {V_b, T}}} = 0. This is the classical
  identity behind force-gradient symplectic integrators. The conjecture is
  that the relation is this identity, seen through the free Lie algebra
  and filtered by locality. Not yet derived by hand.
* **Not yet checked:** planar; N ≥ 4 (does the same mechanism explain
  new_L1 = N(N−2), which is smaller than the number of pairs of edges
  sharing a body?); an explicit hand derivation.

---

## Proposed actions (for decision)

| # | action | depends on |
|---|---|---|
| 1 | Exact ℚ upper bound for 1D d(4), per block, on jaga | nothing |
| 2 | Same for planar d(4); if equal to 5,914, submit a(4) | 1 (method) |
| 3 | Second prime/seed run of planar `d4.py` | nothing |
| 4 | Editorial fixes A1–A5 in the Zenodo paper; A6 in paper 2; A7 in the literature note | your acceptance |
| 5 | Identify the length-4 relation for generic masses and in 2D | nothing |

## Reproducing

`pip install python-flint sympy numpy`, then from the repo root:

```
python docs/pending_review/2026-09/scripts/formal_vs_physical.py 3 2 3 1/r 300   # B1 (planar, ~17 min)
python docs/pending_review/2026-09/scripts/free_lie_depth.py 3 3                 # A7/B2
python docs/pending_review/2026-09/scripts/relation4.py 1                        # B2
python docs/pending_review/2026-09/scripts/exp_exact.py 1,2,3                    # B3 (exact over Q, seconds)
python docs/pending_review/2026-09/scripts/d4.py 1 2500 17                       # B4 1D (~1 min)
python docs/pending_review/2026-09/scripts/d4.py 2 2500 17                       # B4 planar (~30 min)
python docs/pending_review/2026-09/scripts/graded_fast.py 8 1000 all             # B6 (~8 min)
```
