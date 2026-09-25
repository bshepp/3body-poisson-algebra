# Independent review, September 2026

This is a fresh look at the project: the preprint and papers 2–4, the
A395423 claims, and where the computations stood at the last commit
(2026-07-29). All numbers below were recomputed in this folder with an
independent engine (FLINT, exact arithmetic mod p = 2^31 − 1). They do
not depend on the repo's float64 SVD pipeline.

## 0. Overlap with work already in the repo (added after checking)

The first pass of this review did not search the repo for prior work
before testing. That check has now been done:

| item in this review | already in the repo? |
|---|---|
| Ranks > 116 are phantom (§2.4) | **Yes.** `collision_syzygy/COLLISION_SYZYGY_REPORT.md` and the README already identify the >116 ranks as a numerical-rank bug. Paper 2's §"Deep and soft syzygies" (116 → 124 "wake up") was never updated to match, so the fix belongs in paper 2, not in new analysis. |
| The algebra cannot detect integrability (§2.5) | **Yes.** Concluded in Feb 2026 (`docs/peer_review_analysis.md`, `adversarial_analysis.md`); paper 4 already states it. Retracted as a "needs fixing" item. |
| Formal (u free) vs true-function rank (§1) | **Raised, not tested exactly.** `peer_review_analysis.md` §5 flags it; the only evidence was float SVD at physical points. The exact mod-p check here is new. |
| Free Lie algebra comparison (§2.2) | **Partly.** `related_literature_2026-07-19.md` cites A027376, but its free bound (3, 6, 32, 1318) is wrong; the correct depth-filtration bound is 3, 6, 17, 119. |
| Finite GK dimension (§2.1) | **No.** The repo conjectures infinite GK dimension throughout. |
| Pure exponential potential | **No.** Yukawa was run, but via a 4-term Taylor truncation in u (`docs/project_status.md`), i.e. as a composite power potential, not as e^{−μr}/r. |
| Exact d(4), 1D vs planar (§2.3) | **No.** Only float bounds (≥ 5,625 planar); no 1D level 4. |

### Which items apply to the Zenodo paper (`preprint.tex` = `3body_poisson_algebra.pdf`)

* Title/abstract "super-exponential", the Θ(n²) remark, and the infinite-GK
  framing (§2.1): **yes.**
* "Unchanged by the spatial dimension" (Discussion and Future directions
  item 4) (§2.3): **yes.**
* The d(4) ≥ 5,604 float bound: superseded by the exact lower bound 5,914.
* Yukawa listed as confirming universality: it was the truncated version.
* The stability-atlas paragraph ("rank drops at the Lagrange and Euler
  configurations ... encoding stability boundaries"): **yes**, those are
  numerical, as in §2.4.
* Paper 2 syzygies, primes, Bell, cosmology: **not in the Zenodo paper.**
* "Proved exactly for levels 0–3": now actually certified (§1).

## 1. What holds up

**a(0..3) = 3, 6, 17, 116 is correct, and is now certified for the actual
functions on phase space.** The repo's exact rank treats u_ij = 1/r_ij as
a free symbol. That gives an *upper* bound on the dimension of the span of
the real functions: a formal identity is also a functional identity, but
not necessarily the other way round. The float64 SVD supplied the matching
lower bound, but a float SVD is not a proof. `formal_vs_physical.py`
evaluates every generator exactly (mod p) at points that satisfy the
physical constraint: u = 1/(x_i − x_j) in 1D, and u² r² = 1 in the plane
(taking a square root mod p). A rank of an exact evaluation matrix over
GF(p) is a certified lower bound. Results:

| | formal (u free) | physical, 1D | physical, planar |
|---|---|---|---|
| d(0..3) | 3, 6, 17, 116 | 3, 6, 17, 116 | 3, 6, 17, 116 |

Lower bound = upper bound, so the four published terms are established.

Also solid:
* Mass invariance for generic masses (the rank over Q(m1,m2,m3) argument
  is the right argument).
* The move to mod-p methods in July. L3(5) = 5,965 was cross-checked by
  two primes and exact elimination, which is exactly how this should be
  done.
* L1(N) = N(3N−5)/2 and L2(N) = N(4N²−9N+3)/2 (N ≥ 4) with exact
  confirmations through N = 10, including one registered out-of-sample
  prediction (N = 10).
* The audit culture. The repo repeatedly caught and corrected its own
  overclaims.

## 2. What is wrong or needs reframing

### 2.1 "Super-exponential growth" and "infinite GK dimension" are false asymptotically

The filtration A_n = A_{n−1} + {A_{n−1}, A_{n−1}} doubles the maximum
word length at each level: A_n contains Lie words of length up to 2^n.
Any algebra with *polynomial* growth in word length, dim ~ L^k, therefore
already grows like 2^{kn}, which is exponential in n. Four data points
with rising ratios say nothing about the asymptotics.

Worse, the GK dimension is provably **finite**:

* Give u total degree 1. A bracket raises total degree by at most 2:
  D_x can raise it by 3 through ∂u/∂x = −(x_i−x_j)u³, and ∂_p lowers it
  by 1. So a Lie word of length L is a polynomial of degree ≤ 4L − 2.
* Every element therefore lies in the space of polynomial functions of
  degree ≤ 4L on phase space, which has dimension O(L^k), where k is the
  dimension of the space the functions really live on. Every H_ij is
  translation-invariant, so everything is a polynomial in the coordinate
  differences, the momenta and the u_ij, subject to u² r² = 1. That gives
  k ≤ 5 in 1D and k ≤ 10 in the plane (9 if you also use rotation
  invariance).
* Hence GKdim ≤ 5 (1D) and ≤ 10 (planar). In depth-filtration terms,
  d(n+1)/d(n) is eventually at most about 2^5 (1D) or 2^10 (planar).

The growth really is fast through the levels computed; the bound is far
from binding there. But the title, the abstract, the Θ(n²) remark and the
GK conjecture should all go.

### 2.2 d(0), d(1), d(2) carry no physics; the only nontrivial content through level 3 is one relation

`free_lie_depth.py` computes the same filtration for the **free** Lie
algebra on 3 generators: **3, 6, 17, 119**. So:

* 17 is *not* "genuinely irregular"; it is the free count 3 + 3 + 8 + 3.
  (The free bound "3, 6, 32, 1318" in
  `docs/related_literature_2026-07-19.md` is wrong. It sums all Lie words
  of length ≤ 2^n, but the depth filtration reaches only some of them.)
* Split by word length (the 1/r algebra is graded by word length, since
  the weights x:2, p:−1, u:−2 make a length-L word homogeneous of weight
  1 − 3L):

  | length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | total |
  |---|---|---|---|---|---|---|---|---|---|
  | free, inside A_3 | 3 | 3 | 8 | 18 | 24 | 36 | 24 | 3 | 119 |
  | physical, inside A_3 | 3 | 3 | 8 | 17 | 23 | 35 | 24 | 3 | 116 |

  There is exactly **one** non-free relation. It sits at word length 4
  (`relation4.py` prints it; it is unique and has small integer
  coefficients), and the other two missing dimensions are its brackets
  at lengths 5 and 6.

This is most of the explanation for the "universality" of 3, 6, 17, 116.
At low depth, almost any pair potential generates a nearly free Lie
algebra, so almost any potential gives the same counts. That includes a
pure exponential e^{−r}, which I checked (`exp_potential.py`). The
exceptions are the potentials that make H_ij a polynomial of degree ≤ 2:
harmonic, and r¹ in 1D. Quadratic polynomials are closed under the
Poisson bracket, so those algebras are finite-dimensional. This is
Dullin's sp(2n) observation, already in the literature notes. Cubic
r³ is the next degenerate case. The universality is real, but it is a
statement about free Lie algebras plus one relation, not a deep physical
invariant.

### 2.3 Spatial-dimension independence breaks at level 4

`d4.py` computes d(4) exactly mod p. It uses the repo's own derivative
pipeline ({f,g} from the first derivatives of level-3 generators), but in
GF(p) instead of float64, and with the rank split into word-length blocks:

* **1D (full phase space): d(4) = 5,042.** Two independent seeds (random
  masses, points, and sample sizes of 2,500 and 2,200) agree exactly. No
  block is near saturation: the largest block rank is 960.
* **Planar: d(4) = 5,914** (seed 17, S = 2,500; largest block rank 1,213,
  so not saturated). This agrees with the repo's float bound (≥ 5,625) and
  is higher than it. It needs a second prime and seed before it goes to
  OEIS.

Status of these numbers: levels 0–3 are built symbolically (exact
polynomials; coefficients reduced mod p). Level-4 brackets are evaluated
exactly from the symbolic first derivatives of the level-3 generators, so
no floating point is involved anywhere. An exact GF(p) rank of an
evaluation matrix is a **certified lower bound**: d(4) ≥ 5,914 (planar)
and ≥ 5,042 (1D) are proven. It equals the true value with high
probability (Schwartz–Zippel), but it is not yet a theorem. The matching
**upper bound** needs the symbolic level-4 brackets: their monomial
coefficient rank (u free) mod p. If that rank also comes out at 5,914,
a(4) is proved. That symbolic expansion is the heavy, memory-bound step,
and it is the right job for the 256 GB server.

The planar float bound in the repo is d(4) ≥ 5,625. So the claim that the
sequence is independent of spatial dimension (paper 3, the cosmology
section, the OEIS comments) fails at level 4. The finite-GK argument says
it had to fail eventually: the 1D and planar algebras live on spaces of
different dimension.

A related subtlety in 1D: if the bodies are confined to one ordering
(x1 > x2 > x3), the word-length-7 piece has dimension 215; over the full
line (all orderings) it has 272, which equals the formal count. The
definition should say which one is meant.

### 2.4 Paper 2: the "soft syzygies" and ranks above 116 are numerical artifacts

The 40 relations among the 156 level-3 candidates are exact polynomial
identities (exact rank over Q). They therefore hold at every point of
phase space. A rank of 124 or 125 "near the collinear submanifold" is
impossible for functions spanning a 116-dimensional space. The Jacobi
identity does not "fail" anywhere. These are float64 conditioning
effects. For the same reason, any "local rank drop" in the atlas is
numerical: real-analytic functions that are linearly independent on phase
space stay independent on every open ball. The ε^α tier scaling is the
Taylor expansion showing through at small ε. The S3 isotypic counts
(24 A + 28 A′ + 52 E) are legitimate and exact.

**Consequence:** the atlas campaign (42 configurations, AWS, most of the
server time) maps SVD conditioning, not algebraic structure. It should
not be presented as a stability or integrability diagnostic.

### 2.5 Paper 4 (Calogero–Moser) — already settled in the repo (see §0); no action

Integrability is about the centralizer of H = ΣH_ij (enough commuting
integrals). The algebra *generated by the pieces* H_ij is a different
object, so it is unsurprising that it cannot see integrability. After
§2.2 this is expected a priori: 1/r² is a generic potential.

### 2.6 Remove from anything public

The primes/GUE "same universality class as gravity" claim, the
cosmological-consequence section (N ~ 10^80), and the Bell/CHSH
investigation. After §2.2–2.3 their premise (a potential- and
dimension-independent invariant) no longer holds, and they cost
credibility with the exact reviewers A395423 needs.

## 3. Where the project stood (2026-07-29)

* A395423 approved and published on OEIS (3, 6, 17, 116; keyword `more`).
* L3(5) = 5,965 triply verified. The a = 1198 scaling law was falsified.
  The zero-degree-of-freedom binomial fit predicts L3(6) = 17,979, which
  has not been run.
* d(4): float bound ≥ 5,625 (corrected 11,937-candidate census on jaga).
  The mpmath high-precision run was stalled at 4.4%. The exact d(4) was
  listed as the path to extending the OEIS entry.
* The atlas, primes, Bell, and neural side projects were parked.

## 4. Recommended next steps

1. **Extend A395423 with a(4) = 5,914.** Rerun `d4.py 2 <S> <seed>` on
   jaga with a second prime and seed (about 30 min per run on one core
   here). Then close the upper bound symbolically: expand the level-4
   brackets block by block as polynomials mod p and take the rank of the
   monomial coefficients. Formal rank = 5,914 makes it a theorem. Add the 1D sequence as a separate
   OEIS entry (3, 6, 17, 116, 5042), because it differs from the planar
   one.
2. **Rewrite paper 1** around what is now provable: certified
   d(0..4) in 1D and in the plane, the word-length grading, the
   comparison with the free Lie algebra and the single quartic relation,
   the finite GK-dimension bound, and the divergence between 1D and 2D at
   level 4. That is a smaller claim than "super-exponential", but true
   and more interesting.
3. **Understand the quartic relation** and whether the whole N=3 algebra
   is "free modulo a few relations" in a range of degrees. The per-length
   dimensions g_L (1D, all orderings: 3, 3, 8, 17, 44, 103, 272) are the
   natural invariant to report and conjecture about.
4. **N-body:** new_L1 = N(N−2) and new_L2 = 12·C(N,3) look provable by
   hand, via shared kinetic terms and commuting disjoint pairs. A proof
   would be a real theorem.
5. Drop the atlas, Bell, GUE and cosmology material from the paper track.

## Files

| file | what it does |
|---|---|
| `formal_vs_physical.py` | d(0..3) with u free vs u = 1/r on real phase space (1D / 2D) |
| `free_lie_depth.py` | same filtration for the free Lie algebra on k generators |
| `graded.py`, `graded2d.py` | word-length-graded dimensions g_L (1D / planar), formal vs physical |
| `relation4.py` | explicit non-free relation at word length 4 |
| `exp_potential.py` | exponential potential e^{−r} (also gives 3, 6, 17, 116) |
| `d4.py` | exact (mod p) d(4) via derivative pipeline + length blocks |
| `*.log` | outputs of the runs quoted above |

Requires `pip install python-flint sympy numpy`.
