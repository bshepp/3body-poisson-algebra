# Academic Engagement Strategy
## Pairwise Poisson Algebras of the N-Body Problem
### Last updated: March 2026

---

## Papers

| # | Title | File |
|---|-------|------|
| 1 | Super-exponential growth of the Poisson algebra (N=3) | `preprint.tex` |
| 2 | S₃-equivariant jet filtration: tiers, scaling, syzygies | `paper2_s3_filtration.tex` |
| 3 | Universal dimension sequences: N=4, d-independence, universality conjecture | `paper3_universality.tex` |
| 4 | The Poisson Algebra of Pairwise Interactions: A Calogero-Moser Integrability Test | `paper4_calogero_integrability.tex` |

## Key Results (for abstract drafting)

These are the established facts. Use only these in formal submissions.

### Paper 1 results
| Result | Status | Evidence |
|--------|--------|----------|
| d(0)=3, d(1)=6, d(2)=17, d(3)=116 | **Proved** | Exact symbolic computation, SVD gap > 10^6 |
| d(4) >= 5,604 | **Lower bound** | Numerical pipeline, 200K samples at generic config; exact rank via mpmath in progress |
| Mass invariance of d(0)-d(3) | **Proved** | Tested 20+ mass ratios including Tsygvintsev exceptional cases |
| Infinite Gelfand-Kirillov dimension | **Proved** | Follows from super-exponential growth through Level 4 |
| Harmonic (r^2) algebra closes at dim 15 | **Proved** | Zero new generators at Level 4, SVD gap > 10^14 |
| Calogero-Moser (1/r^2) matches Newton (1/r) | **Proved** | Identical sequence [3,6,17,116] |

### Paper 2 results
| Result | Status | Evidence |
|--------|--------|----------|
| Four-tier decomposition 52+44+16+4=116 | **Proved** | SVD spectra at multiple configurations |
| S₃ isotypic decomposition 24A + 28A' + 52E | **Proved** | CG analysis, E-fraction = 2/3 at every level |
| Integer-quantized jet filtration (α = 0, 1, 2, 3) | **Proved** | Epsilon scaling analysis |
| Syzygy decomposition: 32 soft + 8 deep | **Proved** | Direct evaluation + noise-floor taxonomy |

### Paper 3 results
| Result | Status | Evidence |
|--------|--------|----------|
| N=4 dimension sequence [6, 14, 62] | **Proved** | SVD gap 3.4×10¹¹ |
| N=4 mass invariance | **Proved** | 3 configs (equal, hierarchical, mixed) |
| d-independence (N=3 and N=4, d=1,2,3) | **Proved** | All gap ratios > 10¹⁰ |
| Pole-order invariance (1/r, 1/r², 1/r³) | **Proved** | Identical [3,6,17,116] |
| Charge-sign invariance | **Proved** | All-attractive, helium (+2,−1,−1), all-repulsive |
| Local rank drops at symmetric configs | **Observed** | d(3)=110 at Lagrange, 106 at Euler collinear |
| S₃ symmetry in gap ratio landscape | **Observed** | Isosceles curves visible as valleys on shape sphere |
| Shape sphere atlas: 11 configs surveyed | **Complete** | 99,000 grid points; rank 116 at 75–93% across all potentials/charges |
| Cross-potential atlas consistency | **Confirmed** | Critical locus identical topology for 1/r, 1/r², 1/r³, log(r) |
| Charge sensitivity atlas (7 configs) | **Confirmed** | Rank differences sparse (±1–4 at boundaries); gap score more continuous |
| Full 100×100 atlas campaign (19 configs) | **In progress** | Tiers 1-3 at 27–30%; Tier 4 (Yukawa) terminated (subs() stall); lambdify fix applied |
| N=2 Poisson algebra is trivial (dim=1) | **Proved** | bell_test.py Part A: dimension 1 through Level 5 |
| Bell test (CHSH from algebra projections) | **No violation** | max |S| = 1.77 (tidal generators, equilateral); classical bound holds; full-scale run in progress |

### Paper 4 results
| Result | Status | Evidence |
|--------|--------|----------|
| 1D Calogero-Moser (integrable) gives [3,6,17,116] | **Proved** | N=3, d=1, 1/r² via exact_growth_nbody.py; gap ratio 6.85×10⁷ |
| Galperin superintegrable mass ratios invariant | **Proved** | q=3,4,5,6 and generic (1,2.7,0.4) all give d(3)=116 |
| Singularity dichotomy | **Proved** | All singular potentials → [3,6,17,116]; harmonic → [3,6,13,15,15] |
| Dimension sequence is singularity class invariant | **Proved** | Integrable CM matches non-integrable gravity |
| Integrability invisible to pairwise algebra | **Proved** | CM integrals (L², Chevalley J) are non-pairwise |

**Do not claim:**
- That super-exponential growth is a "non-integrability certificate" (disproved by CM comparison)
- That gap ratio minima correspond to "stable orbits" (unproved)
- That gap ratio valleys trace KAM tori (speculative)
- That the gradient of the gap map has physical meaning (unproved)

---

## Conference Targets

### Tier 1: High relevance, actionable deadlines

| Venue | Dates | Location | Deadline | Framing |
|-------|-------|----------|----------|---------|
| GAP XX (RIMS) | Apr 13-24, 2026 | Kyoto | Register now (free) | Moduli spaces, Poisson geometry on shape space |
| Gravity2026 (IBS) | Apr 27 - May 1, 2026 | Daejeon | Abstract: Mar 29 | Algebraic obstructions in gravitational N-body systems |

### Tier 2: Strong fit, later deadlines

| Venue | Dates | Location | Deadline | Framing |
|-------|-------|----------|----------|---------|
| Monopole Moduli (ICTS-TIFR) | Sep 14-25, 2026 | Bengaluru | TBD (~June) | Poisson geometry on configuration moduli space |
| ICMMP 2026 | Oct-Nov 2026 | Hangzhou | TBD (~mid-2026) | Exact algebraic results + novel integer sequence |
| Strings 2026 (SIMIS) | Jul 2026 | Shanghai | TBD | Infinite-dimensional algebras from classical mechanics |

### Tier 3: Networking / seminar opportunities

| Venue | Notes |
|-------|-------|
| Kavli IPMU (Tokyo) | Short trip from Kyoto during GAP XX |
| NCTS (Taipei) | Contact Hsuan-Yi Liao (GAP XX organizer) |
| IBS Center for Geometry and Physics (Pohang) | Yong-Geun Oh is GAP XX speaker |
| IISER Pune / Kolkata | Arrange seminars if visiting ICTS |
| Bangkok Workshop (annual, ~Jan 2027) | Informal; SVD methods connect to random matrix theory |

---

## Audience-Specific Framing

### Symplectic geometers (GAP XX, RIMS, Kavli IPMU)

**Angle:** The shape space of three-body configurations is a 2-sphere with S3 action. The Poisson algebra generated by pairwise Hamiltonians defines a rank function on this moduli space. We compute this rank exactly and show its critical locus coincides with the symmetric configurations (Lagrange, Euler, isosceles).

**Key terms:** moduli space, Poisson geometry, S3 action, rank stratification, shape sphere.

**What they will ask:** Is the rank function semicontinuous? (Yes -- it can only drop at special points.) Can you describe the ideal of relations at Lagrange? (Open problem.) What is the Poisson center? (Unknown, but finite-dimensionality of the harmonic case constrains it.)

**New atlas evidence (Mar 2026):** The shape sphere atlas survey across 11 potential/charge configurations (99,000 grid points) confirms the critical locus tracks S₃ fixed points exclusively. Triptych comparison figures show this structure is universal across potentials.  Figures in `atlas_figures/triptychs/`.

### Gravitational dynamics (Gravity2026, IBS-CTPU)

**Angle:** A purely algebraic invariant of the N-body problem.  The pairwise Poisson algebra's dimension sequence classifies interactions by singularity type (singular vs regular), not integrability status.  No orbit integration required.  Now extended to N=4, with the universality conjecture predicting the sequence depends only on N.

**Key terms:** Poisson bracket, SVD rank estimation, three-body problem, four-body problem, stability atlas, universality.

**What they will ask:** Does this scale to N>3? (Yes — computed for N=4, computationally feasible for N=5.) Can you detect individual stable orbits? (Not directly — the method measures algebraic constraint strength, not specific trajectories.) Is this a non-integrability certificate? (No — the integrable Calogero-Moser system gives the same sequence.)

### Mathematical physics (ICTS, ICMMP, Strings)

**Angle:** The dimension sequence [3, 6, 17, 116, >=4501] defines a new integer sequence not in OEIS. Its mass invariance (a theorem) and potential-type dependence (singular vs. regular) connect to differential Galois theory and Morales-Ramis non-integrability.

**Key terms:** Gelfand-Kirillov dimension, differential Galois group, Morales-Ramis theory, mass invariance, super-exponential growth.

**What they will ask:** What is the exact d(4)? (In progress -- mpmath computation running on AWS with 50-digit precision.) Is the sequence eventually periodic? (Almost certainly not -- growth is super-exponential.) Connection to quantum groups? (Speculative but intriguing.) What about Bell correlations? (Tested -- CHSH computation from algebra projections onto single-body observables in progress.)

---

## Draft Abstracts

### General-purpose (trilogy overview, suitable for most venues)

> **Universal dimension sequences of pairwise Poisson algebras in the N-body problem**
>
> We study the Poisson algebra generated by pairwise interaction Hamiltonians of the N-body problem under iterated brackets.  For N=3 (three bodies), the cumulative dimension sequence is [3, 6, 17, 116] with d(4)≥5,604, implying infinite Gelfand-Kirillov dimension.  For N=4, the sequence [6, 14, 62] is computed for the first time.  Both sequences are mass-invariant and independent of the spatial dimension d (verified at d=1,2,3).  For N=3, the sequence is also independent of the pole order of the potential (1/r, 1/r², 1/r³ all identical) and invariant under sign-flip of couplings (attractive, repulsive, or mixed Coulomb).  The harmonic potential (r²) produces a finite algebra of dimension 15.  The internal structure of the 116-dimensional N=3 algebra is explained by S₃ representation theory: a four-tier decomposition 52+44+16+4=116 with integer-quantized jet filtration scaling exponents.  A comprehensive shape sphere atlas survey — 99,000 grid points across 11 potential/charge configurations — confirms rank 116 at 75–93% of shape space, with the critical locus tracking the S₃ fixed-point set exclusively.  These results motivate the Universality Conjecture: the dimension sequence depends only on N and the singularity class (singular vs regular), and nothing else.

### For Poisson geometry / moduli space venues

> **The Poisson algebra on the shape sphere of the three-body problem**
>
> The shape space of planar triangles, modulo translation, rotation, and scale, is a 2-sphere carrying a natural S₃ action.  We study the pairwise Poisson algebra generated by gravitational Hamiltonians.  The rank is generically 116 but drops to 110 at the Lagrange equilateral configuration and 106 at Euler collinear configurations.  The critical locus traces the three isosceles curves—the symmetry walls of S₃.  A comprehensive atlas survey of 99,000 grid points across 11 potential/charge configurations — including 1/r, 1/r², 1/r³, log(r), and 7 Coulomb charge classes — confirms this critical locus structure is universal: no rank anomalies appear at non-symmetric configurations.  The internal algebra possesses a four-tier decomposition 52+44+16+4=116 explained exactly by the S₃ Clebsch-Gordan rules, with integer-quantized epsilon scaling exponents corresponding to a jet filtration.  The algebra and its tier structure are universal: the same dimension sequence arises for all singular central potentials, all spatial dimensions (d=1,2,3), and all charge configurations (attractive, repulsive, or mixed).

---

## Publishing Outlets

| Journal | Scope fit | Impact | Notes |
|---------|-----------|--------|-------|
| Communications in Mathematical Physics | Excellent | High | Natural home; Poisson algebras + celestial mechanics |
| Journal of Mathematical Physics | Excellent | Medium-high | AIP; broad mathematical physics |
| Celestial Mechanics and Dynamical Astronomy | Excellent | Medium | Springer; domain-specific |
| Asian Journal of Mathematics | Good | Medium (Q3) | International Press; English; ~12 week review |
| Regular and Chaotic Dynamics | Good | Medium | Strong integrability tradition |
| Asia Pacific Journal of Mathematics | Adequate | Lower (Q4) | No mandatory APC; open access |

**Recommendation:** Target CMP or JMP for Paper 1 as the flagship.  Submit Papers 2 and 3 as companion pieces to the same or related journals (e.g., JMP for Paper 2, Letters in Mathematical Physics for Paper 3).  Alternatively, submit all three as a single arXiv package with sequential numbering.  Use conference proceedings for shorter overview versions.

---

## Seminar Tour (if attending GAP XX)

| Stop | Institution | When | Contact route |
|------|-------------|------|---------------|
| 1 | RIMS, Kyoto | During GAP XX (Apr 13-24) | Speak with organizers |
| 2 | Kavli IPMU, Tokyo | After GAP XX | Email citing GAP XX participation |
| 3 | IBS, Daejeon | Late April | Attend Gravity2026, request seminar |
| 4 | NCTS, Taipei | Early May | Contact Hsuan-Yi Liao |
| 5 | ICTS, Bengaluru | September | Apply for Monopole Moduli program |

---

## Action Items

### Immediate (next 30 days)
- [ ] Submit all three papers to arXiv (establish priority for trilogy)
- [ ] Register for GAP XX (free, open now)
- [ ] Submit abstract to Gravity2026 (deadline March 29) — use trilogy overview abstract
- [ ] Prepare 30-minute seminar talk covering all three papers

### Mid-term (March-June 2026)
- [ ] Apply for ICTS Monopole Moduli program when applications open
- [ ] Watch for ICMMP 2026 call for papers
- [ ] Compute N=5 Level 1-2 (Paper 3 prediction #1) — strengthens trilogy
- [ ] Submit to journal after incorporating conference feedback

### Long-term (2026-2027)
- [ ] Submit dimension sequences to OEIS ([3,6,17,116] and [6,14,62])
- [ ] Submit Paper 1 to CMP or JMP; Papers 2-3 as companion pieces
- [ ] Monitor Bangkok Workshop 2027 for next edition

---

## Researchers to Connect With

See `outreach_emails.md` for draft emails to European collaborators (Combot, Tsygvintsev, Maciejewski, Przybylska, Morales-Ruiz).

**At Asian venues:**
- Kenji Fukaya (GAP XX mini-course) -- Floer homology, symplectic geometry
- Hiraku Nakajima (GAP XX mini-course) -- geometric representation theory
- Yong-Geun Oh (IBS, GAP XX speaker) -- contact geometry, Hamiltonian dynamics
- Adeel Khan (Academia Sinica, GAP XX speaker) -- potential Taiwan connection
- Ashoke Sen (ICTS Monopole Moduli organizer) -- theoretical physics
- Hsuan-Yi Liao (GAP XX organizer, NTHU Taiwan) -- gateway to NCTS network
