# Related and citable literature (deep-research pass, 2026-07-19)

Product of an adversarially-verified web research pass (every item below
survived 3-0 verification against primary sources; refuted claims noted at
the end). Two goals: independent mentions of this project's work, and
adjacent citable literature.

## Independent mentions: none exist yet (and why)

- No third-party mention, citation, or discussion of A395423, the Zenodo
  preprint (either DOI), the GitHub repo, nbody.briansheppard.com, or the
  HF dataset was found anywhere.
- Mechanism for the OEIS part (verified): the approved public page
  oeis.org/A395423 is still an allocation stub with no content, and the
  full draft lives under oeis.org/draft/, which OEIS robots.txt disallows
  for crawlers. Until approval, the entry is invisible to search engines
  by construction. Re-run the search after approval.
- The only third-party engagement so far is OEIS editorial review itself
  (Irvine, Marcus, Yuen, Spezia, Zabolotskii; one substantive mathematical
  comment: Zabolotskii's kinetic-terms observation).
- The non-OEIS negatives (Zenodo/GitHub/HF) are "nothing found," not
  verified negatives.

## Apparent novelty

- OEIS search for the consecutive terms 3, 6, 17, 116 returns zero
  results; no prior work computing an increasing pairwise
  bracket-generated filtration L_{n+1} = L_n + {L_n, L_n} for mechanical
  Hamiltonians was found (absence not adversarially proven).

## Citable shortlist (verifier caveats baked in; keep them in citation text)

1. **Dullin, "The Lie-Poisson structure of the reduced n-body problem,"
   Nonlinearity 26 (2013) 1565-1579, arXiv:1207.5883.** Strongest single
   citation. Lemma 1: the symplectic Poisson bracket on quadratic forms
   closes to sp(2m); Theorem 4: Galilean reduction of the n-body problem
   gives sp(2n-2) for any d (n=3: sp(4)). This is the standard-literature
   mechanism behind the harmonic (r^2) closure at dim 15. CAVEAT: his
   sp(4) is a kinematic algebra of quadratic invariants (any potential),
   not an iterated-bracket filtration; homogeneous quadratics only (r^1
   needs the affine extension sp(2m) semidirect h_m). Preprint bib: yes.
   OEIS LINKS: plausible.
2. **Carinena, Falceto, Grabowski, Ranada, "Geometry of Lie integrability
   by quadratures," J. Phys. A 48 (2015) 215206, arXiv:1409.7549** and
   **Maciejewski, Przybylska, Combot, "Non-integrability of the n-body
   problem," arXiv:2502.01426 (accepted JEMS).** The closest published tie
   between bracket filtrations and integrability: MPC Sec. 2 builds an
   iterated-bracket filtration of vector fields and invokes Carinena et
   al. Theorem 9 (filtration Abelian at level n => integrable by n+1
   quadratures). CAVEAT: their series is DECREASING (derived-series-like)
   where ours is an INCREASING generation filtration; say so explicitly
   when citing. MPC Theorem 1.1 is also the state-of-the-art n-body
   non-integrability result (planar n-body, arbitrary positive masses,
   (h,c) != (0,0), Morales-Ramis with solvability weakening).
3. **OEIS A027376** (Witt formula, free Lie algebra on 3 generators): the
   universal upper bound for any 3-generated Lie algebra. Against the
   doubling filtration the free bounds are 3, 6, 32, 1318 (partial sums
   through word length 2^(n-1)) vs observed 3, 6, 17, 116: level 1
   saturates the free bound, levels 2-3 fall strictly below. Belongs in
   A395423 CROSSREFS (Cf.), not LINKS. REFUTED as framed: "116 appears in
   A027376 at degree 6" - do NOT cite that coincidence.
4. **Munthe-Kaas, Owren, "Computations in a free Lie algebra," Phil.
   Trans. R. Soc. A 357 (1999) 957-981.** Cite ONLY for the graded Witt
   formula (the claim that it is a foundational bracket-closure
   computation reference was refuted 1-2).
5. **Kuznetsov, Phys. Lett. A 218 (1996) 212-222, arXiv:solv-int/9509001;
   Jonke, Meljanac, Phys. Lett. B 511 (2001) 276-284, hep-th/0105043;
   Carrillo-Morales, Correa, Lechtenfeld, JHEP 05 (2021) 163,
   arXiv:2101.07274.** Calogero-Moser symmetry/invariant algebras close
   finitely but NONLINEARLY (quadratic/polynomial) - the published
   contrast to open-ended linear growth for the same 1/r^2 pair
   potential. CAVEAT: all three are quantum and 1D N-particle (one with a
   confining trap); generating sets differ from pairwise Hamiltonians.
   Preprint bib: yes. OEIS: no.
6. **Miller, Post, Winternitz, J. Phys. A 46 (2013) 423001,
   arXiv:1309.2694.** Standard superintegrability review: symmetry
   algebras "close polynomially or rationally." CAVEAT: never discusses
   non-closure or growth - do not attribute our dichotomy framing to it.
7. **Low, J. Math. Phys. 55, 022105 (2014), arXiv:1207.6787.** The
   centrally extended inhomogeneous symplectic group sp(2n,R) semidirect
   h_n as MAXIMAL quantum kinematical symmetry; n=2 gives exactly
   Sp(4,R)xH_2, dim 15 - an independent characterization of the harmonic
   closure algebra as a maximal object. CAVEAT: quantum-representation-
   theoretic maximality; pair with a classical reference (e.g. Folland)
   for the degree-<=2 polynomial Poisson statement.
8. **Latini, Marquette, Zhang, Ann. Phys. 426 (2021) 168397,
   arXiv:2010.12822.** Higher-rank Racah algebra R(n) embedded in
   quadratic symmetry algebras (classical and quantum). Preprint bib
   (B2); OEIS: no.
9. **Tsygvintsev, Crelle 537 (2001) 127-149.** Canonical planar
   three-body meromorphic non-integrability (Theorem 6.3, no mass
   restriction). CAVEAT: cite alongside Combot (arXiv:1209.4747) re
   algebraic-potential precision; the mass-exception values attach to his
   later single-integral papers, not Thm 6.3.

## Open questions from the pass

- Third-party mentions of the non-OEIS artifacts: no verified claims
  either way (re-check Zenodo citation trackers, Scholar, HF stats).
- Any literature analog for the r^3 exceptional exponent: nothing
  surfaced (Dullin/Low cover r^1/r^2 quadratic closure only).
- Re-run the Goal-A mention search after A395423 approval.
