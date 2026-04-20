# Supplemental Folder Triage

*Created: April 18, 2026*

Classification of all items in `F:\science-projects\3body-supplemental\` against
the current state of the main project. Status legend:

- **DONE** — implemented and matches main; supplemental file can stay as historical record.
- **PARTIAL** — partially implemented; specific gaps remain.
- **PENDING** — actionable, not started; see "Action plan" below.
- **REFERENCE** — notes / essay / external paper; no implementation expected.
- **NOT WORTH** — superseded or out of current scope.

---

## Table

| File | Status | Notes / Where it lives in main |
|------|--------|-------------------------------|
| `bracket_tree_field_guide.md` | REFERENCE | Catalog/taxonomy. Systems tracked in [`nbody/expansion_configs.py`](../nbody/expansion_configs.py) and §2 of [`docs/project_status.md`](project_status.md). Treat as living taxonomy doc. |
| `expansion.txt` | DONE | Brainstorm list; every entry now in `expansion_configs.py` SCENARIOS or §2 Physical Systems Catalog of `project_status.md`. |
| `Molecular three-body systems. Triat.txt` | DONE | H₃⁺ and O₃ in [`nbody/named_molecular_systems.py`](../nbody/named_molecular_systems.py); log-vortex confirmed (`primes/run_gue_logas.py`). |
| `Now I have a comprehensive picture..txt` | DONE (snapshot) | Status ledger, superseded by [`docs/project_status.md`](project_status.md). |
| `EXTRACT_ALGEBRA_STRUCTURE.md` | MOSTLY DONE | Structure constants, Killing form, derived/lower central series, isomorphism done for 15 potentials at L2 — see [`nbody/isomorphism_test.py`](../nbody/isomorphism_test.py), `results/algebra_structure/`. Gap: U/Vt full SVD save + inter-level Gram projection (minor). |
| `incognito_review_insights (1).md` | REFERENCE | Two-layer framing + helium suggestions absorbed into papers and `docs/conjectures.md`. |
| `Research_in_the_Age_of_AI.md` | REFERENCE | Methodology essay; no code action. |
| `indra_seed (1).md` | REFERENCE / NOT WORTH (yet) | Productization vision. Premature; revisit after Paper 3 ships. |
| `pairwise_hamiltonian_landscape.md` | REFERENCE | Top target (GUE log-gas) DONE in `primes/`. Remaining suggestions live in bracket tree guide. |
| `poisson_numerical_robustness_memo.md` | DONE | Symbolic rank over ℚ supersedes — see [`gap_workplan.md`](gap_workplan.md) §4.4 and `symbolic_rank.py`. |
| `poisson_neural_network_research_brief.md` | DONE | Implemented in [`neural/`](../neural/); shipped in HF dataset (`neural_algebras` table). |
| `arXiv-1712.06698v3` | REFERENCE | Galperin billiards / π paper (Aretxabaleta et al.); not your work. Reading material. |
| `Riemann_zeta_function_absolute_value.png` | REFERENCE | Visual anchor for GUE/zeta narrative in `primes/`. |
| `I now have everything I need. Let m.txt` | PARTIAL | L2 broad sweep (500 exponents) + L3 extended sweep (76) DONE; full 1015-exponent L3 plan aborted on cost (~$1,500). See [`gap_workplan.md`](gap_workplan.md) §3.1. |
| `schwarzschild_scope.md` | PENDING (P1) | Composite supported (`nbody/run_composite_test.py`); Schwarzschild config + atlas not built. |
| `MAP_STRUCTURE_ACROSS_PARAMETER_SPACE.md` | PENDING (P2) | Depends on EXTRACT (now mostly done). New scripts needed. |
| `nbody-atlas-mcp-suggestions.md` | PENDING (P3) | No MCP server yet. |
| `ALCUBIERRE_MARCH30_INTEGRATION.md` | PENDING (P4, low ROI) | Pure physics side-quest, no clear hook to Poisson algebra pipeline. |

---

## Action plan (priority-ordered)

### P1 — Schwarzschild / composite atlas extension
**Source:** `schwarzschild_scope.md`

Composite potentials are already supported through `nbody/run_composite_test.py`
and the PN entries of `expansion_configs.py`. What's missing:

- A registry/scenario entry for `V_eff = -M·u + (L²/2)·u² - M·L²·u³`.
- A 2D atlas scan over `(r/M, L/M)` (or equivalent shape coords) looking for
  algebraic signatures at the event horizon (r = 2M), photon sphere (r = 3M),
  and ISCO (r = 6M).
- Comparison panel against pure 1/r at the same grid.

**Effort:** ~1 day setup + ~$10–30 AWS atlas spend.
**ROI:** HIGH — first GR-flavored result in the atlas; natural follow-up to the
GUE/prime story; plausible Paper 4/5 figure.

### P2 — Structure-constant atlas across (μ, φ)
**Source:** `MAP_STRUCTURE_ACROSS_PARAMETER_SPACE.md`

The L2 structure-constant pipeline already exists (`nbody/isomorphism_test.py`).
Extend it to compute the 32 non-zero structure constants (and Killing eigenvalues
+ center dim) at every grid point of the shape sphere and render as sphere maps.

Add monodromy: parallel-transport the basis around a small loop encircling a
binary collision, check the resulting permutation on basis indices.

**New scripts:**
- `atlas_structure_sweep.py` — main sweep.
- `compute_monodromy.py` — collision loop transport.
- `compare_atlas_structures.py` — cross-potential comparison.

**Effort:** 2–3 days; AWS optional (most can run locally at L2).
**ROI:** HIGH — turns universality from "same dim everywhere" into "same tensor
everywhere" — the strongest possible isomorphism evidence. Paper 3 figure.

### P3 — N-body atlas MCP server
**Source:** `nbody-atlas-mcp-suggestions.md`

Build a small FastMCP (stdio) server exposing read-only tools over the local
`aws_results/` `.npy` files: `list_configurations`, `get_rank`, `get_sv_spectrum`,
`get_gap_ratio`, `get_heatmap_slice`, `compare_configs`. HTTP transport later.

Pairs naturally with the experiment registry (Part B): one extra tool
`list_experiments(category=, status=)` reads the same YAML.

**Effort:** 1–2 days.
**ROI:** MEDIUM-HIGH — agents can interrogate the atlas without screenshots.

### P4 — Alcubierre image-method investigation (low priority)
**Source:** `ALCUBIERRE_MARCH30_INTEGRATION.md`

Pure physics side-quest — literature work plus a Casimir analog in linearized
gravity. No clear hook into the Poisson-algebra pipeline. Keep as a parallel
notebook track only if personally interesting.

### P5 — Full L3 parametric 1/r^n sweep (defer)
**Source:** `I now have everything I need. Let m.txt`

Already aborted at projected ~$1,500. The L2 broad sweep + L3 extended sweep
already cover the universality claim. Only revisit if a specific paper figure
demands a continuous fine-resolution `dim(L3) vs n` curve; if so, follow the
recommended cheap path (coarse L2 first, targeted L3 around r¹/r²/r³).

---

## NOT WORTH pursuing now

- **Indra productization** (`indra_seed (1).md`). Premature; the research isn't
  packaged for external users yet. Revisit after Paper 3 ships.
