# Pending review (September–November 2026)

**Nothing in this folder has been accepted into the project.** It is an
external evaluation plus follow-up computations, kept apart from the
papers, docs, and results so that nothing is mistaken for a settled claim.

| file | what it is |
|---|---|
| `REVIEW_PENDING.md` | The review itself. Part A: what needs fixing in existing material. Part B: new findings (B1–B9). Every item labelled PROVEN / LOWER BOUND / EVIDENCE / PREVIOUSLY KNOWN. |
| `KNOWLEDGE_LEDGER.md` | One-page ledger of every claim the project makes, with status and evidence path. §9 is the "tested and moved on from" list. |
| `literature_check/REPORT.md` | Deep-research pass (2026-11-08): is the per-length sequence already known? Answer: no; closest work is McLachlan & Murua 2019. Note files in `literature_check/notes/`. |
| `scripts/` | Exact (mod p / over ℚ) verification scripts for every number quoted. |
| `logs/` | Their outputs. |

Scripts need `pip install python-flint sympy numpy` and run from the repo root.

Reconciled with `main` on 2026-11-08 (merge of seven post-July commits,
including the author's own a(4) = 5,914 jet-evaluation result and the
hidden-zero fix). See the reconciliation notes at the top of
`REVIEW_PENDING.md` and `KNOWLEDGE_LEDGER.md`.

History: an earlier version was committed as `review_2026_09/` and struck
(reverted) for mixing proven and unproven claims; this folder replaces it.
