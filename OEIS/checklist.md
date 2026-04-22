# Pre-submission checklist (per OEIS Style Sheet)

Use this checklist before clicking **Save** on each new OEIS submission.
Source: <https://oeis.org/wiki/Style_Sheet>.

## Content correctness

- [ ] Every term has been independently verified (cross-CAS where possible)
- [ ] At least one program in the entry reproduces all listed terms when run
- [ ] No term is conjectural unless explicitly flagged in Comments/Extensions
- [ ] Formula and recurrence are proved (or marked "Conjecture:" in front)
- [ ] Cross-CAS confirmation noted in Comments where applicable

## Naming and writing

- [ ] Name is one line, descriptive, uses `a(n)` notation, no vanity
- [ ] All math text is ASCII (no Unicode `≤`, `≥`, `→`, `Pi`, `Σ`, `…`)
- [ ] `Sum_{k=a..b}`, `Product_{k=a..b}`, capital S/P
- [ ] `*` for multiplication, `^` for exponent (not `**`, `²`, `·`, `×`)
- [ ] `mod` not `%`
- [ ] US spelling (behavior, color, neighbor, generalize, labeled)
- [ ] No "AI", "LLM", "ChatGPT", "Claude" in author or content (allowed
      in narrative if relevant, but never as author)

## Field discipline

- [ ] **Name** (`%N`) — one line, descriptive
- [ ] **Data** (`%S`) — at least 4 terms, fits ~260 chars
- [ ] **Offset** (`%O`) — index of first term
- [ ] **Comments** (`%C`) — context, motivation, invariances, links to
      papers explained briefly
- [ ] **References** (`%D`) — books/journal articles (not online-only)
- [ ] **Links** (`%H`) — stable URLs (GitHub commit hash preferred over
      `main`), papers on arXiv, b-file if applicable
- [ ] **Formula** (`%F`) — closed forms, recurrences, generating
      functions, asymptotics; only proved formulas (otherwise prefix
      "Conjecture:")
- [ ] **Example** (`%e`) — show your work for at least one term
- [ ] **Mathematica** (`%t`) — self-contained, runnable
- [ ] **Programs** (`%o`) — Python preferred for reproducibility, or
      PARI / Sage / Magma; signed with `# ~~~~`
- [ ] **Cross-references** (`%Y`) — `Cf. A000000, A000000` for related
      sequences
- [ ] **Keywords** (`%K`) — picked from official list
- [ ] **Author** (`%A`) — `_Brian Sheppard_`

## Keyword cheat-sheet (relevant to our entries)

| Keyword | Meaning | Use? |
|---------|---------|------|
| `nonn` | All terms are nonnegative | always for our entries |
| `fini` | Sequence is finite | yes for harmonic `[3, 6, 13, 15, 15]` |
| `full` | All terms are listed | yes for harmonic |
| `hard` | Terms are hard to compute | yes for `a(4) = 116` and `a(3) = 1260` (cpu-xl jobs needed) |
| `nice` | Sequence is "nice" / interesting | optional editorial flag — let editors decide |
| `easy` | Trivial to compute | yes for the closed-form L2 entry |
| `more` | More terms would be welcome | yes for `[3, 6, 17, 116]` and `[6, 14, 62, 1260]` (next terms unknown) |
| `tabl` | Triangle/table | no — we have plain sequences |
| `cofr` | Continued fraction | no |
| `frac` | Sequence of fractions (split into num/den) | no |
| `sign` | Has negative terms | no |
| `cons` | Decimal expansion of a constant | no |

## After submission

- [ ] Status set to **Editing** while iterating, **Proposed** when ready
- [ ] Reply to any pink-box editor queries in your own words (no AI)
- [ ] When approved, record the assigned A-number in `OEIS/README.md`
- [ ] Update repo `README.md` and `registry/experiments.yaml` to reference
      the new A-number(s)
- [ ] Add cross-references to other A-numbers we know about
      (`A095794` for L1, plus any new A-numbers we get for our other
      sequences)
