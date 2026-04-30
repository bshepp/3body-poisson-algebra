# OEIS submissions — 3body Poisson algebra dimension sequences

This folder collects the integer sequences produced by this codebase that
are candidates for inclusion in the [On-Line Encyclopedia of Integer
Sequences](https://oeis.org/), together with the supporting evidence,
draft entries, and a checklist that maps the OEIS Style Sheet to our
specific situation.

> **Author note.** Brian Sheppard already has an OEIS account. This folder
> is preparatory: it produces draft text in OEIS format that he can
> review and paste into the submission form. Nothing here is auto-submitted.

---

## Candidate inventory

| # | Sequence (first few terms) | What it is | OEIS status | Draft |
|---|----------------------------|------------|-------------|-------|
| 1 | `3, 6, 17, 116` | N=3 cumulative dim under Poisson bracket, singular pairwise potentials (1/r, 1/r², 1/r³, log, Yukawa, composite); d-, mass-, potential-invariant; cross-CAS confirmed (Python + Wolfram) | **Submitted as [A395423](https://oeis.org/A395423)** (2026-04-21) | [`candidates/A395423.md`](candidates/A395423.md) — interactive walkthrough: [`explainer/`](../explainer/) |
| 2 | `6, 14, 62, 1260` | N=4 cumulative dim under Poisson bracket, singular potentials; d=1,2,3 confirmed; mass-invariant; potential-invariant for 1/r², 1/r³, log | **Novel** — search returns no results | [`candidates/A_3body_n4_singular.md`](candidates/A_3body_n4_singular.md) |
| 3 | `3, 6, 13, 15, 15` | N=3 harmonic (V = +r²) cumulative dim; algebra **closes** at dim 15 (verified L=4 in Mathematica with 11,937 brackets) | **Novel** — search returns unrelated factorial sequence | [`candidates/A_3body_n3_harmonic.md`](candidates/A_3body_n3_harmonic.md) |
| 4 | `62, 145, 279, 476, 748, 1107, ...` (N=4..9) | Closed-form L2(N) = N(4N²−9N+3)/2 for the singular potentials; verified for N=4..9 | **Novel** — search returns no results | [`candidates/A_3body_l2_closed_form.md`](candidates/A_3body_l2_closed_form.md) |
| 5 | `1, 6, 14, 25, 39, 56, 76, 99, 125, 154, ...` | Closed-form L1(N) = N(3N−5)/2 for N≥3 (offset 1: a(n)=L1(n+1)) | **Already in OEIS as [A095794](https://oeis.org/A095794)** — `(n+1)(3n−2)/2`, second pentagonal numbers minus 1 | [`candidates/A095794_addition.md`](candidates/A095794_addition.md) (comment + cross-ref only) |

A note on novelty: a "no results" hit on the OEIS sequence search is
strong evidence the sequence is genuinely new, but the editor will
ultimately decide. If an editor identifies an existing entry that
matches, the appropriate response is to add a cross-reference and a
comment to that entry instead of creating a duplicate.

---

## Submission process (from oeis.org/Submit.html and Style Sheet)

### Where you submit

You go to <https://oeis.org/> while logged in and click **Contribute → New
Sequence**. The form is one big page with the field names that appear in
each entry's source view (`%N`, `%S`, `%C`, `%H`, …).

### Required fields

The following are mandatory or near-mandatory for a new sequence:

1. **Name** (`%N`) — one-line description; uses `a(n)` for the n-th term.
   Avoid vanity (don't name it after yourself). Use ASCII only.
2. **Data** (`%S`) — at least 4 terms, ideally 200–500 characters worth.
   Comma-separated integers, no whitespace surprises.
3. **Offset** (`%O`) — the index of the first term. For lists usually 1
   (sometimes 0). The system will compute the second offset value
   (1-based index of the first |a(n)| > 1) automatically.
4. **Keywords** (`%K`) — pick from the
   [official keyword list](https://oeis.org/eishelp2.html#RK). For us
   the relevant ones are `nonn` (nonnegative), `fini` and `full` (if
   the sequence is finite and complete — applies to the harmonic case),
   `hard` (computationally difficult), `nice` (broadly interesting).
5. **Author** (`%A`) — your name in the form `_Brian Sheppard_, Apr 21
   2026`. The underscores produce a link to your user page.

### Highly recommended fields

6. **Comments** (`%C`) — context, motivation, what makes the sequence
   meaningful. Multi-paragraph block uses
   `From _Author_, Date: (Start) ... (End)`.
7. **Links** (`%H`) — papers, b-files, illustrations. **Use stable
   URLs.** A link to the GitHub repo and to the published paper(s) is
   appropriate here.
8. **Formula** (`%F`) — closed-form expressions, generating functions,
   recurrences. Use ASCII conventions: `Sum_{k=1..n}`, `Pi`, `sqrt`,
   `^` for exponent, `*` for multiplication.
9. **Example** (`%e`) — show your work for one or two terms.
10. **Mathematica** (`%t`) and/or **Programs** (`%o`) — self-contained
    code that reproduces the terms. Sign with `(* ~~~~ *)` (Mathematica)
    or `# ~~~~` (Python). The `~~~~` is auto-replaced with your name and
    date by the OEIS server. **You must understand the code you submit.**
11. **Cross-references** (`%Y`) — `Cf. A000000, A000000` style.

### Style rules that bite

- **ASCII only** in math text. No `≤`, `→`, `π`, `∑`, `…`. Use `<=`,
  `->`, `Pi`, `Sum_`, `...`.
- **`a(n)` not `a[n]` or `a_n`**.
- **`Sum_{k=a..b}`**, **`Product_{k=a..b}`** with capital S / P.
- **`*` for multiplication**, `^` for exponentiation (not `**`, `²`,
  `³`, `·`, `×`).
- **US spelling**: behavior, color, neighbor, labeled, generalize, …
- **`mod` not `%`** for the modulo operation.
- Comments and code by the original author of the submission **don't**
  need a `~~~~` signature; later additions do.
- Don't paste your email. Sign with `~~~~` (Wiki tilde syntax).

### AI policy

The OEIS [AI policy](https://oeis.org/wiki/Use_of_AI_for_OEIS_Submissions)
is strict but accommodating:

- AI can be **used to draft** name/comments/formula text and to write
  programs, **as long as the human submitter understands and verifies
  every word and every line of code**.
- AI **cannot be listed as an author** (US copyright bars it).
- AI-generated text **cannot be pasted into editor-query response boxes**
  (the "Pink Boxes").
- **No bulk or serial AI-driven submissions** without prior permission.

For our case: the drafts in `candidates/` were prepared with AI
assistance; before submission Brian needs to read each draft end-to-end,
confirm every term, every formula, and every program is one he stands
behind, and then post the form himself. This is well within the policy
and is the same standard expected of any submission.

### Workflow

1. Pick one candidate from the table above (start with `[3, 6, 17, 116]`
   — it has the most context behind it).
2. Open `candidates/<entry>.md` and review the draft against the OEIS
   field list. Make any corrections you want directly in that file.
3. Open <https://oeis.org/> while logged in, go to **Contribute → New
   Sequence**, and paste field by field.
4. Before clicking **Save**, set status to **Editing** to keep iterating;
   set to **Proposed** when you're ready for the editorial board to look.
5. Editors will leave queries in pink boxes on the draft page; reply
   directly there in your own words (per AI policy, do not paste LLM
   output into pink boxes).
6. Once approved you'll get an A-number — record it back in this folder
   so we know what got accepted.

### B-file (optional but appreciated)

For the closed-form L2 sequence (`62, 145, 279, 476, 748, ...`) we can
trivially compute thousands of terms. A b-file with 10,000 terms is
standard. The format is one `n a(n)` per line, plain ASCII, blank line
at the end:

```
4 62
5 145
6 279
7 476
8 748
9 1107
10 1565
...
```

For the cumulative dimension sequences (`[3, 6, 17, 116]` and friends)
we **cannot** extend a b-file beyond what we have computed: the next
term `a(5)` of `[3, 6, 17, 116]` is conjectured to be ≥ 5,604 but is
not yet known exactly. So those entries should not have b-files; the
limit on terms is genuinely a research frontier.

---

## What's in this folder

```
OEIS/
├── README.md                                 (this file)
├── checklist.md                              (Style Sheet -> our entries)
└── candidates/
    ├── A395423.md                            (submitted as A395423: [3, 6, 17, 116])
    ├── A_3body_n4_singular.md                (draft for [6, 14, 62, 1260])
    ├── A_3body_n3_harmonic.md                (draft for [3, 6, 13, 15, 15])
    ├── A_3body_l2_closed_form.md             (draft for L2(N))
    └── A095794_addition.md                   (cross-ref + comment for L1(N))
```

The `A_3body_*` filenames are placeholders — they will be replaced with
the actual A-numbers once OEIS assigns them.

## See also

- [OEIS Style Sheet](https://oeis.org/wiki/Style_Sheet) — the canonical
  formatting reference
- [OEIS keyword descriptions](https://oeis.org/eishelp2.html#RK)
- [OEIS AI policy](https://oeis.org/wiki/Use_of_AI_for_OEIS_Submissions)
- [b-file submission guide](https://oeis.org/SubmitB.html)
- [Project README](../README.md) — context for what these sequences mean
- [`bench_flint/validation_summary.md`](../bench_flint/validation_summary.md)
  — Phase F oracle independent CAS verification
- [`mathematica/`](../mathematica/) — Wolfram Language reproductions
