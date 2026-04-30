# OEIS A395423 — Interactive Explainer

A static, dependency-free webpage that walks readers through the 11
technical terms in the [OEIS A395423](https://oeis.org/A395423) submission,
one interactive widget at a time.

## Stack

- Plain HTML + CSS + vanilla JS (no framework, no build step).
- [KaTeX](https://katex.org/) (CDN) for math.
- A Web Worker per heavy widget; §0 uses an RK4 3-body integrator.
- Optional precomputed JSON tables produced by `precompute.py` (uses the
  repo's own `nbody/exact_growth_nbody.py`).

## Local preview

```powershell
python -m http.server -d explainer 8000
# then open http://localhost:8000
```

(A bare `file://` open works for §0 except for the Web Worker, which
requires an http origin.)

## Deploy to GitHub Pages

The intended public URL is

    https://bshepp.github.io/3body-poisson-algebra/explainer/

Easiest route: enable Pages on `main` / root, and the `explainer/` folder
is served as-is at the URL above. No build step required.

## Layout

```
explainer/
├── README.md
├── index.html              scrollytelling shell, one <section> per OEIS term
├── style.css
├── js/
│   ├── orbit_worker.js     RK4 integrator (Web Worker)
│   ├── orbit_widget.js     §0 canvas + sliders + presets
│   └── ...                 (more widgets land here as sections come online)
├── data/                   precomputed JSON (bracket tables, dim sequences)
└── precompute.py           one-time generator for data/
```

## Status

| Section | Topic                                                          | State |
|---------|----------------------------------------------------------------|-------|
| §0      | Three-body system (orbit playground)                           | ✅ |
| §1      | Symplectic phase space (pendulum portrait)                     | ✅ |
| §2      | Poisson bracket calculator                                     | ✅ |
| §3      | Pairwise potentials $H_{ij}=V(r_{ij})$                         | ✅ |
| §4      | $L_0,\ L_{n+1}=[L_0,L_n]$ — building the algebra              | ✅ |
| §5      | The dimension sequence $a(n)$ (3, 6, 17, 116)                  | ✅ |
| §6      | Singular potentials and universality                           | ✅ |
| §7      | Spatial dimension $d\in\{1,2,3\}$                              | ✅ |
| §8      | Mass invariance                                                | ✅ |
| §9      | SVD rank gap $>10^{10}$                                        | ✅ |
| §10     | The frontier: $a(4)\geq 5604$                                  | ✅ |

Side-tracks (collapsibles) D1–D5 follow once the spine is up.
