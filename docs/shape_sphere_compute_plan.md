# Shape Sphere — Missing Data Compute Plan

**Status:** ✅ DONE — Plan B implemented (`shape_sphere_atlas.py`); 8×16 smoke scan deployed to viewer; AWS launcher `infra/userdata_sphere.sh` ready for production-resolution scan.
**Owner:** atlas team
**Companion code:** `website/interactive.html` (`renderShapeSphere`, `muPhiToShapeSphere`)

## 1. The shape sphere we're rendering

Following Hsiang–Montgomery / Iwai, for the equal-mass planar 3-body problem
we form Jacobi vectors

$$
\rho_1 = r_2 - r_1, \qquad
\rho_2 = \tfrac{2}{\sqrt{3}}\bigl(r_3 - \tfrac{r_1+r_2}{2}\bigr)
$$

(the `2/√3` chosen so the equilateral triangle satisfies
$|\rho_1|=|\rho_2|$, $\rho_1\!\cdot\!\rho_2 = 0$). The hyperradius is
$R^2 = |\rho_1|^2 + |\rho_2|^2$. The shape sphere coordinates are

$$
\bigl(s_1,s_2,s_3\bigr) = \frac{1}{R^2}\bigl(\,|\rho_1|^2-|\rho_2|^2,\;\;
2\rho_1\!\cdot\!\rho_2,\;\; 2\,\rho_1\!\times\!\rho_2\,\bigr),
\qquad s_1^2+s_2^2+s_3^2 = 1.
$$

In atlas coordinates ($r_1=(0,0)$, $r_2=(1,0)$, $r_3=(\mu\cos\varphi,\mu\sin\varphi)$),
let $A = \mu^2 - \mu\cos\varphi + \tfrac14$. Then

$$
R^2 = 1 + \tfrac{4}{3}A, \qquad
w_1 = 1 - \tfrac{4}{3}A,\quad
w_2 = \tfrac{4}{\sqrt{3}}\bigl(\mu\cos\varphi-\tfrac12\bigr),\quad
w_3 = \tfrac{4}{\sqrt{3}}\,\mu\sin\varphi.
$$

### Landmarks

| Configuration | $(\mu, \varphi)$ | $(s_1, s_2, s_3)$ | Sampled? |
|---|---|---|---|
| Equilateral $L_4$ | $(1,\;+\pi/3)$ | $(0,0,+1)$ — north pole | ✓ |
| Equilateral $L_5$ | $(1,\;-\pi/3)$ | $(0,0,-1)$ — south pole | ✓ |
| Binary $r_2{=}r_3$ | $(1,\;0)$ | $(\tfrac12,\;+\tfrac{\sqrt3}{2},\;0)$ | ✓ |
| Binary $r_1{=}r_3$ | $(0,\;\text{any})$ | $(\tfrac12,\;-\tfrac{\sqrt3}{2},\;0)$ | ✓ |
| **Binary $r_1{=}r_2$** | $\mu\to\infty$ | $(-1,0,0)$ | **✗** |
| Equator ($s_3=0$) | $\varphi\in\{0,\pi\}$ | collinear configs | ✓ partial |

The $r_1{=}r_2$ collision is the single point where the chosen
parameterization fails — we pinned the $r_1\,r_2$ separation at unit length,
so reaching $r_1{=}r_2$ requires $\mu\to\infty$ (every other separation
becomes infinite by comparison). The whole spherical cap around
$(-1,0,0)$ is therefore unsampled by the current atlas.

## 2. What "missing" means quantitatively

The current atlas (`website/data/atlas_*.json`) covers
$\mu\in[0.2,\,3.0]$, $\varphi\in[0,\,\pi]$ (then mirrored to
$\varphi\in[-\pi,\pi]$ for display). The image of this rectangle on the
shape sphere is the complement of a cap around $s_1=-1$. Concretely,

$$
s_1 = \frac{3 - 4A}{3 + 4A}, \qquad A_{\max} = \mu_{\max}^2 + \mu_{\max} + \tfrac14.
$$

For $\mu_{\max}=3$: $A_{\max} = 12.25$, $s_1^{\min} = -49/52 \approx -0.942$.
The unsampled cap subtends
$\theta = \arccos(-0.942) - \arccos(-1) \approx 19.6°$ of polar angle,
which is the obvious blank patch on the rendered sphere near the
"r₁=r₂" placeholder.

## 3. Strategies to fill the cap

Three viable approaches, in increasing order of effort.

### Option A — Extend $\mu$ to large values (cheap, partial)

Run the existing atlas pipeline on $\mu \in [3,\,30]$ (or further) using
the same script set (`atlas_1000.py`, `parametric_atlas_scan.py`). This
shrinks but never closes the cap — at $\mu_{\max}=30$, $A_{\max}=930.25$,
$s_1^{\min} \approx -0.99916$, residual cap $\approx 2.4°$. Costs scale
linearly with the new $\mu$ rows; on the existing AWS spot fleet this is
a one-evening job per potential.

**Pros:** zero new code, drop-in compatible with all downstream tooling.
**Cons:** asymptotic; the singular point itself is never reached;
oversamples the dynamically uninteresting "one body very far away" regime.

### Option B — Reparameterize via the hyperradius (recommended)

Sample the configuration space directly on the shape sphere by inverting
the Jacobi map. Pick a uniform spherical grid (e.g. Fibonacci lattice,
$N \sim 10^4$ points) and for each $(s_1,s_2,s_3)$ recover representative
positions via

$$
|\rho_1|^2 = \tfrac{R^2}{2}(1+s_1),\quad |\rho_2|^2 = \tfrac{R^2}{2}(1-s_1),\quad
\rho_1\!\cdot\!\rho_2 = \tfrac{R^2}{2}s_2,\quad \rho_1\!\times\!\rho_2 = \tfrac{R^2}{2}s_3.
$$

Choose $R=1$ (scale-invariant; rank/gap are scale-invariant for power-law
potentials) and pick a frame for $\rho_1$, e.g. $\rho_1 = (|\rho_1|, 0)$.
Then $\rho_2$ is determined up to sign of $s_3$. Convert to
$(r_1, r_2, r_3)$ by translating so the centroid sits at the origin.

This produces uniform coverage of the shape sphere including all three
binary collisions (now interior points of the grid, with the usual
collision-singularity blow-up which the existing $u_{ij}=1/r_{ij}$ chain
rule handles).

**Pros:** correct geometry, uniform statistical coverage, all three
binary collisions appear symmetrically.
**Cons:** new sampling driver (~150 LOC); need to add a thin adapter
that returns the same `{mu, phi, rank, gap}` schema (or a new
`{s1,s2,s3, rank, gap}` schema and a new viewer code path).

### Option C — Patch atlas: $r_1{=}r_2$-centric chart (targeted)

Run a *second* atlas with $r_1$ and $r_3$ pinned at unit separation
instead of $r_1, r_2$. This is exactly the same code with a relabeling of
particles, and its $(\mu,\varphi)$ rectangle covers the cap around
$(-1,0,0)$ on the original chart. Stitch the two atlases by transforming
both onto the shape sphere.

**Pros:** zero changes to the engine; trivially parallel; produces
exactly the missing cap.
**Cons:** atlas now has two charts that must be reconciled (overlap region
provides a consistency check, which is a feature for validation).

## 4. Recommended sequence

1. **Now (no compute):** ship the current renderer with the (-1, 0, 0)
   "unsampled" placeholder so users see the missing region honestly.
   *(Done in commit shipping this doc.)*
2. **Short-term (Option C):** run a single permuted-particle atlas on
   the same spot-fleet recipe used for `atlas_1000`. Estimate: ~$10–20
   in spot, one wall-clock day per potential. Stitch via the shape-sphere
   map. Use the overlap region as a regression check.
3. **Medium-term (Option B):** add a `shape_sphere_atlas.py` driver that
   samples directly on $S^2$ via the Jacobi inversion above. Becomes the
   primary atlas for any future N=3 work; old $(\mu,\varphi)$ atlases
   stay as legacy / for reproducibility of published figures.
4. **Long-term:** lift the same construction to $N=4$ (shape *space* is
   $\mathbb{CP}^2$ for $N=4$ planar; not a sphere) — out of scope here.

## 5. Open questions

- **Mass ratios.** All formulas above assume equal masses. For unequal
  masses the Jacobi normalization changes; the binary-collision points
  no longer sit at exactly $120°$ on the equator. The existing
  `mass_ratio_sweep.py` data could be reprojected onto a mass-dependent
  shape sphere — worth doing once Option B is in place.
- **Reflection symmetry.** Currently we mirror $\varphi \to -\varphi$ at
  render time (since the atlas only stores $\varphi\ge 0$). Once Option B
  or C is in place, drop the mirror — it should fall out of the data.
- **Display.** The shape sphere rendering currently shows only the
  primary panel. Consider extending to comparison/diff panels for the
  multi-potential view (small JS refactor; orthogonal to this plan).

## 6. Code touchpoints

| File | Change |
|---|---|
| `website/interactive.html` | Already updated to use canonical Jacobi map. Will need to consume new schema if Option B lands. |
| `atlas_1000.py` | Add a `--swap-particles` flag for Option C. |
| (new) `shape_sphere_atlas.py` | Option B driver. |
| `website/data/manifest.json` | Add an `atlas_chart` field once multiple charts exist. |
| `nbody/exact_growth_nbody.py` | No changes — the engine is parameterization-agnostic. |
