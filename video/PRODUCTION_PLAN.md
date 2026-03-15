# Video Production Plan: Three-Body Poisson Algebra

## Overview

Two videos from one research paper:
1. **"The Three-Body Problem Has a Hidden Algebraic Code"** (~15-18 min, accessible)
2. **"Computing the Poisson Algebra of the Three-Body Problem"** (~25-30 min, mathematical)

---

## Computational Work (AWS)

### 1. Complete 1000x1000 Atlas
- **Current state**: 880/1000 rows done for 1/r, eps=5e-3
- **Issue**: Near-collision timeouts in remaining rows
- **Fix**: Skip points with r_min < 0.1, interpolate in post-processing
- **Cost**: ~$2-3 on r6i.4xlarge spot
- **Output**: 1000x1000 gap_map.npy, sv_spectra.npy

### 2. High-Res Lagrange Region (1/r² potential)
- **Region**: μ=[0.3, 2.0], φ=[15°, 105°]
- **Resolution**: 1000x1000
- **Epsilon**: 2e-4 (best feature visibility)
- **Cost**: ~$8-12 (10 parallel spot instances)
- **Output**: Concentric rings + bead structure at video resolution

### 3. Atlas Sweep Animation Data (1/r^n, n=1.0→2.0)
- **Current state**: 11 frames at 50x50 with some cached
- **Need**: 60 frames at 200x200 for smooth 2-second video sweep
- **Cost**: ~$5-10 on r6i.4xlarge
- **Output**: 60 gap_map .npy files → rendered to 4K frames

### 4. Three-Potential Comparison Atlas (video resolution)
- **Need**: 500x500 for each of 1/r, 1/r², r² with consistent colormap
- **Cost**: ~$3-5
- **Output**: Side-by-side comparison at 4K

### Total AWS Budget: ~$20-30

---

## Animation Inventory

### Manim Scenes (video/manim_scenes.py)

| Scene | Video | Duration | Description |
|-------|-------|----------|-------------|
| `ThreeBodyIntro` | 1 | 30s | Three masses, pairwise forces, orbits |
| `PoissonBracketExplain` | 1 | 45s | What {f,g} means operationally |
| `BracketTreeGrowth` | 1 | 60s | Branching tree: 3→6→17→116 |
| `DimensionCounter` | 1 | 30s | Numbers ticking up with growth ratio |
| `SVDCliff` | 1+2 | 45s | Singular values falling off a cliff at 116 |
| `PotentialTable` | 1 | 30s | Build comparison table row by row |
| `CMMSurprise` | 1 | 20s | Highlight: integrable system, same sequence! |
| `PolynomialTrick` | 2 | 60s | u=1/r substitution, chain rule |
| `BracketComputation` | 2 | 90s | Step through {H12, H13} explicitly |
| `TidalCompetition` | 2 | 45s | Force diagram, K1 interpretation |
| `MassInvariance` | 2 | 30s | Table of mass configs, all give 116 |
| `Level4Pipeline` | 2 | 45s | Flowchart: symbolic → lambdify → SVD |

### High-Res Data Renders (video/renders/)

| Render | Source | Resolution | Description |
|--------|--------|------------|-------------|
| `atlas_1r_4k.png` | atlas_1000 data | 3840x2160 | Newton atlas, inferno colormap |
| `atlas_1r2_4k.png` | hires data | 3840x2160 | Calogero-Moser atlas |
| `atlas_harmonic_4k.png` | hires data | 3840x2160 | Harmonic atlas (the boring one) |
| `atlas_comparison_4k.png` | all three | 3840x2160 | Side-by-side triptych |
| `svd_spectrum_4k.png` | exact engine | 3840x2160 | Cliff plot with annotations |
| `lagrange_zoom_4k.png` | lagrange hires | 3840x2160 | Concentric ring detail |
| `s3_symmetry_4k.png` | hires + overlay | 3840x2160 | Isosceles curves on atlas |

### 3D Animations (video/3d/)

| Animation | Tool | Duration | Description |
|-----------|------|----------|-------------|
| `shape_sphere_rotation.mp4` | Matplotlib/Manim | 10s | Slow rotation of shape sphere |
| `atlas_sweep.mp4` | Matplotlib | 3s | Potential exponent n: 1→2 |
| `multi_epsilon_zoom.mp4` | Matplotlib | 5s | Zooming into finer epsilon |
| `three_body_orbit.mp4` | Manim | 15s | Chaotic orbit vs Lagrange |
| `bracket_explosion.mp4` | Manim | 10s | Generators multiplying visually |

---

## File Structure

```
video/
├── PRODUCTION_PLAN.md          # This file
├── manim_scenes.py             # All Manim animation scenes
├── render_4k.py                # High-res static renders from data
├── orbit_sim.py                # Three-body orbit integrator for animation
├── aws_hires_sweep.py          # AWS computation for sweep animation
├── renders/                    # Output: 4K static frames
└── media/                      # Output: Manim renders
```

---

## Render Specifications

- **Resolution**: 3840x2160 (4K UHD) for all final assets
- **Frame rate**: 60fps for animations, 30fps acceptable for data sweeps
- **Colormap**: `inferno` for atlas (consistent with paper)
- **Font**: CMU Serif (matches LaTeX aesthetic)
- **Background**: Dark (#1a1a2e or similar) for Manim scenes
- **Aspect ratio**: 16:9
