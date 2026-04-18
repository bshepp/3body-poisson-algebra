#!/usr/bin/env python3
"""Curated comparison figures for the Figures page.

Renders ~10 named multi-panel figures into figures_v2/comparisons/. Each panel
shares its color scale across configurations so the comparison is honest.

Inputs:
  aws_results/atlas_full/<config>/   -- atlas heatmaps for power-law and named systems
  atlas_targeted/<config>/<region>/  -- targeted regions (Lagrange continuity strip)
  dataset/output/*.parquet           -- neural / tier / dimension-sequences tables
  primes/figures/*.png               -- already-rendered GUE figures
  atlas_output_hires/*               -- knee landscapes (1/r, 1/r^2, harmonic SV spectra)

Output:
  figures_v2/comparisons/<id>.png

Run from the repository root:
    python website/figures_compare.py
    python website/figures_compare.py --force
    python website/figures_compare.py --only neural_classes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures_v2", "comparisons")
AWS_FULL = os.path.join(ROOT, "aws_results", "atlas_full")
TARGETED = os.path.join(ROOT, "atlas_targeted")
HIRES = os.path.join(ROOT, "atlas_output_hires")
DATASETS = os.path.join(ROOT, "dataset", "output")
PRIMES_FIG = os.path.join(ROOT, "primes", "figures")
LEGACY = os.path.join(ROOT, "legacy_figures_archive")

DPI = 150
CMAP = cm.inferno
INVALID_COLOR = "#2a3040"


# ─── helpers ───

def _load_atlas(d):
    if not os.path.isdir(d):
        return None
    paths = ["mu_vals.npy", "phi_vals.npy", "gap_map.npy"]
    for p in paths:
        if not os.path.exists(os.path.join(d, p)):
            return None
    mu = np.load(os.path.join(d, "mu_vals.npy"))
    phi = np.load(os.path.join(d, "phi_vals.npy"))
    gap = np.load(os.path.join(d, "gap_map.npy"))
    rank_p = os.path.join(d, "rank_map.npy")
    rank = np.load(rank_p) if os.path.exists(rank_p) else None
    # Mask invalid cells (rank == -1, gap <= 0) so they show as the no-data color
    invalid = ~np.isfinite(gap) | (gap <= 0)
    if rank is not None:
        invalid = invalid | (rank == -1)
    gap = np.where(invalid, np.nan, gap)
    return mu, phi, gap


def _load_full(name):
    return _load_atlas(os.path.join(AWS_FULL, name))


def _load_targeted(cfg, region):
    d = os.path.join(TARGETED, cfg, region)
    out = _load_atlas(d)
    if out is None:
        # gap_score_map fallback
        if os.path.isdir(d):
            mu = np.load(os.path.join(d, "mu_vals.npy"))
            phi = np.load(os.path.join(d, "phi_vals.npy"))
            gp = os.path.join(d, "gap_score_map.npy")
            if os.path.exists(gp):
                return mu, phi, np.load(gp)
        return None
    return out


def _apply_axes_aspect(ax, mu, phi):
    """No-op kept for call-site compatibility (see figures_render.py)."""
    return None


def _heatmap(ax, mu, phi, log_gap, vmin, vmax, title, cmap=CMAP):
    cmap_bad = cmap.copy() if hasattr(cmap, "copy") else cm.get_cmap(cmap.name)
    cmap_bad.set_bad(color=INVALID_COLOR)
    im = ax.pcolormesh(phi * 180 / np.pi, mu, np.ma.masked_invalid(log_gap),
                       cmap=cmap_bad, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlim(float(phi.min()) * 180 / np.pi, float(phi.max()) * 180 / np.pi)
    ax.set_ylim(float(mu.min()), float(mu.max()))
    _apply_axes_aspect(ax, mu, phi)
    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=10)
    ax.set_ylabel(r"$\mu$", fontsize=10)
    ax.set_title(title, fontsize=11)
    return im


def _shared_range(arrays, lo=2, hi=98):
    pool = []
    for a in arrays:
        v = a[np.isfinite(a)]
        if v.size:
            pool.append(v.ravel())
    if not pool:
        return 0.0, 1.0
    cat = np.concatenate(pool)
    return float(np.percentile(cat, lo)), float(np.percentile(cat, hi))


def _annotate_partial(ax, gap_arr):
    """Add an N-of-M overlay if a fraction of the grid is unmeasured."""
    invalid = ~np.isfinite(gap_arr)
    n_inv = int(invalid.sum())
    if n_inv > 0:
        frac = n_inv / invalid.size
        ax.text(0.985, 0.02, f"{n_inv}/{invalid.size} cells unmeasured ({frac:.0%})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#c8ccd4",
                bbox=dict(boxstyle="round,pad=0.25", fc="#1e2430", ec="#2a3040", alpha=0.85))


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out_path, ROOT)}")


def _row_panel(configs, title, out_id, force):
    """Generic 1xN heatmap panel from a list of (label, full_config_name)."""
    out = os.path.join(OUT, f"{out_id}.png")
    if os.path.exists(out) and not force:
        return False
    loaded = []
    for label, name in configs:
        d = _load_full(name)
        if d is None:
            print(f"  SKIP {out_id}: missing {name}")
            return False
        mu, phi, gap = d  # gap already has invalid cells set to NaN
        # Don't artificially floor at log10(1)=0 -- that crushes valid sub-decade
        # gaps to a single bin. Floor only at 1e-3 to keep log10 finite for the
        # rare near-zero cell that survived the validity mask.
        g = np.where(np.isfinite(gap), gap, np.nan)
        g = np.where(g < 1e-3, 1e-3, g)
        log_gap = np.log10(g)
        loaded.append((label, mu, phi, log_gap, gap))
    arrays = [t[3] for t in loaded]
    vmin, vmax = _shared_range(arrays)
    if vmax - vmin < 0.5:
        mid = 0.5 * (vmin + vmax)
        vmin, vmax = mid - 0.5, mid + 0.5
    n = len(loaded)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), facecolor="white")
    if n == 1:
        axes = [axes]
    last_im = None
    for ax, (label, mu, phi, lg, raw_gap) in zip(axes, loaded):
        last_im = _heatmap(ax, mu, phi, lg, vmin, vmax, label)
        _annotate_partial(ax, raw_gap)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\log_{10}$(gap ratio)", fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    _save(fig, out)
    return True


# ─── individual comparison renderers ───

def cmp_singular_vs_harmonic(force):
    return _row_panel(
        [
            ("1/r (Newton)", "1r_q+1_+1_+1"),
            ("1/r^2 (Calogero-Moser)", "1r2"),
            ("log(r) (2D vortex)", "log"),
            ("r^2 (harmonic)", "1r-2"),
        ],
        "Singular vs harmonic dichotomy",
        "singular_vs_harmonic", force,
    )


def cmp_irrational_exponents(force):
    return _row_panel(
        [
            (r"$1/r^{\pi}$", "1r3p14159"),
            (r"$1/r^{e}$", "1r2p71828"),
            (r"$1/r^{\varphi}$", "1r1p61803"),
        ],
        "Irrational-exponent universality",
        "irrational_exponents", force,
    )


def cmp_exponent_continuity(force):
    """r^1, r^1.01, r^2, r^2.01, r^3, r^4 (singular family)."""
    # Note: 1r3 == 1/r^3, 1r-2 == r^2 etc. We use 1/r^k family for the strip
    # because it covers the exceptional integer transition cleanly.
    return _row_panel(
        [
            ("1/r", "1r_q+1_+1_+1"),
            (r"$1/r^{1.01}$", "1r1p01"),
            (r"$1/r^{2}$", "1r2"),
            (r"$1/r^{2.01}$", "1r2p01"),
            (r"$1/r^{3}$", "1r3"),
        ],
        "Exponent continuity across exceptional integers",
        "exponent_continuity", force,
    )


def cmp_mass_extremes(force):
    return _row_panel(
        [
            ("Equal mass (baseline)", "1r_q+1_+1_+1"),
            ("Sun-Earth-Moon", "sun_earth_moon_1r_m1p0_3e-06_3p7e-08"),
            ("Sun-Jupiter-Asteroid", "sun_jupiter_asteroid_1r_m1p0_0p00095_1e-10"),
            ("Triple BH (LISA)", "triple_bh_lisa_1r_m1p0_0p01_1e-05"),
        ],
        "Mass-ratio extremes -- conditioning failure modes",
        "mass_ratio_extremes", force,
    )


def cmp_charge_departure(force):
    return _row_panel(
        [
            ("He (+2,-1,-1)", "1r_q+2_-1_-1"),
            ("Li+ (+3,-1,-1)", "1r_q+3_-1_-1"),
            ("H2+ (+1,+1,-1)", "1r_q+1_+1_-1"),
        ],
        "Charge-magnitude departure from universal 116 at L3",
        "charge_departure", force,
    )


# ─── data-driven (parquet-backed) comparisons ───

def cmp_quantum_plus_one(force):
    out = os.path.join(OUT, "quantum_plus_one.png")
    if os.path.exists(out) and not force:
        return False
    try:
        import pandas as pd
    except ImportError:
        print("  SKIP quantum_plus_one: pandas not available")
        return False
    df = pd.read_parquet(os.path.join(DATASETS, "dimension_sequences.parquet"))
    sub = df[(df["N"] == 3) & (df["d"] == 1)]
    moyal = sub[sub["bracket_type"] == "moyal"].copy()
    poisson = sub[sub["bracket_type"] == "poisson"].copy()
    # Pair singular potentials that have both
    import pandas as _pd
    pots = []
    for p in moyal["potential"].unique():
        mrow = moyal[moyal["potential"] == p].iloc[0]
        prow = poisson[poisson["potential"] == p]
        if len(prow) == 0:
            continue
        prow = prow.iloc[0]
        if _pd.isna(mrow.get("dim_L3")) or _pd.isna(prow.get("dim_L3")):
            continue
        pots.append((p, int(prow["dim_L3"]), int(mrow["dim_L3"])))
    if not pots:
        print("  SKIP quantum_plus_one: no paired classical/Moyal rows")
        return False
    pots.sort(key=lambda r: r[0])
    labels = [r[0] for r in pots]
    classical = [r[1] for r in pots]
    quantum = [r[2] for r in pots]
    delta = [q - c for c, q in zip(classical, quantum)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 5.5), facecolor="white")
    w = 0.38
    ax.bar(x - w / 2, classical, w, color="#4ac8e8", label="classical (Poisson)")
    ax.bar(x + w / 2, quantum, w, color="#a07ae0", label=r"quantum (Moyal, $\hbar$)")
    for xi, d in zip(x, delta):
        if d:
            ax.text(xi, max(classical[xi], quantum[xi]) + 1.5,
                    f"+{d}" if d > 0 else str(d),
                    ha="center", color="#3ddc84", fontweight="bold")
    ax.axhline(116, ls="--", color="#3ddc84", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("dim L3")
    ax.set_title(r"Classical Poisson vs Moyal quantum: the +1 quantum generator",
                 fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, out)
    return True


def cmp_neural_classes(force):
    out = os.path.join(OUT, "neural_classes.png")
    if os.path.exists(out) and not force:
        return False
    try:
        import pandas as pd
    except ImportError:
        print("  SKIP neural_classes: pandas not available")
        return False
    df = pd.read_parquet(os.path.join(DATASETS, "neural_algebras.parquet"))
    df = df.dropna(subset=["dim_L3", "universality_class_L3"])
    df = df.sort_values(["universality_class_L3", "coupling_type"])
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    classes = sorted(df["universality_class_L3"].unique())
    palette = plt.get_cmap("tab10")
    for i, cls in enumerate(classes):
        sub = df[df["universality_class_L3"] == cls]
        labels = [f"{r['coupling_type']}\nL={r['n_layers']},k={r['width']}"
                  for _, r in sub.iterrows()]
        x = np.arange(len(labels)) + (i * 0.001)  # ordering only; real x below
        # Better: lay out per-class groups along x
    # Simpler layout: one bar per row
    rows = df.reset_index(drop=True)
    x = np.arange(len(rows))
    color_map = {cls: palette(i % 10) for i, cls in enumerate(classes)}
    bars = ax.bar(
        x, rows["dim_L3"],
        color=[color_map[c] for c in rows["universality_class_L3"]],
    )
    ax.axhline(116, ls="--", color="#3ddc84", lw=1, alpha=0.7,
               label="physics universal 116")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['coupling_type']}\nL{r['n_layers']} k{r['width']} {r['loss_function']}"
         for _, r in rows.iterrows()],
        rotation=70, ha="right", fontsize=8,
    )
    ax.set_ylabel("dim L3")
    ax.set_title("Neural-network coupling -> seven universality classes at L3", fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[c]) for c in classes]
    ax.legend(handles + [plt.Line2D([0], [0], color="#3ddc84", ls="--")],
              [f"Class {c}" for c in classes] + ["physics 116"],
              loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    _save(fig, out)
    return True


def cmp_tier_decomposition(force):
    out = os.path.join(OUT, "tier_decomposition.png")
    if os.path.exists(out) and not force:
        return False
    try:
        import pandas as pd
    except ImportError:
        print("  SKIP tier_decomposition: pandas not available")
        return False
    df = pd.read_parquet(os.path.join(DATASETS, "tier_decomposition.parquet"))
    if df.empty:
        return False
    df["key"] = df.apply(lambda r: f"N={int(r['N'])} L{int(r['level'])}", axis=1)
    keys = df["key"].drop_duplicates().tolist()
    irreps = df["irrep_name"].drop_duplicates().tolist()
    palette = plt.get_cmap("Set2")
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.1), 6), facecolor="white")
    bottoms = np.zeros(len(keys))
    for i, irr in enumerate(irreps):
        vals = []
        for k in keys:
            row = df[(df["key"] == k) & (df["irrep_name"] == irr)]
            v = float(row["contribution"].iloc[0]) if len(row) else 0.0
            vals.append(v)
        ax.bar(keys, vals, bottom=bottoms, label=irr, color=palette(i % 8))
        bottoms += np.array(vals)
    ax.set_ylabel("contribution to observed rank")
    ax.set_title(r"$S_3 / S_4$ irrep decomposition by (N, level)", fontsize=13)
    ax.legend(title="irrep", loc="upper left", fontsize=9, ncol=2)
    plt.xticks(rotation=20)
    plt.tight_layout()
    _save(fig, out)
    return True


# ─── primes / GUE ───

def cmp_gue_overlay(force):
    """Composite the existing GUE figures into a 2x2 panel."""
    out = os.path.join(OUT, "gue_overlay.png")
    if os.path.exists(out) and not force:
        return False
    candidates = [
        ("Finite-N GUE scaling",       "finite_n_gue_scaling.png"),
        ("Spacing overlay (finite N)", "finite_n_spacing_overlay.png"),
        ("Singular-potential spacings", "singular_potential_spacing_overlay.png"),
        ("Multi-potential comparison", "multi_potential_r_comparison.png"),
    ]
    have = []
    for label, fn in candidates:
        # Primes figures were archived; restore from legacy archive read-only
        for base in (PRIMES_FIG, os.path.join(LEGACY, "primes", "figures")):
            p = os.path.join(base, fn)
            if os.path.exists(p):
                have.append((label, p))
                break
    if not have:
        print("  SKIP gue_overlay: source PNGs not found")
        return False
    n = len(have)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows), facecolor="white")
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    for i, (label, p) in enumerate(have):
        img = plt.imread(p)
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=11)
        axes[i].axis("off")
    for j in range(len(have), len(axes)):
        axes[j].axis("off")
    fig.suptitle(r"GUE log-gas: spacings agree with random-matrix predictions", fontsize=14,
                 fontweight="bold", y=1.005)
    plt.tight_layout()
    _save(fig, out)
    return True


# ─── knee landscape comparison ───

def _knee_index_grid(sv_arr, sv_floor=1e-10):
    """Compute knee index per (mu, phi) from a (n_mu, n_phi, n_sv) array.

    Returns a 2D array of indices where the steepest log10-ratio drop occurs,
    using only well-conditioned values (>= sv_floor); cells whose entire
    spectrum is sub-floor are returned as NaN. This avoids the artificial
    "knee at index 0" problem when an algebra closes early and the rest of
    the spectrum is effectively zero.
    """
    if sv_arr.ndim != 3:
        return None
    sv = np.where(sv_arr > sv_floor, sv_arr, np.nan)
    log_sv = np.log10(sv)
    drops = log_sv[..., :-1] - log_sv[..., 1:]
    # Mask non-finite drops; leave finite drops where both neighbors are real
    drops = np.where(np.isfinite(drops), drops, -np.inf)
    knee = np.argmax(drops, axis=-1) + 1
    # If every drop in a cell is -inf, there's no knee
    has_real = np.isfinite(drops).any(axis=-1)
    knee = np.where(has_real, knee.astype(float), np.nan)
    return knee


def _knee_loader(parent_path, label):
    """Try several layouts to find an sv_spectra.npy + mu_vals/phi_vals."""
    candidates = [
        parent_path,
        os.path.join(parent_path, "adaptive"),
        os.path.join(parent_path, "eps_5e-04"),
        os.path.join(parent_path, "eps_2e-04"),
        os.path.join(parent_path, "eps_1e-04"),
    ]
    for d in candidates:
        sv_p = os.path.join(d, "sv_spectra.npy")
        mu_p = os.path.join(d, "mu_vals.npy")
        phi_p = os.path.join(d, "phi_vals.npy")
        if os.path.exists(sv_p) and os.path.exists(mu_p) and os.path.exists(phi_p):
            return label, np.load(mu_p), np.load(phi_p), np.load(sv_p)
    return None


def cmp_knee_landscape_set(force):
    out = os.path.join(OUT, "knee_landscape_set.png")
    if os.path.exists(out) and not force:
        return False
    sources = [
        ("1/r (Newton)", os.path.join(HIRES, "1_r")),
        ("1/r^2 (Calogero-Moser)", os.path.join(HIRES, "1_r2")),
        ("r^2 (harmonic)", os.path.join(HIRES, "harmonic")),
    ]
    panels = []
    for label, parent in sources:
        if not os.path.isdir(parent):
            print(f"  SKIP knee panel '{label}' (missing dir)")
            continue
        loaded = _knee_loader(parent, label)
        if loaded is None:
            print(f"  SKIP knee panel '{label}' (no sv_spectra.npy found)")
            continue
        panels.append(loaded)
    if not panels:
        print("  SKIP knee_landscape_set: no panels could be loaded")
        return False
    knees = [(_knee_index_grid(p[3]), p) for p in panels]
    if not any(k is not None and np.isfinite(k).any() for k, _ in knees):
        print("  SKIP knee_landscape_set: no valid knee data")
        return False
    n = len(panels)
    # Per-panel color scale — the harmonic algebra closes at index ~15 while
    # singular ones cliff near 116, so a shared scale washes one out.
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 5.8), facecolor="white")
    if n == 1:
        axes = [axes]
    cmap_bad = cm.viridis.copy()
    cmap_bad.set_bad(color=INVALID_COLOR)
    for ax, (knee, (label, mu, phi, sv)) in zip(axes, knees):
        if knee is None or not np.isfinite(knee).any():
            ax.text(0.5, 0.5, "no knee data", transform=ax.transAxes,
                    ha="center", va="center", color="#7a8090")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontsize=11)
            continue
        finite = knee[np.isfinite(knee)]
        vmin = float(np.percentile(finite, 5))
        vmax = float(np.percentile(finite, 95))
        if vmax - vmin < 1:
            vmax = vmin + 1
        masked = np.ma.masked_invalid(knee)
        im = ax.pcolormesh(phi * 180 / np.pi, mu, masked,
                           cmap=cmap_bad, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlim(float(phi.min()) * 180 / np.pi, float(phi.max()) * 180 / np.pi)
        ax.set_ylim(float(mu.min()), float(mu.max()))
        ax.set_xlabel(r"$\phi$ (degrees)", fontsize=10)
        ax.set_ylabel(r"$\mu$", fontsize=10)
        ax.set_title(f"{label}  (knee {int(round(np.nanmedian(knee)))})", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("knee index", fontsize=9)
    fig.suptitle("Spectral knee landscape -- where the SV cliff occurs across the shape sphere",
                 fontsize=14, fontweight="bold", y=1.04)
    _save(fig, out)
    return True


# ─── registry ───

COMPARISONS = {
    "singular_vs_harmonic": cmp_singular_vs_harmonic,
    "irrational_exponents": cmp_irrational_exponents,
    "exponent_continuity":  cmp_exponent_continuity,
    "mass_ratio_extremes":  cmp_mass_extremes,
    "charge_departure":     cmp_charge_departure,
    "quantum_plus_one":     cmp_quantum_plus_one,
    "neural_classes":       cmp_neural_classes,
    "tier_decomposition":   cmp_tier_decomposition,
    "gue_overlay":          cmp_gue_overlay,
    "knee_landscape_set":   cmp_knee_landscape_set,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=list(COMPARISONS.keys()))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    keys = args.only or list(COMPARISONS.keys())
    n = 0
    for k in keys:
        print(f"  rendering {k}...")
        try:
            wrote = COMPARISONS[k](args.force)
            if wrote:
                n += 1
        except Exception as exc:
            print(f"  ERROR {k}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"Done: {n} new comparison figure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
