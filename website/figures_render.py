#!/usr/bin/env python3
"""Canonical figure renderer for the Figures page.

Walks every atlas configuration that has NPY data and emits one canonical view
per analysis type into figures_v2/. Reproducible: fixed colormaps, fixed colorbar
ranges per analysis type, fixed DPI. Resumable: skips outputs already present.

Sources:
  aws_results/atlas_full/<config>/    -- 100x100 named-system + power-law atlases
  atlas_output_hires/<config>/        -- 100x100 hires sweeps including eps subdirs
  atlas_targeted/<config>/<region>/   -- 50x50 zoomed regions per (config, region)

Outputs (under figures_v2/):
  heatmaps/<id>.png         -- mu/phi gap-ratio heatmap with isosceles overlay
  spheres/<id>.png          -- 3D shape sphere
  triptychs/<id>_vs_baseline.png -- 1/r baseline | this | difference
  spectra/<id>_sv.png       -- mean SV spectrum (when sv_spectra.npy present)

Run from the repository root:
    python website/figures_render.py
    python website/figures_render.py --force
    python website/figures_render.py --only aws_full
    python website/figures_render.py --config 1r3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures_v2")

AWS_FULL = os.path.join(ROOT, "aws_results", "atlas_full")
HIRES = os.path.join(ROOT, "atlas_output_hires")
TARGETED = os.path.join(ROOT, "atlas_targeted")

BASELINE_AWS = "1r_q+1_+1_+1"   # equal-mass equal-charge 1/r in aws_results/atlas_full

DPI = 150
CMAP = cm.inferno
DIFF_CMAP = cm.RdBu_r
INVALID_COLOR = "#2a3040"   # the "no data" color for masked cells (matches site border-bright)


# ─── Geometry helpers ───

def mu_phi_to_sphere(mu, phi):
    """Map shape coordinates (mu, phi) to a unit sphere (s1, s2, s3)."""
    w2_sq = mu ** 2 - mu * np.cos(phi) + 0.25
    N = 1.0 + w2_sq
    return (
        (1.0 - w2_sq) / N,
        2.0 * (mu * np.cos(phi) - 0.5) / N,
        2.0 * mu * np.sin(phi) / N,
    )


def interpolate(mu, phi, data, n_fine=200):
    # Cubic interpolator requires fully finite inputs. Substitute the column
    # mean for any NaN so cubic spline construction succeeds, then restore
    # NaNs over the regions that were originally invalid by interpolating a
    # validity mask in parallel.
    has_nan = not np.isfinite(data).all()
    if has_nan:
        data_filled = np.where(np.isfinite(data), data, np.nanmean(data))
        mask = np.isfinite(data).astype(float)
    else:
        data_filled = data
        mask = None
    interp = RegularGridInterpolator(
        (mu, phi), data_filled, method="cubic", bounds_error=False, fill_value=None
    )
    mu_f = np.linspace(mu[0], mu[-1], n_fine)
    phi_f = np.linspace(phi[0], phi[-1], n_fine)
    MU, PHI = np.meshgrid(mu_f, phi_f, indexing="ij")
    pts = np.stack([MU.ravel(), PHI.ravel()], axis=-1)
    out = interp(pts).reshape(MU.shape)
    if mask is not None:
        m_interp = RegularGridInterpolator(
            (mu, phi), mask, method="linear", bounds_error=False, fill_value=0.0,
        )
        m_out = m_interp(pts).reshape(MU.shape)
        out = np.where(m_out >= 0.5, out, np.nan)
    return MU, PHI, out


def draw_isosceles_curves(ax, mu_range, phi_range_rad):
    """Overlay the three isosceles curves on a (phi-deg, mu) heatmap axis."""
    phi_d = np.linspace(phi_range_rad[0], phi_range_rad[1], 600)
    mu_lo, mu_hi = mu_range
    ax.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.4)
    mu_c2 = 2 * np.cos(phi_d)
    m2 = (mu_c2 >= mu_lo) & (mu_c2 <= mu_hi)
    if m2.any():
        ax.plot(np.degrees(phi_d[m2]), mu_c2[m2],
                color="white", lw=0.7, ls="--", alpha=0.4)
    cos_p = np.cos(phi_d)
    safe = cos_p > 0.01
    mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
    m3 = np.isfinite(mu_c3) & (mu_c3 >= mu_lo) & (mu_c3 <= mu_hi)
    if m3.any():
        ax.plot(np.degrees(phi_d[m3]), mu_c3[m3],
                color="white", lw=0.7, ls="--", alpha=0.4)


# ─── Configuration loaders ───

class AtlasData:
    def __init__(self, mu, phi, rank, gap, label, summary=None, source=""):
        self.mu = np.asarray(mu)
        self.phi = np.asarray(phi)
        self.rank = np.asarray(rank) if rank is not None else None
        self.gap = np.asarray(gap)
        self.label = label
        self.summary = summary or {}
        self.source = source


def _load_npy(d, name):
    p = os.path.join(d, name)
    if os.path.exists(p):
        return np.load(p)
    return None


def load_aws_full(name) -> Optional[AtlasData]:
    d = os.path.join(AWS_FULL, name)
    if not os.path.isdir(d):
        return None
    mu = _load_npy(d, "mu_vals.npy")
    phi = _load_npy(d, "phi_vals.npy")
    gap = _load_npy(d, "gap_map.npy")
    rank = _load_npy(d, "rank_map.npy")
    if mu is None or phi is None or gap is None:
        return None
    cfg = {}
    summary = {}
    cp = os.path.join(d, "config.json")
    sp = os.path.join(d, "summary.json")
    if os.path.exists(cp):
        with open(cp) as f:
            cfg = json.load(f)
    if os.path.exists(sp):
        with open(sp) as f:
            summary = json.load(f)
    label = cfg.get("label", name)
    return AtlasData(mu, phi, rank, gap, label, summary,
                     source=f"aws_results/atlas_full/{name}")


def load_hires_eps(parent, eps_dir) -> Optional[AtlasData]:
    d = os.path.join(parent, eps_dir)
    if not os.path.isdir(d):
        return None
    mu = _load_npy(d, "mu_vals.npy")
    phi = _load_npy(d, "phi_vals.npy")
    gap = _load_npy(d, "gap_map.npy")
    rank = _load_npy(d, "rank_map.npy")
    if mu is None or phi is None or gap is None:
        return None
    cfg = {}
    cp = os.path.join(d, "config.json")
    if os.path.exists(cp):
        with open(cp) as f:
            cfg = json.load(f)
    pot = cfg.get("potential", os.path.basename(parent))
    eps = cfg.get("epsilon", eps_dir.replace("eps_", ""))
    label = f"{pot} (eps={eps})"
    return AtlasData(mu, phi, rank, gap, label, source=f"atlas_output_hires/{os.path.basename(parent)}/{eps_dir}")


def load_targeted_region(config_dir, region) -> Optional[AtlasData]:
    d = os.path.join(TARGETED, config_dir, region)
    if not os.path.isdir(d):
        return None
    mu = _load_npy(d, "mu_vals.npy")
    phi = _load_npy(d, "phi_vals.npy")
    # gap_map is the raw sigma_k / sigma_{k+1} ratio (orders of magnitude).
    # gap_score_map is a normalized log-tier score (~16-19) used as a
    # tier classifier internally. We always render the raw gap ratio.
    gap = _load_npy(d, "gap_map.npy")
    rank = _load_npy(d, "rank_map.npy")
    if mu is None or phi is None or gap is None:
        return None
    cfg = {}
    cp = os.path.join(d, "config.json")
    if os.path.exists(cp):
        with open(cp) as f:
            cfg = json.load(f)
    pot = cfg.get("potential", config_dir)
    label = f"{pot}  {region.replace('_', ' ').title()}"
    return AtlasData(mu, phi, rank, gap, label, source=f"atlas_targeted/{config_dir}/{region}")


# ─── Core renderers ───

def _validity_mask(rank, gap):
    """Cells with no valid measurement: rank == -1 or gap == 0 or non-finite."""
    invalid = ~np.isfinite(gap) | (gap <= 0)
    if rank is not None:
        invalid = invalid | (rank == -1)
    return ~invalid


def _figsize_for(mu, phi, cell_inch=0.13, min_w=5.0, min_h=2.6,
                 max_w=14.0, max_h=10.0, cb_pad=1.6):
    """Pick a figure size that's roughly proportional to the grid shape.

    Each cell gets ~cell_inch * cell_inch of canvas (so 50x50 grids end up
    around 6.5x6.5 inches). We then add cb_pad inches to width for the
    colorbar, and clip to sensible bounds so very thin grids (10x80) and
    very large grids stay readable.

    For very rectangular grids (e.g. 10x80 isosceles ridge) the height is
    intentionally allowed to be small (min_h ~ 2.6") so individual cells
    don't get stretched into tall rectangles. The label/title overhead
    (~0.8") is added to height regardless.
    """
    n_mu = len(mu) if hasattr(mu, "__len__") else 0
    n_phi = len(phi) if hasattr(phi, "__len__") else 0
    if n_mu < 2 or n_phi < 2:
        return (8.5, 6.5)
    w = float(np.clip(n_phi * cell_inch + cb_pad, min_w, max_w))
    h = float(np.clip(n_mu * cell_inch + 0.9, min_h, max_h))
    return (round(w, 1), round(h, 1))


def _apply_axes_aspect(ax, mu, phi):
    """No-op kept for call-site compatibility. We rely on _figsize_for to
    drive cell aspect; using ax.set_aspect with extreme grid ratios collapses
    the data axes."""
    return None


def _log_gap_safe(gap, valid):
    """log10(gap) over valid cells; positive log floor only for cells where
    gap < 1 (those are real sub-decade signal so we just clamp at 1e-3
    rather than a hard 1.0 that crushes them to a single bin)."""
    g = np.where(valid, gap, np.nan)
    g = np.where(g <= 0, np.nan, g)
    g = np.where(g < 1e-3, 1e-3, g)
    return np.log10(g)


def render_heatmap(data: AtlasData, out_path: str, title: Optional[str] = None) -> str:
    valid = _validity_mask(data.rank, data.gap)
    if not valid.any():
        # Render a placeholder card so the figure exists but reads as N/A.
        return _render_na_card(out_path, title or data.label,
                                "No valid samples (every cell failed: rank == -1 / gap == 0).")
    log_gap = _log_gap_safe(data.gap, valid)
    finite_log = log_gap[np.isfinite(log_gap)]
    # Tighter percentiles (2/98) give better dynamic range when the data
    # has a few outliers, which is common in targeted scans.
    vmin = float(np.percentile(finite_log, 2))
    vmax = float(np.percentile(finite_log, 98))
    if vmax - vmin < 0.5:
        # Avoid a near-flat colorbar collapse but use enough range that
        # near-uniform regions don't all paint as the lowest color.
        mid = 0.5 * (vmin + vmax)
        vmin, vmax = mid - 0.5, mid + 0.5

    fs = _figsize_for(data.mu, data.phi)
    fig, ax = plt.subplots(figsize=fs, facecolor="white")
    cmap_with_bad = CMAP.copy() if hasattr(CMAP, "copy") else cm.get_cmap(CMAP.name)
    cmap_with_bad.set_bad(color=INVALID_COLOR)
    masked = np.ma.masked_invalid(log_gap)
    im = ax.pcolormesh(
        data.phi * 180 / np.pi, data.mu, masked,
        cmap=cmap_with_bad, vmin=vmin, vmax=vmax, shading="auto",
    )
    draw_isosceles_curves(
        ax,
        (float(data.mu.min()), float(data.mu.max())),
        (float(data.phi.min()), float(data.phi.max())),
    )
    ax.set_xlim(float(data.phi.min()) * 180 / np.pi, float(data.phi.max()) * 180 / np.pi)
    ax.set_ylim(float(data.mu.min()), float(data.mu.max()))
    _apply_axes_aspect(ax, data.mu, data.phi)
    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=11)
    ax.set_ylabel(r"$\mu$", fontsize=11)
    n_invalid = int((~valid).sum())
    extra = f"  ({n_invalid} of {valid.size} cells unmeasured)" if n_invalid else ""
    ax.set_title((title or data.label) + extra, fontsize=12)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$(gap ratio)", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _render_na_card(out_path: str, title: str, message: str) -> str:
    """Solid card for configurations where no data could be plotted."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor="white")
    ax.set_facecolor(INVALID_COLOR)
    ax.text(0.5, 0.55, title, ha="center", va="center",
            transform=ax.transAxes, fontsize=15, fontweight="bold", color="white")
    ax.text(0.5, 0.42, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="#c8ccd4", wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1e2430")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_sphere(data: AtlasData, out_path: str, title: Optional[str] = None) -> str:
    valid_grid = _validity_mask(data.rank, data.gap)
    if not valid_grid.any():
        return _render_na_card(out_path, title or data.label,
                                "No valid samples for shape sphere.")
    log_gap = _log_gap_safe(data.gap, valid_grid)
    MU, PHI, LG = interpolate(data.mu, data.phi, log_gap)
    valid = np.isfinite(LG)
    if not valid.any():
        return _render_na_card(out_path, title or data.label,
                                "Interpolation produced no finite values.")
    vmin = float(np.nanpercentile(LG[valid], 2))
    vmax = float(np.nanpercentile(LG[valid], 98))
    if vmax - vmin < 0.5:
        mid = 0.5 * (vmin + vmax)
        vmin, vmax = mid - 0.5, mid + 0.5
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    S1, S2, S3 = mu_phi_to_sphere(MU, PHI)
    FC = CMAP(norm(np.where(valid, LG, vmin)))

    fig = plt.figure(figsize=(8.5, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="lightsteelblue", alpha=0.08, shade=False, antialiased=False,
    )
    ax.plot_surface(S1, S2, S3, facecolors=FC,
                    rstride=1, cstride=1, shade=False, antialiased=True, alpha=0.95)
    FCm = FC.copy(); FCm[:, :, 3] = 0.5
    ax.plot_surface(S1, S2, -S3, facecolors=FCm,
                    rstride=1, cstride=1, shade=False, antialiased=True)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
            "k-", alpha=0.25, linewidth=0.8)
    ax.set_xlabel("$s_1$", fontsize=9)
    ax.set_ylabel("$s_2$", fontsize=9)
    ax.set_zlabel("$s_3$", fontsize=9)
    ax.view_init(elev=30, azim=-55)
    ax.set_title(title or data.label, fontsize=13)
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08)
    cb.set_label(r"$\log_{10}$(gap ratio)", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_triptych(data: AtlasData, baseline: AtlasData, out_path: str,
                    title: Optional[str] = None) -> str:
    valid_self = _validity_mask(data.rank, data.gap)
    valid_base = _validity_mask(baseline.rank, baseline.gap)
    if not valid_self.any():
        return _render_na_card(out_path, title or data.label,
                                "No valid samples for triptych.")
    log_self = _log_gap_safe(data.gap, valid_self)
    log_base = _log_gap_safe(baseline.gap, valid_base)
    if log_self.shape != log_base.shape:
        interp = RegularGridInterpolator(
            (baseline.mu, baseline.phi), log_base, method="linear",
            bounds_error=False, fill_value=np.nan,
        )
        MU_g, PHI_g = np.meshgrid(data.mu, data.phi, indexing="ij")
        log_base_resampled = interp(
            np.stack([MU_g.ravel(), PHI_g.ravel()], axis=-1)
        ).reshape(log_self.shape)
        plot_base = log_base_resampled
    else:
        plot_base = log_base
    diff = log_self - plot_base

    valid_a = np.isfinite(log_self)
    valid_b = np.isfinite(plot_base)
    valid_d = np.isfinite(diff)

    pool = np.concatenate([log_self[valid_a].ravel(), plot_base[valid_b].ravel()])
    vmin = float(np.nanpercentile(pool, 2))
    vmax = float(np.nanpercentile(pool, 98))
    if vmax - vmin < 0.5:
        mid = 0.5 * (vmin + vmax)
        vmin, vmax = mid - 0.5, mid + 0.5
    dlim = float(np.nanpercentile(np.abs(diff[valid_d]), 99)) if valid_d.any() else 1.0
    dlim = max(dlim, 0.5)

    cmap_with_bad = CMAP.copy() if hasattr(CMAP, "copy") else cm.get_cmap(CMAP.name)
    cmap_with_bad.set_bad(color=INVALID_COLOR)
    diff_cmap_bad = DIFF_CMAP.copy() if hasattr(DIFF_CMAP, "copy") else cm.get_cmap(DIFF_CMAP.name)
    diff_cmap_bad.set_bad(color=INVALID_COLOR)

    base_w, base_h = _figsize_for(data.mu, data.phi)
    fig, axes = plt.subplots(1, 3, figsize=(base_w * 2.6, base_h), facecolor="white")
    panels = [
        (axes[0], plot_base, baseline.label, cmap_with_bad, vmin, vmax, r"$\log_{10}$(gap)"),
        (axes[1], log_self, data.label, cmap_with_bad, vmin, vmax, r"$\log_{10}$(gap)"),
        (axes[2], diff, "difference", diff_cmap_bad, -dlim, dlim, r"$\Delta\;\log_{10}$(gap)"),
    ]
    for ax, arr, ttl, cmap, vlo, vhi, lbl in panels:
        im = ax.pcolormesh(
            data.phi * 180 / np.pi, data.mu, np.ma.masked_invalid(arr),
            cmap=cmap, vmin=vlo, vmax=vhi, shading="auto",
        )
        draw_isosceles_curves(
            ax,
            (float(data.mu.min()), float(data.mu.max())),
            (float(data.phi.min()), float(data.phi.max())),
        )
        ax.set_xlim(float(data.phi.min()) * 180 / np.pi, float(data.phi.max()) * 180 / np.pi)
        ax.set_ylim(float(data.mu.min()), float(data.mu.max()))
        _apply_axes_aspect(ax, data.mu, data.phi)
        ax.set_xlabel(r"$\phi$ (degrees)", fontsize=11)
        ax.set_ylabel(r"$\mu$", fontsize=11)
        ax.set_title(ttl, fontsize=12)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(lbl, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_sv_spectrum(sv_path: str, out_path: str, title: str) -> str:
    if not os.path.exists(sv_path):
        return ""
    sv = np.load(sv_path)
    # sv shape: (n_mu, n_phi, n_sv) ideally
    if sv.ndim < 2:
        return ""
    flat = sv.reshape(-1, sv.shape[-1])
    finite = np.isfinite(flat).all(axis=1)
    flat = flat[finite]
    if flat.shape[0] == 0:
        return ""
    flat = np.where(flat > 0, flat, np.nan)
    mean = np.nanmean(flat, axis=0)
    p10 = np.nanpercentile(flat, 10, axis=0)
    p90 = np.nanpercentile(flat, 90, axis=0)
    idx = np.arange(1, mean.size + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.fill_between(idx, p10, p90, alpha=0.2, color="C0", label="10th-90th pctl")
    ax.plot(idx, mean, "C0-", lw=1.6, label="mean")
    ax.set_yscale("log")
    ax.set_xlabel("singular value index", fontsize=11)
    ax.set_ylabel("singular value", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ─── Drivers ───

def needs_render(out_path: str, force: bool) -> bool:
    return force or not os.path.exists(out_path)


def render_aws_full(force: bool, only: Optional[set[str]] = None,
                    only_config: Optional[str] = None) -> int:
    if not os.path.isdir(AWS_FULL):
        return 0
    if only is not None and "aws_full" not in only:
        return 0
    baseline = load_aws_full(BASELINE_AWS)
    if baseline is None:
        print(f"  WARN: baseline {BASELINE_AWS} not found; triptychs skipped")
    n = 0
    configs = sorted(os.listdir(AWS_FULL))
    for name in configs:
        if only_config and name != only_config:
            continue
        d = os.path.join(AWS_FULL, name)
        if not os.path.isdir(d):
            continue
        if not os.path.exists(os.path.join(d, "rank_map.npy")):
            continue
        data = load_aws_full(name)
        if data is None:
            continue
        sid = name
        h = os.path.join(OUT, "heatmaps", f"awsfull__{sid}.png")
        s = os.path.join(OUT, "spheres",  f"awsfull__{sid}.png")
        t = os.path.join(OUT, "triptychs", f"awsfull__{sid}_vs_baseline.png")
        try:
            if needs_render(h, force):
                render_heatmap(data, h)
                n += 1
            if needs_render(s, force):
                render_sphere(data, s)
                n += 1
            if baseline and name != BASELINE_AWS and needs_render(t, force):
                render_triptych(data, baseline, t,
                                title=f"{data.label}  vs  {baseline.label} (1/r baseline)")
                n += 1
        except Exception as exc:
            print(f"  ERROR awsfull/{name}: {exc}")
    print(f"  aws_full: {n} new figures")
    return n


def render_hires(force: bool, only: Optional[set[str]] = None,
                 only_config: Optional[str] = None) -> int:
    if not os.path.isdir(HIRES):
        return 0
    if only is not None and "hires" not in only:
        return 0
    n = 0
    for parent in sorted(os.listdir(HIRES)):
        if only_config and parent != only_config:
            continue
        pdir = os.path.join(HIRES, parent)
        if not os.path.isdir(pdir):
            continue
        # Top-level (no eps subdir) or each eps_X subdir
        targets = []
        if os.path.exists(os.path.join(pdir, "rank_map.npy")):
            targets.append((pdir, parent))
        for sub in sorted(os.listdir(pdir)):
            sd = os.path.join(pdir, sub)
            if os.path.isdir(sd) and (sub.startswith("eps_") or sub == "adaptive"):
                if os.path.exists(os.path.join(sd, "rank_map.npy")):
                    targets.append((sd, f"{parent}__{sub}"))
        for d, sid in targets:
            mu = _load_npy(d, "mu_vals.npy")
            phi = _load_npy(d, "phi_vals.npy")
            gap = _load_npy(d, "gap_map.npy")
            rank = _load_npy(d, "rank_map.npy")
            if mu is None or phi is None or gap is None:
                continue
            cfg = {}
            cp = os.path.join(d, "config.json")
            if os.path.exists(cp):
                with open(cp) as f:
                    cfg = json.load(f)
            pot = cfg.get("potential", parent)
            eps = cfg.get("epsilon", "")
            label = f"{pot}" + (f"  eps={eps}" if eps else "")
            data = AtlasData(mu, phi, rank, gap, label,
                             source=os.path.relpath(d, ROOT).replace("\\", "/"))
            h = os.path.join(OUT, "heatmaps", f"hires__{sid}.png")
            s = os.path.join(OUT, "spheres",  f"hires__{sid}.png")
            try:
                if needs_render(h, force):
                    render_heatmap(data, h)
                    n += 1
                if needs_render(s, force):
                    render_sphere(data, s)
                    n += 1
            except Exception as exc:
                print(f"  ERROR hires/{sid}: {exc}")
            sv = os.path.join(d, "sv_spectra.npy")
            if os.path.exists(sv):
                sp = os.path.join(OUT, "spectra", f"hires__{sid}_sv.png")
                if needs_render(sp, force):
                    try:
                        render_sv_spectrum(sv, sp, f"SV spectrum: {label}")
                        n += 1
                    except Exception as exc:
                        print(f"  ERROR sv hires/{sid}: {exc}")
    print(f"  hires: {n} new figures")
    return n


def render_targeted(force: bool, only: Optional[set[str]] = None,
                    only_config: Optional[str] = None) -> int:
    if not os.path.isdir(TARGETED):
        return 0
    if only is not None and "targeted" not in only:
        return 0
    n = 0
    REGIONS = ["lagrange", "euler_strip", "isosceles_ridge",
               "small_mu", "tier_cluster", "charge_hotspot"]
    for cfg_name in sorted(os.listdir(TARGETED)):
        if only_config and cfg_name != only_config:
            continue
        cd = os.path.join(TARGETED, cfg_name)
        if not os.path.isdir(cd):
            continue
        for region in REGIONS:
            data = load_targeted_region(cfg_name, region)
            if data is None:
                continue
            sid = f"targeted__{cfg_name}__{region}"
            h = os.path.join(OUT, "heatmaps", f"{sid}.png")
            try:
                if needs_render(h, force):
                    render_heatmap(data, h, title=data.label)
                    n += 1
            except Exception as exc:
                print(f"  ERROR targeted/{cfg_name}/{region}: {exc}")
            sv = os.path.join(cd, region, "sv_spectra.npy")
            if os.path.exists(sv):
                sp = os.path.join(OUT, "spectra", f"{sid}_sv.png")
                if needs_render(sp, force):
                    try:
                        render_sv_spectrum(sv, sp, f"SV spectrum: {data.label}")
                        n += 1
                    except Exception as exc:
                        print(f"  ERROR sv targeted/{cfg_name}/{region}: {exc}")
    print(f"  targeted: {n} new figures")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-render even if output exists")
    ap.add_argument("--only", choices=["aws_full", "hires", "targeted"],
                    nargs="*", default=None,
                    help="restrict to one or more sources")
    ap.add_argument("--config", default=None,
                    help="restrict to a single config name (matches subdirectory)")
    args = ap.parse_args()
    only = set(args.only) if args.only else None
    os.makedirs(OUT, exist_ok=True)
    print("Rendering canonical figures into figures_v2/ ...")
    total = 0
    total += render_aws_full(args.force, only, args.config)
    total += render_hires(args.force, only, args.config)
    total += render_targeted(args.force, only, args.config)
    print(f"Done: {total} new figure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
