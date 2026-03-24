#!/usr/bin/env python3
"""
Atlas Assembly: Comprehensive Shape-Sphere Atlas Visualization
================================================================

Loads all computed atlas data (full-sphere multi-epsilon and targeted
adaptive regional scans) and produces publication-quality figures.

Output: atlas_figures/

Usage:
    python assemble_atlases.py           # generate all figures
    python assemble_atlases.py --only targeted   # only targeted composites
    python assemble_atlases.py --only fullsphere # only full-sphere pages
    python assemble_atlases.py --only comparison # only cross-potential / charge comparisons
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.ticker as ticker

TARGETED_DIR = "atlas_targeted"
HIRES_DIR = "atlas_output_hires"
OUTPUT_DIR = "atlas_figures"

POTENTIAL_LABELS = {
    "1_r":  r"$1/r$ (Newtonian / Coulomb)",
    "1_r2": r"$1/r^2$ (Calogero-Moser)",
    "1_r3": r"$1/r^3$",
    "log":  r"$\log(r)$ (2D Vortices)",
    "harmonic": r"$r^2$ (Harmonic)",
}

CHARGE_LABELS = {
    "coulomb_+1_+1_+1": r"Coulomb $(+1,+1,+1)$ — Penning trap",
    "coulomb_+1_+1_-1": r"Coulomb $(+1,+1,-1)$ — $\mathrm{H}_2^+$",
    "coulomb_+1_-1_-1": r"Coulomb $(+1,-1,-1)$ — $\mathrm{Ps}^- / \mathrm{H}^-$",
    "coulomb_+2_-1_-1": r"Coulomb $(+2,-1,-1)$ — Helium",
    "coulomb_+3_-1_-1": r"Coulomb $(+3,-1,-1)$ — $\mathrm{Li}^+$",
    "coulomb_1_r2_+2_-1_-1": r"$1/r^2$ Coulomb $(+2,-1,-1)$",
    "coulomb_1_r3_+2_-1_-1": r"$1/r^3$ Coulomb $(+2,-1,-1)$",
}

REGION_LABELS = {
    "lagrange":        "Lagrange equilateral",
    "euler_strip":     "Euler collinear strip",
    "charge_hotspot":  "Charge-sensitivity hotspot",
    "isosceles_ridge": "Isosceles ridge",
    "small_mu":        "Small-\u03bc region",
    "tier_cluster":    "Tier cluster",
}

REFERENCE_POTENTIALS = {"1_r", "1_r2", "1_r3", "log"}

CHARGE_TO_REFERENCE = {
    "coulomb_+1_+1_+1": "1_r",
    "coulomb_+1_+1_-1": "1_r",
    "coulomb_+1_-1_-1": "1_r",
    "coulomb_+2_-1_-1": "1_r",
    "coulomb_+3_-1_-1": "1_r",
    "coulomb_1_r2_+2_-1_-1": "1_r2",
    "coulomb_1_r3_+2_-1_-1": "1_r3",
}


def _dir_label(dirname):
    if dirname in POTENTIAL_LABELS:
        return POTENTIAL_LABELS[dirname]
    if dirname in CHARGE_LABELS:
        return CHARGE_LABELS[dirname]
    return dirname


def _safe_filename(dirname):
    return dirname.replace("+", "p").replace("-", "m").replace("/", "_")


def _draw_isosceles(ax, mu_range, phi_range_rad):
    """Overlay isosceles triangle curves on a (phi_deg, mu) plot."""
    phi_d = np.linspace(phi_range_rad[0], phi_range_rad[1], 500)
    mu_lo, mu_hi = mu_range

    ax.axhline(1.0, color='cyan', lw=0.8, ls='--', alpha=0.5,
               label=r'$\mu=1$ (equilateral)')

    mu_c2 = 2 * np.cos(phi_d)
    m2 = (mu_c2 >= mu_lo) & (mu_c2 <= mu_hi)
    if m2.any():
        ax.plot(np.degrees(phi_d[m2]), mu_c2[m2],
                color='lime', lw=0.8, ls='--', alpha=0.5,
                label=r'$\mu = 2\cos\phi$ (isosceles)')

    cos_p = np.cos(phi_d)
    safe = cos_p > 0.01
    mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
    m3 = np.isfinite(mu_c3) & (mu_c3 >= mu_lo) & (mu_c3 <= mu_hi)
    if m3.any():
        ax.plot(np.degrees(phi_d[m3]), mu_c3[m3],
                color='magenta', lw=0.8, ls='--', alpha=0.5,
                label=r'$\mu = 1/(2\cos\phi)$')


def _load_targeted_region(base_dir, region_name):
    """Load a single targeted region's data. Returns dict or None."""
    rdir = os.path.join(base_dir, region_name)
    rank_path = os.path.join(rdir, "rank_map.npy")
    if not os.path.isfile(rank_path):
        return None

    data = {}
    for name in ["rank_map", "gap_score_map", "optimal_eps_map",
                  "tier_map", "mu_vals", "phi_vals", "gap_map",
                  "n_tested_map", "sv_spectra"]:
        p = os.path.join(rdir, f"{name}.npy")
        if os.path.isfile(p):
            data[name] = np.load(p)

    status_path = os.path.join(rdir, "status.json")
    if os.path.isfile(status_path):
        with open(status_path) as f:
            data["status"] = json.load(f)

    return data


def _load_hires(potential_slug):
    """Load full-sphere 100x100 data for a potential."""
    d = os.path.join(HIRES_DIR, potential_slug)
    if not os.path.isdir(d):
        return None
    data = {}
    for name in ["rank_map", "gap_map", "mu_vals", "phi_vals", "sv_spectra"]:
        p = os.path.join(d, f"{name}.npy")
        if os.path.isfile(p):
            data[name] = np.load(p)
    if "rank_map" not in data:
        return None

    eps_dirs = sorted([e for e in os.listdir(d)
                       if os.path.isdir(os.path.join(d, e))
                       and e.startswith("eps_")])
    data["epsilon_dirs"] = eps_dirs
    data["epsilon_data"] = {}
    for ed in eps_dirs:
        eps_data = {}
        for name in ["rank_map", "gap_map", "mu_vals", "phi_vals"]:
            p = os.path.join(d, ed, f"{name}.npy")
            if os.path.isfile(p):
                eps_data[name] = np.load(p)
        if eps_data:
            data["epsilon_data"][ed] = eps_data

    return data


# ─────────────────────────────────────────────────────────────────────
# A. Full-sphere atlas pages
# ─────────────────────────────────────────────────────────────────────
def generate_fullsphere_pages():
    """One page per potential from atlas_output_hires/."""
    potentials = sorted([d for d in os.listdir(HIRES_DIR)
                         if os.path.isdir(os.path.join(HIRES_DIR, d))
                         and d != "multi_epsilon"])
    if not potentials:
        print("  No full-sphere data found.")
        return

    for pot in potentials:
        data = _load_hires(pot)
        if data is None:
            continue

        label = POTENTIAL_LABELS.get(pot, pot)
        rank = data["rank_map"]
        gap = data.get("gap_map")
        mu = data["mu_vals"]
        phi = data["phi_vals"]
        phi_deg = np.degrees(phi)

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(f"Shape Sphere Atlas — {label}\n"
                     f"100×100 grid, $\\mu \\in [{mu[0]:.1f}, {mu[-1]:.1f}]$, "
                     f"$\\phi \\in [{phi_deg[0]:.0f}°, {phi_deg[-1]:.0f}°]$",
                     fontsize=14, fontweight='bold')

        n_eps = len(data.get("epsilon_data", {}))
        if n_eps > 0:
            gs = GridSpec(2, max(3, n_eps), figure=fig,
                          hspace=0.35, wspace=0.3)
        else:
            gs = GridSpec(1, 2, figure=fig, hspace=0.35, wspace=0.3)

        ax_rank = fig.add_subplot(gs[0, 0])
        rank_lo = max(rank.min() - 1, 112)
        rank_hi = max(rank.max() + 1, 120)
        im = ax_rank.pcolormesh(phi_deg, mu, rank, cmap='RdYlGn',
                                shading='auto', vmin=rank_lo, vmax=rank_hi)
        ax_rank.set_title(f"SVD Rank [{rank.min()}–{rank.max()}]")
        ax_rank.set_xlabel("$\\phi$ (deg)")
        ax_rank.set_ylabel("$\\mu = r_{13}/r_{12}$")
        plt.colorbar(im, ax=ax_rank, label="rank")
        _draw_isosceles(ax_rank, (mu[0], mu[-1]), (phi[0], phi[-1]))

        if gap is not None:
            ax_gap = fig.add_subplot(gs[0, 1])
            log_gap = np.log10(np.where(gap > 0, gap, 1e-10))
            im2 = ax_gap.pcolormesh(phi_deg, mu, log_gap, cmap='inferno',
                                    shading='auto')
            ax_gap.set_title("log₁₀(SVD Gap Ratio)")
            ax_gap.set_xlabel("$\\phi$ (deg)")
            plt.colorbar(im2, ax=ax_gap, label="log₁₀(gap)")
            _draw_isosceles(ax_gap, (mu[0], mu[-1]), (phi[0], phi[-1]))

        if n_eps >= 2:
            ax_hist = fig.add_subplot(gs[0, 2])
            ax_hist.hist(rank.ravel(), bins=range(int(rank.min()),
                         int(rank.max()) + 2), edgecolor='black',
                         alpha=0.7, color='steelblue')
            ax_hist.set_xlabel("Rank")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title("Rank Distribution")

        eps_keys = sorted(data.get("epsilon_data", {}).keys())
        for i, ek in enumerate(eps_keys):
            if i >= gs.ncols:
                break
            ed = data["epsilon_data"][ek]
            if "rank_map" not in ed:
                continue

            ax_e = fig.add_subplot(gs[1, i])
            er = ed["rank_map"]
            em = ed.get("mu_vals", mu)
            ep = np.degrees(ed.get("phi_vals", phi))
            im_e = ax_e.pcolormesh(ep, em, er, cmap='RdYlGn',
                                   shading='auto', vmin=rank_lo, vmax=rank_hi)
            eps_label = ek.replace("eps_", "ε=").replace("_", "")
            ax_e.set_title(f"Rank at {eps_label}")
            ax_e.set_xlabel("$\\phi$ (deg)")
            if i == 0:
                ax_e.set_ylabel("$\\mu$")
            plt.colorbar(im_e, ax=ax_e)
            _draw_isosceles(ax_e, (em[0], em[-1]),
                            (np.radians(ep[0]), np.radians(ep[-1])))

        fname = os.path.join(OUTPUT_DIR, f"fullsphere_{_safe_filename(pot)}.png")
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ─────────────────────────────────────────────────────────────────────
# B. Targeted region composite figures
# ─────────────────────────────────────────────────────────────────────
def generate_targeted_composites():
    """One composite figure per potential/charge config, all 6 regions stitched."""
    configs = sorted([d for d in os.listdir(TARGETED_DIR)
                      if os.path.isdir(os.path.join(TARGETED_DIR, d))
                      and d != "plots" and d != "log_sync"])
    if not configs:
        print("  No targeted data found.")
        return

    regions = ["lagrange", "euler_strip", "charge_hotspot",
               "isosceles_ridge", "small_mu", "tier_cluster"]

    for cfg_dir in configs:
        base = os.path.join(TARGETED_DIR, cfg_dir)
        label = _dir_label(cfg_dir)

        available_regions = []
        for r in regions:
            d = _load_targeted_region(base, r)
            if d is not None:
                available_regions.append((r, d))

        if not available_regions:
            continue

        _make_stitched_figure(cfg_dir, label, available_regions)
        _make_region_details(cfg_dir, label, available_regions)

    print(f"  All targeted composites generated.")


def _make_stitched_figure(cfg_dir, label, region_data):
    """Stitch all regions onto a single (mu, phi) coordinate space."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f"Shape Sphere Atlas — {label}\n"
                 f"Targeted Adaptive Scan — {len(region_data)} regions",
                 fontsize=14, fontweight='bold')

    global_mu_min = min(d["mu_vals"].min() for _, d in region_data)
    global_mu_max = max(d["mu_vals"].max() for _, d in region_data)
    global_phi_min = min(np.degrees(d["phi_vals"]).min()
                         for _, d in region_data)
    global_phi_max = max(np.degrees(d["phi_vals"]).max()
                         for _, d in region_data)

    mu_pad = (global_mu_max - global_mu_min) * 0.05
    phi_pad = (global_phi_max - global_phi_min) * 0.02

    all_ranks = np.concatenate([d["rank_map"].ravel()
                                for _, d in region_data])
    rank_lo = max(all_ranks.min() - 1, 112)
    rank_hi = max(all_ranks.max() + 1, 118)

    titles = ["SVD Rank", "Gap Score",
              "Optimal log₁₀(ε)", "Number of Tiers"]
    cmaps = ["RdYlGn", "inferno", "viridis", "YlOrRd"]
    keys = ["rank_map", "gap_score_map", "optimal_eps_map", "tier_map"]

    for ax_idx, (ax, title, cmap, key) in enumerate(
            zip(axes.flat, titles, cmaps, keys)):
        ax.set_title(title, fontsize=12)

        for rname, rdata in region_data:
            mu = rdata["mu_vals"]
            phi_deg = np.degrees(rdata["phi_vals"])

            if key == "tier_map":
                if key in rdata and rdata[key].ndim == 4:
                    vals = np.sum(rdata[key][:, :, :, 0] >= 0, axis=2)
                else:
                    continue
            elif key == "optimal_eps_map":
                if key in rdata:
                    vals = np.log10(np.where(rdata[key] > 0,
                                             rdata[key], 1e-5))
                else:
                    continue
            elif key in rdata:
                vals = rdata[key]
            else:
                continue

            if key == "rank_map":
                im = ax.pcolormesh(phi_deg, mu, vals, cmap=cmap,
                                   shading='auto',
                                   vmin=rank_lo, vmax=rank_hi)
            else:
                im = ax.pcolormesh(phi_deg, mu, vals, cmap=cmap,
                                   shading='auto')

            rlabel = REGION_LABELS.get(rname, rname)
            ax.annotate(rlabel, xy=(phi_deg.mean(), mu.mean()),
                        fontsize=6, color='white', ha='center',
                        va='center', alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='black', alpha=0.4))

        _draw_isosceles(ax, (global_mu_min, global_mu_max),
                        (np.radians(global_phi_min),
                         np.radians(global_phi_max)))

        ax.set_xlim(global_phi_min - phi_pad, global_phi_max + phi_pad)
        ax.set_ylim(global_mu_min - mu_pad, global_mu_max + mu_pad)
        ax.set_xlabel("$\\phi$ (deg)")
        ax.set_ylabel("$\\mu = r_{13}/r_{12}$")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = os.path.join(OUTPUT_DIR,
                         f"targeted_composite_{_safe_filename(cfg_dir)}.png")
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def _make_region_details(cfg_dir, label, region_data):
    """Per-region detail panels: 6 rows x 4 cols."""
    n = len(region_data)
    fig, axes = plt.subplots(n, 4, figsize=(22, 4.5 * n))
    fig.suptitle(f"Targeted Region Details — {label}",
                 fontsize=14, fontweight='bold')

    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (rname, rdata) in enumerate(region_data):
        mu = rdata["mu_vals"]
        phi_deg = np.degrees(rdata["phi_vals"])
        mu_range = (mu[0], mu[-1])
        phi_range = (rdata["phi_vals"][0], rdata["phi_vals"][-1])

        rank = rdata["rank_map"]
        rank_lo = max(rank.min() - 1, 112)
        rank_hi = max(rank.max() + 1, 118)
        im0 = axes[row, 0].pcolormesh(phi_deg, mu, rank, cmap='RdYlGn',
                                       shading='auto',
                                       vmin=rank_lo, vmax=rank_hi)
        axes[row, 0].set_title(
            f"{REGION_LABELS.get(rname, rname)}\n"
            f"Rank [{rank.min()}–{rank.max()}]", fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0])

        if "gap_score_map" in rdata:
            gs = rdata["gap_score_map"]
            gs_lo = np.percentile(gs, 1)
            gs_hi = np.percentile(gs, 99)
            if gs_hi - gs_lo < 0.1:
                gs_lo, gs_hi = gs.min() - 0.05, gs.max() + 0.05
            im1 = axes[row, 1].pcolormesh(phi_deg, mu, gs, cmap='inferno',
                                           shading='auto',
                                           vmin=gs_lo, vmax=gs_hi)
            axes[row, 1].set_title("Gap Score", fontsize=10)
            plt.colorbar(im1, ax=axes[row, 1])

        if "optimal_eps_map" in rdata:
            oe = rdata["optimal_eps_map"]
            log_eps = np.log10(np.where(oe > 0, oe, 1e-5))
            eps_lo = np.percentile(log_eps, 1)
            eps_hi = np.percentile(log_eps, 99)
            if eps_hi - eps_lo < 0.1:
                eps_lo, eps_hi = log_eps.min() - 0.05, log_eps.max() + 0.05
            im2 = axes[row, 2].pcolormesh(phi_deg, mu, log_eps,
                                           cmap='viridis', shading='auto',
                                           vmin=eps_lo, vmax=eps_hi)
            axes[row, 2].set_title("log₁₀(Optimal ε)", fontsize=10)
            plt.colorbar(im2, ax=axes[row, 2])

        if "tier_map" in rdata and rdata["tier_map"].ndim == 4:
            n_tiers = np.sum(rdata["tier_map"][:, :, :, 0] >= 0, axis=2)
            tier_hi = max(n_tiers.max(), 4)
            im3 = axes[row, 3].pcolormesh(phi_deg, mu, n_tiers,
                                           cmap='YlOrRd', shading='auto',
                                           vmin=1, vmax=tier_hi)
            axes[row, 3].set_title(
                f"Tiers [1–{n_tiers.max()}]", fontsize=10)
            plt.colorbar(im3, ax=axes[row, 3])

        for c in range(4):
            _draw_isosceles(axes[row, c], mu_range, phi_range)
            axes[row, c].set_xlim(phi_deg[0], phi_deg[-1])
            axes[row, c].set_ylim(mu[0], mu[-1])
            if row == n - 1:
                axes[row, c].set_xlabel("$\\phi$ (deg)")
            if c == 0:
                axes[row, c].set_ylabel("$\\mu$")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUTPUT_DIR,
                         f"targeted_details_{_safe_filename(cfg_dir)}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ─────────────────────────────────────────────────────────────────────
# C. Cross-potential comparison
# ─────────────────────────────────────────────────────────────────────
def generate_cross_potential_comparison():
    """Side-by-side rank maps across different potentials for the same region."""
    ref_pots = sorted([d for d in os.listdir(TARGETED_DIR)
                       if d in REFERENCE_POTENTIALS
                       and os.path.isdir(os.path.join(TARGETED_DIR, d))])
    if len(ref_pots) < 2:
        print("  Need >= 2 reference potentials for cross-potential comparison.")
        return

    regions = ["lagrange", "euler_strip", "charge_hotspot",
               "isosceles_ridge", "small_mu", "tier_cluster"]

    for region in regions:
        pot_data = []
        for pot in ref_pots:
            d = _load_targeted_region(os.path.join(TARGETED_DIR, pot), region)
            if d is not None:
                pot_data.append((pot, d))

        if len(pot_data) < 2:
            continue

        n = len(pot_data)
        fig, axes = plt.subplots(2, n, figsize=(7 * n, 12))
        if n == 1:
            axes = axes[:, np.newaxis]
        fig.suptitle(f"Cross-Potential Comparison — "
                     f"{REGION_LABELS.get(region, region)}",
                     fontsize=14, fontweight='bold')

        for col, (pot, data) in enumerate(pot_data):
            mu = data["mu_vals"]
            phi_deg = np.degrees(data["phi_vals"])
            mu_range = (mu[0], mu[-1])
            phi_range = (data["phi_vals"][0], data["phi_vals"][-1])

            rank = data["rank_map"]
            r_lo = max(rank.min() - 1, 112)
            r_hi = max(rank.max() + 1, 120)

            im0 = axes[0, col].pcolormesh(phi_deg, mu, rank, cmap='RdYlGn',
                                           shading='auto',
                                           vmin=r_lo, vmax=r_hi)
            axes[0, col].set_title(
                f"{POTENTIAL_LABELS.get(pot, pot)}\n"
                f"Rank [{rank.min()}–{rank.max()}]", fontsize=11)
            plt.colorbar(im0, ax=axes[0, col])
            _draw_isosceles(axes[0, col], mu_range, phi_range)
            axes[0, col].set_xlabel("$\\phi$ (deg)")
            if col == 0:
                axes[0, col].set_ylabel("$\\mu$")

            if "gap_score_map" in data:
                gs = data["gap_score_map"]
                im1 = axes[1, col].pcolormesh(phi_deg, mu, gs,
                                               cmap='inferno',
                                               shading='auto')
                axes[1, col].set_title("Gap Score", fontsize=11)
                plt.colorbar(im1, ax=axes[1, col])
                _draw_isosceles(axes[1, col], mu_range, phi_range)
                axes[1, col].set_xlabel("$\\phi$ (deg)")
                if col == 0:
                    axes[1, col].set_ylabel("$\\mu$")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fname = os.path.join(
            OUTPUT_DIR,
            f"cross_potential_{_safe_filename(region)}.png")
        fig.savefig(fname, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ─────────────────────────────────────────────────────────────────────
# D. Charge sensitivity comparison
# ─────────────────────────────────────────────────────────────────────
def generate_charge_comparisons():
    """Reference vs charged: rank difference and gap score difference."""
    charge_dirs = sorted([d for d in os.listdir(TARGETED_DIR)
                          if d.startswith("coulomb_")
                          and os.path.isdir(os.path.join(TARGETED_DIR, d))])
    if not charge_dirs:
        print("  No charged configurations found.")
        return

    regions = ["lagrange", "euler_strip", "charge_hotspot",
               "isosceles_ridge", "small_mu", "tier_cluster"]

    for chg_dir in charge_dirs:
        ref_dir = CHARGE_TO_REFERENCE.get(chg_dir)
        if not ref_dir:
            continue
        if not os.path.isdir(os.path.join(TARGETED_DIR, ref_dir)):
            continue

        chg_label = _dir_label(chg_dir)
        ref_label = _dir_label(ref_dir)

        region_diffs = []
        for region in regions:
            ref_data = _load_targeted_region(
                os.path.join(TARGETED_DIR, ref_dir), region)
            chg_data = _load_targeted_region(
                os.path.join(TARGETED_DIR, chg_dir), region)
            if ref_data is None or chg_data is None:
                continue
            if ref_data["rank_map"].shape != chg_data["rank_map"].shape:
                continue
            region_diffs.append((region, ref_data, chg_data))

        if not region_diffs:
            continue

        n = len(region_diffs)
        fig, axes = plt.subplots(n, 3, figsize=(21, 4.5 * n))
        fig.suptitle(f"Charge Sensitivity: {ref_label} → {chg_label}",
                     fontsize=13, fontweight='bold')
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, (rname, ref_d, chg_d) in enumerate(region_diffs):
            mu = ref_d["mu_vals"]
            phi_deg = np.degrees(ref_d["phi_vals"])
            mu_range = (mu[0], mu[-1])
            phi_range = (ref_d["phi_vals"][0], ref_d["phi_vals"][-1])

            rank_diff = chg_d["rank_map"].astype(float) - \
                ref_d["rank_map"].astype(float)
            vlim = max(abs(rank_diff.min()), abs(rank_diff.max()), 0.5)

            im0 = axes[row, 0].pcolormesh(
                phi_deg, mu, rank_diff, cmap='RdBu_r',
                shading='auto', vmin=-vlim, vmax=vlim)
            axes[row, 0].set_title(
                f"{REGION_LABELS.get(rname, rname)}\n"
                f"Rank Difference (Δ={rank_diff.min():.0f} to "
                f"{rank_diff.max():.0f})", fontsize=10)
            plt.colorbar(im0, ax=axes[row, 0], label="charged − ref")
            _draw_isosceles(axes[row, 0], mu_range, phi_range)

            if ("gap_score_map" in ref_d and "gap_score_map" in chg_d
                    and ref_d["gap_score_map"].shape ==
                    chg_d["gap_score_map"].shape):
                gs_diff = chg_d["gap_score_map"] - ref_d["gap_score_map"]
                gs_vlim = max(abs(gs_diff.min()), abs(gs_diff.max()), 0.5)
                im1 = axes[row, 1].pcolormesh(
                    phi_deg, mu, gs_diff, cmap='RdBu_r',
                    shading='auto', vmin=-gs_vlim, vmax=gs_vlim)
                axes[row, 1].set_title("Gap Score Difference", fontsize=10)
                plt.colorbar(im1, ax=axes[row, 1],
                             label="Δ gap score")
                _draw_isosceles(axes[row, 1], mu_range, phi_range)

            if ("optimal_eps_map" in ref_d and "optimal_eps_map" in chg_d
                    and ref_d["optimal_eps_map"].shape ==
                    chg_d["optimal_eps_map"].shape):
                log_ref = np.log10(np.where(
                    ref_d["optimal_eps_map"] > 0,
                    ref_d["optimal_eps_map"], 1e-5))
                log_chg = np.log10(np.where(
                    chg_d["optimal_eps_map"] > 0,
                    chg_d["optimal_eps_map"], 1e-5))
                eps_diff = log_chg - log_ref
                eps_vlim = max(abs(eps_diff.min()),
                               abs(eps_diff.max()), 0.1)
                im2 = axes[row, 2].pcolormesh(
                    phi_deg, mu, eps_diff, cmap='PuOr',
                    shading='auto', vmin=-eps_vlim, vmax=eps_vlim)
                axes[row, 2].set_title("Δ log₁₀(Optimal ε)", fontsize=10)
                plt.colorbar(im2, ax=axes[row, 2],
                             label="Δ log₁₀(ε)")
                _draw_isosceles(axes[row, 2], mu_range, phi_range)

            for c in range(3):
                axes[row, c].set_xlim(phi_deg[0], phi_deg[-1])
                axes[row, c].set_ylim(mu[0], mu[-1])
                if row == n - 1:
                    axes[row, c].set_xlabel("$\\phi$ (deg)")
                if c == 0:
                    axes[row, c].set_ylabel("$\\mu$")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = os.path.join(
            OUTPUT_DIR,
            f"charge_sensitivity_{_safe_filename(chg_dir)}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")

    _generate_charge_overview_panel()


def _generate_charge_overview_panel():
    """All charge configs side by side for the Lagrange region."""
    charge_dirs = sorted([d for d in os.listdir(TARGETED_DIR)
                          if d.startswith("coulomb_")
                          and d in CHARGE_TO_REFERENCE
                          and os.path.isdir(os.path.join(TARGETED_DIR, d))])
    region = "lagrange"
    ref_pots_seen = set()
    panels = []

    for chg_dir in charge_dirs:
        ref_dir = CHARGE_TO_REFERENCE[chg_dir]
        ref_data = _load_targeted_region(
            os.path.join(TARGETED_DIR, ref_dir), region)
        chg_data = _load_targeted_region(
            os.path.join(TARGETED_DIR, chg_dir), region)
        if ref_data is None or chg_data is None:
            continue
        if ref_data["rank_map"].shape != chg_data["rank_map"].shape:
            continue

        if ref_dir not in ref_pots_seen:
            panels.append(("ref", ref_dir, ref_data))
            ref_pots_seen.add(ref_dir)
        panels.append(("chg", chg_dir, chg_data))

    if len(panels) < 2:
        return

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 10))
    fig.suptitle("Charge Sensitivity Overview — Lagrange Region\n"
                 "Reference potentials vs. charged configurations",
                 fontsize=13, fontweight='bold')

    for col, (ptype, dirname, data) in enumerate(panels):
        mu = data["mu_vals"]
        phi_deg = np.degrees(data["phi_vals"])
        rank = data["rank_map"]
        r_lo = max(rank.min() - 1, 112)
        r_hi = max(rank.max() + 1, 120)

        im0 = axes[0, col].pcolormesh(phi_deg, mu, rank, cmap='RdYlGn',
                                       shading='auto',
                                       vmin=r_lo, vmax=r_hi)
        short_label = _dir_label(dirname)
        if len(short_label) > 40:
            short_label = short_label[:37] + "..."
        axes[0, col].set_title(short_label, fontsize=9)
        plt.colorbar(im0, ax=axes[0, col])
        axes[0, col].set_xlabel("$\\phi$ (deg)")
        if col == 0:
            axes[0, col].set_ylabel("$\\mu$")

        if "gap_score_map" in data:
            gs = data["gap_score_map"]
            im1 = axes[1, col].pcolormesh(phi_deg, mu, gs,
                                           cmap='inferno', shading='auto')
            axes[1, col].set_title("Gap Score", fontsize=9)
            plt.colorbar(im1, ax=axes[1, col])
            axes[1, col].set_xlabel("$\\phi$ (deg)")
            if col == 0:
                axes[1, col].set_ylabel("$\\mu$")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = os.path.join(OUTPUT_DIR, "charge_overview_lagrange.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ─────────────────────────────────────────────────────────────────────
# E. Summary dashboard
# ─────────────────────────────────────────────────────────────────────
def generate_summary_dashboard():
    """Summary figure: what was computed, key stats, scenario mapping."""
    targeted_configs = sorted([d for d in os.listdir(TARGETED_DIR)
                               if os.path.isdir(os.path.join(TARGETED_DIR, d))
                               and d not in ("plots", "log_sync")])
    hires_pots = sorted([d for d in os.listdir(HIRES_DIR)
                         if os.path.isdir(os.path.join(HIRES_DIR, d))
                         and d != "multi_epsilon"]) if os.path.isdir(HIRES_DIR) else []

    regions = ["lagrange", "euler_strip", "charge_hotspot",
               "isosceles_ridge", "small_mu", "tier_cluster"]

    stats = []
    for cfg in targeted_configs:
        base = os.path.join(TARGETED_DIR, cfg)
        n_regions = 0
        all_ranks = []
        for r in regions:
            d = _load_targeted_region(base, r)
            if d is not None:
                n_regions += 1
                all_ranks.append(d["rank_map"].ravel())
        if all_ranks:
            combined = np.concatenate(all_ranks)
            stats.append({
                "config": cfg,
                "label": _dir_label(cfg),
                "n_regions": n_regions,
                "rank_min": int(combined.min()),
                "rank_max": int(combined.max()),
                "rank_mode": int(np.median(combined)),
                "n_points": len(combined),
                "pct_116": float(np.mean(combined == 116) * 100),
            })

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    ax_table.set_title("Atlas Survey Summary", fontsize=14,
                       fontweight='bold', pad=20)

    if stats:
        col_labels = ["Configuration", "Regions", "Points",
                      "Rank Range", "% at 116"]
        cell_data = []
        for s in stats:
            cell_data.append([
                s["config"],
                str(s["n_regions"]),
                f"{s['n_points']:,}",
                f"{s['rank_min']}–{s['rank_max']}",
                f"{s['pct_116']:.1f}%",
            ])

        table = ax_table.table(cellText=cell_data, colLabels=col_labels,
                               loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#D6E4F0')

    ax_bar = fig.add_subplot(gs[1, 0])
    if stats:
        names = [s["config"] for s in stats]
        pcts = [s["pct_116"] for s in stats]
        colors = ['#2ecc71' if p > 99 else '#f39c12' if p > 90
                  else '#e74c3c' for p in pcts]
        bars = ax_bar.barh(range(len(names)), pcts, color=colors,
                           edgecolor='gray', alpha=0.85)
        ax_bar.set_yticks(range(len(names)))
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.set_xlabel("% of grid points with rank = 116")
        ax_bar.set_title("Rank Universality Check", fontsize=12)
        ax_bar.axvline(100, color='green', ls='--', alpha=0.5)
        ax_bar.set_xlim(0, 105)
        ax_bar.invert_yaxis()

    ax_hires = fig.add_subplot(gs[1, 1])
    if hires_pots:
        hires_stats = []
        for pot in hires_pots:
            data = _load_hires(pot)
            if data is None:
                continue
            rank = data["rank_map"]
            hires_stats.append({
                "potential": pot,
                "rank_min": int(rank.min()),
                "rank_max": int(rank.max()),
                "pct_116": float(np.mean(rank == 116) * 100),
            })

        if hires_stats:
            ax_hires.set_title("Full-Sphere Atlas Summary", fontsize=12)
            for i, hs in enumerate(hires_stats):
                label = POTENTIAL_LABELS.get(hs["potential"], hs["potential"])
                ax_hires.barh(i, hs["pct_116"],
                              color='steelblue', alpha=0.8,
                              edgecolor='gray')
                ax_hires.text(hs["pct_116"] + 1, i,
                              f"rank {hs['rank_min']}–{hs['rank_max']}",
                              va='center', fontsize=9)
            ax_hires.set_yticks(range(len(hires_stats)))
            ax_hires.set_yticklabels(
                [POTENTIAL_LABELS.get(h["potential"], h["potential"])
                 for h in hires_stats], fontsize=9)
            ax_hires.set_xlabel("% at rank 116")
            ax_hires.set_xlim(0, 115)
            ax_hires.invert_yaxis()
    else:
        ax_hires.text(0.5, 0.5, "No full-sphere data",
                      ha='center', va='center', transform=ax_hires.transAxes)
        ax_hires.axis('off')

    fname = os.path.join(OUTPUT_DIR, "atlas_summary_dashboard.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")

    summary_json = {
        "targeted_configs": stats,
        "hires_potentials": hires_pots,
        "total_targeted_files": sum(1 for _ in Path(TARGETED_DIR).rglob("*")
                                    if _.is_file()),
        "total_hires_files": sum(1 for _ in Path(HIRES_DIR).rglob("*")
                                 if _.is_file())
                              if os.path.isdir(HIRES_DIR) else 0,
        "failed_scenarios": ["tritium_he3 (yukawa)", "dusty_plasma (yukawa)"],
    }
    jpath = os.path.join(OUTPUT_DIR, "atlas_summary.json")
    with open(jpath, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  Saved {jpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble atlas figures from computed data")
    parser.add_argument("--only", type=str, default=None,
                        choices=["fullsphere", "targeted", "comparison",
                                 "charge", "summary"],
                        help="Generate only one category of figures")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("ATLAS ASSEMBLY")
    print("=" * 70)

    if args.only is None or args.only == "fullsphere":
        print("\n[A] Full-sphere atlas pages...")
        generate_fullsphere_pages()

    if args.only is None or args.only == "targeted":
        print("\n[B] Targeted region composites...")
        generate_targeted_composites()

    if args.only is None or args.only == "comparison":
        print("\n[C] Cross-potential comparisons...")
        generate_cross_potential_comparison()

    if args.only is None or args.only == "charge":
        print("\n[D] Charge sensitivity comparisons...")
        generate_charge_comparisons()

    if args.only is None or args.only == "summary":
        print("\n[E] Summary dashboard...")
        generate_summary_dashboard()

    print("\n" + "=" * 70)
    print(f"ALL ATLAS FIGURES SAVED TO {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
