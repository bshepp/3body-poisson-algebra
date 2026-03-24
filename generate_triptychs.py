#!/usr/bin/env python3
"""
Triptych Atlas Series
======================

For each configuration X, produces a figure with rows for each region:
  Panel 1: 1/r gravitational reference (equal-mass)
  Panel 2: Configuration X
  Panel 3: Difference map (X minus reference)

Covers rank maps, gap score maps, and optimal epsilon maps.

Also generates full-sphere triptychs where 100x100 data exists.

Output: atlas_figures/triptychs/
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker

TARGETED_DIR = "atlas_targeted"
HIRES_DIR = "atlas_output_hires"
OUTPUT_DIR = os.path.join("atlas_figures", "triptychs")

REFERENCE = "1_r"

CONFIGS = {
    "1_r2":                   r"$1/r^2$ (Calogero–Moser)",
    "1_r3":                   r"$1/r^3$",
    "log":                    r"$\log(r)$ (2D Vortices)",
    "coulomb_+1_+1_+1":       r"Coulomb $(+1,+1,+1)$ — Penning",
    "coulomb_+1_+1_-1":       r"Coulomb $(+1,+1,-1)$ — H$_2^+$",
    "coulomb_+1_-1_-1":       r"Coulomb $(+1,-1,-1)$ — Ps$^-$/H$^-$",
    "coulomb_+2_-1_-1":       r"Coulomb $(+2,-1,-1)$ — He",
    "coulomb_+3_-1_-1":       r"Coulomb $(+3,-1,-1)$ — Li$^+$",
    "coulomb_1_r2_+2_-1_-1":  r"$1/r^2$ Coulomb $(+2,-1,-1)$",
    "coulomb_1_r3_+2_-1_-1":  r"$1/r^3$ Coulomb $(+2,-1,-1)$",
}

REGION_ORDER = [
    "lagrange", "euler_strip", "charge_hotspot",
    "isosceles_ridge", "small_mu", "tier_cluster",
]

REGION_LABELS = {
    "lagrange":        "Lagrange equilateral",
    "euler_strip":     "Euler collinear strip",
    "charge_hotspot":  "Charge-sensitivity hotspot",
    "isosceles_ridge": "Isosceles ridge",
    "small_mu":        r"Small-$\mu$ region",
    "tier_cluster":    "Tier cluster",
}


def _safe_filename(s):
    return s.replace("+", "p").replace("-", "m").replace("/", "_")


def _draw_isosceles(ax, mu_range, phi_range_rad):
    phi_d = np.linspace(phi_range_rad[0], phi_range_rad[1], 500)
    mu_lo, mu_hi = mu_range
    ax.axhline(1.0, color='white', lw=0.7, ls='--', alpha=0.4)
    mu_c2 = 2 * np.cos(phi_d)
    m2 = (mu_c2 >= mu_lo) & (mu_c2 <= mu_hi)
    if m2.any():
        ax.plot(np.degrees(phi_d[m2]), mu_c2[m2],
                color='white', lw=0.7, ls='--', alpha=0.4)
    cos_p = np.cos(phi_d)
    safe = cos_p > 0.01
    mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
    m3 = np.isfinite(mu_c3) & (mu_c3 >= mu_lo) & (mu_c3 <= mu_hi)
    if m3.any():
        ax.plot(np.degrees(phi_d[m3]), mu_c3[m3],
                color='white', lw=0.7, ls='--', alpha=0.4)


def _load_region(config_dir, region):
    rdir = os.path.join(TARGETED_DIR, config_dir, region)
    rpath = os.path.join(rdir, "rank_map.npy")
    if not os.path.isfile(rpath):
        return None
    data = {}
    for name in ["rank_map", "gap_score_map", "optimal_eps_map",
                  "tier_map", "mu_vals", "phi_vals"]:
        p = os.path.join(rdir, f"{name}.npy")
        if os.path.isfile(p):
            data[name] = np.load(p)
    return data


def _load_hires(config_dir):
    d = os.path.join(HIRES_DIR, config_dir)
    rpath = os.path.join(d, "rank_map.npy")
    if not os.path.isfile(rpath):
        return None
    data = {}
    for name in ["rank_map", "gap_map", "mu_vals", "phi_vals"]:
        p = os.path.join(d, f"{name}.npy")
        if os.path.isfile(p):
            data[name] = np.load(p)
    return data


def generate_rank_triptych(cfg_key, cfg_label):
    """6-row rank triptych: ref | config | difference."""
    available = []
    for region in REGION_ORDER:
        ref = _load_region(REFERENCE, region)
        comp = _load_region(cfg_key, region)
        if ref is not None and comp is not None:
            if ref["rank_map"].shape == comp["rank_map"].shape:
                available.append((region, ref, comp))

    if not available:
        return False

    n = len(available)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4.2 * n + 1.5))
    fig.suptitle(
        f"Rank Triptych: $1/r$ Gravitational  vs  {cfg_label}",
        fontsize=14, fontweight='bold', y=0.995)

    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (region, ref, comp) in enumerate(available):
        mu = ref["mu_vals"]
        phi_deg = np.degrees(ref["phi_vals"])
        mu_range = (mu[0], mu[-1])
        phi_range = (ref["phi_vals"][0], ref["phi_vals"][-1])

        rr = ref["rank_map"]
        cr = comp["rank_map"]
        diff = cr.astype(float) - rr.astype(float)

        all_vals = np.concatenate([rr.ravel(), cr.ravel()])
        r_lo = max(all_vals.min() - 1, 112)
        r_hi = max(all_vals.max() + 1, 118)

        im0 = axes[row, 0].pcolormesh(phi_deg, mu, rr, cmap='RdYlGn',
                                       shading='auto',
                                       vmin=r_lo, vmax=r_hi)
        axes[row, 0].set_ylabel(
            f"{REGION_LABELS.get(region, region)}\n$\\mu$", fontsize=9)
        if row == 0:
            axes[row, 0].set_title(
                r"$1/r$ Gravitational" + "\n" +
                f"rank [{rr.min()}–{rr.max()}]", fontsize=10)
        else:
            axes[row, 0].set_title(
                f"rank [{rr.min()}–{rr.max()}]", fontsize=9)
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.85)
        _draw_isosceles(axes[row, 0], mu_range, phi_range)

        im1 = axes[row, 1].pcolormesh(phi_deg, mu, cr, cmap='RdYlGn',
                                       shading='auto',
                                       vmin=r_lo, vmax=r_hi)
        if row == 0:
            short = cfg_label if len(cfg_label) < 40 else cfg_label[:37] + "…"
            axes[row, 1].set_title(
                short + "\n" +
                f"rank [{cr.min()}–{cr.max()}]", fontsize=10)
        else:
            axes[row, 1].set_title(
                f"rank [{cr.min()}–{cr.max()}]", fontsize=9)
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.85)
        _draw_isosceles(axes[row, 1], mu_range, phi_range)

        vlim = max(abs(diff.min()), abs(diff.max()), 1)
        im2 = axes[row, 2].pcolormesh(phi_deg, mu, diff, cmap='coolwarm',
                                       shading='auto',
                                       vmin=-vlim, vmax=vlim)
        if row == 0:
            axes[row, 2].set_title(
                "Difference\n" +
                f"Δ = [{diff.min():.0f}, {diff.max():.0f}]", fontsize=10)
        else:
            axes[row, 2].set_title(
                f"Δ = [{diff.min():.0f}, {diff.max():.0f}]", fontsize=9)
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.85, label="Δ rank")
        _draw_isosceles(axes[row, 2], mu_range, phi_range)

        for c in range(3):
            axes[row, c].set_xlim(phi_deg[0], phi_deg[-1])
            axes[row, c].set_ylim(mu[0], mu[-1])
            if row == n - 1:
                axes[row, c].set_xlabel("$\\phi$ (deg)", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = os.path.join(OUTPUT_DIR,
                         f"triptych_rank_{_safe_filename(cfg_key)}.png")
    fig.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")
    return True


def generate_gap_triptych(cfg_key, cfg_label):
    """6-row gap score triptych."""
    available = []
    for region in REGION_ORDER:
        ref = _load_region(REFERENCE, region)
        comp = _load_region(cfg_key, region)
        if (ref is not None and comp is not None
                and "gap_score_map" in ref and "gap_score_map" in comp
                and ref["gap_score_map"].shape == comp["gap_score_map"].shape):
            available.append((region, ref, comp))

    if not available:
        return False

    n = len(available)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4.2 * n + 1.5))
    fig.suptitle(
        f"Gap Score Triptych: $1/r$ Gravitational  vs  {cfg_label}",
        fontsize=14, fontweight='bold', y=0.995)

    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (region, ref, comp) in enumerate(available):
        mu = ref["mu_vals"]
        phi_deg = np.degrees(ref["phi_vals"])
        mu_range = (mu[0], mu[-1])
        phi_range = (ref["phi_vals"][0], ref["phi_vals"][-1])

        rg = ref["gap_score_map"]
        cg = comp["gap_score_map"]
        diff = cg - rg

        all_g = np.concatenate([rg.ravel(), cg.ravel()])
        g_lo = np.percentile(all_g, 1)
        g_hi = np.percentile(all_g, 99)
        if g_hi - g_lo < 0.1:
            g_lo, g_hi = all_g.min() - 0.05, all_g.max() + 0.05

        im0 = axes[row, 0].pcolormesh(phi_deg, mu, rg, cmap='inferno',
                                       shading='auto',
                                       vmin=g_lo, vmax=g_hi)
        axes[row, 0].set_ylabel(
            f"{REGION_LABELS.get(region, region)}\n$\\mu$", fontsize=9)
        if row == 0:
            axes[row, 0].set_title(r"$1/r$ Gravitational", fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.85)
        _draw_isosceles(axes[row, 0], mu_range, phi_range)

        im1 = axes[row, 1].pcolormesh(phi_deg, mu, cg, cmap='inferno',
                                       shading='auto',
                                       vmin=g_lo, vmax=g_hi)
        if row == 0:
            short = cfg_label if len(cfg_label) < 40 else cfg_label[:37] + "…"
            axes[row, 1].set_title(short, fontsize=10)
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.85)
        _draw_isosceles(axes[row, 1], mu_range, phi_range)

        vlim = max(abs(diff.min()), abs(diff.max()), 0.5)
        im2 = axes[row, 2].pcolormesh(phi_deg, mu, diff, cmap='coolwarm',
                                       shading='auto',
                                       vmin=-vlim, vmax=vlim)
        if row == 0:
            axes[row, 2].set_title("Difference (Δ gap score)", fontsize=10)
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.85, label="Δ gap")
        _draw_isosceles(axes[row, 2], mu_range, phi_range)

        for c in range(3):
            axes[row, c].set_xlim(phi_deg[0], phi_deg[-1])
            axes[row, c].set_ylim(mu[0], mu[-1])
            if row == n - 1:
                axes[row, c].set_xlabel("$\\phi$ (deg)", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = os.path.join(OUTPUT_DIR,
                         f"triptych_gap_{_safe_filename(cfg_key)}.png")
    fig.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")
    return True


def generate_epsilon_triptych(cfg_key, cfg_label):
    """6-row optimal epsilon triptych."""
    available = []
    for region in REGION_ORDER:
        ref = _load_region(REFERENCE, region)
        comp = _load_region(cfg_key, region)
        if (ref is not None and comp is not None
                and "optimal_eps_map" in ref and "optimal_eps_map" in comp
                and ref["optimal_eps_map"].shape ==
                comp["optimal_eps_map"].shape):
            available.append((region, ref, comp))

    if not available:
        return False

    n = len(available)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4.2 * n + 1.5))
    fig.suptitle(
        f"Optimal $\\varepsilon$ Triptych: $1/r$ Gravitational  vs  "
        f"{cfg_label}",
        fontsize=14, fontweight='bold', y=0.995)

    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (region, ref, comp) in enumerate(available):
        mu = ref["mu_vals"]
        phi_deg = np.degrees(ref["phi_vals"])
        mu_range = (mu[0], mu[-1])
        phi_range = (ref["phi_vals"][0], ref["phi_vals"][-1])

        re = np.log10(np.where(ref["optimal_eps_map"] > 0,
                               ref["optimal_eps_map"], 1e-5))
        ce = np.log10(np.where(comp["optimal_eps_map"] > 0,
                               comp["optimal_eps_map"], 1e-5))
        diff = ce - re

        all_e = np.concatenate([re.ravel(), ce.ravel()])
        e_lo = np.percentile(all_e, 1)
        e_hi = np.percentile(all_e, 99)
        if e_hi - e_lo < 0.1:
            e_lo, e_hi = all_e.min() - 0.05, all_e.max() + 0.05

        im0 = axes[row, 0].pcolormesh(phi_deg, mu, re, cmap='viridis',
                                       shading='auto',
                                       vmin=e_lo, vmax=e_hi)
        axes[row, 0].set_ylabel(
            f"{REGION_LABELS.get(region, region)}\n$\\mu$", fontsize=9)
        if row == 0:
            axes[row, 0].set_title(
                r"$1/r$ Gravitational" + "\n" + r"$\log_{10}(\varepsilon)$",
                fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.85)
        _draw_isosceles(axes[row, 0], mu_range, phi_range)

        im1 = axes[row, 1].pcolormesh(phi_deg, mu, ce, cmap='viridis',
                                       shading='auto',
                                       vmin=e_lo, vmax=e_hi)
        if row == 0:
            short = cfg_label if len(cfg_label) < 40 else cfg_label[:37] + "…"
            axes[row, 1].set_title(
                short + "\n" + r"$\log_{10}(\varepsilon)$",
                fontsize=10)
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.85)
        _draw_isosceles(axes[row, 1], mu_range, phi_range)

        vlim = max(abs(diff.min()), abs(diff.max()), 0.5)
        im2 = axes[row, 2].pcolormesh(phi_deg, mu, diff, cmap='PuOr',
                                       shading='auto',
                                       vmin=-vlim, vmax=vlim)
        if row == 0:
            axes[row, 2].set_title(
                "Difference\n" + r"$\Delta\log_{10}(\varepsilon)$",
                fontsize=10)
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.85,
                     label=r"$\Delta\log_{10}\varepsilon$")
        _draw_isosceles(axes[row, 2], mu_range, phi_range)

        for c in range(3):
            axes[row, c].set_xlim(phi_deg[0], phi_deg[-1])
            axes[row, c].set_ylim(mu[0], mu[-1])
            if row == n - 1:
                axes[row, c].set_xlabel("$\\phi$ (deg)", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fname = os.path.join(OUTPUT_DIR,
                         f"triptych_eps_{_safe_filename(cfg_key)}.png")
    fig.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")
    return True


def generate_fullsphere_triptych(cfg_key, cfg_label):
    """Full 100x100 shape-sphere triptych where both ref and config have data."""
    ref = _load_hires(REFERENCE)
    comp = _load_hires(cfg_key)
    if ref is None or comp is None:
        return False
    if ref["rank_map"].shape != comp["rank_map"].shape:
        return False

    mu = ref["mu_vals"]
    phi_deg = np.degrees(ref["phi_vals"])
    mu_range = (mu[0], mu[-1])
    phi_range = (ref["phi_vals"][0], ref["phi_vals"][-1])

    rr = ref["rank_map"]
    cr = comp["rank_map"]
    diff = cr.astype(float) - rr.astype(float)

    all_vals = np.concatenate([rr.ravel(), cr.ravel()])
    r_lo = max(all_vals.min() - 1, 112)
    r_hi = max(all_vals.max() + 1, 120)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Full Shape Sphere Triptych: $1/r$ Gravitational  vs  {cfg_label}\n"
        f"100×100 grid, $\\mu \\in [{mu[0]:.1f}, {mu[-1]:.1f}]$, "
        f"$\\phi \\in [{phi_deg[0]:.0f}°, {phi_deg[-1]:.0f}°]$",
        fontsize=14, fontweight='bold')

    im0 = axes[0, 0].pcolormesh(phi_deg, mu, rr, cmap='RdYlGn',
                                 shading='auto', vmin=r_lo, vmax=r_hi)
    axes[0, 0].set_title(
        f"$1/r$ Gravitational\nrank [{rr.min()}–{rr.max()}]", fontsize=11)
    axes[0, 0].set_ylabel("$\\mu = r_{13}/r_{12}$")
    plt.colorbar(im0, ax=axes[0, 0], label="rank")
    _draw_isosceles(axes[0, 0], mu_range, phi_range)

    im1 = axes[0, 1].pcolormesh(phi_deg, mu, cr, cmap='RdYlGn',
                                 shading='auto', vmin=r_lo, vmax=r_hi)
    axes[0, 1].set_title(
        f"{cfg_label}\nrank [{cr.min()}–{cr.max()}]", fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], label="rank")
    _draw_isosceles(axes[0, 1], mu_range, phi_range)

    vlim = max(abs(diff.min()), abs(diff.max()), 1)
    im2 = axes[0, 2].pcolormesh(phi_deg, mu, diff, cmap='coolwarm',
                                 shading='auto', vmin=-vlim, vmax=vlim)
    axes[0, 2].set_title(
        f"Rank Difference\nΔ = [{diff.min():.0f}, {diff.max():.0f}]",
        fontsize=11)
    plt.colorbar(im2, ax=axes[0, 2], label="Δ rank")
    _draw_isosceles(axes[0, 2], mu_range, phi_range)

    if "gap_map" in ref and "gap_map" in comp:
        rg = np.log10(np.where(ref["gap_map"] > 0, ref["gap_map"], 1e-10))
        cg = np.log10(np.where(comp["gap_map"] > 0, comp["gap_map"], 1e-10))
        gdiff = cg - rg

        all_g = np.concatenate([rg.ravel(), cg.ravel()])
        g_lo = np.percentile(all_g, 1)
        g_hi = np.percentile(all_g, 99)
        if g_hi - g_lo < 0.1:
            g_lo, g_hi = all_g.min() - 0.05, all_g.max() + 0.05

        im3 = axes[1, 0].pcolormesh(phi_deg, mu, rg, cmap='inferno',
                                     shading='auto', vmin=g_lo, vmax=g_hi)
        axes[1, 0].set_title("log₁₀(Gap Ratio)", fontsize=11)
        axes[1, 0].set_ylabel("$\\mu$")
        axes[1, 0].set_xlabel("$\\phi$ (deg)")
        plt.colorbar(im3, ax=axes[1, 0])
        _draw_isosceles(axes[1, 0], mu_range, phi_range)

        im4 = axes[1, 1].pcolormesh(phi_deg, mu, cg, cmap='inferno',
                                     shading='auto', vmin=g_lo, vmax=g_hi)
        axes[1, 1].set_title("log₁₀(Gap Ratio)", fontsize=11)
        axes[1, 1].set_xlabel("$\\phi$ (deg)")
        plt.colorbar(im4, ax=axes[1, 1])
        _draw_isosceles(axes[1, 1], mu_range, phi_range)

        gvlim = max(abs(gdiff.min()), abs(gdiff.max()), 0.5)
        im5 = axes[1, 2].pcolormesh(phi_deg, mu, gdiff, cmap='coolwarm',
                                     shading='auto',
                                     vmin=-gvlim, vmax=gvlim)
        axes[1, 2].set_title("Δ log₁₀(Gap Ratio)", fontsize=11)
        axes[1, 2].set_xlabel("$\\phi$ (deg)")
        plt.colorbar(im5, ax=axes[1, 2], label="Δ log₁₀(gap)")
        _draw_isosceles(axes[1, 2], mu_range, phi_range)
    else:
        for c in range(3):
            axes[1, c].axis('off')

    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlim(phi_deg[0], phi_deg[-1])
            ax.set_ylim(mu[0], mu[-1])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = os.path.join(
        OUTPUT_DIR, f"triptych_fullsphere_{_safe_filename(cfg_key)}.png")
    fig.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TRIPTYCH ATLAS SERIES")
    print("  Reference: 1/r equal-mass gravitational")
    print(f"  Comparisons: {len(CONFIGS)} configurations")
    print(f"  Regions: {len(REGION_ORDER)}")
    print("=" * 70)

    total = 0

    for cfg_key, cfg_label in sorted(CONFIGS.items()):
        print(f"\n--- {cfg_key} ---")

        if generate_rank_triptych(cfg_key, cfg_label):
            total += 1
        if generate_gap_triptych(cfg_key, cfg_label):
            total += 1
        if generate_epsilon_triptych(cfg_key, cfg_label):
            total += 1
        if generate_fullsphere_triptych(cfg_key, cfg_label):
            total += 1

    print(f"\n{'='*70}")
    print(f"COMPLETE: {total} triptych figures saved to {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
