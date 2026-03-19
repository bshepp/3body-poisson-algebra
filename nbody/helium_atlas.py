#!/usr/bin/env python3
"""
Helium Atlas: Charge-Sign Invariance Comparison Tool
=====================================================

Compares multi-epsilon atlas data for the helium Coulomb configuration
(charges +2, -1, -1) against the all-attractive gravitational 1/r
reference.  Both datasets are produced by multi_epsilon_atlas.py in
the parent directory.

Generates:
  - Side-by-side gap ratio heatmaps at each epsilon (the main signal)
  - Gap ratio difference maps (helium - gravitational)
  - Gap-vs-epsilon profiles at notable configurations
  - Singular value spectrum comparison at notable points
  - Orbit family overlay on the best-epsilon helium gap atlas

Prerequisites:
  Run the scans first from the parent directory:
    python multi_epsilon_atlas.py scan --potential 1/r
    python multi_epsilon_atlas.py scan --charges 2 -1 -1

Usage:
    python helium_atlas.py compare           # side-by-side comparison
    python helium_atlas.py orbits            # orbit family overlay
    python helium_atlas.py all               # both
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time
from pathlib import Path

import functools
print = functools.partial(print, flush=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.join(_SCRIPT_DIR, "..")

HIRES_DIR = os.path.join(_PARENT_DIR, "atlas_output_hires")

CHARGES = (2, -1, -1)

POT_DIR = {'1/r': '1_r', '1/r2': '1_r2', 'harmonic': 'harmonic'}
POT_LABEL = {
    '1/r':      '1/r (Newton)',
    '1/r2':     '1/r^2 (Calogero-Moser)',
    'harmonic': 'r^2 (Harmonic)',
}


def _charged_dir_name(potential_type='1/r'):
    charge_part = "_".join(f"{c:+d}" for c in CHARGES)
    if potential_type == '1/r':
        return f"coulomb_{charge_part}"
    pot_slug = POT_DIR.get(potential_type, potential_type.replace('/', ''))
    return f"coulomb_{pot_slug}_{charge_part}"


EPSILONS = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]

GRID_N = 100


def eps_tag(eps):
    return f"eps_{eps:.0e}".replace("+", "")


def _data_dir(base_name, eps):
    base = os.path.join(HIRES_DIR, base_name)
    if eps == 5e-3:
        return base
    return os.path.join(base, eps_tag(eps))


def _load_eps_data(base_name):
    """Load rank/gap/sv data for all available epsilons."""
    data = {}
    mu, phi = None, None
    for eps in EPSILONS:
        d = _data_dir(base_name, eps)
        rm_path = os.path.join(d, "rank_map.npy")
        if not os.path.exists(rm_path):
            continue
        data[eps] = {
            "rank": np.load(rm_path),
            "gap": np.load(os.path.join(d, "gap_map.npy")),
        }
        sv_path = os.path.join(d, "sv_spectra.npy")
        if os.path.exists(sv_path):
            data[eps]["sv"] = np.load(sv_path)
        if mu is None:
            mu_path = os.path.join(d, "mu_vals.npy")
            phi_path = os.path.join(d, "phi_vals.npy")
            if os.path.exists(mu_path):
                mu = np.load(mu_path)
                phi = np.load(phi_path)
    return data, mu, phi


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _log_gap(gap_arr):
    """Convert gap ratio array to log10, handling inf/zero."""
    safe = np.where(np.isinf(gap_arr), 1e15, gap_arr)
    return np.log10(np.clip(safe, 1, None))


def run_compare(potential_type='1/r'):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    helium_dir = _charged_dir_name(potential_type)
    ref_dir = POT_DIR[potential_type]
    pot_label = POT_LABEL.get(potential_type, potential_type)
    out_dir = os.path.join(_SCRIPT_DIR,
                           f"helium_comparison_{POT_DIR[potential_type]}"
                           if potential_type != '1/r'
                           else "helium_comparison")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"HELIUM vs ALL-ATTRACTIVE: GAP & SV COMPARISON  [{pot_label}]")
    print("=" * 70)

    he_data, he_mu, he_phi = _load_eps_data(helium_dir)
    ref_data, ref_mu, ref_phi = _load_eps_data(ref_dir)

    if not he_data:
        print("\nERROR: No helium atlas data found.")
        print(f"  Expected in: {os.path.join(HIRES_DIR, HELIUM_DIR_NAME)}/")
        print("  Run:  python multi_epsilon_atlas.py scan --charges 2 -1 -1")
        return
    if not ref_data:
        print("\nWARNING: No all-attractive reference data found.")
        print(f"  Expected in: {os.path.join(HIRES_DIR, REF_DIR_NAME)}/")
        print("  Run:  python multi_epsilon_atlas.py scan --potential 1/r")
        print("  Generating helium-only plots.\n")

    common_eps = sorted(set(he_data.keys()) & set(ref_data.keys()),
                        reverse=True) if ref_data else []
    he_eps = sorted(he_data.keys(), reverse=True)

    print(f"  Helium epsilons:    {[f'{e:.0e}' for e in he_eps]}")
    if ref_data:
        ref_eps = sorted(ref_data.keys(), reverse=True)
        print(f"  Reference epsilons: {[f'{e:.0e}' for e in ref_eps]}")
        print(f"  Common epsilons:    {[f'{e:.0e}' for e in common_eps]}")

    mu = he_mu
    phi = he_phi
    phi_deg = np.degrees(phi)

    def _draw_isosceles(ax):
        phi_d = np.linspace(phi[0], phi[-1], 500)
        ax.axhline(1.0, color="cyan", linewidth=1.2, linestyle="--",
                    alpha=0.5)
        mu_c2 = 2 * np.cos(phi_d)
        m2 = (mu_c2 >= mu[0]) & (mu_c2 <= mu[-1])
        ax.plot(np.degrees(phi_d[m2]), mu_c2[m2], color="lime",
                linewidth=1.2, linestyle="--", alpha=0.5)
        cos_p = np.cos(phi_d)
        safe = cos_p > 0.01
        mu_c3 = np.where(safe, 1.0 / (2 * cos_p), np.nan)
        m3 = np.isfinite(mu_c3) & (mu_c3 >= mu[0]) & (mu_c3 <= mu[-1])
        ax.plot(np.degrees(phi_d[m3]), mu_c3[m3], color="magenta",
                linewidth=1.2, linestyle="--", alpha=0.5)

    notable = {
        "Lagrange": (1.0, np.pi / 3),
        "Euler": (0.5, np.pi),
        "Right-isos": (1.0, np.pi / 2),
        "Generic-1": (0.6, 0.8),
        "Generic-2": (1.5, 1.2),
        "Generic-3": (2.0, 0.6),
    }

    def _ij(t_mu, t_phi):
        return (np.argmin(np.abs(mu - t_mu)),
                np.argmin(np.abs(phi - t_phi)))

    # ==================================================================
    # 1. Gap ratio heatmaps: side-by-side + difference
    # ==================================================================
    if common_eps:
        print("\n  Generating gap ratio comparison maps...")
        for eps in common_eps:
            h_lg = _log_gap(he_data[eps]["gap"])
            r_lg = _log_gap(ref_data[eps]["gap"])
            diff_lg = h_lg - r_lg
            vmax_g = max(h_lg.max(), r_lg.max())

            fig, axes = plt.subplots(1, 3, figsize=(22, 6), facecolor="white")

            im0 = axes[0].pcolormesh(phi_deg, mu, r_lg, cmap="inferno",
                                     vmin=0, vmax=vmax_g, shading="auto")
            _draw_isosceles(axes[0])
            axes[0].plot(60, 1.0, "*", color="cyan", markersize=12,
                         markeredgecolor="white", zorder=15)
            axes[0].set_title("All-Attractive 1/r\n(gravitational)")
            axes[0].set_xlabel(r"$\phi$ (degrees)")
            axes[0].set_ylabel(r"$\mu = r_{13}/r_{12}$")
            plt.colorbar(im0, ax=axes[0], label=r"$\log_{10}$(gap ratio)",
                         shrink=0.85)

            im1 = axes[1].pcolormesh(phi_deg, mu, h_lg, cmap="inferno",
                                     vmin=0, vmax=vmax_g, shading="auto")
            _draw_isosceles(axes[1])
            axes[1].plot(60, 1.0, "*", color="cyan", markersize=12,
                         markeredgecolor="white", zorder=15)
            axes[1].set_title("Helium (+2, -1, -1)\n(mixed Coulomb)")
            axes[1].set_xlabel(r"$\phi$ (degrees)")
            plt.colorbar(im1, ax=axes[1], label=r"$\log_{10}$(gap ratio)",
                         shrink=0.85)

            vabs = max(abs(diff_lg.min()), abs(diff_lg.max()), 0.1)
            im2 = axes[2].pcolormesh(phi_deg, mu, diff_lg, cmap="RdBu_r",
                                     vmin=-vabs, vmax=vabs, shading="auto")
            _draw_isosceles(axes[2])
            axes[2].set_title(r"$\Delta\log_{10}$(gap)"
                              "\n(helium - gravitational)")
            axes[2].set_xlabel(r"$\phi$ (degrees)")
            plt.colorbar(im2, ax=axes[2],
                         label=r"$\Delta\log_{10}$(gap ratio)", shrink=0.85)

            corr = np.corrcoef(h_lg.ravel(), r_lg.ravel())[0, 1]
            fig.suptitle(
                f"Gap Ratio Landscape at "
                r"$\varepsilon$"
                f" = {eps:.0e}  |  "
                f"Pearson r = {corr:.4f}  |  "
                r"$\Delta$"
                f" range [{diff_lg.min():.2f}, {diff_lg.max():.2f}]",
                fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            out_path = os.path.join(out_dir,
                                    f"gap_landscape_{eps_tag(eps)}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved {out_path}")

    # ==================================================================
    # 2. Gap-vs-epsilon profiles at notable configurations
    # ==================================================================
    eps_for_curves = common_eps if common_eps else he_eps
    if len(eps_for_curves) >= 2:
        print("\n  Generating gap-vs-epsilon profiles...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
        cmap_lines = plt.get_cmap("tab10")

        for idx, (name, (t_mu, t_phi)) in enumerate(notable.items()):
            i, j = _ij(t_mu, t_phi)
            color = cmap_lines(idx)
            loc_str = (f"(\u03bc={mu[i]:.2f},"
                       f"\u03c6={np.degrees(phi[j]):.0f}\u00b0)")

            he_gaps = [np.log10(max(he_data[e]["gap"][i, j], 1))
                       for e in eps_for_curves if e in he_data]
            he_eps_list = [e for e in eps_for_curves if e in he_data]

            axes[0].plot(he_eps_list, he_gaps, "o-", color=color,
                         linewidth=2, markersize=6,
                         label=f"{name} {loc_str}")

            if ref_data:
                ref_gaps = [np.log10(max(ref_data[e]["gap"][i, j], 1))
                            for e in eps_for_curves if e in ref_data]
                ref_eps_list = [e for e in eps_for_curves if e in ref_data]
                axes[0].plot(ref_eps_list, ref_gaps, "s--", color=color,
                             linewidth=1.5, markersize=5, alpha=0.6)

            if ref_data:
                paired_eps = [e for e in eps_for_curves
                              if e in he_data and e in ref_data]
                diffs = [np.log10(max(he_data[e]["gap"][i, j], 1))
                         - np.log10(max(ref_data[e]["gap"][i, j], 1))
                         for e in paired_eps]
                axes[1].plot(paired_eps, diffs, "o-", color=color,
                             linewidth=2, markersize=6, label=name)

        axes[0].set_xscale("log")
        axes[0].invert_xaxis()
        axes[0].set_xlabel(r"$\varepsilon$", fontsize=13)
        axes[0].set_ylabel(r"$\log_{10}$(gap ratio)", fontsize=13)
        axes[0].set_title("Gap Ratio vs Epsilon\n"
                          "(solid = helium, dashed = gravitational)")
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].grid(True, alpha=0.3)

        if ref_data:
            axes[1].set_xscale("log")
            axes[1].invert_xaxis()
            axes[1].axhline(0, color="gray", linewidth=1, alpha=0.5)
            axes[1].set_xlabel(r"$\varepsilon$", fontsize=13)
            axes[1].set_ylabel(r"$\Delta\log_{10}$(gap ratio)", fontsize=13)
            axes[1].set_title("Gap Difference (helium - gravitational)\n"
                              "at each notable configuration")
            axes[1].legend(fontsize=8, ncol=2)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No reference data",
                         transform=axes[1].transAxes, ha="center",
                         fontsize=14, color="gray")

        fig.suptitle(f"Gap Ratio Profiles: Helium vs All-Attractive [{pot_label}]",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "gap_vs_epsilon_profiles.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {out_path}")

    # ==================================================================
    # 3. Singular value spectrum comparison at notable points
    # ==================================================================
    best_eps = min(common_eps) if common_eps else min(he_eps)
    he_has_sv = "sv" in he_data.get(best_eps, {})
    ref_has_sv = ref_data and "sv" in ref_data.get(best_eps, {})

    if he_has_sv:
        print(f"\n  Generating SV spectrum comparison (eps={best_eps:.0e})...")
        n_notable = len(notable)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
        axes_flat = axes.flatten()

        for idx, (name, (t_mu, t_phi)) in enumerate(notable.items()):
            ax = axes_flat[idx]
            i, j = _ij(t_mu, t_phi)

            he_sv = he_data[best_eps]["sv"][i, j]
            he_sv = he_sv[he_sv > 0]
            he_sv_norm = he_sv / he_sv[0] if len(he_sv) > 0 else he_sv

            ax.semilogy(range(len(he_sv_norm)), he_sv_norm, "o-",
                        color="#E63946", markersize=3, linewidth=1.2,
                        label="Helium", zorder=5)

            if ref_has_sv:
                ref_sv = ref_data[best_eps]["sv"][i, j]
                ref_sv = ref_sv[ref_sv > 0]
                ref_sv_norm = ref_sv / ref_sv[0] if len(ref_sv) > 0 else ref_sv
                ax.semilogy(range(len(ref_sv_norm)), ref_sv_norm, "s-",
                            color="#457B9D", markersize=3, linewidth=1.2,
                            alpha=0.7, label="Gravitational", zorder=4)

            ax.axvline(115.5, color="gray", linewidth=1, linestyle=":",
                       alpha=0.6)
            ax.text(115.5, ax.get_ylim()[0] * 2, " rank=116",
                    fontsize=7, color="gray", va="bottom")
            ax.set_title(f"{name}\n"
                         f"(\u03bc={mu[i]:.2f}, "
                         f"\u03c6={np.degrees(phi[j]):.0f}\u00b0)",
                         fontsize=10)
            ax.set_xlabel("SV index")
            ax.set_ylabel(r"$\sigma_k / \sigma_0$")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"Singular Value Spectra: Helium vs Gravitational\n"
            r"$\varepsilon$"
            f" = {best_eps:.0e}  |  normalized to "
            r"$\sigma_0$",
            fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(out_dir, "sv_spectra_comparison.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {out_path}")

    # ==================================================================
    # 4. Gap correlation scatter plot
    # ==================================================================
    if common_eps and ref_data:
        print("\n  Generating gap correlation scatter...")
        best = min(common_eps)
        h_lg = _log_gap(he_data[best]["gap"])
        r_lg = _log_gap(ref_data[best]["gap"])
        corr = np.corrcoef(h_lg.ravel(), r_lg.ravel())[0, 1]

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
        ax.scatter(r_lg.ravel(), h_lg.ravel(), s=2, alpha=0.3,
                   color="#2A9D8F")
        lims = [min(r_lg.min(), h_lg.min()),
                max(r_lg.max(), h_lg.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.4, label="y = x")
        ax.set_xlabel(r"Gravitational $\log_{10}$(gap)", fontsize=12)
        ax.set_ylabel(r"Helium $\log_{10}$(gap)", fontsize=12)
        ax.set_title(
            f"Gap Ratio Correlation at "
            r"$\varepsilon$"
            f" = {best:.0e}\n"
            f"Pearson r = {corr:.4f}  |  "
            f"N = {h_lg.size} grid points",
            fontsize=13)
        ax.legend(fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "gap_correlation_scatter.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {out_path}")

    # ==================================================================
    # 5. Summary statistics
    # ==================================================================
    print(f"\n{'='*70}")
    print("GAP & SV COMPARISON STATISTICS")
    print(f"{'='*70}")

    summary_lines = []
    summary_lines.append("Helium vs All-Attractive: Gap & SV Comparison\n")
    summary_lines.append("=" * 60 + "\n\n")

    for eps in (common_eps if common_eps else he_eps):
        h_lg = _log_gap(he_data[eps]["gap"])
        line = f"eps = {eps:.0e}:\n"
        line += (f"  Helium log-gap range:       "
                 f"[{h_lg.min():.2f}, {h_lg.max():.2f}]\n")
        line += (f"  Helium log-gap mean/std:    "
                 f"{h_lg.mean():.2f} +/- {h_lg.std():.2f}\n")

        if eps in ref_data:
            r_lg = _log_gap(ref_data[eps]["gap"])
            diff_lg = h_lg - r_lg
            corr = np.corrcoef(h_lg.ravel(), r_lg.ravel())[0, 1]
            line += (f"  Ref log-gap range:          "
                     f"[{r_lg.min():.2f}, {r_lg.max():.2f}]\n")
            line += (f"  Ref log-gap mean/std:       "
                     f"{r_lg.mean():.2f} +/- {r_lg.std():.2f}\n")
            line += (f"  Delta log-gap range:        "
                     f"[{diff_lg.min():.2f}, {diff_lg.max():.2f}]\n")
            line += (f"  Delta log-gap mean/std:     "
                     f"{diff_lg.mean():.2f} +/- {diff_lg.std():.2f}\n")
            line += f"  Pearson correlation:        {corr:.4f}\n"

            h_rm = he_data[eps]["rank"]
            r_rm = ref_data[eps]["rank"]
            rank_diff = h_rm.astype(int) - r_rm.astype(int)
            n_rank_diff = np.sum(rank_diff != 0)
            line += (f"  Rank agreement:             "
                     f"{h_rm.size - n_rank_diff}/{h_rm.size} "
                     f"({100*(1 - n_rank_diff/h_rm.size):.1f}%)\n")
        line += "\n"
        summary_lines.append(line)
        print(f"  {line.rstrip()}")

    if common_eps and ref_data:
        best = min(common_eps)
        h_lg = _log_gap(he_data[best]["gap"])
        r_lg = _log_gap(ref_data[best]["gap"])
        corr = np.corrcoef(h_lg.ravel(), r_lg.ravel())[0, 1]
        diff_lg = h_lg - r_lg

        if corr > 0.95:
            verdict = (f"HIGHLY CORRELATED (r={corr:.4f}): The gap ratio "
                       f"landscape is nearly identical between helium and "
                       f"gravitational potentials. Delta log-gap "
                       f"mean={diff_lg.mean():.3f}, std={diff_lg.std():.3f}. "
                       f"Charge-sign invariance holds at the gap level.")
        elif corr > 0.7:
            verdict = (f"MODERATELY CORRELATED (r={corr:.4f}): The gap "
                       f"landscapes share broad structure but differ in "
                       f"detail. Delta log-gap std={diff_lg.std():.3f}.")
        else:
            verdict = (f"WEAKLY CORRELATED (r={corr:.4f}): The gap "
                       f"landscapes differ substantially. Charge sign "
                       f"materially affects the local algebra separation.")
    else:
        verdict = ("NO COMPARISON: Reference data not available.")

    summary_lines.append(f"\nVERDICT: {verdict}\n")
    print(f"\n  VERDICT: {verdict}")

    summary_path = os.path.join(out_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(summary_lines)
    print(f"\n  Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Orbit family overlay
# ---------------------------------------------------------------------------

def _draw_orbit_diagram(ax, kind, title, color):
    """Draw a small schematic of a 3-body configuration."""
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    nuc_kw = dict(s=140, zorder=10, edgecolors="black", linewidth=0.8)
    elec_kw = dict(s=60, zorder=10, edgecolors="black", linewidth=0.8)

    if kind == "eZe":
        ax.scatter([0], [0], color="#CC0000", marker="o", **nuc_kw)
        ax.scatter([-1], [0], color="#3366FF", marker="o", **elec_kw)
        ax.scatter([1], [0], color="#3366FF", marker="o", **elec_kw)
        ax.plot([-1, 1], [0, 0], "k-", linewidth=0.8, zorder=1)
        ax.text(0, 0.25, "+2", ha="center", fontsize=7, color="#CC0000",
                fontweight="bold")
        ax.text(-1, 0.25, r"e$^-$", ha="center", fontsize=7)
        ax.text(1, 0.25, r"e$^-$", ha="center", fontsize=7)
    elif kind == "Zee":
        ax.scatter([0], [0], color="#CC0000", marker="o", **nuc_kw)
        ax.scatter([0.6], [0], color="#3366FF", marker="o", **elec_kw)
        ax.scatter([1.2], [0], color="#3366FF", marker="o", **elec_kw)
        ax.plot([0, 1.2], [0, 0], "k-", linewidth=0.8, zorder=1)
        ax.text(0, 0.25, "+2", ha="center", fontsize=7, color="#CC0000",
                fontweight="bold")
    elif kind == "Langmuir":
        ax.scatter([0], [0], color="#CC0000", marker="o", **nuc_kw)
        theta = np.linspace(0, 2 * np.pi, 80)
        ax.plot(0.9 * np.cos(theta), 0.9 * np.sin(theta), "--",
                color="#999999", linewidth=0.8)
        ax.scatter([0.9], [0], color="#3366FF", marker="o", **elec_kw)
        ax.scatter([-0.9], [0], color="#3366FF", marker="o", **elec_kw)
        ax.text(0, 0.22, "+2", ha="center", fontsize=7, color="#CC0000",
                fontweight="bold")
    elif kind == "equilateral":
        ax.scatter([0], [0], color="#CC0000", marker="o", **nuc_kw)
        ax.scatter([1], [0], color="#3366FF", marker="o", **elec_kw)
        ax.scatter([0.5], [0.866], color="#3366FF", marker="o", **elec_kw)
        ax.plot([0, 1, 0.5, 0], [0, 0, 0.866, 0], "k-",
                linewidth=0.8, zorder=1)
        ax.text(0, -0.25, "+2", ha="center", fontsize=7, color="#CC0000",
                fontweight="bold")
    elif kind == "frozen_planet":
        ax.scatter([0], [0], color="#CC0000", marker="o", **nuc_kw)
        ax.scatter([0.35], [0], color="#3366FF", marker="o", **elec_kw)
        ax.scatter([1.2], [0], color="#3366FF", marker="o",
                   alpha=0.5, **elec_kw)
        ax.plot([0, 1.2], [0, 0], "k-", linewidth=0.8, zorder=1)
        ax.text(1.2, 0.25, "frozen", ha="center", fontsize=6,
                color="#666666", style="italic")
        ax.text(0, 0.25, "+2", ha="center", fontsize=7, color="#CC0000",
                fontweight="bold")
    elif kind == "wannier":
        ax.scatter([0], [0], color="#CC0000", marker="o", **nuc_kw)
        ang = np.radians(50)
        ax.scatter([0.9], [0], color="#3366FF", marker="o", **elec_kw)
        ax.scatter([0.9 * np.cos(ang)], [0.9 * np.sin(ang)], color="#3366FF",
                   marker="o", **elec_kw)
        ax.plot([0, 0.9], [0, 0], "k--", linewidth=0.8, zorder=1)
        ax.plot([0, 0.9 * np.cos(ang)], [0, 0.9 * np.sin(ang)], "k--",
                linewidth=0.8, zorder=1)
        ax.text(0, -0.25, "+2", ha="center", fontsize=7, color="#CC0000",
                fontweight="bold")

    ax.set_title(title, fontsize=9, fontweight="bold", color=color, pad=2)


def run_orbits(potential_type='1/r'):
    """
    Overlay known classical helium periodic orbit families on the atlas.

    Uses the best available epsilon (smallest) for maximum structure
    visibility.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec

    helium_dir = _charged_dir_name(potential_type)
    pot_label = POT_LABEL.get(potential_type, potential_type)
    out_dir = os.path.join(_SCRIPT_DIR,
                           f"helium_comparison_{POT_DIR[potential_type]}"
                           if potential_type != '1/r'
                           else "helium_comparison")
    os.makedirs(out_dir, exist_ok=True)

    he_data, mu, phi = _load_eps_data(helium_dir)
    if not he_data:
        print("ERROR: Helium atlas data not found. Run scan first.")
        return

    best_eps = min(he_data.keys())
    print(f"\n  Using best epsilon: {best_eps:.0e}")

    h_gap = he_data[best_eps]["gap"]
    h_lg = _log_gap(h_gap)
    phi_deg = np.degrees(phi)

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 5, figure=fig,
                  height_ratios=[1, 4, 1],
                  width_ratios=[1, 1, 4, 1, 1],
                  hspace=0.35, wspace=0.3)

    ax_main = fig.add_subplot(gs[1, 1:4])

    ax_eZe = fig.add_subplot(gs[0, 3:5])
    ax_Zee = fig.add_subplot(gs[0, 0:2])
    ax_lang = fig.add_subplot(gs[2, 3:5])
    ax_equi = fig.add_subplot(gs[2, 2])
    ax_froz = fig.add_subplot(gs[2, 0:2])
    ax_wann = fig.add_subplot(gs[0, 2])

    _draw_orbit_diagram(ax_eZe, "eZe",
                        "eZe collinear\n(electrons opposite sides)",
                        "#DD2222")
    _draw_orbit_diagram(ax_Zee, "Zee",
                        "Zee collinear\n(electrons same side)", "#2255CC")
    _draw_orbit_diagram(ax_lang, "Langmuir",
                        "Langmuir orbit\n(circular, 180\u00b0 apart)",
                        "#DD8800")
    _draw_orbit_diagram(ax_equi, "equilateral",
                        "Equilateral\n(Lagrange)", "#00AA00")
    _draw_orbit_diagram(ax_froz, "frozen_planet",
                        "Frozen planet\n(one electron frozen)", "#6666CC")
    _draw_orbit_diagram(ax_wann, "wannier",
                        r"Wannier ridge ($r_1\!=\!r_2$)", "#CC9900")

    im = ax_main.pcolormesh(phi_deg, mu, h_lg,
                            cmap="inferno", shading="auto", alpha=0.9)

    ax_main.axvline(x=phi_deg[-1], color="#DD2222", linewidth=3,
                    label=r"eZe ($\phi\!=\!180\degree$)", zorder=5)
    ax_main.axvline(x=phi_deg[0], color="#2255CC", linewidth=3,
                    label=r"Zee ($\phi\!=\!0\degree$)", zorder=5)
    ax_main.axhline(y=1.0, color="#CC9900", linewidth=2, linestyle="--",
                    label=r"Wannier ridge ($\mu\!=\!1$)", zorder=5)

    ax_main.scatter([phi_deg[-1]], [1.0], color="#DD8800", marker="D",
                    s=200, edgecolors="white", linewidth=1.5,
                    label="Langmuir / Asym. stretch", zorder=12)
    ax_main.scatter([60.0], [1.0], color="#00AA00", marker="^",
                    s=200, edgecolors="white", linewidth=1.5,
                    label="Equilateral (Lagrange)", zorder=12)

    for mu_lo, mu_hi in [(0.05, 0.35), (3.0, 5.0)]:
        rect = Rectangle((phi_deg[0], mu_lo),
                          30, mu_hi - mu_lo,
                          linewidth=1.5, edgecolor="#6666CC",
                          facecolor="#6666CC", alpha=0.15,
                          linestyle="--", zorder=4)
        ax_main.add_patch(rect)
    ax_main.plot([], [], color="#6666CC", linestyle="--",
                 label="Frozen planet region")

    cbar = plt.colorbar(im, ax=ax_main, pad=0.015, shrink=0.85)
    cbar.set_label(r"$\log_{10}$(gap ratio)", fontsize=12)

    ax_main.set_xlabel(r"$\phi$ (degrees)", fontsize=12)
    ax_main.set_ylabel(r"$\mu = r_{13}/r_{12}$", fontsize=12)
    ax_main.set_xlim(phi_deg.min() - 2, phi_deg.max() + 2)
    ax_main.set_ylim(mu.min(), mu.max())

    ax_main.legend(loc="center right", fontsize=9, framealpha=0.92,
                   borderpad=0.8, handlelength=1.8)

    stats_text = (f"log-gap range: [{h_lg.min():.1f}, {h_lg.max():.1f}]  |  "
                  f"mean: {h_lg.mean():.1f}")
    ax_main.text(0.5, 0.02, stats_text, transform=ax_main.transAxes,
                 ha="center", fontsize=10, color="white",
                 style="italic")

    fig.suptitle(
        "Helium Stability Atlas with Classical Orbit Families\n"
        r"$N\!=\!3,\ d\!=\!2,\ V\!=\!1/r,\ "
        r"\mathrm{charges}\!=\!(+2,\,-1,\,-1)$"
        f"  |  "
        r"$\varepsilon$"
        f" = {best_eps:.0e}",
        fontsize=14, y=0.98)

    out_path = os.path.join(out_dir, "helium_orbits_atlas.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)

    print(f"\n{'='*70}")
    print("ORBIT CONFIGURATIONS vs GAP RATIO")
    print(f"{'='*70}")
    print(f"  {'Orbit':32s} {'Location':22s} {'log10(gap)':14s}")
    print(f"  {'-'*32} {'-'*22} {'-'*14}")

    orbit_slices = [
        ("eZe collinear", "phi=180, mu varies",
         h_lg[:, -1]),
        ("Zee collinear", "phi=0, mu varies",
         h_lg[:, 0]),
        ("Wannier ridge", "mu=1.0, phi varies",
         h_lg[np.argmin(np.abs(mu - 1.0)), :]),
        ("Equilateral", "mu=1.0, phi=60",
         np.array([h_lg[np.argmin(np.abs(mu - 1.0)),
                         np.argmin(np.abs(phi_deg - 60))]])),
        ("Frozen planet (mu<<1)", "mu~0.2, phi~6",
         h_lg[:5, :5].ravel()),
        ("Frozen planet (mu>>1)", "mu~3.0, phi~6",
         h_lg[-5:, :5].ravel()),
    ]

    for name, loc, gaps in orbit_slices:
        if gaps.size == 1:
            gstr = f"{gaps[0]:.2f}"
        else:
            gstr = f"[{gaps.min():.1f} - {gaps.max():.1f}]"
        print(f"  {name:32s} {loc:22s} {gstr:14s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Helium atlas comparison tool")
    ap.add_argument("command", nargs="?", default="all",
                    choices=["compare", "orbits", "all"],
                    help="compare, orbits, or all (default)")
    ap.add_argument("--potential", type=str, default="1/r",
                    choices=["1/r", "1/r2"],
                    help="Potential type to compare (default: 1/r)")
    args = ap.parse_args()

    if args.command in ("compare", "all"):
        run_compare(potential_type=args.potential)

    if args.command in ("orbits", "all"):
        run_orbits(potential_type=args.potential)


if __name__ == "__main__":
    main()
