#!/usr/bin/env python3
"""
Noise plateau mapping (gap_workplan section 4.6).

Sweeps the SVD rank-detection threshold tau across decades and watches
how the reported algebra dimension responds, at:

    Panel A: Equal-mass L=3 atlas (1/r), three exemplar grid points
             (generic / Lagrange / Euler).  Source: atlas_output_hires/.
    Panel B: Mass-ratio sweep at L=2.  33 m_3 values from 1 to 10^10.
             Source: data/mass_ratio_sweep.json (18 SVs per row).
    Panel C: Same as B but for 1/r^2 (Calogero-Moser) at L=3.  Source:
             atlas_output_hires/1_r2/eps_*/sv_spectra.npy.

Outputs:
    figures/noise_plateau.png
    results/noise_plateau/noise_plateau_data.json

For the headline rank=116 result we expect a clean plateau across the
[1e-12, 1e-1] decade at moderate conditioning.  Where the plateau
narrows or fragments, float64 SVD has run out of precision -- the
mechanistic explanation for why extreme mass ratios appear to undercount.

The script is pure post-processing: no fresh symbolic compute.  Run
from the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("matplotlib required: %s" % e, file=sys.stderr)
    raise

REPO = Path(__file__).resolve().parent

# Atlas grid -- 100x100 in (mu, phi).  Matches the convention used in
# stability_atlas.py / dirac_analysis_from_svd.py.
MU = np.linspace(0.5, 2.0, 100)
PHI_DEG = np.linspace(0, 180, 100)

# Three exemplar grid points (mu, phi indices) at d=2.
#   generic  -- some non-symmetric point well away from collinear / Lagrange
#   Lagrange -- equilateral triangle, mu=1, phi=60 deg
#   Euler    -- collinear, phi near 0
EXEMPLAR_GRID_POINTS = {
    "generic":  (int(np.argmin(np.abs(MU - 1.3))),
                 int(np.argmin(np.abs(PHI_DEG - 95)))),
    "Lagrange": (int(np.argmin(np.abs(MU - 1.0))),
                 int(np.argmin(np.abs(PHI_DEG - 60)))),
    "Euler":    (int(np.argmin(np.abs(MU - 1.0))),
                 int(np.argmin(np.abs(PHI_DEG - 5)))),
}

# Thresholds: 31 decades-spaced points over [1e-18, 1e-1].  Dense enough
# to resolve plateau edges; coarse enough to keep the figure readable.
LOG_THRESHOLDS = np.linspace(-18, -1, 35)
THRESHOLDS = 10.0 ** LOG_THRESHOLDS

# Mass-ratio cuts to highlight in Panel B.  Chosen to cover from
# equal-mass to extreme conditioning (matching the gap_workplan request
# for "equal, moderate (10:1), extreme (10^6:1) and beyond").
HIGHLIGHTED_M3 = [1.0, 10.0, 100.0, 1000.0, 1e6, 1e10]


# ---------------------------------------------------------------------------

def reported_rank(sv: np.ndarray, threshold: float) -> int:
    """Count singular values strictly greater than threshold."""
    return int(np.sum(sv > threshold))


def plateau_width(svs: np.ndarray, expected_rank: int) -> tuple[float, float]:
    """Return (tau_low, tau_high) defining the widest interval over which
    the reported rank equals expected_rank.

    Returns (nan, nan) if no such interval exists at this configuration.
    """
    if svs is None or len(svs) == 0:
        return (float("nan"), float("nan"))
    ranks = np.array([reported_rank(svs, t) for t in THRESHOLDS])
    mask = ranks == expected_rank
    if not mask.any():
        return (float("nan"), float("nan"))
    # Longest contiguous run of True
    idxs = np.where(mask)[0]
    # Find contiguous runs
    best_run = (idxs[0], idxs[0])
    cur_start = idxs[0]
    prev = idxs[0]
    for k in idxs[1:]:
        if k == prev + 1:
            prev = k
        else:
            if prev - cur_start > best_run[1] - best_run[0]:
                best_run = (cur_start, prev)
            cur_start = k
            prev = k
    if prev - cur_start > best_run[1] - best_run[0]:
        best_run = (cur_start, prev)
    return (float(THRESHOLDS[best_run[0]]), float(THRESHOLDS[best_run[1]]))


# ---------------------------------------------------------------------------
# Panel A: L=3 atlas SV cubes

def load_atlas_sv(potential_dir: str, eps_str: str) -> np.ndarray:
    p = REPO / "atlas_output_hires" / potential_dir / ("eps_%s" % eps_str) / "sv_spectra.npy"
    return np.load(p, mmap_mode="r")  # (100, 100, 156)


def panel_a_data(potential_dir: str, eps_str: str = "2e-03") -> dict:
    """For each exemplar grid point, return ranks(tau)."""
    cube = load_atlas_sv(potential_dir, eps_str)
    out = {}
    for name, (mu_i, phi_i) in EXEMPLAR_GRID_POINTS.items():
        sv = np.array(cube[mu_i, phi_i, :])  # (156,)
        sv_sorted = np.sort(sv)[::-1]
        ranks = [reported_rank(sv_sorted, t) for t in THRESHOLDS]
        out[name] = {
            "mu": float(MU[mu_i]),
            "phi_deg": float(PHI_DEG[phi_i]),
            "sv_top10":   sv_sorted[:10].tolist(),
            "sv_around_gap": sv_sorted[max(0, 116 - 5): 116 + 5].tolist(),
            "ranks": ranks,
            "plateau_at_116": plateau_width(sv_sorted, 116),
        }
    return {
        "potential_dir": potential_dir,
        "eps": eps_str,
        "points": out,
    }


# ---------------------------------------------------------------------------
# Panel B: Mass-ratio sweep at L=2

def panel_b_data() -> dict:
    path = REPO / "data" / "mass_ratio_sweep.json"
    sweep = json.loads(path.read_text())
    out_rows = []
    for row in sweep:
        sv = np.array(row["level_2"]["singular_values"])
        sv_sorted = np.sort(sv)[::-1]
        ranks = [reported_rank(sv_sorted, t) for t in THRESHOLDS]
        plat = plateau_width(sv_sorted, 17)
        out_rows.append({
            "m3":         row["m3"],
            "m3_log10":   row["m3_log10"],
            "sv":         sv_sorted.tolist(),
            "ranks":      ranks,
            "plateau_at_17": plat,
        })
    return {
        "level": 2,
        "expected_rank": 17,
        "rows": out_rows,
    }


# ---------------------------------------------------------------------------
# Panel C: L=3 mass-ratio plateau (1/r^2 atlas, equal mass only -- no
# L=3 SV vectors at unequal masses are stored in the project.  Instead
# we read the 1/r^2 atlas at 5 epsilons and treat each epsilon as a
# proxy for a different "effective conditioning" regime: smaller eps =
# more conditioning sensitivity.

def panel_c_data() -> dict:
    out_eps = {}
    for eps_str in ["2e-03", "1e-03", "5e-04", "2e-04", "1e-04"]:
        cube = load_atlas_sv("1_r2", eps_str)
        plats = []
        for mu_i, phi_i in zip(*np.where(np.ones(cube.shape[:2], dtype=bool))):
            sv = np.sort(np.array(cube[mu_i, phi_i, :]))[::-1]
            tau_lo, tau_hi = plateau_width(sv, 116)
            if not np.isnan(tau_lo):
                plats.append(np.log10(tau_hi) - np.log10(tau_lo))
            # short-circuit after a representative sample to keep this
            # fast; 1000 grid points is plenty for a histogram
            if len(plats) >= 1000:
                break
        # Convert array to handle NaNs
        widths = np.array(plats, dtype=float)
        out_eps[eps_str] = {
            "n_samples":    len(widths),
            "mean_log_width": float(np.mean(widths)),
            "median_log_width": float(np.median(widths)),
            "min_log_width": float(np.min(widths)),
            "max_log_width": float(np.max(widths)),
            "histogram": list(zip(*np.histogram(widths, bins=15))),
        }
        # Tuple-of-list zip back to JSON-friendly form
        hist_counts, hist_edges = np.histogram(widths, bins=15)
        out_eps[eps_str]["histogram"] = {
            "counts": hist_counts.tolist(),
            "edges":  hist_edges.tolist(),
        }
    return {
        "potential_dir": "1_r2",
        "expected_rank": 116,
        "by_eps": out_eps,
    }


# ---------------------------------------------------------------------------
# Plotting

def make_figure(panel_a_1r: dict, panel_a_1r2: dict, panel_b: dict,
                panel_c: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A1 -- 1/r equal-mass L=3 plateaus at three exemplar points
    ax = axes[0, 0]
    for name, info in panel_a_1r["points"].items():
        ax.plot(THRESHOLDS, info["ranks"], marker=".", label=name)
    ax.set_xscale("log")
    ax.set_xlim(THRESHOLDS[-1], THRESHOLDS[0])
    ax.set_ylim(0, 160)
    ax.axhline(116, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel(r"SVD threshold $\tau$")
    ax.set_ylabel("Reported rank")
    ax.set_title("Panel A1: 1/r equal-mass L=3 atlas (eps=2e-3)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel A2 -- 1/r^2 equal-mass L=3 plateaus at three exemplar points
    ax = axes[0, 1]
    for name, info in panel_a_1r2["points"].items():
        ax.plot(THRESHOLDS, info["ranks"], marker=".", label=name)
    ax.set_xscale("log")
    ax.set_xlim(THRESHOLDS[-1], THRESHOLDS[0])
    ax.set_ylim(0, 160)
    ax.axhline(116, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel(r"SVD threshold $\tau$")
    ax.set_ylabel("Reported rank")
    ax.set_title("Panel A2: 1/r$^2$ equal-mass L=3 atlas (eps=2e-3)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B -- Mass-ratio plateau at L=2
    ax = axes[1, 0]
    rows = panel_b["rows"]
    # Highlight the chosen m3 values
    cmap = plt.cm.viridis
    log_m3_vals = [r["m3_log10"] for r in rows]
    for row in rows:
        if any(abs(row["m3"] - tgt) / max(tgt, 1.0) < 1e-3
               for tgt in HIGHLIGHTED_M3):
            color = cmap((row["m3_log10"]) / max(log_m3_vals))
            ax.plot(THRESHOLDS, row["ranks"], marker=".",
                    color=color,
                    label=("m_3=%g" % row["m3"]).replace("e+0", "e"))
    ax.set_xscale("log")
    ax.set_xlim(THRESHOLDS[-1], THRESHOLDS[0])
    ax.set_ylim(0, 18)
    ax.axhline(17, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel(r"SVD threshold $\tau$")
    ax.set_ylabel("Reported rank")
    ax.set_title("Panel B: L=2 mass-ratio plateau (1/r, equal $x_3$)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C -- L=3 plateau widths at varying eps (proxy for
    # conditioning regime)
    ax = axes[1, 1]
    eps_strs = ["2e-03", "1e-03", "5e-04", "2e-04", "1e-04"]
    for eps_str in eps_strs:
        info = panel_c["by_eps"][eps_str]
        edges = info["histogram"]["edges"]
        centers = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
        counts = info["histogram"]["counts"]
        ax.plot(centers, counts, marker=".", label="eps=%s" % eps_str)
    ax.set_xlabel(r"Plateau width $\log_{10}(\tau_{high}/\tau_{low})$")
    ax.set_ylabel("Number of grid points (sample of 1000)")
    ax.set_title("Panel C: L=3 plateau-width histogram (1/r$^2$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Noise plateau mapping -- gap_workplan section 4.6",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/noise_plateau",
                    help="Directory for JSON output")
    ap.add_argument("--fig", default="figures/noise_plateau.png",
                    help="Figure output path")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = REPO / args.fig
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    print("Panel A1: 1/r equal-mass atlas ...", flush=True)
    pa1 = panel_a_data("1_r")
    print("Panel A2: 1/r^2 equal-mass atlas ...", flush=True)
    pa2 = panel_a_data("1_r2")
    print("Panel B: L=2 mass-ratio sweep ...", flush=True)
    pb = panel_b_data()
    print("Panel C: 1/r^2 L=3 plateau-width histograms ...", flush=True)
    pc = panel_c_data()

    bundle = {
        "panels": {
            "A1_1r":   pa1,
            "A2_1r2":  pa2,
            "B_l2_mr": pb,
            "C_l3_1r2_widths": pc,
        },
        "thresholds": THRESHOLDS.tolist(),
        "log_thresholds": LOG_THRESHOLDS.tolist(),
        "exemplar_grid_points": {
            k: {"mu_index": v[0], "phi_index": v[1],
                "mu": float(MU[v[0]]), "phi_deg": float(PHI_DEG[v[1]])}
            for k, v in EXEMPLAR_GRID_POINTS.items()
        },
    }
    json_path = out_dir / "noise_plateau_data.json"
    json_path.write_text(json.dumps(bundle, indent=2))
    print("Wrote %s" % json_path)

    print("Making figure ...", flush=True)
    make_figure(pa1, pa2, pb, pc, fig_path)
    print("Wrote %s" % fig_path)

    # ---- Headline numbers for the findings doc ----
    print()
    print("=" * 60)
    print("HEADLINE NUMBERS")
    print("=" * 60)
    print("Panel A1 (1/r generic, plateau at rank=116):", flush=True)
    plat = pa1["points"]["generic"]["plateau_at_116"]
    if not np.isnan(plat[0]):
        print("  tau in [%.2e, %.2e]  --  log10 width = %.1f"
              % (plat[0], plat[1], np.log10(plat[1] / plat[0])))
    else:
        print("  NO plateau at rank 116 found")
    print()
    print("Panel B (L=2 plateau width at rank=17 vs m_3):")
    for row in pb["rows"]:
        if any(abs(row["m3"] - tgt) / max(tgt, 1.0) < 1e-3
               for tgt in HIGHLIGHTED_M3):
            plat = row["plateau_at_17"]
            if not np.isnan(plat[0]):
                width = np.log10(plat[1] / plat[0])
                print("  m_3=%-12g -> plateau width = %4.1f decades"
                      % (row["m3"], width))
            else:
                print("  m_3=%-12g -> NO plateau at rank 17" % row["m3"])


if __name__ == "__main__":
    main()
