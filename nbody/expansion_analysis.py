#!/usr/bin/env python3
"""
Multi-System Universality Survey -- Analysis and Plotting

Generates comparative visualizations from the dimension sequence and atlas
results produced by run_expansion_dimseq.py and run_expansion_atlas.py.

Plots
-----
1. Universality Table    -- heatmap of dim sequences across all scenarios
2. Growth Curves         -- dimension vs level for all scenarios (log scale)
3. SVD Comparison        -- overlaid singular value spectra by potential type
4. Atlas Comparison Grid -- rank maps from representative scenarios
5. Category Summaries    -- per-category detail plots

Usage:
    python expansion_analysis.py                      # from local results
    python expansion_analysis.py --pull-s3             # download results first
    python expansion_analysis.py --dimseq-only         # only dim seq plots
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "expansion_results")

S3_BUCKET = os.environ.get("S3_BUCKET", "3body-compute-290318")

EXPECTED = [3, 6, 17, 116]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def pull_from_s3():
    """Download results from S3."""
    print("Pulling dimension sequence results from S3...")
    subprocess.run([
        "aws", "s3", "cp",
        f"s3://{S3_BUCKET}/results/expansion_dimseq/expansion_dimseq_summary.json",
        os.path.join(SCRIPT_DIR, "expansion_dimseq_summary.json"),
    ], capture_output=True)

    subprocess.run([
        "aws", "s3", "cp",
        f"s3://{S3_BUCKET}/results/expansion_dimseq/expansion_dimseq_completion.json",
        os.path.join(SCRIPT_DIR, "expansion_dimseq_completion.json"),
    ], capture_output=True)

    print("Pulling atlas results from S3...")
    subprocess.run([
        "aws", "s3", "sync",
        f"s3://{S3_BUCKET}/atlas_targeted/",
        os.path.join(PROJECT_DIR, "atlas_targeted/"),
    ], capture_output=True, timeout=600)


def load_dimseq_results():
    """Load dimension sequence results from completion manifest."""
    path = os.path.join(SCRIPT_DIR, "expansion_dimseq_completion.json")
    if not os.path.exists(path):
        print(f"  No results found at {path}")
        return {}

    with open(path) as f:
        manifest = json.load(f)

    results = {}
    for key, entry in manifest.items():
        if entry.get("status") == "complete":
            result = entry.get("result")
            if isinstance(result, list):
                results[key] = result
            elif isinstance(result, dict) and "error" not in result:
                results[key] = result
    return results


def plot_universality_table(results):
    """
    Centerpiece: a colored table showing dimension sequences for all scenarios.
    Green = matches [3,6,17,116], red = different, grey = not computed.
    """
    from expansion_configs import SCENARIOS, CATEGORIES

    ensure_output_dir()

    ordered_keys = []
    for cat_key, _ in CATEGORIES:
        for key in SCENARIOS:
            if SCENARIOS[key]["category"] == cat_key:
                ordered_keys.append(key)

    n_scenarios = len(ordered_keys)
    n_levels = 4

    fig, ax = plt.subplots(figsize=(12, max(8, n_scenarios * 0.45 + 2)))

    category_colors = {
        "gravitational": "#E8F5E9",
        "atomic": "#E3F2FD",
        "nuclear": "#FFF3E0",
        "plasma": "#F3E5F5",
        "pn": "#FFEBEE",
        "exotic": "#F5F5F5",
    }

    for i, key in enumerate(ordered_keys):
        cfg = SCENARIOS[key]
        cat = cfg["category"]
        bg_color = category_colors.get(cat, "#FFFFFF")

        ax.add_patch(plt.Rectangle(
            (-0.5, n_scenarios - i - 1.5), n_levels + 5, 1,
            facecolor=bg_color, edgecolor="none", zorder=0
        ))

        seq = results.get(key)
        label = cfg["label"]
        pot = cfg["potential"]

        ax.text(-0.3, n_scenarios - i - 1, label,
                ha="right", va="center", fontsize=9, fontweight="bold")
        ax.text(-0.1, n_scenarios - i - 1, f"({pot})",
                ha="right", va="center", fontsize=7, color="grey")

        if seq is None or (isinstance(seq, dict) and "error" in seq):
            for j in range(n_levels):
                ax.add_patch(plt.Rectangle(
                    (j - 0.4, n_scenarios - i - 1.4), 0.8, 0.8,
                    facecolor="#E0E0E0", edgecolor="white", linewidth=1
                ))
                ax.text(j, n_scenarios - i - 1, "—",
                        ha="center", va="center", fontsize=10, color="grey")
        else:
            for j in range(min(len(seq), n_levels)):
                val = seq[j]
                expected = EXPECTED[j] if j < len(EXPECTED) else None

                if expected is not None and val == expected:
                    color = "#4CAF50"
                    text_color = "white"
                elif expected is not None:
                    color = "#F44336"
                    text_color = "white"
                else:
                    color = "#FFC107"
                    text_color = "black"

                ax.add_patch(plt.Rectangle(
                    (j - 0.4, n_scenarios - i - 1.4), 0.8, 0.8,
                    facecolor=color, edgecolor="white", linewidth=1.5,
                    zorder=2
                ))
                ax.text(j, n_scenarios - i - 1, str(val),
                        ha="center", va="center", fontsize=11,
                        fontweight="bold", color=text_color, zorder=3)

    for j in range(n_levels):
        ax.text(j, n_scenarios - 0.3, f"d{j}",
                ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(j, n_scenarios + 0.3, f"({EXPECTED[j]})",
                ha="center", va="center", fontsize=8, color="grey")

    current_cat = None
    for i, key in enumerate(ordered_keys):
        cat = SCENARIOS[key]["category"]
        if cat != current_cat:
            current_cat = cat
            cat_label = dict(
                (k, l) for k, l in
                [("gravitational", "Gravitational"),
                 ("atomic", "Atomic/Coulomb"),
                 ("nuclear", "Nuclear"),
                 ("plasma", "Plasma"),
                 ("pn", "Post-Newtonian"),
                 ("exotic", "Exotic")]
            ).get(cat, cat)
            ax.axhline(y=n_scenarios - i - 0.5, color="grey",
                       linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xlim(-5, n_levels - 0.3)
    ax.set_ylim(-0.6, n_scenarios + 0.8)
    ax.axis("off")

    n_match = sum(1 for s in results.values()
                  if isinstance(s, list)
                  and all(s[j] == EXPECTED[j]
                          for j in range(min(len(s), len(EXPECTED)))))
    n_total = len(results)
    n_diff = n_total - n_match

    ax.set_title(
        f"Universality Survey: Dimension Sequences [3, 6, 17, 116]\n"
        f"{n_match}/{n_total} match  |  {n_diff} differ  |  "
        f"{n_scenarios - n_total} pending",
        fontsize=14, fontweight="bold", pad=20
    )

    legend_items = [
        ("#4CAF50", "Matches expected"),
        ("#F44336", "Differs from expected"),
        ("#E0E0E0", "Not yet computed"),
    ]
    for idx, (color, label) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle(
            (n_levels - 3 + idx * 1.5, -0.3), 0.3, 0.3,
            facecolor=color, edgecolor="grey", linewidth=0.5
        ))
        ax.text(n_levels - 3 + idx * 1.5 + 0.4, -0.15, label,
                fontsize=7, va="center")

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "universality_table.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_growth_curves(results):
    """Dimension vs level for all scenarios on log scale."""
    from expansion_configs import SCENARIOS, CATEGORIES

    ensure_output_dir()

    cat_colors = {
        "gravitational": "#2E7D32",
        "atomic": "#1565C0",
        "nuclear": "#E65100",
        "plasma": "#7B1FA2",
        "pn": "#C62828",
        "exotic": "#424242",
    }
    cat_markers = {
        "gravitational": "o",
        "atomic": "s",
        "nuclear": "^",
        "plasma": "D",
        "pn": "v",
        "exotic": "P",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(range(4), EXPECTED, "k--", linewidth=2.5, alpha=0.6, zorder=10,
            label="Expected [3, 6, 17, 116]")

    plotted_cats = set()
    for key, seq in results.items():
        if not isinstance(seq, list):
            continue
        cfg = SCENARIOS.get(key, {})
        cat = cfg.get("category", "unknown")
        color = cat_colors.get(cat, "grey")
        marker = cat_markers.get(cat, "x")
        label = cfg.get("label", key)

        cat_label = None
        if cat not in plotted_cats:
            cat_label = dict(
                [("gravitational", "Gravitational"),
                 ("atomic", "Atomic/Coulomb"),
                 ("nuclear", "Nuclear (Yukawa)"),
                 ("plasma", "Plasma"),
                 ("pn", "Post-Newtonian"),
                 ("exotic", "Exotic")]
            ).get(cat, cat)
            plotted_cats.add(cat)

        levels = list(range(len(seq)))
        ax.plot(levels, seq, color=color, marker=marker, markersize=5,
                alpha=0.7, linewidth=1.2, label=cat_label)

    ax.set_yscale("log")
    ax.set_xlabel("Bracket Level", fontsize=12)
    ax.set_ylabel("Algebra Dimension", fontsize=12)
    ax.set_title("Growth Curves: All Scenarios", fontsize=14, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Level 0", "Level 1", "Level 2", "Level 3"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "growth_curves.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"  Saved: {fname}")


def plot_category_summary(results, category, cat_label):
    """Per-category bar chart of dimension sequences."""
    from expansion_configs import SCENARIOS

    ensure_output_dir()

    cat_scenarios = {k: v for k, v in results.items()
                     if isinstance(v, list)
                     and k in SCENARIOS
                     and SCENARIOS[k]["category"] == category}
    if not cat_scenarios:
        return

    labels = [SCENARIOS[k]["label"] for k in cat_scenarios]
    sequences = list(cat_scenarios.values())

    fig, axes = plt.subplots(1, 4, figsize=(14, max(4, len(labels) * 0.5 + 1)),
                             sharey=True)

    for lv, ax in enumerate(axes):
        vals = [seq[lv] if lv < len(seq) else 0 for seq in sequences]
        exp = EXPECTED[lv]
        colors = ["#4CAF50" if v == exp else "#F44336" for v in vals]

        y_pos = range(len(labels))
        ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.6)

        ax.axvline(x=exp, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"Level {lv} (exp: {exp})", fontsize=10)
        ax.set_xlabel("Dimension")

        if lv == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)

        for j, v in enumerate(vals):
            ax.text(v + 0.5, j, str(v), va="center", fontsize=8)

    fig.suptitle(f"{cat_label} -- Dimension Sequences",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"category_{category}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_atlas_comparison():
    """Grid of rank maps from atlas-enabled scenarios."""
    from expansion_configs import SCENARIOS, get_atlas_scenarios

    ensure_output_dir()
    atlas_dir = os.path.join(PROJECT_DIR, "atlas_targeted")

    atlas_keys = get_atlas_scenarios()
    available = []

    for key in atlas_keys:
        cfg = SCENARIOS[key]
        pot_dir_name = cfg["potential"].replace("/", "_").replace("^", "")
        region_dir = os.path.join(atlas_dir, pot_dir_name, "lagrange")
        rank_file = os.path.join(region_dir, "rank_map.npy")
        if os.path.exists(rank_file):
            available.append((key, cfg, rank_file, region_dir))

    if not available:
        print("  No atlas data found for comparison grid.")
        return

    n = len(available)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    vmin, vmax = 0, 120

    for idx, (key, cfg, rank_file, region_dir) in enumerate(available):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        rank_map = np.load(rank_file)
        mu_file = os.path.join(region_dir, "mu_vals.npy")
        phi_file = os.path.join(region_dir, "phi_vals.npy")

        if os.path.exists(mu_file) and os.path.exists(phi_file):
            mu_vals = np.load(mu_file)
            phi_vals = np.load(phi_file)
            extent = [mu_vals[0], mu_vals[-1],
                      np.degrees(phi_vals[0]), np.degrees(phi_vals[-1])]
        else:
            extent = None

        im = ax.imshow(rank_map.T, origin="lower", aspect="auto",
                       extent=extent, vmin=vmin, vmax=vmax,
                       cmap="viridis")
        ax.set_title(f"{cfg['label']}\n({cfg['potential']})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("mu", fontsize=8)
        ax.set_ylabel("phi (deg)", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(available), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle("Atlas Comparison: Lagrange Region Rank Maps",
                 fontsize=14, fontweight="bold")
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Rank")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    fname = os.path.join(OUTPUT_DIR, "atlas_comparison_grid.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def generate_text_summary(results):
    """Write a text summary with documented-only scenarios."""
    from expansion_configs import SCENARIOS, CATEGORIES, DOCUMENTED_ONLY

    ensure_output_dir()

    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-SYSTEM UNIVERSALITY SURVEY -- RESULTS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    n_match = 0
    n_differ = 0
    n_error = 0
    n_pending = 0

    for key in SCENARIOS:
        seq = results.get(key)
        if seq is None:
            n_pending += 1
        elif isinstance(seq, dict) and "error" in seq:
            n_error += 1
        elif isinstance(seq, list):
            match = all(seq[j] == EXPECTED[j]
                        for j in range(min(len(seq), len(EXPECTED))))
            if match:
                n_match += 1
            else:
                n_differ += 1

    lines.append(f"Total scenarios:     {len(SCENARIOS)}")
    lines.append(f"Match [3,6,17,116]:  {n_match}")
    lines.append(f"Differ:              {n_differ}")
    lines.append(f"Errors:              {n_error}")
    lines.append(f"Pending:             {n_pending}")
    lines.append("")

    if n_differ == 0 and n_pending == 0 and n_error == 0:
        lines.append("VERDICT: UNIVERSALITY HOLDS across all tested systems")
    elif n_differ > 0:
        lines.append("VERDICT: UNIVERSALITY BROKEN -- see details below")
    else:
        lines.append("VERDICT: INCOMPLETE -- some scenarios pending or errored")

    lines.append("")
    lines.append("-" * 70)

    for cat_key, cat_label in CATEGORIES:
        cat_scenarios = [k for k in SCENARIOS
                         if SCENARIOS[k]["category"] == cat_key]
        if not cat_scenarios:
            continue

        lines.append(f"\n{cat_label}:")
        lines.append("-" * 40)

        for key in cat_scenarios:
            cfg = SCENARIOS[key]
            seq = results.get(key)

            if seq is None:
                status = "PENDING"
                seq_str = "—"
            elif isinstance(seq, dict) and "error" in seq:
                status = f"ERROR: {seq['error']}"
                seq_str = "—"
            elif isinstance(seq, list):
                match = all(seq[j] == EXPECTED[j]
                            for j in range(min(len(seq), len(EXPECTED))))
                status = "MATCH" if match else "DIFFERS"
                seq_str = str(seq)
            else:
                status = "UNKNOWN"
                seq_str = str(seq)

            lines.append(f"  {cfg['label']:.<40} {seq_str:>20}  [{status}]")

    lines.append("")
    lines.append("=" * 70)
    lines.append("SCENARIOS NOT COMPUTED (documented only)")
    lines.append("=" * 70)

    for key, info in DOCUMENTED_ONLY.items():
        lines.append(f"\n  {info['label']}")
        lines.append(f"  Potential: {info['potential_form']}")
        lines.append(f"  Why skipped: {info['reason_skipped']}")
        lines.append(f"  Pathway forward: {info['pathway']}")

    text = "\n".join(lines)

    fname = os.path.join(OUTPUT_DIR, "universality_survey_summary.txt")
    with open(fname, "w") as f:
        f.write(text)
    print(f"  Saved: {fname}")

    print("\n" + text)


def main():
    ap = argparse.ArgumentParser(
        description="Multi-System Universality Survey -- analysis and plots")
    ap.add_argument("--pull-s3", action="store_true",
                    help="Download results from S3 before analysis")
    ap.add_argument("--dimseq-only", action="store_true",
                    help="Only generate dimension sequence plots")
    args = ap.parse_args()

    if args.pull_s3:
        pull_from_s3()

    sys.path.insert(0, SCRIPT_DIR)

    print("=" * 70)
    print("MULTI-SYSTEM UNIVERSALITY SURVEY -- ANALYSIS")
    print("=" * 70)
    print(f"  Output directory: {OUTPUT_DIR}")
    print()

    results = load_dimseq_results()
    print(f"  Loaded {len(results)} scenario results\n")

    print("--- Universality Table ---")
    plot_universality_table(results)

    print("\n--- Growth Curves ---")
    plot_growth_curves(results)

    print("\n--- Category Summaries ---")
    from expansion_configs import CATEGORIES
    for cat_key, cat_label in CATEGORIES:
        plot_category_summary(results, cat_key, cat_label)

    if not args.dimseq_only:
        print("\n--- Atlas Comparison Grid ---")
        plot_atlas_comparison()

    print("\n--- Text Summary ---")
    generate_text_summary(results)

    print(f"\n  All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
