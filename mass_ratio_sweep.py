#!/usr/bin/env python3
"""
Mass Ratio Sweep — track dimension sequence and SVD gaps as mass
ratio varies continuously from equal (1:1:1) to extreme (1:1:1e6).

Runs through level 2 (fast, ~2 min each) for many mass ratios.
Outputs JSON data and generates multi-panel diagnostic plots.
"""
import os, sys, json, argparse
import numpy as np
from time import time
from sympy import Rational, Add

os.environ["PYTHONUNBUFFERED"] = "1"

from exact_growth import (
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    total_deriv,
    poisson_bracket, simplify_generator,
    sample_phase_space, lambdify_generators, svd_gap_analysis,
)

sys.setrecursionlimit(50000)

M3_VALUES = [
    1, 1.01, 1.05, 1.1, 1.2, 1.5,
    2, 3, 5, 10, 20, 50,
    100, 500, 1000, 5000,
    1e4, 1e5, 1e6,
]


def build_hamiltonians(m1, m2, m3):
    M1 = Rational(m1).limit_denominator(10**9)
    M2 = Rational(m2).limit_denominator(10**9)
    M3 = Rational(m3).limit_denominator(10**9)
    T1 = (px1**2 + py1**2) / (2 * M1)
    T2 = (px2**2 + py2**2) / (2 * M2)
    T3 = (px3**2 + py3**2) / (2 * M3)
    H12 = T1 + T2 - M1 * M2 * u12
    H13 = T1 + T3 - M1 * M3 * u13
    H23 = T2 + T3 - M2 * M3 * u23
    return H12, H13, H23


def compute_sweep_point(m3, max_level=2, n_samples=500, seed=42):
    """Compute dimension sequence for masses (1, 1, m3) through max_level."""
    t_total = time()
    H12, H13, H23 = build_hamiltonians(1, 1, m3)

    all_exprs = [H12, H13, H23]
    all_levels = [0, 0, 0]
    computed_pairs = {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}

    # Level 1
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        expr = poisson_bracket(all_exprs[i], all_exprs[j])
        expr = simplify_generator(expr)
        all_exprs.append(expr)
        all_levels.append(1)

    # Level 2
    if max_level >= 2:
        frontier = [i for i, lv in enumerate(all_levels) if lv == 1]
        n_existing = len(all_exprs)
        for i in frontier:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                expr = simplify_generator(expr)
                all_exprs.append(expr)
                all_levels.append(2)

    # SVD
    Z_qp, Z_u = sample_phase_space(n_samples, seed)
    evaluate = lambdify_generators(all_exprs)
    eval_matrix = evaluate(Z_qp, Z_u)

    result = {"m3": float(m3), "m3_log10": float(np.log10(m3)) if m3 > 0 else 0}
    for lv in range(max_level + 1):
        mask = [i for i, l in enumerate(all_levels) if l <= lv]
        sub = eval_matrix[:, mask]
        _, s, _ = np.linalg.svd(sub, full_matrices=False)

        # Find best gap
        best_gap = 1.0
        best_idx = -1
        for k in range(min(len(s) - 1, sub.shape[1] - 1)):
            if s[k + 1] > 1e-300:
                gap = s[k] / s[k + 1]
            else:
                gap = float("inf")
            if gap > best_gap and k >= 2:
                best_gap = gap
                best_idx = k
        if best_gap > 1e4:
            rank = best_idx + 1
        else:
            rank = int(np.sum(s / s[0] > 1e-10))

        svs = s.tolist()
        result[f"level_{lv}"] = {
            "dim": rank,
            "n_candidates": len(mask),
            "gap_ratio": float(best_gap) if best_gap < 1e300 else None,
            "gap_index": best_idx + 1 if best_idx >= 0 else None,
            "singular_values": svs[:min(25, len(svs))],
        }

    dims = [result[f"level_{lv}"]["dim"] for lv in range(max_level + 1)]
    result["dims"] = dims
    result["elapsed_s"] = time() - t_total
    return result


def plot_results(results, output_path="mass_ratio_sweep.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m3_vals = [r["m3"] for r in results]
    log_m3 = [r["m3_log10"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mass Ratio Sweep: (1, 1, m₃) gravitational 3-body",
                 fontsize=14, fontweight="bold")

    # Panel A: Dimensions at each level
    ax = axes[0, 0]
    for lv in [0, 1, 2]:
        dims = [r[f"level_{lv}"]["dim"] for r in results]
        ax.plot(log_m3, dims, "o-", label=f"Level {lv}", markersize=5)
    expected = {0: 3, 1: 6, 2: 17}
    for lv, exp in expected.items():
        ax.axhline(exp, color="gray", ls="--", alpha=0.5)
        ax.text(log_m3[-1] + 0.1, exp, f"{exp}", va="center", fontsize=8,
                color="gray")
    ax.set_xlabel("log₁₀(m₃/m₁)")
    ax.set_ylabel("Dimension")
    ax.set_title("A: Dimension vs mass ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: SVD gap ratio at each level
    ax = axes[0, 1]
    for lv in [1, 2]:
        gaps = []
        for r in results:
            g = r[f"level_{lv}"].get("gap_ratio")
            gaps.append(g if g and g < 1e20 else None)
        valid = [(x, g) for x, g in zip(log_m3, gaps) if g is not None]
        if valid:
            xs, gs = zip(*valid)
            ax.semilogy(xs, gs, "o-", label=f"Level {lv}", markersize=5)
    ax.axhline(1e4, color="red", ls="--", alpha=0.5, label="Detection threshold")
    ax.set_xlabel("log₁₀(m₃/m₁)")
    ax.set_ylabel("Gap ratio (sv[k]/sv[k+1])")
    ax.set_title("B: SVD gap ratio vs mass ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Level-2 SV spectrum for select ratios
    ax = axes[1, 0]
    select_indices = []
    select_m3 = [1, 10, 1000, 1e6]
    for target in select_m3:
        best = min(range(len(results)),
                   key=lambda i: abs(results[i]["m3"] - target))
        select_indices.append(best)
    for idx in select_indices:
        r = results[idx]
        svs = r["level_2"]["singular_values"]
        ax.semilogy(range(1, len(svs) + 1), svs, "o-",
                    label=f"m₃={r['m3']:g}", markersize=3)
    ax.axvline(17.5, color="red", ls="--", alpha=0.5, label="Expected dim=17")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("C: Level-2 SV spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Summary table
    ax = axes[1, 1]
    ax.axis("off")
    col_labels = ["m₃", "L0", "L1", "L2", "Gap(L2)"]
    table_data = []
    for r in results:
        m3_str = f"{r['m3']:g}"
        dims = r["dims"]
        gap = r["level_2"].get("gap_ratio")
        gap_str = f"{gap:.1e}" if gap and gap < 1e20 else "inf"
        table_data.append([m3_str, str(dims[0]), str(dims[1]),
                           str(dims[2]), gap_str])
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("D: Summary", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Mass ratio sweep")
    ap.add_argument("--max-level", type=int, default=2)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="mass_ratio_sweep")
    args = ap.parse_args()

    print("=" * 70)
    print("MASS RATIO SWEEP")
    print(f"  Family: (1, 1, m3)")
    print(f"  m3 values: {len(M3_VALUES)} points from {M3_VALUES[0]} to {M3_VALUES[-1]}")
    print(f"  Max level: {args.max_level}, Samples: {args.samples}")
    print("=" * 70)

    results = []
    for i, m3 in enumerate(M3_VALUES):
        print(f"\n[{i+1}/{len(M3_VALUES)}] m3 = {m3:g}  "
              f"(ratio = {m3:.6g}:1)", flush=True)
        r = compute_sweep_point(m3, max_level=args.max_level,
                                n_samples=args.samples, seed=args.seed)
        results.append(r)
        print(f"  dims = {r['dims']}  "
              f"gap(L2) = {r['level_2'].get('gap_ratio', 'N/A')}  "
              f"[{r['elapsed_s']:.1f}s]")

    # Save JSON
    json_path = f"{args.output}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Data saved: {json_path}")

    # Plot
    plot_results(results, f"{args.output}.png")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    unique_seqs = set(tuple(r["dims"]) for r in results)
    print(f"  Unique dimension sequences: {len(unique_seqs)}")
    for seq in sorted(unique_seqs):
        count = sum(1 for r in results if tuple(r["dims"]) == seq)
        m3s = [r["m3"] for r in results if tuple(r["dims"]) == seq]
        print(f"    {list(seq)}: {count} configs (m3 = {m3s})")

    if len(unique_seqs) == 1:
        print("\n  *** MASS INVARIANCE CONFIRMED through level 2 ***")
        print("  All mass ratios produce the same dimension sequence.")
    else:
        print("\n  *** TRANSITION DETECTED ***")
        for r in results:
            if tuple(r["dims"]) != tuple(results[0]["dims"]):
                print(f"  First deviation at m3 = {r['m3']}")
                break


if __name__ == "__main__":
    main()
