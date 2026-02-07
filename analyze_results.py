#!/usr/bin/env python3
"""
Post-hoc Analysis of Lie Algebra Growth Results
=================================================

Loads checkpointed expressions, performs SVD analysis with
configurable parameters, searches OEIS for the dimension sequence,
and generates publication-quality plots.

Usage
-----
    python analyze_results.py                  # standard analysis
    python analyze_results.py --samples 2000   # higher precision
    python analyze_results.py --oeis           # search OEIS for sequence
"""

import os
import sys
import argparse
import pickle
import numpy as np
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"

import sympy as sp
from exact_growth import (
    ALL_VARS, sample_phase_space, svd_gap_analysis,
    _make_flat_func,
)


def load_all_checkpoints(checkpoint_dir="checkpoints"):
    """Load the highest-level checkpoint."""
    files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pkl")]
    )
    if not files:
        print("No checkpoints found!")
        sys.exit(1)

    path = os.path.join(checkpoint_dir, files[-1])
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    print(f"Loaded: {path}  (level {data['level']}, "
          f"{len(data['exprs'])} expressions)")
    return data


def lambdify_all(exprs):
    """Lambdify expressions with fallback for complex ones."""
    print(f"Lambdifying {len(exprs)} expressions...")
    t0 = time()
    funcs = []
    for idx, expr in enumerate(exprs):
        if (idx + 1) % 20 == 0 or idx == len(exprs) - 1:
            print(f"  {idx+1}/{len(exprs)}  [{time()-t0:.1f}s]")
        try:
            f = sp.lambdify(ALL_VARS, expr, modules="numpy", cse=True)
        except RecursionError:
            f = _make_flat_func(expr, f"_f{idx}")
        funcs.append(f)
    print(f"Total: {time()-t0:.1f}s")
    return funcs


def evaluate_all(funcs, Z_qp, Z_u):
    """Evaluate all lambdified functions at sample points."""
    args = ([Z_qp[:, i] for i in range(12)] +
            [Z_u[:, i] for i in range(3)])
    cols = []
    for f in funcs:
        val = f(*args)
        cols.append(np.atleast_1d(val).ravel())
    return np.column_stack(cols)


def run_analysis(args):
    data = load_all_checkpoints()
    all_exprs = data["exprs"]
    all_names = data["names"]
    all_levels = data["levels"]
    max_level = max(all_levels)

    # Phase-space sampling
    Z_qp, Z_u = sample_phase_space(args.samples, seed=args.seed)
    print(f"Sample points: {Z_qp.shape[0]}")

    # Lambdify and evaluate
    funcs = lambdify_all(all_exprs)
    print("Evaluating...", end=" ", flush=True)
    t0 = time()
    eval_matrix = evaluate_all(funcs, Z_qp, Z_u)
    print(f"done [{time()-t0:.1f}s]")
    print(f"Matrix shape: {eval_matrix.shape}")

    # Per-level SVD
    dims = {}
    new_gens = {}
    for lv in range(max_level + 1):
        mask = [i for i, l in enumerate(all_levels) if l <= lv]
        sub = eval_matrix[:, mask]
        rank, svals = svd_gap_analysis(sub, label=f"(level ≤ {lv})")
        dims[lv] = rank
        if lv == 0:
            new_gens[lv] = rank
        else:
            new_gens[lv] = rank - dims[lv - 1]
        print(f"  ==> dim(≤{lv}) = {rank},  new at level {lv}: {new_gens[lv]}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    dim_seq = [dims[lv] for lv in range(max_level + 1)]
    new_seq = [new_gens[lv] for lv in range(max_level + 1)]

    a114491 = [2, 3, 6, 17, 69, 407, 3808, 75165]

    print(f"\n  {'Level':>5} | {'Cum.Dim':>8} | {'New':>5} | "
          f"{'Ratio':>8} | {'A114491':>8} | {'Match':>6}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for lv in range(max_level + 1):
        d = dims[lv]
        n = new_gens[lv]
        r = dims[lv] / dims[lv - 1] if lv > 0 and dims[lv - 1] > 0 else 0
        a = a114491[lv + 1] if lv + 1 < len(a114491) else "?"
        m = "YES" if d == a else "no"
        print(f"  {lv:>5} | {d:>8} | {n:>5} | {r:>8.2f} | "
              f"{str(a):>8} | {m:>6}")

    print(f"\n  Dimension sequence:   {dim_seq}")
    print(f"  New-generator sequence: {new_seq}")

    # Growth rate analysis
    if len(dim_seq) >= 3:
        log_dims = np.log(np.array(dim_seq, dtype=float))
        levels = np.arange(len(dim_seq), dtype=float)
        # Fit  log(d) = a * level + b  (exponential growth)
        coeffs = np.polyfit(levels, log_dims, 1)
        growth_rate = np.exp(coeffs[0])
        print(f"\n  Exponential fit: d(L) ~ {np.exp(coeffs[1]):.2f} * "
              f"{growth_rate:.2f}^L")
        print(f"  Gelfand-Kirillov dimension estimate: "
              f"{coeffs[0]:.2f} (log growth rate)")

    # OEIS search hint
    if args.oeis:
        print(f"\n  Search OEIS for: {','.join(str(d) for d in dim_seq)}")
        print(f"  https://oeis.org/search?q={','.join(str(d) for d in dim_seq)}")

    # Plots
    if not args.no_plot:
        _make_plots(dims, new_gens, eval_matrix, all_levels, max_level)


def _make_plots(dims, new_gens, eval_matrix, all_levels, max_level):
    """Generate analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Growth curve (linear)
        lvls = sorted(dims.keys())
        d_vals = [dims[l] for l in lvls]
        axes[0, 0].plot(lvls, d_vals, "o-", linewidth=2, markersize=8)
        axes[0, 0].set_xlabel("Bracket Level")
        axes[0, 0].set_ylabel("Cumulative Dimension")
        axes[0, 0].set_title("Algebra Dimension Growth")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Growth curve (log scale)
        axes[0, 1].semilogy(lvls, d_vals, "o-", linewidth=2, markersize=8,
                            label="Measured")

        # A114491 comparison
        a114491 = [3, 6, 17, 69, 407, 3808]
        a_lvls = list(range(min(len(a114491), max_level + 1)))
        axes[0, 1].semilogy(a_lvls, a114491[:len(a_lvls)], "s--",
                            linewidth=1.5, markersize=6, alpha=0.6,
                            label="A114491")
        axes[0, 1].set_xlabel("Bracket Level")
        axes[0, 1].set_ylabel("Cumulative Dimension (log)")
        axes[0, 1].set_title("Growth Comparison")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. New generators per level
        n_vals = [new_gens[l] for l in lvls]
        axes[1, 0].bar(lvls, n_vals, alpha=0.7)
        axes[1, 0].set_xlabel("Bracket Level")
        axes[1, 0].set_ylabel("New Generators")
        axes[1, 0].set_title("New Independent Generators per Level")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # 4. Full SVD spectrum
        norms = np.linalg.norm(eval_matrix, axis=0)
        norms[norms < 1e-15] = 1.0
        _, s, _ = np.linalg.svd(eval_matrix / norms, full_matrices=False)
        axes[1, 1].semilogy(range(1, len(s) + 1), s / s[0],
                            "o-", markersize=2, alpha=0.7)
        axes[1, 1].set_xlabel("Index")
        axes[1, 1].set_ylabel("Relative Singular Value")
        axes[1, 1].set_title("SVD Spectrum (all generators)")
        axes[1, 1].grid(True, alpha=0.3)

        # Mark key positions
        for pos in [3, 6, 17, 69, 116]:
            if pos < len(s):
                axes[1, 1].axvline(x=pos, color="red", linestyle="--",
                                   alpha=0.3)
                axes[1, 1].text(pos, 0.3, f" {pos}", color="red",
                                fontsize=8)

        plt.tight_layout()
        plt.savefig("growth_analysis.png", dpi=150)
        print(f"\n  Plots saved to growth_analysis.png")
        plt.close()
    except Exception as e:
        print(f"\n  (Plots skipped: {e})")


def main():
    ap = argparse.ArgumentParser(
        description="Analyse Lie-algebra growth results"
    )
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space sample points (default: 500)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--oeis", action="store_true",
                    help="Print OEIS search URL for dimension sequence")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip plot generation")
    args = ap.parse_args()

    run_analysis(args)


if __name__ == "__main__":
    main()
