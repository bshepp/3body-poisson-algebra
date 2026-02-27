#!/usr/bin/env python3
"""
Atlas Diagnostic Checks
========================

1. Investigate >116 rank anomalies in the grid scan
2. Control epsilon sweep at a generic non-special configuration

Usage:
    python atlas_diagnostics.py anomalies    # just the data inspection
    python atlas_diagnostics.py control      # epsilon sweep at generic point
    python atlas_diagnostics.py              # both
"""

import os
import sys
import numpy as np
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"


def investigate_anomalies():
    """Load saved atlas data and inspect points with rank > 116."""
    print("=" * 70)
    print("INVESTIGATING RANK > 116 ANOMALIES")
    print("=" * 70)

    for pot_dir, pot_name, expected_rank in [
        ("atlas_output/1_r", "1/r (Newton)", 116),
        ("atlas_output/1_r2", "1/r2 (Calogero-Moser)", 116),
        ("atlas_output/harmonic", "Harmonic (r^2)", 15),
    ]:
        if not os.path.exists(os.path.join(pot_dir, "rank_map.npy")):
            print(f"\n  {pot_name}: no saved data, skipping")
            continue

        mu = np.load(os.path.join(pot_dir, "mu_vals.npy"))
        phi = np.load(os.path.join(pot_dir, "phi_vals.npy"))
        rank_map = np.load(os.path.join(pot_dir, "rank_map.npy"))
        gap_map = np.load(os.path.join(pot_dir, "gap_map.npy"))

        print(f"\n  {pot_name}:")
        print(f"    Grid: {rank_map.shape[0]} x {rank_map.shape[1]}")
        print(f"    Expected rank: {expected_rank}")
        print(f"    Rank range: [{rank_map.min()}, {rank_map.max()}]")
        print(f"    Unique ranks: {sorted(set(rank_map.flatten()))}")

        anomalous = np.argwhere(rank_map > expected_rank)
        if len(anomalous) == 0:
            print(f"    No anomalies (all points at or below {expected_rank})")
            continue

        print(f"    Anomalous points (rank > {expected_rank}): {len(anomalous)}")
        print()
        print(f"    {'mu':>8s}  {'phi':>8s}  {'phi_deg':>8s}  {'rank':>5s}  "
              f"{'gap':>10s}  {'location':>20s}")
        print(f"    {'----':>8s}  {'----':>8s}  {'-------':>8s}  {'----':>5s}  "
              f"{'---':>10s}  {'--------':>20s}")

        for idx in anomalous:
            i, j = idx
            mu_val = mu[i]
            phi_val = phi[j]
            phi_deg = phi_val * 180 / np.pi
            r = rank_map[i, j]
            g = gap_map[i, j]

            near_edge = []
            if i <= 1 or i >= len(mu) - 2:
                near_edge.append("mu-edge")
            if j <= 1 or j >= len(phi) - 2:
                near_edge.append("phi-edge")
            if mu_val < 0.3:
                near_edge.append("near-collision-13")
            if mu_val > 2.5:
                near_edge.append("near-collision-12")
            if phi_val < 0.2 or phi_val > np.pi - 0.2:
                near_edge.append("near-collinear")

            loc = ", ".join(near_edge) if near_edge else "interior"

            print(f"    {mu_val:8.3f}  {phi_val:8.3f}  {phi_deg:8.1f}  "
                  f"{r:5d}  {g:10.2e}  {loc:>20s}")

        normal_gaps = gap_map[rank_map == expected_rank]
        anomaly_gaps = gap_map[rank_map > expected_rank]
        print(f"\n    Gap ratio comparison:")
        print(f"      Normal (rank={expected_rank}):  "
              f"median={np.median(normal_gaps):.2e}, "
              f"min={normal_gaps.min():.2e}, max={normal_gaps.max():.2e}")
        print(f"      Anomalous (rank>{expected_rank}): "
              f"median={np.median(anomaly_gaps):.2e}, "
              f"min={anomaly_gaps.min():.2e}, max={anomaly_gaps.max():.2e}")


def control_epsilon_sweep():
    """Run epsilon sweep at a generic non-special point."""
    print("\n" + "=" * 70)
    print("CONTROL EPSILON SWEEP AT GENERIC CONFIGURATION")
    print("=" * 70)

    from stability_atlas import AtlasConfig, PoissonAlgebra, ShapeSpace

    config = AtlasConfig(
        potential_type='1/r',
        max_level=3,
        n_phase_samples=400,
        svd_gap_threshold=1e4,
    )

    print("\nBuilding algebra (this takes ~12 min for level 3)...")
    algebra = PoissonAlgebra(config)

    epsilons = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

    configs_to_test = {
        "generic_scalene": (1.5, 1.0),
        "generic_2": (0.8, 0.7),
        "generic_3": (2.0, 1.8),
        "lagrange": (1.0, np.pi / 3),
        "euler_collinear": (0.5, np.pi),
    }

    for name, (mu_val, phi_val) in configs_to_test.items():
        positions = ShapeSpace.shape_to_positions(mu_val, phi_val)
        phi_deg = phi_val * 180 / np.pi

        print(f"\n  {name} (mu={mu_val:.3f}, phi={phi_deg:.1f} deg):")

        ranks = []
        for eps in epsilons:
            rank, svs, info = algebra.compute_rank_at_configuration(
                positions, config.max_level,
                n_samples=config.n_phase_samples,
                epsilon=eps
            )
            ranks.append(rank)

            n_above = np.sum(svs > 1e-10 * svs[0]) if len(svs) > 0 else 0
            print(f"    eps={eps:.1e}  rank={rank}  gap={info['max_gap_ratio']:.2e}  "
                  f"n_meaningful_svs={n_above}")

        varies = len(set(ranks)) > 1
        if varies:
            print(f"  --> RANK VARIES: {ranks}")
        else:
            print(f"  --> Rank stable at {ranks[0]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("anomalies", "both"):
        investigate_anomalies()

    if mode in ("control", "both"):
        control_epsilon_sweep()

    print("\nDone.")
