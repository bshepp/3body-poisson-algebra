#!/usr/bin/env python3
"""Validate all dataset output files."""

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "dataset" / "output"
README = ROOT / "dataset" / "README.md"

TABLES = [
    "dimension_sequences", "structure_constants", "charge_sensitivity",
    "mass_invariance", "level4_convergence", "spectral_statistics",
    "physical_systems", "bell_test", "scaling_formulas",
    "tier_decomposition", "contextuality", "convergence_trajectories",
    "neural_algebras",
]


def test_yaml_frontmatter():
    content = README.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "Could not find YAML frontmatter"
    meta = yaml.safe_load(parts[1])
    assert meta["license"] == "mit"
    assert len(meta["configs"]) == 13, f"Expected 13 configs, got {len(meta['configs'])}"
    print(f"YAML frontmatter: {len(meta['configs'])} configs, license={meta['license']}  OK")


def test_parquet_files():
    total = 0
    for t in TABLES:
        df = pd.read_parquet(OUTPUT / f"{t}.parquet")
        total += len(df)
        print(f"  {t}: {len(df)} rows x {len(df.columns)} cols  OK")
    print(f"  Total: {total} rows")


def test_dataset_info():
    info = json.loads((OUTPUT / "dataset_info.json").read_text())
    assert len(info["splits"]) == 13, f"Expected 13 splits, got {len(info['splits'])}"
    print(f"dataset_info.json: {len(info['splits'])} splits  OK")


def test_structure_constants_shape():
    df = pd.read_parquet(OUTPUT / "structure_constants.parquet")
    for _, row in df.iterrows():
        tensor = json.loads(row["structure_constants"])
        shape = (len(tensor), len(tensor[0]), len(tensor[0][0]))
        expected = (row["algebra_dim"], row["algebra_dim"], row["algebra_dim"])
        assert shape == expected, f"Shape {shape} != {expected} for {row['potential']}"
    print(f"Structure constants: all {len(df)} tensors correct shape  OK")


def test_dimension_sequences_valid():
    df = pd.read_parquet(OUTPUT / "dimension_sequences.parquet")
    for idx, row in df.iterrows():
        seq = json.loads(row["dimension_sequence"])
        assert isinstance(seq, list) and all(isinstance(x, int) for x in seq)
    # Validate flattened columns exist and match
    assert "dim_L0" in df.columns, "Missing dim_L0 column"
    for idx, row in df.iterrows():
        seq = json.loads(row["dimension_sequence"])
        for i in range(min(len(seq), 5)):
            col = f"dim_L{i}"
            assert pd.notna(row[col]) and int(row[col]) == seq[i], f"Mismatch at row {idx} {col}"
    print(f"Dimension sequences: all {len(df)} entries valid + flattened dims match  OK")


def test_charge_sensitivity():
    df = pd.read_parquet(OUTPUT / "charge_sensitivity.parquet")
    for _, row in df.iterrows():
        charges = json.loads(row["charges"])
        assert isinstance(charges, list) and len(charges) == 3
    assert "dim_L0" in df.columns, "Missing dim_L0 in charge_sensitivity"
    print(f"Charge sensitivity: all {len(df)} charge vectors valid + flattened dims  OK")


def test_mass_invariance():
    df = pd.read_parquet(OUTPUT / "mass_invariance.parquet")
    moderate = df[df["m3"] <= 10000]
    assert all(moderate["level_2_dim"] == 17), "Not all moderate-mass level_2_dim == 17"
    for _, row in df.iterrows():
        svs = json.loads(row["level_2_singular_values"])
        assert len(svs) >= 12, f"Expected >= 12 SVs, got {len(svs)} for m3={row['m3']}"
    assert len(df) >= 30, f"Expected >= 30 mass ratio rows, got {len(df)}"
    print(f"Mass invariance: {len(df)} rows, moderate-mass dim=17 verified  OK")


def test_physical_systems():
    df = pd.read_parquet(OUTPUT / "physical_systems.parquet")
    assert len(df) >= 13, f"Expected >= 13 physical systems, got {len(df)}"
    assert "category" in df.columns
    assert "matches_universal" in df.columns
    assert "dim_L0" in df.columns
    categories = set(df["category"])
    assert "astrophysical" in categories
    assert "atomic" in categories
    yukawa_systems = df[df["category"].isin(["nuclear", "plasma"])]
    if len(yukawa_systems) > 0:
        assert all(yukawa_systems["matches_universal"]), \
            "Yukawa systems should all match universal sequence"
    print(f"Physical systems: {len(df)} systems across {len(categories)} categories  OK")


def test_bell_test():
    df = pd.read_parquet(OUTPUT / "bell_test.parquet")
    assert len(df) == 9, f"Expected 9 Bell test rows, got {len(df)}"
    assert all(~df["significant_violation"]), "Unexpected CHSH violation"
    assert all(df["max_abs_S"] < df["classical_bound"]), "S exceeds classical bound"
    print(f"Bell test: {len(df)} strata x variants, no violations  OK")


def test_scaling_formulas():
    df = pd.read_parquet(OUTPUT / "scaling_formulas.parquet")
    assert len(df) == 5, f"Expected 5 scaling formula rows, got {len(df)}"
    statuses = set(df["formula_status"])
    assert "falsified" in statuses, "Missing falsified formula"
    assert "verified" in statuses, "Missing verified formula"
    print(f"Scaling formulas: {len(df)} formulas, statuses={statuses}  OK")


def test_tier_decomposition():
    df = pd.read_parquet(OUTPUT / "tier_decomposition.parquet")
    assert len(df) >= 10, f"Expected >= 10 tier decomposition rows, got {len(df)}"
    groups = set(df["symmetry_group"])
    assert "S3" in groups, "Missing S3 decomposition"
    assert "S4" in groups, "Missing S4 decomposition"
    assert all(df["contribution"] >= 0), "Negative contribution found"
    print(f"Tier decomposition: {len(df)} rows, groups={groups}  OK")


def test_contextuality():
    df = pd.read_parquet(OUTPUT / "contextuality.parquet")
    assert len(df) >= 10, f"Expected >= 10 contextuality rows, got {len(df)}"
    assert all(df["n_commuting_pairs"] == 0), "Expected zero commuting pairs in all algebras"
    assert all(~df["contextual"]), "Unexpected contextuality found"
    assert all(df["ks_colorable"]), "Expected all KS colorable"
    print(f"Contextuality: {len(df)} algebras, all non-contextual (0 commuting pairs)  OK")


def test_spectral_statistics():
    df = pd.read_parquet(OUTPUT / "spectral_statistics.parquet")
    assert len(df) >= 10, f"Expected >= 10 spectral statistics rows, got {len(df)}"
    n4_rows = df[df["source_type"] == "n4_atlas_1d_slice"]
    assert len(n4_rows) >= 3, f"Expected >= 3 N=4 atlas slices, got {len(n4_rows)}"
    for _, row in n4_rows.iterrows():
        assert row["rank_mode"] == 62, \
            f"N=4 atlas slice {row['config']} mode rank {row['rank_mode']} != 62"
        assert row["n_points"] >= 50, \
            f"N=4 atlas slice {row['config']} has too few points: {row['n_points']}"
    print(f"Spectral statistics: {len(df)} rows, {len(n4_rows)} N=4 slices (all mode=62)  OK")


def test_convergence_trajectories():
    df = pd.read_parquet(OUTPUT / "convergence_trajectories.parquet")
    assert len(df) >= 40, f"Expected >= 40 convergence rows, got {len(df)}"
    configs = df.groupby(["N", "d", "potential", "level"])
    for name, group in configs:
        ranks = group.sort_values("n_samples")["rank"].values
        # Rank should be monotonically non-decreasing with sample count
        for i in range(len(ranks) - 1):
            assert ranks[i] <= ranks[i + 1], \
                f"Rank decreased for {name}: {ranks[i]} > {ranks[i+1]}"
    print(f"Convergence trajectories: {len(df)} rows, {len(configs)} configs, monotonic  OK")


def test_neural_algebras():
    df = pd.read_parquet(OUTPUT / "neural_algebras.parquet")
    assert len(df) >= 20, f"Expected >= 20 neural algebra rows, got {len(df)}"
    assert "n_layers" in df.columns
    assert "coupling_type" in df.columns
    assert "dimension_sequence" in df.columns

    for _, row in df.iterrows():
        dims = json.loads(row["dimension_sequence"])
        assert isinstance(dims, list) and all(isinstance(x, int) for x in dims)
        assert dims[0] >= 1, "Level-0 dimension must be >= 1"

    gradient_l3 = df[(df["n_layers"] == 3) & (df["coupling_type"] == "gradient") &
                     (df["loss_function"] == "mse") & (df["activation"] == "linear") &
                     (df["width"] == 1)]
    if len(gradient_l3) > 0:
        dims = json.loads(gradient_l3.iloc[0]["dimension_sequence"])
        assert dims == [3, 6, 17, 119], \
            f"L=3 gradient MSE linear should be [3,6,17,119], got {dims}"

    couplings = set(df["coupling_type"])
    assert len(couplings) >= 3, f"Expected >= 3 coupling types, got {couplings}"

    print(f"Neural algebras: {len(df)} configs, coupling types={couplings}  OK")


def main():
    print("=== Dataset Validation ===\n")
    test_yaml_frontmatter()
    test_parquet_files()
    test_dataset_info()
    test_structure_constants_shape()
    test_dimension_sequences_valid()
    test_charge_sensitivity()
    test_mass_invariance()
    test_physical_systems()
    test_bell_test()
    test_spectral_statistics()
    test_scaling_formulas()
    test_tier_decomposition()
    test_contextuality()
    test_convergence_trajectories()
    test_neural_algebras()
    print("\n*** ALL TESTS PASSED ***")


if __name__ == "__main__":
    main()
