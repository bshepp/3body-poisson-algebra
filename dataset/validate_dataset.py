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
]


def test_yaml_frontmatter():
    content = README.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "Could not find YAML frontmatter"
    meta = yaml.safe_load(parts[1])
    assert meta["license"] == "mit"
    assert len(meta["configs"]) == 9, f"Expected 9 configs, got {len(meta['configs'])}"
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
    assert len(info["splits"]) == 9, f"Expected 9 splits, got {len(info['splits'])}"
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
    assert all(df["level_2_dim"] == 17), "Not all level_2_dim == 17"
    for _, row in df.iterrows():
        svs = json.loads(row["level_2_singular_values"])
        assert len(svs) == 18
    print(f"Mass invariance: all {len(df)} rows have dim=17 and 18 SVs  OK")


def test_physical_systems():
    df = pd.read_parquet(OUTPUT / "physical_systems.parquet")
    assert len(df) >= 10, f"Expected >= 10 physical systems, got {len(df)}"
    assert "category" in df.columns
    assert "matches_universal" in df.columns
    assert "dim_L0" in df.columns
    categories = set(df["category"])
    assert "astrophysical" in categories
    assert "atomic" in categories
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
    test_scaling_formulas()
    print("\n*** ALL TESTS PASSED ***")


if __name__ == "__main__":
    main()
