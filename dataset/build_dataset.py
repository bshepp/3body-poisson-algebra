#!/usr/bin/env python3
"""
Build Hugging Face dataset: Pairwise Poisson Algebras of the N-Body Problem.

Reads ~200 JSON result files from the 3-body project and normalizes them
into 12 Parquet tables suitable for upload to Hugging Face.

Usage:
    pip install pandas pyarrow
    python dataset/build_dataset.py

Output:
    dataset/output/dimension_sequences.parquet
    dataset/output/structure_constants.parquet
    dataset/output/charge_sensitivity.parquet
    dataset/output/mass_invariance.parquet
    dataset/output/level4_convergence.parquet
    dataset/output/spectral_statistics.parquet
    dataset/output/physical_systems.parquet
    dataset/output/bell_test.parquet
    dataset/output/scaling_formulas.parquet
    dataset/output/tier_decomposition.parquet
    dataset/output/contextuality.parquet
    dataset/output/convergence_trajectories.parquet
    dataset/output/dataset_info.json
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "dataset" / "output"


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Split 1: dimension_sequences
# ---------------------------------------------------------------------------

def _potential_from_label(label: str) -> str:
    """Normalize potential labels to canonical names."""
    mapping = {
        "1/r": "1/r", "1/r^2": "1/r^2", "1/r^3": "1/r^3", "1/r^4": "1/r^4",
        "r^2": "r^2", "r^4": "r^4", "log": "log",
        "composite_u1_2": "composite(u+u^2)",
        "composite_u1_2_3": "composite(u+u^2+u^3)",
        "composite_u4": "composite(u^4)",
        "quantum_1/r": "1/r", "quantum_1/r^2": "1/r^2",
        "quantum_1/r^3": "1/r^3", "quantum_1/r^4": "1/r^4",
        "quantum_log": "log",
        "quantum_r^1": "r^1", "quantum_r^2": "r^2",
        "quantum_r^3": "r^3", "quantum_r^4": "r^4",
        "quantum_composite_u1_2": "composite(u+u^2)",
        "neural_gradient_product": "neural",
    }
    return mapping.get(label, label)


def build_dimension_sequences() -> pd.DataFrame:
    rows = []

    # --- Standard symbolic_rank files ---
    rank_dir = ROOT / "results" / "symbolic_rank"
    for fp in sorted(rank_dir.glob("rank_N*.json")):
        data = load_json(fp)
        N = data.get("N", 3)
        d = data.get("d", 1)
        label = data.get("potential_label", data.get("potential", "1/r"))
        potential = _potential_from_label(label)
        is_quantum = data.get("quantum", False)
        bracket = "moyal" if is_quantum else "poisson"
        rows.append({
            "N": N, "d": d, "potential": potential,
            "bracket_type": bracket,
            "masses": None, "charges": None,
            "external_potential": None,
            "max_level": data.get("max_level"),
            "dimension_sequence": data.get("cumulative_rank"),
            "new_per_level": data.get("new_per_level"),
            "is_exact": True,
            "computation_method": "symbolic_QQ",
            "sympy_version": data.get("sympy_version"),
            "computation_time_s": data.get("computation_time_seconds"),
            "physical_system": None,
            "source_file": str(fp.relative_to(ROOT)),
        })

    # --- Mass-specific symbolic rank files ---
    mass_files = {
        "rank_symbolic.json": ("symbolic", None),
        "rank_1_1_1.json": (["1", "1", "1"], None),
        "rank_1_2_3.json": (["1", "2", "3"], None),
        "rank_1_1_5over2.json": (["1", "1", "5/2"], None),
        "rank_1_1_1over100.json": (["1", "1", "1/100"], None),
        "rank_1_1_1over10000.json": (["1", "1", "1/10000"], None),
    }
    for fname, (masses, _) in mass_files.items():
        fp = rank_dir / fname
        if not fp.exists():
            continue
        data = load_json(fp)
        rows.append({
            "N": 3, "d": 1, "potential": "1/r",
            "bracket_type": "poisson",
            "masses": masses, "charges": None,
            "external_potential": None,
            "max_level": max(len(data.get("cumulative_rank", [])), 0),
            "dimension_sequence": data.get("cumulative_rank"),
            "new_per_level": data.get("new_per_level"),
            "is_exact": True,
            "computation_method": "symbolic_QQ_m1m2m3" if masses != "symbolic" else "symbolic_QQ_symbolic",
            "sympy_version": data.get("sympy_version"),
            "computation_time_s": data.get("computation_time_seconds"),
            "physical_system": None,
            "source_file": str(fp.relative_to(ROOT)),
        })

    # --- GUE comparison results ---
    gue_fp = ROOT / "primes" / "results" / "gue_comparison.json"
    if gue_fp.exists():
        gue = load_json(gue_fp)
        for r in gue.get("results", []):
            phys = r.get("physical_model")
            has_ext = "harmonic" in r.get("description", "").lower()
            rows.append({
                "N": 3, "d": 1,
                "potential": "log" if "log" in r["name"] else ("composite" if "composite" in r["name"] else "r^2" if "harmonic" in r["name"] else "1/r+r^2"),
                "bracket_type": "poisson",
                "masses": None, "charges": None,
                "external_potential": "harmonic_omega_1" if has_ext else None,
                "max_level": r.get("max_level", 3),
                "dimension_sequence": r.get("dimensions"),
                "new_per_level": None,
                "is_exact": False,
                "computation_method": "numerical_svd",
                "sympy_version": None,
                "computation_time_s": r.get("elapsed_seconds"),
                "physical_system": phys,
                "source_file": str(gue_fp.relative_to(ROOT)),
            })

    # --- Quantum GUE ---
    qgue_fp = ROOT / "primes" / "results" / "quantum_gue.json"
    if qgue_fp.exists():
        qgue = load_json(qgue_fp)
        rows.append({
            "N": qgue.get("n_bodies", 3),
            "d": qgue.get("d_spatial", 1),
            "potential": qgue.get("potential", "log"),
            "bracket_type": "moyal",
            "masses": None, "charges": None,
            "external_potential": f"harmonic_omega_{qgue['external_potential']['omega']}" if qgue.get("external_potential") else None,
            "max_level": len(qgue.get("dimension_sequence", [])) - 1,
            "dimension_sequence": qgue.get("dimension_sequence"),
            "new_per_level": None,
            "is_exact": False,
            "computation_method": "numerical_svd",
            "sympy_version": qgue.get("sympy_version"),
            "computation_time_s": qgue.get("elapsed_seconds"),
            "physical_system": "GUE_quantum_log_gas",
            "source_file": str(qgue_fp.relative_to(ROOT)),
        })

    # --- Energy bound results ---
    eb_fp = ROOT / "results" / "energy_bound" / "energy_bound_results.json"
    if eb_fp.exists():
        eb = load_json(eb_fp)
        rows.append({
            "N": 3, "d": 2, "potential": "1/r",
            "bracket_type": "poisson",
            "masses": None, "charges": None,
            "external_potential": None,
            "max_level": 3,
            "dimension_sequence": [3, 6, 17, eb.get("classical_rank_ad_H", 116)],
            "new_per_level": None,
            "is_exact": True,
            "computation_method": "symbolic_QQ",
            "sympy_version": None,
            "computation_time_s": eb.get("total_time_s"),
            "physical_system": "energy_bound_analysis",
            "source_file": str(eb_fp.relative_to(ROOT)),
        })

    # --- N=4 potential universality ---
    n4_fp = ROOT / "nbody" / "n4_potential_universality_results.json"
    if n4_fp.exists():
        n4 = load_json(n4_fp)
        for r in n4.get("results", []):
            rows.append({
                "N": r.get("N", 4), "d": r.get("d", 1),
                "potential": r.get("potential", "1/r"),
                "bracket_type": "poisson",
                "masses": None, "charges": None,
                "external_potential": None,
                "max_level": r.get("max_level", 2),
                "dimension_sequence": r.get("sequence"),
                "new_per_level": None,
                "is_exact": False,
                "computation_method": "numerical_svd",
                "sympy_version": None,
                "computation_time_s": r.get("elapsed_s"),
                "physical_system": None,
                "source_file": str(n4_fp.relative_to(ROOT)),
            })

    # --- Yukawa mu-sweep (Taylor-expansion composite) ---
    yuk_fp = ROOT / "results" / "yukawa_dimseq.json"
    if yuk_fp.exists():
        yuk = load_json(yuk_fp)
        for r in yuk.get("mu_sweep", []):
            if "error" in r:
                continue
            rows.append({
                "N": r.get("N", 3), "d": r.get("d", 1),
                "potential": f"yukawa(mu={r.get('mu', '?')})",
                "bracket_type": "poisson",
                "masses": None, "charges": None,
                "external_potential": None,
                "max_level": r.get("taylor_order", 3),
                "dimension_sequence": r.get("dimension_sequence"),
                "new_per_level": None,
                "is_exact": False,
                "computation_method": "numerical_svd",
                "sympy_version": None,
                "computation_time_s": r.get("computation_time_s"),
                "physical_system": None,
                "source_file": str(yuk_fp.relative_to(ROOT)),
            })

    # --- Extended L3 exponent sweep (1/r^n and r^n families) ---
    l3_ext_fp = ROOT / "results" / "l3_exponent_sweep_extended.json"
    if l3_ext_fp.exists():
        l3e = load_json(l3_ext_fp)
        for r in l3e.get("results", []):
            if "error" in r:
                continue
            rows.append({
                "N": r.get("N", 3), "d": r.get("d", 1),
                "potential": r.get("potential_label", ""),
                "bracket_type": "poisson",
                "masses": None, "charges": None,
                "external_potential": None,
                "max_level": r.get("max_level", 3),
                "dimension_sequence": r.get("dimension_sequence"),
                "new_per_level": None,
                "is_exact": False,
                "computation_method": r.get("method", "numerical_svd"),
                "sympy_version": None,
                "computation_time_s": r.get("computation_time_s"),
                "physical_system": None,
                "source_file": str(l3_ext_fp.relative_to(ROOT)),
            })

    # --- Level-2 exponent sweep (fine-grained 1/r^n and r^n families) ---
    l2_fp = ROOT / "results" / "level2_exponent_sweep.json"
    if l2_fp.exists():
        l2 = load_json(l2_fp)
        for r in l2.get("results", []):
            rows.append({
                "N": r.get("N", 3), "d": r.get("d", 1),
                "potential": r.get("potential_label", ""),
                "bracket_type": "poisson",
                "masses": None, "charges": None,
                "external_potential": None,
                "max_level": r.get("max_level", 2),
                "dimension_sequence": r.get("dimension_sequence"),
                "new_per_level": None,
                "is_exact": False,
                "computation_method": r.get("method", "numerical_svd"),
                "sympy_version": None,
                "computation_time_s": r.get("computation_time_s"),
                "physical_system": None,
                "source_file": str(l2_fp.relative_to(ROOT)),
            })

    # Deduplicate symbolic_rank rows: keep highest max_level per (N, d, potential, bracket_type)
    seen = {}
    deduped = []
    for r in rows:
        if r["computation_method"] in ("symbolic_QQ",) and r["physical_system"] is None and r["masses"] is None:
            key = (r["N"], r["d"], r["potential"], r["bracket_type"])
            if key in seen:
                prev_idx, prev_level = seen[key]
                if (r["max_level"] or 0) > (prev_level or 0):
                    deduped[prev_idx] = None
                    seen[key] = (len(deduped), r["max_level"])
                    deduped.append(r)
                else:
                    deduped.append(None)
            else:
                seen[key] = (len(deduped), r["max_level"])
                deduped.append(r)
        else:
            deduped.append(r)
    rows = [r for r in deduped if r is not None]

    df = pd.DataFrame(rows)

    # Flatten dimension_sequence into individual level columns for HF Viewer
    for i in range(5):
        col = f"dim_L{i}"
        df[col] = df["dimension_sequence"].apply(
            lambda seq, idx=i: seq[idx] if isinstance(seq, list) and len(seq) > idx else None
        )
        df[col] = df[col].astype("Int64")

    for col in ["dimension_sequence", "new_per_level", "masses", "charges"]:
        df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
    return df


# ---------------------------------------------------------------------------
# Split 2: structure_constants
# ---------------------------------------------------------------------------

def build_structure_constants() -> pd.DataFrame:
    rows = []
    sc_base = ROOT / "results" / "algebra_structure"
    rank_dir = ROOT / "results" / "symbolic_rank"

    for sc_dir in sorted(sc_base.iterdir()):
        sc_fp = sc_dir / "structure_constants_exact.json"
        if not sc_fp.exists():
            continue

        dir_name = sc_dir.name
        rank_name = f"rank_{dir_name}.json"
        rank_fp = rank_dir / rank_name

        parts = dir_name.split("_", 2)
        raw_potential = parts[-1] if len(parts) > 2 else dir_name
        potential_map = {
            "1r": "1/r", "1r2": "1/r^2", "1r3": "1/r^3",
            "r2": "r^2", "r4": "r^4", "log": "log",
            "composite_u1_2": "composite(u+u^2)",
            "composite_u1_2_3": "composite(u+u^2+u^3)",
            "composite_u4": "composite(u^4)",
        }
        potential = potential_map.get(raw_potential, _potential_from_label(raw_potential))

        sc_data = load_json(sc_fp)

        metadata = {}
        if rank_fp.exists():
            rank_data = load_json(rank_fp)
            metadata = rank_data.get("structure", {})

        algebra_dim = len(sc_data) if isinstance(sc_data, list) else 17
        rows.append({
            "potential": potential,
            "algebra_dim": algebra_dim,
            "structure_constants": json.dumps(sc_data),
            "killing_signature": json.dumps(metadata.get("killing_signature")),
            "is_semisimple": metadata.get("is_semisimple"),
            "is_solvable": metadata.get("is_solvable"),
            "solvability_length": metadata.get("solvability_length"),
            "is_nilpotent": metadata.get("is_nilpotent"),
            "nilpotency_class": metadata.get("nilpotency_class"),
            "center_dimension": metadata.get("center_dimension"),
            "derived_series": json.dumps(metadata.get("derived_series")),
            "lower_central_series": json.dumps(metadata.get("lower_central_series")),
            "source_file": str(sc_fp.relative_to(ROOT)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split 3: charge_sensitivity
# ---------------------------------------------------------------------------

def _parse_charge_entries(data, source_path, default_method="numerical_SVD"):
    """Parse charge sensitivity entries from a manifest file."""
    rows = []
    for key, entry in data.items():
        r = entry.get("result", entry)
        charges_raw = r.get("charges", {})
        masses_raw = r.get("masses", {})

        if isinstance(charges_raw, dict):
            charges = [charges_raw.get("1"), charges_raw.get("2"), charges_raw.get("3")]
        else:
            charges = charges_raw

        if isinstance(masses_raw, dict):
            masses = [masses_raw.get("1"), masses_raw.get("2"), masses_raw.get("3")]
        else:
            masses = masses_raw

        dims = r.get("dims", [])
        rows.append({
            "experiment_key": key,
            "label": r.get("label"),
            "charges": json.dumps(charges),
            "masses": json.dumps(masses),
            "n_samples": r.get("n_samples"),
            "method": r.get("method", default_method),
            "d_spatial": r.get("d_spatial"),
            "dimension_sequence": json.dumps(dims),
            "dim_L0": dims[0] if len(dims) > 0 else None,
            "dim_L1": dims[1] if len(dims) > 1 else None,
            "dim_L2": dims[2] if len(dims) > 2 else None,
            "dim_L3": dims[3] if len(dims) > 3 else None,
            "matches_116": r.get("matches_116"),
            "physical_system": r.get("label"),
            "computation_time_s": r.get("elapsed_seconds"),
            "source_file": source_path,
        })
    return rows


def build_charge_sensitivity() -> pd.DataFrame:
    rows = []
    fp_main = ROOT / "results" / "charge_sensitivity" / "charge_sensitivity_completion.json"
    if fp_main.exists():
        rows.extend(_parse_charge_entries(
            load_json(fp_main), str(fp_main.relative_to(ROOT)),
            default_method="numerical_SVD"))

    fp_d1 = ROOT / "results" / "charge_sensitivity" / "charge_sweep_qqn_d1.json"
    if fp_d1.exists():
        rows.extend(_parse_charge_entries(
            load_json(fp_d1), str(fp_d1.relative_to(ROOT)),
            default_method="exact_symbolic_QQ"))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["dim_L0", "dim_L1", "dim_L2", "dim_L3"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Split 4: mass_invariance
# ---------------------------------------------------------------------------

def build_mass_invariance() -> pd.DataFrame:
    fp = ROOT / "data" / "mass_ratio_sweep.json"
    if not fp.exists():
        return pd.DataFrame()

    data = load_json(fp)
    rows = []
    for entry in data:
        rows.append({
            "m3": entry["m3"],
            "m3_log10": entry["m3_log10"],
            "level_0_dim": entry["level_0"]["dim"],
            "level_1_dim": entry["level_1"]["dim"],
            "level_2_dim": entry["level_2"]["dim"],
            "level_2_gap_ratio": entry["level_2"]["gap_ratio"],
            "level_2_singular_values": json.dumps(entry["level_2"]["singular_values"]),
            "dims": json.dumps(entry["dims"]),
            "elapsed_s": entry.get("elapsed_s"),
            "source_file": str(fp.relative_to(ROOT)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split 5: level4_convergence
# ---------------------------------------------------------------------------

def build_level4_convergence() -> pd.DataFrame:
    rows = []
    results_dir = ROOT / "results"

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("level4_"):
            continue
        fp = d / "results.json"
        if not fp.exists():
            continue

        data = load_json(fp)
        config = data.get("config", d.name.split("_")[1])

        rows.append({
            "config": config,
            "n_samples": data.get("n_samples"),
            "dimension_sequence": json.dumps(data.get("dims")),
            "new_per_level": json.dumps(data.get("new_per_level")),
            "d4_lower_bound": data.get("d4_lower_bound"),
            "max_gap_ratio": data.get("max_gap_ratio"),
            "max_gap_index": data.get("max_gap_index"),
            "elapsed_seconds": data.get("elapsed_seconds"),
            "timestamp": data.get("timestamp"),
            "mu": data.get("mu"),
            "phi": data.get("phi"),
            "epsilon": data.get("epsilon"),
            "source_file": str(fp.relative_to(ROOT)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split 6: spectral_statistics
# ---------------------------------------------------------------------------

def build_spectral_statistics() -> pd.DataFrame:
    rows = []

    # --- Targeted atlas configs ---
    atlas_fp = ROOT / "atlas_figures" / "atlas_summary.json"
    if atlas_fp.exists():
        atlas = load_json(atlas_fp)
        for cfg in atlas.get("targeted_configs", []):
            rows.append({
                "config": cfg["config"],
                "label": cfg["label"],
                "source_type": "targeted_atlas",
                "n_regions": cfg.get("n_regions"),
                "rank_min": cfg.get("rank_min"),
                "rank_max": cfg.get("rank_max"),
                "rank_mode": cfg.get("rank_mode"),
                "n_points": cfg.get("n_points"),
                "pct_116": cfg.get("pct_116"),
                "source_file": str(atlas_fp.relative_to(ROOT)),
            })

    # --- Atlas full (irrational exponents) ---
    atlas_full_dir = ROOT / "results" / "atlas_full"
    if atlas_full_dir.exists():
        for d in sorted(atlas_full_dir.iterdir()):
            fp = d / "summary.json"
            if not fp.exists():
                continue
            data = load_json(fp)
            exponent_name = d.name.replace("atlas-", "")
            rows.append({
                "config": exponent_name,
                "label": data.get("label", exponent_name),
                "source_type": "atlas_full_irrational",
                "n_regions": None,
                "rank_min": None,
                "rank_max": None,
                "rank_mode": None,
                "n_points": data.get("total_points", data.get("valid_points")),
                "pct_116": data.get("rank_116_fraction", 0) * 100 if isinstance(data.get("rank_116_fraction"), (int, float)) else None,
                "source_file": str(fp.relative_to(ROOT)),
            })

    # --- N=4 atlas 1D slices ---
    n4_atlas_fp = ROOT / "results" / "n4_atlas_1d.json"
    if n4_atlas_fp.exists():
        n4_data = load_json(n4_atlas_fp)
        for sl in n4_data.get("slices", []):
            rows.append({
                "config": f"N4_d1_{sl['slice_name']}",
                "label": sl["description"],
                "source_type": "n4_atlas_1d_slice",
                "n_regions": None,
                "rank_min": sl.get("rank_min"),
                "rank_max": sl.get("rank_max"),
                "rank_mode": sl.get("mode_rank"),
                "n_points": sl.get("n_points"),
                "pct_116": None,
                "source_file": str(n4_atlas_fp.relative_to(ROOT)),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split 7: physical_systems
# ---------------------------------------------------------------------------

SYSTEM_CATEGORIES = {
    "sun_earth_moon": "astrophysical",
    "sun_jupiter_asteroid": "astrophysical",
    "three_cluster_stars": "astrophysical",
    "binary_star_planet": "astrophysical",
    "three_galaxies": "astrophysical",
    "triple_bh_lisa": "astrophysical",
    "binary_bh_ns": "astrophysical",
    "helium": "atomic",
    "lithium_ion": "atomic",
    "h_minus_ion": "atomic",
    "positronium_neg": "atomic",
    "muonic_helium": "atomic",
    "h2_plus_ion": "atomic",
    "penning_trap": "ion_trap",
    "two_d_vortices": "condensed_matter",
    "h3_plus": "molecular",
    "ozone_nuclei": "molecular",
    "tritium_he3": "nuclear",
    "dusty_plasma": "plasma",
    "p_n_n_scattering": "nuclear",
}

SYSTEM_LABELS = {
    "sun_earth_moon": "Sun-Earth-Moon",
    "sun_jupiter_asteroid": "Sun-Jupiter-Asteroid",
    "three_cluster_stars": "Three Cluster Stars",
    "binary_star_planet": "Binary Star + Planet",
    "three_galaxies": "Three Galaxies",
    "triple_bh_lisa": "Triple Black Hole (LISA)",
    "binary_bh_ns": "Binary BH + Neutron Star",
    "helium": "Helium Atom",
    "lithium_ion": "Lithium Ion (Li+)",
    "h_minus_ion": "Hydrogen Anion (H-)",
    "positronium_neg": "Positronium Ion (Ps-)",
    "muonic_helium": "Muonic Helium",
    "h2_plus_ion": "Hydrogen Molecular Ion (H2+)",
    "penning_trap": "Penning Trap (3 ions)",
    "two_d_vortices": "2D Point Vortices",
    "h3_plus": "Trihydrogen Cation (H3+)",
    "ozone_nuclei": "Ozone Nuclei (O3)",
    "tritium_he3": "Tritium / He-3 (Yukawa)",
    "dusty_plasma": "Dusty Plasma (Yukawa)",
    "p_n_n_scattering": "Proton-Neutron-Neutron (Yukawa)",
}


def build_physical_systems() -> pd.DataFrame:
    fp = ROOT / "results" / "expansion_dimseq" / "expansion_dimseq_completion.json"
    if not fp.exists():
        return pd.DataFrame()

    data = load_json(fp)
    rows = []
    for sys_name, entry in data.items():
        if entry.get("status") != "complete":
            continue
        dims = entry.get("result", [])
        matches = dims == [3, 6, 17, 116]
        rows.append({
            "system_name": sys_name,
            "system_label": SYSTEM_LABELS.get(sys_name, sys_name),
            "category": SYSTEM_CATEGORIES.get(sys_name, "other"),
            "dimension_sequence": json.dumps(dims),
            "dim_L0": dims[0] if len(dims) > 0 else None,
            "dim_L1": dims[1] if len(dims) > 1 else None,
            "dim_L2": dims[2] if len(dims) > 2 else None,
            "dim_L3": dims[3] if len(dims) > 3 else None,
            "matches_universal": matches,
            "completed_at": entry.get("completed_at"),
            "source_file": str(fp.relative_to(ROOT)),
        })

    # --- Yukawa physical systems ---
    yuk_fp = ROOT / "results" / "yukawa_dimseq.json"
    if yuk_fp.exists():
        yuk = load_json(yuk_fp)
        for r in yuk.get("physical_systems", []):
            if "error" in r:
                continue
            sys_name = r.get("system_name", "unknown")
            dims = r.get("dimension_sequence", [])
            matches = dims == [3, 6, 17, 116]
            rows.append({
                "system_name": sys_name,
                "system_label": SYSTEM_LABELS.get(sys_name, r.get("system_label", sys_name)),
                "category": SYSTEM_CATEGORIES.get(sys_name, r.get("category", "other")),
                "dimension_sequence": json.dumps(dims),
                "dim_L0": dims[0] if len(dims) > 0 else None,
                "dim_L1": dims[1] if len(dims) > 1 else None,
                "dim_L2": dims[2] if len(dims) > 2 else None,
                "dim_L3": dims[3] if len(dims) > 3 else None,
                "matches_universal": matches,
                "completed_at": None,
                "source_file": str(yuk_fp.relative_to(ROOT)),
            })

    df = pd.DataFrame(rows)
    for col in ["dim_L0", "dim_L1", "dim_L2", "dim_L3"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Split 8: bell_test
# ---------------------------------------------------------------------------

def build_bell_test() -> pd.DataFrame:
    fp = ROOT / "nbody" / "bell_test_results" / "chsh_summary.json"
    if not fp.exists():
        return pd.DataFrame()

    data = load_json(fp)
    overall = data.get("overall", {})
    rows = []

    for stratum_name, variants in data.get("strata", {}).items():
        for variant_name, v in variants.items():
            rows.append({
                "stratum": stratum_name,
                "variant": variant_name,
                "max_abs_S": v.get("max_abs_S"),
                "max_S": v.get("max_S"),
                "ci_95_lower": v["ci_95"][0] if v.get("ci_95") else None,
                "ci_95_upper": v["ci_95"][1] if v.get("ci_95") else None,
                "optimal_angles_deg": json.dumps(v.get("optimal_angles_deg")),
                "significant_violation": v.get("significant_violation"),
                "n_samples": v.get("n_samples"),
                "classical_bound": overall.get("classical_bound", 2.0),
                "tsirelson_bound": overall.get("tsirelson_bound"),
                "source_file": str(fp.relative_to(ROOT)),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split 9: scaling_formulas
# ---------------------------------------------------------------------------

def build_scaling_formulas() -> pd.DataFrame:
    fp = ROOT / "results" / "analysis" / "nbody_scaling_formulas.json"
    if not fp.exists():
        return pd.DataFrame()

    data = load_json(fp)
    formulas = data.get("formulas", {})
    rows = []

    formula_entries = [
        ("L0", formulas.get("L0", {})),
        ("L1", formulas.get("L1", {})),
        ("L2_original", formulas.get("L2_original", {})),
        ("L2_corrected", formulas.get("L2_corrected", {})),
        ("L3", formulas.get("L3", {})),
    ]

    for level_key, f in formula_entries:
        status = "unknown"
        if f.get("trivial"):
            status = "verified_trivial"
        elif f.get("status") == "FALSIFIED":
            status = "falsified"
        elif f.get("verified_for"):
            status = "verified"
        elif f.get("expression") == "unknown":
            status = "unknown"

        notes_parts = []
        if f.get("note"):
            notes_parts.append(f["note"])
        if f.get("boundary_note"):
            notes_parts.append(f["boundary_note"])
        if f.get("graph_theoretic_conjecture"):
            notes_parts.append(f"Conjecture: {f['graph_theoretic_conjecture']}")

        rows.append({
            "level": level_key,
            "formula_expression": f.get("expression"),
            "formula_status": status,
            "leading_term": f.get("leading_term"),
            "new_per_level_formula": json.dumps(f.get("new_per_level")) if isinstance(f.get("new_per_level"), dict) else (f.get("new_per_level") or f.get("new_per_level_formula")),
            "verified_N_values": json.dumps(f.get("verified_for")),
            "failed_N_values": json.dumps(f.get("failed_at")),
            "predictions": json.dumps(f.get("predictions")),
            "data_points": json.dumps(f.get("data_points")),
            "notes": " | ".join(notes_parts) if notes_parts else None,
            "source_file": str(fp.relative_to(ROOT)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split 10: tier_decomposition
# ---------------------------------------------------------------------------

def build_tier_decomposition() -> pd.DataFrame:
    fp = ROOT / "results" / "tier_decomposition" / "s3_s4_decomposition.json"
    if not fp.exists():
        return pd.DataFrame()

    data = load_json(fp)
    rows = []

    for group_key, group_data in data.items():
        N = group_data["N"]
        sym = group_data["symmetry_group"]
        irreps = group_data["irreps"]
        irrep_dims = group_data["irrep_dims"]
        observed_rank = group_data["observed_rank"]

        for level_key, level_data in group_data["levels"].items():
            level_num = int(level_key.replace("L", ""))
            mults = level_data["mults"]
            for i, irrep_name in enumerate(irreps):
                rows.append({
                    "N": N,
                    "symmetry_group": sym,
                    "level": level_num,
                    "total_candidates": level_data["dim"],
                    "observed_rank": observed_rank[level_num] if level_num < len(observed_rank) else None,
                    "irrep_name": irrep_name,
                    "irrep_dim": irrep_dims[i],
                    "multiplicity": mults[i],
                    "contribution": mults[i] * irrep_dims[i],
                    "source_file": str(fp.relative_to(ROOT)),
                })

        # Total row
        total = group_data["total"]
        for i, irrep_name in enumerate(irreps):
            rows.append({
                "N": N,
                "symmetry_group": sym,
                "level": -1,
                "total_candidates": total["dim"],
                "observed_rank": observed_rank[-1] if observed_rank else None,
                "irrep_name": irrep_name,
                "irrep_dim": irrep_dims[i],
                "multiplicity": total["mults"][i],
                "contribution": total["mults"][i] * irrep_dims[i],
                "source_file": str(fp.relative_to(ROOT)),
            })

    df = pd.DataFrame(rows)
    for col in ["N", "level", "total_candidates", "observed_rank",
                 "irrep_dim", "multiplicity", "contribution"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Split 11: contextuality
# ---------------------------------------------------------------------------

def build_contextuality() -> pd.DataFrame:
    fp = ROOT / "nbody" / "contextuality_results" / "contextuality_summary.json"
    if not fp.exists():
        return pd.DataFrame()

    data = load_json(fp)
    rows = []
    for alg in data.get("algebras", []):
        name = alg["algebra"]
        parts = name.split("_", 2)
        N = int(parts[0][1:]) if len(parts) > 0 else 3
        d = int(parts[1][1:]) if len(parts) > 1 else 2
        pot_raw = parts[2] if len(parts) > 2 else "1r"
        potential_map = {
            "1r": "1/r", "1r2": "1/r^2", "1r3": "1/r^3", "1r4": "1/r^4",
            "r1": "r^1", "r2": "r^2", "r4": "r^4", "r6": "r^6",
            "r8": "r^8", "r10": "r^10", "r3": "r^3",
            "log": "log",
            "composite_u1_2": "composite(u+u^2)",
            "composite_u1_2_3": "composite(u+u^2+u^3)",
            "composite_u4": "composite(u^4)",
        }
        potential = potential_map.get(pot_raw, pot_raw)

        rows.append({
            "N": N,
            "d": d,
            "potential": potential,
            "algebra_dim": alg["dim"],
            "n_commuting_pairs": alg["commuting_pairs"],
            "total_pairs": alg["total_pairs"],
            "commutativity_fraction": alg["commuting_pairs"] / alg["total_pairs"] if alg["total_pairs"] > 0 else 0,
            "ks_colorable": alg["ks_colorable"],
            "contextual": alg["contextual"],
            "pm_constructible": alg["pm_constructible"],
            "source_file": str(fp.relative_to(ROOT)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split 12: convergence_trajectories
# ---------------------------------------------------------------------------

def build_convergence_trajectories() -> pd.DataFrame:
    fp = ROOT / "results" / "convergence_trajectories.json"
    if not fp.exists():
        return pd.DataFrame()

    data = load_json(fp)
    df = pd.DataFrame(data)
    df["source_file"] = str(fp.relative_to(ROOT))
    return df


# ---------------------------------------------------------------------------
# Split 13: neural_algebras
# ---------------------------------------------------------------------------

def build_neural_algebras() -> pd.DataFrame:
    nn_dir = ROOT / "results" / "neural_algebras"
    if not nn_dir.exists():
        return pd.DataFrame()

    # Known universality classes at L=3 (assigned by dimension sequence)
    UNIV_CLASSES_L3 = {
        (3, 6, 17, 119): "A_119_gradient",
        (3, 6, 17, 115): "B_115_directional",
        (3, 6, 17, 111): "C_111_gradient_sum",
        (3, 6, 17, 104): "D_104_gradient_cubic",
        (3, 6, 17, 87):  "E_87_natural_gradient",
        (3, 6, 17, 62):  "F_62_cross_entropy",
        (3, 5, 11, 47):  "G_47_hessian",
    }

    rows = []
    for fp in sorted(nn_dir.glob("nn_*.json")):
        data = load_json(fp)
        dims = data.get("dimension_sequence", [])

        matches_physical = False
        if len(dims) >= 4:
            matches_physical = dims[3] == 116

        class_label = None
        if data.get("n_layers") == 3 and len(dims) >= 4:
            class_label = UNIV_CLASSES_L3.get(tuple(dims))

        rows.append({
            "n_layers": data.get("n_layers"),
            "width": data.get("width"),
            "activation": data.get("activation"),
            "loss_function": data.get("loss_function"),
            "coupling_type": data.get("coupling_type"),
            "max_level": data.get("max_level"),
            "dimension_sequence": json.dumps(dims),
            "new_per_level": json.dumps(data.get("new_per_level")),
            "dim_L0": dims[0] if len(dims) > 0 else None,
            "dim_L1": dims[1] if len(dims) > 1 else None,
            "dim_L2": dims[2] if len(dims) > 2 else None,
            "dim_L3": dims[3] if len(dims) > 3 else None,
            "matches_physical_116": matches_physical,
            "universality_class_L3": class_label,
            "n_generators": data.get("n_generators"),
            "stabilized": data.get("stabilized"),
            "is_exact": data.get("computation_method") == "exact_gaussian_elimination_QQ",
            "computation_method": data.get("computation_method"),
            "computation_time_s": data.get("computation_time_seconds"),
            "source_file": str(fp.relative_to(ROOT)),
        })

    df = pd.DataFrame(rows)
    for col in ["n_layers", "width", "max_level", "dim_L0",
                 "dim_L1", "dim_L2", "dim_L3", "n_generators"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_dataset_info(tables: dict[str, pd.DataFrame]):
    info = {
        "description": "Pairwise Poisson Algebras of the N-Body Problem",
        "version": "2.0.0",
        "splits": {},
    }
    for name, df in tables.items():
        info["splits"][name] = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
        }
    with open(OUTPUT / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    builders = {
        "dimension_sequences": build_dimension_sequences,
        "structure_constants": build_structure_constants,
        "charge_sensitivity": build_charge_sensitivity,
        "mass_invariance": build_mass_invariance,
        "level4_convergence": build_level4_convergence,
        "spectral_statistics": build_spectral_statistics,
        "physical_systems": build_physical_systems,
        "bell_test": build_bell_test,
        "scaling_formulas": build_scaling_formulas,
        "tier_decomposition": build_tier_decomposition,
        "contextuality": build_contextuality,
        "convergence_trajectories": build_convergence_trajectories,
        "neural_algebras": build_neural_algebras,
    }

    tables = {}
    for name, builder in builders.items():
        print(f"Building {name}...")
        df = builder()
        fp = OUTPUT / f"{name}.parquet"
        df.to_parquet(fp, index=False, engine="pyarrow")
        tables[name] = df
        print(f"  -> {len(df)} rows, {len(df.columns)} columns -> {fp.name}")

    write_dataset_info(tables)
    print(f"\nAll tables written to {OUTPUT}")

    total_rows = sum(len(df) for df in tables.values())
    print(f"Total: {total_rows} rows across {len(tables)} tables")


if __name__ == "__main__":
    main()
