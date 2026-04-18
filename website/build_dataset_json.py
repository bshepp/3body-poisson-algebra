#!/usr/bin/env python3
"""Build website JSON files from the curated dataset and image inventory.

Reads:
  dataset/output/*.parquet       (13 curated tables, 993 rows total)
  dataset/output/dataset_info.json
  data/image_inventory.json

Writes:
  website/data/datasets/manifest.json
  website/data/datasets/<table>.json   (one per table)
  website/data/figures/manifest.json

Run from the repository root:
    python website/build_dataset_json.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

try:
    import pandas as pd
except ImportError:
    sys.stderr.write("pandas is required. pip install pandas pyarrow\n")
    raise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(ROOT, "dataset", "output")
INFO_PATH = os.path.join(PARQUET_DIR, "dataset_info.json")
IMG_INV_PATH = os.path.join(ROOT, "data", "image_inventory.json")

OUT_DATA = os.path.join(ROOT, "website", "data")
OUT_DATASETS = os.path.join(OUT_DATA, "datasets")
OUT_FIGURES = os.path.join(OUT_DATA, "figures")

HF_BASE = "https://huggingface.co/datasets/bshepp/pairwise-poisson-algebras/resolve/main"

# ─── Manual per-table metadata (matches dataset/README.md descriptions) ───
TABLE_META: dict[str, dict[str, Any]] = {
    "dimension_sequences": {
        "label": "Dimension Sequences",
        "section": "dim",
        "summary": "Master catalog: one row per (N, d, potential, bracket_type) configuration. Includes flattened dim_L0..dim_L4 columns alongside the JSON dimension_sequence.",
        "default_sort": ["N", "d", "potential"],
    },
    "physical_systems": {
        "label": "Physical Systems",
        "section": "dim",
        "summary": "Named physical systems (helium, triple BH, Penning trap, ...) with their computed dimension sequences across astrophysical, atomic, ion-trap, and condensed-matter categories.",
        "default_sort": ["category", "system_name"],
    },
    "neural_algebras": {
        "label": "Neural Algebras",
        "section": "dim",
        "summary": "Lie algebras from neural network training dynamics. Sweeps depth, width, coupling type, loss, and activation. Reveals seven L=3 universality classes.",
        "default_sort": ["coupling_type", "n_layers"],
    },
    "scaling_formulas": {
        "label": "Scaling Formulas",
        "section": "dim",
        "summary": "Closed-form scaling formulas for algebra dimension as a function of N. Documents the original conjecture, falsification at N=7, and the corrected formula.",
        "default_sort": ["level"],
    },
    "mass_invariance": {
        "label": "Mass Invariance",
        "section": "sweep",
        "summary": "Sweep of m3 from 1 to 10^10 (m1=m2=1). Shows dimension invariance through moderate ratios and conditioning degradation at extremes.",
        "default_sort": ["m3"],
    },
    "charge_sensitivity": {
        "label": "Charge Sensitivity",
        "section": "sweep",
        "summary": "Tests whether the algebra dimension depends on particle charges q_i across 38 configurations. Mixed-sign charges break the universal 116 at level 3.",
        "default_sort": ["experiment_key"],
    },
    "level4_convergence": {
        "label": "Level-4 Convergence",
        "section": "sweep",
        "summary": "Level-4 lower bounds from numerical SVD at increasing sample sizes across global, euler, lagrange, and scalene configurations.",
        "default_sort": ["config", "n_samples"],
    },
    "convergence_trajectories": {
        "label": "Convergence Trajectories",
        "section": "sweep",
        "summary": "Per-(N,d,potential,level) convergence: rank and gap ratio as a function of n_samples and n_candidates.",
        "default_sort": ["N", "d", "potential", "level"],
    },
    "structure_constants": {
        "label": "Structure Constants",
        "section": "algebra",
        "summary": "Exact rational structure constants C^k_ij for the bracket algebra (N=3) across 16 potentials, with Killing form, derived/lower-central series, and (semi)simplicity/solvability/nilpotency invariants.",
        "default_sort": ["potential"],
    },
    "spectral_statistics": {
        "label": "Spectral Statistics",
        "section": "algebra",
        "summary": "Rank distributions across phase space from atlas scans: rank min/max/mode and pct of points achieving rank 116.",
        "default_sort": ["config"],
    },
    "tier_decomposition": {
        "label": "Tier Decomposition",
        "section": "algebra",
        "summary": "S3/S4 irrep decomposition per (N, level): irrep multiplicities and total contribution to the observed rank.",
        "default_sort": ["N", "level", "irrep_name"],
    },
    "bell_test": {
        "label": "Bell Test (CHSH)",
        "section": "quantum",
        "summary": "CHSH Bell inequality tests on the Poisson algebra across three phase-space strata x three observable variants. All respect the classical bound.",
        "default_sort": ["stratum", "variant"],
    },
    "contextuality": {
        "label": "Contextuality",
        "section": "quantum",
        "summary": "Per-(N, d, potential) commuting-pair fractions and Kochen-Specker / Peres-Mermin colorability flags.",
        "default_sort": ["N", "d", "potential"],
    },
}

SECTIONS: dict[str, dict[str, str]] = {
    "dim": {"label": "Dimension Sequences", "order": "1"},
    "sweep": {"label": "Physical Sweeps", "order": "2"},
    "algebra": {"label": "Algebra Structure", "order": "3"},
    "quantum": {"label": "Quantum / Contextuality", "order": "4"},
}


def normalize_value(v: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-safe primitives."""
    if v is None:
        return None
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                return None
            return f
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, np.ndarray):
            return [normalize_value(x) for x in v.tolist()]
    except ImportError:
        pass
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
    if isinstance(v, (list, tuple)):
        return [normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): normalize_value(x) for k, x in v.items()}
    return v


def expand_structure_constants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the flat dim x dim x dim string-rational tensor with a sparse
    list of nonzero entries plus a small summary block, to keep payloads sane
    while preserving full information."""
    out = []
    for r in rows:
        r2 = dict(r)
        sc = r.get("structure_constants")
        if isinstance(sc, str):
            try:
                sc = json.loads(sc)
            except Exception:
                sc = None
        nonzero: list[list[Any]] = []
        if isinstance(sc, list):
            dim = len(sc)
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        c = sc[i][j][k]
                        if isinstance(c, str):
                            cs = c.strip()
                        else:
                            cs = str(c)
                        if cs and cs != "0":
                            nonzero.append([i, j, k, cs])
            r2["structure_constants"] = {
                "dim": dim,
                "nonzero_count": len(nonzero),
                "nonzero": nonzero,
            }
        # Parse small JSON-encoded fields for direct use in the browser
        for jcol in ("killing_signature", "derived_series", "lower_central_series"):
            v = r.get(jcol)
            if isinstance(v, str):
                try:
                    r2[jcol] = json.loads(v)
                except Exception:
                    pass
        out.append(r2)
    return out


def export_tables(info: dict[str, Any]) -> dict[str, Any]:
    os.makedirs(OUT_DATASETS, exist_ok=True)
    splits = info.get("splits", {})
    manifest_tables: list[dict[str, Any]] = []
    total_rows = 0
    for table_name, split_info in splits.items():
        meta = TABLE_META.get(table_name)
        if meta is None:
            print(f"  WARN: no metadata for table '{table_name}', skipping")
            continue
        parquet_path = os.path.join(PARQUET_DIR, f"{table_name}.parquet")
        if not os.path.exists(parquet_path):
            print(f"  WARN: missing {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        records = df.to_dict(orient="records")
        records = [{k: normalize_value(v) for k, v in r.items()} for r in records]

        if table_name == "structure_constants":
            records = expand_structure_constants(records)

        out_path = os.path.join(OUT_DATASETS, f"{table_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, separators=(",", ":"), ensure_ascii=False)

        # Build per-column metadata
        columns = []
        for col in df.columns:
            ser = df[col]
            dtype = str(ser.dtype)
            sample = None
            for v in ser.dropna().head(1).tolist():
                sample = v
                break
            columns.append({"name": col, "dtype": dtype, "sample": str(sample)[:80] if sample is not None else None})

        size_kb = round(os.path.getsize(out_path) / 1024, 1)
        n_rows = int(len(records))
        total_rows += n_rows
        entry = {
            "id": table_name,
            "label": meta["label"],
            "section": meta["section"],
            "summary": meta["summary"],
            "n_rows": n_rows,
            "n_columns": int(len(df.columns)),
            "columns": columns,
            "default_sort": meta.get("default_sort", []),
            "json": f"data/datasets/{table_name}.json",
            "size_kb": size_kb,
            "parquet_url": f"{HF_BASE}/{table_name}.parquet",
        }
        manifest_tables.append(entry)
        print(f"  {table_name:25s} {n_rows:5d} rows  {size_kb:7.1f} KB")

    section_blocks = []
    for sid, sinfo in sorted(SECTIONS.items(), key=lambda kv: kv[1]["order"]):
        section_blocks.append({
            "id": sid,
            "label": sinfo["label"],
            "tables": [t["id"] for t in manifest_tables if t["section"] == sid],
        })

    manifest = {
        "version": info.get("version", "unknown"),
        "generated_from": "dataset/output/*.parquet",
        "hf_base": HF_BASE,
        "total_rows": total_rows,
        "total_tables": len(manifest_tables),
        "sections": section_blocks,
        "tables": manifest_tables,
    }
    with open(os.path.join(OUT_DATASETS, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


# ─── Figures manifest ───

# Categories from data/image_inventory.json that are NOT figures we want to expose.
DROP_CATEGORIES = {
    "12_animations_videos",         # mp4s + intermediate frames belong elsewhere
    "14_manim_scene_videos",        # videos
    "15_manim_tex_svgs",            # build artifacts
    "16_manim_partial_movie_clips", # build artifacts
    "17_aws_results_mirrors",       # duplicates of primary
}

CATEGORY_LABELS = {
    "1_atlas_landscapes_shape_spheres": "Shape Sphere Atlases",
    "2_triptychs": "Triptych Compositions",
    "3_svd_spectra": "SVD Spectra (N-body)",
    "4_spectral_depth": "Spectral Depth Mining",
    "5_potential_comparisons": "Potential Comparisons",
    "6_nbody_scaling": "N-body / Helium / Bell",
    "7_mass_ratio_sweeps": "Mass Ratio Sweeps",
    "8_4k_renders": "4K Renders",
    "9_paper_figures": "Paper Figures",
    "10_clebsch_gordan": "Clebsch-Gordan / S3",
    "11_level4_analysis": "Level-4 Analysis",
    "13_targeted_atlas_scans": "Targeted Atlas Scans",
}

EXT_OK = {".png", ".svg", ".gif"}

TAG_RULES = [
    ("triptych", lambda p: "/triptychs/" in p or "triptych_" in os.path.basename(p)),
    ("atlas", lambda p: "atlas" in p.lower() or "shape_sphere" in p),
    ("targeted", lambda p: "/atlas_targeted/" in p or "targeted_" in os.path.basename(p)),
    ("hires", lambda p: "hires" in p or "_4k" in p or "/atlas_output_hires/" in p),
    ("charge", lambda p: "coulomb" in p or "charge" in p),
    ("epsilon", lambda p: "/multi_epsilon/" in p or "_eps_" in p or "epsilon" in p),
    ("spectral", lambda p: "/spectral_depth/" in p or "spectral_" in p or "/sv_analysis/" in p),
    ("svd", lambda p: "svd" in p or "sv_landscapes" in p or "sv116" in p),
    ("nbody", lambda p: "/nbody/" in p or re.search(r"_N\d+_d\d+", p) is not None),
    ("level4", lambda p: "level4" in p),
    ("helium", lambda p: "helium" in p),
    ("bell", lambda p: "bell" in p),
    ("paper", lambda p: "/calogero_paper/" in p or os.path.basename(p).startswith("fig_") or p.startswith("fig_")),
    ("animation", lambda p: p.endswith(".gif") or "/animations/" in p or "atlas_animation" in p),
    ("mass-ratio", lambda p: "mass_ratio" in p),
]


def humanize(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\bfullsphere\b", "Full Sphere", base, flags=re.I)
    base = re.sub(r"\b1 r\b", "1/r", base)
    base = re.sub(r"\b1 r2\b", "1/r2", base)
    base = re.sub(r"\b1 r3\b", "1/r3", base)
    base = re.sub(r"\b1 r4\b", "1/r4", base)
    base = re.sub(r"\bp(\d)\b", r"+\1", base)
    base = re.sub(r"\bm(\d)\b", r"-\1", base)
    base = re.sub(r"\s+", " ", base).strip()
    parts = base.split()
    out = []
    for p in parts:
        if p.lower() in {"sv", "svd", "cg", "lqg", "s3", "p2", "4k", "ps", "bh"}:
            out.append(p.upper())
        elif p.lower() in {"vs", "of", "to", "and"}:
            out.append(p.lower())
        elif "/" in p:
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:])
    return " ".join(out) if out else base


def derive_tags(path: str) -> list[str]:
    p = path.replace("\\", "/")
    return [name for name, fn in TAG_RULES if fn(p)]


def export_figures(img_inv: dict[str, Any]) -> dict[str, Any]:
    os.makedirs(OUT_FIGURES, exist_ok=True)
    figures: list[dict[str, Any]] = []
    seen = set()
    categories_used: dict[str, int] = {}
    cats = img_inv.get("categories", {})
    for cat_id, cat in cats.items():
        if cat_id in DROP_CATEGORIES:
            continue
        files = cat.get("files") or []
        if not files:
            continue
        cat_label = CATEGORY_LABELS.get(cat_id, cat.get("description", cat_id))
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in EXT_OK:
                continue
            norm = f.replace("\\", "/")
            if norm in seen:
                continue
            seen.add(norm)
            tags = derive_tags(norm)
            figures.append({
                "name": humanize(norm),
                "path": norm,
                "category": cat_id,
                "category_label": cat_label,
                "ext": ext.lstrip("."),
                "tags": tags,
            })
            categories_used[cat_id] = categories_used.get(cat_id, 0) + 1

    # Sort by category order then by name within category
    cat_order = {cid: i for i, cid in enumerate(CATEGORY_LABELS.keys())}
    figures.sort(key=lambda x: (cat_order.get(x["category"], 999), x["name"].lower()))

    manifest = {
        "generated_from": "data/image_inventory.json",
        "n_figures": len(figures),
        "categories": [
            {"id": cid, "label": CATEGORY_LABELS.get(cid, cid), "count": categories_used[cid]}
            for cid in CATEGORY_LABELS
            if cid in categories_used
        ],
        "tags": sorted({t for fig in figures for t in fig["tags"]}),
        "figures": figures,
    }
    out_path = os.path.join(OUT_FIGURES, "manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"), ensure_ascii=False)
    print(f"  figures/manifest.json: {len(figures)} figures across "
          f"{len(manifest['categories'])} categories ({os.path.getsize(out_path)//1024} KB)")
    return manifest


def main() -> int:
    if not os.path.exists(INFO_PATH):
        sys.stderr.write(f"missing {INFO_PATH}\n")
        return 1
    if not os.path.exists(IMG_INV_PATH):
        sys.stderr.write(f"missing {IMG_INV_PATH}\n")
        return 1

    print("Building dataset JSON files...")
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    ds_manifest = export_tables(info)
    print(f"  total: {ds_manifest['total_rows']} rows across "
          f"{ds_manifest['total_tables']} tables")

    print("Building figures manifest...")
    with open(IMG_INV_PATH, "r", encoding="utf-8") as f:
        img_inv = json.load(f)
    export_figures(img_inv)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
