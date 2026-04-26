#!/usr/bin/env python3
"""Build website/data/figures/manifest.json from the figures_v2/ tree.

The manifest is the source of truth for the Figures page. Each figure gets:
  - canonical id, name, path
  - system + analysis classification (drives the System / Analysis browse modes)
  - n, d, caption (one-sentence descriptor)
  - data_links (back-links into the Datasets page)
  - groups (membership in curated comparison sets, drives the Comparisons mode)

Run from the repository root:
    python website/build_figures_manifest.py
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures_v2")
OUT_DIR = os.path.join(ROOT, "website", "data", "figures")
OUT_PATH = os.path.join(OUT_DIR, "manifest.json")


# ─── System classification ───
# Each system has a stable id, a display label, and a one-line summary used
# in the Figures rail.

SYSTEMS = {
    "1_r":           {"label": "1/r (Newton)",          "summary": "Gravitational potential — universal baseline.",      "order": 1},
    "1_r2":          {"label": "1/r^2 (Calogero-Moser)","summary": "Inverse-square; integrable yet universal sequence.",  "order": 2},
    "1_r3":          {"label": "1/r^3",                 "summary": "Cubic singularity — universal sequence.",            "order": 3},
    "1_r4":          {"label": "1/r^4",                 "summary": "Quartic singularity.",                                "order": 4},
    "1_r_irrational":{"label": "1/r^k (irrational k)",  "summary": "Irrational exponents pi, e, phi — all universal.",    "order": 5},
    "1_r_continuity":{"label": "1/r^k (sweep)",         "summary": "Continuity strip across exceptional integers.",       "order": 6},
    "log":           {"label": "log(r)",                "summary": "Logarithmic potential — 2D vortices, GUE log-gas.",   "order": 7},
    "harmonic":      {"label": "r^2 (harmonic)",        "summary": "Spring potential — the only finite algebra.",        "order": 8},
    "rn_poly":       {"label": "r^n (polynomial)",      "summary": "Polynomial potentials beyond harmonic.",              "order": 9},
    "coulomb":       {"label": "Coulomb (charged)",     "summary": "Three-body Coulomb — varying charges and masses.",    "order": 10},
    "coulomb_1_r2":  {"label": "Coulomb 1/r^2",         "summary": "Inverse-square Coulomb variants.",                    "order": 11},
    "coulomb_1_r3":  {"label": "Coulomb 1/r^3",         "summary": "Inverse-cube Coulomb variants.",                      "order": 12},
    "atomic":        {"label": "Atomic systems",        "summary": "He, H-, H2+, Li+, Ps-, muonic He.",                   "order": 13},
    "astrophysical": {"label": "Astrophysical",         "summary": "Sun-Earth-Moon, Sun-Jupiter-Asteroid, BH triples, dark matter.", "order": 14},
    "neural":        {"label": "Neural networks",       "summary": "SGD-with-momentum on linear networks.",               "order": 15},
    "gue":           {"label": "GUE log-gas",           "summary": "Dyson model for Riemann-zeta zero spacings.",         "order": 16},
    "quantum":       {"label": "Quantum (Moyal)",       "summary": "Quantum bracket deformation: +1 generator at L3.",    "order": 17},
    "tier":          {"label": "Tier decomposition",    "summary": "S3 / S4 irrep structure of generators.",              "order": 18},
}

ANALYSES = {
    "heatmap":     {"label": "Gap-ratio heatmap",       "summary": "log10(sigma_116/sigma_117) over the (mu, phi) shape sphere.", "order": 1},
    "sphere":      {"label": "Shape sphere",            "summary": "Same data wrapped onto the unit sphere S^2.",                "order": 2},
    "triptych":    {"label": "Triptych vs baseline",    "summary": "1/r baseline | this config | difference.",                   "order": 3},
    "spectrum":    {"label": "Singular-value spectrum", "summary": "Mean SV spectrum across the scan grid (10-90 percentile band).", "order": 4},
    "comparison":  {"label": "Curated comparison",      "summary": "Multi-panel figures with shared color scale.",               "order": 5},
    "knee":        {"label": "Spectral knee",           "summary": "Index of the steepest log-drop in the SV spectrum.",         "order": 6},
    "bar_chart":   {"label": "Algebra summary chart",   "summary": "Cross-system comparisons of dimensions / classes.",          "order": 7},
}

# ─── Curated comparison groups (drive the Comparisons tab) ───
# Each group lists the figure ids that comprise it (members are populated by
# matching against figure ids generated below).

GROUPS = [
    {
        "id": "singular_vs_harmonic",
        "label": "Singular vs harmonic dichotomy",
        "summary": "Four atlases stacked: 1/r, 1/r^2, log, r^2. Three are universal; r^2 is the lone finite case.",
        "primary_figure": "comparison__singular_vs_harmonic",
        "members": ["awsfull__1r_q+1_+1_+1", "awsfull__1r2", "awsfull__log", "awsfull__1r-2"],
    },
    {
        "id": "irrational_exponents",
        "label": "Irrational exponents",
        "summary": "1/r^pi, 1/r^e, 1/r^phi all give the universal sequence.",
        "primary_figure": "comparison__irrational_exponents",
        "members": ["awsfull__1r3p14159", "awsfull__1r2p71828", "awsfull__1r1p61803"],
    },
    {
        "id": "exponent_continuity",
        "label": "Exponent continuity",
        "summary": "Sweep across the exceptional integer exponents to visualise the sharpness of the transition.",
        "primary_figure": "comparison__exponent_continuity",
        "members": ["awsfull__1r_q+1_+1_+1", "awsfull__1r1p01", "awsfull__1r2", "awsfull__1r2p01", "awsfull__1r3"],
    },
    {
        "id": "mass_ratio_extremes",
        "label": "Mass-ratio extremes",
        "summary": "Equal mass | Sun-Earth-Moon | Sun-Jupiter-Asteroid | LISA triple BH. Extreme ratios degrade conditioning.",
        "primary_figure": "comparison__mass_ratio_extremes",
        "members": [
            "awsfull__1r_q+1_+1_+1",
            "awsfull__sun_earth_moon_1r_m1p0_3e-06_3p7e-08",
            "awsfull__sun_jupiter_asteroid_1r_m1p0_0p00095_1e-10",
            "awsfull__triple_bh_lisa_1r_m1p0_0p01_1e-05",
        ],
    },
    {
        "id": "charge_departure",
        "label": "Charge-magnitude departure",
        "summary": "He (+2,-1,-1) vs Li+ (+3,-1,-1) vs H2+ (+1,+1,-1) -- three real departures from L3=116.",
        "primary_figure": "comparison__charge_departure",
        "members": ["awsfull__1r_q+2_-1_-1", "awsfull__1r_q+3_-1_-1", "awsfull__1r_q+1_+1_-1"],
    },
    {
        "id": "quantum_plus_one",
        "label": "Quantum +1 generator",
        "summary": "Replacing the Poisson bracket with the Moyal bracket adds exactly one extra dimension at L3.",
        "primary_figure": "comparison__quantum_plus_one",
        "members": [],
    },
    {
        "id": "neural_classes",
        "label": "Neural-network classes",
        "summary": "Seven coupling-defined universality classes against the physics universal 116.",
        "primary_figure": "comparison__neural_classes",
        "members": [],
    },
    {
        "id": "tier_decomposition",
        "label": "S3 / S4 tier decomposition",
        "summary": "Stacked irrep contributions per (N, level).",
        "primary_figure": "comparison__tier_decomposition",
        "members": [],
    },
    {
        "id": "gue_overlay",
        "label": "GUE log-gas overlay",
        "summary": "Spacing distributions match random-matrix predictions for the GUE log-gas.",
        "primary_figure": "comparison__gue_overlay",
        "members": [],
    },
    {
        "id": "knee_landscape_set",
        "label": "Spectral knee landscape (cross-potential)",
        "summary": "Side-by-side knee landscapes for 1/r, 1/r^2, harmonic.",
        "primary_figure": "comparison__knee_landscape_set",
        "members": [],
    },
]


# ─── Stem -> classification ───

def classify(folder: str, fname: str) -> dict:
    """Return classification dict (system, n, d, caption, data_links, groups, name).

    folder is one of "heatmaps", "spheres", "triptychs", "spectra", "comparisons".
    fname is the file name without extension.
    """
    info: dict = {"system": "1_r", "n": 3, "d": None, "groups": [], "data_links": [], "caption": ""}

    # Comparisons: id == fname; a row from GROUPS has the rest
    if folder == "comparisons":
        for g in GROUPS:
            if g["id"] == fname:
                info.update({
                    "system": _comparison_system(g["id"]),
                    "groups": [g["id"]],
                    "caption": g["summary"],
                    "n": _comparison_n(g["id"]),
                    "d": _comparison_d(g["id"]),
                    "data_links": _comparison_data_links(g["id"]),
                })
                return info
        info["caption"] = f"Comparison: {fname.replace('_', ' ')}"
        return info

    # awsfull / hires / targeted parsing
    parts = fname.split("__")
    if not parts:
        return info
    src = parts[0]            # awsfull | hires | targeted
    cfg = parts[1] if len(parts) > 1 else ""
    sub = parts[2] if len(parts) > 2 else ""   # eps_X / region

    info["d"] = 2 if src == "awsfull" else 2
    sys_id, sys_caption = classify_config(src, cfg, sub)
    info["system"] = sys_id
    info["caption"] = sys_caption

    # Group memberships
    full_id = f"{src}__{cfg}"
    for g in GROUPS:
        if full_id in g["members"]:
            info["groups"].append(g["id"])

    # Data links: best-effort link to dimension_sequences row
    info["data_links"] = build_data_links(src, cfg, sub)
    return info


def classify_config(src: str, cfg: str, sub: str) -> tuple[str, str]:
    """Map (src, cfg, sub) -> (system_id, caption)."""
    # Coulomb charge configurations
    if cfg.startswith("coulomb_1_r2"):
        return "coulomb_1_r2", "Inverse-square Coulomb (+2,-1,-1)."
    if cfg.startswith("coulomb_1_r3"):
        return "coulomb_1_r3", "Inverse-cube Coulomb (+2,-1,-1)."
    if cfg.startswith("coulomb_") or cfg.startswith("1r_q") or "1r2_q+2" in cfg or "1r3_q+2" in cfg:
        return "coulomb", f"Three-body Coulomb {cfg.replace('coulomb_', '').replace('1r_', '')}."
    # Named atomic / astrophysical
    if cfg.startswith("h2_plus_ion") or cfg.startswith("h_minus_ion") or \
       cfg.startswith("lithium_ion") or cfg.startswith("muonic_helium") or \
       cfg.startswith("positronium"):
        return "atomic", f"Atomic system: {cfg.split('_1r_')[0]}."
    if cfg.startswith("sun_earth_moon"):
        return "astrophysical", "Sun-Earth-Moon mass ratios."
    if cfg.startswith("sun_jupiter_asteroid"):
        return "astrophysical", "Sun-Jupiter-Asteroid mass ratios."
    if cfg.startswith("triple_bh_lisa"):
        return "astrophysical", "LISA triple black-hole system."
    if cfg.startswith("binary_bh_ns"):
        return "astrophysical", "Binary BH + neutron star."
    if cfg.startswith("binary_star_planet"):
        return "astrophysical", "Binary star + planet."
    if cfg.startswith("dark_matter"):
        return "astrophysical", "Dark matter halo three-body."
    # Power laws
    if cfg in ("1r_q+1_+1_+1", "1_r"):
        return "1_r", "Equal-mass equal-charge 1/r baseline (Newton)."
    if cfg in ("1r2", "1_r2"):
        return "1_r2", "1/r^2 inverse-square (Calogero-Moser)."
    if cfg in ("1r3", "1_r3"):
        return "1_r3", "1/r^3 inverse-cube."
    if cfg == "1r-2":
        return "harmonic", "r^2 harmonic potential."
    if cfg == "1r-5":
        return "rn_poly", "r^5 polynomial potential."
    if cfg == "1r0":
        return "log", "log(r) (encoded as 1/r^0)."
    if cfg in ("log",):
        return "log", "log(r) potential."
    if cfg in ("1r1p61803", "1r2p71828", "1r3p14159"):
        return "1_r_irrational", f"1/r^{cfg.replace('1r','').replace('p','.')} (irrational)."
    if cfg in ("1r1p01", "1r2p01"):
        return "1_r_continuity", f"1/r^{cfg.replace('1r','').replace('p','.')} (continuity sweep)."
    if cfg.startswith("harmonic"):
        return "harmonic", "r^2 harmonic potential."
    return "1_r", f"{cfg} ({src})"


def build_data_links(src: str, cfg: str, sub: str) -> list[dict]:
    """Best-effort link to the dimension_sequences table for this config."""
    if src == "awsfull":
        if cfg.startswith("1r_q+1_+1_+1") or cfg in ("1r2", "1r3", "1r-2", "1r-5"):
            potmap = {
                "1r_q+1_+1_+1": "1/r", "1r2": "1/r^2", "1r3": "1/r^3",
                "1r-2": "r^2", "1r-5": "r^5",
            }
            pot = potmap.get(cfg)
            if pot:
                return [{
                    "table": "dimension_sequences",
                    "label": f"dim sequences for {pot}",
                    "filter": f"potential={pot}",
                }]
        if cfg == "log":
            return [{"table": "dimension_sequences", "label": "log(r) entries",
                     "filter": "potential=log"}]
        if cfg.startswith("sun_earth_moon") or cfg.startswith("sun_jupiter_asteroid") \
           or cfg.startswith("triple_bh_lisa") or cfg.startswith("binary_") \
           or cfg.startswith("dark_matter"):
            return [{"table": "physical_systems", "label": "named physical systems"}]
        if cfg.startswith("1r_q") or cfg.startswith("1r2_q") or cfg.startswith("1r3_q"):
            return [{"table": "charge_sensitivity", "label": "charge sweeps"}]
        if cfg.startswith("h2_plus_ion") or cfg.startswith("h_minus_ion") \
           or cfg.startswith("lithium_ion") or cfg.startswith("muonic_helium") \
           or cfg.startswith("positronium"):
            return [{"table": "physical_systems", "label": "atomic systems"}]
    return []


def _comparison_system(gid: str) -> str:
    return {
        "singular_vs_harmonic": "harmonic",
        "irrational_exponents": "1_r_irrational",
        "exponent_continuity":  "1_r_continuity",
        "mass_ratio_extremes":  "astrophysical",
        "charge_departure":     "coulomb",
        "quantum_plus_one":     "quantum",
        "neural_classes":       "neural",
        "tier_decomposition":   "tier",
        "gue_overlay":          "gue",
        "knee_landscape_set":   "harmonic",
    }.get(gid, "1_r")


def _comparison_n(gid: str) -> int:
    return 3


def _comparison_d(gid: str) -> Optional[int]:
    if gid in {"neural_classes", "tier_decomposition"}:
        return None
    return 2


def _comparison_data_links(gid: str) -> list[dict]:
    return {
        "singular_vs_harmonic": [
            {"table": "dimension_sequences", "label": "All dim sequences"},
        ],
        "irrational_exponents": [
            {"table": "dimension_sequences", "label": "Irrational exponent rows",
             "filter": "potential=1/r^"},
        ],
        "exponent_continuity": [
            {"table": "dimension_sequences", "label": "All r^k rows"},
        ],
        "mass_ratio_extremes": [
            {"table": "mass_invariance", "label": "m3 sweep"},
            {"table": "physical_systems", "label": "named astrophysical systems"},
        ],
        "charge_departure": [
            {"table": "charge_sensitivity", "label": "charge sweep"},
        ],
        "quantum_plus_one": [
            {"table": "dimension_sequences", "label": "Moyal vs Poisson rows",
             "filter": "bracket_type=moyal"},
        ],
        "neural_classes": [
            {"table": "neural_algebras", "label": "21 NN configurations"},
        ],
        "tier_decomposition": [
            {"table": "tier_decomposition", "label": "irrep table"},
        ],
        "gue_overlay": [
            {"table": "dimension_sequences", "label": "log-gas / GUE rows",
             "filter": "potential=log"},
        ],
        "knee_landscape_set": [
            {"table": "spectral_statistics", "label": "rank distributions"},
        ],
    }.get(gid, [])


# ─── Display name from path ───

POTENTIAL_NAMES = {
    "1r_q+1_+1_+1": "Newton (1/r, equal mass)",
    "1r-2": "r^2 (harmonic)",
    "1r-5": "r^5 polynomial",
    "1r0": "log(r) (1/r^0 encoding)",
    "1r1p01": "1/r^1.01",
    "1r1p61803": "1/r^phi (golden ratio)",
    "1r2": "1/r^2 (Calogero-Moser)",
    "1r2p01": "1/r^2.01",
    "1r2p71828": "1/r^e (Euler's number)",
    "1r3": "1/r^3",
    "1r3p14159": "1/r^pi",
    "log": "log(r)",
    "1r_q+1_+1_-1": "Coulomb (+1,+1,-1) -- H2+",
    "1r_q+1_-1_-1": "Coulomb (+1,-1,-1) -- H- / Ps-",
    "1r_q+2_-1_-1": "Coulomb (+2,-1,-1) -- He",
    "1r_q+3_-1_-1": "Coulomb (+3,-1,-1) -- Li+",
    "1r2_q+2_-1_-1": "1/r^2 Coulomb (+2,-1,-1)",
    "1r3_q+2_-1_-1": "1/r^3 Coulomb (+2,-1,-1)",
    "h2_plus_ion_1r_q+1_+1_-1_m1836p0_1836p0_1p0": "H2+ (proton-proton-electron)",
    "h_minus_ion_1r_q+1_-1_-1_m1836p0_1p0_1p0": "H- (proton + 2e-)",
    "lithium_ion_1r_q+3_-1_-1_m12789p0_1p0_1p0": "Li+ ion",
    "muonic_helium_1r_q+2_-1_-1_m7294p0_1p0_207p0": "Muonic helium",
    "positronium_neg_1r_q+1_-1_-1": "Positronium (negative)",
    "sun_earth_moon_1r_m1p0_3e-06_3p7e-08": "Sun-Earth-Moon",
    "sun_jupiter_asteroid_1r_m1p0_0p00095_1e-10": "Sun-Jupiter-Asteroid",
    "triple_bh_lisa_1r_m1p0_0p01_1e-05": "Triple BH (LISA)",
    "binary_bh_ns_1r_m1p0_1p0_0p047": "Binary BH + neutron star",
    "binary_star_planet_1r_m1p0_1p0_0p001": "Binary star + planet",
    "dark_matter_1r_m1p0_5p0_10p0": "Dark matter halo",
    "1_r": "1/r (Newton)",
    "1_r2": "1/r^2",
    "harmonic": "r^2 (harmonic)",
    "coulomb_+2_-1_-1": "Coulomb (+2,-1,-1)",
    "coulomb_1_r2_+2_-1_-1": "1/r^2 Coulomb (+2,-1,-1)",
}

REGION_NAMES = {
    "lagrange":        "Lagrange equilateral",
    "euler_strip":     "Euler collinear strip",
    "isosceles_ridge": "Isosceles ridge",
    "small_mu":        "Small-mu region",
    "tier_cluster":    "Tier cluster",
    "charge_hotspot":  "Charge-sensitivity hotspot",
}


def display_name(folder: str, fname: str) -> str:
    if folder == "comparisons":
        for g in GROUPS:
            if g["id"] == fname:
                return g["label"]
        return fname.replace("_", " ").title()
    parts = fname.split("__")
    src = parts[0]
    cfg = parts[1] if len(parts) > 1 else ""
    sub = parts[2] if len(parts) > 2 else ""
    pot_label = POTENTIAL_NAMES.get(cfg, cfg)
    if sub.startswith("eps_"):
        eps = sub.replace("eps_", "")
        suffix = f" -- eps={eps}"
    elif sub == "adaptive":
        suffix = " -- adaptive"
    elif sub in REGION_NAMES:
        suffix = f" -- {REGION_NAMES[sub]}"
    elif sub:
        suffix = f" -- {sub}"
    else:
        suffix = ""
    src_tag = {"awsfull": "", "hires": " [hires]", "targeted": " [targeted]"}.get(src, "")
    return f"{pot_label}{suffix}{src_tag}"


# ─── Main build ───

def main() -> int:
    if not os.path.isdir(FIG_DIR):
        print(f"missing {FIG_DIR}")
        return 1

    figures: list[dict] = []
    for sub in sorted(os.listdir(FIG_DIR)):
        sub_dir = os.path.join(FIG_DIR, sub)
        if not os.path.isdir(sub_dir):
            continue
        analysis = {
            "heatmaps": "heatmap",
            "spheres": "sphere",
            "triptychs": "triptych",
            "spectra": "spectrum",
            "comparisons": "comparison",
        }.get(sub, sub.rstrip("s"))
        for fn in sorted(os.listdir(sub_dir)):
            if not fn.lower().endswith(".png"):
                continue
            stem = os.path.splitext(fn)[0]
            stem_for_class = stem.replace("_vs_baseline", "")
            cls = classify(sub, stem_for_class)
            figures.append({
                "id": f"{sub}__{stem}",
                "name": display_name(sub, stem_for_class) + (" (triptych)" if "_vs_baseline" in stem else ""),
                "path": f"../figures_v2/{sub}/{fn}",
                "system": cls["system"],
                "analysis": analysis,
                "n": cls["n"],
                "d": cls["d"],
                "caption": cls["caption"],
                "data_links": cls["data_links"],
                "groups": cls["groups"],
            })

    # Resolve group members from generated ids (so the manifest references
    # actual figure ids, not symbolic awsfull__X stubs).
    fig_ids = {f["id"] for f in figures}
    fig_by_stem = {}
    for f in figures:
        # For grouping, key by source__config (no analysis prefix)
        path_id = f["id"]
        # path_id like "heatmaps__awsfull__1r_q+1_+1_+1"
        if "__" in path_id:
            tail = path_id.split("__", 1)[1]
            fig_by_stem.setdefault(tail, []).append(f["id"])

    enriched_groups = []
    for g in GROUPS:
        members_resolved: list[str] = []
        # primary first if present
        primary = f"comparisons__{g['id']}"
        if primary in fig_ids:
            members_resolved.append(primary)
        # then heatmap members
        for stub in g.get("members", []):
            for fid in fig_by_stem.get(stub, []):
                if fid not in members_resolved and "heatmaps__" in fid:
                    members_resolved.append(fid)
        enriched = dict(g)
        enriched["members"] = members_resolved
        # Sync primary_figure to the actual generated id convention
        # (`comparisons__<id>`); fall back to the first resolved member.
        if primary in fig_ids:
            enriched["primary_figure"] = primary
        elif members_resolved:
            enriched["primary_figure"] = members_resolved[0]
        enriched_groups.append(enriched)
        # Tag every member figure with the group
        for fid in members_resolved:
            for f in figures:
                if f["id"] == fid and g["id"] not in f["groups"]:
                    f["groups"].append(g["id"])

    # System / analysis facets with counts
    sys_counts: dict[str, int] = {}
    ana_counts: dict[str, int] = {}
    for f in figures:
        sys_counts[f["system"]] = sys_counts.get(f["system"], 0) + 1
        ana_counts[f["analysis"]] = ana_counts.get(f["analysis"], 0) + 1

    systems_block = []
    for sid in sorted(sys_counts.keys(), key=lambda s: SYSTEMS.get(s, {}).get("order", 999)):
        s = SYSTEMS.get(sid, {"label": sid, "summary": ""})
        systems_block.append({
            "id": sid,
            "label": s["label"],
            "summary": s.get("summary", ""),
            "count": sys_counts[sid],
        })
    analyses_block = []
    for aid in sorted(ana_counts.keys(), key=lambda a: ANALYSES.get(a, {}).get("order", 999)):
        a = ANALYSES.get(aid, {"label": aid, "summary": ""})
        analyses_block.append({
            "id": aid,
            "label": a["label"],
            "summary": a.get("summary", ""),
            "count": ana_counts[aid],
        })

    manifest = {
        "n_figures": len(figures),
        "n_systems": len(systems_block),
        "n_analyses": len(analyses_block),
        "n_groups": len(enriched_groups),
        "systems":  systems_block,
        "analyses": analyses_block,
        "groups":   enriched_groups,
        "figures":  figures,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"), ensure_ascii=False)
    print(f"  figures: {len(figures)}")
    print(f"  systems: {len(systems_block)}")
    print(f"  analyses: {len(analyses_block)}")
    print(f"  groups: {len(enriched_groups)}")
    print(f"  wrote {os.path.relpath(OUT_PATH, ROOT)} "
          f"({os.path.getsize(OUT_PATH) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
