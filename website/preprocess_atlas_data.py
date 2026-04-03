#!/usr/bin/env python3
"""
Preprocess atlas .npy data into JSON files for the interactive web explorer.

Outputs:
  website/data/manifest.json          — index of all scans + supplementary datasets
  website/data/atlas_{id}.json        — per-scan: rank_map, gap_map, mu_vals, phi_vals
  website/data/atlas_{id}_sv.bin      — per-scan: sv_spectra as Float32 binary (on-demand)
  website/data/mass_ratio.json        — mass ratio sweep data
  website/data/nbody_scaling.json     — N-body scaling results

Run from the repository root:
    python website/preprocess_atlas_data.py
"""

import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_HIRES = os.path.join(ROOT, "atlas_output_hires")
HELIUM_DIR = os.path.join(ROOT, "nbody", "atlas_helium_d2")
MASS_RATIO_FILE = os.path.join(ROOT, "data", "mass_ratio_sweep.json")
NBODY_SCALING_FILE = os.path.join(ROOT, "nbody", "n_body_scaling_results.json")
OUT_DIR = os.path.join(ROOT, "website", "data")

os.makedirs(OUT_DIR, exist_ok=True)


def scan_id_from_path(path):
    """Generate a stable short ID from a scan directory path."""
    rel = os.path.relpath(path, ATLAS_HIRES).replace("\\", "/")
    return rel.replace("/", "_").replace("+", "p").replace("-", "m")


def export_scan(scan_dir, scan_id, manifest_entry):
    """Export one scan directory to JSON + binary."""
    rank = np.load(os.path.join(scan_dir, "rank_map.npy"))
    gap = np.load(os.path.join(scan_dir, "gap_map.npy"))
    mu = np.load(os.path.join(scan_dir, "mu_vals.npy"))
    phi = np.load(os.path.join(scan_dir, "phi_vals.npy"))

    # JSON: rank_map (int), gap_map (float, clamped for JSON safety),
    #        mu_vals, phi_vals
    gap_safe = np.where(np.isfinite(gap), gap, 0.0)

    atlas_data = {
        "mu": mu.tolist(),
        "phi": phi.tolist(),
        "rank": rank.astype(int).tolist(),
        "gap": gap_safe.tolist(),
        "grid_n": int(rank.shape[0]),
    }

    json_path = os.path.join(OUT_DIR, f"atlas_{scan_id}.json")
    with open(json_path, "w") as f:
        json.dump(atlas_data, f, separators=(",", ":"))
    manifest_entry["json"] = f"data/atlas_{scan_id}.json"

    # Binary: sv_spectra as Float32 (row-major)
    sv_path = os.path.join(scan_dir, "sv_spectra.npy")
    if os.path.exists(sv_path):
        sv = np.load(sv_path).astype(np.float32)
        # Replace non-finite with 0
        sv = np.where(np.isfinite(sv), sv, 0.0).astype(np.float32)
        bin_path = os.path.join(OUT_DIR, f"atlas_{scan_id}_sv.bin")
        sv.tofile(bin_path)
        manifest_entry["sv_bin"] = f"data/atlas_{scan_id}_sv.bin"
        manifest_entry["sv_shape"] = list(sv.shape)

    sz = os.path.getsize(json_path)
    print(f"  {scan_id}: JSON {sz//1024}KB", end="")
    if "sv_bin" in manifest_entry:
        sz2 = os.path.getsize(bin_path)
        print(f" + SV {sz2//1024//1024}MB", end="")
    print()


def main():
    manifest = {"scans": [], "datasets": {}}

    # ── Atlas hires scans ──
    print("Exporting atlas_output_hires scans...")
    for dirpath, dirnames, filenames in os.walk(ATLAS_HIRES):
        if "config.json" not in filenames:
            continue
        if "rank_map.npy" not in filenames:
            continue

        config = json.load(open(os.path.join(dirpath, "config.json")))
        scan_id = scan_id_from_path(dirpath)

        entry = {
            "id": scan_id,
            "source": "atlas_hires",
            "potential": config.get("potential", "unknown"),
            "epsilon": config.get("epsilon", config.get("eps_range", "adaptive")),
            "level": config.get("level", 3),
            "n_generators": config.get("n_generators", 0),
            "n_samples": config.get("n_samples", 0),
            "grid_n": config.get("grid_n", 0),
            "mu_range": config.get("mu_range", []),
            "phi_range": config.get("phi_range", []),
        }
        if "charges" in config:
            entry["charges"] = config["charges"]
        if "mode" in config:
            entry["mode"] = config["mode"]

        export_scan(dirpath, scan_id, entry)
        manifest["scans"].append(entry)

    # ── Helium atlas ──
    if os.path.isdir(HELIUM_DIR) and os.path.exists(
        os.path.join(HELIUM_DIR, "rank_map.npy")
    ):
        print("Exporting helium atlas...")
        config = json.load(open(os.path.join(HELIUM_DIR, "config.json")))
        scan_id = "helium_d2"
        entry = {
            "id": scan_id,
            "source": "helium",
            "potential": config.get("potential", "coulomb_helium"),
            "epsilon": config.get("epsilon", 0),
            "level": config.get("level", 3),
            "n_generators": config.get("n_generators", 0),
            "n_samples": config.get("n_samples", 0),
            "grid_n": config.get("grid_n", 0),
            "mu_range": config.get("mu_range", []),
            "phi_range": config.get("phi_range", []),
        }
        if "charges" in config:
            entry["charges"] = config["charges"]
        export_scan(HELIUM_DIR, scan_id, entry)
        manifest["scans"].append(entry)

    # ── Mass ratio sweep ──
    if os.path.exists(MASS_RATIO_FILE):
        print("Including mass ratio sweep...")
        with open(MASS_RATIO_FILE) as f:
            mass_data = json.load(f)
        # Slim down: keep dim sequences and gap ratios, drop raw singular values
        slim = []
        for item in mass_data:
            s = {"m3": item["m3"], "m3_log10": item.get("m3_log10", 0)}
            for lvl in ("level_0", "level_1", "level_2"):
                if lvl in item:
                    s[lvl] = {
                        "dim": item[lvl]["dim"],
                        "gap_ratio": item[lvl].get("gap_ratio", 0),
                    }
            slim.append(s)

        out_path = os.path.join(OUT_DIR, "mass_ratio.json")
        with open(out_path, "w") as f:
            json.dump(slim, f, separators=(",", ":"))
        manifest["datasets"]["mass_ratio"] = {
            "path": "data/mass_ratio.json",
            "n_entries": len(slim),
            "description": "Dimension sequence vs mass ratio m3 (m1=m2=1)",
        }
        print(f"  mass_ratio.json: {os.path.getsize(out_path)//1024}KB")

    # ── N-body scaling ──
    if os.path.exists(NBODY_SCALING_FILE):
        print("Including N-body scaling results...")
        with open(NBODY_SCALING_FILE) as f:
            nbody_data = json.load(f)
        out_path = os.path.join(OUT_DIR, "nbody_scaling.json")
        with open(out_path, "w") as f:
            json.dump(nbody_data, f, separators=(",", ":"))
        manifest["datasets"]["nbody_scaling"] = {
            "path": "data/nbody_scaling.json",
            "n_entries": len(nbody_data),
            "description": "Dimension sequences for N=5..9 bodies (d=1, 1/r potential)",
        }
        print(f"  nbody_scaling.json: {os.path.getsize(out_path)//1024}KB")

    # ── Sort scans by potential, then epsilon ──
    def sort_key(s):
        eps = s["epsilon"] if isinstance(s["epsilon"], (int, float)) else 999
        return (s["potential"], eps)

    manifest["scans"].sort(key=sort_key)

    # ── Write manifest ──
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {len(manifest['scans'])} scans, "
          f"{len(manifest['datasets'])} supplementary datasets")
    print(f"Output directory: {OUT_DIR}")

    # Total size
    total = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR)
    )
    print(f"Total output size: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
