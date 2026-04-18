#!/usr/bin/env python3
"""Archive legacy figure PNGs into legacy_figures_archive/.

Source of truth: data/image_inventory.json plus a few well-known directories
that aren't fully enumerated there. Moves PNGs (preserving relative paths)
into legacy_figures_archive/, leaving NPY data and other source files alone.

Skips the new figures_v2/ output tree.

Run from the repository root:
    python scripts/archive_legacy_figures.py
    python scripts/archive_legacy_figures.py --dry-run
    python scripts/archive_legacy_figures.py --restore       # move back
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV_PATH = os.path.join(ROOT, "data", "image_inventory.json")
ARCHIVE = os.path.join(ROOT, "legacy_figures_archive")

# Directories to scan exhaustively for *.png (above and beyond the inventory)
SCAN_DIRS = [
    "atlas_figures",
    "atlas_targeted",
    "spectral_depth",
    "potential_comparison_plots",
    "primes/figures",
    "calogero_paper",
    "papers/calogero",
    "video/renders",
    "video",
    "results",
    "atlas_output_hires",
    "nbody",
    "3d",
    "figures",
    "atlas_animation",
    "atlas_1000",
    "aws_results",
    "data",
]
# Plus any *.png at the repo root.
SCAN_ROOT_PNGS = True

# Directories to skip even if they fall under SCAN_DIRS
SKIP_PREFIXES = (
    "legacy_figures_archive",
    "figures_v2",
    "website/data",
    "website/assets",      # canonical site assets (HF thumbnail etc.)
    "assets",              # repo-level assets folder
    "papers",              # paper PDFs - leave alone (only papers/calogero scanned)
    "aws_code_mirror",     # mirror of source workspace
    ".playwright-mcp",     # browser automation screenshots
    ".git",
)


def collect_from_inventory() -> set[str]:
    if not os.path.exists(INV_PATH):
        return set()
    with open(INV_PATH, "r", encoding="utf-8") as f:
        inv = json.load(f)
    out: set[str] = set()
    for cat in inv.get("categories", {}).values():
        for f in cat.get("files") or []:
            if isinstance(f, str) and f.lower().endswith(".png"):
                out.add(f.replace("\\", "/"))
    return out


def collect_from_scan() -> set[str]:
    out: set[str] = set()
    for sd in SCAN_DIRS:
        abs_dir = os.path.join(ROOT, sd)
        if not os.path.isdir(abs_dir):
            continue
        for dp, _, files in os.walk(abs_dir):
            for fn in files:
                if not fn.lower().endswith(".png"):
                    continue
                rel = os.path.relpath(os.path.join(dp, fn), ROOT).replace("\\", "/")
                out.add(rel)
    if SCAN_ROOT_PNGS:
        for fn in os.listdir(ROOT):
            full = os.path.join(ROOT, fn)
            if os.path.isfile(full) and fn.lower().endswith(".png"):
                out.add(fn)
    return out


def filter_paths(paths: Iterable[str]) -> list[str]:
    keep = []
    for p in paths:
        if any(p.startswith(s.rstrip("/") + "/") or p == s for s in SKIP_PREFIXES):
            continue
        # Existence check (inventory may reference deleted files)
        if not os.path.exists(os.path.join(ROOT, p)):
            continue
        keep.append(p)
    return sorted(set(keep))


def move_path(rel: str, dry: bool) -> None:
    src = os.path.join(ROOT, rel)
    dst = os.path.join(ARCHIVE, rel)
    if dry:
        print(f"  would move {rel}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def restore() -> int:
    if not os.path.isdir(ARCHIVE):
        print("Nothing to restore.")
        return 0
    n = 0
    for dp, _, files in os.walk(ARCHIVE):
        for fn in files:
            src = os.path.join(dp, fn)
            rel = os.path.relpath(src, ARCHIVE).replace("\\", "/")
            dst = os.path.join(ROOT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            n += 1
    print(f"Restored {n} files.")
    return 0


def ensure_gitignore() -> None:
    gi = os.path.join(ROOT, ".gitignore")
    needle = "legacy_figures_archive/"
    if not os.path.exists(gi):
        return
    with open(gi, "r", encoding="utf-8") as f:
        content = f.read()
    if needle in content:
        return
    with open(gi, "a", encoding="utf-8") as f:
        f.write("\n# Archived legacy PNGs (moved aside by scripts/archive_legacy_figures.py)\n")
        f.write("legacy_figures_archive/\n")
        f.write("\n# New canonical rendered figures (regenerate via website/figures_render.py)\n")
        f.write("figures_v2/\n")
    print("  added legacy_figures_archive/ and figures_v2/ to .gitignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.restore:
        return restore()

    inv_set = collect_from_inventory()
    scan_set = collect_from_scan()
    paths = filter_paths(inv_set | scan_set)
    print(f"Found {len(paths)} legacy PNGs ({len(inv_set)} from inventory, "
          f"{len(scan_set)} from filesystem scan).")

    if args.dry_run:
        for p in paths[:30]:
            print(f"  {p}")
        if len(paths) > 30:
            print(f"  ... and {len(paths) - 30} more.")
        print("Dry run; no files moved.")
        return 0

    os.makedirs(ARCHIVE, exist_ok=True)
    for p in paths:
        move_path(p, dry=False)

    # Also clean now-empty directories under the scan dirs
    for sd in SCAN_DIRS:
        abs_dir = os.path.join(ROOT, sd)
        if not os.path.isdir(abs_dir):
            continue
        for dp, _, files in os.walk(abs_dir, topdown=False):
            try:
                if not os.listdir(dp):
                    os.rmdir(dp)
            except OSError:
                pass

    ensure_gitignore()
    print(f"Moved {len(paths)} PNGs into {os.path.relpath(ARCHIVE, ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
