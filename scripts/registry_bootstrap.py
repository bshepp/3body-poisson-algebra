#!/usr/bin/env python3
"""Bootstrap registry/experiments.yaml from discovered scripts.

Walks the tracked roots (via ``registry.loader.discover_scripts``), extracts a
docstring summary from each .py file, infers a category from path heuristics,
and writes entries with ``status: needs_review``. Any IDs already present in
the registry are preserved untouched (re-runs are idempotent).

Usage::

    python scripts/registry_bootstrap.py            # dry-run; print plan
    python scripts/registry_bootstrap.py --write    # write to registry/experiments.yaml
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from registry.loader import discover_scripts, load, REGISTRY_PATH  # noqa: E402

CATEGORY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"infra/.*launch_"), "infra"),
    (re.compile(r"^test_|/test_|^validate_|/validate_"), "test"),
    (re.compile(r"^diagnostic_|/diagnose_"), "diagnostic"),
    (re.compile(r"^viz_|^render_|^animate_|^plot_|/figures_|/preprocess_"), "viz"),
    (re.compile(r"website/"), "website"),
    (re.compile(r"dataset/"), "dataset"),
    (re.compile(r"quantum|moyal"), "quantum"),
    (re.compile(r"spectral|level2_spec|finite_n_gue"), "spectral"),
    (re.compile(r"isomorphism|structure_|killing|derived_series"), "structure"),
    (re.compile(r"atlas|targeted_adaptive|stability|landscape|sphere"), "atlas"),
    (re.compile(r"sweep|exponent|dimseq|symbolic_rank|expansion_dimseq|n_scaling|yukawa"), "dimseq"),
    (re.compile(r"exact_growth|nbody/exact|3d/exact|nn_poisson|nn_algebra|quantum_algebra"), "engine"),
    (re.compile(r"analyze_|compare_|117|expansion_analysis|summarize_"), "analysis"),
]

DEFAULT_CATEGORY = "analysis"

SCOPE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^nbody/"), "nbody"),
    (re.compile(r"^primes/"), "primes"),
    (re.compile(r"^neural/"), "neural"),
    (re.compile(r"^3d/"), "3body/3d"),
    (re.compile(r"^dataset/"), "dataset"),
    (re.compile(r"^website/"), "website"),
    (re.compile(r"^infra/"), "infra"),
    (re.compile(r"^scripts/"), "tooling"),
]


def infer_category(rel_posix: str) -> str:
    for pat, cat in CATEGORY_RULES:
        if pat.search(rel_posix):
            return cat
    return DEFAULT_CATEGORY


def infer_scope(rel_posix: str) -> str:
    for pat, scope in SCOPE_RULES:
        if pat.search(rel_posix):
            return scope
    return "3body"


def make_id(rel_posix: str) -> str:
    """Canonicalize 'nbody/run_helium.py' -> 'nbody_run_helium'."""
    stem = rel_posix.replace("/", "_").removesuffix(".py")
    stem = re.sub(r"[^a-z0-9_]", "_", stem.lower())
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def extract_docstring(path: Path) -> str:
    try:
        src = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return ""
    doc = ast.get_docstring(mod)
    if not doc:
        return ""
    first_para = doc.strip().split("\n\n", 1)[0]
    summary = " ".join(line.strip() for line in first_para.splitlines() if line.strip())
    if len(summary) > 400:
        summary = summary[:397] + "..."
    return summary


def build_entry(rel: Path) -> dict:
    rel_posix = rel.as_posix()
    eid = make_id(rel_posix)
    description = extract_docstring(REPO_ROOT / rel) or f"(no docstring) {rel.name}"
    return {
        "id": eid,
        "path": rel_posix,
        "category": infer_category(rel_posix),
        "scope": infer_scope(rel_posix),
        "status": "needs_review",
        "description": description,
        "outputs": [],
        "tags": [],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap experiment registry from discovered scripts")
    p.add_argument("--write", action="store_true", help="Write to registry/experiments.yaml (else dry run)")
    args = p.parse_args()

    existing = load()
    existing_paths = {e.get("path") for e in existing if e.get("path")}
    existing_ids = {e.get("id") for e in existing if e.get("id")}

    discovered = discover_scripts()
    new_entries: list[dict] = []
    for rel in discovered:
        rel_posix = rel.as_posix()
        if rel_posix in existing_paths:
            continue
        entry = build_entry(rel)
        if entry["id"] in existing_ids:
            entry["id"] = f"{entry['id']}_auto"
        new_entries.append(entry)
        existing_ids.add(entry["id"])

    print(f"Discovered: {len(discovered)} scripts")
    print(f"Already registered: {len(existing_paths)}")
    print(f"New entries to add: {len(new_entries)}")

    if not new_entries:
        print("Nothing to do.")
        return 0

    merged = list(existing) + new_entries
    merged.sort(key=lambda e: (e.get("category", ""), e.get("id", "")))

    if not args.write:
        print("\n--- Sample of new entries (first 5) ---")
        print(yaml.safe_dump(new_entries[:5], sort_keys=False, allow_unicode=True))
        print("Run with --write to commit.")
        return 0

    REGISTRY_PATH.write_text(
        _file_header() + yaml.safe_dump(merged, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(merged)} entries to {REGISTRY_PATH.relative_to(REPO_ROOT)}")
    return 0


def _file_header() -> str:
    return (
        "# Experiment registry - single source of truth.\n"
        "#\n"
        "# Auto-bootstrapped entries have status: needs_review. Curate them to\n"
        "# canonical state by editing this file directly. New scripts should be\n"
        "# added with `python scripts/registry_new.py <path> --category <c>`.\n"
        "#\n"
        "# Status values: planned | wip | complete | broken | superseded | archived | needs_review\n"
        "# Categories: dimseq | atlas | structure | quantum | spectral | infra |\n"
        "#             analysis | viz | dataset | test | diagnostic | engine |\n"
        "#             website | docs\n"
        "\n"
    )


if __name__ == "__main__":
    sys.exit(main())
