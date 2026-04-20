#!/usr/bin/env python3
"""Scaffold a new entry in registry/experiments.yaml for a fresh script.

Usage::

    python scripts/registry_new.py path/to/new_script.py \\
        --category dimseq \\
        --scope nbody/N3 \\
        --status wip \\
        --description "Short summary"

If --description is omitted, the script's module docstring is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from registry.loader import REGISTRY_PATH, load  # noqa: E402

VALID_CATEGORIES = [
    "dimseq", "atlas", "structure", "quantum", "spectral", "infra",
    "analysis", "viz", "dataset", "test", "diagnostic", "engine",
    "website", "docs",
]

VALID_STATUSES = [
    "planned", "wip", "complete", "broken", "superseded", "archived", "needs_review",
]


def _make_id(rel_posix: str) -> str:
    import re
    stem = rel_posix.replace("/", "_").removesuffix(".py")
    stem = re.sub(r"[^a-z0-9_]", "_", stem.lower())
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def _extract_docstring(path: Path) -> str:
    import ast
    try:
        src = path.read_text(encoding="utf-8")
        mod = ast.parse(src)
        doc = ast.get_docstring(mod) or ""
    except (OSError, UnicodeDecodeError, SyntaxError):
        return ""
    para = doc.strip().split("\n\n", 1)[0]
    return " ".join(line.strip() for line in para.splitlines() if line.strip())


def main() -> int:
    p = argparse.ArgumentParser(description="Scaffold a new registry entry")
    p.add_argument("path", help="Repo-relative path to the script (e.g. nbody/new_thing.py)")
    p.add_argument("--id", help="Override auto-derived id")
    p.add_argument("--category", required=True, choices=VALID_CATEGORIES)
    p.add_argument("--scope", default=None, help="e.g. nbody/N3, primes, neural")
    p.add_argument("--status", default="wip", choices=VALID_STATUSES)
    p.add_argument("--description", default=None, help="Override docstring summary")
    p.add_argument("--tag", action="append", default=[], help="Tag (repeatable)")
    p.add_argument("--write", action="store_true", help="Write to registry/experiments.yaml")
    args = p.parse_args()

    rel = Path(args.path).as_posix()
    full = REPO_ROOT / rel
    if not full.exists():
        print(f"WARNING: {rel} does not exist on disk yet.", file=sys.stderr)

    eid = args.id or _make_id(rel)
    description = args.description or _extract_docstring(full) or f"(no docstring) {Path(rel).name}"

    entry = {
        "id": eid,
        "path": rel,
        "category": args.category,
        "scope": args.scope,
        "status": args.status,
        "description": description,
        "outputs": [],
        "tags": args.tag,
    }

    print("Proposed entry:")
    print(yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))

    entries = load()
    if any(e.get("id") == eid for e in entries):
        print(f"ERROR: id {eid!r} already exists in registry.", file=sys.stderr)
        return 1
    if any(e.get("path") == rel for e in entries):
        print(f"ERROR: path {rel!r} already in registry.", file=sys.stderr)
        return 1

    if not args.write:
        print("Dry run; pass --write to commit.")
        return 0

    entries.append(entry)
    REGISTRY_PATH.write_text(
        _file_header() + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Appended {eid} to {REGISTRY_PATH.relative_to(REPO_ROOT)}")
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
