#!/usr/bin/env python3
"""Render the registry to docs/registry_status.md and website/data/registry.json.

Usage::

    python -m registry.build_status_table          # writes both
    python -m registry.build_status_table --md-only
    python -m registry.build_status_table --json-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from registry.loader import dump_markdown, load  # noqa: E402

MARKDOWN_OUT = REPO_ROOT / "docs" / "registry_status.md"
JSON_OUT = REPO_ROOT / "website" / "data" / "registry.json"


def write_markdown() -> None:
    MARKDOWN_OUT.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUT.write_text(dump_markdown() + "\n", encoding="utf-8")
    print(f"Wrote {MARKDOWN_OUT.relative_to(REPO_ROOT)}")


def write_json() -> None:
    entries = load()
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_from": "registry/experiments.yaml",
        "count": len(entries),
        "entries": entries,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {JSON_OUT.relative_to(REPO_ROOT)}")


def main() -> int:
    p = argparse.ArgumentParser(description="Render registry to markdown + JSON")
    p.add_argument("--md-only", action="store_true")
    p.add_argument("--json-only", action="store_true")
    args = p.parse_args()

    if args.json_only:
        write_json()
    elif args.md_only:
        write_markdown()
    else:
        write_markdown()
        write_json()
    return 0


if __name__ == "__main__":
    sys.exit(main())
