"""Registry loader, validator, and lint check.

Usage::

    from registry import load, find, check

    entries = load()
    yukawa = find(category="dimseq", tags=["yukawa"])
    issues = check()  # list[str]; empty means clean

CLI::

    python -m registry.loader --check        # exit 1 on issues
    python -m registry.loader --list         # print id\tstatus\tpath
    python -m registry.loader --markdown     # print markdown table

Tracked roots: repository top-level + nbody/, primes/, neural/, 3d/, dataset/,
website/build_*.py, infra/launch_*.py. Excluded paths are listed in
``EXCLUDED_DIRS`` and ``EXCLUDED_FILES`` below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "registry" / "experiments.yaml"
SCHEMA_PATH = REPO_ROOT / "registry" / "schema.json"

TRACKED_ROOTS: tuple[str, ...] = (
    ".",
    "nbody",
    "primes",
    "neural",
    "3d",
    "dataset",
)

TRACKED_GLOBS: tuple[tuple[str, str], ...] = (
    ("website", "build_*.py"),
    ("website", "figures_*.py"),
    ("website", "preprocess_*.py"),
    ("website", "render_*.py"),
    ("infra", "launch_*.py"),
    ("scripts", "*.py"),
)

EXCLUDED_DIRS: frozenset[str] = frozenset({
    "__pycache__",
    "tmp",
    "aws_code_mirror",
    "aws_results",
    "checkpoints",
    "aws_checkpoints",
    "legacy_figures_archive",
    ".github",
    ".playwright-mcp",
    ".claude",
    ".venv",
    "venv",
    "node_modules",
})

EXCLUDED_FILES: frozenset[str] = frozenset({
    "setup.py",
    "conftest.py",
})


# --------------------------------------------------------------------------- #
# YAML loading (use PyYAML if available, else a tiny built-in fallback)
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load the registry. "
            "Install with: pip install pyyaml"
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the registry as a list of dict entries."""
    p = Path(path) if path else REGISTRY_PATH
    if not p.exists():
        return []
    data = _load_yaml(p)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Registry at {p} must be a YAML list")
    return data


def find(
    *,
    category: str | None = None,
    status: str | None = None,
    scope: str | None = None,
    tag: str | None = None,
    tags: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter registry entries by any combination of fields."""
    entries = load()
    if category is not None:
        entries = [e for e in entries if e.get("category") == category]
    if status is not None:
        entries = [e for e in entries if e.get("status") == status]
    if scope is not None:
        entries = [e for e in entries if e.get("scope") == scope]
    if tag is not None:
        entries = [e for e in entries if tag in (e.get("tags") or [])]
    if tags is not None:
        wanted = set(tags)
        entries = [e for e in entries if wanted.issubset(set(e.get("tags") or []))]
    return entries


def latest(category: str | None = None) -> dict[str, Any] | None:
    """Return the entry with the most recent ``latest_result.date`` field."""
    entries = find(category=category) if category else load()
    dated = [e for e in entries if (e.get("latest_result") or {}).get("date")]
    if not dated:
        return None
    return max(dated, key=lambda e: e["latest_result"]["date"])


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

def _validate_schema(entries: list[dict[str, Any]]) -> list[str]:
    """Validate entries against schema.json. Returns list of issue strings."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return [
            "jsonschema not installed; skipping schema validation. "
            "Install with: pip install jsonschema"
        ]
    if not SCHEMA_PATH.exists():
        return [f"Schema file missing: {SCHEMA_PATH}"]
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    issues: list[str] = []
    validator = jsonschema.Draft7Validator(schema)
    for err in validator.iter_errors(entries):
        loc = "/".join(str(p) for p in err.absolute_path)
        issues.append(f"schema: {loc}: {err.message}")
    return issues


# --------------------------------------------------------------------------- #
# Filesystem walk
# --------------------------------------------------------------------------- #

def _is_excluded(rel: Path) -> bool:
    parts = rel.parts
    for part in parts:
        if part in EXCLUDED_DIRS:
            return True
    if rel.name in EXCLUDED_FILES:
        return True
    if rel.name.startswith("_"):
        return True
    return False


def discover_scripts() -> list[Path]:
    """Walk tracked roots and return repo-relative paths to .py scripts."""
    found: set[Path] = set()
    for root in TRACKED_ROOTS:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        if root == ".":
            for entry in root_path.iterdir():
                if entry.is_file() and entry.suffix == ".py":
                    rel = entry.relative_to(REPO_ROOT)
                    if not _is_excluded(rel):
                        found.add(rel)
        else:
            for entry in root_path.rglob("*.py"):
                rel = entry.relative_to(REPO_ROOT)
                if not _is_excluded(rel):
                    found.add(rel)
    for parent, pattern in TRACKED_GLOBS:
        parent_path = REPO_ROOT / parent
        if not parent_path.exists():
            continue
        for match in parent_path.glob(pattern):
            if match.is_file():
                rel = match.relative_to(REPO_ROOT)
                if not _is_excluded(rel):
                    found.add(rel)
    return sorted(found)


# --------------------------------------------------------------------------- #
# Lint check
# --------------------------------------------------------------------------- #

def check(strict_outputs: bool = False) -> list[str]:
    """Run all consistency checks. Returns list of issue strings (empty = clean).

    Checks performed:
      1. YAML loads and is a list.
      2. JSON Schema validation (if jsonschema available).
      3. Every registry entry has a unique id.
      4. Every registry ``path`` exists on disk.
      5. Every discovered .py script in tracked roots is registered (or excluded).
      6. ``superseded_by`` values point to existing ids.
      7. If ``strict_outputs``, every ``status: complete`` entry has at least
         one existing ``outputs`` path.
    """
    issues: list[str] = []
    try:
        entries = load()
    except Exception as exc:  # noqa: BLE001
        return [f"failed to load registry: {exc}"]

    issues.extend(_validate_schema(entries))

    seen_ids: dict[str, int] = {}
    for i, e in enumerate(entries):
        eid = e.get("id")
        if not eid:
            issues.append(f"entry #{i}: missing id")
            continue
        if eid in seen_ids:
            issues.append(f"duplicate id: {eid} (entries #{seen_ids[eid]} and #{i})")
        else:
            seen_ids[eid] = i

    for e in entries:
        path = e.get("path")
        if not path:
            issues.append(f"{e.get('id', '?')}: missing path")
            continue
        if e.get("status") == "archived":
            continue
        if e.get("status") == "planned":
            continue
        full = REPO_ROOT / path
        if not full.exists():
            issues.append(f"{e['id']}: path does not exist: {path}")

    for e in entries:
        sup = e.get("superseded_by")
        if sup and sup not in seen_ids:
            issues.append(f"{e['id']}: superseded_by={sup!r} not in registry")

    registered_paths = {
        Path(e["path"]).as_posix()
        for e in entries
        if e.get("path") and e.get("status") not in {"planned"}
    }
    discovered = discover_scripts()
    for rel in discovered:
        if rel.as_posix() not in registered_paths:
            issues.append(f"unregistered script: {rel.as_posix()}")

    if strict_outputs:
        for e in entries:
            if e.get("status") != "complete":
                continue
            outs = e.get("outputs") or []
            if not outs:
                issues.append(f"{e['id']}: status=complete but no outputs listed")
                continue
            existing = [o for o in outs if (REPO_ROOT / o).exists()]
            if not existing:
                issues.append(
                    f"{e['id']}: status=complete but none of the listed "
                    f"outputs exist: {outs}"
                )

    return issues


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

_STATUS_ORDER = ["wip", "complete", "planned", "broken", "needs_review", "superseded", "archived"]


def dump_markdown() -> str:
    """Render the registry as a markdown document grouped by category."""
    entries = load()
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_cat.setdefault(e.get("category", "uncategorized"), []).append(e)

    lines: list[str] = []
    lines.append("# Experiment Registry")
    lines.append("")
    lines.append(f"*Auto-generated from `registry/experiments.yaml` — {len(entries)} entries.*")
    lines.append("")

    counts: dict[str, int] = {}
    for e in entries:
        counts[e.get("status", "?")] = counts.get(e.get("status", "?"), 0) + 1
    summary = " · ".join(
        f"{s}: {counts[s]}" for s in _STATUS_ORDER if s in counts
    )
    if summary:
        lines.append(f"**Status:** {summary}")
        lines.append("")

    for cat in sorted(by_cat):
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| ID | Status | Path | Description |")
        lines.append("|----|--------|------|-------------|")
        for e in sorted(by_cat[cat], key=lambda x: x.get("id", "")):
            desc = (e.get("description") or "").replace("\n", " ").strip()
            if len(desc) > 120:
                desc = desc[:117] + "..."
            lines.append(
                f"| `{e.get('id', '?')}` "
                f"| {e.get('status', '?')} "
                f"| `{e.get('path', '?')}` "
                f"| {desc} |"
            )
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _main() -> int:
    p = argparse.ArgumentParser(description="Experiment registry tools")
    p.add_argument("--check", action="store_true", help="Run consistency checks; exit 1 on issues")
    p.add_argument("--strict-outputs", action="store_true", help="With --check, also verify outputs exist")
    p.add_argument("--list", action="store_true", help="Print one line per entry")
    p.add_argument("--markdown", action="store_true", help="Print markdown table")
    p.add_argument("--discover", action="store_true", help="List discovered scripts (debug)")
    args = p.parse_args()

    did_anything = False

    if args.discover:
        for rel in discover_scripts():
            print(rel.as_posix())
        did_anything = True

    if args.list:
        for e in load():
            print(f"{e.get('id', '?'):40s}  {e.get('status', '?'):12s}  {e.get('path', '?')}")
        did_anything = True

    if args.markdown:
        print(dump_markdown())
        did_anything = True

    if args.check or not did_anything:
        issues = check(strict_outputs=args.strict_outputs)
        if issues:
            print(f"Registry check FAILED with {len(issues)} issue(s):", file=sys.stderr)
            for i in issues:
                print(f"  - {i}", file=sys.stderr)
            return 1
        print(f"Registry check OK ({len(load())} entries)")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(_main())
