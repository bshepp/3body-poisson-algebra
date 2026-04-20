# Experiment Registry

Single source of truth for every script in the project: what it does, status,
latest result, how to run, related docs.

## Files

| File | Role |
|------|------|
| `experiments.yaml` | The registry. Hand-edited; CI-checked. |
| `schema.json` | JSON Schema for entries. |
| `loader.py` | Python loader + validator + lint check + CLI. |
| `build_status_table.py` | Renders `docs/registry_status.md` and `website/data/registry.json`. |
| `__init__.py` | Re-exports `load`, `find`, `latest`, `check`, `dump_markdown`. |

Companion CLIs in `scripts/`:

| Script | Purpose |
|--------|---------|
| `scripts/registry_bootstrap.py` | One-shot: walk tracked roots and seed `needs_review` entries for any unregistered `.py`. Idempotent. |
| `scripts/registry_curate.py` | Apply hand-written canonical patches keyed by entry id. |
| `scripts/registry_new.py` | Scaffold a single new entry from the CLI when adding a script. |

## Quick start

```bash
pip install pyyaml jsonschema

# Lint check (run in CI; non-zero exit on any issue)
python -m registry.loader --check

# Add a new script entry interactively
python scripts/registry_new.py nbody/new_thing.py --category dimseq --scope nbody/N3 --write

# Sweep up any newly-added scripts that weren't registered manually
python scripts/registry_bootstrap.py --write

# Render markdown + JSON for the website / docs
python -m registry.build_status_table
```

## Programmatic use

```python
from registry import find, latest

complete_dimseq = find(category="dimseq", status="complete")
last = latest("dimseq")
print(last["id"], last["latest_result"]["sequence"])
```

## What gets tracked

Tracked roots (see `loader.py`):

- repo top-level `*.py`
- `nbody/`, `primes/`, `neural/`, `3d/`, `dataset/` (recursive)
- `website/build_*.py`, `website/figures_*.py`, `website/preprocess_*.py`, `website/render_*.py`
- `infra/launch_*.py`
- `scripts/*.py`

Excluded: `__pycache__`, `tmp/`, `aws_code_mirror/`, `aws_results/`, `checkpoints/`,
`legacy_figures_archive/`, `.github/`, anything starting with `_`.

## Status values

- `planned` — design exists; no code yet. `path` need not exist on disk.
- `wip` — actively in progress; results may be partial.
- `complete` — works, has been run, result is canonical.
- `broken` — needs repair; describe the failure mode in `description`.
- `superseded` — set `superseded_by: <id-of-replacement>`.
- `archived` — kept for history; not maintained.
- `needs_review` — bootstrapped automatically; status not yet curated.

## Pre-commit hook (optional, recommended)

Add to `.git/hooks/pre-commit`:

```bash
#!/usr/bin/env bash
python -m registry.loader --check || {
    echo "Registry lint failed. Edit registry/experiments.yaml to fix."
    exit 1
}
```

Then `chmod +x .git/hooks/pre-commit`.

## CI

`.github/workflows/regression.yml` includes a `registry-lint` job that runs
`python -m registry.loader --check` on every push and pull request. Build fails
if any tracked `.py` is missing from the registry, any registry `path` does not
exist on disk (for non-`planned`/`archived` entries), any `superseded_by` points
to an unknown id, or schema validation fails.
