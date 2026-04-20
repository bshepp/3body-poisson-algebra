"""Experiment registry package.

A single machine-readable index of every script in the 3body project: what it
does, status, latest result, how to run, and cross-references to docs and
supplemental design memos.

Single source of truth: ``registry/experiments.yaml``.
Schema: ``registry/schema.json``.
Loader: ``registry/loader.py``.
"""

from .loader import (
    load,
    find,
    latest,
    check,
    dump_markdown,
    REGISTRY_PATH,
    SCHEMA_PATH,
)

__all__ = [
    "load",
    "find",
    "latest",
    "check",
    "dump_markdown",
    "REGISTRY_PATH",
    "SCHEMA_PATH",
]
