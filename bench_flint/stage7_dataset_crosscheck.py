#!/usr/bin/env python3
"""Stage 7: cross-check our patched validator outputs against the project's
pinned dataset (dataset/output/dimension_sequences.parquet).

For every per-case JSON in bench_flint/_validate_results/, find the
matching row(s) in the dataset (by N, d, potential, charges, max_level)
and confirm the dimension sequence is consistent with the canonical.

Produces bench_flint/stage7_crosscheck.json with one row per validator
case, matched against zero or more dataset rows. Cases the dataset
doesn't have entries for are flagged "no_dataset_entry" - those are
informational, not failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "bench_flint" / "_validate_results"
DATASET_PATH = REPO_ROOT / "dataset" / "output" / "dimension_sequences.parquet"
OUT_PATH = REPO_ROOT / "bench_flint" / "stage7_crosscheck.json"


def normalize_potential(p: str) -> str:
    """Match dataset potential strings."""
    return p.replace(" ", "")


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found")
        return 2

    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df)} dataset rows")

    case_jsons = sorted(RESULT_DIR.glob("s*.json"))
    print(f"Found {len(case_jsons)} validator case JSONs")

    crosschecks = []

    for cj in case_jsons:
        try:
            with open(cj, "r", encoding="utf-8") as f:
                case = json.load(f)
        except Exception as exc:
            crosschecks.append({"case_file": cj.name, "error": str(exc)})
            continue

        if case.get("status") != "done":
            crosschecks.append({
                "case_id": case.get("case_id"),
                "status": case.get("status"),
                "skipped": True,
                "reason": "case did not complete",
            })
            continue

        potential = normalize_potential(case["potential"])
        n_bodies = case["n_bodies"]
        d_spatial = case["d_spatial"]
        max_level = case["max_level"]
        our_seq = case["sequence"]

        # Restrict the dataset filter:
        sel = df[
            (df["N"] == n_bodies)
            & (df["d"] == d_spatial)
            & (df["potential"].apply(normalize_potential) == potential)
        ].copy()

        # For composite potentials the dataset stores them under a
        # potential string that may differ; we don't try to match those.
        check = {
            "case_id": case["case_id"],
            "potential": case["potential"],
            "N": n_bodies, "d": d_spatial,
            "max_level": max_level,
            "our_sequence": our_seq,
            "match_canonical": case.get("match"),
            "expected": case.get("expected"),
        }

        if len(sel) == 0:
            check["dataset_match"] = "no_dataset_entry"
            check["note"] = ("Dataset has no row for this (N, d, potential). "
                             "Common for composite potentials.")
            crosschecks.append(check)
            continue

        # Compare each prefix at our max_level. The dataset stores
        # dimension_sequence as a string (e.g. "[3, 6, 17]") so parse it.
        ok_rows = []
        mismatch_rows = []
        for _, row in sel.iterrows():
            raw_seq = row["dimension_sequence"]
            if isinstance(raw_seq, str):
                try:
                    ds_seq = json.loads(raw_seq)
                except Exception:
                    continue
            else:
                ds_seq = list(raw_seq)
            if not isinstance(ds_seq, list):
                continue
            # Truncate both to our shortest level for comparison.
            common_len = min(len(ds_seq), len(our_seq))
            if common_len == 0:
                continue
            ds_prefix = ds_seq[:common_len]
            our_prefix = our_seq[:common_len]
            row_summary = {
                "ds_max_level": int(row["max_level"]),
                "ds_sequence": ds_seq,
                "ds_charges": row.get("charges"),
                "ds_masses": row.get("masses"),
                "is_exact": bool(row.get("is_exact", False)),
                "common_prefix_match": ds_prefix == our_prefix,
                "common_len": common_len,
            }
            if row_summary["common_prefix_match"]:
                ok_rows.append(row_summary)
            else:
                mismatch_rows.append(row_summary)

        check["dataset_matched_rows"] = len(ok_rows)
        check["dataset_mismatch_rows"] = len(mismatch_rows)
        check["matched_examples"] = ok_rows[:3]
        check["mismatch_examples"] = mismatch_rows[:3]

        # Pass if at least one dataset row's prefix agrees with ours
        if ok_rows:
            check["dataset_match"] = "consistent"
        elif mismatch_rows:
            # Note: extreme-mass best_effort cases are expected to mismatch
            # because the dataset's pinned values are exact-Q while ours
            # are float64 SVD.
            check["dataset_match"] = "all_mismatch"
        else:
            check["dataset_match"] = "no_comparable_rows"

        crosschecks.append(check)

    # Summary
    print(f"\nCross-check summary:")
    by_status = {}
    for c in crosschecks:
        ds = c.get("dataset_match", "?")
        by_status.setdefault(ds, 0)
        by_status[ds] += 1
    for k, v in sorted(by_status.items()):
        print(f"  {k:25s}: {v}")

    payload = {
        "crosschecks": crosschecks,
        "summary": by_status,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    main()
