#!/usr/bin/env python3
"""
Audit all local atlas data against S3 to detect stale/incomplete syncs.

Checks:
  1. Local npy files with suspicious zero-fill patterns
  2. Checkpoint vs summary consistency
  3. S3 comparison: re-downloads each npy and checks for differences

Usage:
    python audit_atlas_data.py              # audit + report only
    python audit_atlas_data.py --fix        # also replace stale local files
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import numpy as np

DATA_DIR = os.path.join("aws_results", "atlas_full")


def check_local(name, path):
    """Check a single atlas config for internal consistency."""
    issues = []

    gap_path = os.path.join(path, "gap_map.npy")
    rank_path = os.path.join(path, "rank_map.npy")
    cp_path = os.path.join(path, "checkpoint.json")
    sum_path = os.path.join(path, "summary.json")

    if not os.path.exists(gap_path) or not os.path.exists(rank_path):
        issues.append("MISSING: gap_map.npy or rank_map.npy")
        return issues

    gap = np.load(gap_path)
    rank = np.load(rank_path)

    n_rows, n_cols = gap.shape
    zero_gap_rows = []
    neg1_rank_rows = []

    for i in range(n_rows):
        if np.all(gap[i] == 0):
            zero_gap_rows.append(i)
        if np.all(rank[i] == -1):
            neg1_rank_rows.append(i)

    if zero_gap_rows:
        n_zero = len(zero_gap_rows)
        first = zero_gap_rows[0]
        issues.append(
            f"ZERO-FILL: {n_zero}/{n_rows} rows are all-zero in gap_map "
            f"(first empty row: {first})"
        )

    if neg1_rank_rows:
        n_neg = len(neg1_rank_rows)
        first = neg1_rank_rows[0]
        issues.append(
            f"UNCOMPUTED: {n_neg}/{n_rows} rows are all -1 in rank_map "
            f"(first: {first})"
        )

    cp = None
    if os.path.exists(cp_path):
        with open(cp_path) as f:
            cp = json.load(f)

    summary = None
    if os.path.exists(sum_path):
        with open(sum_path) as f:
            summary = json.load(f)

    if cp and summary:
        cp_row = cp.get("last_completed_row", -1)
        total = cp.get("total_rows", n_rows)
        if cp_row < total - 1 and summary.get("valid_points", 0) == n_rows * n_cols:
            issues.append(
                f"MISMATCH: checkpoint says row {cp_row}/{total-1} "
                f"but summary claims {summary['valid_points']} valid points"
            )

    if summary:
        claimed_r116 = summary.get("rank_116_count", 0)
        actual_r116 = int(np.sum(rank == 116))
        if claimed_r116 != actual_r116:
            issues.append(
                f"RANK MISMATCH: summary claims {claimed_r116} rank-116 "
                f"but npy has {actual_r116}"
            )

    return issues


def compare_s3(name, local_path):
    """Download npy files from S3 and compare to local copies."""
    diffs = {}
    s3_prefix = f"s3://3body-compute-290318/atlas_full/{name}/"

    for fname in ["gap_map.npy", "rank_map.npy"]:
        local_file = os.path.join(local_path, fname)
        if not os.path.exists(local_file):
            continue

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_prefix + fname, tmp_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                diffs[fname] = "S3_DOWNLOAD_FAILED"
                continue

            local_arr = np.load(local_file)
            s3_arr = np.load(tmp_path)

            if local_arr.shape != s3_arr.shape:
                diffs[fname] = f"SHAPE: local={local_arr.shape} s3={s3_arr.shape}"
            elif not np.array_equal(local_arr, s3_arr):
                n_diff = int(np.sum(local_arr != s3_arr))
                local_zeros = int(np.sum(local_arr == 0))
                s3_zeros = int(np.sum(s3_arr == 0))
                diffs[fname] = (
                    f"DATA DIFFERS: {n_diff} cells  "
                    f"(local zeros={local_zeros}, s3 zeros={s3_zeros})"
                )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return diffs


def fix_from_s3(name, local_path):
    """Replace local npy files with S3 versions."""
    s3_prefix = f"s3://3body-compute-290318/atlas_full/{name}/"
    fixed = []
    for fname in ["gap_map.npy", "rank_map.npy", "checkpoint.json", "summary.json"]:
        local_file = os.path.join(local_path, fname)
        result = subprocess.run(
            ["aws", "s3", "cp", s3_prefix + fname, local_file],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            fixed.append(fname)
    return fixed


def main():
    parser = argparse.ArgumentParser(description="Audit atlas data integrity")
    parser.add_argument("--fix", action="store_true",
                        help="Replace stale local files from S3")
    parser.add_argument("--skip-s3", action="store_true",
                        help="Skip S3 comparison (local checks only)")
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    configs = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))

    print(f"Auditing {len(configs)} atlas configs in {DATA_DIR}/\n")

    problems = {}
    clean = []

    for name in configs:
        path = os.path.join(DATA_DIR, name)
        local_issues = check_local(name, path)

        s3_diffs = {}
        if not args.skip_s3:
            s3_diffs = compare_s3(name, path)

        all_issues = local_issues + [
            f"S3 STALE ({k}): {v}" for k, v in s3_diffs.items()
            if v != "S3_DOWNLOAD_FAILED"
        ]
        s3_missing = [k for k, v in s3_diffs.items() if v == "S3_DOWNLOAD_FAILED"]

        if all_issues:
            problems[name] = all_issues
            print(f"  PROBLEM  {name}")
            for issue in all_issues:
                print(f"           -> {issue}")
            if s3_missing:
                print(f"           (S3 download failed for: {', '.join(s3_missing)})")

            if args.fix and any("STALE" in i or "ZERO-FILL" in i or "MISMATCH" in i
                                for i in all_issues):
                fixed = fix_from_s3(name, path)
                print(f"           FIXED: re-downloaded {', '.join(fixed)}")
        else:
            clean.append(name)
            status = "ok"
            if s3_missing:
                status += f" (no S3 copy: {', '.join(s3_missing)})"
            print(f"       OK  {name}  {status}")

    print(f"\n{'='*60}")
    print(f"  TOTAL: {len(configs)}  |  CLEAN: {len(clean)}  |  PROBLEMS: {len(problems)}")
    if problems:
        print(f"\n  Configs needing attention:")
        for name in problems:
            print(f"    - {name}")
        if not args.fix:
            print(f"\n  Run with --fix to auto-repair from S3")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
