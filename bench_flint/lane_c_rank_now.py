"""Download the latest Lane C L=4 mod-p checkpoint and report current rank.

Each completed bracket contributes one column to the GF(p) matrix.  The
rank of the partial matrix is a *lower bound* on dim(L_k) at the
corresponding target level.  Cap: rank <= n_samples (currently 120).

Usage:
    python bench_flint/lane_c_rank_now.py
    python bench_flint/lane_c_rank_now.py --keep   # keep local pickle copy

Env overrides:
    LANE_C_BUCKET    default 3body-compute-290318
    LANE_C_PREFIX    default lane_c
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter

try:
    from flint import nmod_mat
    HAVE_FLINT = True
except ImportError:
    HAVE_FLINT = False
    import numpy as np


def s3_download(bucket: str, key: str, dest: str) -> None:
    subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", dest, "--quiet"],
        check=True,
    )


def rank_mod_p(cols: list[list[int]], prime: int, n_rows: int) -> int:
    if HAVE_FLINT:
        # nmod_mat takes a flat row-major list
        flat = [0] * (n_rows * len(cols))
        for c_idx, col in enumerate(cols):
            for r_idx, v in enumerate(col):
                flat[r_idx * len(cols) + c_idx] = int(v) % prime
        M = nmod_mat(n_rows, len(cols), flat, prime)
        return int(M.rank())
    else:
        # NumPy fallback (slow / not exact for big primes, but works for sanity)
        M = np.array(cols, dtype=np.int64).T % prime  # (n_rows, n_cols)
        # Gaussian elimination mod p
        A = M.copy()
        rows, ncols = A.shape
        r = 0
        for c in range(ncols):
            piv = None
            for k in range(r, rows):
                if A[k, c] % prime != 0:
                    piv = k
                    break
            if piv is None:
                continue
            if piv != r:
                A[[r, piv]] = A[[piv, r]]
            inv = pow(int(A[r, c]) % prime, prime - 2, prime)
            A[r] = (A[r] * inv) % prime
            for k in range(rows):
                if k != r and A[k, c] % prime != 0:
                    A[k] = (A[k] - A[k, c] * A[r]) % prime
            r += 1
            if r == rows:
                break
        return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("LANE_C_BUCKET", "3body-compute-290318"))
    ap.add_argument("--prefix", default=os.environ.get("LANE_C_PREFIX", "lane_c"))
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep local pickle copy")
    ap.add_argument("--out", default=None, help="local path for pickle (defaults to temp)")
    args = ap.parse_args()

    key = f"{args.prefix}/checkpoints/level_{args.level}_modp.pkl"
    dest = args.out or os.path.join(tempfile.gettempdir(), f"level_{args.level}_modp.pkl")

    print(f"[fetch] s3://{args.bucket}/{key} -> {dest}")
    s3_download(args.bucket, key, dest)
    sz = os.path.getsize(dest)
    print(f"[fetch] {sz} bytes")

    with open(dest, "rb") as fh:
        d = pickle.load(fh)

    prime = d["prime"]
    n_samples = d["n_samples"]
    n_lower = d["n_lower"]
    columns = d["columns"]
    processed = d["processed_pairs"]

    target_flags = [c.get("target_level", False) for c in columns]
    cnt = Counter(target_flags)
    print(f"[meta]  L={d['level']}  prime={prime}  n_samples={n_samples}  "
          f"n_lower={n_lower}")
    print(f"[meta]  columns={len(columns)}  "
          f"(lower-level={cnt[False]}, target-level={cnt[True]})")
    print(f"[meta]  processed_pairs at target level = {len(processed)}")

    # 1) Rank of just the lower-level (L < target) span
    lower_cols = [c["col"] for c in columns if not c.get("target_level", False)]
    print(f"[rank] computing rank of {len(lower_cols)} lower-level columns "
          f"({n_samples} rows)...", flush=True)
    r_lower = rank_mod_p(lower_cols, prime, n_samples)
    print(f"[rank]   dim(span of L<{args.level} generators) = {r_lower}")

    # 2) Rank of full partial matrix (lower + completed target columns)
    all_cols = [c["col"] for c in columns]
    print(f"[rank] computing rank of {len(all_cols)} total columns...", flush=True)
    r_full = rank_mod_p(all_cols, prime, n_samples)
    print(f"[rank]   dim(span including {cnt[True]} L={args.level} brackets) = {r_full}")

    new_at_level = r_full - r_lower
    print()
    print(f"[result] LOWER BOUND on dim(L_{args.level}) so far: "
          f"{new_at_level} new + {r_lower} prior = {r_full} cumulative")
    if r_full >= n_samples:
        print(f"[warn]  rank saturates n_samples={n_samples}; increase samples "
              f"to certify higher dims")

    if not args.keep and not args.out:
        try:
            os.remove(dest)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
