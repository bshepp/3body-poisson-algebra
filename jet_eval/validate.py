"""Validation harness for the jet-evaluation engine.

Runs the known-value gauntlet — every rank the project has already
established by independent means:

  N=3 d=2, L<=3 : cumulative ranks must be [3, 6, 17, 116]
  N=4 d=1, L<=3 : final cumulative rank must be 1260
  N=5 d=1, L<=2 : cumulative ranks must be [10, 25, 145]

Usage: python validate.py [--case n3|n4|n5|all]
"""
import argparse
import sys
from time import time

import numpy as np

from nbody_jets import NBodyJets
from modp_linalg import IncrementalBasis

PRIME = (1 << 31) - 1


def run_case(name, N, d, max_level, expected, S, seed=20260730,
             batch=None):
    t0 = time()
    print(f"[{name}] N={N} d={d} L<={max_level} S={S} p={PRIME}",
          flush=True)
    eng = NBodyJets(N, d, max_level, PRIME)
    eng.enumerate_trees()
    counts = [len(eng.per_level[L]) for L in range(max_level + 1)]
    print(f"[{name}] tree counts per level: {counts}", flush=True)

    rng = np.random.default_rng(seed)
    batch = batch or S
    vals = {L: [] for L in range(max_level + 1)}
    for lo in range(0, S, batch):
        n = min(batch, S - lo)
        pts = rng.integers(1, PRIME, size=(n, eng.nv), dtype=np.int64)
        vb = eng.values_batch(pts)
        for L in range(max_level + 1):
            vals[L].append(vb[L])
        print(f"[{name}]   batch {lo + n}/{S} done "
              f"({time() - t0:.0f}s)", flush=True)

    basis = IncrementalBasis(PRIME)
    cumulative = []
    for L in range(max_level + 1):
        rows = np.concatenate(vals[L], axis=1)
        basis.add(rows)
        cumulative.append(basis.rank)
    ok = cumulative == expected
    print(f"[{name}] cumulative ranks = {cumulative}  expected {expected}"
          f"  -> {'PASS' if ok else 'FAIL'}   [{time() - t0:.0f}s]",
          flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all",
                    choices=["n3", "n4", "n5", "all"])
    args = ap.parse_args()
    results = []
    if args.case in ("n3", "all"):
        results.append(run_case("n3", 3, 2, 3, [3, 6, 17, 116], S=256))
    if args.case in ("n5", "all"):
        results.append(run_case("n5", 5, 1, 2, [10, 25, 145], S=256))
    if args.case in ("n4", "all"):
        results.append(run_case("n4", 4, 1, 3, [6, 14, 62, 1260],
                                S=2048, batch=512))
    print("=" * 50)
    print("ALL PASS" if all(results) else "SOME FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
