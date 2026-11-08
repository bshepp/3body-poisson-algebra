"""N=3 d=2 Level-4 census rank via jet evaluation — the a(4) shot.

Evaluates all level <= 4 bracket trees (the full corrected 11,937-tree
L4 census) at S random F_p^15 points and takes exact mod-p cumulative
ranks. The L<=3 prefix must reproduce [3, 6, 17, 116] (built-in
validation); the L4 number is then a certified lower bound for a(4),
equal to rank_p of the census with Schwartz-Zippel confidence, and a
candidate exact a(4) for A395423 (keyword `more`).

Usage: python run_l4.py [--samples 8192] [--batch 1024]
       [--prime 2147483647] [--seed 20260730] [--out FILE.json]
"""
import argparse
import json
import sys
from time import time

import numpy as np

from nbody_jets import NBodyJets
from modp_linalg import IncrementalBasis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--prime", type=int, default=(1 << 31) - 1)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage", default="all",
                    choices=["eval", "rank", "all"],
                    help="two-stage mode so each stage fits a tool "
                         "timeout; eval saves l4_values.npz")
    ap.add_argument("--values", default="l4_values.npz")
    ap.add_argument("--upow", type=int, default=1,
                    help="potential V = u^upow (1/r^upow)")
    args = ap.parse_args()
    S, P = args.samples, args.prime

    t0 = time()
    print(f"[l4] N=3 d=2 L<=4  V=1/r^{args.upow}  S={S} p={P} "
          f"seed={args.seed} stage={args.stage}", flush=True)
    eng = NBodyJets(3, 2, 4, P, upow=args.upow)
    eng.enumerate_trees()
    counts = [len(eng.per_level[L]) for L in range(5)]
    print(f"[l4] tree counts per level: {counts} "
          f"(L4 census must be 11937: "
          f"{'OK' if counts[4] == 11937 else 'MISMATCH'})", flush=True)

    if args.stage in ("eval", "all"):
        rng = np.random.default_rng(args.seed)
        vals = {L: [] for L in range(5)}
        for lo in range(0, S, args.batch):
            n = min(args.batch, S - lo)
            pts = rng.integers(1, P, size=(n, eng.nv), dtype=np.int64)
            vb = eng.values_batch(pts)
            for L in range(5):
                vals[L].append(vb[L].astype(np.uint32))
            print(f"[l4]   batch {lo + n}/{S} done ({time() - t0:.0f}s)",
                  flush=True)
        np.savez(args.values,
                 **{f"L{L}": np.concatenate(vals[L], axis=1)
                    for L in range(5)})
        print(f"[l4] eval stage done, values saved to {args.values} "
              f"({time() - t0:.0f}s)", flush=True)
        if args.stage == "eval":
            return 0

    z = np.load(args.values)
    basis = IncrementalBasis(P)
    cumulative = []
    for L in range(5):
        rows = z[f"L{L}"].astype(np.uint64)
        t1 = time()
        for lo in range(0, rows.shape[0], 3000):
            basis.add(rows[lo:lo + 3000])
        cumulative.append(basis.rank)
        print(f"[l4] cumulative rank through L{L} = {basis.rank}  "
              f"(+{time() - t1:.0f}s)", flush=True)

    ok_prefix = cumulative[:4] == [3, 6, 17, 116]
    print("=" * 60, flush=True)
    print(f"[l4] cumulative = {cumulative}", flush=True)
    print(f"[l4] L<=3 prefix {'VALID [3,6,17,116]' if ok_prefix else 'INVALID — DO NOT TRUST L4'}",
          flush=True)
    print(f"[l4] a(4) candidate (mod-p census rank) = {cumulative[4]}",
          flush=True)
    if cumulative[4] >= S:
        print("[l4] WARNING: saturated n_samples — rerun larger",
              flush=True)
    print(f"[l4] total {time() - t0:.0f}s", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "method": "jet-evaluation mod-p census rank "
                          "(no symbolic generation)",
                "N": 3, "d": 2,
                "potential": ("1/r" if args.upow == 1
                              else f"1/r^{args.upow}"),
                "max_level": 4,
                "prime": P, "seed": args.seed, "n_samples": S,
                "tree_counts": counts,
                "cumulative_ranks": cumulative,
                "prefix_valid": ok_prefix,
                "a4_candidate": int(cumulative[4]),
                "t_total_s": round(time() - t0, 1),
            }, fh, indent=2)
        print(f"[l4] wrote {args.out}", flush=True)
    return 0 if ok_prefix else 1


if __name__ == "__main__":
    sys.exit(main())
