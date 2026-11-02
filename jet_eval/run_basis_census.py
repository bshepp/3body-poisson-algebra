"""Basis-reduced jet-evaluation census ranks mod p (N-body, any level).

Same mathematics as run_l4.py / NBodyJets: bracket trees evaluated as
truncated Taylor jets at random F_p points, exact mod-p cumulative
ranks, Schwartz-Zippel confidence. The difference is enumeration. At
every level k < M only a maximal independent subset of the level-k
brackets is carried forward (chosen at S_sel selection points). Since
    L_k = L_{k-1} + {L_{k-1}, L_{k-1}}
and the bracket is bilinear, brackets of a spanning set of L_{k-1}
against a spanning set of L_{<=k-1} span {L_{k-1}, L_{k-1}} exactly, so
the cumulative ranks equal those of the full tree census while the
top-level candidate count drops from the full census (25.6M trees for
N=6 L3) to ~ new_{M-1} x dim L_{M-1} (~38K).

Top-level values (order-0 bracket of order-1 jets) are computed for all
frontier x selected pairs at once as a gradient bilinear form
    {f, g} = sum_{b,k} (Df/dq_bk Dg/dp_bk - Df/dp_bk Dg/dq_bk),
with D/dq the total derivative through u_ij = 1/r_ij (chain rule as in
NBodyJets.bracket).

Epistemics are unchanged from run_l4.py: the final rank is a certified
lower bound on the mod-p (hence rational) dimension, equal to it with
probability ~ 1 - rank*deg/p per run, PROVIDED S_sel and S both exceed
the ranks they resolve (checked and reported).

Memory. Top-level values are written straight into a .npy memmap
(--values), so a run can be split into --stage eval and --stage rank
and the rank stage re-run without re-evaluating. The rank stage uses
LeanIncrementalBasis: uint32 basis storage and column-chunked limb
GEMMs, peak ~3x the basis instead of ~8x (the dense kernel ran this
machine out of RAM on the N=6 L3 top level, 38K x 24.6K).

Usage:
  python run_basis_census.py --N 6 --d 1 --L 3 --samples 24576 \
      [--sel-samples 4096] [--batch 512] [--prime P] [--seed SEED] \
      [--upow 1] [--stage all|eval|rank] [--values FILE.npy] \
      [--check-kernels] [--out FILE.json]
"""
import argparse
import json
import os
import sys
from time import time

import numpy as np

from nbody_jets import NBodyJets
from modp_linalg import IncrementalBasis, rref_modp, matmul_modp, _submod


def log(msg):
    print(f"[bc] {msg}", flush=True)


class LeanIncrementalBasis:
    """IncrementalBasis with the same algorithm (reduce block against the
    RREF basis via pivot coefficients, RREF the remainder, back-reduce the
    basis, keep rows sorted by pivot) but uint32 basis storage and
    column/row-chunked GEMMs so peak memory stays near 3x the basis."""

    def __init__(self, prime, col_chunk=4096, row_chunk=3000):
        self.prime = prime
        self._p = np.uint64(prime)
        self.basis = None  # uint32 (rank, n)
        self.pivs = []
        self.col_chunk = col_chunk
        self.row_chunk = row_chunk

    def _reduce(self, rows):
        A = rows[:, self.pivs]  # (k, r) copy, before columns change
        n = rows.shape[1]
        for lo in range(0, n, self.col_chunk):
            hi = min(n, lo + self.col_chunk)
            Bc = self.basis[:, lo:hi].astype(np.uint64)
            rows[:, lo:hi] = _submod(rows[:, lo:hi],
                                     matmul_modp(A, Bc, self.prime),
                                     self.prime)
        return rows

    def add(self, rows):
        rows = rows[np.any(rows, axis=1)] if rows.size else rows
        if rows.shape[0] == 0:
            return self.rank
        if self.basis is not None and self.pivs:
            rows = self._reduce(np.ascontiguousarray(rows))
            rows = rows[np.any(rows, axis=1)]
            if rows.shape[0] == 0:
                return self.rank
        RN, pivN = rref_modp(rows, self.prime)
        if self.basis is None:
            self.basis = RN.astype(np.uint32)
            self.pivs = list(pivN)
        elif RN.shape[0]:
            C = self.basis[:, pivN].astype(np.uint64)  # (r, k')
            r = self.basis.shape[0]
            for lo in range(0, r, self.row_chunk):
                hi = min(r, lo + self.row_chunk)
                blk = self.basis[lo:hi].astype(np.uint64)
                blk = _submod(blk, matmul_modp(C[lo:hi], RN, self.prime),
                              self.prime)
                self.basis[lo:hi] = blk.astype(np.uint32)
            self.basis = np.concatenate(
                [self.basis, RN.astype(np.uint32)], axis=0)
            self.pivs = self.pivs + list(pivN)
            order = np.argsort(self.pivs)
            self.basis = self.basis[order]
            self.pivs = [self.pivs[j] for j in order]
        return self.rank

    @property
    def rank(self):
        return 0 if self.basis is None else self.basis.shape[0]


class BasisCensus:
    def __init__(self, N, d, M, prime, upow=1):
        self.eng = NBodyJets(N, d, M, prime, upow=upow)
        self.N, self.d, self.M, self.p = N, d, M, prime
        self.m = len(self.eng.pairs)
        # tree registry: id -> (level, (a, b)); base trees have pair None
        self.level = [0] * self.m
        self.pair = [None] * self.m
        self.sel = {0: list(range(self.m))}  # refined by select()
        self.cand_counts = {}
        self.sel_ranks = {}
        self.sel_saturated = False

    # ------------------------------------------------------------ trees
    def _candidates(self, k):
        """Unordered pairs (a, b): a in sel[k-1], b in sel[<=k-1], a != b."""
        lows = [t for lv in range(k) for t in self.sel[lv]]
        seen = set()
        out = []
        for a in self.sel[k - 1]:
            for b in lows:
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                out.append((a, b))
        return out

    def _register(self, k, a, b):
        self.level.append(k)
        self.pair.append((a, b))
        return len(self.level) - 1

    def _jets_for_selected(self, pts, upto):
        """Jets of every selected tree with level <= upto at pts.
        Base jets order M, level-k jets order M-k."""
        eng, M = self.eng, self.M
        chains = eng.chain_jets(pts, max(M - 1, 0))
        jets = {t: j for t, j in enumerate(eng.base_jets(pts))}
        for k in range(1, upto + 1):
            for t in self.sel[k]:
                a, b = self.pair[t]
                jets[t] = eng.bracket(jets[a], jets[b], M - k, chains)
        return jets, chains

    # -------------------------------------------------------- selection
    def select(self, rng, S_sel):
        """Phase 1: choose independent spanning subsets at levels < M."""
        eng, M, p = self.eng, self.M, self.p
        pts = rng.integers(1, p, size=(S_sel, eng.nv), dtype=np.int64)
        jets, chains = self._jets_for_selected(pts, 0)
        # level 0
        V = np.stack([jets[t][0] for t in self.sel[0]])
        _, pivs = rref_modp(np.ascontiguousarray(V.T), p)
        self.sel[0] = [self.sel[0][j] for j in pivs]
        self.cand_counts[0] = self.m
        self.sel_ranks[0] = len(pivs)
        low = V[pivs]
        log(f"select L0: {self.m} base -> rank {len(pivs)}")
        for k in range(1, M):
            t1 = time()
            cands = self._candidates(k)
            self.cand_counts[k] = len(cands)
            cj = [eng.bracket(jets[a], jets[b], M - k, chains)
                  for a, b in cands]
            Vc = np.stack([j[0] for j in cj])
            V = np.concatenate([low, Vc], axis=0)
            n_low = low.shape[0]
            _, pivs = rref_modp(np.ascontiguousarray(V.T), p)
            assert all(j in pivs for j in range(n_low)), \
                "selected lower trees lost independence (impossible)"
            new = [j - n_low for j in pivs if j >= n_low]
            ids = []
            for j in new:
                a, b = cands[j]
                t = self._register(k, a, b)
                jets[t] = cj[j]
                ids.append(t)
            self.sel[k] = ids
            self.sel_ranks[k] = len(pivs)
            low = V[pivs]
            log(f"select L{k}: {len(cands)} candidates -> {len(new)} new, "
                f"cumulative rank {len(pivs)}  ({time() - t1:.1f}s)")
        self.sel_saturated = self.sel_ranks[M - 1] > S_sel // 2
        return self.sel_ranks

    # ------------------------------------------------------- evaluation
    def _grad_form(self, jets, chains, ids):
        """Effective (total) gradients of the order>=1 jets of trees
        `ids`: returns Gq, Gp arrays of shape (n, N*d, S)."""
        eng, p = self.eng, np.uint64(self.p)
        N, d = self.N, self.d
        S = next(iter(jets.values())).shape[1]
        nqd = N * d
        Gq = np.zeros((len(ids), nqd, S), dtype=np.uint64)
        Gp = np.zeros((len(ids), nqd, S), dtype=np.uint64)
        idx = eng.ctx.index
        nv = eng.nv

        def lin(v):
            e = [0] * nv
            e[v] = 1
            return idx[tuple(e)]

        for r, t in enumerate(ids):
            J = jets[t]
            for b in range(1, N + 1):
                for k in range(d):
                    vq, vp = eng.qi(b, k), eng.pi(b, k)
                    Gp[r, (b - 1) * d + k] = J[lin(vp)]
                    acc = J[lin(vq)].copy()
                    for (i, j) in eng.pairs:
                        if b != i and b != j:
                            continue
                        vu = eng.ui(i, j)
                        cu = J[lin(vu)]
                        if cu.any():
                            acc = (acc + cu * chains[(vu, vq)][0]) % p
                    Gq[r, (b - 1) * d + k] = acc
        return Gq, Gp

    def top_pairs(self):
        """Candidate index pairs for the top level, as index arrays into
        (frontier list, all-selected list)."""
        M = self.M
        front = self.sel[M - 1]
        alls = [t for lv in range(M) for t in self.sel[lv]]
        pos = {t: i for i, t in enumerate(alls)}
        seen = set()
        ia, ib = [], []
        for i, a in enumerate(front):
            for b in alls:
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                ia.append(i)
                ib.append(pos[b])
        self.cand_counts[M] = len(ia)
        return front, alls, np.array(ia), np.array(ib)

    def evaluate(self, rng, S, batch, values_path):
        """Phase 2: values of all selected trees (levels < M) and of all
        top-level candidates at S fresh points. Top-level values go to a
        uint32 .npy memmap at values_path (n_top, S); lower levels are
        returned in memory and also saved to values_path + '.low.npz'.
        Returns dict level -> (n, S) uint32 (top level is the memmap)."""
        eng, M, p = self.eng, self.M, np.uint64(self.p)
        front, alls, ia, ib = self.top_pairs()
        pos = {t: i for i, t in enumerate(alls)}
        fi = np.array([pos[t] for t in front])
        nqd = self.N * self.d
        top = np.lib.format.open_memmap(values_path, mode="w+",
                                        dtype=np.uint32,
                                        shape=(len(ia), S))
        low = {L: np.zeros((len(self.sel[L]), S), dtype=np.uint32)
               for L in range(M)}
        done, t0 = 0, time()
        while done < S:
            n = min(batch, S - done)
            pts = rng.integers(1, self.p, size=(n, eng.nv), dtype=np.int64)
            jets, chains = self._jets_for_selected(pts, M - 1)
            for L in range(M):
                low[L][:, done:done + n] = np.stack(
                    [jets[t][0] for t in self.sel[L]]).astype(np.uint32)
            Gq, Gp = self._grad_form(jets, chains, alls)
            acc = np.zeros((len(front), len(alls), n), dtype=np.uint64)
            for i in range(nqd):
                acc = (acc + (Gq[fi, i, None, :] * Gp[None, :, i, :]) % p) % p
                acc = (acc + (p - (Gp[fi, i, None, :] * Gq[None, :, i, :]) % p)) % p
            top[:, done:done + n] = acc[ia, ib].astype(np.uint32)
            del acc, Gq, Gp, jets, chains
            done += n
            log(f"eval {done}/{S} points ({time() - t0:.0f}s)")
        top.flush()
        np.savez(values_path + ".low.npz",
                 **{f"L{L}": low[L] for L in range(M)},
                 meta=np.array(json.dumps(self.meta())))
        vals = dict(low)
        vals[M] = top
        return vals

    def meta(self):
        return {
            "candidate_counts": [self.cand_counts[k]
                                 for k in range(self.M + 1)],
            "selected_counts": [len(self.sel[k]) for k in range(self.M)],
            "selection_ranks": [self.sel_ranks[k] for k in range(self.M)],
            "sel_saturated": bool(self.sel_saturated),
        }

    @staticmethod
    def load_values(values_path, M):
        top = np.load(values_path, mmap_mode="r")
        z = np.load(values_path + ".low.npz")
        vals = {L: z[f"L{L}"] for L in range(M)}
        vals[M] = top
        return vals, json.loads(str(z["meta"]))

    def ranks(self, vals, kernel="lean"):
        basis = (LeanIncrementalBasis(self.p) if kernel == "lean"
                 else IncrementalBasis(self.p))
        out = []
        for L in range(self.M + 1):
            rows = vals[L]
            t1 = time()
            for lo in range(0, rows.shape[0], 3000):
                basis.add(np.array(rows[lo:lo + 3000], dtype=np.uint64))
            out.append(basis.rank)
            log(f"[{kernel}] cumulative rank through L{L} = {basis.rank} "
                f"({rows.shape[0]} rows, +{time() - t1:.0f}s)")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--L", type=int, required=True, help="max level M")
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--sel-samples", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--prime", type=int, default=(1 << 31) - 1)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--upow", type=int, default=1)
    ap.add_argument("--stage", default="all", choices=["all", "eval", "rank"])
    ap.add_argument("--values", default=None,
                    help=".npy memmap for top-level values "
                         "(default: bc_values_N{N}_d{d}_L{L}_s{seed}.npy)")
    ap.add_argument("--check-kernels", action="store_true",
                    help="also rank with the dense kernel and require "
                         "agreement (validation only; memory-hungry)")
    ap.add_argument("--expect", default=None,
                    help="comma-separated known cumulative ranks to check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    values_path = args.values or (
        f"bc_values_N{args.N}_d{args.d}_L{args.L}_s{args.seed}.npy")

    t0 = time()
    log(f"N={args.N} d={args.d} L<={args.L} V=1/r^{args.upow} "
        f"S={args.samples} S_sel={args.sel_samples} p={args.prime} "
        f"seed={args.seed} stage={args.stage} values={values_path}")
    bc = BasisCensus(args.N, args.d, args.L, args.prime, upow=args.upow)

    if args.stage in ("all", "eval"):
        rng = np.random.default_rng(args.seed)
        bc.select(rng, args.sel_samples)
        if bc.sel_saturated:
            log("WARNING: selection rank exceeds S_sel/2 -- raise "
                "--sel-samples")
        vals = bc.evaluate(rng, args.samples, args.batch, values_path)
        meta = bc.meta()
        log(f"eval stage done ({time() - t0:.0f}s); values at "
            f"{values_path}")
        if args.stage == "eval":
            return 0
    else:
        vals, meta = BasisCensus.load_values(values_path, args.L)
        log(f"loaded values: top {vals[args.L].shape}, meta {meta}")

    cum = bc.ranks(vals, kernel="lean")
    kernels_agree = True
    if args.check_kernels:
        cum_dense = bc.ranks(vals, kernel="dense")
        kernels_agree = cum_dense == cum
        log(f"dense kernel = {cum_dense}; agree: {kernels_agree}")
    t_total = time() - t0

    sel_ranks = meta["selection_ranks"]
    sel_consistent = all(cum[k] == sel_ranks[k] for k in range(args.L))
    saturated = cum[-1] > args.samples - 64
    verdict = []
    if args.expect:
        exp = [int(x) for x in args.expect.split(",")]
        ok = cum[:len(exp)] == exp
        verdict.append(f"expected {exp}: {'MATCH' if ok else 'MISMATCH'}")
    log("=" * 60)
    log(f"cumulative ranks = {cum}")
    log(f"candidates per level = {meta['candidate_counts']}")
    log(f"selected per level = {meta['selected_counts']} + top")
    log(f"selection/evaluation lower-level ranks consistent: "
        f"{sel_consistent}")
    if saturated:
        log("WARNING: top rank saturated n_samples -- rerun larger")
    for v in verdict:
        log(v)
    log(f"total {t_total:.0f}s")
    if args.out:
        json.dump({
            "method": "basis-reduced jet-evaluation mod-p census rank "
                      "(no symbolic generation; independent spanning "
                      "subsets carried below the top level)",
            "N": args.N, "d": args.d,
            "potential": "1/r" if args.upow == 1 else f"1/r^{args.upow}",
            "max_level": args.L, "prime": args.prime, "seed": args.seed,
            "n_samples": args.samples, "n_sel_samples": args.sel_samples,
            "candidate_counts": meta["candidate_counts"],
            "selected_counts": meta["selected_counts"],
            "selection_ranks": sel_ranks,
            "cumulative_ranks": [int(x) for x in cum],
            "selection_consistent": bool(sel_consistent),
            "saturated": bool(saturated),
            "rank_kernel": "lean" + (" (dense agreed)"
                                     if args.check_kernels else ""),
            "expected": args.expect,
            "verdict": verdict,
            "values_file": os.path.basename(values_path),
            "t_total_s": round(t_total, 1),
        }, open(args.out, "w"), indent=2)
        log(f"wrote {args.out}")
    good = (sel_consistent and not saturated and kernels_agree
            and all(v.endswith("MATCH") and "MISMATCH" not in v
                    for v in verdict))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
