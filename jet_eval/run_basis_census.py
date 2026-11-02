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

Usage:
  python run_basis_census.py --N 6 --d 1 --L 3 --samples 24576 \
      [--sel-samples 4096] [--batch 512] [--prime P] [--seed SEED] \
      [--upow 1] [--out FILE.json]
"""
import argparse
import json
import sys
from time import time

import numpy as np

from nbody_jets import NBodyJets
from modp_linalg import IncrementalBasis, rref_modp


def log(msg):
    print(f"[bc] {msg}", flush=True)


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

    def evaluate(self, rng, S, batch):
        """Phase 2: values of all selected trees (levels < M) and of all
        top-level candidates at S fresh points. Returns dict level ->
        (n, S) uint32."""
        eng, M, p = self.eng, self.M, np.uint64(self.p)
        front, alls, ia, ib = self.top_pairs()
        pos = {t: i for i, t in enumerate(alls)}
        fi = np.array([pos[t] for t in front])
        nqd = self.N * self.d
        vals = {L: [] for L in range(M + 1)}
        done, t0 = 0, time()
        while done < S:
            n = min(batch, S - done)
            pts = rng.integers(1, self.p, size=(n, eng.nv), dtype=np.int64)
            jets, chains = self._jets_for_selected(pts, M - 1)
            for L in range(M):
                vals[L].append(np.stack([jets[t][0] for t in self.sel[L]])
                               .astype(np.uint32))
            Gq, Gp = self._grad_form(jets, chains, alls)
            acc = np.zeros((len(front), len(alls), n), dtype=np.uint64)
            for i in range(nqd):
                acc = (acc + (Gq[fi, i, None, :] * Gp[None, :, i, :]) % p) % p
                acc = (acc + (p - (Gp[fi, i, None, :] * Gq[None, :, i, :]) % p)) % p
            vals[M].append(acc[ia, ib].astype(np.uint32))
            done += n
            log(f"eval {done}/{S} points ({time() - t0:.0f}s)")
        return {L: np.concatenate(v, axis=1) for L, v in vals.items()}

    def ranks(self, vals):
        basis = IncrementalBasis(self.p)
        out = []
        for L in range(self.M + 1):
            rows = vals[L]
            t1 = time()
            for lo in range(0, rows.shape[0], 3000):
                basis.add(rows[lo:lo + 3000].astype(np.uint64))
            out.append(basis.rank)
            log(f"cumulative rank through L{L} = {basis.rank} "
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
    ap.add_argument("--expect", default=None,
                    help="comma-separated known cumulative ranks to check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time()
    log(f"N={args.N} d={args.d} L<={args.L} V=1/r^{args.upow} "
        f"S={args.samples} S_sel={args.sel_samples} p={args.prime} "
        f"seed={args.seed}")
    bc = BasisCensus(args.N, args.d, args.L, args.prime, upow=args.upow)
    rng = np.random.default_rng(args.seed)
    sel_ranks = bc.select(rng, args.sel_samples)
    if bc.sel_saturated:
        log("WARNING: selection rank exceeds S_sel/2 -- raise --sel-samples")
    vals = bc.evaluate(rng, args.samples, args.batch)
    cum = bc.ranks(vals)
    t_total = time() - t0

    sel_consistent = all(cum[k] == sel_ranks[k] for k in range(args.L))
    saturated = cum[-1] > args.samples - 64
    verdict = []
    if args.expect:
        exp = [int(x) for x in args.expect.split(",")]
        ok = cum[:len(exp)] == exp
        verdict.append(f"expected {exp}: {'MATCH' if ok else 'MISMATCH'}")
    log("=" * 60)
    log(f"cumulative ranks = {cum}")
    log(f"candidates per level = "
        f"{[bc.cand_counts[k] for k in range(args.L + 1)]}")
    log(f"selected per level = {[len(bc.sel[k]) for k in range(args.L)]}"
        f" + top")
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
            "candidate_counts": [bc.cand_counts[k]
                                 for k in range(args.L + 1)],
            "selected_counts": [len(bc.sel[k]) for k in range(args.L)],
            "selection_ranks": [sel_ranks[k] for k in range(args.L)],
            "cumulative_ranks": [int(x) for x in cum],
            "selection_consistent": bool(sel_consistent),
            "saturated": bool(saturated),
            "expected": args.expect,
            "verdict": verdict,
            "t_total_s": round(t_total, 1),
        }, open(args.out, "w"), indent=2)
        log(f"wrote {args.out}")
    good = (sel_consistent and not saturated
            and all(v.endswith("MATCH") and "MISMATCH" not in v
                    for v in verdict))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
