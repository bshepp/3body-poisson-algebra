"""Jet evaluation of N-body Poisson-bracket trees mod p.

Evaluates every bracket tree of the level filtration numerically at
random sample points in F_p, via truncated Taylor jets (jets.py) —
no symbolic expression is ever constructed. A level-L tree needs only
order-(Lmax - L) jets, and lower-level jets are shared by all higher
trees, so the cost per tree is a handful of truncated jet products.

Variable layout mirrors nbody.exact_growth_nbody.NBodyAlgebra:
q (body-major, coord-minor), then p, then u_ij ordered by
combinations. u_ij = 1/r_ij is an independent coordinate; position
derivatives go through the chain rule
    d u_ij / d q_ik = -(q_ik - q_jk) * u_ij^3
exactly as in the symbolic engine (independent-u Schwartz-Zippel,
same convention as every prior mod-p leg of this project).

Tree enumeration mirrors symbolic_rank_nbody.build_generators:
level 1 = all pairs of base Hamiltonians; level >= 2 = frontier x all
existing, deduplicated by unordered pair across levels. This
reproduces the full candidate census (zero trees included — they
contribute zero rows and cannot change any rank).
"""
from itertools import combinations

import numpy as np

from jets import JetContext


class NBodyJets:
    def __init__(self, N, d, max_level, prime=(1 << 31) - 1):
        self.N = N
        self.d = d
        self.max_level = max_level
        self.prime = prime
        self.n_q = N * d
        self.pairs = list(combinations(range(1, N + 1), 2))
        self.n_u = len(self.pairs)
        self.nv = 2 * N * d + self.n_u
        self.ctx = JetContext(self.nv, max_level, prime)

    # variable indices
    def qi(self, body, k):
        return (body - 1) * self.d + k

    def pi(self, body, k):
        return self.N * self.d + (body - 1) * self.d + k

    def ui(self, i, j):
        return 2 * self.N * self.d + self.pairs.index((i, j))

    def base_jets(self, pts):
        """Jets (order max_level) of the C(N,2) pair Hamiltonians at
        pts (S, nv). Unit masses, V = 1/r i.e. + u_ij."""
        ctx = self.ctx
        M = self.max_level
        inv2 = (self.prime + 1) // 2
        H = []
        for (i, j) in self.pairs:
            acc = ctx.zero(M, pts.shape[0])
            for b in (i, j):
                for k in range(self.d):
                    pv = ctx.variable(self.pi(b, k), pts[:, self.pi(b, k)], M)
                    acc = ctx.add(acc, ctx.scale(ctx.mult(pv, pv, M), inv2))
            uv = ctx.variable(self.ui(i, j), pts[:, self.ui(i, j)], M)
            acc = ctx.add(acc, uv)
            H.append(acc)
        return H

    def chain_jets(self, pts, order):
        """chain[(u_index, q_index)] = jet of d u_ij / d q_bk."""
        ctx = self.ctx
        out = {}
        for (i, j) in self.pairs:
            vu = self.ui(i, j)
            uj = ctx.variable(vu, pts[:, vu], order)
            u3 = ctx.mult(ctx.mult(uj, uj, order), uj, order)
            for k in range(self.d):
                qi_ = ctx.variable(self.qi(i, k), pts[:, self.qi(i, k)],
                                   order)
                qj_ = ctx.variable(self.qi(j, k), pts[:, self.qi(j, k)],
                                   order)
                diff = ctx.sub(qi_, qj_)
                base = ctx.mult(diff, u3, order)
                # d u/d q_ik = -(q_ik - q_jk) u^3 ; d u/d q_jk = +...
                out[(vu, self.qi(i, k))] = ctx.scale(base, self.prime - 1)
                out[(vu, self.qi(j, k))] = base
        return out

    def bracket(self, fj, gj, m_out, chains):
        """{f, g} as an order-m_out jet, from order-(m_out+1) jets."""
        ctx = self.ctx
        S = fj.shape[1]
        out = ctx.zero(m_out, S)
        # cache partials
        df = {}
        dg = {}

        def d_of(jet, cache, v):
            if v not in cache:
                cache[v] = ctx.deriv(jet, v, m_out)
            return cache[v]

        for b in range(1, self.N + 1):
            for k in range(self.d):
                vq = self.qi(b, k)
                vp = self.pi(b, k)
                Dqf = d_of(fj, df, vq)
                Dqg = d_of(gj, dg, vq)
                for (i, j) in self.pairs:
                    if b != i and b != j:
                        continue
                    vu = self.ui(i, j)
                    cj = chains[(vu, vq)]
                    dfu = d_of(fj, df, vu)
                    if dfu.any():
                        Dqf = ctx.add(Dqf, ctx.mult(dfu, cj, m_out))
                    dgu = d_of(gj, dg, vu)
                    if dgu.any():
                        Dqg = ctx.add(Dqg, ctx.mult(dgu, cj, m_out))
                t1 = ctx.mult(Dqf, d_of(gj, dg, vp), m_out)
                t2 = ctx.mult(d_of(fj, df, vp), Dqg, m_out)
                out = ctx.add(out, ctx.sub(t1, t2))
        return out

    def enumerate_trees(self):
        """(level, a, b) tree list mirroring build_generators; returns
        list of per-level lists of (a_index, b_index) into the growing
        tree list, plus the level of every tree."""
        m = self.n_u  # wrong name guard: base gens count = C(N,2)
        m = len(self.pairs)
        levels = [0] * m
        tree_pairs = [None] * m
        done = set()
        per_level = {0: list(range(m))}
        for L in range(1, self.max_level + 1):
            new = []
            if L == 1:
                cand = list(combinations(range(m), 2))
            else:
                frontier = [t for t, lv in enumerate(levels) if lv == L - 1]
                base = len(levels)
                cand = [(a, b) for a in frontier for b in range(base)
                        if a != b]
            for a, b in cand:
                key = frozenset((a, b))
                if key in done:
                    continue
                done.add(key)
                tree_pairs.append((a, b))
                levels.append(L)
                new.append(len(levels) - 1)
            per_level[L] = new
        self.levels = levels
        self.tree_pairs = tree_pairs
        self.per_level = per_level
        return per_level

    def values_batch(self, pts):
        """Evaluate every tree's value at pts (S, nv).
        Returns dict level -> (n_trees_level, S) uint64 array."""
        ctx = self.ctx
        M = self.max_level
        chains = self.chain_jets(pts, max(M - 1, 0))
        jets = list(self.base_jets(pts))  # index-aligned with trees
        vals = {0: np.stack([j[0] for j in jets])}
        for L in range(1, M + 1):
            m_out = M - L
            rows = []
            for t in self.per_level[L]:
                a, b = self.tree_pairs[t]
                j = self.bracket(jets[a], jets[b], m_out, chains)
                jets.append(j)
                rows.append(j[0])
            vals[L] = (np.stack(rows) if rows
                       else np.zeros((0, pts.shape[0]), dtype=np.uint64))
        return vals
