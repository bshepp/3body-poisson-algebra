#!/usr/bin/env python3
"""
N-Body Symbolic Rank Over Q — Exact Algebraic Dimension
=========================================================

Computes the exact rank of the Poisson algebra generators for arbitrary
N-body systems over Q, using the monomial-coefficient matrix and exact
Gaussian elimination via DomainMatrix.

No SVD, no thresholds, no numerical approximation.

Supports multiprocessing (--workers) for bracket computation on Linux
(uses fork to share SymPy expression trees without serialization).

Usage
-----
    # N=3 baseline (should give [3, 6, 17, 116])
    python symbolic_rank_nbody.py -N 3 -d 2 --max-level 3

    # N=5, 1D, through level 3, 15 workers
    python symbolic_rank_nbody.py -N 5 -d 1 --max-level 3 --workers 15

    # N=6, 2D, through level 2
    python symbolic_rank_nbody.py -N 6 -d 2 --max-level 2

    # 1/r^4 singular potential (N=3, 2D)
    python symbolic_rank_nbody.py -N 3 -d 2 --potential composite --composite -1 4 --max-level 3

    # r^4 quartic spring (N=3, 2D)
    python symbolic_rank_nbody.py -N 3 -d 2 --potential r^4 --max-level 4
"""

import os
import sys
import json
import signal
import argparse
from time import time, strftime
from multiprocessing import Pool, cpu_count

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
from sympy import Integer, Rational, Add, Poly, expand, cancel, diff, Symbol

import pickle

from exact_growth_nbody import NBodyAlgebra, COORD_LABELS


# =====================================================================
# Module-level globals for multiprocessing workers (fork-inherited)
# =====================================================================
_WORKER_ENGINE = None
_WORKER_EXPRS = None
_WORKER_NAMES = None
_WORKER_PHASE_VARS = None
_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n  [SIGTERM] Received at {strftime('%Y-%m-%dT%H:%M:%SZ')} "
          f"-- finishing current bracket batch...", flush=True)


def _compute_one_bracket(args):
    """Worker function: compute one Poisson bracket {f_i, f_j}.

    Uses fork-inherited _WORKER_ENGINE and _WORKER_EXPRS to avoid
    pickling large SymPy expression trees.
    """
    idx, i, j = args
    t0 = time()
    expr = _WORKER_ENGINE._poisson_bracket(_WORKER_EXPRS[i], _WORKER_EXPRS[j])
    t_b = time() - t0
    t0 = time()
    expr = _WORKER_ENGINE._simplify(expr)
    t_s = time() - t0
    nterms = len(Add.make_args(expr))
    ni, nj = _WORKER_NAMES[i], _WORKER_NAMES[j]
    bname = f"{{{ni},{nj}}}"
    return (idx, i, j, bname, expr, nterms, t_b, t_s)


def _extract_one_monomial(idx):
    """Worker function: expand one expression and extract its monomial dict.

    Uses fork-inherited _WORKER_EXPRS and _WORKER_PHASE_VARS.
    Returns (idx, monom_dict) where monom_dict maps monomial tuples to coefficients.
    """
    expanded = expand(_WORKER_EXPRS[idx])
    p = Poly(expanded, *_WORKER_PHASE_VARS, domain='QQ')
    return (idx, p.as_dict())


class NBodySymbolicRank:
    """Exact rank computation for N-body Poisson algebras."""

    def __init__(self, n_bodies, d_spatial, potential, potential_params=None,
                 masses=None, charges=None, quantum=False,
                 external_potential=None):
        self.n_bodies = n_bodies
        self.d_spatial = d_spatial
        self.potential = potential
        self.quantum = quantum
        self.external_potential = external_potential

        if potential in ('r^4', 'r^2'):
            self._init_polynomial_spring(n_bodies, d_spatial, masses, potential)
            if quantum:
                from quantum_algebra import hbar_sym
                self.hbar_sym = hbar_sym
        elif quantum:
            from quantum_algebra import QuantumNBodyAlgebra, hbar_sym
            self.algebra = QuantumNBodyAlgebra(
                n_bodies, d_spatial, potential,
                potential_params=potential_params,
                masses=masses, charges=charges,
                external_potential=external_potential)
            self.phase_vars = list(self.algebra.all_vars)
            self.uses_u = True
            self.hbar_sym = hbar_sym
        else:
            self.algebra = NBodyAlgebra(
                n_bodies, d_spatial, potential,
                potential_params=potential_params,
                masses=masses, charges=charges,
                external_potential=external_potential)
            self.phase_vars = list(self.algebra.all_vars)
            self.uses_u = True

    def _init_polynomial_spring(self, n_bodies, d_spatial, masses, potential):
        """Build polynomial potential Hamiltonians directly.

        r^2 (harmonic) and r^4 (quartic spring) are polynomial in positions,
        no u_ij needed.
        """
        from itertools import combinations
        labels = COORD_LABELS[d_spatial]
        power = {'r^2': 1, 'r^4': 2}[potential]

        self.q_vars = []
        self.p_vars = []
        self.q_by_body = {}
        self.p_by_body = {}

        for body in range(1, n_bodies + 1):
            q_body, p_body = [], []
            for c in labels:
                q_sym = Symbol(f"{c}{body}", real=True)
                p_sym = Symbol(f"p{c}{body}", real=True)
                self.q_vars.append(q_sym)
                self.p_vars.append(p_sym)
                q_body.append(q_sym)
                p_body.append(p_sym)
            self.q_by_body[body] = q_body
            self.p_by_body[body] = p_body

        self.phase_vars = self.q_vars + self.p_vars
        self.uses_u = False

        if masses is None:
            masses_dict = {b: Integer(1) for b in range(1, n_bodies + 1)}
        else:
            masses_dict = masses

        body_pairs = list(combinations(range(1, n_bodies + 1), 2))
        self.hamiltonian_list = []
        self.hamiltonian_names = []

        for bi, bj in body_pairs:
            mi, mj = masses_dict[bi], masses_dict[bj]
            KE_i = sum(p**2 for p in self.p_by_body[bi]) / (2 * mi)
            KE_j = sum(p**2 for p in self.p_by_body[bj]) / (2 * mj)
            r_sq = sum((self.q_by_body[bi][k] - self.q_by_body[bj][k])**2
                       for k in range(d_spatial))
            V = r_sq ** power
            H = KE_i + KE_j + V
            name = f"H{bi}{bj}"
            self.hamiltonian_list.append(H)
            self.hamiltonian_names.append(name)

        self.algebra = None

    def _poisson_bracket(self, f, g):
        """Compute bracket: Poisson {f,g} or quantum [f,g]/(i*hbar)."""
        if self.quantum and self.uses_u:
            return self.algebra.commutator(f, g)
        if self.uses_u:
            return self.algebra.poisson_bracket(f, g)

        # Classical Poisson bracket
        pb = Integer(0)
        for q, p in zip(self.q_vars, self.p_vars):
            pb += diff(f, q) * diff(g, p) - diff(f, p) * diff(g, q)

        if not self.quantum:
            return pb

        # Moyal bracket for polynomial phase-space functions (no u_ij).
        #
        # [f,g]_M / (i*hbar) = sum_{n=0}^{...} (-1)^n (hbar/2)^{2n}
        #                       / (2n+1)! * P_{2n+1}(f,g)
        #
        # P_m(f,g) = sum_{k=0}^{m} (-1)^k C(m,k)
        #   * [sum over DOF distributions of (m-k) q-derivs and k p-derivs]
        #   * (multi-deriv f) * (multi-deriv g)
        #
        # Implementation: distribute derivatives across DOFs via
        # multinomial partitions.

        from quantum_algebra import hbar_sym
        from math import factorial, comb

        result = pb
        f_exp = expand(f)
        g_exp = expand(g)

        max_deg = 0
        for v in self.q_vars + self.p_vars:
            max_deg = max(max_deg, sp.degree(f_exp, v),
                          sp.degree(g_exp, v))

        n_dof = len(self.q_vars)

        for n in range(1, max_deg // 2 + 1):
            order = 2 * n + 1
            coeff = Rational((-1)**n, factorial(order) * 2**(2*n))

            p_term = Integer(0)
            for k in range(order + 1):
                sign = (-1)**k
                nq = order - k  # q-derivatives on f, p-derivatives on g
                np_ = k         # p-derivatives on f, q-derivatives on g

                for qa in self._partitions(nq, n_dof):
                    mn_q = factorial(nq)
                    skip = False
                    for v in qa:
                        mn_q //= factorial(v)

                    for pa in self._partitions(np_, n_dof):
                        mn_p = factorial(np_)
                        for v in pa:
                            mn_p //= factorial(v)

                        df = f_exp
                        for j in range(n_dof):
                            if qa[j] > 0:
                                df = diff(df, self.q_vars[j], qa[j])
                            if pa[j] > 0:
                                df = diff(df, self.p_vars[j], pa[j])
                        if df == 0:
                            continue

                        dg = g_exp
                        for j in range(n_dof):
                            if pa[j] > 0:
                                dg = diff(dg, self.q_vars[j], pa[j])
                            if qa[j] > 0:
                                dg = diff(dg, self.p_vars[j], qa[j])
                        if dg == 0:
                            continue

                        p_term += sign * mn_q * mn_p * df * dg

            if p_term != 0:
                result += hbar_sym**(2*n) * coeff * p_term

        return result

    @staticmethod
    def _partitions(n, k):
        """Generate all k-tuples of non-negative integers summing to n."""
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for rest in NBodySymbolicRank._partitions(n - i, k - 1):
                yield (i,) + rest

    def _simplify(self, expr):
        if self.uses_u:
            return cancel(expr)
        return expand(expr)

    def _ckpt_path(self, checkpoint_dir, tag):
        """Return path for a checkpoint file."""
        if checkpoint_dir is None:
            return None
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, f"{tag}.pkl")

    def save_checkpoint(self, checkpoint_dir, tag, data):
        """Save a checkpoint to disk."""
        path = self._ckpt_path(checkpoint_dir, tag)
        if path is None:
            return
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        print(f"  [checkpoint] Saved {tag} ({os.path.getsize(path)/1024/1024:.1f} MB)")

    def load_checkpoint(self, checkpoint_dir, tag):
        """Load a checkpoint if it exists. Returns None if not found."""
        path = self._ckpt_path(checkpoint_dir, tag)
        if path is None or not os.path.exists(path):
            return None
        print(f"  [checkpoint] Loading {tag}...")
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"  [checkpoint] Loaded {tag} ({os.path.getsize(path)/1024/1024:.1f} MB)")
        return data

    def build_generators(self, max_level, checkpoint_dir=None, n_workers=1):
        """Build generators through max_level, with optional per-level
        checkpointing and multiprocessing.

        On Linux, n_workers > 1 uses multiprocessing.Pool with fork()
        to share expression trees without serialization overhead.
        """
        global _WORKER_ENGINE, _WORKER_EXPRS, _WORKER_NAMES

        print(f"Building generators: N={self.n_bodies}, d={self.d_spatial}, "
              f"potential={self.potential}, max_level={max_level}, "
              f"workers={n_workers}")
        t_total = time()

        ckpt = self.load_checkpoint(checkpoint_dir, f"generators_level{max_level}")
        if ckpt is not None:
            all_exprs, all_names, all_levels, computed_pairs = ckpt
            print(f"  Resumed from complete checkpoint: {len(all_exprs)} generators")
            return all_exprs, all_names, all_levels

        if self.potential in ('r^4', 'r^2'):
            all_exprs = list(self.hamiltonian_list)
            all_names = list(self.hamiltonian_names)
        else:
            all_exprs = list(self.algebra.hamiltonian_list)
            all_names = list(self.algebra.hamiltonian_names)

        n_l0 = len(all_exprs)
        all_levels = [0] * n_l0
        computed_pairs = set()

        resume_level = -1
        for lv in range(max_level, -1, -1):
            ckpt = self.load_checkpoint(checkpoint_dir, f"generators_level{lv}")
            if ckpt is not None:
                all_exprs, all_names, all_levels, computed_pairs = ckpt
                resume_level = lv
                print(f"  Resumed from level {lv} checkpoint: "
                      f"{len(all_exprs)} generators")
                break

        if resume_level < 0:
            print(f"\n--- Level 0: {n_l0} Pairwise Hamiltonians ---")
            for name, expr in zip(all_names, all_exprs):
                nterms = len(Add.make_args(expr))
                print(f"  {name}: {nterms} terms")

            for i in range(n_l0):
                for j in range(i + 1, n_l0):
                    computed_pairs.add(frozenset({i, j}))

            if max_level >= 1:
                print(f"\n--- Level 1: Brackets of {n_l0} Hamiltonians ---")
                for i in range(n_l0):
                    for j in range(i + 1, n_l0):
                        ni, nj = all_names[i], all_names[j]
                        bname = f"{{{ni},{nj}}}"
                        t0 = time()
                        expr = self._poisson_bracket(all_exprs[i], all_exprs[j])
                        expr = self._simplify(expr)
                        nterms = len(Add.make_args(expr))
                        print(f"  {bname} -> {nterms} terms [{time()-t0:.1f}s]")
                        all_exprs.append(expr)
                        all_names.append(bname)
                        all_levels.append(1)

                self.save_checkpoint(checkpoint_dir, "generators_level1",
                                     (all_exprs, all_names, all_levels, computed_pairs))
            resume_level = 1

        # Levels 2+
        for level in range(max(2, resume_level + 1), max_level + 1):
            print(f"\n--- Level {level} ---")
            t_level = time()
            frontier = [i for i, lv in enumerate(all_levels) if lv == level - 1]
            n_existing = len(all_exprs)

            # Pre-enumerate all work items for this level
            work_items = []
            for i in frontier:
                for j in range(n_existing):
                    if i == j:
                        continue
                    pair = frozenset({i, j})
                    if pair in computed_pairs:
                        continue
                    computed_pairs.add(pair)
                    work_items.append((len(work_items), i, j))

            n_total = len(work_items)
            print(f"  {n_total} candidate brackets "
                  f"(frontier={len(frontier)}, existing={n_existing})")

            # Try to resume intra-level checkpoint
            intra_ckpt = self.load_checkpoint(
                checkpoint_dir, f"generators_level{level}_partial")
            completed_results = {}
            if intra_ckpt is not None:
                completed_results = intra_ckpt
                print(f"  Resumed {len(completed_results)}/{n_total} "
                      f"brackets from intra-level checkpoint")

            remaining = [(idx, i, j) for idx, i, j in work_items
                         if idx not in completed_results]
            print(f"  {len(remaining)} brackets remaining, "
                  f"{n_workers} worker(s)")

            use_mp = (n_workers > 1 and len(remaining) > 1
                      and os.name != 'nt')

            if use_mp:
                _WORKER_ENGINE = self
                _WORKER_EXPRS = all_exprs
                _WORKER_NAMES = all_names

                n_done = len(completed_results)
                t_batch = time()
                ckpt_interval = max(50, n_total // 20)

                chunksize = max(1, min(50, len(remaining) // (n_workers * 20)))
                print(f"  Pool chunksize: {chunksize}", flush=True)

                with Pool(processes=n_workers) as pool:
                    for result in pool.imap_unordered(
                            _compute_one_bracket, remaining,
                            chunksize=chunksize):
                        idx, i, j, bname, expr, nterms, t_b, t_s = result
                        completed_results[idx] = (bname, expr)
                        n_done += 1

                        if n_done % 100 == 0 or n_done == n_total:
                            elapsed = time() - t_batch
                            rate = n_done / elapsed if elapsed > 0 else 0
                            eta = (n_total - n_done) / rate if rate > 0 else 0
                            print(f"  [{n_done:>6d}/{n_total}] {bname}  "
                                  f"bracket {t_b:.1f}s  simplify {t_s:.1f}s  "
                                  f"-> {nterms} terms  "
                                  f"({rate:.1f}/s, ETA {eta/60:.0f}min)",
                                  flush=True)
                        elif n_done % 10 == 0:
                            print(f"  [{n_done:>6d}/{n_total}] {bname}  "
                                  f"-> {nterms} terms", flush=True)

                        if (n_done % ckpt_interval == 0
                                and checkpoint_dir is not None):
                            self.save_checkpoint(
                                checkpoint_dir,
                                f"generators_level{level}_partial",
                                completed_results)

                        if _shutdown_requested:
                            print(f"  [SHUTDOWN] Saving partial results "
                                  f"({n_done}/{n_total})...", flush=True)
                            pool.terminate()
                            pool.join()
                            self.save_checkpoint(
                                checkpoint_dir,
                                f"generators_level{level}_partial",
                                completed_results)
                            print(f"  [SHUTDOWN] Checkpoint saved. Exiting.")
                            sys.exit(0)

                _WORKER_ENGINE = None
                _WORKER_EXPRS = None
                _WORKER_NAMES = None
            else:
                # Sequential fallback (Windows or single worker)
                n_done = len(completed_results)
                for idx, i, j in remaining:
                    t0 = time()
                    expr = self._poisson_bracket(all_exprs[i], all_exprs[j])
                    t_b = time() - t0
                    t0 = time()
                    expr = self._simplify(expr)
                    t_s = time() - t0

                    nterms = len(Add.make_args(expr))
                    ni, nj = all_names[i], all_names[j]
                    bname = f"{{{ni},{nj}}}"
                    completed_results[idx] = (bname, expr)
                    n_done += 1

                    print(f"  [{n_done:>6d}/{n_total}] {bname}  "
                          f"bracket {t_b:.1f}s  simplify {t_s:.1f}s  "
                          f"-> {nterms} terms")

                    if (_shutdown_requested and checkpoint_dir is not None):
                        self.save_checkpoint(
                            checkpoint_dir,
                            f"generators_level{level}_partial",
                            completed_results)
                        print(f"  [SHUTDOWN] Checkpoint saved. Exiting.")
                        sys.exit(0)

            # Reassemble results in original order
            for idx in range(n_total):
                bname, expr = completed_results[idx]
                all_exprs.append(expr)
                all_names.append(bname)
                all_levels.append(level)

            print(f"  Level {level}: {n_total} candidates "
                  f"in {time()-t_level:.1f}s")
            self.save_checkpoint(checkpoint_dir, f"generators_level{level}",
                                 (all_exprs, all_names, all_levels, computed_pairs))

            # Clean up intra-level partial checkpoint
            partial_path = self._ckpt_path(
                checkpoint_dir, f"generators_level{level}_partial")
            if partial_path and os.path.exists(partial_path):
                os.remove(partial_path)
                print(f"  [checkpoint] Removed partial checkpoint for level {level}")

        print(f"\nTotal generators: {len(all_exprs)} "
              f"in {time()-t_total:.1f}s")
        return all_exprs, all_names, all_levels

    def extract_monomial_matrix(self, exprs, n_workers=1):
        """Extract monomial-coefficient matrix over QQ."""
        global _WORKER_EXPRS, _WORKER_PHASE_VARS

        print(f"\nExtracting monomial-coefficient matrix "
              f"({len(exprs)} generators, {n_workers} workers)...")
        t0 = time()

        all_monoms = set()
        n_gen = len(exprs)
        use_mp = (n_workers > 1 and n_gen > 100 and os.name != 'nt')

        if use_mp:
            _WORKER_EXPRS = exprs
            _WORKER_PHASE_VARS = self.phase_vars

            poly_list = [None] * n_gen
            chunksize = max(1, min(100, n_gen // (n_workers * 10)))
            n_done = 0
            t_batch = time()

            with Pool(processes=n_workers) as pool:
                for idx, monom_dict in pool.imap_unordered(
                        _extract_one_monomial, range(n_gen),
                        chunksize=chunksize):
                    poly_list[idx] = monom_dict
                    all_monoms.update(monom_dict.keys())
                    n_done += 1

                    if n_done % 5000 == 0 or n_done == n_gen:
                        elapsed = time() - t_batch
                        rate = n_done / elapsed if elapsed > 0 else 0
                        eta = (n_gen - n_done) / rate if rate > 0 else 0
                        print(f"  Processed {n_done}/{n_gen} generators, "
                              f"{len(all_monoms)} distinct monomials "
                              f"({rate:.0f}/s, ETA {eta/60:.1f}min)",
                              flush=True)

            _WORKER_EXPRS = None
            _WORKER_PHASE_VARS = None
        else:
            poly_list = []
            for idx, expr in enumerate(exprs):
                expanded = expand(expr)
                p = Poly(expanded, *self.phase_vars, domain='QQ')
                monom_dict = p.as_dict()
                poly_list.append(monom_dict)
                all_monoms.update(monom_dict.keys())
                if (idx + 1) % 20 == 0 or idx == len(exprs) - 1:
                    print(f"  Processed {idx+1}/{n_gen} generators, "
                          f"{len(all_monoms)} distinct monomials so far")

        monom_list = sorted(all_monoms)
        monom_to_idx = {m: i for i, m in enumerate(monom_list)}

        n_mon = len(monom_list)
        n_nonzero = sum(len(d) for d in poly_list)
        density = n_nonzero / (n_gen * n_mon) if n_gen * n_mon > 0 else 0

        print(f"  Matrix dimensions: {n_gen} x {n_mon}")
        print(f"  Non-zero entries: {n_nonzero}")
        print(f"  Density: {density:.4f}")
        print(f"  Extraction time: {time()-t0:.1f}s")

        return poly_list, monom_list, monom_to_idx

    def compute_exact_rank(self, poly_list, monom_list, monom_to_idx, levels,
                           checkpoint_dir=None):
        """Compute exact rank using DomainMatrix over QQ (or QQ[hbar] for quantum)."""
        from sympy.polys.matrices import DomainMatrix
        from sympy.polys.domains import QQ

        if self.quantum:
            domain = QQ[self.hbar_sym]
            domain_label = f"QQ[hbar]"
        else:
            domain = QQ
            domain_label = "QQ"

        n_mon = len(monom_list)
        results = {}

        # Try to load a rank checkpoint
        ckpt = self.load_checkpoint(checkpoint_dir, "rank_results") \
            if checkpoint_dir else None
        if ckpt is not None:
            results = ckpt
            print(f"  Resumed rank checkpoint: levels {sorted(results.keys())}")

        for max_lv in range(max(levels) + 1):
            if max_lv in results:
                print(f"\n  Rank through level {max_lv}: "
                      f"{results[max_lv]} (from checkpoint)")
                continue

            mask = [i for i, lv in enumerate(levels) if lv <= max_lv]
            n_sel = len(mask)

            print(f"\n  Rank through level {max_lv} "
                  f"({n_sel} generators x {n_mon} monomials, "
                  f"domain={domain_label})...",
                  end=" ", flush=True)
            t0 = time()

            rows = []
            for i in mask:
                row = [domain.zero] * n_mon
                for monom, coeff in poly_list[i].items():
                    col = monom_to_idx[monom]
                    row[col] = domain.convert(coeff)
                rows.append(row)

            dm = DomainMatrix(rows, (n_sel, n_mon), domain)
            rank = dm.rank()
            elapsed = time() - t0

            print(f"rank = {rank}  [{elapsed:.1f}s]")
            results[max_lv] = rank

            self.save_checkpoint(checkpoint_dir, "rank_results", results)

        return results

    def select_basis(self, poly_list, monom_list, monom_to_idx, rank):
        """Select `rank` linearly independent generators via row echelon form.

        Returns the indices of the selected generators.
        """
        from sympy.polys.matrices import DomainMatrix
        from sympy.polys.domains import QQ

        if self.quantum:
            domain = QQ[self.hbar_sym]
        else:
            domain = QQ

        print(f"\n  Selecting basis of {rank} independent generators...",
              end=" ", flush=True)
        t0 = time()

        n_gen = len(poly_list)
        n_mon = len(monom_list)

        rows = []
        for i in range(n_gen):
            row = [domain.zero] * n_mon
            for monom, coeff in poly_list[i].items():
                col = monom_to_idx[monom]
                row[col] = domain.convert(coeff)
            rows.append(row)

        selected = []
        current_rows = []
        for i in range(n_gen):
            trial = current_rows + [rows[i]]
            dm = DomainMatrix(trial, (len(trial), n_mon), domain)
            if dm.rank() == len(trial):
                selected.append(i)
                current_rows.append(rows[i])
                if len(selected) == rank:
                    break

        print(f"done [{time()-t0:.1f}s]")
        print(f"    Selected generators: {selected[:10]}{'...' if len(selected) > 10 else ''}")
        return selected

    def compute_structure_constants(self, exprs, names, levels,
                                    poly_list, monom_list, monom_to_idx,
                                    basis_indices, checkpoint_dir=None):
        """Compute structure constants C[i,j,k] exactly over Q.

        For basis generators e_i, e_j: {e_i, e_j} = sum_k C[i,j,k] * e_k

        Returns C as a float array (converted from exact rationals) and
        the exact rational values as a nested list for serialization.
        """
        import numpy as np
        from sympy.polys.matrices import DomainMatrix
        from sympy.polys.domains import QQ
        from fractions import Fraction

        r = len(basis_indices)
        n_mon = len(monom_list)

        print(f"\n{'='*70}")
        print(f"STRUCTURE CONSTANTS (rank {r})")
        print(f"{'='*70}")
        print(f"  Computing {r}*({r}-1)/2 = {r*(r-1)//2} brackets...")
        t_total = time()

        basis_dm_rows = []
        for idx in basis_indices:
            row = [QQ.zero] * n_mon
            for monom, coeff in poly_list[idx].items():
                col = monom_to_idx[monom]
                row[col] = QQ.convert(coeff)
            basis_dm_rows.append(row)

        basis_dm = DomainMatrix(basis_dm_rows, (r, n_mon), QQ)

        C_exact = [[[None for _ in range(r)] for _ in range(r)] for _ in range(r)]
        C_float = np.zeros((r, r, r))

        # Resume from checkpoint if available
        resume_from = 0
        sc_ckpt_path = None
        if checkpoint_dir:
            sc_ckpt_path = os.path.join(checkpoint_dir, "structure_constants_partial.pkl")
            if os.path.exists(sc_ckpt_path):
                import pickle
                with open(sc_ckpt_path, 'rb') as f:
                    ckpt = pickle.load(f)
                C_float = ckpt['C_float']
                C_exact = ckpt['C_exact']
                resume_from = ckpt['n_computed']
                print(f"  Resuming from checkpoint: {resume_from}/{r*(r-1)//2} brackets")

        n_computed = 0
        for a in range(r):
            i = basis_indices[a]
            for b in range(a + 1, r):
                j = basis_indices[b]
                n_computed += 1

                if n_computed <= resume_from:
                    continue

                bracket = self._poisson_bracket(exprs[i], exprs[j])
                bracket = self._simplify(bracket)

                expanded = expand(bracket)
                p = Poly(expanded, *self.phase_vars, domain='QQ')
                bracket_dict = p.as_dict()

                rhs = [QQ.zero] * n_mon
                for monom, coeff in bracket_dict.items():
                    if monom in monom_to_idx:
                        rhs[monom_to_idx[monom]] = QQ.convert(coeff)

                rhs_dm = DomainMatrix([[v] for v in rhs], (n_mon, 1), QQ)
                system = basis_dm.transpose()
                aug = system.hstack(rhs_dm)

                aug_rref, pivots = aug.rref()

                coeffs = [QQ.zero] * r
                for row_idx in range(min(r, aug_rref.shape[0])):
                    if row_idx < len(pivots) and pivots[row_idx] < r:
                        col = pivots[row_idx]
                        coeffs[col] = aug_rref[row_idx, r].element

                for k in range(r):
                    val = coeffs[k]
                    C_exact[a][b][k] = str(Fraction(int(val.numerator),
                                                     int(val.denominator))
                                           if hasattr(val, 'numerator')
                                           else Fraction(val))
                    C_exact[b][a][k] = str(-Fraction(C_exact[a][b][k]))
                    C_float[a, b, k] = float(Fraction(C_exact[a][b][k]))
                    C_float[b, a, k] = -C_float[a, b, k]

                if n_computed % 50 == 0 or n_computed == r*(r-1)//2:
                    print(f"    [{n_computed}/{r*(r-1)//2}] "
                          f"[{time()-t_total:.1f}s]")

                # Checkpoint every 500 brackets
                if sc_ckpt_path and n_computed % 500 == 0:
                    import pickle
                    ckpt = {'C_float': C_float, 'C_exact': C_exact,
                            'n_computed': n_computed}
                    with open(sc_ckpt_path + '.tmp', 'wb') as f:
                        pickle.dump(ckpt, f)
                    os.replace(sc_ckpt_path + '.tmp', sc_ckpt_path)
                    print(f"    [checkpoint saved at {n_computed}]")

        for a in range(r):
            for k in range(r):
                C_exact[a][a][k] = "0"

        print(f"  Structure constants computed in {time()-t_total:.1f}s")
        n_nonzero = int(np.count_nonzero(C_float))
        print(f"  Non-zero entries: {n_nonzero} / {r**3} "
              f"({100*n_nonzero/r**3:.1f}%)")

        return C_float, C_exact


def compute_killing_form(C):
    """Compute the Killing form from structure constants.

    K[i,j] = trace(ad_i @ ad_j) = sum_{k,l} C[i,k,l] * C[j,l,k]

    Returns (K, eigenvalues, signature).
    """
    import numpy as np

    r = C.shape[0]
    K = np.zeros((r, r))
    for i in range(r):
        ad_i = C[i]  # (r, r)
        for j in range(i, r):
            ad_j = C[j]  # (r, r)
            val = np.trace(ad_i @ ad_j.T)
            K[i, j] = val
            K[j, i] = val

    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues.sort()

    tol = 1e-10 * np.max(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 1e-10
    n_pos = int(np.sum(eigenvalues > tol))
    n_neg = int(np.sum(eigenvalues < -tol))
    n_zero = int(np.sum(np.abs(eigenvalues) <= tol))
    signature = (n_pos, n_neg, n_zero)

    return K, eigenvalues, signature


def compute_derived_series(C, max_depth=10):
    """Compute the derived series: L^(0) = L, L^(n+1) = [L^(n), L^(n)].

    Returns (dimensions, is_solvable, solvability_length).
    """
    import numpy as np

    r = C.shape[0]
    current_basis = np.eye(r)
    dimensions = [r]

    for depth in range(1, max_depth + 1):
        dim = current_basis.shape[0]
        if dim == 0:
            break

        brackets = []
        for a in range(dim):
            for b in range(a + 1, dim):
                v_a = current_basis[a]
                v_b = current_basis[b]
                bracket = np.einsum('i,j,ijk->k', v_a, v_b, C)
                if np.linalg.norm(bracket) > 1e-12:
                    brackets.append(bracket)

        if len(brackets) == 0:
            dimensions.append(0)
            return dimensions, True, depth

        bracket_matrix = np.array(brackets)
        _, s_b, Vt_b = np.linalg.svd(bracket_matrix, full_matrices=False)
        tol = 1e-10 * s_b[0] if len(s_b) > 0 else 1e-10
        new_rank = int(np.sum(s_b > tol))

        if new_rank == 0:
            dimensions.append(0)
            return dimensions, True, depth

        current_basis = Vt_b[:new_rank]
        dimensions.append(new_rank)

        if new_rank == dimensions[-2]:
            return dimensions, False, None

    return dimensions, False, None


def compute_lower_central_series(C, max_depth=10):
    """Lower central series: L_0 = L, L_{n+1} = [L, L_n].

    Returns (dimensions, is_nilpotent, nilpotency_class).
    """
    import numpy as np

    r = C.shape[0]
    current_basis = np.eye(r)
    dimensions = [r]

    for depth in range(1, max_depth + 1):
        dim_current = current_basis.shape[0]
        if dim_current == 0:
            break

        brackets = []
        for a in range(r):
            e_a = np.zeros(r)
            e_a[a] = 1.0
            for b in range(dim_current):
                v_b = current_basis[b]
                bracket = np.einsum('i,j,ijk->k', e_a, v_b, C)
                if np.linalg.norm(bracket) > 1e-12:
                    brackets.append(bracket)

        if len(brackets) == 0:
            dimensions.append(0)
            return dimensions, True, depth

        bracket_matrix = np.array(brackets)
        _, s_b, Vt_b = np.linalg.svd(bracket_matrix, full_matrices=False)
        tol = 1e-10 * s_b[0] if len(s_b) > 0 else 1e-10
        new_rank = int(np.sum(s_b > tol))

        if new_rank == 0:
            dimensions.append(0)
            return dimensions, True, depth

        current_basis = Vt_b[:new_rank]
        dimensions.append(new_rank)

        if new_rank == dimensions[-2]:
            return dimensions, False, None

    return dimensions, False, None


def compute_center(C):
    """Find the center Z(L): elements commuting with everything.

    Returns (center_basis, center_dim).
    """
    import numpy as np

    r = C.shape[0]
    # z in Z(L) iff sum_j z[j] * C[j, i, k] = 0 for all i, k
    # Reshape: A[i*r + k, j] = C[j, i, k]
    A = C.transpose(1, 2, 0).reshape(r * r, r)  # (r^2, r)

    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = 1e-10 * s[0] if len(s) > 0 and s[0] > 0 else 1e-10
    null_dim = int(np.sum(s < tol))

    center_basis = Vt[-null_dim:] if null_dim > 0 else np.empty((0, r))
    return center_basis, null_dim


def extract_monomial_stats(poly_list, names, levels, phase_vars,
                           n_q_vars, n_p_vars, n_u_vars):
    """Extract per-generator monomial degree statistics."""
    records = []
    for idx, (monom_dict, name, level) in enumerate(zip(poly_list, names, levels)):
        n_monomials = len(monom_dict)
        max_degree = 0
        q_degrees = []
        p_degrees = []
        u_degrees = []

        for monom_tuple in monom_dict.keys():
            td = sum(monom_tuple)
            max_degree = max(max_degree, td)
            qd = sum(monom_tuple[:n_q_vars])
            pd = sum(monom_tuple[n_q_vars:n_q_vars + n_p_vars])
            ud = sum(monom_tuple[n_q_vars + n_p_vars:])
            q_degrees.append(qd)
            p_degrees.append(pd)
            u_degrees.append(ud)

        records.append({
            "name": name,
            "level": level,
            "n_monomials": n_monomials,
            "max_degree": max_degree,
            "max_q_degree": max(q_degrees) if q_degrees else 0,
            "max_p_degree": max(p_degrees) if p_degrees else 0,
            "max_u_degree": max(u_degrees) if u_degrees else 0,
        })
    return records


def main():
    ap = argparse.ArgumentParser(
        description="Exact algebraic rank for N-body Poisson algebra")
    ap.add_argument("-N", type=int, default=3, help="Number of bodies")
    ap.add_argument("-d", type=int, default=2, help="Spatial dimension")
    ap.add_argument("--potential", default="1/r",
                    help="Potential type (1/r, 1/r^2, 1/r^3, r^4, composite)")
    ap.add_argument("--composite", nargs=2, type=str, action='append',
                    metavar=('COEFF', 'POWER'),
                    help="Composite potential term: coeff power (e.g. -1 4)")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Maximum bracket level")
    ap.add_argument("--structure", action="store_true",
                    help="Compute full algebraic structure (structure constants, "
                         "Killing form, derived series, center)")
    ap.add_argument("--quantum", action="store_true",
                    help="Use quantum commutator [f,g]/(i*hbar) instead of "
                         "Poisson bracket {f,g}")
    ap.add_argument("--output", default=None, help="Output JSON file")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Directory for per-level checkpoints (enables resume)")
    ap.add_argument("--external-potential", nargs=2, action='append',
                    metavar=('KEY', 'VALUE'),
                    help="External potential parameter: key value "
                         "(e.g. --external-potential omega 1). "
                         "Can be repeated for multiple parameters.")
    ap.add_argument("--workers", type=int,
                    default=1,
                    help="Number of multiprocessing workers for bracket "
                         "computation (default: 1, sequential). "
                         "Uses fork on Linux; ignored on Windows.")
    args = ap.parse_args()

    n_workers = args.workers
    if os.name == 'nt' and n_workers > 1:
        print(f"WARNING: --workers={n_workers} ignored on Windows (no fork). "
              f"Using sequential mode.", flush=True)
        n_workers = 1

    signal.signal(signal.SIGTERM, _sigterm_handler)

    mode = "QUANTUM COMMUTATOR" if args.quantum else "POISSON"
    print("=" * 70)
    print(f"N-BODY SYMBOLIC RANK OVER Q ({mode})")
    print("=" * 70)
    print(f"SymPy version: {sp.__version__}")
    print(f"N = {args.N}, d = {args.d}")
    print(f"Potential: {args.potential}")
    print(f"Max level: {args.max_level}")
    print(f"Workers: {n_workers}")
    if args.quantum:
        print(f"Bracket: [f,g]/(i*hbar) (Moyal/Weyl quantization)")
        print(f"Domain: QQ[hbar]")

    t_grand = time()

    potential_params = None
    if args.potential == 'composite' and args.composite:
        potential_params = [(Rational(c), int(p)) for c, p in args.composite]
        print(f"Composite terms: {potential_params}")

    external_potential = None
    if args.external_potential:
        external_potential = {}
        for key, value in args.external_potential:
            external_potential[key] = float(value)
        print(f"External potential: {external_potential}")

    pot_label = args.potential
    if args.potential == 'composite' and potential_params:
        powers = [str(p) for _, p in potential_params]
        pot_label = f"composite_u{'_'.join(powers)}"
    if external_potential:
        ext_tag = "_".join(f"{k}{v}" for k, v in external_potential.items())
        pot_label = f"{pot_label}_ext_{ext_tag}"
    if args.quantum:
        pot_label = f"quantum_{pot_label}"

    engine = NBodySymbolicRank(
        args.N, args.d, args.potential,
        potential_params=potential_params,
        quantum=args.quantum,
        external_potential=external_potential)

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Checkpointing to: {ckpt_dir}")

    # Phase 1: Build generators
    exprs, names, levels = engine.build_generators(args.max_level,
                                                   checkpoint_dir=ckpt_dir,
                                                   n_workers=n_workers)

    # Phase 2: Extract monomial-coefficient matrix
    poly_list, monom_list, monom_to_idx = engine.extract_monomial_matrix(
        exprs, n_workers=n_workers)

    # Phase 3: Exact rank
    print("\n" + "=" * 70)
    print("EXACT RANK COMPUTATION")
    print("=" * 70)

    rank_results = engine.compute_exact_rank(
        poly_list, monom_list, monom_to_idx, levels, checkpoint_dir=ckpt_dir)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    cumulative = [rank_results[lv] for lv in sorted(rank_results)]
    print(f"  Cumulative rank: {cumulative}")
    new_per_level = [cumulative[0]]
    for i in range(1, len(cumulative)):
        new_per_level.append(cumulative[i] - cumulative[i-1])
    print(f"  New per level:   {new_per_level}")
    print(f"  Total generators: {len(exprs)}")
    print(f"  Total monomials:  {len(monom_list)}")
    total_time = time() - t_grand
    print(f"  Total time: {total_time:.1f}s")

    stabilized = len(cumulative) >= 2 and cumulative[-1] == cumulative[-2]
    if stabilized:
        print(f"\n  STABILIZED at dim = {cumulative[-1]}")
    else:
        print(f"\n  GROWING (not stabilized through level {args.max_level})")

    # Save results
    output = {
        "N": args.N,
        "d": args.d,
        "potential": args.potential,
        "potential_label": pot_label,
        "max_level": args.max_level,
        "quantum": args.quantum,
        "bracket_type": "commutator/(i*hbar)" if args.quantum else "Poisson",
        "n_generators": len(exprs),
        "n_monomials": len(monom_list),
        "cumulative_rank": cumulative,
        "new_per_level": new_per_level,
        "matrix_density": sum(len(d) for d in poly_list)
            / (len(exprs) * len(monom_list))
            if len(exprs) * len(monom_list) > 0 else 0,
        "computation_time_seconds": round(total_time, 1),
        "sympy_version": sp.__version__,
        "stabilized": stabilized,
    }

    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "results", "symbolic_rank")
        os.makedirs(out_dir, exist_ok=True)
        pot_tag = pot_label.replace("/", "").replace("^", "")
        out_path = os.path.join(out_dir, f"rank_N{args.N}_d{args.d}_{pot_tag}.json")

    # Phase 4: Structure extraction (optional)
    if args.structure:
        import numpy as np

        final_rank = cumulative[-1]

        basis_indices = engine.select_basis(
            poly_list, monom_list, monom_to_idx, final_rank)

        C_float, C_exact = engine.compute_structure_constants(
            exprs, names, levels, poly_list, monom_list, monom_to_idx,
            basis_indices, checkpoint_dir=ckpt_dir)

        # Killing form
        print(f"\n{'='*70}")
        print("KILLING FORM")
        print(f"{'='*70}")
        K, k_eigs, signature = compute_killing_form(C_float)
        print(f"  Signature: ({signature[0]}+, {signature[1]}-, "
              f"{signature[2]} zero)")
        print(f"  Semisimple: {signature[2] == 0}")
        print(f"  Trace: {np.trace(K):.6g}")
        if signature[0] + signature[1] > 0:
            print(f"  Max eigenvalue: {k_eigs[-1]:.6g}")
            print(f"  Min non-zero eigenvalue: "
                  f"{k_eigs[signature[2]] if signature[2] < len(k_eigs) else 'N/A'}")

        # Derived series
        print(f"\n{'='*70}")
        print("DERIVED SERIES")
        print(f"{'='*70}")
        derived_dims, is_solvable, solv_len = compute_derived_series(C_float)
        print(f"  Dimensions: {derived_dims}")
        print(f"  Solvable: {is_solvable}")
        if is_solvable:
            print(f"  Solvability length: {solv_len}")

        # Lower central series
        print(f"\n{'='*70}")
        print("LOWER CENTRAL SERIES")
        print(f"{'='*70}")
        lcs_dims, is_nilpotent, nilp_class = compute_lower_central_series(C_float)
        print(f"  Dimensions: {lcs_dims}")
        print(f"  Nilpotent: {is_nilpotent}")
        if is_nilpotent:
            print(f"  Nilpotency class: {nilp_class}")

        # Center
        print(f"\n{'='*70}")
        print("CENTER")
        print(f"{'='*70}")
        center_basis, center_dim = compute_center(C_float)
        print(f"  Center dimension: {center_dim}")

        # Monomial structure
        n_q = len(engine.phase_vars) // 2 if not engine.uses_u else len(engine.algebra.q_vars)
        n_p = n_q
        n_u = len(engine.algebra.u_vars) if engine.uses_u else 0
        monomial_stats = extract_monomial_stats(
            poly_list, names, levels, engine.phase_vars, n_q, n_p, n_u)

        level_monomial_summary = {}
        for lv in range(max(levels) + 1):
            lv_records = [r for r in monomial_stats if r["level"] == lv]
            if lv_records:
                level_monomial_summary[lv] = {
                    "count": len(lv_records),
                    "avg_monomials": sum(r["n_monomials"] for r in lv_records) / len(lv_records),
                    "max_degree": max(r["max_degree"] for r in lv_records),
                }

        print(f"\n{'='*70}")
        print("MONOMIAL STRUCTURE")
        print(f"{'='*70}")
        for lv, stats in sorted(level_monomial_summary.items()):
            print(f"  Level {lv}: {stats['count']} generators, "
                  f"avg {stats['avg_monomials']:.0f} monomials, "
                  f"max degree {stats['max_degree']}")

        # Add structure data to output
        output["structure"] = {
            "basis_indices": basis_indices,
            "killing_signature": list(signature),
            "killing_trace": float(np.trace(K)),
            "is_semisimple": bool(signature[2] == 0),
            "derived_series": derived_dims,
            "is_solvable": is_solvable,
            "solvability_length": solv_len,
            "lower_central_series": lcs_dims,
            "is_nilpotent": is_nilpotent,
            "nilpotency_class": nilp_class,
            "center_dimension": center_dim,
            "monomial_summary": {str(k): v for k, v in level_monomial_summary.items()},
        }

        # Save structure constants and Killing form to separate files
        struct_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "results", "algebra_structure",
            f"N{args.N}_d{args.d}_{pot_tag}")
        os.makedirs(struct_dir, exist_ok=True)

        np.save(os.path.join(struct_dir, "structure_constants.npy"), C_float)
        np.save(os.path.join(struct_dir, "killing_form.npy"), K)
        np.save(os.path.join(struct_dir, "killing_eigenvalues.npy"), k_eigs)
        if center_dim > 0:
            np.save(os.path.join(struct_dir, "center_basis.npy"), center_basis)

        with open(os.path.join(struct_dir, "structure_constants_exact.json"), "w") as f:
            json.dump(C_exact, f)

        print(f"\n  Structure data saved to {struct_dir}/")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
