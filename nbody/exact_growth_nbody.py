#!/usr/bin/env python3
# Track: N-Body Extension | Poisson algebra for arbitrary particle count
# Parent project: ../preprint.tex (planar 3-body results)
"""
N-Body Poisson Algebra Growth Engine
=====================================

Generalized engine supporting arbitrary particle count N, spatial
dimension d, and potential type.  Subsumes the 3-body engine
(3d/exact_growth_nd.py) as the special case N=3.

The polynomial trick u_ij = 1/r_ij works for all N and d -- only the
number of auxiliary variables C(N,2) and coordinate dimensions N*d
change.

Usage
-----
    python exact_growth_nbody.py -N 4 -d 2 --max-level 2
    python exact_growth_nbody.py -N 3 -d 2 --max-level 3   # validate
    python exact_growth_nbody.py -N 3 -d 2 --potential 1/r^3 --max-level 3
"""

import os
import sys
import argparse
import pickle
import numpy as np
from time import time
from itertools import combinations

import sympy as sp
from sympy import Symbol, symbols, diff, Integer, Rational, cancel, expand

os.environ["PYTHONUNBUFFERED"] = "1"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COORD_LABELS = {1: ["x"], 2: ["x", "y"], 3: ["x", "y", "z"]}

VALID_POTENTIALS = ("1/r", "1/r^2", "1/r^3")


class NBodyAlgebra:
    """Poisson algebra engine for the N-body problem in d spatial dimensions.

    Parameters
    ----------
    n_bodies : int
        Number of particles (>= 2).
    d_spatial : int
        Spatial dimension (1, 2, or 3).
    potential : str
        Potential type: '1/r', '1/r^2', or '1/r^3'.
    masses : dict, optional
        Masses {1: m1, 2: m2, ...}.  Defaults to equal unit masses.
    charges : dict, optional
        Charges {1: q1, 2: q2, ...}.  When provided, the potential for
        pair (i,j) is q_i * q_j * u_ij^p (positive = repulsive, negative
        = attractive).  When None, uses -m_i * m_j * u_ij^p (all-attractive).
    checkpoint_dir : str, optional
        Directory for checkpoint files.
    """

    def __init__(self, n_bodies=3, d_spatial=2, potential="1/r",
                 masses=None, charges=None, checkpoint_dir=None):
        if n_bodies < 2:
            raise ValueError(f"n_bodies must be >= 2 (got {n_bodies})")
        if d_spatial not in (1, 2, 3):
            raise ValueError(f"d_spatial must be 1, 2, or 3 (got {d_spatial})")
        if potential not in VALID_POTENTIALS:
            raise ValueError(f"potential must be one of {VALID_POTENTIALS} "
                             f"(got {potential!r})")

        self.N = n_bodies
        self.d = d_spatial
        self.potential = potential
        self.n_q = n_bodies * d_spatial
        self.n_p = n_bodies * d_spatial
        self.n_phase = 2 * n_bodies * d_spatial

        self.body_pairs = list(combinations(range(1, n_bodies + 1), 2))
        self.n_pairs = len(self.body_pairs)
        self.charges = charges

        tag = f"N{n_bodies}_d{d_spatial}_{potential.replace('/', '').replace('^', '')}"
        if charges is not None:
            charge_str = "_".join(f"q{k}{v:+g}" for k, v in sorted(charges.items()))
            tag += f"_{charge_str}"
        default_ckpt = os.path.join(_SCRIPT_DIR, f"checkpoints_{tag}")
        self.checkpoint_dir = checkpoint_dir or default_ckpt

        self._build_symbols()
        self._build_chain_rule()
        self._build_hamiltonians(masses, charges)

    # -----------------------------------------------------------------
    # Symbol construction
    # -----------------------------------------------------------------

    def _build_symbols(self):
        labels = COORD_LABELS[self.d]

        self.q_vars = []
        self.p_vars = []
        self.q_by_body = {}
        self.p_by_body = {}

        for body in range(1, self.N + 1):
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

        self.u_vars = []
        self.u_by_pair = {}
        for bi, bj in self.body_pairs:
            u = Symbol(f"u{bi}{bj}", positive=True)
            self.u_vars.append(u)
            self.u_by_pair[(bi, bj)] = u

        self.all_vars = self.q_vars + self.p_vars + self.u_vars

    # -----------------------------------------------------------------
    # Chain rule:  du_ij / dq_k = -(q_k - q_partner) * u_ij^3
    # -----------------------------------------------------------------

    def _build_chain_rule(self):
        self.chain_rule = {}
        for (bi, bj), u_var in zip(self.body_pairs, self.u_vars):
            for k in range(self.d):
                qi_k = self.q_by_body[bi][k]
                qj_k = self.q_by_body[bj][k]
                self.chain_rule[(u_var, qi_k)] = -(qi_k - qj_k) * u_var ** 3
                self.chain_rule[(u_var, qj_k)] = -(qj_k - qi_k) * u_var ** 3

    # -----------------------------------------------------------------
    # Hamiltonians:  H_ij = T_i + T_j + V(u_ij)
    # -----------------------------------------------------------------

    def _build_hamiltonians(self, masses=None, charges=None):
        if masses is None:
            masses = {b: Integer(1) for b in range(1, self.N + 1)}
        self.masses = masses

        def kinetic(body):
            m = masses[body]
            return sum(p ** 2 for p in self.p_by_body[body]) / (2 * m)

        pot_power = {"1/r": 1, "1/r^2": 2, "1/r^3": 3}[self.potential]

        self.hamiltonians = {}
        self.hamiltonian_list = []
        self.hamiltonian_names = []

        for bi, bj in self.body_pairs:
            u = self.u_by_pair[(bi, bj)]
            mi, mj = masses[bi], masses[bj]
            if charges is not None:
                qi = Integer(charges[bi]) if isinstance(charges[bi], int) else charges[bi]
                qj = Integer(charges[bj]) if isinstance(charges[bj], int) else charges[bj]
                H = kinetic(bi) + kinetic(bj) + qi * qj * u ** pot_power
            else:
                H = kinetic(bi) + kinetic(bj) - mi * mj * u ** pot_power
            name = f"H{bi}{bj}"
            self.hamiltonians[name] = H
            self.hamiltonian_list.append(H)
            self.hamiltonian_names.append(name)

    # -----------------------------------------------------------------
    # Derivatives and Poisson bracket
    # -----------------------------------------------------------------

    def total_deriv(self, expr, var):
        """df/dvar with chain rule through u_ij for position variables."""
        result = diff(expr, var)
        if var in self.p_vars:
            return result
        for u_var in self.u_vars:
            key = (u_var, var)
            if key in self.chain_rule:
                df_du = diff(expr, u_var)
                if df_du != 0:
                    result += df_du * self.chain_rule[key]
        return result

    def poisson_bracket(self, f, g):
        """{f, g} = sum_k (df/dq_k dg/dp_k - df/dp_k dg/dq_k)."""
        result = Integer(0)
        for q, p in zip(self.q_vars, self.p_vars):
            df_dq = self.total_deriv(f, q)
            dg_dp = diff(g, p)
            df_dp = diff(f, p)
            dg_dq = self.total_deriv(g, q)
            result += df_dq * dg_dp - df_dp * dg_dq
        return result

    def simplify_generator(self, expr):
        return cancel(expr)

    # -----------------------------------------------------------------
    # Pre-computed derivatives for faster brackets
    # -----------------------------------------------------------------

    def precompute_derivatives(self, exprs, names=None):
        n = len(exprs)
        all_derivs = []
        t0 = time()

        print(f"  Pre-computing derivatives for {n} expressions "
              f"({2 * self.n_q} derivs each)...")
        for idx, expr in enumerate(exprs):
            derivs = {"dq": [], "dp": []}
            for q in self.q_vars:
                derivs["dq"].append(expand(self.total_deriv(expr, q)))
            for p in self.p_vars:
                derivs["dp"].append(expand(diff(expr, p)))
            all_derivs.append(derivs)
            if (idx + 1) % 10 == 0 or idx == n - 1:
                name = names[idx] if names else f"expr_{idx}"
                print(f"    {idx+1}/{n}  [{time()-t0:.1f}s]  "
                      f"last: {name}", flush=True)

        total_terms = sum(
            sum(len(sp.Add.make_args(d)) for d in dd["dq"])
            + sum(len(sp.Add.make_args(d)) for d in dd["dp"])
            for dd in all_derivs
        )
        elapsed = time() - t0
        print(f"  Derivatives done: {n * 2 * self.n_q} derivs, "
              f"{total_terms} total terms  [{elapsed:.1f}s]")
        return all_derivs

    def poisson_bracket_from_derivs(self, derivs_f, derivs_g):
        """Compute {f, g} from pre-computed derivative arrays."""
        result = Integer(0)
        for k in range(self.n_q):
            result += (derivs_f["dq"][k] * derivs_g["dp"][k]
                       - derivs_f["dp"][k] * derivs_g["dq"][k])
        return result

    # -----------------------------------------------------------------
    # Phase-space sampling
    # -----------------------------------------------------------------

    def sample_phase_space(self, n, seed=42, pos_range=3.0, mom_range=1.0,
                           min_sep=0.5):
        """Sample n phase-space points with all pairwise separations > min_sep."""
        N, d = self.N, self.d
        n_phase = self.n_phase
        rng = np.random.RandomState(seed)
        pts = np.empty((0, n_phase))

        for _ in range(200):
            bs = max((n - pts.shape[0]) * 5, 256)
            b = np.zeros((bs, n_phase))
            b[:, :N * d] = rng.uniform(-pos_range, pos_range, (bs, N * d))
            b[:, N * d:] = rng.uniform(-mom_range, mom_range, (bs, N * d))

            ok = np.ones(bs, dtype=bool)
            for bi, bj in self.body_pairs:
                si = (bi - 1) * d
                sj = (bj - 1) * d
                r_sq = np.sum((b[:, si:si+d] - b[:, sj:sj+d]) ** 2, axis=1)
                ok &= (r_sq > min_sep ** 2)

            pts = np.vstack([pts, b[ok]])
            if pts.shape[0] >= n:
                break
        pts = pts[:n]

        u_cols = []
        for bi, bj in self.body_pairs:
            si = (bi - 1) * d
            sj = (bj - 1) * d
            r = np.sqrt(np.sum((pts[:, si:si+d] - pts[:, sj:sj+d]) ** 2,
                               axis=1))
            u_cols.append(1.0 / r)

        return pts, np.column_stack(u_cols)

    # -----------------------------------------------------------------
    # Lambdify
    # -----------------------------------------------------------------

    def _expr_to_chunked_lines(self, expr, target_var, indent="    ",
                               max_terms_per_line=50):
        terms = sp.Add.make_args(expr)
        if len(terms) <= max_terms_per_line:
            return [f"{indent}{target_var} = {sp.pycode(expr)}"]
        lines = [f"{indent}{target_var} = 0"]
        for i in range(0, len(terms), max_terms_per_line):
            chunk_expr = sp.Add(*terms[i:i + max_terms_per_line])
            lines.append(
                f"{indent}{target_var} += {sp.pycode(chunk_expr)}")
        return lines

    def _make_flat_func(self, expr, func_name="_f"):
        replacements, reduced = sp.cse([expr])
        var_str = ", ".join(str(v) for v in self.all_vars)
        lines = [f"def {func_name}({var_str}):"]
        for sym, sub_expr in replacements:
            lines.extend(
                self._expr_to_chunked_lines(sub_expr, str(sym)))
        lines.extend(
            self._expr_to_chunked_lines(reduced[0], "_result"))
        lines.append("    return _result")
        code = "\n".join(lines)
        namespace = {"sqrt": np.sqrt, "math": __import__("math")}
        exec(compile(code, "<generated>", "exec"), namespace)
        return namespace[func_name]

    def lambdify_generators(self, exprs):
        n = len(exprs)
        n_phase = self.n_phase
        n_u = self.n_pairs
        t0 = time()

        if n <= 50:
            print(f"    Lambdifying {n} expressions (standard)...",
                  end=" ", flush=True)
            func = sp.lambdify(self.all_vars, exprs,
                               modules="numpy", cse=True)
            print(f"done [{time() - t0:.1f}s]")

            def evaluate(Z_qp, Z_u):
                n_pts = Z_qp.shape[0]
                args = ([Z_qp[:, i] for i in range(n_phase)] +
                        [Z_u[:, i] for i in range(n_u)])
                vals = func(*args)
                cols = []
                for v in vals:
                    arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
                    if arr.shape[0] == 1:
                        arr = np.full(n_pts, arr[0])
                    cols.append(arr)
                return np.column_stack(cols)
            return evaluate

        print(f"    Lambdifying {n} expressions individually...",
              flush=True)
        funcs = []
        for idx, expr in enumerate(exprs):
            if (idx + 1) % 20 == 0 or idx == n - 1:
                print(f"      {idx+1}/{n}  [{time()-t0:.1f}s]",
                      flush=True)
            try:
                f = sp.lambdify(self.all_vars, expr,
                                modules="numpy", cse=True)
            except RecursionError:
                f = self._make_flat_func(expr, f"_f{idx}")
            funcs.append(f)

        print(f"    Total lambdify time: {time() - t0:.1f}s")

        def evaluate(Z_qp, Z_u):
            n_pts = Z_qp.shape[0]
            args = ([Z_qp[:, i] for i in range(n_phase)] +
                    [Z_u[:, i] for i in range(n_u)])
            cols = []
            for f in funcs:
                val = f(*args)
                arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
                if arr.shape[0] == 1:
                    arr = np.full(n_pts, arr[0])
                cols.append(arr)
            return np.column_stack(cols)
        return evaluate

    # -----------------------------------------------------------------
    # SVD gap analysis
    # -----------------------------------------------------------------

    def svd_gap_analysis(self, eval_matrix, label=""):
        norms = np.linalg.norm(eval_matrix, axis=0)
        norms[norms < 1e-15] = 1.0
        M = eval_matrix / norms

        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        noise_threshold = 1e-8 * s[0]
        n_meaningful = int(np.sum(s > noise_threshold))

        best_gap_ratio = 1.0
        best_gap_idx = -1
        for i in range(min(n_meaningful, len(s) - 1)):
            if s[i + 1] > noise_threshold:
                gap = s[i] / s[i + 1]
            else:
                gap = s[i] / max(s[i + 1], 1e-300)
            if gap > best_gap_ratio and i >= 2:
                best_gap_ratio = gap
                best_gap_idx = i

        print(f"\n  SVD SPECTRUM  {label}")
        print(f"  {'Idx':>5} | {'Singular Value':>18} | {'Gap Ratio':>12} "
              f"| {'Rel to Max':>12}")
        print(f"  {'-'*5}-+-{'-'*18}-+-{'-'*12}-+-{'-'*12}")

        for i in range(len(s)):
            rel = s[i] / s[0] if s[0] > 0 else 0.0
            if i < len(s) - 1 and s[i + 1] > 1e-300:
                gap = s[i] / s[i + 1]
            else:
                gap = float("inf")

            show = (i < 30 or
                    (best_gap_idx >= 0 and abs(i - best_gap_idx) < 5) or
                    gap > 100.0 or
                    i >= len(s) - 5)
            if show:
                marker = ""
                if i == best_gap_idx:
                    marker += f"  *** GAP ({gap:.1e}) ***"
                elif gap > 100 and i != best_gap_idx:
                    marker += f"  (gap {gap:.1e})"
                print(f"  {i+1:>5} | {s[i]:>18.12f} | {gap:>12.2f} "
                      f"| {rel:>12.2e}{marker}")

        if best_gap_ratio > 1e4:
            rank = best_gap_idx + 1
            print(f"\n  DEFINITIVE GAP at index {rank}: "
                  f"ratio {best_gap_ratio:.2e}x")
            print(f"    sv({rank}) = {s[rank-1]:.6e},  "
                  f"sv({rank+1}) = {s[rank]:.6e}")
        elif best_gap_ratio > 10:
            rank = best_gap_idx + 1
            print(f"\n  Gap at index {rank}: ratio {best_gap_ratio:.1f}x "
                  f"(moderate)")
        else:
            rank = n_meaningful
            print(f"\n  No clear gap found "
                  f"(best ratio {best_gap_ratio:.1f}x)")
            print(f"  Using noise-floor threshold: rank = {rank}")

        return rank, s

    # -----------------------------------------------------------------
    # Jacobi identity verification
    # -----------------------------------------------------------------

    def verify_jacobi_symbolic(self, a, b, c, a_name, b_name, c_name):
        print(f"    Jacobi ({a_name}, {b_name}, {c_name}) "
              f"[symbolic]...", end=" ", flush=True)
        t0 = time()
        bc = self.poisson_bracket(b, c)
        ca = self.poisson_bracket(c, a)
        ab = self.poisson_bracket(a, b)
        j1 = self.poisson_bracket(a, bc)
        j2 = self.poisson_bracket(b, ca)
        j3 = self.poisson_bracket(c, ab)
        result = cancel(j1 + j2 + j3)
        ok = result == 0
        elapsed = time() - t0
        status = ("EXACT ZERO" if ok
                  else f"NONZERO ({len(sp.Add.make_args(result))} terms)")
        print(f"{status}  [{elapsed:.1f}s]")
        return ok

    def verify_jacobi_numerical(self, a, b, c, a_name, b_name, c_name,
                                n_pts=20, seed=123):
        print(f"    Jacobi ({a_name}, {b_name}, {c_name}) "
              f"[numerical]...", end=" ", flush=True)
        t0 = time()
        bc = self.poisson_bracket(b, c)
        ca = self.poisson_bracket(c, a)
        ab = self.poisson_bracket(a, b)
        print(f"inner [{time()-t0:.1f}s]...", end=" ", flush=True)

        j1 = self.poisson_bracket(a, bc)
        j2 = self.poisson_bracket(b, ca)
        j3 = self.poisson_bracket(c, ab)
        print(f"outer [{time()-t0:.1f}s]...", end=" ", flush=True)

        total = j1 + j2 + j3
        f = sp.lambdify(self.all_vars, total, modules="numpy")
        Z_qp, Z_u = self.sample_phase_space(n_pts, seed)
        n_u = self.n_pairs
        args = ([Z_qp[:, i] for i in range(self.n_phase)] +
                [Z_u[:, i] for i in range(n_u)])
        vals = np.array(f(*args))
        max_err = np.max(np.abs(vals))
        elapsed = time() - t0

        ok = max_err < 1e-10
        status = f"max |err| = {max_err:.2e}" + (" OK" if ok else " FAIL")
        print(f"{status}  [{elapsed:.1f}s]")
        return ok

    # -----------------------------------------------------------------
    # Checkpoints
    # -----------------------------------------------------------------

    def save_checkpoint(self, level, all_exprs, all_names, all_levels):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"level_{level}.pkl")
        data = {
            "level": level,
            "n_bodies": self.N,
            "d_spatial": self.d,
            "potential": self.potential,
            "charges": self.charges,
            "exprs": all_exprs,
            "names": all_names,
            "levels": all_levels,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    Checkpoint saved: {path}")

    def load_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        files = sorted(
            f for f in os.listdir(self.checkpoint_dir)
            if f.endswith(".pkl")
        )
        if not files:
            return None
        path = os.path.join(self.checkpoint_dir, files[-1])
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if data.get("d_spatial") != self.d:
            print(f"    WARNING: checkpoint d={data.get('d_spatial')} "
                  f"!= current d={self.d}, skipping")
            return None
        if data.get("n_bodies") != self.N:
            print(f"    WARNING: checkpoint N={data.get('n_bodies')} "
                  f"!= current N={self.N}, skipping")
            return None
        if data.get("potential") != self.potential:
            print(f"    WARNING: checkpoint potential={data.get('potential')!r} "
                  f"!= current {self.potential!r}, skipping")
            return None
        print(f"    Loaded checkpoint: {path}  (level {data['level']})")
        return data

    # -----------------------------------------------------------------
    # Main computation
    # -----------------------------------------------------------------

    def compute_growth(self, max_level=3, n_samples=500, seed=42,
                       resume=False):
        N, d = self.N, self.d
        n_pairs = self.n_pairs
        dim_label = {1: "1D (linear)", 2: "2D (planar)", 3: "3D (spatial)"}

        print("=" * 70)
        print(f"POISSON ALGEBRA GROWTH  —  "
              f"N={N} bodies, {dim_label.get(d, f'{d}D')}, "
              f"V = {self.potential}")
        print(f"  Polynomial representation: u_ij = 1/r_ij")
        print("=" * 70)
        print(f"  Bodies: {N},  Pairs: {n_pairs}")
        print(f"  Spatial dimension: d = {d}")
        print(f"  Phase space: {self.n_phase}D  "
              f"({self.n_q} positions + {self.n_p} momenta) "
              f"+ {n_pairs} auxiliary u_ij")
        print(f"  Potential: {self.potential}")
        if self.charges is not None:
            print(f"  Charges: {self.charges}")
            for bi, bj in self.body_pairs:
                qi, qj = self.charges[bi], self.charges[bj]
                sign = "repulsive" if qi * qj > 0 else "attractive"
                print(f"    ({bi},{bj}): q{bi}*q{bj} = {qi*qj:+g}  ({sign})")
        print(f"  Symbols: q = {self.q_vars}")
        print(f"           p = {self.p_vars}")
        print(f"           u = {self.u_vars}")
        print(f"  Max level: {max_level},  "
              f"Samples: {n_samples},  Seed: {seed}")
        print()

        start_level = 0
        all_exprs = []
        all_names = []
        all_levels = []
        computed_pairs = set()

        if resume:
            ckpt = self.load_checkpoint()
            if ckpt is not None:
                all_exprs = ckpt["exprs"]
                all_names = ckpt["names"]
                all_levels = ckpt["levels"]
                start_level = ckpt["level"] + 1
                for i in range(len(all_exprs)):
                    for j in range(i + 1, len(all_exprs)):
                        if max(all_levels[i], all_levels[j]) < start_level - 1:
                            computed_pairs.add(frozenset({i, j}))

        # -- Level 0: pairwise Hamiltonians --
        if start_level <= 0:
            print(f"--- Level 0: {n_pairs} Pairwise Hamiltonians ---")
            for name, expr in zip(self.hamiltonian_names,
                                  self.hamiltonian_list):
                all_exprs.append(expr)
                all_names.append(name)
                all_levels.append(0)
                print(f"  {name}: {len(sp.Add.make_args(expr))} terms")
            for i in range(n_pairs):
                for j in range(i + 1, n_pairs):
                    computed_pairs.add(frozenset({i, j}))
            self.save_checkpoint(0, all_exprs, all_names, all_levels)

        # -- Level 1: all pairwise brackets of level-0 generators --
        if start_level <= 1:
            n_l0 = n_pairs
            print(f"\n--- Level 1: Brackets of {n_l0} Hamiltonians ---")
            for i in range(n_l0):
                for j in range(i + 1, n_l0):
                    ni, nj = all_names[i], all_names[j]
                    bracket_name = f"{{{ni},{nj}}}"
                    print(f"  Computing {bracket_name}...",
                          end=" ", flush=True)
                    t0 = time()
                    expr = self.poisson_bracket(all_exprs[i], all_exprs[j])
                    expr = self.simplify_generator(expr)
                    elapsed = time() - t0
                    nterms = len(sp.Add.make_args(expr))
                    print(f"{nterms} terms  [{elapsed:.1f}s]")
                    all_exprs.append(expr)
                    all_names.append(bracket_name)
                    all_levels.append(1)
            self.save_checkpoint(1, all_exprs, all_names, all_levels)

        # -- Levels 2+ --
        for level in range(max(2, start_level), max_level + 1):
            print(f"\n--- Level {level} ---")
            t_level = time()

            frontier_indices = [
                i for i, lv in enumerate(all_levels) if lv == level - 1
            ]
            n_existing = len(all_exprs)
            n_candidates = 0
            new_exprs = []
            new_names = []

            for i in frontier_indices:
                for j in range(n_existing):
                    if i == j:
                        continue
                    pair = frozenset({i, j})
                    if pair in computed_pairs:
                        continue
                    computed_pairs.add(pair)

                    n_candidates += 1
                    ni, nj = all_names[i], all_names[j]
                    bracket_name = f"{{{ni},{nj}}}"

                    print(f"  [{n_candidates:>4d}] {bracket_name}...",
                          end=" ", flush=True)
                    t0 = time()
                    expr = self.poisson_bracket(
                        all_exprs[i], all_exprs[j])
                    t_bracket = time() - t0
                    print(f"bracket {t_bracket:.1f}s...",
                          end=" ", flush=True)

                    t0s = time()
                    expr = self.simplify_generator(expr)
                    t_simp = time() - t0s
                    nterms = len(sp.Add.make_args(expr))
                    print(f"simplify {t_simp:.1f}s  -> {nterms} terms")

                    new_exprs.append(expr)
                    new_names.append(bracket_name)

            for expr, name in zip(new_exprs, new_names):
                all_exprs.append(expr)
                all_names.append(name)
                all_levels.append(level)

            elapsed_level = time() - t_level
            print(f"\n  Level {level}: {len(new_exprs)} candidates "
                  f"computed in {elapsed_level:.1f}s")
            self.save_checkpoint(level, all_exprs, all_names, all_levels)

        # -- Jacobi identity --
        print("\n" + "=" * 70)
        print("JACOBI IDENTITY VERIFICATION")
        print("=" * 70)

        if len(all_exprs) >= 3:
            self.verify_jacobi_symbolic(
                all_exprs[0], all_exprs[1], all_exprs[2],
                all_names[0], all_names[1], all_names[2],
            )
        if len(all_exprs) > n_pairs + 2:
            self.verify_jacobi_numerical(
                all_exprs[0], all_exprs[1], all_exprs[n_pairs],
                all_names[0], all_names[1], all_names[n_pairs],
            )

        # -- Numerical evaluation and SVD --
        print("\n" + "=" * 70)
        print("NUMERICAL EVALUATION AND SVD ANALYSIS")
        print("=" * 70)

        Z_qp, Z_u = self.sample_phase_space(n_samples, seed)
        print(f"  Sample points: {Z_qp.shape[0]}")

        evaluate = self.lambdify_generators(all_exprs)
        print("    Evaluating at sample points...", end=" ", flush=True)
        t0 = time()
        eval_matrix = evaluate(Z_qp, Z_u)
        print(f"done [{time() - t0:.1f}s]")
        print(f"    Evaluation matrix shape: {eval_matrix.shape}")

        level_dims = {}
        for lv in range(max_level + 1):
            mask = [i for i, l in enumerate(all_levels) if l <= lv]
            sub = eval_matrix[:, mask]
            rank, svals = self.svd_gap_analysis(
                sub, label=f"(through level {lv})")
            level_dims[lv] = rank
            print(f"  ==> Dimension through level {lv}: {rank}")

        rank_full, svals_full = self.svd_gap_analysis(
            eval_matrix, label="(ALL generators)")

        # -- Summary --
        print("\n" + "=" * 70)
        print(f"DIMENSION SUMMARY  (N={N}, d={d}, V={self.potential})")
        print("=" * 70)

        ref_n3 = {0: 3, 1: 6, 2: 17, 3: 116}

        for lv in range(max_level + 1):
            dim = level_dims[lv]
            if N == 3:
                ref = ref_n3.get(lv, "?")
                match_str = ("MATCH" if dim == ref else "MISMATCH"
                             if isinstance(ref, int) else "")
                print(f"  Level {lv}: dim = {dim:>5d}    "
                      f"N=3 reference = {ref}    [{match_str}]")
            else:
                print(f"  Level {lv}: dim = {dim:>5d}")

        seq = [level_dims[lv] for lv in range(max_level + 1)]
        print(f"\n  Dimension sequence: {seq}")

        # SVD plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogy(range(1, len(svals_full) + 1),
                        svals_full / svals_full[0],
                        "o-", markersize=3)
            ax.set_xlabel("Index")
            ax.set_ylabel("Singular value (relative to max)")
            ax.set_title(f"Poisson Algebra SVD Spectrum "
                         f"(N={N}, d={d}, V={self.potential})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = os.path.join(_SCRIPT_DIR,
                                 f"svd_spectrum_N{N}_d{d}.png")
            plt.savefig(fname, dpi=150)
            print(f"\n  SVD plot saved to {fname}")
            plt.close()
        except Exception as e:
            print(f"\n  (Plot skipped: {e})")

        return level_dims


def main():
    ap = argparse.ArgumentParser(
        description="Poisson algebra growth for the N-body problem")
    ap.add_argument("-N", "--bodies", type=int, default=3,
                    help="Number of bodies (default: 3)")
    ap.add_argument("-d", "--dim", type=int, default=2,
                    help="Spatial dimension: 1, 2, or 3 (default: 2)")
    ap.add_argument("--potential", type=str, default="1/r",
                    choices=VALID_POTENTIALS,
                    help="Potential type (default: 1/r)")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Maximum bracket level (default: 2)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space sample points (default: 500)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
    args = ap.parse_args()

    alg = NBodyAlgebra(
        n_bodies=args.bodies,
        d_spatial=args.dim,
        potential=args.potential,
    )
    alg.compute_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
