#!/usr/bin/env python3
# Track: 3D Extension | Spatial three-body Poisson algebra
# Parent project: ../preprint.tex (planar/2D results)
# See README.md in this directory for context.
"""
N-Dimensional Poisson Algebra Growth Engine
============================================

Parameterized version of the planar engine (../exact_growth.py)
supporting arbitrary spatial dimension d = 1, 2, or 3.

The polynomial trick u_ij = 1/r_ij works identically in all
dimensions -- only the chain rule table grows with d.

The planar case (d=2) reproduces the published sequence
[3, 6, 17, 116].  The spatial case (d=3) is the primary target
of this track.

Usage
-----
    python exact_growth_nd.py -d 3 --max-level 2
    python exact_growth_nd.py -d 2 --max-level 3   # validate
    python exact_growth_nd.py -d 1 --max-level 3   # bonus
"""

import os
import sys
import argparse
import pickle
import numpy as np
from time import time

import sympy as sp
from sympy import Symbol, symbols, diff, Integer, cancel, expand

os.environ["PYTHONUNBUFFERED"] = "1"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COORD_LABELS = {1: ["x"], 2: ["x", "y"], 3: ["x", "y", "z"]}


class ThreeBodyAlgebra:
    """Poisson algebra engine for the 3-body problem in d spatial dimensions.

    Parameters
    ----------
    d_spatial : int
        Spatial dimension (1, 2, or 3).
    masses : dict, optional
        Masses {1: m1, 2: m2, 3: m3}.  Defaults to equal unit masses.
    checkpoint_dir : str, optional
        Directory for checkpoint files.  Defaults to checkpoints_dN/
        inside the script directory.
    """

    def __init__(self, d_spatial=3, masses=None, checkpoint_dir=None):
        if d_spatial not in (1, 2, 3):
            raise ValueError(f"d_spatial must be 1, 2, or 3 (got {d_spatial})")

        self.d = d_spatial
        self.n_q = 3 * d_spatial
        self.n_p = 3 * d_spatial
        self.n_phase = 6 * d_spatial

        default_ckpt = os.path.join(_SCRIPT_DIR, f"checkpoints_d{d_spatial}")
        self.checkpoint_dir = checkpoint_dir or default_ckpt

        self._build_symbols()
        self._build_chain_rule()
        self._build_hamiltonians(masses)

    # -----------------------------------------------------------------
    # Symbol construction
    # -----------------------------------------------------------------

    def _build_symbols(self):
        labels = COORD_LABELS[self.d]

        self.q_vars = []
        self.p_vars = []
        self.q_by_body = {}
        self.p_by_body = {}

        for body in (1, 2, 3):
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

        self.u12, self.u13, self.u23 = symbols("u12 u13 u23", positive=True)
        self.u_vars = [self.u12, self.u13, self.u23]
        self.all_vars = self.q_vars + self.p_vars + self.u_vars

    # -----------------------------------------------------------------
    # Chain rule:  du_ij / dq_k = -(q_k - q_partner) * u_ij^3
    # -----------------------------------------------------------------

    def _build_chain_rule(self):
        self.chain_rule = {}
        pairs = [
            (self.u12, 1, 2),
            (self.u13, 1, 3),
            (self.u23, 2, 3),
        ]
        for u_var, bi, bj in pairs:
            for k in range(self.d):
                qi_k = self.q_by_body[bi][k]
                qj_k = self.q_by_body[bj][k]
                self.chain_rule[(u_var, qi_k)] = -(qi_k - qj_k) * u_var ** 3
                self.chain_rule[(u_var, qj_k)] = -(qj_k - qi_k) * u_var ** 3

    # -----------------------------------------------------------------
    # Hamiltonians:  H_ij = T_i + T_j - m_i m_j u_ij
    # -----------------------------------------------------------------

    def _build_hamiltonians(self, masses=None):
        if masses is None:
            masses = {1: Integer(1), 2: Integer(1), 3: Integer(1)}
        self.masses = masses

        def kinetic(body):
            m = masses[body]
            return sum(p ** 2 for p in self.p_by_body[body]) / (2 * m)

        m1, m2, m3 = masses[1], masses[2], masses[3]
        self.H12 = kinetic(1) + kinetic(2) - m1 * m2 * self.u12
        self.H13 = kinetic(1) + kinetic(3) - m1 * m3 * self.u13
        self.H23 = kinetic(2) + kinetic(3) - m2 * m3 * self.u23

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
    # Pre-computed derivatives for CSE-optimised brackets
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
        """Sample points and compute u_ij = 1/r_ij for each."""
        d = self.d
        n_phase = self.n_phase
        rng = np.random.RandomState(seed)
        pts = np.empty((0, n_phase))

        for _ in range(200):
            bs = max((n - pts.shape[0]) * 5, 256)
            b = np.zeros((bs, n_phase))
            b[:, :3 * d] = rng.uniform(-pos_range, pos_range, (bs, 3 * d))
            b[:, 3 * d:] = rng.uniform(-mom_range, mom_range, (bs, 3 * d))

            pos1 = b[:, 0:d]
            pos2 = b[:, d:2 * d]
            pos3 = b[:, 2 * d:3 * d]

            r12_sq = np.sum((pos1 - pos2) ** 2, axis=1)
            r13_sq = np.sum((pos1 - pos3) ** 2, axis=1)
            r23_sq = np.sum((pos2 - pos3) ** 2, axis=1)

            ok = ((r12_sq > min_sep ** 2) &
                  (r13_sq > min_sep ** 2) &
                  (r23_sq > min_sep ** 2))
            pts = np.vstack([pts, b[ok]])
            if pts.shape[0] >= n:
                break
        pts = pts[:n]

        pos1 = pts[:, 0:d]
        pos2 = pts[:, d:2 * d]
        pos3 = pts[:, 2 * d:3 * d]

        u12_v = 1.0 / np.sqrt(np.sum((pos1 - pos2) ** 2, axis=1))
        u13_v = 1.0 / np.sqrt(np.sum((pos1 - pos3) ** 2, axis=1))
        u23_v = 1.0 / np.sqrt(np.sum((pos2 - pos3) ** 2, axis=1))

        return pts, np.column_stack([u12_v, u13_v, u23_v])

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
        t0 = time()

        if n <= 50:
            print(f"    Lambdifying {n} expressions (standard)...",
                  end=" ", flush=True)
            func = sp.lambdify(self.all_vars, exprs,
                               modules="numpy", cse=True)
            print(f"done [{time() - t0:.1f}s]")

            def evaluate(Z_qp, Z_u):
                args = ([Z_qp[:, i] for i in range(n_phase)] +
                        [Z_u[:, i] for i in range(3)])
                vals = func(*args)
                return np.column_stack(vals)
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
            args = ([Z_qp[:, i] for i in range(n_phase)] +
                    [Z_u[:, i] for i in range(3)])
            cols = []
            for f in funcs:
                val = f(*args)
                cols.append(np.atleast_1d(val).ravel())
            return np.column_stack(cols)
        return evaluate

    # -----------------------------------------------------------------
    # SVD gap analysis (dimension-independent)
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

        ref_indices = {3, 6, 15, 17, 116}
        for i in range(len(s)):
            rel = s[i] / s[0] if s[0] > 0 else 0.0
            if i < len(s) - 1 and s[i + 1] > 1e-300:
                gap = s[i] / s[i + 1]
            else:
                gap = float("inf")

            show = (i < 30 or
                    (best_gap_idx >= 0 and abs(i - best_gap_idx) < 5) or
                    gap > 100.0 or
                    i >= len(s) - 5 or
                    (i + 1) in ref_indices)
            if show:
                marker = ""
                if (i + 1) in ref_indices:
                    marker = f"  <-- {i+1} (2D ref)"
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
        args = ([Z_qp[:, i] for i in range(self.n_phase)] +
                [Z_u[:, i] for i in range(3)])
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
            "d_spatial": self.d,
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
            print(f"    WARNING: checkpoint is for d={data.get('d_spatial')}, "
                  f"skipping (current d={self.d})")
            return None
        print(f"    Loaded checkpoint: {path}  (level {data['level']})")
        return data

    # -----------------------------------------------------------------
    # Main computation
    # -----------------------------------------------------------------

    def compute_growth(self, max_level=3, n_samples=500, seed=42,
                       resume=False):
        d = self.d
        dim_label = {1: "1D (linear)", 2: "2D (planar)", 3: "3D (spatial)"}

        print("=" * 70)
        print(f"POISSON ALGEBRA GROWTH  —  "
              f"{dim_label.get(d, f'{d}D')}")
        print(f"  Polynomial representation: u_ij = 1/r_ij")
        print("=" * 70)
        print(f"  Spatial dimension: d = {d}")
        print(f"  Phase space: {self.n_phase}D  "
              f"({self.n_q} positions + {self.n_p} momenta) "
              f"+ 3 auxiliary u_ij")
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
                        # A pair (i,j) is computed at level max(levels)+1,
                        # so it was already computed if max(levels) < start_level - 1
                        if max(all_levels[i], all_levels[j]) < start_level - 1:
                            computed_pairs.add(frozenset({i, j}))

        # -- Level 0 --
        if start_level <= 0:
            print("--- Level 0: Pairwise Hamiltonians ---")
            for name, expr in [("H12", self.H12), ("H13", self.H13),
                                ("H23", self.H23)]:
                all_exprs.append(expr)
                all_names.append(name)
                all_levels.append(0)
                print(f"  {name}: {len(sp.Add.make_args(expr))} terms")
            for i in range(3):
                for j in range(i + 1, 3):
                    computed_pairs.add(frozenset({i, j}))
            self.save_checkpoint(0, all_exprs, all_names, all_levels)

        # -- Level 1 --
        if start_level <= 1:
            print("\n--- Level 1: Tidal-competition generators ---")
            pairs_l1 = [
                ("K1", "{H12,H13}", 0, 1),
                ("K2", "{H12,H23}", 0, 2),
                ("K3", "{H13,H23}", 1, 2),
            ]
            for short, full, i, j in pairs_l1:
                print(f"  Computing {full}...", end=" ", flush=True)
                t0 = time()
                expr = self.poisson_bracket(all_exprs[i], all_exprs[j])
                expr = self.simplify_generator(expr)
                elapsed = time() - t0
                nterms = len(sp.Add.make_args(expr))
                print(f"{nterms} terms  [{elapsed:.1f}s]")
                all_exprs.append(expr)
                all_names.append(short)
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

        self.verify_jacobi_symbolic(
            all_exprs[0], all_exprs[1], all_exprs[2],
            "H12", "H13", "H23",
        )
        if len(all_exprs) > 5:
            self.verify_jacobi_numerical(
                all_exprs[0], all_exprs[1], all_exprs[3],
                "H12", "H13", "K1",
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
        print(f"DIMENSION SUMMARY  (d = {d})")
        print("=" * 70)

        ref_2d = {0: 3, 1: 6, 2: 17, 3: 116}

        for lv in range(max_level + 1):
            dim = level_dims[lv]
            ref = ref_2d.get(lv, "?")
            if d == 2:
                match_str = "MATCH" if dim == ref else "MISMATCH"
                print(f"  Level {lv}: dim = {dim:>5d}    "
                      f"2D reference = {ref}    [{match_str}]")
            else:
                print(f"  Level {lv}: dim = {dim:>5d}    "
                      f"(2D reference = {ref})")

        seq = [level_dims[lv] for lv in range(max_level + 1)]
        ref_seq = [ref_2d.get(lv, "?") for lv in range(max_level + 1)]
        print(f"\n  Dimension sequence (d={d}): {seq}")
        if d != 2:
            matches = all(
                level_dims[lv] == ref_2d.get(lv)
                for lv in range(max_level + 1)
            )
            if matches:
                print(f"  *** MATCHES 2D sequence {ref_seq} ***")
            else:
                print(f"  *** DIFFERS from 2D sequence {ref_seq} ***")

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
            ax.set_title(f"Poisson Algebra SVD Spectrum (d={d})")
            ax.grid(True, alpha=0.3)
            for pos in (3, 6, 17, 116):
                if pos <= len(svals_full):
                    ax.axvline(x=pos, color="red", ls="--", alpha=0.3)
                    ax.text(pos, 0.5, f" {pos}", color="red", fontsize=8)
            plt.tight_layout()
            fname = os.path.join(_SCRIPT_DIR,
                                 f"svd_spectrum_d{d}.png")
            plt.savefig(fname, dpi=150)
            print(f"\n  SVD plot saved to {fname}")
            plt.close()
        except Exception as e:
            print(f"\n  (Plot skipped: {e})")

        return level_dims


def main():
    ap = argparse.ArgumentParser(
        description="Poisson algebra growth for the d-dimensional "
                    "3-body problem")
    ap.add_argument("-d", "--dim", type=int, default=3,
                    help="Spatial dimension: 1, 2, or 3 (default: 3)")
    ap.add_argument("--max-level", type=int, default=3,
                    help="Maximum bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space sample points (default: 500)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
    args = ap.parse_args()

    alg = ThreeBodyAlgebra(d_spatial=args.dim)
    alg.compute_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
