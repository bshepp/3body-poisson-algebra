#!/usr/bin/env python3
"""
Exact Lie Algebra Growth via Symbolic Computation
===================================================

Uses SymPy for EXACT Poisson brackets with a key performance
optimisation: auxiliary symbols for inverse distances (u_ij = 1/r_ij)
keep all expressions polynomial, making differentiation and
simplification fast.

The chain rule  df/dx = ∂f/∂x + Σ (∂f/∂u_ij)(du_ij/dx)  is handled
explicitly.  At evaluation time, u_ij is substituted with 1/r_ij.

Usage
-----
    python exact_growth.py                  # levels 0-3 (default)
    python exact_growth.py --max-level 2    # quick test
    python exact_growth.py --resume         # resume from checkpoint
"""

import os
import sys
import argparse
import pickle
import numpy as np
from time import time

sys.setrecursionlimit(100000)

import sympy as sp
from sympy import symbols, diff, Integer, cancel, Rational, expand

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

# =====================================================================
# Phase-space variables  (planar 3-body, equal masses, G = 1)
# =====================================================================
# Positions
x1, y1, x2, y2, x3, y3 = symbols("x1 y1 x2 y2 x3 y3", real=True)
# Momenta
px1, py1, px2, py2, px3, py3 = symbols(
    "px1 py1 px2 py2 px3 py3", real=True
)

# Auxiliary symbols for inverse distances:  u_ij = 1 / r_ij
# These are treated as INDEPENDENT variables in symbolic computation.
# The chain rule is applied explicitly in poisson_bracket().
u12, u13, u23 = symbols("u12 u13 u23", positive=True)

Q_VARS = [x1, y1, x2, y2, x3, y3]
P_VARS = [px1, py1, px2, py2, px3, py3]
U_VARS = [u12, u13, u23]

# For lambdify: ALL_VARS order is q's, p's, u's
ALL_VARS = Q_VARS + P_VARS + U_VARS

CHECKPOINT_DIR = "checkpoints"


# =====================================================================
# Chain-rule data for du_ij / dq_k
#
# u_ij = 1/r_ij = ((qi_x - qj_x)^2 + (qi_y - qj_y)^2)^(-1/2)
# du_ij/dq_k = -(q_k_component - q_k_partner) * u_ij^3
#              only if body k is one of i, j (0 otherwise)
# =====================================================================
def _build_chain_rule_table():
    """
    Returns a dict:  du_var / dq_var -> SymPy expression
    (polynomial in positions and u's, no sqrt).
    """
    table = {}
    # u12: depends on x1,y1,x2,y2
    table[(u12, x1)] = -(x1 - x2) * u12 ** 3
    table[(u12, y1)] = -(y1 - y2) * u12 ** 3
    table[(u12, x2)] = -(x2 - x1) * u12 ** 3
    table[(u12, y2)] = -(y2 - y1) * u12 ** 3
    # u13: depends on x1,y1,x3,y3
    table[(u13, x1)] = -(x1 - x3) * u13 ** 3
    table[(u13, y1)] = -(y1 - y3) * u13 ** 3
    table[(u13, x3)] = -(x3 - x1) * u13 ** 3
    table[(u13, y3)] = -(y3 - y1) * u13 ** 3
    # u23: depends on x2,y2,x3,y3
    table[(u23, x2)] = -(x2 - x3) * u23 ** 3
    table[(u23, y2)] = -(y2 - y3) * u23 ** 3
    table[(u23, x3)] = -(x3 - x2) * u23 ** 3
    table[(u23, y3)] = -(y3 - y2) * u23 ** 3
    return table


CHAIN_RULE = _build_chain_rule_table()


def total_deriv(expr, var):
    """
    Total derivative of expr w.r.t. a position variable,
    accounting for the implicit dependence through u_ij:

      df/dvar = ∂f/∂var + Σ_ij (∂f/∂u_ij)(du_ij/dvar)

    For momentum variables, u_ij does not depend on p, so
    total_deriv == partial deriv.
    """
    result = diff(expr, var)
    if var in P_VARS:
        return result
    # Add chain-rule contributions from u variables
    for u_var in U_VARS:
        key = (u_var, var)
        if key in CHAIN_RULE:
            df_du = diff(expr, u_var)
            if df_du != 0:
                result += df_du * CHAIN_RULE[key]
    return result


# =====================================================================
# Exact Poisson bracket (using total derivatives)
# =====================================================================
def poisson_bracket(f, g):
    """
    {f, g} = Σ_i ( (df/dq_i)(dg/dp_i) - (df/dp_i)(dg/dq_i) )

    Uses total derivatives for position variables (chain rule through u_ij).
    Momentum derivatives are simple partials (u_ij doesn't depend on p).
    """
    result = Integer(0)
    for q, p in zip(Q_VARS, P_VARS):
        df_dq = total_deriv(f, q)
        dg_dp = diff(g, p)       # partial is fine for momenta
        df_dp = diff(f, p)
        dg_dq = total_deriv(g, q)
        result += df_dq * dg_dp - df_dp * dg_dq
    return result


def simplify_generator(expr):
    """
    Light simplification: cancel() handles rational functions well.
    Since we avoid sqrt, expressions stay polynomial in (q, p, u).
    """
    return cancel(expr)


# =====================================================================
# Pre-computed derivatives for CSE-optimised bracket computation
# =====================================================================
def precompute_derivatives(exprs, names=None, n_workers=1):
    """
    Pre-compute and cancel() ALL partial derivatives needed for
    Poisson brackets.

    For each expression, computes:
      - total_deriv(expr, q_i) for each of 6 position variables
      - diff(expr, p_i)        for each of 6 momentum variables

    Each derivative is simplified with cancel() once.  When these
    pre-computed derivatives are used in poisson_bracket_from_derivs(),
    the bracket becomes a simple multiply-and-add — no differentiation,
    no chain rule — giving a dramatic speedup.

    Parameters
    ----------
    exprs : list of SymPy expressions
    names : list of str, optional
    n_workers : int
        Number of parallel workers for the pre-computation (1 = serial)

    Returns
    -------
    list of dict
        Each dict has keys "dq" (list of 6 derivatives) and
        "dp" (list of 6 derivatives).
    """
    from multiprocessing import Pool as _Pool

    n = len(exprs)
    all_derivs = [None] * n
    t0 = time()

    if n_workers > 1:
        print(f"  Pre-computing derivatives for {n} expressions "
              f"({n_workers} workers)...")
        work = [(idx, exprs[idx]) for idx in range(n)]
        with _Pool(processes=n_workers) as pool:
            for idx, derivs in pool.imap_unordered(
                    _compute_one_deriv_set, work):
                all_derivs[idx] = derivs
                name = names[idx] if names else f"expr_{idx}"
                done = sum(1 for d in all_derivs if d is not None)
                if done % 10 == 0 or done == n:
                    print(f"    {done}/{n}  [{time()-t0:.1f}s]  "
                          f"last: {name}", flush=True)
    else:
        print(f"  Pre-computing derivatives for {n} expressions "
              f"(serial)...")
        for idx, expr in enumerate(exprs):
            _, derivs = _compute_one_deriv_set((idx, expr))
            all_derivs[idx] = derivs
            if (idx + 1) % 10 == 0 or idx == n - 1:
                name = names[idx] if names else f"expr_{idx}"
                print(f"    {idx+1}/{n}  [{time()-t0:.1f}s]  "
                      f"last: {name}", flush=True)

    elapsed = time() - t0
    total_terms = sum(
        sum(len(sp.Add.make_args(d)) for d in dd["dq"])
        + sum(len(sp.Add.make_args(d)) for d in dd["dp"])
        for dd in all_derivs
    )
    print(f"  Derivatives done: {n*12} derivs, "
          f"{total_terms} total terms  [{elapsed:.1f}s]")

    return all_derivs


def _compute_one_deriv_set(args):
    """Worker function for parallel derivative pre-computation.

    Uses expand() instead of cancel() because all expressions
    in our u_ij polynomial representation are polynomial (no
    rational functions), so there's nothing to cancel.  expand()
    is dramatically faster and ensures a canonical additive form.
    """
    idx, expr = args
    derivs = {"dq": [], "dp": []}

    for q in Q_VARS:
        d = total_deriv(expr, q)
        d = expand(d)
        derivs["dq"].append(d)

    for p in P_VARS:
        d = diff(expr, p)
        # Momentum derivs of polynomials are already simple
        d = expand(d)
        derivs["dp"].append(d)

    return idx, derivs


def poisson_bracket_from_derivs(derivs_f, derivs_g):
    """
    Compute {f, g} using PRE-COMPUTED derivatives.

    This is the CSE-optimised bracket: instead of differentiating
    the full expressions every time, we reuse derivatives that were
    computed and simplified once.

    The Poisson bracket is:
        {f,g} = Σ_k (df/dq_k * dg/dp_k - df/dp_k * dg/dq_k)

    Parameters
    ----------
    derivs_f, derivs_g : dict
        Each has keys "dq" (list of 6) and "dp" (list of 6).

    Returns
    -------
    SymPy expression (unsimplified sum of products)
    """
    result = Integer(0)
    for k in range(6):
        df_dq = derivs_f["dq"][k]
        dg_dp = derivs_g["dp"][k]
        df_dp = derivs_f["dp"][k]
        dg_dq = derivs_g["dq"][k]
        result += df_dq * dg_dp - df_dp * dg_dq
    return result


# =====================================================================
# Hamiltonians (in terms of u_ij)
# =====================================================================
T1 = (px1 ** 2 + py1 ** 2) / 2
T2 = (px2 ** 2 + py2 ** 2) / 2
T3 = (px3 ** 2 + py3 ** 2) / 2

H12 = T1 + T2 - u12         # V12 = -1/r12 = -u12
H13 = T1 + T3 - u13
H23 = T2 + T3 - u23

VALID_POTENTIALS = ('1/r', '1/r2', 'harmonic')


def build_hamiltonians(potential_type='1/r', masses=None, coupling=1):
    """Build pairwise Hamiltonians for a given potential type.

    Parameters
    ----------
    potential_type : str
        '1/r' (Newtonian), '1/r2' (Calogero-Moser), or 'harmonic'.
    masses : tuple of 3 floats, optional
        Particle masses (m1, m2, m3). Default (1, 1, 1).
    coupling : float
        Coupling constant (default 1).

    Returns (H12, H13, H23) as SymPy expressions.
    """
    if masses is None:
        masses = (1, 1, 1)
    m1, m2, m3 = [Rational(m).limit_denominator(100000) if isinstance(m, float)
                  else m for m in masses]

    KE1 = (px1**2 + py1**2) / (2 * m1)
    KE2 = (px2**2 + py2**2) / (2 * m2)
    KE3 = (px3**2 + py3**2) / (2 * m3)

    g = Rational(coupling).limit_denominator(100000) if isinstance(coupling, float) else coupling

    if potential_type == '1/r':
        return (KE1 + KE2 - g * m1 * m2 * u12,
                KE1 + KE3 - g * m1 * m3 * u13,
                KE2 + KE3 - g * m2 * m3 * u23)

    if potential_type == '1/r2':
        return (KE1 + KE2 - g * m1 * m2 * u12**2,
                KE1 + KE3 - g * m1 * m3 * u13**2,
                KE2 + KE3 - g * m2 * m3 * u23**2)

    if potential_type == 'harmonic':
        r12_sq = (x1 - x2)**2 + (y1 - y2)**2
        r13_sq = (x1 - x3)**2 + (y1 - y3)**2
        r23_sq = (x2 - x3)**2 + (y2 - y3)**2
        return (KE1 + KE2 + g * r12_sq,
                KE1 + KE3 + g * r13_sq,
                KE2 + KE3 + g * r23_sq)

    raise ValueError(f"Unknown potential type: {potential_type!r}. "
                     f"Valid: {VALID_POTENTIALS}")


# =====================================================================
# Checkpoint helpers
# =====================================================================
def save_checkpoint(level, all_exprs, all_names, all_levels):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"level_{level}.pkl")
    data = {
        "level": level,
        "exprs": all_exprs,
        "names": all_names,
        "levels": all_levels,
    }
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    Checkpoint saved: {path}")


def load_checkpoint():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pkl")]
    )
    if not files:
        return None
    path = os.path.join(CHECKPOINT_DIR, files[-1])
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    print(f"    Loaded checkpoint: {path}  (level {data['level']})")
    return data


# =====================================================================
# Phase-space sampling
# =====================================================================
def sample_phase_space(n, seed=42, pos_range=3.0, mom_range=1.0,
                       min_sep=0.5):
    """
    Sample phase-space points and compute u_ij = 1/r_ij for each.
    Returns (Z_qp, Z_u):
      Z_qp: (N, 12)  positions and momenta
      Z_u:  (N, 3)   inverse distances [u12, u13, u23]
    """
    rng = np.random.RandomState(seed)
    pts = np.empty((0, 12))
    for _ in range(200):
        bs = max((n - pts.shape[0]) * 5, 256)
        b = np.zeros((bs, 12))
        b[:, :6] = rng.uniform(-pos_range, pos_range, (bs, 6))
        b[:, 6:] = rng.uniform(-mom_range, mom_range, (bs, 6))
        dx12 = b[:, 0] - b[:, 2]; dy12 = b[:, 1] - b[:, 3]
        dx13 = b[:, 0] - b[:, 4]; dy13 = b[:, 1] - b[:, 5]
        dx23 = b[:, 2] - b[:, 4]; dy23 = b[:, 3] - b[:, 5]
        ok = ((dx12**2 + dy12**2 > min_sep**2) &
              (dx13**2 + dy13**2 > min_sep**2) &
              (dx23**2 + dy23**2 > min_sep**2))
        pts = np.vstack([pts, b[ok]])
        if pts.shape[0] >= n:
            break
    pts = pts[:n]

    # Compute u_ij = 1/r_ij
    dx12 = pts[:, 0] - pts[:, 2]; dy12 = pts[:, 1] - pts[:, 3]
    dx13 = pts[:, 0] - pts[:, 4]; dy13 = pts[:, 1] - pts[:, 5]
    dx23 = pts[:, 2] - pts[:, 4]; dy23 = pts[:, 3] - pts[:, 5]
    u12_vals = 1.0 / np.sqrt(dx12**2 + dy12**2)
    u13_vals = 1.0 / np.sqrt(dx13**2 + dy13**2)
    u23_vals = 1.0 / np.sqrt(dx23**2 + dy23**2)

    Z_u = np.column_stack([u12_vals, u13_vals, u23_vals])
    return pts, Z_u


# =====================================================================
# Lambdify a list of SymPy expressions (with CSE)
# =====================================================================
def _expr_to_chunked_lines(expr, target_var, indent="    ",
                           max_terms_per_line=50):
    """
    Convert a SymPy expression into chunked Python lines that avoid
    deep AST nesting.  Instead of one giant return expression, we
    accumulate into a variable in chunks.
    """
    terms = sp.Add.make_args(expr)
    if len(terms) <= max_terms_per_line:
        return [f"{indent}{target_var} = {sp.pycode(expr)}"]

    lines = [f"{indent}{target_var} = 0"]
    for i in range(0, len(terms), max_terms_per_line):
        chunk = terms[i:i + max_terms_per_line]
        chunk_expr = sp.Add(*chunk)
        lines.append(f"{indent}{target_var} += {sp.pycode(chunk_expr)}")
    return lines


def _make_flat_func(expr, func_name="_f"):
    """
    Build a flat (non-nested) numpy function for a SymPy expression.
    Uses CSE for efficiency and chunked code generation to avoid
    Python's compile() recursion limit on deeply nested ASTs.
    """
    replacements, reduced = sp.cse([expr])

    var_str = ", ".join(str(v) for v in ALL_VARS)
    lines = [f"def {func_name}({var_str}):"]

    for sym, sub_expr in replacements:
        lines.extend(_expr_to_chunked_lines(sub_expr, str(sym)))

    lines.extend(_expr_to_chunked_lines(reduced[0], "_result"))
    lines.append("    return _result")

    code = "\n".join(lines)
    namespace = {
        "sqrt": np.sqrt,
        "math": __import__("math"),
    }
    exec(compile(code, "<generated>", "exec"), namespace)
    return namespace[func_name]


def _expr_to_chunked_lines(expr, target_var, indent="    ",
                           max_terms_per_line=50):
    """Break large additions into chunked += lines to avoid deep AST."""
    from sympy import pycode
    terms = sp.Add.make_args(expr)
    if len(terms) <= max_terms_per_line:
        return [f"{indent}{target_var} = {pycode(expr)}"]
    lines = [f"{indent}{target_var} = 0"]
    for i in range(0, len(terms), max_terms_per_line):
        chunk_expr = sp.Add(*terms[i:i + max_terms_per_line])
        lines.append(
            f"{indent}{target_var} += {pycode(chunk_expr)}")
    return lines


def _make_flat_func(expr, label="_f"):
    """Fallback for expressions too deeply nested for compile().

    Uses CSE to flatten the expression tree, writes chunked assignment
    code to a temp file, and imports it -- bypassing the recursive
    compiler even for very large Yukawa-type expressions.
    """
    import tempfile
    import importlib.util
    from sympy import cse as _cse, pycode

    replacements, (reduced,) = _cse(expr, optimizations='basic')

    var_names = [str(v) for v in ALL_VARS]
    sig = ", ".join(var_names)
    lines = [
        "import numpy as _np",
        "from numpy import exp, log, sqrt, sin, cos, abs, power",
        f"def {label}({sig}):",
    ]
    for sym, sub in replacements:
        lines.extend(_expr_to_chunked_lines(sub, str(sym)))
    lines.extend(_expr_to_chunked_lines(reduced, "_result"))
    lines.append("    return _result")
    code = "\n".join(lines)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="flat_")
    tmp.write(code)
    tmp.flush()
    tmp.close()

    spec = importlib.util.spec_from_file_location(f"_flat_{label}", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, label)


def lambdify_generators(exprs):
    """
    Convert a list of SymPy expressions to fast numpy callables.

    For small expression sets (< 50), uses standard lambdify.
    For large sets, builds flat functions individually to avoid
    recursion-limit issues with Python's compiler.
    """
    n = len(exprs)
    t0 = time()

    if n <= 50:
        print(f"    Lambdifying {n} expressions (standard)...",
              end=" ", flush=True)
        func = sp.lambdify(ALL_VARS, exprs, modules="numpy", cse=True)
        print(f"done [{time() - t0:.1f}s]")

        def evaluate(Z_qp, Z_u):
            args = ([Z_qp[:, i] for i in range(12)] +
                    [Z_u[:, i] for i in range(3)])
            vals = func(*args)
            return np.column_stack(vals)

        return evaluate

    LAYER_LAMBDIFY = 0
    LAYER_FLAT_CSE = 1
    LAYER_XREPLACE = 2

    print(f"    Lambdifying {n} expressions individually...", flush=True)

    funcs = []
    layers = []
    counts = [0, 0, 0]

    for idx, expr in enumerate(exprs):
        if (idx + 1) % 20 == 0 or idx == n - 1:
            print(f"      {idx+1}/{n}  [{time()-t0:.1f}s]  "
                  f"(lambdify:{counts[0]} flat:{counts[1]} "
                  f"xreplace:{counts[2]})", flush=True)

        # Layer 1: standard lambdify
        try:
            f = sp.lambdify(ALL_VARS, expr, modules="numpy", cse=False)
            funcs.append(f)
            layers.append(LAYER_LAMBDIFY)
            counts[0] += 1
            continue
        except (RecursionError, Exception):
            pass

        # Layer 2: CSE + chunked temp-file (still vectorised)
        try:
            f = _make_flat_func(expr, label=f"g{idx}")
            funcs.append(f)
            layers.append(LAYER_FLAT_CSE)
            counts[1] += 1
            continue
        except (RecursionError, Exception) as e:
            print(f"      [{idx}] flat-CSE also failed ({e.__class__.__name__}), "
                  f"falling back to xreplace", flush=True)

        # Layer 3: point-by-point xreplace (slow, last resort)
        var_syms = list(ALL_VARS)
        captured_expr = expr

        def _make_xreplace(ex, vs):
            def _subs_eval(*args):
                n_pts = len(args[0]) if hasattr(args[0], '__len__') else 1
                result = np.empty(n_pts)
                for i in range(n_pts):
                    pt = {v: (float(a[i]) if hasattr(a, '__len__') else float(a))
                          for v, a in zip(vs, args)}
                    try:
                        result[i] = complex(ex.xreplace(pt)).real
                    except Exception:
                        result[i] = 0.0
                return result
            return _subs_eval

        funcs.append(_make_xreplace(captured_expr, var_syms))
        layers.append(LAYER_XREPLACE)
        counts[2] += 1

    print(f"    Lambdify results: {counts[0]} lambdify, "
          f"{counts[1]} flat-CSE, {counts[2]} xreplace")
    print(f"    Total lambdify time: {time() - t0:.1f}s")

    def evaluate(Z_qp, Z_u):
        args = ([Z_qp[:, i] for i in range(12)] +
                [Z_u[:, i] for i in range(3)])
        n_pts = Z_qp.shape[0]
        cols = []
        t_eval = time()
        for idx, (f, layer) in enumerate(zip(funcs, layers)):
            if layer == LAYER_XREPLACE:
                print(f"      WARNING: eval {idx+1}/{n} using xreplace "
                      f"(slow!)", flush=True)
            val = f(*args)
            arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
            if arr.shape[0] == 1:
                arr = np.full(n_pts, arr[0])
            elif arr.shape[0] < n_pts:
                arr = np.resize(arr, n_pts)
            cols.append(arr[:n_pts])
            if (idx + 1) % 20 == 0:
                print(f"      eval {idx+1}/{n}  [{time()-t_eval:.1f}s]",
                      flush=True)
        return np.column_stack(cols)

    return evaluate


# =====================================================================
# SVD gap analysis
# =====================================================================
def svd_gap_analysis(eval_matrix, label=""):
    """
    Compute SVD, print spectrum, identify gaps.
    Returns (rank, singular_values).

    Gap detection: finds the largest ratio s[i]/s[i+1] where BOTH
    s[i] and s[i+1] are above machine zero.  Also detects the
    transition from meaningful values to machine-zero (the
    "noise floor" boundary).
    """
    # Column-normalise
    norms = np.linalg.norm(eval_matrix, axis=0)
    norms[norms < 1e-15] = 1.0
    M = eval_matrix / norms

    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # --- Determine the noise floor ---
    # For exact symbolic expressions evaluated in float64, the noise
    # floor is typically at ~1e-10 relative to max.  Any singular
    # value below this is numerical noise, not algebraic content.
    noise_threshold = 1e-8 * s[0]   # relative threshold
    n_meaningful = int(np.sum(s > noise_threshold))

    # --- Find the largest gap among meaningful singular values ---
    best_gap_ratio = 1.0
    best_gap_idx = -1
    for i in range(min(n_meaningful, len(s) - 1)):
        if s[i + 1] > noise_threshold:
            gap = s[i] / s[i + 1]
        else:
            # This is the transition to noise floor
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

        # Print selectively: first 30, around the gap, key positions, tail
        show = (i < 30 or
                (best_gap_idx >= 0 and abs(i - best_gap_idx) < 5) or
                gap > 100.0 or
                i >= len(s) - 5 or
                (i + 1) in (17, 52, 69, 116, 407))
        if show:
            marker = ""
            if i + 1 == 3:
                marker = "  <-- 3 (A114491)"
            elif i + 1 == 6:
                marker = "  <-- 6 (A114491)"
            elif i + 1 == 17:
                marker = "  <-- 17 (A114491)"
            elif i + 1 == 69:
                marker = "  <-- 69 (A114491)"
            elif i + 1 == 116:
                marker = "  <-- 116"
            if i == best_gap_idx:
                marker += f"  *** GAP ({gap:.1e}) ***"
            elif gap > 100 and i != best_gap_idx:
                marker += f"  (gap {gap:.1e})"
            print(f"  {i+1:>5} | {s[i]:>18.12f} | {gap:>12.2f} "
                  f"| {rel:>12.2e}{marker}")

    # Determine rank from the definitive gap
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
        print(f"\n  No clear gap found (best ratio {best_gap_ratio:.1f}x)")
        print(f"  Using noise-floor threshold: rank = {rank}")

    return rank, s


# =====================================================================
# Jacobi identity verification
# =====================================================================
def verify_jacobi_symbolic(a_expr, b_expr, c_expr,
                           a_name, b_name, c_name):
    """Exact symbolic check: {A,{B,C}} + {B,{C,A}} + {C,{A,B}} == 0"""
    print(f"    Jacobi ({a_name}, {b_name}, {c_name}) [symbolic]...",
          end=" ", flush=True)
    t0 = time()

    bc = poisson_bracket(b_expr, c_expr)
    ca = poisson_bracket(c_expr, a_expr)
    ab = poisson_bracket(a_expr, b_expr)

    j1 = poisson_bracket(a_expr, bc)
    j2 = poisson_bracket(b_expr, ca)
    j3 = poisson_bracket(c_expr, ab)

    result = cancel(j1 + j2 + j3)
    ok = result == 0
    elapsed = time() - t0

    status = "EXACT ZERO" if ok else f"NONZERO ({len(sp.Add.make_args(result))} terms)"
    print(f"{status}  [{elapsed:.1f}s]")
    return ok


def verify_jacobi_numerical(a_expr, b_expr, c_expr,
                             a_name, b_name, c_name,
                             n_pts=20, seed=123):
    """
    Numerically verify Jacobi using exact symbolic expressions
    evaluated at random points.  Still highly reliable since only
    float64 rounding introduces error.
    """
    print(f"    Jacobi ({a_name}, {b_name}, {c_name}) [numerical]...",
          end=" ", flush=True)
    t0 = time()

    bc = poisson_bracket(b_expr, c_expr)
    ca = poisson_bracket(c_expr, a_expr)
    ab = poisson_bracket(a_expr, b_expr)
    print(f"inner [{time()-t0:.1f}s]...", end=" ", flush=True)

    j1 = poisson_bracket(a_expr, bc)
    j2 = poisson_bracket(b_expr, ca)
    j3 = poisson_bracket(c_expr, ab)
    print(f"outer [{time()-t0:.1f}s]...", end=" ", flush=True)

    total = j1 + j2 + j3

    f = sp.lambdify(ALL_VARS, total, modules="numpy")
    Z_qp, Z_u = sample_phase_space(n_pts, seed)
    args = [Z_qp[:, i] for i in range(12)] + [Z_u[:, i] for i in range(3)]
    vals = np.array(f(*args))
    max_err = np.max(np.abs(vals))
    elapsed = time() - t0

    ok = max_err < 1e-10
    status = f"max |err| = {max_err:.2e}" + (" OK" if ok else " FAIL")
    print(f"{status}  [{elapsed:.1f}s]")
    return ok


# =====================================================================
# Main computation
# =====================================================================
def compute_exact_growth(max_level=3, n_samples=500, seed=42,
                         resume=False, potential_type='1/r',
                         masses=None, coupling=1):
    pot_label = {'1/r': 'Newtonian (1/r)',
                 '1/r2': 'Calogero-Moser (1/r²)',
                 'harmonic': 'Harmonic (r²)'}.get(potential_type, potential_type)
    mass_str = (f"m=({masses[0]},{masses[1]},{masses[2]})"
                if masses else "m1=m2=m3=1")
    print("=" * 70)
    print("EXACT LIE ALGEBRA GROWTH  (SymPy symbolic computation)")
    print("  Polynomial representation: u_ij = 1/r_ij auxiliary vars")
    print("=" * 70)
    print(f"  Potential: {pot_label},  {mass_str},  g={coupling}")
    print(f"  Max level: {max_level},  Samples: {n_samples},  Seed: {seed}")
    print()

    start_level = 0
    all_exprs = []
    all_names = []
    all_levels = []
    computed_pairs = set()

    if resume:
        ckpt = load_checkpoint()
        if ckpt is not None:
            all_exprs = ckpt["exprs"]
            all_names = ckpt["names"]
            all_levels = ckpt["levels"]
            start_level = ckpt["level"] + 1
            for i in range(len(all_exprs)):
                for j in range(i + 1, len(all_exprs)):
                    if all_levels[i] + all_levels[j] < start_level:
                        computed_pairs.add(frozenset({i, j}))

    # ------------------------------------------------------------------
    # Level 0
    # ------------------------------------------------------------------
    if start_level <= 0:
        print("--- Level 0: Pairwise Hamiltonians ---")
        if potential_type == '1/r' and masses is None and coupling == 1:
            h12, h13, h23 = H12, H13, H23
        else:
            h12, h13, h23 = build_hamiltonians(potential_type, masses, coupling)
        for name, expr in [("H12", h12), ("H13", h13), ("H23", h23)]:
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(0)
            print(f"  {name}: {len(sp.Add.make_args(expr))} terms")

        for i in range(3):
            for j in range(i + 1, 3):
                computed_pairs.add(frozenset({i, j}))

        save_checkpoint(0, all_exprs, all_names, all_levels)

    # ------------------------------------------------------------------
    # Level 1
    # ------------------------------------------------------------------
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
            expr = poisson_bracket(all_exprs[i], all_exprs[j])
            expr = simplify_generator(expr)
            elapsed = time() - t0
            nterms = len(sp.Add.make_args(expr))
            print(f"{nterms} terms  [{elapsed:.1f}s]")
            all_exprs.append(expr)
            all_names.append(short)
            all_levels.append(1)

        save_checkpoint(1, all_exprs, all_names, all_levels)

    # ------------------------------------------------------------------
    # Levels 2+
    # ------------------------------------------------------------------
    for level in range(max(2, start_level), max_level + 1):
        print(f"\n--- Level {level} ---")
        t_level = time()

        frontier_indices = [
            i for i, lv in enumerate(all_levels) if lv == level - 1
        ]
        n_existing = len(all_exprs)

        n_candidates = 0
        new_exprs_this_level = []
        new_names_this_level = []

        for i in frontier_indices:
            for j in range(n_existing):
                if i == j:
                    continue
                pair = frozenset({i, j})
                if pair in computed_pairs:
                    continue
                computed_pairs.add(pair)

                n_candidates += 1
                ni = all_names[i]
                nj = all_names[j]
                bracket_name = f"{{{ni},{nj}}}"

                print(f"  [{n_candidates:>4d}] {bracket_name}...",
                      end=" ", flush=True)
                t0 = time()
                expr = poisson_bracket(all_exprs[i], all_exprs[j])
                t_bracket = time() - t0
                print(f"bracket {t_bracket:.1f}s...", end=" ", flush=True)

                t0s = time()
                expr = simplify_generator(expr)
                t_simp = time() - t0s

                nterms = len(sp.Add.make_args(expr))
                print(f"simplify {t_simp:.1f}s  -> {nterms} terms")

                new_exprs_this_level.append(expr)
                new_names_this_level.append(bracket_name)

        for expr, name in zip(new_exprs_this_level, new_names_this_level):
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(level)

        elapsed_level = time() - t_level
        print(f"\n  Level {level}: {len(new_exprs_this_level)} candidates "
              f"computed in {elapsed_level:.1f}s")

        save_checkpoint(level, all_exprs, all_names, all_levels)

    # ------------------------------------------------------------------
    # Jacobi identity verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("JACOBI IDENTITY VERIFICATION")
    print("=" * 70)

    # Exact symbolic check for the fundamental triple
    verify_jacobi_symbolic(
        all_exprs[0], all_exprs[1], all_exprs[2],
        "H12", "H13", "H23",
    )

    # Numerical-from-exact for triples involving K generators
    if len(all_exprs) > 5:
        verify_jacobi_numerical(
            all_exprs[0], all_exprs[1], all_exprs[3],
            "H12", "H13", "K1",
        )
        verify_jacobi_numerical(
            all_exprs[0], all_exprs[3], all_exprs[4],
            "H12", "K1", "K2",
        )

    # ------------------------------------------------------------------
    # Numerical evaluation and SVD
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("NUMERICAL EVALUATION AND SVD ANALYSIS")
    print("=" * 70)

    Z_qp, Z_u = sample_phase_space(n_samples, seed)
    print(f"  Sample points: {Z_qp.shape[0]}")

    evaluate = lambdify_generators(all_exprs)
    print("    Evaluating at sample points...", end=" ", flush=True)
    t0 = time()
    eval_matrix = evaluate(Z_qp, Z_u)
    print(f"done [{time() - t0:.1f}s]")
    print(f"    Evaluation matrix shape: {eval_matrix.shape}")

    # Per-level SVD
    level_dims = {}
    for lv in range(max_level + 1):
        mask = [i for i, l in enumerate(all_levels) if l <= lv]
        sub = eval_matrix[:, mask]
        rank, svals = svd_gap_analysis(
            sub, label=f"(through level {lv})"
        )
        level_dims[lv] = rank
        print(f"  ==> Dimension through level {lv}: {rank}")

    # Full SVD
    rank_full, svals_full = svd_gap_analysis(
        eval_matrix, label="(ALL generators)"
    )

    # Summary
    print("\n" + "=" * 70)
    print("DIMENSION SUMMARY")
    print("=" * 70)

    a114491 = [2, 3, 6, 17, 69, 407, 3808, 75165]

    for lv in range(max_level + 1):
        dim = level_dims[lv]
        prediction = a114491[lv + 1] if lv + 1 < len(a114491) else "?"
        match = "MATCH" if dim == prediction else "no match"
        print(f"  Level {lv}: dim = {dim:>5d}    "
              f"A114491({lv+1}) = {prediction}    [{match}]")

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
        ax.set_title("Exact Lie Algebra: Singular Value Spectrum")
        ax.grid(True, alpha=0.3)

        for pos, label in [(3, "3"), (6, "6"), (17, "17"),
                           (69, "69"), (407, "407")]:
            if pos <= len(svals_full):
                ax.axvline(x=pos, color="red", linestyle="--", alpha=0.4)
                ax.text(pos, 0.5, f" {label}", color="red", fontsize=9)

        plt.tight_layout()
        plt.savefig("exact_svd_spectrum.png", dpi=150)
        print(f"\n  SVD plot saved to exact_svd_spectrum.png")
        plt.close()
    except Exception as e:
        print(f"\n  (Plot skipped: {e})")


def main():
    ap = argparse.ArgumentParser(
        description="Exact Lie-algebra growth via symbolic Poisson brackets"
    )
    ap.add_argument("--max-level", type=int, default=3,
                    help="Maximum bracket level (default: 3)")
    ap.add_argument("--samples", type=int, default=500,
                    help="Phase-space sample points (default: 500)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last checkpoint")
    ap.add_argument("--potential", default="1/r",
                    choices=VALID_POTENTIALS,
                    help="Potential type (default: 1/r)")
    ap.add_argument("--masses", nargs=3, type=float, default=None,
                    metavar=("M1", "M2", "M3"),
                    help="Particle masses (default: 1 1 1)")
    ap.add_argument("--coupling", type=float, default=1.0,
                    help="Coupling constant (default: 1.0)")
    args = ap.parse_args()

    compute_exact_growth(
        max_level=args.max_level,
        n_samples=args.samples,
        seed=args.seed,
        resume=args.resume,
        potential_type=args.potential,
        masses=tuple(args.masses) if args.masses else None,
        coupling=args.coupling,
    )


if __name__ == "__main__":
    main()
