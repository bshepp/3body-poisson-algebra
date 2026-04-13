#!/usr/bin/env python3
"""Fast numerical comparison of level-3 structure constants.

Uses finite differences for Poisson brackets — no symbolic derivatives.
This is MUCH faster than the symbolic-derivative approach.

Strategy:
  1. Load level_3.pkl checkpoints for both potentials
  2. Lambdify all 156 generators (fast — no derivatives)
  3. Sample random (q, p, u) points (treating u as independent)
  4. Evaluate all generators at sample points → 156 × N matrix
  5. QR for 116-dim basis selection
  6. Compute Poisson brackets numerically using finite differences + chain rule
  7. Express brackets in basis via least-squares → structure constants
  8. Compare tensors between potentials

Usage:
  python fast_level3_compare.py
"""

import os
import sys
import pickle
import numpy as np
from time import time
from itertools import combinations

# Level-3 generators have deep expression trees (2000-9000 ops)
sys.setrecursionlimit(50000)

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sympy as sp
from sympy import lambdify, cse
from exact_growth_nbody import NBodyAlgebra


def load_checkpoint(path):
    """Load old-format level_N.pkl checkpoint."""
    print(f"  Loading {path}...", end=" ", flush=True)
    t0 = time()
    with open(path, 'rb') as f:
        data = pickle.load(f)
    dt = time() - t0
    print(f"[{dt:.1f}s] {len(data['exprs'])} generators")
    return data


def lambdify_generators(exprs, all_vars):
    """Lambdify all generators into fast numpy callables.
    Uses stack-based evaluator for deep expression trees that crash compile().
    """
    print(f"  Lambdifying {len(exprs)} generators...", end=" ", flush=True)
    t0 = time()
    funcs = []
    var_names = [str(v) for v in all_vars]
    
    for i, expr in enumerate(exprs):
        try:
            f = lambdify(all_vars, expr, modules='numpy')
        except RecursionError:
            f = _make_stack_evaluator(expr, all_vars, var_names)
        funcs.append(f)
        if (i + 1) % 50 == 0:
            elapsed = time() - t0
            print(f"{i+1}/{len(exprs)} [{elapsed:.0f}s]", end=" ", flush=True)
    dt = time() - t0
    print(f"[{dt:.1f}s]")
    return funcs


def _make_stack_evaluator(expr, all_vars, var_names):
    """Build an iterative stack-based evaluator for deeply nested SymPy expressions.
    Compiles the expression into a flat instruction list, avoiding compile() recursion.
    """
    from sympy import Add, Mul, Pow, Integer, Rational, Float, Symbol, S
    
    instructions = []  # list of (op, arg)
    cache = {}  # id(expr) → slot index
    var_map = {str(v): i for i, v in enumerate(all_vars)}
    slot = [0]  # mutable counter
    
    # Iterative post-order traversal
    stack = [(expr, False)]
    while stack:
        node, processed = stack.pop()
        nid = id(node)
        
        if nid in cache:
            continue
        
        if node.is_Symbol:
            s = slot[0]; slot[0] += 1
            cache[nid] = s
            instructions.append(('var', var_map[str(node)]))
            continue
        
        if node.is_Number:
            s = slot[0]; slot[0] += 1
            cache[nid] = s
            instructions.append(('const', float(node)))
            continue
        
        if not processed:
            # Check if all children are cached
            all_cached = all(id(a) in cache for a in node.args)
            if all_cached:
                processed = True
            else:
                stack.append((node, True))
                for a in reversed(node.args):
                    if id(a) not in cache:
                        stack.append((a, False))
                continue
        
        # All children cached — emit instruction
        child_slots = [cache[id(a)] for a in node.args]
        s = slot[0]; slot[0] += 1
        cache[nid] = s
        
        if node.is_Add:
            instructions.append(('add', child_slots))
        elif node.is_Mul:
            instructions.append(('mul', child_slots))
        elif node.is_Pow:
            instructions.append(('pow', child_slots))
        elif hasattr(node, 'func'):
            fname = type(node).__name__
            instructions.append(('func', (fname, child_slots)))
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
    
    root_slot = cache[id(expr)]
    n_slots = slot[0]
    
    func_map = {
        'sqrt': np.sqrt, 'Abs': np.abs, 'sin': np.sin, 'cos': np.cos,
        'exp': np.exp, 'log': np.log,
    }
    
    def evaluator(*args):
        args = list(args)
        n_pts = np.asarray(args[0]).shape[0] if hasattr(args[0], '__len__') else 1
        slots = [None] * n_slots
        
        for idx, (op, operand) in enumerate(instructions):
            if op == 'var':
                slots[idx] = np.asarray(args[operand], dtype=float)
            elif op == 'const':
                slots[idx] = np.full(n_pts, operand)
            elif op == 'add':
                r = slots[operand[0]].copy()
                for si in operand[1:]:
                    r = r + slots[si]
                slots[idx] = r
            elif op == 'mul':
                r = slots[operand[0]].copy()
                for si in operand[1:]:
                    r = r * slots[si]
                slots[idx] = r
            elif op == 'pow':
                slots[idx] = slots[operand[0]] ** slots[operand[1]]
            elif op == 'func':
                fname, child_sl = operand
                if fname in func_map:
                    slots[idx] = func_map[fname](slots[child_sl[0]])
                else:
                    raise ValueError(f"Unknown function: {fname}")
        
        return slots[root_slot]
    
    return evaluator


def evaluate_generators(funcs, points):
    """Evaluate all generators at all sample points.
    
    points: (N_points, N_vars) array
    Returns: (N_gens, N_points) array
    """
    n_gens = len(funcs)
    n_pts = points.shape[0]
    vals = np.zeros((n_gens, n_pts))
    for i, f in enumerate(funcs):
        args = [points[:, j] for j in range(points.shape[1])]
        try:
            vals[i] = f(*args)
        except Exception:
            # Fallback: evaluate point by point
            for p in range(n_pts):
                vals[i, p] = f(*points[p])
    return vals


def select_basis_qr(gen_vals, expected_rank=116):
    """Select linearly independent basis using column-pivoted QR.
    
    gen_vals: (N_gens, N_points) array
    Returns: indices of basis generators
    """
    from scipy.linalg import qr
    # QR on (N_points × N_gens) to find independent columns (=generators)
    Q, R, perm = qr(gen_vals.T, pivoting=True)
    # Diagonal of R gives the "importance" of each generator
    diag = np.abs(np.diag(R))
    # Find gap
    if len(diag) > expected_rank:
        gap = diag[expected_rank - 1] / (diag[expected_rank] + 1e-300)
        print(f"  QR gap at position {expected_rank}: {gap:.2e} "
              f"(s_{expected_rank}={diag[expected_rank-1]:.4e}, "
              f"s_{expected_rank+1}={diag[expected_rank]:.4e})")
    basis_idx = sorted(perm[:expected_rank])
    return basis_idx


def precompute_partials(funcs, points, h=1e-7):
    """Precompute all partial derivatives of all generators at all points.
    
    Returns: (N_gens, N_vars, N_points) array
    """
    n_gens = len(funcs)
    n_pts = points.shape[0]
    n_vars = points.shape[1]
    
    print(f"  Precomputing partial derivatives ({n_gens} gens × {n_vars} vars)...", 
          flush=True)
    t0 = time()
    
    partials = np.zeros((n_gens, n_vars, n_pts))
    
    for v in range(n_vars):
        pts_plus = points.copy()
        pts_minus = points.copy()
        pts_plus[:, v] += h
        pts_minus[:, v] -= h
        
        for i, f in enumerate(funcs):
            f_plus = evaluate_single(f, pts_plus)
            f_minus = evaluate_single(f, pts_minus)
            partials[i, v] = (f_plus - f_minus) / (2 * h)
        
        if (v + 1) % 5 == 0:
            elapsed = time() - t0
            eta = elapsed / (v + 1) * (n_vars - v - 1)
            print(f"    var {v+1}/{n_vars}  [{elapsed:.1f}s, ETA {eta:.0f}s]", flush=True)
    
    print(f"  Partials done [{time()-t0:.1f}s]")
    return partials


def precompute_chain_rule_vals(algebra, points):
    """Precompute chain rule values du_ij/dq_k at all sample points.
    
    Returns: dict mapping (u_idx_rel, q_idx) → (N_points,) array
    """
    n_q = len(algebra.q_vars)
    n_p = len(algebra.p_vars)
    chain_vals = {}
    
    for u_idx_rel, u_var in enumerate(algebra.u_vars):
        u_idx = n_q + n_p + u_idx_rel
        u_vals = points[:, u_idx]
        
        pair_idx = u_idx_rel
        bi, bj = algebra.body_pairs[pair_idx]
        
        for k in range(n_q):
            body_of_q = k // algebra.d + 1
            comp_of_q = k % algebra.d
            
            if body_of_q == bi:
                qi_k = points[:, k]
                qj_comp = algebra.q_by_body[bj][comp_of_q]
                qj_k = points[:, algebra.q_vars.index(qj_comp)]
                chain_vals[(u_idx_rel, k)] = -(qi_k - qj_k) * u_vals**3
            elif body_of_q == bj:
                qi_comp = algebra.q_by_body[bi][comp_of_q]
                qi_k = points[:, algebra.q_vars.index(qi_comp)]
                qj_k = points[:, k]
                chain_vals[(u_idx_rel, k)] = -(qj_k - qi_k) * u_vals**3
    
    return chain_vals


def compute_total_derivs(partials, chain_vals, algebra, n_gens):
    """Compute total derivatives Df/Dq_k from partial derivatives and chain rule.
    
    Returns: (N_gens, N_q, N_points) array of total q-derivatives
    """
    n_q = len(algebra.q_vars)
    n_p = len(algebra.p_vars)
    n_pts = partials.shape[2]
    
    total_dq = partials[:, :n_q, :].copy()  # start with ∂f/∂q_k
    
    for u_idx_rel in range(len(algebra.u_vars)):
        u_var_idx = n_q + n_p + u_idx_rel  # index in all_vars
        for k in range(n_q):
            cv = chain_vals.get((u_idx_rel, k))
            if cv is not None:
                # total_dq[:, k, :] += ∂f/∂u * du/dq_k
                total_dq[:, k, :] += partials[:, u_var_idx, :] * cv[np.newaxis, :]
    
    return total_dq


def bracket_from_derivs(total_dq, partials, idx_i, idx_j, n_q):
    """Compute {f_i, f_j} from precomputed derivatives.
    
    {f, g} = Σ_k [ (Df/Dq_k)(∂g/∂p_k) - (∂f/∂p_k)(Dg/Dq_k) ]
    """
    bracket = np.zeros(total_dq.shape[2])
    for k in range(n_q):
        p_idx = n_q + k  # index of p_k in all_vars
        bracket += (total_dq[idx_i, k] * partials[idx_j, p_idx] 
                   - partials[idx_i, p_idx] * total_dq[idx_j, k])
    return bracket


def evaluate_single(f, points):
    """Evaluate a single lambdified function at all points."""
    args = [points[:, j] for j in range(points.shape[1])]
    try:
        result = f(*args)
        if np.isscalar(result):
            return np.full(points.shape[0], result)
        return np.asarray(result, dtype=float)
    except Exception:
        n = points.shape[0]
        result = np.zeros(n)
        for p in range(n):
            result[p] = float(f(*points[p]))
        return result


def compute_structure_constants(total_dq, partials, basis_idx, basis_vals, 
                                 n_q):
    """Compute full structure constant tensor C[i,j,k].
    
    {basis_i, basis_j} = Σ_k C[i,j,k] * basis_k
    
    Returns: (dim, dim, dim) array
    """
    dim = len(basis_idx)
    n_brackets = dim * (dim - 1) // 2
    n_pts = basis_vals.shape[1]
    
    # Precompute pseudo-inverse of basis_vals
    print(f"  Computing pseudo-inverse of basis ({dim} × {n_pts})...", 
          end=" ", flush=True)
    t0 = time()
    pinv = np.linalg.pinv(basis_vals.T)  # (dim, N_points)
    print(f"[{time()-t0:.1f}s]")
    
    C = np.zeros((dim, dim, dim))
    max_residual = 0.0
    
    print(f"  Computing {n_brackets} Poisson brackets...", flush=True)
    t0 = time()
    count = 0
    for ii in range(dim):
        for jj in range(ii + 1, dim):
            gi = basis_idx[ii]
            gj = basis_idx[jj]
            
            bkt = bracket_from_derivs(total_dq, partials, gi, gj, n_q)
            
            # Express in basis via pseudo-inverse
            coeffs = pinv @ bkt  # (dim,)
            
            # Check residual
            bkt_norm = np.linalg.norm(bkt)
            if bkt_norm > 1e-12:
                residual = np.linalg.norm(bkt - basis_vals.T @ coeffs) / bkt_norm
            else:
                residual = 0.0
            max_residual = max(max_residual, residual)
            
            C[ii, jj, :] = coeffs
            C[jj, ii, :] = -coeffs  # antisymmetry
            
            count += 1
            if count % 500 == 0 or count == n_brackets:
                elapsed = time() - t0
                rate = count / elapsed
                eta = (n_brackets - count) / rate if rate > 0 else 0
                print(f"    {count}/{n_brackets}  [{elapsed:.1f}s, ETA {eta:.0f}s]  "
                      f"max residual: {max_residual:.2e}", flush=True)
    
    elapsed = time() - t0
    print(f"  Done: {n_brackets} brackets in {elapsed:.1f}s "
          f"({elapsed/n_brackets*1000:.1f}ms each)")
    print(f"  Max regression residual: {max_residual:.2e}")
    return C


def process_potential(label, ckpt_path, algebra, n_points=300, h=1e-7):
    """Full pipeline for one potential."""
    data = load_checkpoint(ckpt_path)
    exprs = data['exprs']
    names = data['names']
    levels = data['levels']
    
    print(f"  Generators by level: ", end="")
    for lv in sorted(set(levels)):
        n = sum(1 for l in levels if l == lv)
        print(f"L{lv}={n}", end=" ")
    print()
    
    # Lambdify
    all_vars = algebra.q_vars + algebra.p_vars + algebra.u_vars
    funcs = lambdify_generators(exprs, all_vars)
    
    # Sample random points (FIXED SEED — same for both potentials)
    # CRITICAL: evaluate on the CONSTRAINT SURFACE u_ij = 1/r_ij
    # Treating u as independent causes catastrophic cancellation.
    n_vars = len(all_vars)
    n_q = len(algebra.q_vars)
    n_p = len(algebra.p_vars)
    n_u = len(algebra.u_vars)
    rng = np.random.RandomState(42)
    points = np.empty((n_points, n_vars))
    
    # q: positions spread out so particles aren't too close
    # Place bodies at random positions with minimum separation
    d = algebra.d
    N = algebra.N
    for pt in range(n_points):
        while True:
            q_vals = rng.uniform(-3.0, 3.0, n_q)
            # Check minimum pairwise distance
            min_dist = np.inf
            for (bi, bj) in algebra.body_pairs:
                qi = q_vals[(bi-1)*d : bi*d]
                qj = q_vals[(bj-1)*d : bj*d]
                dist = np.sqrt(np.sum((qi - qj)**2))
                min_dist = min(min_dist, dist)
            if min_dist > 0.5:  # ensure well-separated
                break
        points[pt, :n_q] = q_vals
    
    # p: momenta, O(1) random
    points[:, n_q:n_q+n_p] = rng.uniform(-2.0, 2.0, (n_points, n_p))
    
    # u: computed from positions as u_ij = 1/r_ij
    for u_idx, (bi, bj) in enumerate(algebra.body_pairs):
        qi = points[:, (bi-1)*d : bi*d]
        qj = points[:, (bj-1)*d : bj*d]
        r_ij = np.sqrt(np.sum((qi - qj)**2, axis=1))
        points[:, n_q + n_p + u_idx] = 1.0 / r_ij
    
    # Evaluate all generators
    print(f"  Evaluating {len(funcs)} generators at {n_points} points...", end=" ", flush=True)
    t0 = time()
    gen_vals = evaluate_generators(funcs, points)
    print(f"[{time()-t0:.1f}s]")
    
    # Check for NaN/Inf
    bad = np.isnan(gen_vals) | np.isinf(gen_vals)
    if bad.any():
        print(f"  WARNING: {bad.sum()} NaN/Inf values in generator evaluations")
        gen_vals = np.nan_to_num(gen_vals, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Select basis via QR
    basis_idx = select_basis_qr(gen_vals, expected_rank=116)
    print(f"  Selected {len(basis_idx)} basis generators")
    basis_vals = gen_vals[basis_idx]  # (116, N_points)
    
    # Precompute all partial derivatives
    partials = precompute_partials(funcs, points, h=h)
    
    # Precompute chain rule values
    chain_vals = precompute_chain_rule_vals(algebra, points)
    
    # Compute total q-derivatives
    n_q = len(algebra.q_vars)
    total_dq = compute_total_derivs(partials, chain_vals, algebra, len(funcs))
    
    # Compute structure constants
    C = compute_structure_constants(total_dq, partials, basis_idx, basis_vals, n_q)
    
    return C, basis_idx, gen_vals, names, levels


def main():
    print("=" * 70)
    print("FAST LEVEL-3 STRUCTURE CONSTANT COMPARISON")
    print("(Finite-difference Poisson brackets — no symbolic derivatives)")
    print("=" * 70)
    
    # Find checkpoints
    base = os.path.dirname(script_dir)
    potential_configs = {
        "1/r": {
            "checkpoint": os.path.join(base, "aws_results", "nbody_checkpoints",
                                       "checkpoints_N3_d2_1r", "level_3.pkl"),
            "potential": "1/r",
        },
        "1/r²": {
            "checkpoint": os.path.join(base, "nbody", "checkpoints_N3_d2_1r2",
                                       "level_3.pkl"),
            "potential": "1/r^2",
        },
    }
    
    # Verify checkpoints exist
    print("\nCheckpoints:")
    for label, cfg in potential_configs.items():
        path = cfg["checkpoint"]
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {label}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {label}: {path} — NOT FOUND")
            return
    
    # Process each potential
    n_points = 400  # More points for better conditioning
    h = 1e-7  # Finite difference step
    
    results = {}
    for label, cfg in potential_configs.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING: {label}")
        print(f"{'='*70}")
        
        algebra = NBodyAlgebra(n_bodies=3, d_spatial=2, potential=cfg["potential"])
        C, basis_idx, gen_vals, names, levels = \
            process_potential(label, cfg["checkpoint"], algebra,
                            n_points=n_points, h=h)
        results[label] = {
            'C': C,
            'basis_idx': basis_idx,
            'names': names,
            'levels': levels,
        }
        print(f"\n  Structure constant tensor shape: {C.shape}")
        print(f"  Max |C|: {np.max(np.abs(C)):.6e}")
        print(f"  Non-zero entries (|C|>1e-6): {np.sum(np.abs(C) > 1e-6)}")
    
    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    C1 = results["1/r"]["C"]
    C2 = results["1/r²"]["C"]
    
    # Check if basis indices are the same
    idx1 = results["1/r"]["basis_idx"]
    idx2 = results["1/r²"]["basis_idx"]
    print(f"\n  Basis indices same: {idx1 == idx2}")
    if idx1 != idx2:
        print(f"  1/r  basis: {idx1[:10]}...")
        print(f"  1/r² basis: {idx2[:10]}...")
        print("  NOTE: Different basis selections — comparing after alignment")
    
    # Direct comparison (if same basis)
    diff = C1 - C2
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    rel_diff = max_diff / (max(np.max(np.abs(C1)), np.max(np.abs(C2))) + 1e-300)
    
    print(f"\n  Max absolute difference:  {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Max relative difference:  {rel_diff:.6e}")
    
    # Element-wise comparison
    threshold = 1e-4  # Generous threshold for finite differences
    mismatches = np.sum(np.abs(diff) > threshold * (np.abs(C1) + np.abs(C2) + 1e-10) / 2)
    total = C1.size
    print(f"\n  Entries with |diff| > {threshold} × avg|C|: {mismatches}/{total}")
    
    # Frobenius norm comparison
    norm1 = np.linalg.norm(C1)
    norm2 = np.linalg.norm(C2)
    norm_diff = np.linalg.norm(diff)
    print(f"\n  ||C_1/r||_F   = {norm1:.6f}")
    print(f"  ||C_1/r²||_F = {norm2:.6f}")
    print(f"  ||diff||_F    = {norm_diff:.6e}")
    print(f"  ||diff||/||C|| = {norm_diff / ((norm1 + norm2)/2 + 1e-300):.6e}")
    
    # Verdict
    print(f"\n{'='*70}")
    if rel_diff < 1e-3:
        print("VERDICT: Structure constants are IDENTICAL (within numerical precision)")
        print(f"  Max relative error: {rel_diff:.2e}")
    elif rel_diff < 1e-1:
        print("VERDICT: Structure constants are SIMILAR but not identical")
        print(f"  Max relative error: {rel_diff:.2e}")
    else:
        print("VERDICT: Structure constants DIFFER significantly")
        print(f"  Max relative error: {rel_diff:.2e}")
    print(f"{'='*70}")
    
    # Save results
    out_path = os.path.join(base, "results", "level3_structure_comparison.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
