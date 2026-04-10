#!/usr/bin/env python3
"""
1D Cross-Section of Algebraic Structure Across Parameter Space
==============================================================

Sweeps mu from near-collision (mu->0) to large separation (mu=5) at
fixed phi=pi/3, computing *numerical* structure constants at each point
for a given potential. Compares 1/r (singular) vs r^4 (smooth) to detect
whether the singularity leaves an imprint in the algebra's embedding.

The symbolic generators are the same at every point (the abstract Lie
algebra is universal). What changes is the *numerical values* of the
structure constants when the algebra is evaluated at phase-space samples
localized near a particular triangle configuration.

Usage
-----
    # 1/r potential, 200 mu points
    python structure_cross_section.py --potential 1/r --n-mu 200

    # r^4 potential
    python structure_cross_section.py --potential r^4 --n-mu 200

    # Both potentials in one run
    python structure_cross_section.py --potential 1/r --n-mu 200
    python structure_cross_section.py --potential r^4 --n-mu 200
"""

import os
import sys
import json
import argparse
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import sympy as sp
from sympy import Rational, Integer, Symbol, diff, expand

from exact_growth_nbody import NBodyAlgebra, COORD_LABELS
from symbolic_rank_nbody import (
    NBodySymbolicRank,
    compute_killing_form,
    compute_derived_series,
    compute_lower_central_series,
    compute_center,
)


def build_localized_samples(mu, phi, n_bodies, d_spatial, n_samples,
                            seed=42, epsilon=0.05, mom_range=1.0,
                            min_sep=0.01):
    """Sample phase space localized around a triangle defined by (mu, phi).

    Triangle vertices:
      body 1: origin
      body 2: (1, 0)
      body 3: (mu*cos(phi), mu*sin(phi))

    Positions perturbed by Gaussian noise of scale epsilon.
    Momenta drawn uniformly.
    """
    base_pos = np.array([
        0.0, 0.0,
        1.0, 0.0,
        mu * np.cos(phi),
        mu * np.sin(phi),
    ])

    rng = np.random.RandomState(seed)
    n_phase = 2 * n_bodies * d_spatial
    n_q = n_bodies * d_spatial

    pts = np.zeros((n_samples, n_phase))
    u_cols = np.zeros((n_samples, 3))
    accepted = 0
    attempts = 0

    while accepted < n_samples and attempts < n_samples * 500:
        q = base_pos + rng.randn(n_q) * epsilon
        p = rng.randn(n_q) * mom_range

        dx12 = q[0] - q[2]; dy12 = q[1] - q[3]
        dx13 = q[0] - q[4]; dy13 = q[1] - q[5]
        dx23 = q[2] - q[4]; dy23 = q[3] - q[5]

        r12 = np.sqrt(dx12**2 + dy12**2)
        r13 = np.sqrt(dx13**2 + dy13**2)
        r23 = np.sqrt(dx23**2 + dy23**2)

        if min(r12, r13, r23) < min_sep:
            attempts += 1
            continue

        pts[accepted, :n_q] = q
        pts[accepted, n_q:] = p
        u_cols[accepted] = [1.0 / r12, 1.0 / r13, 1.0 / r23]
        accepted += 1
        attempts += 1

    if accepted < n_samples:
        print(f"  WARNING: only {accepted}/{n_samples} samples accepted "
              f"at mu={mu:.4f}")
        pts = pts[:accepted]
        u_cols = u_cols[:accepted]

    return pts, u_cols


def lambdify_all_derivatives(engine, exprs, names):
    """Pre-compute symbolic derivatives and lambdify everything once.

    Returns:
      gen_funcs: list of callables for generator evaluation
      dq_funcs:  list of lists of callables (n_gen x n_q)
      dp_funcs:  list of lists of callables (n_gen x n_q)
    """
    n_gen = len(exprs)
    n_q = len(engine.q_vars)
    all_vars = list(engine.all_vars) if hasattr(engine, 'all_vars') else (
        list(engine.q_vars) + list(engine.p_vars) + list(engine.u_vars))

    print(f"  Lambdifying {n_gen} generators...", flush=True)
    t0 = time()
    gen_funcs = []
    for idx, expr in enumerate(exprs):
        gen_funcs.append(sp.lambdify(all_vars, expr, modules="numpy"))
        if (idx + 1) % 20 == 0:
            print(f"    generators: {idx+1}/{n_gen}  [{time()-t0:.1f}s]",
                  flush=True)
    print(f"    generators done [{time()-t0:.1f}s]")

    print(f"  Computing symbolic derivatives ({n_gen} x {2*n_q})...",
          flush=True)
    t1 = time()
    dq_funcs = []
    dp_funcs = []
    for idx, expr in enumerate(exprs):
        dq_row = []
        dp_row = []
        for k in range(n_q):
            dfdq = expand(engine.total_deriv(expr, engine.q_vars[k]))
            dfdp = expand(diff(expr, engine.p_vars[k]))
            dq_row.append(sp.lambdify(all_vars, dfdq, modules="numpy"))
            dp_row.append(sp.lambdify(all_vars, dfdp, modules="numpy"))
        dq_funcs.append(dq_row)
        dp_funcs.append(dp_row)
        if (idx + 1) % 10 == 0 or idx == n_gen - 1:
            print(f"    derivs: {idx+1}/{n_gen}  [{time()-t1:.1f}s]",
                  flush=True)
    print(f"    derivatives done [{time()-t1:.1f}s]")

    return gen_funcs, dq_funcs, dp_funcs


def lambdify_all_polynomial(engine, exprs, names):
    """For polynomial potentials (r^4) that don't use u variables.

    The NBodySymbolicRank with r^4 sets uses_u=False and algebra=None,
    so we can't use engine.total_deriv. Instead, derivatives are plain
    partial derivatives since there's no chain rule.
    """
    n_gen = len(exprs)
    n_q = len(engine.q_vars)
    phase_vars = list(engine.phase_vars)

    print(f"  Lambdifying {n_gen} generators (polynomial)...", flush=True)
    t0 = time()
    gen_funcs = []
    for idx, expr in enumerate(exprs):
        gen_funcs.append(sp.lambdify(phase_vars, expr, modules="numpy"))
        if (idx + 1) % 20 == 0:
            print(f"    generators: {idx+1}/{n_gen}  [{time()-t0:.1f}s]",
                  flush=True)
    print(f"    generators done [{time()-t0:.1f}s]")

    print(f"  Computing symbolic derivatives ({n_gen} x {2*n_q})...",
          flush=True)
    t1 = time()
    dq_funcs = []
    dp_funcs = []
    for idx, expr in enumerate(exprs):
        dq_row = []
        dp_row = []
        for k in range(n_q):
            dfdq = expand(diff(expr, engine.q_vars[k]))
            dfdp = expand(diff(expr, engine.p_vars[k]))
            dq_row.append(sp.lambdify(phase_vars, dfdq, modules="numpy"))
            dp_row.append(sp.lambdify(phase_vars, dfdp, modules="numpy"))
        dq_funcs.append(dq_row)
        dp_funcs.append(dp_row)
        if (idx + 1) % 10 == 0 or idx == n_gen - 1:
            print(f"    derivs: {idx+1}/{n_gen}  [{time()-t1:.1f}s]",
                  flush=True)
    print(f"    derivatives done [{time()-t1:.1f}s]")

    return gen_funcs, dq_funcs, dp_funcs


def evaluate_at_samples(gen_funcs, dq_funcs, dp_funcs, Z_qp, Z_u,
                        uses_u, n_gen, n_q):
    """Evaluate generators and derivatives at sample points.

    Returns:
      eval_matrix: (n_samples, n_gen)
      deriv_dq:    (n_gen, n_q, n_samples)
      deriv_dp:    (n_gen, n_q, n_samples)
    """
    n_samples = Z_qp.shape[0]
    n_phase = Z_qp.shape[1]

    if uses_u:
        args = ([Z_qp[:, i] for i in range(n_phase)] +
                [Z_u[:, i] for i in range(Z_u.shape[1])])
    else:
        args = [Z_qp[:, i] for i in range(n_phase)]

    eval_matrix = np.zeros((n_samples, n_gen))
    for idx in range(n_gen):
        val = gen_funcs[idx](*args)
        eval_matrix[:, idx] = np.atleast_1d(np.asarray(val, dtype=float)
                                             ).ravel()[:n_samples]

    deriv_dq = np.zeros((n_gen, n_q, n_samples))
    deriv_dp = np.zeros((n_gen, n_q, n_samples))
    for idx in range(n_gen):
        for k in range(n_q):
            val = dq_funcs[idx][k](*args)
            deriv_dq[idx, k, :] = np.atleast_1d(
                np.asarray(val, dtype=float)).ravel()[:n_samples]
            val = dp_funcs[idx][k](*args)
            deriv_dp[idx, k, :] = np.atleast_1d(
                np.asarray(val, dtype=float)).ravel()[:n_samples]

    return eval_matrix, deriv_dq, deriv_dp


def compute_numerical_structure_constants(eval_matrix, deriv_dq, deriv_dp,
                                          basis_indices, n_q):
    """Compute structure constants numerically from derivative evaluations.

    For basis generators e_a, e_b, computes {e_a, e_b} numerically using
    the Poisson bracket formula, then solves for C[a,b,k] via least-squares:

        bracket_values = eval_matrix[:, basis] @ C[a,b,:]

    Returns C as a (rank, rank, rank) float array.
    """
    n_samples = eval_matrix.shape[0]
    rank = len(basis_indices)

    basis_matrix = eval_matrix[:, basis_indices]

    C = np.zeros((rank, rank, rank))

    for a in range(rank):
        i = basis_indices[a]
        for b in range(a + 1, rank):
            j = basis_indices[b]

            bracket_vals = np.zeros(n_samples)
            for k in range(n_q):
                bracket_vals += (deriv_dq[i, k, :] * deriv_dp[j, k, :]
                                 - deriv_dp[i, k, :] * deriv_dq[j, k, :])

            coeffs, residuals, _, _ = np.linalg.lstsq(
                basis_matrix, bracket_vals, rcond=None)

            C[a, b, :] = coeffs
            C[b, a, :] = -coeffs

    return C


def get_canonical_basis(eval_matrix, rank):
    """Select canonical basis indices via column-pivoted QR."""
    from scipy.linalg import qr
    norms = np.linalg.norm(eval_matrix, axis=0)
    norms[norms < 1e-15] = 1.0
    M = eval_matrix / norms
    _, _, perm = qr(M, pivoting=True)
    return sorted(perm[:rank].tolist())


def extract_at_point(mu, phi, gen_funcs, dq_funcs, dp_funcs,
                     canonical_basis, rank, n_gen, n_q, uses_u,
                     n_samples=300, seed=42):
    """Extract full algebraic structure at a single (mu, phi) point."""
    t0 = time()

    Z_qp, Z_u = build_localized_samples(
        mu, phi, n_bodies=3, d_spatial=2,
        n_samples=n_samples, seed=seed,
        epsilon=max(0.02, mu * 0.05),
        min_sep=max(0.005, mu * 0.01))

    actual_samples = Z_qp.shape[0]
    if actual_samples < rank + 5:
        return {
            "mu": mu, "phi": phi, "status": "insufficient_samples",
            "n_samples": actual_samples,
        }

    eval_matrix, deriv_dq, deriv_dp = evaluate_at_samples(
        gen_funcs, dq_funcs, dp_funcs, Z_qp, Z_u,
        uses_u, n_gen, n_q)

    basis_matrix = eval_matrix[:, canonical_basis]
    rank_check = np.linalg.matrix_rank(basis_matrix, tol=1e-8)
    if rank_check < rank:
        return {
            "mu": mu, "phi": phi, "status": "rank_drop",
            "rank_at_point": int(rank_check), "expected_rank": rank,
        }

    C = compute_numerical_structure_constants(
        eval_matrix, deriv_dq, deriv_dp, canonical_basis, n_q)

    K, k_eigs, signature = compute_killing_form(C)
    derived_dims, is_solvable, solv_len = compute_derived_series(C)
    lcs_dims, is_nilpotent, nilp_class = compute_lower_central_series(C)
    center_basis, center_dim = compute_center(C)

    sc_norm = float(np.linalg.norm(C))
    nonzero_mask = np.abs(C) > 1e-10
    n_nonzero = int(np.sum(nonzero_mask))
    nonzero_values = C[nonzero_mask].tolist()

    norms_local = np.linalg.norm(basis_matrix, axis=0)
    norms_local[norms_local < 1e-15] = 1.0
    M_local = basis_matrix / norms_local
    G = M_local.T @ M_local / actual_samples
    gram_det = float(np.linalg.det(G))
    cond_number = float(np.linalg.cond(M_local))

    elapsed = time() - t0

    return {
        "mu": float(mu),
        "phi": float(phi),
        "status": "ok",
        "sc_norm": sc_norm,
        "n_nonzero_sc": n_nonzero,
        "nonzero_sc_values": nonzero_values,
        "killing_eigenvalues": k_eigs.tolist(),
        "killing_signature": list(signature),
        "killing_trace": float(np.trace(K)),
        "derived_series": derived_dims,
        "is_solvable": is_solvable,
        "solvability_length": solv_len,
        "lower_central_series": lcs_dims,
        "is_nilpotent": is_nilpotent,
        "nilpotency_class": nilp_class,
        "center_dimension": int(center_dim),
        "gram_determinant": gram_det,
        "condition_number": cond_number,
        "n_samples_used": actual_samples,
        "time_seconds": elapsed,
    }


def run_cross_section(args):
    """Main cross-section computation."""
    print("=" * 70)
    print("STRUCTURE CONSTANT CROSS-SECTION")
    print("=" * 70)
    print(f"Potential: {args.potential}")
    print(f"mu range: [{args.mu_min}, {args.mu_max}], {args.n_mu} points")
    print(f"phi (fixed): {args.phi:.4f} rad ({np.degrees(args.phi):.1f} deg)")
    print(f"Samples per point: {args.n_samples}")
    print(f"Max level: {args.max_level}")

    t_grand = time()

    potential_params = None
    if args.potential == 'composite' and args.composite:
        potential_params = [(Rational(c), int(p)) for c, p in args.composite]

    print(f"\n--- Step 1: Building symbolic generators ---")
    t0 = time()
    engine = NBodySymbolicRank(3, 2, args.potential,
                               potential_params=potential_params)
    exprs, names, levels = engine.build_generators(args.max_level)
    n_gen = len(exprs)
    n_q = 6
    print(f"  {n_gen} generators through level {args.max_level} "
          f"[{time()-t0:.1f}s]")

    print(f"\n--- Step 2: Determining rank ---")
    poly_list, monom_list, monom_to_idx = engine.extract_monomial_matrix(exprs)
    rank_results = engine.compute_exact_rank(poly_list, monom_list,
                                             monom_to_idx, levels)
    final_rank = rank_results[max(rank_results.keys())]
    print(f"  Rank through level {args.max_level}: {final_rank}")

    print(f"\n--- Step 3: Lambdifying generators and derivatives ---")
    if engine.uses_u:
        algebra = engine.algebra
        gen_funcs, dq_funcs, dp_funcs = lambdify_all_derivatives(
            algebra, exprs, names)
    else:
        gen_funcs, dq_funcs, dp_funcs = lambdify_all_polynomial(
            engine, exprs, names)

    print(f"\n--- Step 4: Establishing canonical basis at reference ---")
    ref_mu, ref_phi = 1.0, np.pi / 3
    Z_ref, Z_u_ref = build_localized_samples(
        ref_mu, ref_phi, n_bodies=3, d_spatial=2,
        n_samples=args.n_samples, seed=args.seed)

    eval_ref, _, _ = evaluate_at_samples(
        gen_funcs, dq_funcs, dp_funcs, Z_ref, Z_u_ref,
        engine.uses_u, n_gen, n_q)
    canonical_basis = get_canonical_basis(eval_ref, final_rank)
    print(f"  Canonical basis: {len(canonical_basis)} generators "
          f"(indices: {canonical_basis[:5]}...)")

    mu_vals = np.linspace(args.mu_min, args.mu_max, args.n_mu)

    pot_tag = args.potential.replace("/", "").replace("^", "")
    if args.potential == 'composite' and potential_params:
        powers = [str(p) for _, p in potential_params]
        pot_tag = f"composite_u{'_'.join(powers)}"

    out_dir = os.path.join("results", "structure_cross_section", pot_tag)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "potential": args.potential,
        "potential_tag": pot_tag,
        "n_mu": args.n_mu,
        "mu_range": [args.mu_min, args.mu_max],
        "phi": args.phi,
        "n_samples_per_point": args.n_samples,
        "max_level": args.max_level,
        "rank": final_rank,
        "canonical_basis_indices": canonical_basis,
        "n_generators": n_gen,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n--- Step 5: Sweeping {args.n_mu} mu values ---")
    results = []
    t_sweep = time()

    for i, mu in enumerate(mu_vals):
        result = extract_at_point(
            mu, args.phi, gen_funcs, dq_funcs, dp_funcs,
            canonical_basis, final_rank, n_gen, n_q, engine.uses_u,
            n_samples=args.n_samples, seed=args.seed + i)

        results.append(result)

        if (i + 1) % 10 == 0 or i == 0 or i == len(mu_vals) - 1:
            elapsed = time() - t_sweep
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (args.n_mu - i - 1) / rate if rate > 0 else 0
            status = result.get("status", "?")
            sc_norm = result.get("sc_norm", float("nan"))
            center = result.get("center_dimension", -1)
            print(f"  [{i+1}/{args.n_mu}] mu={mu:.4f} "
                  f"status={status} sc_norm={sc_norm:.4f} "
                  f"center={center} "
                  f"[{elapsed:.1f}s, ETA {eta:.0f}s]", flush=True)

        if (i + 1) % 50 == 0:
            _save_checkpoint(out_dir, results, mu_vals[:i+1])

    print(f"\n--- Step 6: Saving results ---")
    _save_checkpoint(out_dir, results, mu_vals)

    ok_results = [r for r in results if r.get("status") == "ok"]

    summary = {
        "potential": args.potential,
        "n_total": len(results),
        "n_ok": len(ok_results),
        "n_rank_drop": sum(1 for r in results
                           if r.get("status") == "rank_drop"),
        "n_insufficient": sum(1 for r in results
                              if r.get("status") == "insufficient_samples"),
    }

    if ok_results:
        sc_norms = [r["sc_norm"] for r in ok_results]
        summary["sc_norm_range"] = [min(sc_norms), max(sc_norms)]
        summary["sc_norm_mean"] = np.mean(sc_norms)
        summary["sc_norm_std"] = float(np.std(sc_norms))

        center_dims = [r["center_dimension"] for r in ok_results]
        summary["center_dim_values"] = sorted(set(center_dims))
        summary["center_dim_uniform"] = len(set(center_dims)) == 1

        kill_sigs = [tuple(r["killing_signature"]) for r in ok_results]
        summary["killing_signature_values"] = [list(s)
                                                for s in sorted(set(kill_sigs))]
        summary["killing_signature_uniform"] = len(set(kill_sigs)) == 1

        solv_lens = [r.get("solvability_length") for r in ok_results
                     if r.get("solvability_length") is not None]
        if solv_lens:
            summary["solvability_length_values"] = sorted(set(solv_lens))

        cond_numbers = [r["condition_number"] for r in ok_results]
        summary["condition_number_range"] = [min(cond_numbers),
                                              max(cond_numbers)]

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    elapsed_total = time() - t_grand
    print(f"\n{'='*70}")
    print(f"CROSS-SECTION COMPLETE")
    print(f"  Potential: {args.potential}")
    print(f"  {len(ok_results)}/{len(results)} points OK")
    print(f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    if ok_results:
        print(f"  SC norm range: [{summary['sc_norm_range'][0]:.6f}, "
              f"{summary['sc_norm_range'][1]:.6f}]")
        print(f"  SC norm std/mean: "
              f"{summary['sc_norm_std']/summary['sc_norm_mean']:.4f}")
        print(f"  Center dim: {summary['center_dim_values']}")
        print(f"  Killing signature: {summary['killing_signature_values']}")
    print(f"  Results saved to: {out_dir}/")
    print(f"{'='*70}")

    return summary


def _save_checkpoint(out_dir, results, mu_vals):
    """Save intermediate results."""
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return

    mu_arr = np.array([r["mu"] for r in ok])
    sc_norm_arr = np.array([r["sc_norm"] for r in ok])
    center_arr = np.array([r["center_dimension"] for r in ok])
    cond_arr = np.array([r["condition_number"] for r in ok])

    np.savez(os.path.join(out_dir, "cross_section_data.npz"),
             mu=mu_arr, sc_norm=sc_norm_arr,
             center_dim=center_arr, condition_number=cond_arr)


def main():
    ap = argparse.ArgumentParser(
        description="1D cross-section of algebraic structure vs mu")
    ap.add_argument("--potential", default="1/r",
                    help="Potential type (1/r, 1/r^2, r^4, composite)")
    ap.add_argument("--composite", nargs=2, type=str, action='append',
                    metavar=('COEFF', 'POWER'),
                    help="Composite potential term: coeff power")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Max bracket level (default: 2 for rank 17)")
    ap.add_argument("--n-mu", type=int, default=200,
                    help="Number of mu grid points (default: 200)")
    ap.add_argument("--mu-min", type=float, default=0.05,
                    help="Minimum mu (default: 0.05)")
    ap.add_argument("--mu-max", type=float, default=5.0,
                    help="Maximum mu (default: 5.0)")
    ap.add_argument("--phi", type=float, default=None,
                    help="Fixed phi in radians (default: pi/3)")
    ap.add_argument("--n-samples", type=int, default=300,
                    help="Phase-space samples per point (default: 300)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.phi is None:
        args.phi = np.pi / 3

    run_cross_section(args)


if __name__ == "__main__":
    main()
