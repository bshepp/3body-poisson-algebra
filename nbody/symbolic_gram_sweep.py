#!/usr/bin/env python3
"""
Symbolic Gram Determinant and Generator Norms vs Configuration
================================================================

Computes the Gram matrix G_ij(mu) and generator norms ||e_i(mu)||^2
as *exact* rational functions of the configuration parameter mu, then
evaluates them along a 1D sweep.

For a triangle configuration:
  body 1: origin (0, 0)
  body 2: (1, 0)
  body 3: (mu*cos(phi), mu*sin(phi))

we substitute positions and u_ij values, leaving only momenta p as
free variables. Each generator becomes a polynomial in p whose
coefficients are algebraic functions of mu. The "norm" is the sum
of squared coefficients (the squared L2 norm of the coefficient
vector in the p-monomial basis).

The Gram matrix G_ij(mu) = sum_alpha c_i,alpha(mu) * c_j,alpha(mu)
(where alpha indexes p-monomials) is computed symbolically for rank-17
(level 2), giving an exact rational function of mu.

det(G(mu)) measures the "volume" spanned by the generators at
configuration mu. When it vanishes, generators become linearly
dependent. When it diverges, generators blow up (singularity).

Usage
-----
    python symbolic_gram_sweep.py --potential 1/r --n-mu 200
    python symbolic_gram_sweep.py --potential r^4 --n-mu 200
"""

import os
import sys
import json
import argparse
from time import time

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
from sympy import (Symbol, Rational, Integer, sqrt, cos, sin, expand,
                   Poly, Add, cancel, simplify, pi, log as sp_log,
                   Abs, oo, zoo)

from exact_growth_nbody import NBodyAlgebra, COORD_LABELS
from symbolic_rank_nbody import NBodySymbolicRank


def build_config_substitution(mu_sym, phi_val, engine):
    """Build substitution dict: positions and u_ij in terms of mu.

    Triangle:
      body 1: (0, 0)
      body 2: (1, 0)
      body 3: (mu*cos(phi), mu*sin(phi))

    Returns substitution dict for q-variables and u-variables.
    """
    q_vars = engine.q_vars if hasattr(engine, 'q_vars') else engine.algebra.q_vars

    cos_phi = cos(phi_val)
    sin_phi = sin(phi_val)

    x3 = mu_sym * cos_phi
    y3 = mu_sym * sin_phi

    subs = {}

    if hasattr(engine, 'q_by_body'):
        q_by = engine.q_by_body
    else:
        q_by = engine.algebra.q_by_body

    subs[q_by[1][0]] = Integer(0)
    subs[q_by[1][1]] = Integer(0)
    subs[q_by[2][0]] = Integer(1)
    subs[q_by[2][1]] = Integer(0)
    subs[q_by[3][0]] = x3
    subs[q_by[3][1]] = y3

    if hasattr(engine, 'uses_u') and engine.uses_u:
        alg = engine.algebra
        r12_sq = Integer(1)
        r13_sq = x3**2 + y3**2
        r23_sq = (x3 - 1)**2 + y3**2

        r12_sq_expanded = expand(r12_sq)
        r13_sq_expanded = expand(r13_sq)
        r23_sq_expanded = expand(r23_sq)

        u12 = alg.u_by_pair[(1, 2)]
        u13 = alg.u_by_pair[(1, 3)]
        u23 = alg.u_by_pair[(2, 3)]

        subs[u12] = Integer(1)
        subs[u13] = 1 / sqrt(r13_sq_expanded)
        subs[u23] = 1 / sqrt(r23_sq_expanded)

    return subs


def substitute_and_collect(expr, subs_dict, p_vars, mu_sym):
    """Substitute configuration into a generator, collect as polynomial in p.

    Returns a dict {p_monomial_tuple: coeff(mu)} where coefficients are
    symbolic expressions in mu.
    """
    substituted = expr.xreplace(subs_dict)
    substituted = expand(substituted)

    try:
        poly = Poly(substituted, *p_vars, domain='EX')
        return poly.as_dict()
    except Exception:
        poly = Poly(substituted, *p_vars)
        return poly.as_dict()


def compute_gram_entry(coeffs_i, coeffs_j):
    """Compute G_ij = sum_alpha c_i,alpha * c_j,alpha."""
    result = Integer(0)
    common_monoms = set(coeffs_i.keys()) & set(coeffs_j.keys())
    for m in common_monoms:
        result += coeffs_i[m] * coeffs_j[m]
    return result


def compute_norm_squared(coeffs):
    """Compute ||e||^2 = sum_alpha c_alpha^2."""
    result = Integer(0)
    for c in coeffs.values():
        result += c**2
    return result


def run_gram_sweep(args):
    """Main symbolic Gram computation."""
    import numpy as np

    print("=" * 70)
    print("SYMBOLIC GRAM DETERMINANT SWEEP")
    print("=" * 70)
    print(f"Potential: {args.potential}")
    print(f"mu range: [{args.mu_min}, {args.mu_max}], {args.n_mu} points")
    print(f"phi: pi/3 ({sp.pi/3})")
    print(f"Max level: {args.max_level}")

    t_grand = time()

    potential_params = None
    if args.potential == 'composite' and args.composite:
        potential_params = [(Rational(c), int(p)) for c, p in args.composite]

    mu_sym = Symbol('mu', positive=True)
    phi_val = pi / 3

    print(f"\n--- Step 1: Building symbolic generators ---")
    t0 = time()
    engine = NBodySymbolicRank(3, 2, args.potential,
                               potential_params=potential_params)
    exprs, names, levels = engine.build_generators(args.max_level)
    n_gen = len(exprs)
    print(f"  {n_gen} generators through level {args.max_level} "
          f"[{time()-t0:.1f}s]")

    print(f"\n--- Step 2: Determining rank and selecting basis ---")
    poly_list, monom_list, monom_to_idx = engine.extract_monomial_matrix(exprs)
    rank_results = engine.compute_exact_rank(poly_list, monom_list,
                                             monom_to_idx, levels)
    final_rank = rank_results[max(rank_results.keys())]
    print(f"  Rank: {final_rank}")

    basis_indices = engine.select_basis(poly_list, monom_list, monom_to_idx,
                                        final_rank)
    basis_exprs = [exprs[i] for i in basis_indices]
    basis_names = [names[i] for i in basis_indices]
    print(f"  Selected {len(basis_indices)} basis generators")

    print(f"\n--- Step 3: Building configuration substitution ---")
    subs_dict = build_config_substitution(mu_sym, phi_val, engine)
    print(f"  Substitution has {len(subs_dict)} replacements")
    for var, val in subs_dict.items():
        print(f"    {var} -> {val}")

    if hasattr(engine, 'uses_u') and engine.uses_u:
        p_vars = list(engine.algebra.p_vars)
    else:
        p_vars = list(engine.p_vars)

    print(f"\n--- Step 4: Substituting into basis generators ---")
    t0 = time()
    basis_coeffs = []
    for idx, (expr, name) in enumerate(zip(basis_exprs, basis_names)):
        coeffs = substitute_and_collect(expr, subs_dict, p_vars, mu_sym)
        basis_coeffs.append(coeffs)
        n_terms = len(coeffs)
        print(f"  [{idx+1}/{final_rank}] {name}: {n_terms} p-monomials "
              f"[{time()-t0:.1f}s]", flush=True)

    print(f"\n--- Step 5: Computing symbolic norms ---")
    t0 = time()
    norm_sq_exprs = []
    for idx, (coeffs, name) in enumerate(zip(basis_coeffs, basis_names)):
        nsq = compute_norm_squared(coeffs)
        nsq_simplified = cancel(expand(nsq))
        norm_sq_exprs.append(nsq_simplified)
        print(f"  [{idx+1}/{final_rank}] ||{name}||^2 : "
              f"{len(Add.make_args(nsq_simplified))} terms "
              f"[{time()-t0:.1f}s]", flush=True)

    print(f"\n--- Step 6: Computing symbolic Gram matrix ---")
    t0 = time()
    G_symbolic = [[None]*final_rank for _ in range(final_rank)]
    for i in range(final_rank):
        G_symbolic[i][i] = norm_sq_exprs[i]
        for j in range(i+1, final_rank):
            gij = compute_gram_entry(basis_coeffs[i], basis_coeffs[j])
            gij_simplified = cancel(expand(gij))
            G_symbolic[i][j] = gij_simplified
            G_symbolic[j][i] = gij_simplified
        if (i + 1) % 5 == 0 or i == final_rank - 1:
            print(f"  row {i+1}/{final_rank} [{time()-t0:.1f}s]", flush=True)

    print(f"\n  Checking Gram entries for irrational terms...")
    w_expr = mu_sym**2 - mu_sym + 1
    sqrt_w = sp.sqrt(w_expr)
    n_irrational = 0
    for i in range(final_rank):
        for j in range(i, final_rank):
            expr = G_symbolic[i][j]
            test = expr.subs(sqrt_w, 0) - expr
            if test != 0:
                n_irrational += 1
    if n_irrational == 0:
        print(f"    All {final_rank*(final_rank+1)//2} entries are "
              f"rational in mu")
    else:
        print(f"    {n_irrational} entries contain sqrt(mu^2 - mu + 1)")
    has_sqrt = (n_irrational > 0)

    print(f"\n--- Step 7: Computing symbolic Gram determinant "
          f"(rationalized Bareiss) ---")
    t0 = time()
    gram_det_simplified = None
    det_computed = False
    lcm_denom = None
    lcm_denom_factored = None
    lcm_denom_in_mu = None
    lcm_denom_in_mu_factored = None
    det_numerator = None
    det_numerator_factored = None
    total_denom_power = None

    from sympy.polys.matrices import DomainMatrix as DM_class
    from sympy.polys.domains import QQ as QQ_domain

    s_sym = sp.Symbol('s', positive=True)
    relation = s_sym**2 - w_expr

    print(f"  Substituting sqrt(mu^2-mu+1) -> s and extracting "
          f"rational numer/denom in QQ[mu,s]...", flush=True)

    numers = [[None]*final_rank for _ in range(final_rank)]
    denoms = [[None]*final_rank for _ in range(final_rank)]
    n_forced = 0
    for i in range(final_rank):
        for j in range(final_rank):
            expr = G_symbolic[i][j]
            expr_s = expr.subs(sqrt_w, s_sym)
            expr_s = cancel(expand(expr_s))
            n, d = sp.fraction(expr_s)
            n, d = expand(n), expand(d)
            numers[i][j] = n
            denoms[i][j] = d
            if d != Integer(1) and d != 1:
                n_forced += 1
        if (i + 1) % 5 == 0 or i == final_rank - 1:
            print(f"    row {i+1}/{final_rank} [{time()-t0:.1f}s]",
                  flush=True)
    print(f"    {n_forced} entries had non-trivial denominators")

    poly_vars = [mu_sym, s_sym] if has_sqrt else [mu_sym]
    ok = True
    for i in range(final_rank):
        for j in range(final_rank):
            for label, e in [("n", numers[i][j]), ("d", denoms[i][j])]:
                try:
                    Poly(e, *poly_vars, domain='QQ')
                except Exception as exc:
                    print(f"    FAIL: G[{i},{j}] {label} not in "
                          f"QQ[mu,s]: {e} => {exc}")
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        print("  Cannot proceed with DomainMatrix; "
              "will use numerical det only")
    else:
        QQmus_ring = QQ_domain[mu_sym, s_sym] if has_sqrt else QQ_domain[mu_sym]

        print(f"  Computing LCM of all denominators...", flush=True)
        denom_set = {}
        for i in range(final_rank):
            for j in range(final_rank):
                d_str = str(denoms[i][j])
                if d_str not in denom_set:
                    denom_set[d_str] = denoms[i][j]
        unique_denoms = list(denom_set.values())
        print(f"    {len(unique_denoms)} distinct denominators")
        for idx, d in enumerate(unique_denoms[:10]):
            print(f"      [{idx}] {d}")

        lcm_denom = unique_denoms[0]
        for d in unique_denoms[1:]:
            lcm_denom = sp.lcm(
                Poly(lcm_denom, *poly_vars, domain='QQ'),
                Poly(d, *poly_vars, domain='QQ')).as_expr()
        lcm_denom = expand(lcm_denom)
        print(f"  LCM denominator D(mu,s) = {lcm_denom}")
        print(f"    D has {len(Add.make_args(lcm_denom))} terms")

        try:
            lcm_denom_factored = sp.factor(lcm_denom)
            print(f"    D factored = {lcm_denom_factored}")
        except Exception as e:
            print(f"    factoring D failed: {e}")
            lcm_denom_factored = lcm_denom

        lcm_denom_in_mu = expand(lcm_denom.subs(s_sym, sqrt_w))
        try:
            lcm_denom_in_mu_factored = sp.factor(lcm_denom_in_mu)
        except Exception:
            lcm_denom_in_mu_factored = lcm_denom_in_mu
        print(f"  D(mu) [substituting s->sqrt(w)] = "
              f"{lcm_denom_in_mu_factored}")

        print(f"  Clearing denominators...", flush=True)
        G_cleared = [[None]*final_rank for _ in range(final_rank)]
        for i in range(final_rank):
            for j in range(final_rank):
                if denoms[i][j] == Integer(1) or denoms[i][j] == 1:
                    multiplier = lcm_denom
                else:
                    multiplier_poly = sp.quo(
                        Poly(lcm_denom, *poly_vars, domain='QQ'),
                        Poly(denoms[i][j], *poly_vars, domain='QQ'),
                        *poly_vars, domain='QQ')
                    multiplier = multiplier_poly.as_expr()
                G_cleared[i][j] = expand(numers[i][j] * multiplier)

        print(f"  Reducing G_cleared entries modulo "
              f"s^2 = mu^2 - mu + 1...", flush=True)
        if has_sqrt:
            for i in range(final_rank):
                for j in range(final_rank):
                    p = Poly(G_cleared[i][j], s_sym)
                    reduced = Integer(0)
                    for monom, coeff in p.as_dict().items():
                        s_power = monom[0]
                        q, r = divmod(s_power, 2)
                        rational_part = expand(coeff * w_expr**q)
                        if r == 1:
                            reduced += rational_part * s_sym
                        else:
                            reduced += rational_part
                    G_cleared[i][j] = expand(reduced)

        print(f"  Building DomainMatrix over QQ[mu,s]...", flush=True)
        dm_rows = []
        for i in range(final_rank):
            row = []
            for j in range(final_rank):
                row.append(QQmus_ring.from_sympy(G_cleared[i][j]))
            dm_rows.append(row)
        print(f"    DomainMatrix constructed [{time()-t0:.1f}s]")

        dm = DM_class(dm_rows, (final_rank, final_rank), QQmus_ring)

        print(f"  Computing det(G_cleared) via Bareiss...", flush=True)
        t_det = time()
        det_cleared = dm.det()
        det_elapsed = time() - t_det
        print(f"    det computed in {det_elapsed:.1f}s")

        det_poly_raw = QQmus_ring.to_sympy(det_cleared)
        det_poly_raw = expand(det_poly_raw)
        print(f"    raw det has "
              f"{len(Add.make_args(det_poly_raw))} terms")

        if has_sqrt:
            print(f"  Reducing det modulo s^2 = mu^2 - mu + 1...",
                  flush=True)
            det_p = Poly(det_poly_raw, s_sym)
            det_rational = Integer(0)
            det_irrational = Integer(0)
            for monom, coeff in det_p.as_dict().items():
                s_power = monom[0]
                q, r = divmod(s_power, 2)
                coeff_exp = expand(coeff * w_expr**q)
                if r == 0:
                    det_rational += coeff_exp
                else:
                    det_irrational += coeff_exp
            det_rational = expand(det_rational)
            det_irrational = expand(det_irrational)
            print(f"    Rational part: "
                  f"{len(Add.make_args(det_rational))} terms")
            print(f"    Irrational coeff (of sqrt(w)): "
                  f"{len(Add.make_args(det_irrational))} terms")
            if det_irrational == 0:
                print(f"    sqrt cancels in det -- det is rational!")
                det_numerator = det_rational
            else:
                print(f"    det has irrational part; using full form")
                det_numerator = det_rational + det_irrational * sqrt_w
        else:
            det_numerator = det_poly_raw

        total_denom_expr = lcm_denom_in_mu ** final_rank
        gram_det_simplified = cancel(det_numerator / total_denom_expr)
        print(f"  det(G(mu)) = det(G_cleared) / D(mu)^{final_rank}")
        print(f"    simplified det has "
              f"{len(Add.make_args(gram_det_simplified))} terms")

        print(f"\n  Factoring numerator and denominator...", flush=True)
        t_fac = time()
        det_numer_final, det_denom_final = sp.fraction(
            cancel(gram_det_simplified))
        det_numer_final = expand(det_numer_final)
        det_denom_final = expand(det_denom_final)

        try:
            det_numerator_factored = sp.factor(det_numer_final)
            print(f"    Numerator factored [{time()-t_fac:.1f}s]")
            nf_str = str(det_numerator_factored)
            if len(nf_str) > 500:
                print(f"    numer factors = {nf_str[:500]}...")
            else:
                print(f"    numer factors = {det_numerator_factored}")
        except Exception as e:
            print(f"    Numerator factoring failed: {e}")
            det_numerator_factored = det_numer_final

        try:
            det_denom_factored = sp.factor(det_denom_final)
            print(f"    Denominator factored = {det_denom_factored}")
        except Exception as e:
            print(f"    Denominator factoring failed: {e}")
            det_denom_factored = det_denom_final

        print(f"\n  === SINGULARITY POLYNOMIAL ===")
        print(f"  Poles (blow-up) at zeros of: {det_denom_factored}")
        print(f"  Zeros (degeneration) at zeros of: "
              f"{det_numerator_factored}")
        print(f"  ================================")

        det_computed = True
        print(f"\n  Total Step 7 time: {time()-t0:.1f}s")

    print(f"\n--- Step 8: Evaluating along mu sweep ---")
    mu_vals = np.linspace(args.mu_min, args.mu_max, args.n_mu)
    t0 = time()

    norm_sq_funcs = []
    for nsq in norm_sq_exprs:
        norm_sq_funcs.append(sp.lambdify(mu_sym, nsq, modules="mpmath"))

    gram_entry_funcs = [[None]*final_rank for _ in range(final_rank)]
    for i in range(final_rank):
        for j in range(i, final_rank):
            gram_entry_funcs[i][j] = sp.lambdify(
                mu_sym, G_symbolic[i][j], modules="mpmath")
            gram_entry_funcs[j][i] = gram_entry_funcs[i][j]

    if det_computed:
        det_func = sp.lambdify(mu_sym, gram_det_simplified, modules="mpmath")
    print(f"  Lambdified [{time()-t0:.1f}s]")

    results = []
    t_sweep = time()

    for k, mu_val in enumerate(mu_vals):
        mu_f = float(mu_val)
        point = {"mu": mu_f, "status": "ok"}

        try:
            norms = []
            for i, f in enumerate(norm_sq_funcs):
                val = float(f(mu_f))
                norms.append(val)
            point["norm_sq"] = norms
            point["norm_sq_sum"] = sum(norms)
            point["norm_sq_max"] = max(norms)
            point["norm_sq_min"] = min(norms)

            G_numerical = np.zeros((final_rank, final_rank))
            for i in range(final_rank):
                for j in range(i, final_rank):
                    val = float(gram_entry_funcs[i][j](mu_f))
                    G_numerical[i][j] = val
                    G_numerical[j][i] = val

            eigs = np.linalg.eigvalsh(G_numerical)
            eigs.sort()
            point["gram_eigenvalues"] = eigs.tolist()
            point["gram_min_eigenvalue"] = float(eigs[0])
            point["gram_max_eigenvalue"] = float(eigs[-1])
            point["gram_condition_number"] = (
                float(eigs[-1] / eigs[0]) if eigs[0] > 0 else float('inf'))

            point["gram_det_from_eigs"] = float(np.prod(eigs))
            point["gram_log_det"] = float(np.sum(np.log(np.abs(eigs) + 1e-300)))

            if det_computed:
                point["gram_det_exact"] = float(det_func(mu_f))

            point["gram_rank"] = int(np.sum(
                eigs > 1e-10 * eigs[-1])) if eigs[-1] > 0 else 0

        except Exception as e:
            point["status"] = "error"
            point["error"] = str(e)

        results.append(point)

        if (k + 1) % 10 == 0 or k == 0 or k == len(mu_vals) - 1:
            elapsed = time() - t_sweep
            rate = (k + 1) / elapsed if elapsed > 0 else 0
            eta = (args.n_mu - k - 1) / rate if rate > 0 else 0
            status = point.get("status", "?")
            nsq_sum = point.get("norm_sq_sum", float("nan"))
            cond = point.get("gram_condition_number", float("nan"))
            det_val = point.get("gram_det_from_eigs", float("nan"))
            print(f"  [{k+1}/{args.n_mu}] mu={mu_f:.4f} "
                  f"||e||^2_sum={nsq_sum:.4e} cond={cond:.4e} "
                  f"det={det_val:.4e} [{elapsed:.1f}s, ETA {eta:.0f}s]",
                  flush=True)

    print(f"\n--- Step 9: Saving results ---")
    pot_tag = args.potential.replace("/", "").replace("^", "")
    if args.potential == 'composite' and potential_params:
        powers = [str(p) for _, p in potential_params]
        pot_tag = f"composite_u{'_'.join(powers)}"

    out_dir = os.path.join("results", "symbolic_gram_sweep", pot_tag)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "potential": args.potential,
        "potential_tag": pot_tag,
        "n_mu": args.n_mu,
        "mu_range": [args.mu_min, args.mu_max],
        "phi": "pi/3",
        "max_level": args.max_level,
        "rank": final_rank,
        "basis_indices": basis_indices,
        "basis_names": basis_names,
        "det_computed_symbolically": det_computed,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    norm_expressions = {}
    for i, (name, nsq) in enumerate(zip(basis_names, norm_sq_exprs)):
        norm_expressions[name] = str(nsq)
    with open(os.path.join(out_dir, "norm_expressions.json"), "w") as f:
        json.dump(norm_expressions, f, indent=2)

    if det_computed and gram_det_simplified is not None:
        lcm_mu = lcm_denom_in_mu if lcm_denom is not None else None
        lcm_mu_fac = (lcm_denom_in_mu_factored
                      if lcm_denom is not None else None)

        with open(os.path.join(out_dir, "gram_det_expression.txt"), "w") as f:
            f.write(f"det(G(mu)) = {gram_det_simplified}\n\n")
            if lcm_mu is not None:
                f.write(f"LCM denominator D(mu) = {lcm_mu}\n")
                f.write(f"D(mu) factored = {lcm_mu_fac}\n")
                f.write(f"D(mu,s) [s=sqrt(mu^2-mu+1)] = {lcm_denom}\n")
                f.write(f"D(mu,s) factored = {lcm_denom_factored}\n\n")
            if det_numerator is not None:
                f.write(f"det(G_cleared) [numerator before division] = "
                        f"{det_numerator}\n\n")
            if det_numerator_factored is not None:
                f.write(f"Numerator of det(G) factored = "
                        f"{det_numerator_factored}\n\n")
            det_n, det_d = sp.fraction(cancel(gram_det_simplified))
            f.write(f"Final numerator = {det_n}\n")
            f.write(f"Final denominator = {det_d}\n")
            try:
                det_d_fac = sp.factor(det_d)
                f.write(f"Final denominator factored = {det_d_fac}\n\n")
            except Exception:
                det_d_fac = det_d
            try:
                det_n_fac = sp.factor(det_n)
            except Exception:
                det_n_fac = det_n
            f.write(f"Poles (denominator roots) are at zeros of: "
                    f"{det_d_fac}\n")
            f.write(f"Zeros (degeneration loci) are at zeros of: "
                    f"{det_n_fac}\n")

        singularity_data = {
            "lcm_denominator": str(lcm_mu) if lcm_mu else None,
            "lcm_denominator_factored": (
                str(lcm_mu_fac) if lcm_mu_fac else None),
            "det_gram_simplified": str(gram_det_simplified),
            "det_numerator_factored": (
                str(det_numerator_factored)
                if det_numerator_factored else None),
            "final_numerator": str(det_n),
            "final_denominator": str(det_d),
            "total_denom_power": final_rank,
        }
        with open(os.path.join(out_dir, "singularity_polynomial.json"),
                  "w") as f:
            json.dump(singularity_data, f, indent=2)

    gram_str = {}
    for i in range(final_rank):
        for j in range(i, final_rank):
            key = f"G_{i}_{j}"
            gram_str[key] = str(G_symbolic[i][j])
    with open(os.path.join(out_dir, "gram_matrix_expressions.json"), "w") as f:
        json.dump(gram_str, f, indent=2)

    with open(os.path.join(out_dir, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    ok_results = [r for r in results if r.get("status") == "ok"]
    summary = {
        "potential": args.potential,
        "n_total": len(results),
        "n_ok": len(ok_results),
    }
    if ok_results:
        import numpy as np
        norm_sums = [r["norm_sq_sum"] for r in ok_results]
        conds = [r["gram_condition_number"] for r in ok_results
                 if r["gram_condition_number"] < float('inf')]
        dets = [r["gram_det_from_eigs"] for r in ok_results]
        min_eigs = [r["gram_min_eigenvalue"] for r in ok_results]

        summary["norm_sq_sum_range"] = [min(norm_sums), max(norm_sums)]
        summary["gram_condition_range"] = (
            [min(conds), max(conds)] if conds else None)
        summary["gram_det_range"] = [min(dets), max(dets)]
        summary["gram_min_eig_range"] = [min(min_eigs), max(min_eigs)]
        summary["gram_rank_values"] = sorted(set(
            r["gram_rank"] for r in ok_results))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(os.path.join(out_dir, "gram_sweep_data.npz"),
             mu=np.array([r["mu"] for r in ok_results]),
             norm_sq_sum=np.array([r["norm_sq_sum"] for r in ok_results]),
             gram_condition=np.array([
                 r["gram_condition_number"] for r in ok_results]),
             gram_det=np.array([r["gram_det_from_eigs"] for r in ok_results]),
             gram_min_eig=np.array([
                 r["gram_min_eigenvalue"] for r in ok_results]),
             gram_max_eig=np.array([
                 r["gram_max_eigenvalue"] for r in ok_results]))

    elapsed_total = time() - t_grand
    print(f"\n{'='*70}")
    print(f"SYMBOLIC GRAM SWEEP COMPLETE")
    print(f"  Potential: {args.potential}")
    print(f"  {len(ok_results)}/{len(results)} points OK")
    print(f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    if ok_results:
        print(f"  ||e||^2 sum range: "
              f"[{summary['norm_sq_sum_range'][0]:.6e}, "
              f"{summary['norm_sq_sum_range'][1]:.6e}]")
        if summary.get("gram_condition_range"):
            print(f"  Gram condition range: "
                  f"[{summary['gram_condition_range'][0]:.6e}, "
                  f"{summary['gram_condition_range'][1]:.6e}]")
        print(f"  Gram det range: "
              f"[{summary['gram_det_range'][0]:.6e}, "
              f"{summary['gram_det_range'][1]:.6e}]")
        print(f"  Gram rank: {summary['gram_rank_values']}")
    print(f"  Results saved to: {out_dir}/")
    if det_computed:
        print(f"  Gram determinant expression saved (exact in mu)")
        if lcm_denom_in_mu_factored is not None:
            print(f"  LCM denominator D(mu) = "
                  f"{lcm_denom_in_mu_factored}")
        if det_numerator_factored is not None:
            nf_str = str(det_numerator_factored)
            if len(nf_str) > 200:
                print(f"  det(G) numerator factored = {nf_str[:200]}...")
            else:
                print(f"  det(G) numerator factored = "
                      f"{det_numerator_factored}")
        det_n, det_d = sp.fraction(cancel(gram_det_simplified))
        print(f"  Poles at zeros of: {sp.factor(det_d)}")
    print(f"{'='*70}")

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Symbolic Gram determinant and generator norms vs mu")
    ap.add_argument("--potential", default="1/r",
                    help="Potential type (1/r, r^4, composite)")
    ap.add_argument("--composite", nargs=2, type=str, action='append',
                    metavar=('COEFF', 'POWER'),
                    help="Composite potential term")
    ap.add_argument("--max-level", type=int, default=2,
                    help="Max bracket level (default: 2 for rank 17)")
    ap.add_argument("--n-mu", type=int, default=200,
                    help="Number of mu grid points")
    ap.add_argument("--mu-min", type=float, default=0.05,
                    help="Minimum mu")
    ap.add_argument("--mu-max", type=float, default=5.0,
                    help="Maximum mu")
    args = ap.parse_args()

    run_gram_sweep(args)


if __name__ == "__main__":
    main()
