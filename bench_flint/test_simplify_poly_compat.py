#!/usr/bin/env python3
"""
E0 — Poly-compat probe (gate for Phase E)
==========================================

Purpose
-------
Phase E will swap `cancel -> together` inside
`nbody/symbolic_rank_nbody.NBodySymbolicRank._simplify`. The downstream
pipeline calls `Poly(expand(simplified_expr), *phase_vars, domain='QQ')` to
extract the monomial-coefficient matrix used for exact rank determination.

This probe verifies the patch is safe at the Poly layer: for a handful of
raw L<=2 brackets, we check that

    set(Poly(expand(together(b))).monoms()) ==
    set(Poly(expand(cancel(b))).monoms())

and that the two Polys are identically equal (zero difference). If this
holds, the patch is provably equivalent to the existing engine for the
rank phase. If it fails (e.g. `together` leaves a Pow-of-Add that Poly
chokes on), Phase E is aborted.

Output
------
bench_flint/poly_compat_check.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import sympy as sp
from sympy import Add, Poly, cancel, expand, together

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "nbody"))

from symbolic_rank_nbody import NBodySymbolicRank  # noqa: E402


OUTPUT = REPO_ROOT / "bench_flint" / "poly_compat_check.json"


def gather_raw_brackets(engine, max_pairs_per_level=4):
    """Build raw (unsimplified) brackets up through L=2.

    Returns list of dicts {level, name, raw_expr}.
    """
    h_exprs = list(engine.algebra.hamiltonian_list)
    h_names = list(engine.algebra.hamiltonian_names)
    n_l0 = len(h_exprs)

    samples = []

    # L=1: raw brackets of pairwise Hamiltonians
    l1_exprs = []
    l1_names = []
    count = 0
    for i in range(n_l0):
        for j in range(i + 1, n_l0):
            raw = engine._poisson_bracket(h_exprs[i], h_exprs[j])
            l1_exprs.append(raw)
            l1_names.append(f"{{{h_names[i]},{h_names[j]}}}")
            if count < max_pairs_per_level:
                samples.append({
                    "level": 1, "name": l1_names[-1], "raw_expr": raw,
                })
                count += 1

    # For L=2 we need *some* L=1 generators to bracket against L=0 / L=1.
    # We use the *cancel*-simplified L=1s as "operands" so this probe is
    # independent of the patch under test (we only care about the L=2
    # *output* being Poly-extractable both ways).
    l1_simp = [cancel(e) for e in l1_exprs]

    # L=2: bracket a couple of L=1's against L=0 Hamiltonians and against
    # each other. Sample sparingly to keep the probe under a minute.
    count = 0
    for i in range(min(2, n_l0)):
        for j in range(min(3, len(l1_simp))):
            raw = engine._poisson_bracket(h_exprs[i], l1_simp[j])
            samples.append({
                "level": 2,
                "name": f"{{{h_names[i]},{l1_names[j]}}}",
                "raw_expr": raw,
            })
            count += 1
            if count >= max_pairs_per_level:
                break
        if count >= max_pairs_per_level:
            break

    # And one or two L=2 from L=1 x L=1
    if len(l1_simp) >= 2:
        raw = engine._poisson_bracket(l1_simp[0], l1_simp[1])
        samples.append({
            "level": 2,
            "name": f"{{{l1_names[0]},{l1_names[1]}}}",
            "raw_expr": raw,
        })

    return samples


def check_one_bracket(engine, sample):
    raw = sample["raw_expr"]
    phase = engine.phase_vars

    t0 = time.time()
    s_together = together(raw)
    t_together_simplify = time.time() - t0

    t0 = time.time()
    s_cancel = cancel(raw)
    t_cancel_simplify = time.time() - t0

    # Poly extraction (matching the pipeline: expand then Poly)
    t0 = time.time()
    p_together = Poly(expand(s_together), *phase, domain='QQ')
    t_together_poly = time.time() - t0

    t0 = time.time()
    p_cancel = Poly(expand(s_cancel), *phase, domain='QQ')
    t_cancel_poly = time.time() - t0

    monoms_together = set(p_together.monoms())
    monoms_cancel = set(p_cancel.monoms())

    monoms_match = monoms_together == monoms_cancel

    diff_poly = p_together - p_cancel
    polys_equal = diff_poly.is_zero

    return {
        "level": sample["level"],
        "name": sample["name"],
        "raw_count_ops": int(sp.count_ops(raw)),
        "together_count_ops": int(sp.count_ops(s_together)),
        "cancel_count_ops": int(sp.count_ops(s_cancel)),
        "together_simplify_s": round(t_together_simplify, 4),
        "cancel_simplify_s": round(t_cancel_simplify, 4),
        "together_poly_s": round(t_together_poly, 4),
        "cancel_poly_s": round(t_cancel_poly, 4),
        "n_monoms_together": len(monoms_together),
        "n_monoms_cancel": len(monoms_cancel),
        "monoms_match": bool(monoms_match),
        "polys_equal": bool(polys_equal),
        "monoms_only_in_together": sorted(monoms_together - monoms_cancel)[:5],
        "monoms_only_in_cancel": sorted(monoms_cancel - monoms_together)[:5],
    }


def main():
    print(f"sympy {sp.__version__}")
    print("E0: Poly-compat probe (gate for Phase E)")
    print("=" * 60)

    t_global = time.time()

    # Build a small engine: N=3, d=2, 1/r — same shape as the canonical pin.
    engine = NBodySymbolicRank(n_bodies=3, d_spatial=2, potential="1/r")
    print(f"Engine: N={engine.n_bodies}, d={engine.d_spatial}, "
          f"potential={engine.potential}, uses_u={engine.uses_u}")
    print(f"Phase vars ({len(engine.phase_vars)}): {engine.phase_vars}")

    print("\nGenerating raw L<=2 brackets ...")
    samples = gather_raw_brackets(engine, max_pairs_per_level=4)
    print(f"  -> {len(samples)} sample brackets")

    results = []
    for k, sample in enumerate(samples):
        print(f"\n  [{k+1}/{len(samples)}] {sample['name']} (L={sample['level']})")
        rec = check_one_bracket(engine, sample)
        marker = "OK " if (rec["monoms_match"] and rec["polys_equal"]) else "FAIL"
        print(f"    {marker}  monoms: T={rec['n_monoms_together']} "
              f"C={rec['n_monoms_cancel']}  match={rec['monoms_match']}  "
              f"polys_equal={rec['polys_equal']}")
        print(f"    count_ops: raw={rec['raw_count_ops']} "
              f"together={rec['together_count_ops']} "
              f"cancel={rec['cancel_count_ops']}")
        results.append(rec)

    n_total = len(results)
    n_match = sum(1 for r in results if r["monoms_match"] and r["polys_equal"])
    overall_match = (n_match == n_total)

    summary = {
        "sympy_version": sp.__version__,
        "engine": {
            "n_bodies": engine.n_bodies,
            "d_spatial": engine.d_spatial,
            "potential": engine.potential,
            "uses_u": bool(engine.uses_u),
            "phase_vars": [str(v) for v in engine.phase_vars],
        },
        "n_total": n_total,
        "n_match": n_match,
        "overall_match": overall_match,
        "elapsed_total_s": round(time.time() - t_global, 3),
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    os.replace(tmp, OUTPUT)

    print("\n" + "=" * 60)
    print(f"overall_match = {overall_match}  ({n_match}/{n_total})")
    print(f"elapsed = {summary['elapsed_total_s']}s")
    print(f"output  = {OUTPUT.relative_to(REPO_ROOT)}")

    return 0 if overall_match else 1


if __name__ == "__main__":
    sys.exit(main())
