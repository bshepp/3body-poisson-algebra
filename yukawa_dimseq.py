#!/usr/bin/env python3
"""
Yukawa potential dimension sequence survey.

Computes Poisson algebra dimension sequences for the Yukawa potential
V(r) = exp(-mu*r)/r = u * exp(-mu/u)  (where u = 1/r)
at N=3 bodies, d=1, max_level=3.

Strategy: Uses Taylor-expansion composite representation.
  exp(-mu/u) = sum_{k=0}^{K} (-mu/u)^k / k!
  => V = u * exp(-mu/u) ≈ sum_{k=0}^{K} (-mu)^k / k! * u^{1-k}
This converts the transcendental potential into a polynomial composite
that the existing NBodyAlgebra pipeline handles efficiently.

d=1 results are equivalent to d=2 for dimension sequences (proven
by exact symbolic rank for other potentials). Physical systems are
therefore computed at d=1 for efficiency.
"""

import sys
import os
import json
from time import time, strftime
from math import factorial

sys.setrecursionlimit(100000)
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbody"))
from exact_growth_nbody import NBodyAlgebra
from sympy import Rational, Integer

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "yukawa_dimseq.json")

TAYLOR_K = 3

MU_VALUES = [
    Rational(1, 10),   # 0.1  (long Debye length, dusty plasma regime)
    Rational(1, 2),    # 0.5
    Rational(7, 10),   # 0.7  (nuclear force scale)
    Rational(1),       # 1.0
    Rational(2),       # 2.0  (short-range screening)
    Rational(5),       # 5.0  (very short range)
]

PHYSICAL_SYSTEMS = {
    "tritium_he3": {
        "label": "Tritium / He-3 (3 Nucleons)",
        "category": "nuclear",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": None,
        "mu": Rational(7, 10),
        "description": "Three nucleons with Yukawa nuclear force (mu ~ 0.7 fm^-1)",
    },
    "dusty_plasma": {
        "label": "Three Dust Grains in Dusty Plasma",
        "category": "plasma",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": {1: 1, 2: 1, 3: 1},
        "mu": Rational(1, 10),
        "description": "Screened Coulomb / Debye-Yukawa (long Debye length, mu ~ 0.1)",
    },
    "p_n_n_scattering": {
        "label": "Proton-Neutron-Neutron",
        "category": "nuclear",
        "masses": {1: Integer(1), 2: Rational(10014, 10000), 3: Rational(10014, 10000)},
        "charges": None,
        "mu": Rational(7, 10),
        "description": "p-n-n scattering (proton slightly lighter than neutron)",
    },
}

N_SAMPLES = 500
MAX_LEVEL = 3


def yukawa_taylor_terms(mu, K):
    """Build composite potential_params for Taylor expansion of Yukawa."""
    terms = []
    for k in range(K + 1):
        coeff = Rational((-mu) ** k, factorial(k))
        power = 1 - k
        if coeff != 0:
            terms.append((coeff, power))
    return terms


def compute_yukawa_taylor(N, d, mu, K, max_level, n_samples, seed=42,
                          masses=None, charges=None, label=""):
    """Compute dimension sequence for Yukawa via Taylor composite."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  N={N} d={d} mu={mu} Taylor K={K}")
    print(f"{'='*60}", flush=True)

    terms = yukawa_taylor_terms(mu, K)
    print(f"  Composite terms: {len(terms)} (u-powers {1} down to {1-K})")

    t0 = time()
    try:
        alg = NBodyAlgebra(
            n_bodies=N,
            d_spatial=d,
            potential="composite",
            potential_params=terms,
            masses=masses,
            charges=charges,
        )
        dims = alg.compute_growth(
            max_level=max_level,
            n_samples=n_samples,
            seed=seed,
            resume=True,
        )
        elapsed = time() - t0
        seq = [dims[lv] for lv in range(max_level + 1)]
        print(f"\n  RESULT (Taylor K={K}): {seq}  ({elapsed:.1f}s)")
        return {
            "dimension_sequence": seq,
            "computation_time_s": round(elapsed, 1),
            "method": "numerical_svd",
            "n_samples": n_samples,
            "taylor_order": K,
        }
    except Exception as e:
        elapsed = time() - t0
        print(f"\n  FAILED (Taylor K={K}): {e}  ({elapsed:.1f}s)")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "computation_time_s": round(elapsed, 1),
            "taylor_order": K,
        }


def load_existing():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"mu_sweep": [], "physical_systems": []}


def save_results(data):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved to {RESULTS_PATH}")


def main():
    print("=" * 70)
    print("YUKAWA POTENTIAL DIMENSION SEQUENCE SURVEY")
    print("(via Taylor-expansion composite, K=3)")
    print("=" * 70)
    print(f"  Started: {strftime('%Y-%m-%d %H:%M:%S')}")

    data = load_existing()
    completed_mus = {
        str(r.get("mu"))
        for r in data.get("mu_sweep", [])
        if "error" not in r
    }
    completed_phys = {
        r.get("system_name")
        for r in data.get("physical_systems", [])
        if "error" not in r
    }

    # --- Phase 1: mu sweep at d=1 ---
    print("\n\n" + "#" * 70)
    print("# PHASE 1: mu sweep at N=3, d=1")
    print("#" * 70)

    for mu in MU_VALUES:
        if str(mu) in completed_mus:
            print(f"\n  Skipping d=1 mu={mu} (already computed)")
            continue

        result = compute_yukawa_taylor(
            N=3, d=1, mu=mu, K=TAYLOR_K,
            max_level=MAX_LEVEL, n_samples=N_SAMPLES,
            label=f"Yukawa mu={mu} (d=1)",
        )
        result.update({"N": 3, "d": 1, "mu": str(mu), "potential": "yukawa"})
        data.setdefault("mu_sweep", []).append(result)
        save_results(data)

    # --- Phase 2: Named physical systems (d=1) ---
    print("\n\n" + "#" * 70)
    print("# PHASE 2: Named physical systems (d=1)")
    print("#" * 70)

    for sys_name, cfg in PHYSICAL_SYSTEMS.items():
        if sys_name in completed_phys:
            print(f"\n  Skipping {sys_name} (already computed)")
            continue

        mu = cfg["mu"]
        result = compute_yukawa_taylor(
            N=3, d=1, mu=mu, K=TAYLOR_K,
            max_level=MAX_LEVEL, n_samples=N_SAMPLES,
            masses=cfg.get("masses"), charges=cfg.get("charges"),
            label=f"{cfg['label']}",
        )

        result.update({
            "system_name": sys_name,
            "system_label": cfg["label"],
            "category": cfg["category"],
            "N": 3,
            "d": 1,
            "mu": str(mu),
            "potential": "yukawa",
            "masses": {str(k): str(v) for k, v in cfg["masses"].items()},
            "charges": cfg.get("charges"),
            "description": cfg["description"],
        })
        data.setdefault("physical_systems", []).append(result)
        save_results(data)

    # --- Summary ---
    print("\n\n" + "=" * 70)
    print("YUKAWA SURVEY -- SUMMARY")
    print("=" * 70)

    print("\n  mu sweep results:")
    for r in data.get("mu_sweep", []):
        if "error" in r:
            print(f"  N={r['N']} d={r['d']} mu={r['mu']}  ERROR: {r['error'][:60]}")
        else:
            seq = r["dimension_sequence"]
            match = "UNIVERSAL" if seq == [3, 6, 17, 116] else f"DIFFERENT: {seq}"
            print(f"  N={r['N']} d={r['d']} mu={r['mu']}  {match}  ({r['computation_time_s']:.1f}s)")

    print("\n  Physical systems:")
    for r in data.get("physical_systems", []):
        if "error" in r:
            print(f"  {r['system_label']}  ERROR: {r['error'][:60]}")
        else:
            seq = r["dimension_sequence"]
            match = "UNIVERSAL" if seq == [3, 6, 17, 116] else f"DIFFERENT: {seq}"
            print(f"  {r['system_label']}  {match}  ({r['computation_time_s']:.1f}s)")

    print(f"\n  Results saved to {RESULTS_PATH}")
    print(f"  Completed: {strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
