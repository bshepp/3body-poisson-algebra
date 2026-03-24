#!/usr/bin/env python3
"""
Phase 1: Validate survey gravitational results on corrected SymPy.
Re-runs three_galaxies (1:2:3) and binary_star_planet (1:1:0.001)
to level 3 to check if [3,5,13,69] was real or a SymPy artifact.
"""
import sys, os, json, pickle, numpy as np
from time import time
from sympy import Integer, Rational

sys.setrecursionlimit(200000)
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nbody"))
from exact_growth_nbody import NBodyAlgebra

CONFIGS = {
    "three_galaxies": {
        "label": "Three Merging Galaxies (1:2:3)",
        "masses": {1: Integer(1), 2: Integer(2), 3: Integer(3)},
        "charges": None,
    },
    "binary_star_planet": {
        "label": "Binary Star + Planet (1:1:0.001)",
        "masses": {1: Integer(1), 2: Integer(1), 3: Rational(1, 1000)},
        "charges": None,
    },
    "equal_mass_control": {
        "label": "Equal Masses Control (1:1:1)",
        "masses": {1: Integer(1), 2: Integer(1), 3: Integer(1)},
        "charges": None,
    },
}

MAX_LEVEL = 3
N_SAMPLES = 500
SEED = 42


def run_config(name, cfg):
    print(f"\n{'='*70}")
    print(f"CONFIG: {cfg['label']}")
    print(f"  masses = {cfg['masses']}")
    print(f"  charges = {cfg['charges']}")
    print(f"  max_level = {MAX_LEVEL}, samples = {N_SAMPLES}")
    print(f"{'='*70}")

    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2,
        potential="1/r",
        masses=cfg["masses"],
        charges=cfg["charges"],
    )
    level_dims = alg.compute_growth(
        max_level=MAX_LEVEL,
        n_samples=N_SAMPLES,
        seed=SEED,
    )
    dims = [level_dims[lv] for lv in range(MAX_LEVEL + 1)]
    print(f"\n  DIMENSION SEQUENCE: {dims}")
    print(f"  Expected [3,6,17,116]: {dims == [3, 6, 17, 116]}")

    return {"name": name, "label": cfg["label"], "dims": dims,
            "masses": str(cfg["masses"])}


def main():
    print("=" * 70)
    print("SURVEY VALIDATION: Corrected SymPy Pipeline")
    print(f"Python {sys.version}")
    try:
        import sympy
        print(f"SymPy {sympy.__version__}")
    except Exception:
        pass
    print(f"NumPy {np.__version__}")
    print("=" * 70)

    results = {}
    for name, cfg in CONFIGS.items():
        t0 = time()
        res = run_config(name, cfg)
        res["elapsed_s"] = time() - t0
        results[name] = res
        print(f"\n  Elapsed: {res['elapsed_s']:.1f}s")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, res in results.items():
        match = "MATCH" if res["dims"] == [3, 6, 17, 116] else "DIFFERENT"
        print(f"  {res['label']:<45s} {res['dims']}  [{match}]")

    all_match = all(r["dims"] == [3, 6, 17, 116] for r in results.values())
    print(f"\n  All match [3,6,17,116]: {all_match}")
    if all_match:
        print("  CONCLUSION: [3,5,13,69] was a SymPy version artifact.")
        print("  Mass invariance holds universally for gravitational 3-body.")
    else:
        print("  CONCLUSION: Mass ratio DOES affect the dimension sequence.")
        print("  The transition is REAL and needs further investigation.")

    summary = {name: {"dims": r["dims"], "elapsed_s": r["elapsed_s"]}
               for name, r in results.items()}
    with open("validate_survey_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to validate_survey_results.json")


if __name__ == "__main__":
    main()
