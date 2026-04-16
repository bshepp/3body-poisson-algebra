#!/usr/bin/env python3
"""
Named Molecular Systems
========================

Computes dimension sequences for recognizable molecular and chemical
systems to add cross-disciplinary appeal to the dataset.

Systems:
  - H3+: trihydrogen cation (three protons, simplest polyatomic molecule)
  - O3: ozone nuclei (three oxygen-16 nuclei)

These are Coulomb 1/r systems with specific nuclear masses. Mass invariance
guarantees [3, 6, 17, 116], but named entries connect to chemistry.

Usage:
    python named_molecular_systems.py
"""

import sys
import os
import json
from time import time, strftime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra
from sympy import Integer

RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), "..",
    "results", "expansion_dimseq", "expansion_dimseq_completion.json"
)

SYSTEMS = {
    "h3_plus": {
        "label": "H3+ (trihydrogen cation)",
        "masses": {1: 1836, 2: 1836, 3: 1836},
        "charges": {1: 1, 2: 1, 3: 1},
        "potential": "1/r",
        "category": "molecular",
        "description": "Simplest polyatomic molecule, key in interstellar chemistry",
    },
    "ozone_nuclei": {
        "label": "O3 (ozone nuclei)",
        "masses": {1: 29164, 2: 29164, 3: 29164},
        "charges": {1: 8, 2: 8, 3: 8},
        "potential": "1/r",
        "category": "molecular",
        "description": "Ozone nuclear framework, three oxygen-16 nuclei",
    },
}

N_SAMPLES = 2000
MAX_LEVEL = 3
SEED = 42


def run_system(name, spec):
    """Run a single named system."""
    print(f"\n{'=' * 60}")
    print(f"  {spec['label']}")
    print(f"  Masses: {spec['masses']}")
    print(f"  Charges: {spec['charges']}")
    print(f"{'=' * 60}", flush=True)

    masses = spec["masses"]
    charges = spec["charges"]

    alg = NBodyAlgebra(
        n_bodies=3,
        d_spatial=1,
        potential="1/r",
        charges=charges,
        masses=masses,
    )

    t0 = time()
    level_dims = alg.compute_growth(
        max_level=MAX_LEVEL,
        n_samples=N_SAMPLES,
        seed=SEED,
        resume=True,
    )
    elapsed = time() - t0

    dims = [level_dims[lv] for lv in range(MAX_LEVEL + 1)]
    matches = dims == [3, 6, 17, 116]

    print(f"  Result: {dims}")
    print(f"  Matches universal: {matches}")
    print(f"  Time: {elapsed:.1f}s")

    return dims, elapsed


def main():
    print("=" * 60)
    print("NAMED MOLECULAR SYSTEMS")
    print("=" * 60)

    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    for name, spec in SYSTEMS.items():
        if name in manifest and manifest[name].get("status") == "complete":
            print(f"\n  {name}: already complete, skipping")
            continue

        dims, elapsed = run_system(name, spec)

        manifest[name] = {
            "status": "complete",
            "result": dims,
            "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        with open(RESULTS_FILE, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print("ALL SYSTEMS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
