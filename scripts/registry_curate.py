#!/usr/bin/env python3
"""Apply canonical-state patches to selected registry entries.

This is a one-shot curation script: it loads ``registry/experiments.yaml``,
applies a hand-edited dict of patches keyed by entry id, and writes the file
back. Re-running it is idempotent (later runs overwrite the same fields with
the same values).

The patches encode what we know from README.md, docs/project_status.md, and
docs/gap_workplan.md about each headline script. Anything not patched here
remains ``status: needs_review`` and can be curated incrementally.

Usage::

    python scripts/registry_curate.py            # dry-run
    python scripts/registry_curate.py --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from registry.loader import REGISTRY_PATH, load  # noqa: E402

# --------------------------------------------------------------------------- #
# Canonical patches keyed by entry id.
# Each value is a dict of field overrides applied on top of the bootstrapped
# entry. Only listed fields are overwritten; others are preserved.
# --------------------------------------------------------------------------- #

PATCHES: dict[str, dict] = {
    # --- Core engines --------------------------------------------------------
    "exact_growth": {
        "category": "engine",
        "scope": "nbody/N3",
        "status": "complete",
        "description": (
            "Core symbolic Poisson bracket engine for N=3, planar. Builds "
            "pairwise Hamiltonians, iterates brackets to a configurable level, "
            "evaluates numerically, and runs SVD to extract dimension sequence. "
            "Reference: produces [3, 6, 17, 116] for N=3, 1/r."
        ),
        "outputs": ["checkpoints/level_3.pkl"],
        "latest_result": {
            "date": "2026-04-15",
            "summary": "Levels 0-3 give [3, 6, 17, 116] for N=3, 1/r (canonical baseline).",
            "sequence": [3, 6, 17, 116],
        },
        "runtime": "~10 min local",
        "run_command": "python exact_growth.py",
        "related_docs": ["README.md", "docs/project_status.md"],
        "tags": ["core", "engine", "n3", "symbolic"],
    },
    "exact_growth_cm": {
        "category": "engine",
        "scope": "nbody/N3",
        "status": "complete",
        "description": "Calogero-Moser (1/r^2) variant of the core engine. Produces [3, 6, 17, 116].",
        "latest_result": {"date": "2026-04-08", "sequence": [3, 6, 17, 116], "summary": "Universal."},
        "run_command": "python exact_growth_cm.py",
        "tags": ["core", "engine", "calogero-moser", "n3"],
    },
    "nbody_exact_growth_nbody": {
        "category": "engine",
        "scope": "nbody",
        "status": "complete",
        "description": (
            "NBodyAlgebra: arbitrary N, d, potential (single or composite), masses, charges, "
            "external trap. The general-purpose engine the rest of the project sits on."
        ),
        "tags": ["core", "engine", "general"],
    },
    "nbody_quantum_algebra": {
        "category": "engine",
        "scope": "nbody",
        "status": "complete",
        "description": (
            "QuantumNBodyAlgebra: Moyal-bracket version of NBodyAlgebra. "
            "Validated via Jacobi and hbar->0 limit. Produces [3,6,17,117] for N=3, 1/r."
        ),
        "latest_result": {
            "date": "2026-04-10",
            "summary": "Quantum N=3 1/r gives [3, 6, 17, 117] - one extra generator.",
            "sequence": [3, 6, 17, 117],
        },
        "tags": ["core", "engine", "quantum", "moyal"],
    },
    "nbody_symbolic_rank_nbody": {
        "category": "dimseq",
        "scope": "nbody",
        "status": "complete",
        "description": (
            "Exact symbolic rank over Q (or Q(m1,m2,m3)) for arbitrary N, d, potential. "
            "Supersedes float64 SVD for definitive dimension determination, "
            "especially at extreme mass ratios where SVD conditioning fails."
        ),
        "outputs": ["results/symbolic_rank/"],
        "run_command": "python nbody/symbolic_rank_nbody.py -N 3 -d 1 --max-level 3",
        "tags": ["dimseq", "exact", "symbolic", "rank"],
    },
    "symbolic_rank": {
        "category": "dimseq",
        "scope": "nbody/N3",
        "status": "complete",
        "description": (
            "Symbolic rank over Q(m1,m2,m3) for N=3 - the mass-invariance proof. "
            "Establishes [3, 6, 17, 116] as an algebraic theorem for all positive masses."
        ),
        "latest_result": {
            "date": "2026-04-10",
            "summary": "Mass invariance proved over Q(m1,m2,m3); rank = 116 at L3.",
            "sequence": [3, 6, 17, 116],
        },
        "runtime": "~3.2 h on AWS r6i.8xlarge",
        "tags": ["dimseq", "exact", "mass-invariance", "headline"],
    },

    # --- Headline dimseq results --------------------------------------------
    "nbody_run_n4_potential_universality": {
        "category": "dimseq",
        "scope": "nbody/N4",
        "status": "complete",
        "description": "N=4 potential universality: 1/r, 1/r^2, 1/r^3, log(r) all give [6, 14, 62].",
        "outputs": ["nbody/n4_potential_universality_results.json"],
        "latest_result": {
            "date": "2026-04-11",
            "summary": "All four potentials give [6, 14, 62]; SVD gaps 7e12-2e13.",
            "sequence": [6, 14, 62],
        },
        "tags": ["dimseq", "n4", "universality"],
    },
    "yukawa_dimseq": {
        "category": "dimseq",
        "scope": "nbody/N3",
        "status": "complete",
        "description": (
            "Yukawa potential survey via Taylor-composite K=3. Confirms [3,6,17,116] "
            "for mu in {0.1, 0.5, 0.7, 1.0, 2.0, 5.0} plus tritium/He-3, dusty plasma, p-n-n."
        ),
        "outputs": ["results/yukawa_dimseq.json"],
        "latest_result": {
            "date": "2026-04-16",
            "summary": "All 9 Yukawa configs give [3, 6, 17, 116].",
            "sequence": [3, 6, 17, 116],
        },
        "runtime": "~10 min local; ~15 min AWS r6i.2xlarge",
        "run_command": "python yukawa_dimseq.py",
        "related_docs": ["docs/project_status.md"],
        "tags": ["dimseq", "yukawa", "universality", "composite", "screening"],
    },
    "nbody_named_molecular_systems": {
        "category": "dimseq",
        "scope": "nbody/N3",
        "status": "complete",
        "description": "H3+ (m=1836) and O3 (m=29164) - both [3, 6, 17, 116]. Mass invariance for named molecules.",
        "latest_result": {
            "date": "2026-04-15",
            "summary": "H3+ and O3 both give [3, 6, 17, 116].",
            "sequence": [3, 6, 17, 116],
        },
        "related_supplemental": ["Molecular three-body systems. Triat.txt"],
        "tags": ["dimseq", "molecular", "mass-invariance"],
    },
    "nbody_charge_sweep_d1": {
        "category": "dimseq",
        "scope": "nbody/N3",
        "status": "complete",
        "description": (
            "Charge magnitude sweep at d=1: integer charges (+1,+q,-1) for q=1..20 "
            "all give [3, 6, 17, 116]. Phase 3 of the charge sensitivity campaign."
        ),
        "outputs": ["results/charge_sensitivity/charge_sweep_qqn_d1.json"],
        "latest_result": {"date": "2026-04-15", "summary": "All 20 charge magnitudes universal.", "sequence": [3, 6, 17, 116]},
        "tags": ["dimseq", "charge", "universality"],
    },
    "nbody_level2_exponent_sweep": {
        "category": "dimseq",
        "scope": "nbody/N3",
        "status": "complete",
        "description": "500-exponent sweep at L2 for 1/r^n and r^n. 498/500 give universal L2 dim 17; only r^1 (15) and r^2 (13).",
        "outputs": ["results/level2_exponent_sweep.json"],
        "latest_result": {"date": "2026-04-15", "summary": "Universality confirmed at L2 across 500 exponents.", "sequence": []},
        "tags": ["dimseq", "exponent-sweep", "L2"],
    },
    "l3_exponent_sweep": {
        "category": "dimseq",
        "scope": "nbody/N3",
        "status": "complete",
        "description": "Extended L3 exponent sweep: 76 successful data points across 1/r^n and r^n.",
        "outputs": ["results/l3_exponent_sweep_extended.json"],
        "latest_result": {"date": "2026-04-16", "summary": "Continuous universality at L3 except r^1, r^{near 2}.", "sequence": []},
        "tags": ["dimseq", "exponent-sweep", "L3"],
    },

    # --- Structure constants -------------------------------------------------
    "nbody_isomorphism_test": {
        "category": "structure",
        "scope": "nbody/N3",
        "status": "complete",
        "description": (
            "Cross-potential isomorphism test for L2 algebras. 12 non-harmonic 17-dim "
            "algebras canonically isomorphic via Killing eigenvalues, ad-rank multisets, "
            "Casimir trace. r^1 identified as filiform L_{5,2}. r^3 L3 deep analysis."
        ),
        "latest_result": {"date": "2026-04-15", "summary": "Isomorphism confirmed across 12 non-harmonic potentials at L2.", "sequence": []},
        "related_supplemental": ["EXTRACT_ALGEBRA_STRUCTURE.md"],
        "tags": ["structure", "isomorphism", "killing", "L2"],
    },
    "nbody_compare_level3_structure": {
        "category": "structure",
        "scope": "nbody/N3",
        "status": "wip",
        "description": "Level-3 structure comparison across potentials. AWS-scale workload.",
        "tags": ["structure", "L3"],
    },
    "nbody_run_composite_test": {
        "category": "dimseq",
        "scope": "nbody/N3",
        "status": "complete",
        "description": (
            "Composite potential universality: V = sum_k c_k * r^{-p_k}. "
            "Two-term and three-term composites both give [3, 6, 17, 116]."
        ),
        "latest_result": {"date": "2026-04-08", "summary": "Composites confirmed universal.", "sequence": [3, 6, 17, 116]},
        "tags": ["dimseq", "composite", "universality"],
    },

    # --- GUE / primes --------------------------------------------------------
    "primes_run_gue_logas": {
        "category": "dimseq",
        "scope": "primes",
        "status": "complete",
        "description": (
            "Dyson log-gas computation across 4 configs (pure log, GUE composite, "
            "Penning trap, harmonic-only). All singular configs give [3, 6, 17, 116]: "
            "GUE belongs to the same universality class as Newtonian gravity."
        ),
        "outputs": ["primes/results/gue_comparison.json"],
        "latest_result": {
            "date": "2026-04-11",
            "summary": "Universality CONFIRMED for log-gas / GUE composite / Penning trap.",
            "sequence": [3, 6, 17, 116],
        },
        "runtime": "~3 min on AWS r6i.2xlarge",
        "tags": ["dimseq", "gue", "primes", "zeta", "headline"],
    },
    "primes_run_quantum_gue": {
        "category": "quantum",
        "scope": "primes",
        "status": "complete",
        "description": "Quantum Moyal bracket on GUE composite. Confirms [3, 6, 17, 116] under hbar deformation (no +1 growth, unlike pure 1/r).",
        "outputs": ["primes/results/quantum_gue.json"],
        "tags": ["quantum", "gue", "moyal"],
    },
    "primes_level2_spectral_analysis": {
        "category": "spectral",
        "scope": "primes",
        "status": "complete",
        "description": (
            "Kirillov coadjoint orbit spectral analysis - Bohigas-Giannoni-Schmit "
            "conjecture in algebraic structure. 1/r tensor: GUE-GSE; r^2: Poisson."
        ),
        "outputs": ["primes/figures/"],
        "tags": ["spectral", "BGS", "kirillov", "primes"],
    },

    # --- Atlas / scans -------------------------------------------------------
    "stability_atlas": {
        "category": "atlas",
        "scope": "nbody/N3",
        "status": "complete",
        "description": "Exact-engine atlas scanner. Drives most of the (mu, phi) shape-sphere atlases.",
        "tags": ["atlas", "scanner"],
    },
    "multi_epsilon_atlas": {
        "category": "atlas",
        "scope": "nbody/N3",
        "status": "complete",
        "description": "Multi-epsilon adaptive structure atlas. Supports --charges, multiprocessing, spot instances.",
        "tags": ["atlas", "adaptive", "multi-epsilon"],
    },
    "targeted_adaptive_scan": {
        "category": "atlas",
        "scope": "nbody/N3",
        "status": "complete",
        "description": "High-resolution adaptive scans of named regions (Lagrange, Euler, etc.) for any potential.",
        "tags": ["atlas", "adaptive", "lagrange", "euler"],
    },
    "n4_atlas_1d": {
        "category": "atlas",
        "scope": "nbody/N4",
        "status": "complete",
        "description": "First atlas data for N=4: three 1D slices. All 300 points show rank 62 (no drops).",
        "outputs": ["results/n4_atlas_1d.json"],
        "latest_result": {"date": "2026-04-16", "summary": "Complete rank stability at rank 62.", "sequence": []},
        "tags": ["atlas", "n4"],
    },

    # --- Quantum / Bell / contextuality -------------------------------------
    "nbody_bell_test": {
        "category": "quantum",
        "scope": "nbody",
        "status": "complete",
        "description": "Bell test on the pairwise algebra. Local results in nbody/bell_test_results/.",
        "tags": ["quantum", "bell"],
    },
    "nbody_contextuality_test": {
        "category": "quantum",
        "scope": "nbody",
        "status": "complete",
        "description": "Contextuality test (composite-potential variant).",
        "outputs": ["nbody/contextuality_results/"],
        "tags": ["quantum", "contextuality"],
    },
    "nbody_energy_bound_search": {
        "category": "quantum",
        "scope": "nbody/N3",
        "status": "complete",
        "description": (
            "Energy-bound / 117th generator commutant search. "
            "Quantum commutant=40 < classical=41: quantization removes one conservation law. "
            "Energy bound via this approach not possible (Case C)."
        ),
        "tags": ["quantum", "energy-bound", "117th"],
    },
    "nbody_analyze_117th": {
        "category": "analysis",
        "scope": "nbody/N3",
        "status": "complete",
        "description": "Analysis of the 117th quantum generator: sum-of-squares, negative semi-definite, P3 octupole.",
        "tags": ["analysis", "117th", "quantum"],
    },

    # --- N-body scaling ------------------------------------------------------
    "nbody_run_n_scaling_aws": {
        "category": "dimseq",
        "scope": "nbody",
        "status": "complete",
        "description": "AWS launcher for the N-body rank scaling sweep. Confirmed N=3..8 L2; N=4 L3 = [6,14,62,1260].",
        "outputs": ["nbody/n_body_scaling_results.json"],
        "tags": ["dimseq", "n-scaling", "aws"],
    },
    "nbody_sweep_nbody_ranks": {
        "category": "dimseq",
        "scope": "nbody",
        "status": "complete",
        "description": "Local N-body rank sweep harness.",
        "tags": ["dimseq", "n-scaling"],
    },

    # --- Dataset / pipeline --------------------------------------------------
    "dataset_build_dataset": {
        "category": "dataset",
        "scope": "dataset",
        "status": "complete",
        "description": (
            "Build the 13-table HF dataset from ~200 result JSONs. "
            "Outputs Parquet; uploaded to huggingface.co/datasets/bshepp/pairwise-poisson-algebras."
        ),
        "outputs": ["dataset/output/"],
        "run_command": "python dataset/build_dataset.py",
        "related_docs": ["dataset/README.md"],
        "tags": ["dataset", "huggingface", "pipeline"],
    },
    "dataset_validate_dataset": {
        "category": "test",
        "scope": "dataset",
        "status": "complete",
        "description": "Validation suite for all dataset splits.",
        "tags": ["dataset", "validation"],
    },
    "nbody_expansion_configs": {
        "category": "engine",
        "scope": "nbody",
        "status": "complete",
        "description": (
            "Scenario registry for the multi-system universality survey. 21 physical "
            "three-body systems with potentials, masses, charges, external traps. "
            "De facto system-level registry for the project."
        ),
        "tags": ["engine", "scenarios", "registry"],
    },

    # --- Test / regression ---------------------------------------------------
    "test_regression": {
        "category": "test",
        "scope": "3body",
        "status": "complete",
        "description": (
            "Regression suite. Checks SymPy >= 1.13.3, NBodyAlgebra (N=3, 1/r) levels 0-2 "
            "give [3, 6, 17], and ThreeBodyAlgebra d=2 levels 0-1 give [3, 6]."
        ),
        "run_command": "python test_regression.py",
        "related_docs": ["README.md"],
        "tags": ["test", "regression", "ci"],
    },

    # --- Neural network ------------------------------------------------------
    "neural_nn_poisson": {
        "category": "engine",
        "scope": "neural",
        "status": "complete",
        "description": "Neural-network Poisson algebra: 3-layer linear NN with gradient-product coupling. [3, 6, 17, 119].",
        "latest_result": {
            "date": "2026-04-17",
            "summary": "Neural algebra: [3, 6, 17, 119] - 3 extra generators vs gravity.",
            "sequence": [3, 6, 17, 119],
        },
        "related_supplemental": ["poisson_neural_network_research_brief.md"],
        "tags": ["engine", "neural", "gradient-product"],
    },
    "neural_nn_extra_generators": {
        "category": "analysis",
        "scope": "neural",
        "status": "complete",
        "description": "Identification and characterization of the 3 extra neural generators (S3 standard rep, all involve H_23).",
        "tags": ["analysis", "neural", "117th-style"],
    },

    # --- Diagnostics: legacy SVD diagnostics, mostly archived ----------------
    "diagnostic_aws": {"category": "diagnostic", "status": "archived", "description": "Legacy AWS SVD diagnostic; superseded by symbolic rank pipeline.", "tags": ["diagnostic", "legacy"]},
    "diagnostic_compare": {"category": "diagnostic", "status": "archived", "description": "Legacy compare diagnostic; superseded.", "tags": ["diagnostic", "legacy"]},
    "diagnostic_fast_svd": {"category": "diagnostic", "status": "archived", "description": "Legacy fast-SVD diagnostic; superseded.", "tags": ["diagnostic", "legacy"]},
    "diagnostic_level3": {"category": "diagnostic", "status": "archived", "description": "Legacy level-3 diagnostic; superseded.", "tags": ["diagnostic", "legacy"]},

    # --- Known broken --------------------------------------------------------
    "parametric_atlas_scan": {
        "category": "atlas",
        "scope": "nbody/N3",
        "status": "broken",
        "description": (
            "Parametric 1/r^n exponent atlas. Aborted on cost (~$1500 projected). "
            "Needs redesign per gap_workplan.md 3.1: coarse L2 first, then targeted L3."
        ),
        "related_docs": ["docs/gap_workplan.md"],
        "related_supplemental": ["I now have everything I need. Let m.txt"],
        "tags": ["atlas", "parametric", "blocked"],
    },
    "level4_mpmath_rank": {
        "category": "dimseq",
        "scope": "nbody/N3",
        "status": "wip",
        "description": (
            "Level-4 mpmath high-precision rank computation. Spot-reclaimed at 4.4% "
            "(667/15000 rows). Checkpoint on S3; needs relaunch."
        ),
        "related_docs": ["docs/gap_workplan.md"],
        "tags": ["dimseq", "level4", "mpmath", "blocked"],
    },

    # --- Registry tooling itself --------------------------------------------
    "scripts_registry_bootstrap": {
        "category": "infra",
        "scope": "tooling",
        "status": "complete",
        "description": "Bootstrap registry/experiments.yaml from discovered scripts. Idempotent; only adds new entries.",
        "run_command": "python scripts/registry_bootstrap.py --write",
        "related_docs": ["docs/registry_status.md"],
        "tags": ["registry", "tooling"],
    },
    "scripts_registry_curate": {
        "category": "infra",
        "scope": "tooling",
        "status": "complete",
        "description": "Apply canonical-state patches to selected registry entries. Edit PATCHES dict and re-run.",
        "run_command": "python scripts/registry_curate.py --write",
        "tags": ["registry", "tooling"],
    },
    "scripts_registry_new": {
        "category": "infra",
        "scope": "tooling",
        "status": "complete",
        "description": "CLI to scaffold a new registry entry for a fresh script.",
        "run_command": "python scripts/registry_new.py <path> --category <c>",
        "tags": ["registry", "tooling"],
    },
    "scripts_archive_legacy_figures": {
        "category": "infra",
        "scope": "tooling",
        "status": "complete",
        "description": "Moves legacy PNGs into legacy_figures_archive/.",
        "tags": ["tooling", "figures"],
    },

    # ======================================================================
    # 2026-05-11 curation pass: analysis (52) + atlas (13) categories
    # ======================================================================
    # ------ analysis: 3D engine drivers (complete) ------------------------
    "3d_run_1d": {
        "category": "analysis", "scope": "3body/d=1", "status": "complete",
        "description": "Driver: 1D (linear) Poisson algebra computation. Established d-independence for N=3 at d=1.",
        "run_command": "python 3d/run_1d.py",
        "tags": ["analysis", "d=1", "engine-driver"],
    },
    "3d_run_3d": {
        "category": "analysis", "scope": "3body/d=3", "status": "complete",
        "description": "Driver: 3D (spatial) Poisson algebra computation. Established d-independence for N=3 at d=3.",
        "run_command": "python 3d/run_3d.py",
        "tags": ["analysis", "d=3", "engine-driver"],
    },
    # ------ analysis: legacy numerical engine, superseded -----------------
    "algebra_growth": {
        "category": "analysis", "scope": "3body", "status": "superseded",
        "superseded_by": "exact_growth",
        "description": "Original Phase 1 numerical-finite-difference engine. Produced unstable level-3 estimates (39-103). Superseded by exact_growth.py polynomial-u trick.",
        "tags": ["analysis", "legacy", "phase1"],
    },
    "analyze_results": {
        "category": "analysis", "scope": "3body", "status": "complete",
        "description": "Post-hoc analysis of early growth runs. One-shot script that consumed algebra_growth output to characterize sensitivity.",
        "tags": ["analysis", "phase1"],
    },
    # ------ analysis: Level-4 pipeline (complete) -------------------------
    "aws_level4": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Level-4 dimension multi-config AWS pipeline. Produced d(4)>=5604 lower bound across multiple configurations.",
        "outputs": ["results/level4_global_*/", "results/level4_lagrange_*/", "results/level4_euler_*/", "results/level4_scalene_*/"],
        "tags": ["analysis", "level4", "aws"],
    },
    "level4_comparison": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Level-4 comparison chart (gap_workplan Phase 1 Item 1.5). Bar chart + convergence curves across 4 config types.",
        "outputs": ["level4_comparison_chart.png", "level4_convergence_curves.png"],
        "related_docs": ["docs/gap_workplan.md"],
        "tags": ["analysis", "level4", "phase1"],
    },
    "level4_highsample": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Level-4 high-sample bound tightening. Pushed d(4) bounds with up to 200K samples at generic configs.",
        "tags": ["analysis", "level4"],
    },
    # ------ analysis: sanity / utility (complete) -------------------------
    "check_1r2": {
        "category": "analysis", "scope": "3body", "status": "complete",
        "description": "1/r^2 sanity check script: validates Calogero-Moser variant produces [3, 6, 17, 116].",
        "tags": ["analysis", "sanity", "calogero-moser"],
    },
    "clebsch_gordan_analysis": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Canonical S_3 representation decomposition of the N=3 Poisson algebra. Establishes n_E = 52 = Tier 1 size exactly. Referenced from paper 2 (S_3 jet filtration).",
        "outputs": ["potential_comparison_plots/quantization_analysis.md"],
        "related_docs": ["papers/paper2_s3_filtration.tex", "docs/s4_tier_predictions.md"],
        "tags": ["analysis", "representation-theory", "s3", "tier-decomposition"],
    },
    "cm_comparison": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Calogero-Moser vs Newtonian gravity bracket-algebra comparison. Established universality across singular potential types.",
        "tags": ["analysis", "calogero-moser", "universality"],
    },
    "data_inventory": {
        "category": "analysis", "scope": "3body", "status": "complete",
        "description": "Complete data inventory utility across all results/ subdirectories.",
        "outputs": ["data/data_inventory.json"],
        "tags": ["analysis", "utility", "inventory"],
    },
    # ------ analysis: Dirac-constraint exploration (complete) -------------
    "dirac_analysis_from_svd": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Dirac constraint analysis using pre-computed SVD data. Comprehensive noise-floor taxonomy: 32 syzygies + 8 true zeros + significant generators. Settled the question of whether 40 vanishing SVs are constraints (answer: 8 constraints + 32 syzygies).",
        "tags": ["analysis", "dirac", "syzygy", "noise-floor"],
    },
    "dirac_constraint_test": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Tests whether the 40 noise-floor generators are Dirac constraints. Result: 32 are syzygies (Jacobi-identity consequences), 8 are true constraints (translation invariance).",
        "tags": ["analysis", "dirac", "syzygy"],
    },
    "dirac_direct_eval": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Direct generator evaluation to identify which level-3 generators vanish everywhere on phase space (true zeros) vs only at generic points (syzygies).",
        "tags": ["analysis", "dirac", "syzygy"],
    },
    # ------ analysis: viz / figure generation -----------------------------
    "generate_triptychs": {
        "category": "analysis", "scope": "3body", "status": "superseded",
        "superseded_by": "website_figures_compare",
        "description": "Original triptych atlas series renderer. Superseded by the website/figures_compare.py pipeline (Apr 18, 2026 rebuild).",
        "tags": ["analysis", "viz", "legacy"],
    },
    "rerender_frames": {
        "category": "analysis", "scope": "3body", "status": "complete",
        "description": "Atlas animation re-render utility with adaptive colorscale. One-shot fix for early animation issues.",
        "tags": ["analysis", "viz", "utility"],
    },
    # ------ analysis: hires Lagrange (complete tool) ----------------------
    "hires_lagrange_scan": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "High-resolution Lagrange-region scan tool. Used to resolve the concentric ring features around the equilateral point.",
        "tags": ["analysis", "lagrange", "hires"],
    },
    # ------ analysis: explicitly archived ---------------------------------
    "mass_ratio_bisection": {
        "category": "analysis", "scope": "3body", "status": "archived",
        "description": "HISTORICAL ARTIFACT. The docstring of the script itself flags it as DO NOT USE FOR NEW ANALYSIS. Predates the symbolic rank proof.",
        "tags": ["analysis", "legacy", "do-not-use"],
    },
    # ------ analysis: 117th generator chain (complete) --------------------
    "nbody_analyze_117th_compact": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Compact 10-term relative-coordinate form of the 117th quantum generator.",
        "tags": ["analysis", "117th", "quantum"],
    },
    "nbody_analyze_117th_definite": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Sign-definiteness proof for the 117th generator: g = -(9/4)*[(A-B)^2 + A^2] is negative semi-definite.",
        "tags": ["analysis", "117th", "quantum"],
    },
    "nbody_analyze_117th_physics": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Physical interpretation of the 117th generator: NOT a conserved quantity ([G, H_total] != 0).",
        "tags": ["analysis", "117th", "quantum"],
    },
    "nbody_identify_117th": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Identifies the 117th generator: indices into the basis, Legendre P_3 octupole structure, S_3 average.",
        "tags": ["analysis", "117th", "quantum"],
    },
    "nbody_identify_117th_rank": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Rank of the 48 hbar^2 corrections among themselves (probes quantum L3 algebra structure).",
        "tags": ["analysis", "117th", "quantum"],
    },
    # ------ analysis: N-body scaling --------------------------------------
    "nbody_analyze_scaling": {
        "category": "analysis", "scope": "nbody", "status": "complete",
        "description": "N-body dimension-sequence scaling analysis. Derived L2(N) = N(4N^2-9N+3)/2 for N>=4 by fitting new_L2 = 12*C(N,3).",
        "outputs": ["results/analysis/nbody_scaling_formulas.json"],
        "tags": ["analysis", "scaling", "n-body"],
    },
    # ------ analysis: multi-system survey analysis ------------------------
    "nbody_expansion_analysis": {
        "category": "analysis", "scope": "nbody", "status": "complete",
        "description": "Multi-system universality survey: analysis + plotting across 21 physical systems.",
        "outputs": ["results/expansion_dimseq/"],
        "tags": ["analysis", "universality-survey", "multi-system"],
    },
    # ------ analysis: L3 structure constant comparison chain (complete) ---
    "nbody_fast_level3_compare": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Fast numerical L3 structure constant comparison via finite differences. One of four L3 comparison methods.",
        "tags": ["analysis", "L3", "structure-constants"],
    },
    "nbody_numerical_level3_compare": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Numerical SVD-based L3 structure constant comparison between potentials.",
        "tags": ["analysis", "L3", "structure-constants"],
    },
    "nbody_symbolic_level3_compare": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Exact symbolic L3 structure constant comparison over QQ (~37h/potential on AWS).",
        "tags": ["analysis", "L3", "structure-constants", "exact"],
    },
    # ------ analysis: helium / atomic -------------------------------------
    "nbody_run_helium": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Helium atom Poisson algebra. Confirms sign of interaction doesn't matter: helium (+2, -1, -1) gives [3, 6, 17, 116].",
        "latest_result": {"date": "2026-03-19", "summary": "Helium charge-sign invariance confirmed.",
                          "sequence": [3, 6, 17, 116]},
        "tags": ["analysis", "helium", "atomic", "charge-sign"],
    },
    # ------ analysis: N=4 priority runs, superseded by general engine ------
    "nbody_run_n4_d1": {
        "category": "analysis", "scope": "nbody/N4", "status": "superseded",
        "superseded_by": "nbody_symbolic_rank_nbody",
        "description": "Priority 4a: N=4, d=1 through L=2. Now subsumed by the general nbody/symbolic_rank_nbody.py engine.",
        "tags": ["analysis", "n4", "legacy"],
    },
    "nbody_run_n4_d2": {
        "category": "analysis", "scope": "nbody/N4", "status": "superseded",
        "superseded_by": "nbody_symbolic_rank_nbody",
        "description": "Priority 1: N=4, d=2, 1/r through L=2. Subsumed by general engine.",
        "tags": ["analysis", "n4", "legacy"],
    },
    "nbody_run_n4_d3": {
        "category": "analysis", "scope": "nbody/N4", "status": "superseded",
        "superseded_by": "nbody_symbolic_rank_nbody",
        "description": "Priority 4b: N=4, d=3 through L=2. Subsumed by general engine.",
        "tags": ["analysis", "n4", "legacy"],
    },
    "nbody_run_n4_mass": {
        "category": "analysis", "scope": "nbody/N4", "status": "complete",
        "description": "N=4 mass invariance test. Confirmed: 3 configs (equal, hierarchical, mixed) all give [6, 14, 62].",
        "latest_result": {"date": "2026-03-15",
                          "summary": "N=4 mass invariance confirmed across 3 configurations.",
                          "sequence": [6, 14, 62]},
        "tags": ["analysis", "n4", "mass-invariance"],
    },
    "nbody_run_potential_1r3": {
        "category": "analysis", "scope": "nbody/N3", "status": "superseded",
        "superseded_by": "nbody_symbolic_rank_nbody",
        "description": "Priority 2: N=3, d=2, 1/r^3 through L=3. Now subsumed by the general engine with --potential 1/r3.",
        "tags": ["analysis", "1/r3", "legacy"],
    },
    # ------ analysis: post-Newtonian chain (complete) ---------------------
    "nbody_run_pn_aws": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "AWS orchestrator for composite-potential / post-Newtonian dimension-sequence runs.",
        "tags": ["analysis", "post-newtonian", "aws", "composite"],
    },
    "nbody_run_pn_mass_test": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Post-Newtonian mass invariance test.",
        "tags": ["analysis", "post-newtonian", "mass-invariance"],
    },
    "nbody_run_post_newtonian": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Post-Newtonian (1PN) three-body Poisson algebra. Confirms PN correction does not change the algebra: [3, 6, 17, 116].",
        "latest_result": {"date": "2026-03-22",
                          "summary": "1PN preserves [3, 6, 17, 116].",
                          "sequence": [3, 6, 17, 116]},
        "tags": ["analysis", "post-newtonian"],
    },
    # ------ analysis: symbolic n proof chain ------------------------------
    "nbody_symbolic_n_level3": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "L3 verification at 10 concrete rational p-values for V = -u^p potentials.",
        "tags": ["analysis", "symbolic", "exponent-sweep"],
    },
    "nbody_symbolic_n_proof": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Numerical survey of dimension vs potential exponent p. Established [3, 6, 17] is universal at L2 across 79+ rational p-values; the p=-2 harmonic exception is infinitely sharp.",
        "tags": ["analysis", "exponent-survey", "universality"],
    },
    # ------ analysis: neural chain ----------------------------------------
    "neural_identify_extra_generators": {
        "category": "analysis", "scope": "neural", "status": "complete",
        "description": "Identifies the 3 extra generators in the gradient-product neural algebra not present in the physical N=3 universal set.",
        "tags": ["analysis", "neural"],
    },
    "neural_physics_vs_neural": {
        "category": "analysis", "scope": "neural", "status": "complete",
        "description": "Direct comparison: physical N=3 algebra vs neural gradient-product algebra in the same 6D phase space.",
        "tags": ["analysis", "neural"],
    },
    "neural_summarize_results": {
        "category": "analysis", "scope": "neural", "status": "complete",
        "description": "Summary table of all saved neural-algebra results.",
        "tags": ["analysis", "neural", "summary"],
    },
    # ------ analysis: potential comparison + CM exact ---------------------
    "potential_comparison": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Potential comparison study: 1/r vs 1/r^2 vs r^2. Foundational comparison establishing that singular and integrable potentials produce identical dimension sequences (1/r and 1/r^2 both [3,6,17,116]) while harmonic closes finite at dim 15.",
        "run_command": "python potential_comparison.py",
        "tags": ["analysis", "universality", "harmonic"],
    },
    "run_cm_exact": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Exact Calogero-Moser run by patching exact_growth.py Hamiltonians. Reuses the full infrastructure.",
        "tags": ["analysis", "calogero-moser", "exact"],
    },
    # ------ analysis: primes chain ----------------------------------------
    "primes_check_1r_closure": {
        "category": "analysis", "scope": "primes", "status": "complete",
        "description": "Closure test for the 1/r potential algebra. Result: NOT closed at 116 (infinite-dimensional).",
        "tags": ["analysis", "primes", "closure"],
    },
    "primes_closure_check": {
        "category": "analysis", "scope": "primes", "status": "complete",
        "description": "L=4 dimension computation for 1/r and log potentials. Confirms infinite growth.",
        "tags": ["analysis", "primes", "closure", "L4"],
    },
    "primes_hilbert_polya_search": {
        "category": "analysis", "scope": "primes", "status": "superseded",
        "superseded_by": "primes_check_1r_closure",
        "description": "Hilbert-Polya operator search via GUE Lie algebra structure. Superseded by the infinite-dimensionality discovery: the 116 generators do not form a closed Lie subalgebra (Jacobi fails for the SC tensor), so the HP route via this construction does not work.",
        "tags": ["analysis", "primes", "hilbert-polya", "superseded"],
    },
    "primes_launch_gue": {
        "category": "analysis", "scope": "primes", "status": "complete",
        "description": "AWS EC2 spot launcher for the GUE log-gas Poisson algebra computation. Ran the 4-config GUE study.",
        "tags": ["analysis", "primes", "gue", "aws-launcher"],
    },
    "primes_multi_potential_r_comparison": {
        "category": "analysis", "scope": "primes", "status": "complete",
        "description": "Multi-potential <r> (coadjoint orbit spacing ratio) comparison across all singular potentials. Confirms <r> ~ 0.64 universal.",
        "tags": ["analysis", "primes", "BGS", "spectral"],
    },
    # ------ analysis: quantization analysis -------------------------------
    "quantization_analysis": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Tier-structure statistics + LQG/Bekenstein-style integer-scaling hypothesis tests. Foundational for Paper 2 (S_3 jet filtration).",
        "outputs": ["potential_comparison_plots/quantization_analysis.md"],
        "related_docs": ["papers/paper2_s3_filtration.tex"],
        "tags": ["analysis", "tier-decomposition", "lqg", "phase1"],
    },
    # ------ analysis: SV116 + unequal mass -------------------------------
    "sv116_analytical": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "SV #116 analytical prediction (gap_workplan Phase 1 Item 1.6). Achieved R^2 = 0.630 correlation between analytical prediction and observed SV #116.",
        "outputs": ["spectral_depth/sv116_predicted_vs_observed.png"],
        "related_docs": ["docs/gap_workplan.md"],
        "tags": ["analysis", "sv116", "phase1"],
    },
    "unequal_mass_study": {
        "category": "analysis", "scope": "nbody/N3", "status": "complete",
        "description": "Unequal-mass dimension-sequence study. The original [3,5,13,69] vs [3,6,17,116] artifact study; superseded by the symbolic-rank-over-Q proof of mass invariance for all positive masses.",
        "run_command": "python unequal_mass_study.py",
        "tags": ["analysis", "mass-invariance", "historical"],
    },

    # ====================== atlas (13) ====================================
    "analyze_atlas_data": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "Diagnostic for atlas_1000 data: clean-merge + excluded-block check.",
        "tags": ["atlas", "diagnostic", "utility"],
    },
    "assemble_atlases": {
        "category": "atlas", "scope": "nbody/N3", "status": "superseded",
        "superseded_by": "website_figures_render",
        "description": "Original atlas assembly + comprehensive shape-sphere visualization. Superseded by the website/figures_render.py + figures_compare.py pipeline (Apr 18, 2026 rebuild).",
        "tags": ["atlas", "viz", "legacy"],
    },
    "atlas_1000": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "1000x1000 parallelized high-resolution shape-sphere scan. Produced the canonical hi-res atlases.",
        "outputs": ["aws_results/atlas_1000/"],
        "tags": ["atlas", "hires", "1000x1000"],
    },
    "atlas_diagnostics": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "Atlas diagnostic checks. Validates the atlas pipeline output.",
        "tags": ["atlas", "diagnostic"],
    },
    "audit_atlas_data": {
        "category": "atlas", "scope": "3body", "status": "complete",
        "description": "Audit local atlas data against S3 to detect stale/incomplete syncs (fixes the s3 sync size-only artifact for fixed-size numpy arrays).",
        "tags": ["atlas", "diagnostic", "s3"],
    },
    "cg_atlas_comparison": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "Clebsch-Gordan atlas comparison (gap_workplan Phase 1 Item 1.4). Lagrange=33 doublets (E-frac=0.57), Euler=14, Iso-90=24.",
        "outputs": ["spectral_depth/cg_predicted_vs_observed.png",
                    "spectral_depth/cg_residual_map.png"],
        "related_docs": ["docs/gap_workplan.md"],
        "tags": ["atlas", "cg", "phase1"],
    },
    "full_atlas_scan": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "Full shape-sphere atlas scanner with SVD gap analysis and mass-aware momentum sampling. Core atlas tool.",
        "run_command": "python full_atlas_scan.py --resolution 100 --samples 500",
        "tags": ["atlas", "core-tool", "scanner"],
    },
    "merge_atlas_1000": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "Selective merge of 1000x1000 atlas data across multiple AWS spot runs.",
        "tags": ["atlas", "utility"],
    },
    "nbody_helium_atlas": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "Helium atlas: charge-sign invariance comparison tool. Used for the (+2, -1, -1) atlas vs gravitational baseline.",
        "tags": ["atlas", "helium", "charge-sign"],
    },
    "run_expansion_atlas": {
        "category": "atlas", "scope": "nbody", "status": "complete",
        "description": "AWS orchestrator for the Multi-System Universality Survey stability atlases. Produced the 18 completed atlas configurations.",
        "tags": ["atlas", "universality-survey", "aws"],
    },
    "shape_sphere": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "Shape sphere gap-ratio landscape renderer. The original shape-sphere visualization (canonical for paper 1).",
        "tags": ["atlas", "shape-sphere", "viz"],
    },
    "shape_sphere_hires": {
        "category": "atlas", "scope": "nbody/N3", "status": "superseded",
        "superseded_by": "shape_sphere_atlas",
        "description": "High-resolution shape-sphere atlas; superseded by shape_sphere_atlas.py (canonical Hsiang-Montgomery Jacobi projection, see recent commits d33a55e and f7e8b29).",
        "tags": ["atlas", "shape-sphere", "legacy"],
    },
    "sv_landscape_viz": {
        "category": "atlas", "scope": "nbody/N3", "status": "complete",
        "description": "Singular-value landscape visualizations across the shape sphere. Used for the spectral depth mining figures.",
        "tags": ["atlas", "viz", "sv-landscape"],
    },
}


def main() -> int:
    p = argparse.ArgumentParser(description="Apply canonical patches to registry")
    p.add_argument("--write", action="store_true", help="Write changes to registry/experiments.yaml")
    args = p.parse_args()

    entries = load()
    by_id = {e["id"]: e for e in entries}

    missing: list[str] = []
    applied: list[str] = []

    for eid, patch in PATCHES.items():
        target = by_id.get(eid)
        if target is None:
            missing.append(eid)
            continue
        for key, value in patch.items():
            target[key] = value
        applied.append(eid)

    print(f"Applied {len(applied)} patches.")
    if missing:
        print(f"WARNING: {len(missing)} patch ids not found in registry:")
        for m in missing:
            print(f"  - {m}")

    if not args.write:
        print("Dry run; pass --write to commit.")
        return 0

    REGISTRY_PATH.write_text(
        _file_header() + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} entries.")
    return 0


def _file_header() -> str:
    return (
        "# Experiment registry - single source of truth.\n"
        "#\n"
        "# Auto-bootstrapped entries have status: needs_review. Curate them to\n"
        "# canonical state by editing this file directly. New scripts should be\n"
        "# added with `python scripts/registry_new.py <path> --category <c>`.\n"
        "#\n"
        "# Status values: planned | wip | complete | broken | superseded | archived | needs_review\n"
        "# Categories: dimseq | atlas | structure | quantum | spectral | infra |\n"
        "#             analysis | viz | dataset | test | diagnostic | engine |\n"
        "#             website | docs\n"
        "\n"
    )


if __name__ == "__main__":
    sys.exit(main())
