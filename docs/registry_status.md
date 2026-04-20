# Experiment Registry

*Auto-generated from `registry/experiments.yaml` — 164 entries.*

**Status:** wip: 3 · complete: 41 · planned: 8 · broken: 1 · needs_review: 107 · archived: 4

## analysis

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `3d_run_1d` | needs_review | `3d/run_1d.py` | Run the 1D (linear) Poisson algebra computation. |
| `3d_run_3d` | needs_review | `3d/run_3d.py` | Run the 3D (spatial) Poisson algebra computation. |
| `algebra_growth` | needs_review | `algebra_growth.py` | Three-Body Lie Algebra Growth Characterization ================================================ |
| `analyze_results` | needs_review | `analyze_results.py` | Post-hoc Analysis of Lie Algebra Growth Results ================================================= |
| `aws_level4` | needs_review | `aws_level4.py` | Level 4 Lie Algebra Dimension -- Multi-Configuration Pipeline =======================================================... |
| `check_1r2` | needs_review | `check_1r2.py` | (no docstring) check_1r2.py |
| `clebsch_gordan_analysis` | needs_review | `clebsch_gordan_analysis.py` | Decompose the Poisson algebra under S_3 x SO(2) and check whether the tier structure 52 + 44 + 16 + 4 = 116 is predic... |
| `cm_comparison` | needs_review | `cm_comparison.py` | Calogero-Moser vs Gravitational: Bracket Algebra Comparison =========================================================... |
| `data_inventory` | needs_review | `data_inventory.py` | Complete data inventory across all datasets. |
| `dirac_analysis_from_svd` | needs_review | `dirac_analysis_from_svd.py` | Dirac constraint analysis using ONLY pre-computed SVD data. |
| `dirac_constraint_test` | needs_review | `dirac_constraint_test.py` | Test whether the 40 noise-floor generators are Dirac constraints. |
| `dirac_direct_eval` | needs_review | `dirac_direct_eval.py` | Direct evaluation of all generators to identify which are constraints. |
| `generate_triptychs` | needs_review | `generate_triptychs.py` | Triptych Atlas Series ====================== |
| `hires_lagrange_scan` | needs_review | `hires_lagrange_scan.py` | High-Resolution Lagrange Region Scan ====================================== Focused 1000x1000 grid scan centered on t... |
| `level4_comparison` | needs_review | `level4_comparison.py` | Level-4 Comparison — Phase 1 Item 1.5 ====================================== |
| `level4_highsample` | needs_review | `level4_highsample.py` | Level 4 High-Sample Bound Tightening ===================================== |
| `mass_ratio_bisection` | needs_review | `mass_ratio_bisection.py` | HISTORICAL ARTIFACT — DO NOT USE FOR NEW ANALYSIS =================================================== This script and... |
| `nbody_analyze_117th` | complete | `nbody/analyze_117th.py` | Analysis of the 117th quantum generator: sum-of-squares, negative semi-definite, P3 octupole. |
| `nbody_analyze_117th_compact` | needs_review | `nbody/analyze_117th_compact.py` | Extract the compact 10-term relative-coordinate form and analyze it. |
| `nbody_analyze_117th_definite` | needs_review | `nbody/analyze_117th_definite.py` | Final analysis: clean mathematical form and sign-definiteness proof. Key finding: g = (u12*u13)^p * F(w12, w13) with ... |
| `nbody_analyze_117th_physics` | needs_review | `nbody/analyze_117th_physics.py` | Physical interpretation: does g commute with the Hamiltonian? Is it a conserved quantity? |
| `nbody_analyze_scaling` | needs_review | `nbody/analyze_scaling.py` | Analyze N-body dimension sequence scaling. |
| `nbody_expansion_analysis` | needs_review | `nbody/expansion_analysis.py` | Multi-System Universality Survey -- Analysis and Plotting |
| `nbody_fast_level3_compare` | needs_review | `nbody/fast_level3_compare.py` | Fast numerical comparison of level-3 structure constants. |
| `nbody_identify_117th` | needs_review | `nbody/identify_117th.py` | Identify the 117th Generator ============================= |
| `nbody_identify_117th_rank` | needs_review | `nbody/identify_117th_rank.py` | Determine the rank of the 48 hbar^2 corrections among themselves. |
| `nbody_numerical_level3_compare` | needs_review | `nbody/numerical_level3_compare.py` | Numerical comparison of level-3 structure constants between potentials. |
| `nbody_run_helium` | needs_review | `nbody/run_helium.py` | Helium atom Poisson algebra: does the sign of the interaction matter? |
| `nbody_run_n4_d1` | needs_review | `nbody/run_n4_d1.py` | Priority 4a: N=4, d=1 through Level 2. |
| `nbody_run_n4_d2` | needs_review | `nbody/run_n4_d2.py` | Priority 1: N=4, d=2, 1/r through Level 2. |
| `nbody_run_n4_d3` | needs_review | `nbody/run_n4_d3.py` | Priority 4b: N=4, d=3 through Level 2. |
| `nbody_run_n4_mass` | needs_review | `nbody/run_n4_mass.py` | Priority 3: N=4 mass invariance test. |
| `nbody_run_pn_aws` | needs_review | `nbody/run_pn_aws.py` | AWS orchestrator for composite-potential and post-Newtonian runs. |
| `nbody_run_pn_mass_test` | needs_review | `nbody/run_pn_mass_test.py` | Post-Newtonian mass invariance test. |
| `nbody_run_post_newtonian` | needs_review | `nbody/run_post_newtonian.py` | Post-Newtonian (1PN) three-body Poisson algebra. |
| `nbody_run_potential_1r3` | needs_review | `nbody/run_potential_1r3.py` | Priority 2: N=3, d=2, 1/r^3 potential through Level 3. |
| `nbody_symbolic_level3_compare` | needs_review | `nbody/symbolic_level3_compare.py` | Symbolic level-3 structure constant comparison across potentials. |
| `nbody_symbolic_n_level3` | needs_review | `nbody/symbolic_n_level3.py` | Phase 4: Level-3 verification at 10 concrete rational p-values. |
| `nbody_symbolic_n_proof` | needs_review | `nbody/symbolic_n_proof.py` | Numerical survey of Poisson algebra dimension vs potential exponent. |
| `neural_identify_extra_generators` | needs_review | `neural/identify_extra_generators.py` | Identify the 3 extra generators in the gradient-product neural algebra that are NOT present in the physical N=3 unive... |
| `neural_nn_extra_generators` | complete | `neural/nn_extra_generators.py` | Identification and characterization of the 3 extra neural generators (S3 standard rep, all involve H_23). |
| `neural_physics_vs_neural` | needs_review | `neural/physics_vs_neural.py` | Direct comparison: physical N=3 algebra vs neural gradient-product algebra in the SAME 6D phase space. |
| `neural_summarize_results` | needs_review | `neural/summarize_results.py` | Print a table of all saved neural algebra results. |
| `potential_comparison` | needs_review | `potential_comparison.py` | Potential Comparison Study: 1/r vs 1/r² vs r² ================================================ |
| `primes_check_1r_closure` | needs_review | `primes/check_1r_closure.py` | Quick check: does the 1/r potential algebra close at 116? |
| `primes_closure_check` | needs_review | `primes/closure_check.py` | Compute level-4 dimension of the Poisson algebra for 1/r and log potentials. |
| `primes_hilbert_polya_search` | needs_review | `primes/hilbert_polya_search.py` | Hilbert-Pólya operator search via GUE Lie algebra structure. |
| `primes_launch_gue` | needs_review | `primes/launch_gue.py` | Launch EC2 instance for GUE log-gas Poisson algebra computation. |
| `primes_multi_potential_r_comparison` | needs_review | `primes/multi_potential_r_comparison.py` | Multi-potential comparison of coadjoint orbit spacing ratio <r>. |
| `quantization_analysis` | needs_review | `quantization_analysis.py` | Definitive LQG/Bekenstein Quantization Test |
| `rerender_frames` | needs_review | `rerender_frames.py` | Re-render atlas animation frames with adaptive colorscale. |
| `run_cm_exact` | needs_review | `run_cm_exact.py` | Run exact CM computation by patching exact_growth.py's Hamiltonians. Reuses the full infrastructure including _make_f... |
| `s4_tier_analysis` | needs_review | `s4_tier_analysis.py` | S_4 Representation Decomposition for the N=4 Poisson Algebra. |
| `sv116_analytical` | needs_review | `sv116_analytical.py` | SV #116 Analytical Prediction — Phase 1 Item 1.6 ================================================== |
| `unequal_mass_study` | needs_review | `unequal_mass_study.py` | Unequal Mass Study — Dimension Sequence vs Mass Configuration =======================================================... |

## atlas

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `analyze_atlas_data` | needs_review | `analyze_atlas_data.py` | Quick diagnostic of atlas_1000 data: clean merge + excluded block. |
| `assemble_atlases` | needs_review | `assemble_atlases.py` | Atlas Assembly: Comprehensive Shape-Sphere Atlas Visualization ======================================================... |
| `atlas_1000` | needs_review | `atlas_1000.py` | 1000x1000 Atlas: Parallelized High-Resolution Shape Sphere Scan |
| `atlas_diagnostics` | needs_review | `atlas_diagnostics.py` | Atlas Diagnostic Checks ======================== |
| `audit_atlas_data` | needs_review | `audit_atlas_data.py` | Audit all local atlas data against S3 to detect stale/incomplete syncs. |
| `cg_atlas_comparison` | needs_review | `cg_atlas_comparison.py` | CG Atlas Comparison — Phase 1 Item 1.4 ======================================= |
| `full_atlas_scan` | needs_review | `full_atlas_scan.py` | Full shape-sphere atlas scanner with corrected SVD gap analysis and mass-aware momentum sampling. |
| `merge_atlas_1000` | needs_review | `merge_atlas_1000.py` | Selective merge of 1000x1000 atlas data. |
| `multi_epsilon_atlas` | complete | `multi_epsilon_atlas.py` | Multi-epsilon adaptive structure atlas. Supports --charges, multiprocessing, spot instances. |
| `n4_atlas_1d` | complete | `n4_atlas_1d.py` | First atlas data for N=4: three 1D slices. All 300 points show rank 62 (no drops). |
| `nbody_helium_atlas` | needs_review | `nbody/helium_atlas.py` | Helium Atlas: Charge-Sign Invariance Comparison Tool ===================================================== |
| `parametric_atlas_scan` | broken | `parametric_atlas_scan.py` | Parametric 1/r^n exponent atlas. Aborted on cost (~$1500 projected). Needs redesign per gap_workplan.md 3.1: coarse L... |
| `run_expansion_atlas` | needs_review | `run_expansion_atlas.py` | AWS orchestrator for the Multi-System Universality Survey -- stability atlases. |
| `shape_sphere` | needs_review | `shape_sphere.py` | Shape Sphere Gap Ratio Landscape ================================= |
| `shape_sphere_hires` | needs_review | `shape_sphere_hires.py` | High-Resolution Shape Sphere Atlas ==================================== |
| `stability_atlas` | complete | `stability_atlas.py` | Exact-engine atlas scanner. Drives most of the (mu, phi) shape-sphere atlases. |
| `sv_landscape_viz` | needs_review | `sv_landscape_viz.py` | Singular Value Landscape Visualizations ======================================== |
| `targeted_adaptive_scan` | complete | `targeted_adaptive_scan.py` | High-resolution adaptive scans of named regions (Lagrange, Euler, etc.) for any potential. |

## dataset

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `dataset_build_dataset` | complete | `dataset/build_dataset.py` | Build the 13-table HF dataset from ~200 result JSONs. Outputs Parquet; uploaded to huggingface.co/datasets/bshepp/pai... |

## diagnostic

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `bench_flint` | complete | `bench_flint/bench.py` | python-flint vs default ground-types benchmark on the actual project workload. Three benches: (1) micro cancel() on a... |
| `bench_simplify_phases` | complete | `bench_flint/run_simplify_phases.py` | Three-phase smarter-simplify experiment. Tests whether `together` can replace `cancel` in NBodyAlgebra.simplify_gener... |
| `bench_validate_simplify_patch` | complete | `bench_flint/validate_simplify_patch.py` | Paranoid 8-stage validator for the simplify_generator patch. Runs each case as a separate Python subprocess under ben... |
| `diagnostic_aws` | archived | `diagnostic_aws.py` | Legacy AWS SVD diagnostic; superseded by symbolic rank pipeline. |
| `diagnostic_compare` | archived | `diagnostic_compare.py` | Legacy compare diagnostic; superseded. |
| `diagnostic_fast_svd` | archived | `diagnostic_fast_svd.py` | Legacy fast-SVD diagnostic; superseded. |
| `diagnostic_level3` | archived | `diagnostic_level3.py` | Legacy level-3 diagnostic; superseded. |
| `primes_diagnose_brackets` | needs_review | `primes/diagnose_brackets.py` | Diagnose bracket computation: compare symbolic vs numerical brackets. |

## dimseq

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `convergence_trajectory_sweep` | needs_review | `convergence_trajectory_sweep.py` | Convergence Trajectory Sweep ============================= |
| `l3_exponent_sweep` | complete | `l3_exponent_sweep.py` | Extended L3 exponent sweep: 76 successful data points across 1/r^n and r^n. |
| `level4_mpmath_rank` | wip | `level4_mpmath_rank.py` | Level-4 mpmath high-precision rank computation. Spot-reclaimed at 4.4% (667/15000 rows). Checkpoint on S3; needs rela... |
| `mass_ratio_sweep` | needs_review | `mass_ratio_sweep.py` | Mass Ratio Sweep — track dimension sequence and SVD gaps as mass ratio varies continuously from equal (1:1:1) to extr... |
| `nbody_charge_sensitivity_sweep` | needs_review | `nbody/charge_sensitivity_sweep.py` | Charge Sensitivity Sweep ========================= |
| `nbody_charge_sweep_d1` | complete | `nbody/charge_sweep_d1.py` | Charge magnitude sweep at d=1: integer charges (+1,+q,-1) for q=1..20 all give [3, 6, 17, 116]. Phase 3 of the charge... |
| `nbody_fractional_exponent_sweep` | needs_review | `nbody/fractional_exponent_sweep.py` | Fractional exponent sweep: test dimension sequence sensitivity to infinitesimal variations in the potential exponent ... |
| `nbody_level2_exponent_sweep` | complete | `nbody/level2_exponent_sweep.py` | 500-exponent sweep at L2 for 1/r^n and r^n. 498/500 give universal L2 dim 17; only r^1 (15) and r^2 (13). |
| `nbody_named_molecular_systems` | complete | `nbody/named_molecular_systems.py` | H3+ (m=1836) and O3 (m=29164) - both [3, 6, 17, 116]. Mass invariance for named molecules. |
| `nbody_rn_exponent_sweep` | needs_review | `nbody/rn_exponent_sweep.py` | r^n exponent sweep around n=2: test dimension sequence for polynomial-type potentials V(r) = r^n using the composite ... |
| `nbody_run_composite_test` | complete | `nbody/run_composite_test.py` | Composite potential universality: V = sum_k c_k * r^{-p_k}. Two-term and three-term composites both give [3, 6, 17, 1... |
| `nbody_run_expansion_dimseq` | needs_review | `nbody/run_expansion_dimseq.py` | AWS orchestrator for the Multi-System Universality Survey -- dimension sequences. |
| `nbody_run_n4_potential_universality` | complete | `nbody/run_n4_potential_universality.py` | N=4 potential universality: 1/r, 1/r^2, 1/r^3, log(r) all give [6, 14, 62]. |
| `nbody_run_n_scaling_aws` | complete | `nbody/run_n_scaling_aws.py` | AWS launcher for the N-body rank scaling sweep. Confirmed N=3..8 L2; N=4 L3 = [6,14,62,1260]. |
| `nbody_run_n_scaling_probe` | needs_review | `nbody/run_n_scaling_probe.py` | N-body scaling probe: how high can N go at d=1 for levels 0, 1, 2? |
| `nbody_run_schwarzschild` | wip | `nbody/run_schwarzschild.py` | Schwarzschild effective-potential dimension-sequence sweep. V_eff = -M*u + (L^2/2)*u^2 - M*L^2*u^3 expressed as a 3-t... |
| `nbody_sweep_nbody_ranks` | complete | `nbody/sweep_nbody_ranks.py` | Local N-body rank sweep harness. |
| `nbody_symbolic_gram_sweep` | needs_review | `nbody/symbolic_gram_sweep.py` | Symbolic Gram Determinant and Generator Norms vs Configuration ======================================================... |
| `nbody_symbolic_rank_nbody` | complete | `nbody/symbolic_rank_nbody.py` | Exact symbolic rank over Q (or Q(m1,m2,m3)) for arbitrary N, d, potential. Supersedes float64 SVD for definitive dime... |
| `planned_replay_schwarzschild_l3` | planned | `nbody/run_schwarzschild.py` | Re-run the Schwarzschild L3 sweep that died yesterday under the cancel engine. With the patched together engine the 4... |
| `primes_run_gue_logas` | complete | `primes/run_gue_logas.py` | Dyson log-gas computation across 4 configs (pure log, GUE composite, Penning trap, harmonic-only). All singular confi... |
| `symbolic_rank` | complete | `symbolic_rank.py` | Symbolic rank over Q(m1,m2,m3) for N=3 - the mass-invariance proof. Establishes [3, 6, 17, 116] as an algebraic theor... |
| `yukawa_dimseq` | complete | `yukawa_dimseq.py` | Yukawa potential survey via Taylor-composite K=3. Confirms [3,6,17,116] for mu in {0.1, 0.5, 0.7, 1.0, 2.0, 5.0} plus... |

## engine

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `3d_exact_growth_nd` | needs_review | `3d/exact_growth_nd.py` | N-Dimensional Poisson Algebra Growth Engine ============================================ |
| `exact_growth` | complete | `exact_growth.py` | Core symbolic Poisson bracket engine for N=3, planar. Builds pairwise Hamiltonians, iterates brackets to a configurab... |
| `exact_growth_cm` | complete | `exact_growth_cm.py` | Calogero-Moser (1/r^2) variant of the core engine. Produces [3, 6, 17, 116]. |
| `nbody_exact_growth_nbody` | complete | `nbody/exact_growth_nbody.py` | NBodyAlgebra: arbitrary N, d, potential (single or composite), masses, charges, external trap. The general-purpose en... |
| `nbody_expansion_configs` | complete | `nbody/expansion_configs.py` | Scenario registry for the multi-system universality survey. 21 physical three-body systems with potentials, masses, c... |
| `nbody_quantum_algebra` | complete | `nbody/quantum_algebra.py` | QuantumNBodyAlgebra: Moyal-bracket version of NBodyAlgebra. Validated via Jacobi and hbar->0 limit. Produces [3,6,17,... |
| `neural_nn_algebra` | needs_review | `neural/nn_algebra.py` | Generalized Neural Network Poisson Algebra Engine =================================================== |
| `neural_nn_poisson` | complete | `neural/nn_poisson.py` | Neural-network Poisson algebra: 3-layer linear NN with gradient-product coupling. [3, 6, 17, 119]. |
| `planned_simplify_patch_3d` | planned | `3d/exact_growth_nd.py` | Patch 3d/exact_growth_nd.py simplify_generator (line 171-172) and the Jacobi cancel at line 421 from cancel to togeth... |
| `planned_simplify_patch_planar` | planned | `exact_growth.py` | Patch exact_growth.py simplify_generator (line 144-149) and the inline cancel call at line 722 (Jacobi verification) ... |
| `planned_simplify_patch_symbolic_rank` | planned | `nbody/symbolic_rank_nbody.py` | Patch nbody/symbolic_rank_nbody.py _simplify (lines 318-320) from cancel(expr) to together(expr) following the same p... |
| `simplify_generator_patch` | complete | `nbody/exact_growth_nbody.py` | Patched nbody/exact_growth_nbody.py simplify_generator() from cancel(expr) to together(expr). Single-line change at l... |

## infra

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `infra_launch_1r2` | needs_review | `infra/launch_1r2.py` | Launch c6i.4xlarge for plain 1/r^2 atlas (100x100, equal masses/charges, 16 workers). |
| `infra_launch_atlas_instances` | needs_review | `infra/launch_atlas_instances.py` | Launch AWS EC2 instances for full atlas scans. |
| `infra_launch_gram_sweep` | needs_review | `infra/launch_gram_sweep.py` | Launch EC2 instance for symbolic Gram determinant sweep. |
| `infra_launch_nbody_scaling` | needs_review | `infra/launch_nbody_scaling.py` | Launch EC2 instance for N-body scaling probe. |
| `infra_launch_parametric` | needs_review | `infra/launch_parametric.py` | Launch AWS EC2 instances for parametric exponent sweep (1/r^n, n in [-5, +5]). |
| `infra_launch_quantum_rank` | needs_review | `infra/launch_quantum_rank.py` | Launch EC2 instance for quantum commutator algebra rank computation. |
| `infra_launch_structure_level3` | needs_review | `infra/launch_structure_level3.py` | Launch EC2 instance for level-3 structure extraction. |
| `infra_launch_structure_xsection` | needs_review | `infra/launch_structure_xsection.py` | Launch EC2 instance for 1D structure cross-section. |
| `planned_mcp_server` | planned | `mcp/server.py` | FastMCP (stdio) server exposing read-only tools over local aws_results/ .npy atlas data: list_configurations, get_ran... |
| `scripts_archive_legacy_figures` | complete | `scripts/archive_legacy_figures.py` | Moves legacy PNGs into legacy_figures_archive/. |
| `scripts_registry_bootstrap` | complete | `scripts/registry_bootstrap.py` | Bootstrap registry/experiments.yaml from discovered scripts. Idempotent; only adds new entries. |
| `scripts_registry_curate` | complete | `scripts/registry_curate.py` | Apply canonical-state patches to selected registry entries. Edit PATCHES dict and re-run. |
| `scripts_registry_new` | complete | `scripts/registry_new.py` | CLI to scaffold a new registry entry for a fresh script. |

## quantum

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `nbody_bell_test` | complete | `nbody/bell_test.py` | Bell test on the pairwise algebra. Local results in nbody/bell_test_results/. |
| `nbody_contextuality_test` | complete | `nbody/contextuality_test.py` | Contextuality test (composite-potential variant). |
| `nbody_energy_bound_search` | complete | `nbody/energy_bound_search.py` | Energy-bound / 117th generator commutant search. Quantum commutant=40 < classical=41: quantization removes one conser... |
| `primes_run_quantum_gue` | complete | `primes/run_quantum_gue.py` | Quantum Moyal bracket on GUE composite. Confirms [3, 6, 17, 116] under hbar deformation (no +1 growth, unlike pure 1/r). |

## spectral

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `primes_finite_n_gue_comparison` | needs_review | `primes/finite_n_gue_comparison.py` | Finite-N GUE comparison for bracket tensor spectral statistics. |
| `primes_level2_spectral_analysis` | complete | `primes/level2_spectral_analysis.py` | Kirillov coadjoint orbit spectral analysis - Bohigas-Giannoni-Schmit conjecture in algebraic structure. 1/r tensor: G... |
| `spectral_depth_mining` | needs_review | `spectral_depth_mining.py` | Spectral Depth Mining — Phase 1 Items 1.1 + 1.2 + 1.3 ======================================================= |

## structure

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `nbody_compare_level3_structure` | wip | `nbody/compare_level3_structure.py` | Level-3 structure comparison across potentials. AWS-scale workload. |
| `nbody_isomorphism_test` | complete | `nbody/isomorphism_test.py` | Cross-potential isomorphism test for L2 algebras. 12 non-harmonic 17-dim algebras canonically isomorphic via Killing ... |
| `nbody_structure_cross_section` | needs_review | `nbody/structure_cross_section.py` | 1D Cross-Section of Algebraic Structure Across Parameter Space ======================================================... |
| `planned_atlas_structure_sweep` | planned | `atlas_structure_sweep.py` | Atlas of structure constants over the (mu, phi) shape sphere. Compute the 32 non-zero structure constants (and Killin... |
| `planned_compare_atlas_structures` | planned | `compare_atlas_structures.py` | Cross-potential comparison of atlas structure-constant tensors. Renders difference maps between universal-class poten... |
| `planned_compute_monodromy` | planned | `compute_monodromy.py` | Parallel-transport the structure-constant basis around a small loop encircling a binary collision and check the resul... |

## test

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `3d_validate_2d` | needs_review | `3d/validate_2d.py` | Validate the ND engine by reproducing the known 2D results. |
| `dataset_validate_dataset` | complete | `dataset/validate_dataset.py` | Validation suite for all dataset splits. |
| `nbody_validate_n3` | needs_review | `nbody/validate_n3.py` | Validate the N-body engine by reproducing known N=3, d=2 results. |
| `test_adaptive_infra` | needs_review | `test_adaptive_infra.py` | Smoke test for adaptive scan infrastructure. |
| `test_cse` | needs_review | `test_cse.py` | Quick test: validate CSE-optimised bracket computation matches direct computation, and measure the speedup. |
| `test_regression` | complete | `test_regression.py` | Regression suite. Checks SymPy >= 1.13.3, NBodyAlgebra (N=3, 1/r) levels 0-2 give [3, 6, 17], and ThreeBodyAlgebra d=... |
| `validate_survey_masses` | needs_review | `validate_survey_masses.py` | Phase 1: Validate survey gravitational results on corrected SymPy. Re-runs three_galaxies (1:2:3) and binary_star_pla... |

## viz

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `animate_atlas` | needs_review | `animate_atlas.py` | Animate the stability atlas as the potential exponent sweeps from 1/r to 1/r^2. |
| `plot_mass_ratio_results` | needs_review | `plot_mass_ratio_results.py` | Comprehensive multi-panel plot of mass ratio sweep results. Combines local level-2 sweep data with AWS level-3 valida... |
| `render_1r2_triptych` | needs_review | `render_1r2_triptych.py` | Render triptych: 1/r^2 (left) | 1/r^-2 (center) | difference (right). |
| `render_full_atlas` | needs_review | `render_full_atlas.py` | Render shape-sphere atlas triptychs for all completed full_atlas configs. |
| `render_lagrange_zoom` | needs_review | `render_lagrange_zoom.py` | Zoomed triptych around the Lagrange equilateral point: 1/r vs 1/r^3. |
| `render_teaser` | needs_review | `render_teaser.py` | Render standalone 1/r^2 gap ratio panel for preprint teaser figure. |
| `viz_atlas_1000` | needs_review | `viz_atlas_1000.py` | 1000x1000 Atlas Visualization Suite ===================================== |
| `viz_comprehensive` | needs_review | `viz_comprehensive.py` | Comprehensive Visualization Suite ==================================== |
| `website_figures_compare` | needs_review | `website/figures_compare.py` | Curated comparison figures for the Figures page. |
| `website_figures_render` | needs_review | `website/figures_render.py` | Canonical figure renderer for the Figures page. |
| `website_preprocess_atlas_data` | needs_review | `website/preprocess_atlas_data.py` | Preprocess atlas .npy data into JSON files for the interactive web explorer. |

## website

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `website_build_dataset_json` | needs_review | `website/build_dataset_json.py` | Build website JSON files from the curated dataset and image inventory. |
| `website_build_figures_manifest` | needs_review | `website/build_figures_manifest.py` | Build website/data/figures/manifest.json from the figures_v2/ tree. |
| `website_render_knee_index` | needs_review | `website/render_knee_index.py` | Render the 1/r² (Calogero-Moser) knee-index map as a single-panel figure for embedding on the research dashboard. |

