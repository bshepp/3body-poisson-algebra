# Experiment Registry

*Auto-generated from `registry/experiments.yaml` — 180 entries.*

**Status:** wip: 3 · complete: 114 · planned: 7 · broken: 1 · needs_review: 41 · superseded: 9 · archived: 5

## analysis

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `3d_run_1d` | complete | `3d/run_1d.py` | Driver: 1D (linear) Poisson algebra computation. Established d-independence for N=3 at d=1. |
| `3d_run_3d` | complete | `3d/run_3d.py` | Driver: 3D (spatial) Poisson algebra computation. Established d-independence for N=3 at d=3. |
| `algebra_growth` | superseded | `algebra_growth.py` | Original Phase 1 numerical-finite-difference engine. Produced unstable level-3 estimates (39-103). Superseded by exac... |
| `analyze_results` | complete | `analyze_results.py` | Post-hoc analysis of early growth runs. One-shot script that consumed algebra_growth output to characterize sensitivity. |
| `aws_level4` | complete | `aws_level4.py` | Level-4 dimension multi-config AWS pipeline. Produced d(4)>=5604 lower bound across multiple configurations. |
| `check_1r2` | complete | `check_1r2.py` | 1/r^2 sanity check script: validates Calogero-Moser variant produces [3, 6, 17, 116]. |
| `clebsch_gordan_analysis` | complete | `clebsch_gordan_analysis.py` | Canonical S_3 representation decomposition of the N=3 Poisson algebra. Establishes n_E = 52 = Tier 1 size exactly. Re... |
| `cm_comparison` | complete | `cm_comparison.py` | Calogero-Moser vs Newtonian gravity bracket-algebra comparison. Established universality across singular potential ty... |
| `data_inventory` | complete | `data_inventory.py` | Complete data inventory utility across all results/ subdirectories. |
| `dirac_analysis_from_svd` | complete | `dirac_analysis_from_svd.py` | Dirac constraint analysis using pre-computed SVD data. Comprehensive noise-floor taxonomy: 32 syzygies + 8 true zeros... |
| `dirac_constraint_test` | complete | `dirac_constraint_test.py` | Tests whether the 40 noise-floor generators are Dirac constraints. Result: 32 are syzygies (Jacobi-identity consequen... |
| `dirac_direct_eval` | complete | `dirac_direct_eval.py` | Direct generator evaluation to identify which level-3 generators vanish everywhere on phase space (true zeros) vs onl... |
| `generate_triptychs` | superseded | `generate_triptychs.py` | Original triptych atlas series renderer. Superseded by the website/figures_compare.py pipeline (Apr 18, 2026 rebuild). |
| `hires_lagrange_scan` | complete | `hires_lagrange_scan.py` | High-resolution Lagrange-region scan tool. Used to resolve the concentric ring features around the equilateral point. |
| `level4_comparison` | complete | `level4_comparison.py` | Level-4 comparison chart (gap_workplan Phase 1 Item 1.5). Bar chart + convergence curves across 4 config types. |
| `level4_highsample` | complete | `level4_highsample.py` | Level-4 high-sample bound tightening. Pushed d(4) bounds with up to 200K samples at generic configs. |
| `mass_ratio_bisection` | archived | `mass_ratio_bisection.py` | HISTORICAL ARTIFACT. The docstring of the script itself flags it as DO NOT USE FOR NEW ANALYSIS. Predates the symboli... |
| `nbody_analyze_117th` | complete | `nbody/analyze_117th.py` | Analysis of the 117th quantum generator: sum-of-squares, negative semi-definite, P3 octupole. |
| `nbody_analyze_117th_compact` | complete | `nbody/analyze_117th_compact.py` | Compact 10-term relative-coordinate form of the 117th quantum generator. |
| `nbody_analyze_117th_definite` | complete | `nbody/analyze_117th_definite.py` | Sign-definiteness proof for the 117th generator: g = -(9/4)*[(A-B)^2 + A^2] is negative semi-definite. |
| `nbody_analyze_117th_physics` | complete | `nbody/analyze_117th_physics.py` | Physical interpretation of the 117th generator: NOT a conserved quantity ([G, H_total] != 0). |
| `nbody_analyze_scaling` | complete | `nbody/analyze_scaling.py` | N-body dimension-sequence scaling analysis. Derived L2(N) = N(4N^2-9N+3)/2 for N>=4 by fitting new_L2 = 12*C(N,3). |
| `nbody_expansion_analysis` | complete | `nbody/expansion_analysis.py` | Multi-system universality survey: analysis + plotting across 21 physical systems. |
| `nbody_fast_level3_compare` | complete | `nbody/fast_level3_compare.py` | Fast numerical L3 structure constant comparison via finite differences. One of four L3 comparison methods. |
| `nbody_identify_117th` | complete | `nbody/identify_117th.py` | Identifies the 117th generator: indices into the basis, Legendre P_3 octupole structure, S_3 average. |
| `nbody_identify_117th_rank` | complete | `nbody/identify_117th_rank.py` | Rank of the 48 hbar^2 corrections among themselves (probes quantum L3 algebra structure). |
| `nbody_numerical_level3_compare` | complete | `nbody/numerical_level3_compare.py` | Numerical SVD-based L3 structure constant comparison between potentials. |
| `nbody_run_helium` | complete | `nbody/run_helium.py` | Helium atom Poisson algebra. Confirms sign of interaction doesn't matter: helium (+2, -1, -1) gives [3, 6, 17, 116]. |
| `nbody_run_n4_d1` | superseded | `nbody/run_n4_d1.py` | Priority 4a: N=4, d=1 through L=2. Now subsumed by the general nbody/symbolic_rank_nbody.py engine. |
| `nbody_run_n4_d2` | superseded | `nbody/run_n4_d2.py` | Priority 1: N=4, d=2, 1/r through L=2. Subsumed by general engine. |
| `nbody_run_n4_d3` | superseded | `nbody/run_n4_d3.py` | Priority 4b: N=4, d=3 through L=2. Subsumed by general engine. |
| `nbody_run_n4_mass` | complete | `nbody/run_n4_mass.py` | N=4 mass invariance test. Confirmed: 3 configs (equal, hierarchical, mixed) all give [6, 14, 62]. |
| `nbody_run_pn_aws` | complete | `nbody/run_pn_aws.py` | AWS orchestrator for composite-potential / post-Newtonian dimension-sequence runs. |
| `nbody_run_pn_mass_test` | complete | `nbody/run_pn_mass_test.py` | Post-Newtonian mass invariance test. |
| `nbody_run_post_newtonian` | complete | `nbody/run_post_newtonian.py` | Post-Newtonian (1PN) three-body Poisson algebra. Confirms PN correction does not change the algebra: [3, 6, 17, 116]. |
| `nbody_run_potential_1r3` | superseded | `nbody/run_potential_1r3.py` | Priority 2: N=3, d=2, 1/r^3 through L=3. Now subsumed by the general engine with --potential 1/r3. |
| `nbody_symbolic_level3_compare` | complete | `nbody/symbolic_level3_compare.py` | Exact symbolic L3 structure constant comparison over QQ (~37h/potential on AWS). |
| `nbody_symbolic_n_level3` | complete | `nbody/symbolic_n_level3.py` | L3 verification at 10 concrete rational p-values for V = -u^p potentials. |
| `nbody_symbolic_n_proof` | complete | `nbody/symbolic_n_proof.py` | Numerical survey of dimension vs potential exponent p. Established [3, 6, 17] is universal at L2 across 79+ rational ... |
| `neural_identify_extra_generators` | complete | `neural/identify_extra_generators.py` | Identifies the 3 extra generators in the gradient-product neural algebra not present in the physical N=3 universal set. |
| `neural_nn_extra_generators` | complete | `neural/nn_extra_generators.py` | Identification and characterization of the 3 extra neural generators (S3 standard rep, all involve H_23). |
| `neural_physics_vs_neural` | complete | `neural/physics_vs_neural.py` | Direct comparison: physical N=3 algebra vs neural gradient-product algebra in the same 6D phase space. |
| `neural_summarize_results` | complete | `neural/summarize_results.py` | Summary table of all saved neural-algebra results. |
| `noise_plateau_mapping` | complete | `noise_plateau_mapping.py` | SVD rank-threshold plateau mapping (gap_workplan section 4.6). Sweeps tau from 1e-1 down to 1e-18 over: (A) the 1/r a... |
| `potential_comparison` | complete | `potential_comparison.py` | Potential comparison study: 1/r vs 1/r^2 vs r^2. Foundational comparison establishing that singular and integrable po... |
| `primes_check_1r_closure` | complete | `primes/check_1r_closure.py` | Closure test for the 1/r potential algebra. Result: NOT closed at 116 (infinite-dimensional). |
| `primes_closure_check` | complete | `primes/closure_check.py` | L=4 dimension computation for 1/r and log potentials. Confirms infinite growth. |
| `primes_hilbert_polya_search` | superseded | `primes/hilbert_polya_search.py` | Hilbert-Polya operator search via GUE Lie algebra structure. Superseded by the infinite-dimensionality discovery: the... |
| `primes_launch_gue` | complete | `primes/launch_gue.py` | AWS EC2 spot launcher for the GUE log-gas Poisson algebra computation. Ran the 4-config GUE study. |
| `primes_multi_potential_r_comparison` | complete | `primes/multi_potential_r_comparison.py` | Multi-potential <r> (coadjoint orbit spacing ratio) comparison across all singular potentials. Confirms <r> ~ 0.64 un... |
| `quantization_analysis` | complete | `quantization_analysis.py` | Tier-structure statistics + LQG/Bekenstein-style integer-scaling hypothesis tests. Foundational for Paper 2 (S_3 jet ... |
| `rerender_frames` | complete | `rerender_frames.py` | Atlas animation re-render utility with adaptive colorscale. One-shot fix for early animation issues. |
| `run_cm_exact` | complete | `run_cm_exact.py` | Exact Calogero-Moser run by patching exact_growth.py Hamiltonians. Reuses the full infrastructure. |
| `s4_tier_analysis` | complete | `s4_tier_analysis.py` | Pure representation-theoretic decomposition of the N=4 pairwise Poisson algebra under S_4. Computes Clebsch-Gordan co... |
| `sv116_analytical` | complete | `sv116_analytical.py` | SV #116 analytical prediction (gap_workplan Phase 1 Item 1.6). Achieved R^2 = 0.630 correlation between analytical pr... |
| `unequal_mass_study` | complete | `unequal_mass_study.py` | Unequal-mass dimension-sequence study. The original [3,5,13,69] vs [3,6,17,116] artifact study; superseded by the sym... |

## atlas

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `analyze_atlas_data` | complete | `analyze_atlas_data.py` | Diagnostic for atlas_1000 data: clean-merge + excluded-block check. |
| `assemble_atlases` | superseded | `assemble_atlases.py` | Original atlas assembly + comprehensive shape-sphere visualization. Superseded by the website/figures_render.py + fig... |
| `atlas_1000` | complete | `atlas_1000.py` | 1000x1000 parallelized high-resolution shape-sphere scan. Produced the canonical hi-res atlases. |
| `atlas_diagnostics` | complete | `atlas_diagnostics.py` | Atlas diagnostic checks. Validates the atlas pipeline output. |
| `audit_atlas_data` | complete | `audit_atlas_data.py` | Audit local atlas data against S3 to detect stale/incomplete syncs (fixes the s3 sync size-only artifact for fixed-si... |
| `cg_atlas_comparison` | complete | `cg_atlas_comparison.py` | Clebsch-Gordan atlas comparison (gap_workplan Phase 1 Item 1.4). Lagrange=33 doublets (E-frac=0.57), Euler=14, Iso-90... |
| `full_atlas_scan` | complete | `full_atlas_scan.py` | Full shape-sphere atlas scanner with SVD gap analysis and mass-aware momentum sampling. Core atlas tool. |
| `merge_atlas_1000` | complete | `merge_atlas_1000.py` | Selective merge of 1000x1000 atlas data across multiple AWS spot runs. |
| `multi_epsilon_atlas` | complete | `multi_epsilon_atlas.py` | Multi-epsilon adaptive structure atlas. Supports --charges, multiprocessing, spot instances. |
| `n4_atlas_1d` | complete | `n4_atlas_1d.py` | First atlas data for N=4: three 1D slices. All 300 points show rank 62 (no drops). |
| `nbody_helium_atlas` | complete | `nbody/helium_atlas.py` | Helium atlas: charge-sign invariance comparison tool. Used for the (+2, -1, -1) atlas vs gravitational baseline. |
| `parametric_atlas_scan` | broken | `parametric_atlas_scan.py` | Parametric 1/r^n exponent atlas. Aborted on cost (~$1500 projected). Needs redesign per gap_workplan.md 3.1: coarse L... |
| `run_expansion_atlas` | complete | `run_expansion_atlas.py` | AWS orchestrator for the Multi-System Universality Survey stability atlases. Produced the 18 completed atlas configur... |
| `shape_sphere` | complete | `shape_sphere.py` | Shape sphere gap-ratio landscape renderer. The original shape-sphere visualization (canonical for paper 1). |
| `shape_sphere_atlas` | complete | `shape_sphere_atlas.py` | Direct S^2 sampling of the N=3 shape sphere using the canonical Hsiang-Montgomery Jacobi projection, filling the prev... |
| `shape_sphere_hires` | superseded | `shape_sphere_hires.py` | High-resolution shape-sphere atlas; superseded by shape_sphere_atlas.py (canonical Hsiang-Montgomery Jacobi projectio... |
| `stability_atlas` | complete | `stability_atlas.py` | Exact-engine atlas scanner. Drives most of the (mu, phi) shape-sphere atlases. |
| `sv_landscape_viz` | complete | `sv_landscape_viz.py` | Singular-value landscape visualizations across the shape sphere. Used for the spectral depth mining figures. |
| `targeted_adaptive_scan` | complete | `targeted_adaptive_scan.py` | High-resolution adaptive scans of named regions (Lagrange, Euler, etc.) for any potential. |

## dataset

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `dataset_build_dataset` | complete | `dataset/build_dataset.py` | Build the 13-table HF dataset from ~200 result JSONs. Outputs Parquet; uploaded to huggingface.co/datasets/bshepp/pai... |

## diagnostic

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `bench_flint` | complete | `bench_flint/bench.py` | python-flint vs default ground-types benchmark on the actual project workload. Three benches: (1) micro cancel() on a... |
| `bench_simplify_identity` | complete | `bench_flint/test_simplify_identity.py` | Symbolic-identity spot check for the simplify_generator patch. Generates raw L=0..L=2 brackets via NBodyAlgebra.poiss... |
| `bench_simplify_phases` | complete | `bench_flint/run_simplify_phases.py` | Three-phase smarter-simplify experiment. Tests whether `together` can replace `cancel` in NBodyAlgebra.simplify_gener... |
| `bench_simplify_poly_compat` | complete | `bench_flint/test_simplify_poly_compat.py` | Phase E0 gate. Verifies that for raw L=1 and L=2 Poisson brackets in NBodySymbolicRank(N=3, d=2, potential='1/r'),   ... |
| `bench_symbolic_rank_n4` | complete | `bench_flint/bench_symbolic_rank_n4.py` | Phase E3 N=4 d=1 1/r generation-only timing comparison. Pass 1: L=2 head-to-head, both cancel and together. Pass 2: L... |
| `bench_validate_simplify_patch` | complete | `bench_flint/validate_simplify_patch.py` | Paranoid 8-stage validator for the simplify_generator patch. Runs each case as a separate Python subprocess under ben... |
| `bench_validate_symbolic_rank_patch` | complete | `bench_flint/validate_symbolic_rank_patch.py` | Phase E2 validator for the symbolic_rank_nbody simplify patch. Five N=3 cases pinned against results/symbolic_rank/ra... |
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
| `mass_ratio_sweep` | needs_review | `mass_ratio_sweep.py` | Mass Ratio Sweep â€” track dimension sequence and SVD gaps as mass ratio varies continuously from equal (1:1:1) to ex... |
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
| `planned_replay_schwarzschild_l3` | complete | `nbody/run_schwarzschild.py` | Re-ran the Schwarzschild L3 sweep that died yesterday under the cancel engine. With the patched together engine the 4... |
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
| `simplify_generator_patch` | complete | `nbody/exact_growth_nbody.py` | Patched nbody/exact_growth_nbody.py simplify_generator() from cancel(expr) to together(expr). Single-line change at l... |
| `simplify_patch_symbolic_rank` | complete | `nbody/symbolic_rank_nbody.py` | Patched nbody/symbolic_rank_nbody.py _simplify (line 318-330) from cancel(expr) to together(expr) following the same ... |

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
| `launch_lane_c` | complete | `infra/launch_lane_c.py` | AWS launcher for Lane C: r6a.16xlarge SPOT on AL2023 with python3.11 and python-flint. Boots the streaming mod-p N=3 ... |
| `launch_qeps` | complete | `infra/launch_qeps.py` | AWS launcher for the Q(eps) symbolic-nullspace job: computes the exact left-null kernel of the (4,3) binary-collision... |
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
| `spectral_depth_mining` | needs_review | `spectral_depth_mining.py` | Spectral Depth Mining â€” Phase 1 Items 1.1 + 1.2 + 1.3 ======================================================= |

## structure

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `harmonic_lie_algebra_id` | complete | `harmonic_lie_algebra_id.py` | Identifies the 15-dim Lie algebra of the harmonic (r^2) 3-body planar Poisson algebra. Reads exact rational structure... |
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
| `test_shape_sphere_atlas` | complete | `test_shape_sphere_atlas.py` | Regression test for shape_sphere_atlas.py covering binary-collision marking and Jacobi projection invariants. |
| `validate_survey_masses` | needs_review | `validate_survey_masses.py` | Phase 1: Validate survey gravitational results on corrected SymPy. Re-runs three_galaxies (1:2:3) and binary_star_pla... |

## validation

| ID | Status | Path | Description |
|----|--------|------|-------------|
| `mathematica_l4_backup` | planned | `mathematica/poisson_n3_d2_l4_backup.wl` | L=4 1/r backup oracle. Held in reserve in case the HF Jobs cpu-xl Phase D2 jobs (69e820f1cd8c002f31e0140b for 1/r and... |
| `mathematica_oracle_n3_d2` | complete | `mathematica/poisson_n3_d2.wl` | Phase F independent CAS oracle. Reproduces the cumulative-rank dimension sequence [3, 6, 17, 116] for N=3 d=2 at L=3 ... |
| `mathematica_oracle_n3_d2_harmonic` | complete | `mathematica/poisson_n3_d2_harmonic.wl` | Phase F.2 harmonic CAS oracle. Reproduces [3, 6, 13, 15, 15] for the harmonic potential H_ij = T_i + T_j + r_ij^2 at ... |
| `sage_oracle_n3_d2` | complete | `sage/poisson_n3_d2.sage` | Phase G.1 SageMath sanity runner. Reproduces the cumulative-rank dimension sequence [3, 6, 17, 116] for N=3 d=2 at L=... |
| `sage_oracle_n3_d2_engine` | complete | `sage/poisson_n3_d2_engine.sage` | Shared SageMath engine for the planar 3-body Poisson algebra (Phase G.1 third-leg oracle). Mirrors mathematica/poisso... |
| `sage_oracle_n3_d2_harmonic` | complete | `sage/poisson_n3_d2_harmonic.sage` | Phase G.1 SageMath harmonic-closure runner. Reproduces [3, 6, 13, 15, 15] for the harmonic potential H_ij = T_i + T_j... |

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
| `website_render_knee_index` | needs_review | `website/render_knee_index.py` | Render the 1/rÂ² (Calogero-Moser) knee-index map as a single-panel figure for embedding on the research dashboard. |

