#!/usr/bin/env python3
"""
NBodyAlgebra Diagnostic — Steps 3-5
Compare symbolic Hamiltonians, brackets, and evaluation matrices
between exact_growth.py and NBodyAlgebra engines.
"""
import sys
import numpy as np
sys.setrecursionlimit(100000)

import sympy as sp
from sympy import Integer, cancel, diff, Add

# =====================================================================
# Step 3: Compare symbolic Hamiltonians
# =====================================================================
print("=" * 70)
print("STEP 3: COMPARE SYMBOLIC HAMILTONIANS")
print("=" * 70)

# Engine A: exact_growth.py
from exact_growth import (
    H12 as H12_A, H13 as H13_A, H23 as H23_A,
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    poisson_bracket as pb_A, total_deriv as td_A,
    sample_phase_space as sample_A, lambdify_generators as lambdify_A,
)

# Engine B: NBodyAlgebra
sys.path.insert(0, "nbody")
from exact_growth_nbody import NBodyAlgebra

alg = NBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r",
                   masses={1: Integer(1), 2: Integer(1), 3: Integer(1)})

H12_B = alg.hamiltonian_list[0]
H13_B = alg.hamiltonian_list[1]
H23_B = alg.hamiltonian_list[2]

for name, ha, hb in [("H12", H12_A, H12_B), ("H13", H13_A, H13_B),
                      ("H23", H23_A, H23_B)]:
    diff_expr = cancel(sp.expand(ha) - sp.expand(hb))
    print(f"\n  {name} (engine A): {ha}")
    print(f"  {name} (engine B): {hb}")
    print(f"  Difference:        {diff_expr}")
    print(f"  IDENTICAL: {diff_expr == 0}")

# =====================================================================
# Step 3b: Compare variable ordering
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3b: COMPARE VARIABLE ORDERING")
print("=" * 70)
print(f"  Engine A Q_VARS: {Q_VARS}")
print(f"  Engine B q_vars: {alg.q_vars}")
print(f"  Q match: {Q_VARS == alg.q_vars}")
print()
print(f"  Engine A P_VARS: {P_VARS}")
print(f"  Engine B p_vars: {alg.p_vars}")
print(f"  P match: {P_VARS == alg.p_vars}")
print()
print(f"  Engine A U_VARS: {U_VARS}")
print(f"  Engine B u_vars: {alg.u_vars}")
print(f"  U match: {U_VARS == alg.u_vars}")
print()
print(f"  Engine A ALL_VARS: {ALL_VARS}")
print(f"  Engine B all_vars: {alg.all_vars}")
print(f"  ALL match: {ALL_VARS == alg.all_vars}")

# =====================================================================
# Step 3c: Compare chain rule tables
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3c: COMPARE CHAIN RULE TABLES")
print("=" * 70)
from exact_growth import CHAIN_RULE

all_match = True
for key in CHAIN_RULE:
    u_var, q_var = key
    val_a = CHAIN_RULE[key]
    if key in alg.chain_rule:
        val_b = alg.chain_rule[key]
        d = cancel(val_a - val_b)
        if d != 0:
            print(f"  MISMATCH at {key}: A={val_a}, B={val_b}, diff={d}")
            all_match = False
    else:
        print(f"  MISSING in engine B: {key}")
        all_match = False

for key in alg.chain_rule:
    if key not in CHAIN_RULE:
        print(f"  EXTRA in engine B: {key} = {alg.chain_rule[key]}")
        all_match = False

print(f"  Chain rule tables IDENTICAL: {all_match}")
print(f"  Engine A has {len(CHAIN_RULE)} entries")
print(f"  Engine B has {len(alg.chain_rule)} entries")

# =====================================================================
# Step 4: Compare level-1 brackets
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: COMPARE LEVEL-1 BRACKETS")
print("=" * 70)

print("\n  Computing {H12, H13} in engine A...", end=" ", flush=True)
K1_A = cancel(pb_A(H12_A, H13_A))
print(f"{len(Add.make_args(K1_A))} terms")

print("  Computing {H12, H13} in engine B...", end=" ", flush=True)
K1_B = cancel(alg.poisson_bracket(H12_B, H13_B))
print(f"{len(Add.make_args(K1_B))} terms")

diff_K1 = cancel(sp.expand(K1_A) - sp.expand(K1_B))
print(f"  K1 difference: {diff_K1}")
print(f"  K1 IDENTICAL: {diff_K1 == 0}")

print("\n  Computing {H12, H23} in engine A...", end=" ", flush=True)
K2_A = cancel(pb_A(H12_A, H23_A))
print(f"{len(Add.make_args(K2_A))} terms")

print("  Computing {H12, H23} in engine B...", end=" ", flush=True)
K2_B = cancel(alg.poisson_bracket(H12_B, H23_B))
print(f"{len(Add.make_args(K2_B))} terms")

diff_K2 = cancel(sp.expand(K2_A) - sp.expand(K2_B))
print(f"  K2 difference: {diff_K2}")
print(f"  K2 IDENTICAL: {diff_K2 == 0}")

print("\n  Computing {H13, H23} in engine A...", end=" ", flush=True)
K3_A = cancel(pb_A(H13_A, H23_A))
print(f"{len(Add.make_args(K3_A))} terms")

print("  Computing {H13, H23} in engine B...", end=" ", flush=True)
K3_B = cancel(alg.poisson_bracket(H13_B, H23_B))
print(f"{len(Add.make_args(K3_B))} terms")

diff_K3 = cancel(sp.expand(K3_A) - sp.expand(K3_B))
print(f"  K3 difference: {diff_K3}")
print(f"  K3 IDENTICAL: {diff_K3 == 0}")

# =====================================================================
# Step 5: Compare evaluation matrices (level 0 + level 1)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: COMPARE EVALUATION MATRICES (through level 1)")
print("=" * 70)

seed = 42
n_samples = 500

# Engine A sampling
Z_qp_A, Z_u_A = sample_A(n_samples, seed)
print(f"  Engine A: Z_qp shape={Z_qp_A.shape}, Z_u shape={Z_u_A.shape}")

# Engine B sampling
Z_qp_B, Z_u_B = alg.sample_phase_space(n_samples, seed)
print(f"  Engine B: Z_qp shape={Z_qp_B.shape}, Z_u shape={Z_u_B.shape}")

# Compare sample points
qp_diff = np.max(np.abs(Z_qp_A - Z_qp_B))
u_diff = np.max(np.abs(Z_u_A - Z_u_B))
print(f"  Max |Z_qp_A - Z_qp_B|: {qp_diff:.2e}")
print(f"  Max |Z_u_A - Z_u_B|:   {u_diff:.2e}")

# Evaluate level 0+1 generators in both engines
exprs_A = [H12_A, H13_A, H23_A, K1_A, K2_A, K3_A]
exprs_B = [H12_B, H13_B, H23_B, K1_B, K2_B, K3_B]
names = ["H12", "H13", "H23", "K1", "K2", "K3"]

eval_A_fn = lambdify_A(exprs_A)
eval_B_fn = alg.lambdify_generators(exprs_B)

mat_A = eval_A_fn(Z_qp_A, Z_u_A)
mat_B = eval_B_fn(Z_qp_B, Z_u_B)

print(f"\n  Evaluation matrix shapes: A={mat_A.shape}, B={mat_B.shape}")

print("\n  Column-by-column comparison:")
for i, name in enumerate(names):
    col_diff = np.max(np.abs(mat_A[:, i] - mat_B[:, i]))
    col_rel = col_diff / max(np.max(np.abs(mat_A[:, i])), 1e-300)
    print(f"    {name}: max |A-B| = {col_diff:.2e}, "
          f"relative = {col_rel:.2e}, "
          f"MATCH = {col_diff < 1e-10}")

# SVD comparison
_, s_A, _ = np.linalg.svd(mat_A, full_matrices=False)
_, s_B, _ = np.linalg.svd(mat_B, full_matrices=False)

print("\n  SVD singular values comparison (through level 1):")
print(f"    {'idx':>5} | {'sv_A':>18} | {'sv_B':>18} | {'|diff|':>12}")
for i in range(len(s_A)):
    print(f"    {i+1:>5} | {s_A[i]:>18.12f} | {s_B[i]:>18.12f} | "
          f"{abs(s_A[i]-s_B[i]):>12.2e}")

print(f"\n  Rank A: {np.sum(s_A > 1e-10)}")
print(f"  Rank B: {np.sum(s_B > 1e-10)}")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY (through level 1)")
print("=" * 70)
