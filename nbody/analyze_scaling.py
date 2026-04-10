#!/usr/bin/env python3
"""Analyze N-body dimension sequence scaling."""

from math import comb
from fractions import Fraction

data = {
    3: [3, 6, 17, 116],
    4: [6, 14, 62],
    5: [10, 25, 145],
    6: [15, 39, 279],
}

Ns = [3, 4, 5, 6]

print("=" * 70)
print("N-BODY DIMENSION SEQUENCE ANALYSIS")
print("=" * 70)

# Level 0
print("\nLevel 0: C(N,2) = N(N-1)/2")

# Level 1 -- find formula
L1 = {N: data[N][1] for N in data}
new_L1 = {N: L1[N] - comb(N, 2) for N in Ns}
print(f"\nLevel 1 values:  {[L1[N] for N in Ns]}")
print(f"New at L1:       {[new_L1[N] for N in Ns]}")

# new_L1 = {3:3, 4:8, 5:15, 6:24}
# Check (N-2)*N
print("\nTest: new_L1 = N*(N-2):")
for N in Ns:
    v = N * (N - 2)
    ok = "YES" if v == new_L1[N] else "no"
    print(f"  N={N}: N*(N-2)={v}, actual={new_L1[N]}, {ok}")

# Therefore L1 = C(N,2) + N(N-2) = N(N-1)/2 + N^2 - 2N = (3N^2 - 5N)/2
print("\nTest: L1 = N(3N-5)/2:")
for N in Ns:
    v = N * (3 * N - 5) // 2
    ok = "YES" if v == L1[N] else "no"
    print(f"  N={N}: {v}, actual={L1[N]}, {ok}")

print("\nPredictions for L1:")
for N in [7, 8, 10, 20]:
    print(f"  N={N}: L1 = {N * (3 * N - 5) // 2}")

# Level 2
L2 = {N: data[N][2] for N in data}
new_L2 = {N: L2[N] - L1[N] for N in Ns}
print(f"\nLevel 2 values:  {[L2[N] for N in Ns]}")
print(f"New at L2:       {[new_L2[N] for N in Ns]}")

# Verify polynomial: L2 = (13N^3 - 42N^2 + 83N - 120)/6
print("\nTest: L2 = (13N^3 - 42N^2 + 83N - 120)/6:")
for N in Ns:
    num = 13 * N**3 - 42 * N**2 + 83 * N - 120
    v = num // 6
    r = num % 6
    ok = "YES" if v == L2[N] and r == 0 else "no"
    print(f"  N={N}: {v} (rem {r}), actual={L2[N]}, {ok}")

print("\nPredictions for L2:")
for N in [7, 8, 10, 20]:
    v = (13 * N**3 - 42 * N**2 + 83 * N - 120) // 6
    print(f"  N={N}: L2 = {v}")

# Analyze new_L2: 11, 48, 120, 240
print("\n--- Analyzing new_at_L2 = 11, 48, 120, 240 ---")
diffs = []
vals = [new_L2[N] for N in Ns]
for i in range(1, len(vals)):
    diffs.append(vals[i] - vals[i-1])
print(f"  First differences: {diffs}")
diffs2 = [diffs[i] - diffs[i-1] for i in range(1, len(diffs))]
print(f"  Second differences: {diffs2}")

# Quadratic fit: new_L2 = a*N^2 + b*N + c using N=4,5,6 (ignoring N=3 anomaly?)
import numpy as np
A = np.array([[N**2, N, 1] for N in Ns], dtype=float)
b_vec = np.array(vals, dtype=float)
# Use least-squares for 3 coefficients with 4 data points
coeffs, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
print(f"\nQuadratic fit: {coeffs[0]:.2f}*N^2 + {coeffs[1]:.2f}*N + {coeffs[2]:.2f}")
for N in Ns:
    pred = coeffs[0]*N**2 + coeffs[1]*N + coeffs[2]
    print(f"  N={N}: predicted={pred:.1f}, actual={new_L2[N]}")

# Try more formulas for new_L2
print("\n  Brute search: new_L2 = a*C(N,2) + b*C(N,3) + c*C(N,4) + d:")
found = False
for a in range(-30, 31):
    for b in range(-30, 31):
        for c in range(-30, 31):
            for d in range(-30, 31):
                if all(a*comb(N,2) + b*comb(N,3) + c*comb(N,4) + d == new_L2[N]
                       for N in Ns):
                    print(f"    FOUND: {a}*C(N,2) + {b}*C(N,3) + {c}*C(N,4) + {d}")
                    found = True
if not found:
    print("    No match found")

# Also brute search for L2 itself
print("\n  Brute search: L2 = a*C(N,2) + b*C(N,3) + c*C(N,4) + d:")
found = False
for a in range(-30, 31):
    for b in range(-30, 31):
        for c in range(-30, 31):
            for d in range(-30, 31):
                if all(a*comb(N,2) + b*comb(N,3) + c*comb(N,4) + d == L2[N]
                       for N in Ns):
                    print(f"    FOUND: {a}*C(N,2) + {b}*C(N,3) + {c}*C(N,4) + {d}")
                    found = True
if not found:
    print("    No match found")

# Try: L2 = a*C(N,2)^2 + b*C(N,2) + c
print("\n  Testing L2 = a*C(N,2)^2 + b*C(N,2) + c:")
A2 = np.array([[comb(N,2)**2, comb(N,2), 1] for N in Ns], dtype=float)
b2 = np.array([L2[N] for N in Ns], dtype=float)
coeffs2, res, _, _ = np.linalg.lstsq(A2, b2, rcond=None)
print(f"  Fit: {coeffs2[0]:.6f}*C(N,2)^2 + {coeffs2[1]:.6f}*C(N,2) + {coeffs2[2]:.6f}")
for i, c in enumerate(coeffs2):
    print(f"    ~ {Fraction(c).limit_denominator(100)}")
for N in Ns:
    c2 = comb(N,2)
    pred = coeffs2[0]*c2**2 + coeffs2[1]*c2 + coeffs2[2]
    print(f"  N={N}: predicted={pred:.2f}, actual={L2[N]}")

# Level 3 -- only N=3 data
print("\n\n--- Level 3 ---")
print(f"  N=3: L3=116, new=99")
print(f"  Only one data point. N=5 level 3 running on AWS.")

# Growth summary
print("\n\n--- SUMMARY ---")
print(f"  L0(N) = C(N,2) = N(N-1)/2")
print(f"  L1(N) = N(3N-5)/2")
print(f"  L2(N) = (13N^3 - 42N^2 + 83N - 120)/6")
print(f"  L3(N) = only N=3 data: 116. Need more data points.")
print()
print("  Growth is super-polynomial: leading terms go as")
print("    L0 ~ N^2/2")
print("    L1 ~ 3N^2/2")
print("    L2 ~ 13N^3/6")
print("    L3 ~ ? (need N=4,5 data)")
print()
print("  The leading degree in N increases with level,")
print("  suggesting the algebra dimension grows as N^(level+1).")
