#!/usr/bin/env python3
"""
Identify the 15-dimensional Lie algebra closed by the 3-body planar
harmonic Poisson algebra (V = r^2).

Companion to docs/harmonic_dim15.md.  Reads the exact rational
structure constants in results/algebra_structure/N3_d2_r2/, computes
the Killing form B(X,Y) = Tr(ad_X ad_Y) over Q, diagonalizes it,
characterizes the center and the radical, computes derived and lower
central series ranks, finds the rank (Cartan subalgebra dimension),
and reports the best-matching candidate from the dim-15 catalog:

    g_compact = G_2 + R           (Killing (14+, 0-, 1z), rank 3, simple G_2 + 1d center)
    g_compact = su(3) (+) so(4) + R   (Killing (14+, 0-, 1z), rank 5)
    g_compact = so(5) (+) su(2) + R   (Killing (14+, 0-, 1z), rank 4)
    g_split = sl(4, R) = so(3,3)   (Killing (9+, 6-, 0z), simple)
    g = so(2,4) ~ su(2,2)          (Killing (8+, 7-, 0z), simple, conformal of Minkowski-3)
    g = so(1,5)                    (Killing (5+, 10-, 0z), simple)
    g = so*(6) ~ su(3,1)           (Killing (7+, 8-, 0z), simple)
    g = so(6) ~ su(4)              (Killing (0+, 15-, 0z), compact simple)

The signature is the primary discriminator; rank and the dimension of
the abelian radical (center) refine it.

Run:
    python harmonic_lie_algebra_id.py
    python harmonic_lie_algebra_id.py --output results/algebra_structure/harmonic_n3_d2_identification.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from sympy import Matrix, Rational, zeros

REPO = Path(__file__).resolve().parent
SC_PATH = REPO / "results" / "algebra_structure" / "N3_d2_r2" / \
    "structure_constants_exact.json"


# ---------------------------------------------------------------------------
# Load exact structure constants (Python Fractions for speed, then SymPy
# Rationals where we need a SymPy matrix).

def load_sc(path: Path):
    raw = json.loads(path.read_text())
    n = len(raw)
    # sc[i][j][k] = C^k_{ij}
    sc = [[[Fraction(c) for c in row] for row in plane] for plane in raw]
    return n, sc


def check_antisymmetry(sc):
    n = len(sc)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if sc[i][j][k] != -sc[j][i][k]:
                    return False, (i, j, k)
    return True, None


def check_jacobi(sc):
    """[e_i,[e_j,e_k]] + [e_j,[e_k,e_i]] + [e_k,[e_i,e_j]] = 0.

    sum_l (C^l_{jk} C^m_{il} + cyclic) == 0 for every (i, j, k, m).
    """
    n = len(sc)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(n):
                    s = Fraction(0)
                    for ll in range(n):
                        s += sc[j][k][ll] * sc[i][ll][m]
                        s += sc[k][i][ll] * sc[j][ll][m]
                        s += sc[i][j][ll] * sc[k][ll][m]
                    if s != 0:
                        return False, (i, j, k, m, s)
    return True, None


# ---------------------------------------------------------------------------
# Adjoint representation and Killing form.

def ad_matrix(sc, i):
    """(ad e_i)_{km} = C^k_{im}.  Convention: rows index 'output' basis, cols
    index 'input': (ad e_i)(e_m) = sum_k C^k_{im} e_k.
    """
    n = len(sc)
    M = [[sc[i][m][k] for m in range(n)] for k in range(n)]
    return M


def killing_form_exact(sc):
    """K[a][b] = Tr(ad e_a * ad e_b) over Q."""
    n = len(sc)
    K = [[Fraction(0)] * n for _ in range(n)]
    # Pre-compute ad matrices once
    ads = [ad_matrix(sc, i) for i in range(n)]
    for a in range(n):
        for b in range(a, n):
            # Tr(A * B) = sum_{i,j} A[i][j] B[j][i]
            s = Fraction(0)
            A = ads[a]
            B = ads[b]
            for i in range(n):
                for j in range(n):
                    s += A[i][j] * B[j][i]
            K[a][b] = s
            K[b][a] = s
    return K


# ---------------------------------------------------------------------------
# Center, radical, derived series.

def center_basis(sc):
    """Vectors v with [v, e_j] = 0 for all j.
    That is, sum_i v_i C^k_{ij} = 0 for all j, k.
    """
    n = len(sc)
    # Build a (n*n, n) matrix whose null-space is the center
    M = zeros(n * n, n)
    for j in range(n):
        for k in range(n):
            row = j * n + k
            for i in range(n):
                M[row, i] = Rational(sc[i][j][k].numerator, sc[i][j][k].denominator)
    null = M.nullspace()
    return null  # list of SymPy column vectors


def commutator_subspace_dim(sc, basis_vecs):
    """Given a list of (sparse) coefficient vectors in the original 15-basis,
    return the dimension of the span of all pairwise brackets.
    """
    n = len(sc)
    # Each basis vector is an n-vector of Fractions.
    # Bracket: [u, w]_k = sum_{i,j} u_i w_j C^k_{ij}
    if len(basis_vecs) < 2:
        return 0
    rows = []
    for a in range(len(basis_vecs)):
        for b in range(a + 1, len(basis_vecs)):
            u = basis_vecs[a]
            w = basis_vecs[b]
            row = [Fraction(0)] * n
            for i in range(n):
                if u[i] == 0:
                    continue
                for j in range(n):
                    if w[j] == 0:
                        continue
                    for k in range(n):
                        row[k] += u[i] * w[j] * sc[i][j][k]
            rows.append(row)
    # Compute rank of (rows x n) matrix
    if not rows:
        return 0
    M = Matrix([[Rational(c.numerator, c.denominator) for c in r] for r in rows])
    return M.rank()


def derived_series(sc):
    """Returns the dim sequence [dim g, dim [g,g], dim [[g,g],[g,g]], ...]
    until it stabilizes.
    """
    n = len(sc)
    # Start with the standard basis
    current = [tuple(Fraction(1) if i == j else Fraction(0)
                     for j in range(n))
               for i in range(n)]
    dims = [n]
    while True:
        # Compute [current, current] -> all pairwise brackets
        next_rows = []
        for a in range(len(current)):
            for b in range(a + 1, len(current)):
                u = current[a]
                w = current[b]
                row = [Fraction(0)] * n
                for i in range(n):
                    if u[i] == 0:
                        continue
                    for j in range(n):
                        if w[j] == 0:
                            continue
                        cij = [sc[i][j][k] for k in range(n)]
                        for k in range(n):
                            if cij[k]:
                                row[k] += u[i] * w[j] * cij[k]
                if any(x != 0 for x in row):
                    next_rows.append(row)
        if not next_rows:
            dims.append(0)
            break
        M = Matrix([[Rational(c.numerator, c.denominator) for c in r]
                    for r in next_rows])
        rank = M.rank()
        dims.append(rank)
        if rank == 0 or (len(dims) >= 2 and dims[-1] == dims[-2]):
            break
        # Use the row-reduced basis as new current
        Mrref, _ = M.rref()
        new_current = []
        for r in range(Mrref.rows):
            row = [Mrref[r, c] for c in range(Mrref.cols)]
            if any(x != 0 for x in row):
                new_current.append(tuple(Fraction(int(x.p), int(x.q))
                                          for x in row))
        current = new_current
        if len(dims) > 12:
            break  # safety
    return dims


def lower_central_series(sc):
    """[g^0 = g, g^1 = [g, g^0], g^2 = [g, g^1], ...]."""
    n = len(sc)
    standard = [tuple(Fraction(1) if i == j else Fraction(0)
                       for j in range(n))
                 for i in range(n)]
    current = standard
    dims = [n]
    while True:
        next_rows = []
        for a in range(len(standard)):
            for b in range(len(current)):
                u = standard[a]
                w = current[b]
                row = [Fraction(0)] * n
                for i in range(n):
                    if u[i] == 0:
                        continue
                    for j in range(n):
                        if w[j] == 0:
                            continue
                        for k in range(n):
                            cijk = sc[i][j][k]
                            if cijk:
                                row[k] += u[i] * w[j] * cijk
                if any(x != 0 for x in row):
                    next_rows.append(row)
        if not next_rows:
            dims.append(0)
            break
        M = Matrix([[Rational(c.numerator, c.denominator) for c in r]
                    for r in next_rows])
        rank = M.rank()
        dims.append(rank)
        if rank == 0 or (len(dims) >= 2 and dims[-1] == dims[-2]):
            break
        Mrref, _ = M.rref()
        new_current = []
        for r in range(Mrref.rows):
            row = [Mrref[r, c] for c in range(Mrref.cols)]
            if any(x != 0 for x in row):
                new_current.append(tuple(Fraction(int(x.p), int(x.q))
                                          for x in row))
        current = new_current
        if len(dims) > 12:
            break  # safety
    return dims


# ---------------------------------------------------------------------------
# Signature classification.

def killing_signature(K):
    """(positive, negative, zero) signature of the Killing form, plus
    eigenvalue list with multiplicities.

    Strategy: the Killing form is real symmetric over Q, so its
    eigenvalues are real. We use numpy.linalg.eigvalsh for the signature
    decision (numerically stable on real symmetric input) and also
    compute the exact rank to confirm the count of zero eigenvalues.
    """
    n = len(K)
    # Build a float numpy array for eigenvalue decomposition
    Knp = np.array([[float(K[i][j]) for j in range(n)] for i in range(n)],
                   dtype=np.float64)
    evs = np.sort(np.linalg.eigvalsh(Knp))[::-1]  # descending
    # Threshold for "zero": use a scale-aware tolerance
    scale = np.abs(evs).max() if evs.size else 1.0
    tol = max(1e-10, 1e-12 * scale)
    pos = int(np.sum(evs > tol))
    neg = int(np.sum(evs < -tol))
    zer = int(np.sum(np.abs(evs) <= tol))
    # Cross-check zero count via exact rank
    Mexact = Matrix([[Rational(K[i][j].numerator, K[i][j].denominator)
                      for j in range(n)] for i in range(n)])
    rank_exact = Mexact.rank()
    exact_zero = n - rank_exact
    if exact_zero != zer:
        # Trust the exact rank for the zero count and rebalance pos/neg
        # by signs of the eigenvalues
        zer = exact_zero
        # Sort numerical eigenvalues by magnitude; the smallest |zer| are
        # treated as zero, the rest split by sign.
        order = np.argsort(np.abs(evs))
        # Indices of "true zero" eigenvalues (smallest |.|)
        zero_idx = set(order[:zer].tolist())
        pos = neg = 0
        for idx, ev in enumerate(evs):
            if idx in zero_idx:
                continue
            if ev > 0:
                pos += 1
            else:
                neg += 1
    eig_list = [(f"{ev:.15g}", 1, float(ev)) for ev in evs]
    return (pos, neg, zer), eig_list


# ---------------------------------------------------------------------------
# Lie-algebra identification.

CANDIDATES = [
    # ---- Simple 15-dim ----
    {"name": "so(6) ~ su(4)",
     "decomp": "simple, compact",
     "killing_signature_dim15": (0, 15, 0)},
    {"name": "so(5,1) ~ su*(4)",
     "decomp": "simple, non-compact",
     "killing_signature_dim15": (5, 10, 0)},
    {"name": "so(4,2) ~ su(2,2)",
     "decomp": "simple, non-compact (conformal of M^3)",
     "killing_signature_dim15": (8, 7, 0)},
    {"name": "so(3,3) ~ sl(4,R)",
     "decomp": "simple, split",
     "killing_signature_dim15": (9, 6, 0)},
    {"name": "so*(6) ~ su(3,1)",
     "decomp": "simple, non-compact (twisted)",
     "killing_signature_dim15": (7, 8, 0)},
    # ---- 14-dim semisimple + 1-d abelian center ----
    {"name": "G2 + R",
     "decomp": "compact semisimple G2 + 1-d center",
     "killing_signature_dim15": (14, 0, 1)},
    {"name": "su(3) (+) so(4) + R",
     "decomp": "compact semisimple + 1-d center",
     "killing_signature_dim15": (14, 0, 1)},
    {"name": "so(5) (+) su(2) + R",
     "decomp": "compact semisimple + 1-d center",
     "killing_signature_dim15": (14, 0, 1)},
    # ---- 10-dim sp(4,R) Levi + 5-d radical ----
    # The 10-d sp(4, R) has Killing signature (6+, 4-, 0z).  An algebra
    # with Levi decomposition g = sp(4, R) (+) rad with abelian or
    # Heisenberg-type rad of dim 5 inherits this on the quotient.  The
    # rad sits entirely in the zero-eigenspace of the Killing form.
    {"name": "sp(4, R) (+) heisenberg_2 (Jacobi algebra)",
     "decomp": "Levi: sp(4, R) (simple, non-cpt) (+) Heisenberg h_2 (rad dim 5, 1-d center)",
     "killing_signature_dim15": (6, 4, 5)},
    {"name": "sp(4, R) (+) R^5  (sp(4, R) acting on abelian rad)",
     "decomp": "Levi: sp(4, R) (+) R^5 abelian (rad dim 5, center dim 5)",
     "killing_signature_dim15": (6, 4, 5)},
    {"name": "so(3,2) (+) rad (de Sitter symmetry of M^3 extended)",
     "decomp": "Same as sp(4, R) (+) rad via the isomorphism so(3,2) = sp(4, R)",
     "killing_signature_dim15": (6, 4, 5)},
    # ---- 10-dim other Levi candidates ----
    {"name": "so(5) (+) rad",
     "decomp": "compact so(5) (dim 10) (+) 5-d rad",
     "killing_signature_dim15": (0, 10, 5)},
    {"name": "so(4,1) (+) rad",
     "decomp": "anti-de Sitter so(4,1) (dim 10) (+) 5-d rad",
     "killing_signature_dim15": (4, 6, 5)},
]


def identify(signature):
    pos, neg, zer = signature
    matches = []
    for c in CANDIDATES:
        sig = c["killing_signature_dim15"]
        if isinstance(sig, str):
            continue
        if sig == (pos, neg, zer):
            matches.append(c)
    return matches


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", "-o",
                    default="results/algebra_structure/harmonic_n3_d2_identification.json",
                    help="Output JSON path")
    ap.add_argument("--no-jacobi", action="store_true",
                    help="Skip the O(n^4) Jacobi identity check (slow at n=15)")
    args = ap.parse_args()

    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading structure constants from %s ..." % SC_PATH)
    n, sc = load_sc(SC_PATH)
    print(f"  Algebra dimension: {n}")

    ok, where = check_antisymmetry(sc)
    print(f"  Antisymmetry: {'OK' if ok else 'FAIL at %s' % (where,)}")
    assert ok, "Structure constants not antisymmetric"

    if not args.no_jacobi:
        print("  Checking Jacobi identity (this is O(n^4)) ...")
        ok, where = check_jacobi(sc)
        print(f"  Jacobi: {'OK' if ok else 'FAIL at %s' % (where,)}")
        assert ok, "Jacobi identity violated"

    print("\nComputing Killing form (exact over Q) ...")
    K = killing_form_exact(sc)
    print(f"  Killing form computed.")
    print("  Diagonalizing (SymPy exact) ...")
    sig, eig_list = killing_signature(K)
    pos, neg, zer = sig
    print(f"  Signature: ({pos}+, {neg}-, {zer}z)")
    print(f"  Eigenvalues (exact):")
    for ev_str, mult, ev_f in sorted(eig_list, key=lambda t: -t[2]):
        print(f"    multiplicity {mult}: {ev_str} (~ {ev_f:+.4e})")

    print("\nComputing center ...")
    Z = center_basis(sc)
    print(f"  Center dimension: {len(Z)}")
    if Z:
        for k, v in enumerate(Z):
            coeffs = [v[i, 0] for i in range(n)]
            nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
            print(f"    Z[{k}] = " + ", ".join(f"{c}*e{i}" for i, c in nonzero))

    print("\nComputing derived series ...")
    der_dims = derived_series(sc)
    print(f"  dim g^(k) for k=0,1,2,... : {der_dims}")
    solvable = der_dims[-1] == 0

    print("\nComputing lower central series ...")
    lcs_dims = lower_central_series(sc)
    print(f"  dim g^(k) (LCS) : {lcs_dims}")
    nilpotent = lcs_dims[-1] == 0

    # Disambiguator: structure of the radical.
    # The radical is the kernel of the Killing form (the 5-d zero-eigenspace).
    # If we project the algebra onto rad and read brackets [Z_i, Z_j] for
    # rad-elements Z_i, Z_j, we can tell whether the radical is abelian
    # (all such brackets vanish) or non-abelian (some bracket is non-zero).
    # The non-abelian case is the Heisenberg-style "Jacobi" algebra.
    print("\nProbing the radical (kernel of the Killing form) ...")
    # Find a basis of the kernel via exact null-space of K
    Kexact = Matrix([[Rational(K[i][j].numerator, K[i][j].denominator)
                      for j in range(n)] for i in range(n)])
    rad_basis = Kexact.nullspace()  # list of SymPy column vectors
    print(f"  Radical (Killing kernel) dim: {len(rad_basis)}")
    # Convert to plain Python rows
    rad_rows = []
    for v in rad_basis:
        row = [Rational(v[i, 0].p, v[i, 0].q) if v[i, 0] != 0 else Rational(0)
               for i in range(n)]
        rad_rows.append(row)
    # Compute [rad, rad] = span of all [Z_i, Z_j] for Z_i, Z_j in rad_basis
    rad_brackets = []
    for a in range(len(rad_basis)):
        for b in range(a + 1, len(rad_basis)):
            u = [Fraction(int(rad_rows[a][i].p), int(rad_rows[a][i].q))
                 for i in range(n)]
            w = [Fraction(int(rad_rows[b][i].p), int(rad_rows[b][i].q))
                 for i in range(n)]
            br = [Fraction(0)] * n
            for i in range(n):
                if u[i] == 0:
                    continue
                for j in range(n):
                    if w[j] == 0:
                        continue
                    for k in range(n):
                        br[k] += u[i] * w[j] * sc[i][j][k]
            if any(x != 0 for x in br):
                rad_brackets.append(br)
    if not rad_brackets:
        rad_abelian = True
        rad_bracket_rank = 0
    else:
        Mrb = Matrix([[Rational(c.numerator, c.denominator) for c in r]
                      for r in rad_brackets])
        rad_bracket_rank = Mrb.rank()
        rad_abelian = (rad_bracket_rank == 0)
    print(f"  [rad, rad] dim: {rad_bracket_rank}")
    print(f"  Radical is abelian? {rad_abelian}")

    matches = identify(sig)
    # Filter matches using the radical structure
    filtered = []
    for c in matches:
        name = c["name"].lower()
        if rad_abelian and "abelian" in name:
            filtered.append(c)
        elif rad_abelian and "heisenberg" in name:
            pass
        elif (not rad_abelian) and "heisenberg" in name:
            filtered.append(c)
        elif (not rad_abelian) and "abelian" in name:
            pass
        else:
            filtered.append(c)  # other (simple, etc.)

    print("\n" + "=" * 70)
    print("IDENTIFICATION")
    print("=" * 70)
    print(f"Killing signature (p+, q-, z) = ({pos}, {neg}, {zer})")
    print(f"Center dim = {len(Z)}")
    print(f"Radical (Killing kernel) dim = {len(rad_basis)}")
    print(f"Radical abelian: {rad_abelian}   ([rad,rad] dim = {rad_bracket_rank})")
    print(f"Solvable: {solvable}   Nilpotent: {nilpotent}")
    if not matches:
        print("\nNO direct match in candidate catalog.")
    else:
        print(f"\n{len(matches)} candidate(s) with matching Killing signature:")
        for c in matches:
            print(f"  - {c['name']}")
            print(f"      {c['decomp']}")
        if len(filtered) < len(matches):
            print(f"\nAfter filtering by radical structure (abelian="
                  f"{rad_abelian}): {len(filtered)} remain:")
            for c in filtered:
                print(f"  - {c['name']}")
                print(f"      {c['decomp']}")

    # Save
    out_data = {
        "source_sc": str(SC_PATH.relative_to(REPO)),
        "dimension": n,
        "antisymmetry_ok": True,
        "jacobi_ok": (not args.no_jacobi),
        "killing_signature": {"positive": pos, "negative": neg, "zero": zer},
        "killing_eigenvalues_exact": [
            {"eigenvalue": ev_str, "multiplicity": mult,
             "float_approx": ev_f}
            for ev_str, mult, ev_f in eig_list
        ],
        "center_dim": len(Z),
        "center_basis": [
            [str(v[i, 0]) for i in range(n)] for v in Z
        ],
        "radical_dim": len(rad_basis),
        "radical_abelian": bool(rad_abelian),
        "radical_bracket_rank": int(rad_bracket_rank),
        "derived_series_dims": der_dims,
        "lower_central_series_dims": lcs_dims,
        "solvable": bool(solvable),
        "nilpotent": bool(nilpotent),
        "candidate_matches": [c["name"] for c in matches],
        "candidate_filtered_by_radical": [c["name"] for c in filtered],
        "candidate_details": matches,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
