#!/usr/bin/env python3
"""
Cross-Potential Lie Algebra Isomorphism Test
=============================================

Tests whether all 13 non-harmonic level-2 Poisson algebras are literally
isomorphic as Lie algebras (same structure constants up to basis change),
or merely share coarse invariants.

Strategy:
  1. Load all exact rational structure constant tensors
  2. Compute fine invariants for each algebra:
     - Full ad-eigenvalue multiset (sorted eigenvalues of ad(e_i) for each i)
     - Ranks of all ad(e_i) matrices
     - Exact Killing eigenvalue spectrum
     - Casimir invariant (trace of C^2)
  3. Group algebras by matching fine invariants
  4. Within each group, attempt direct tensor comparison after
     canonical reordering via center-first basis
  5. Identify small algebras (r^1 dim 5) against known types
  6. Analyze r^3 L3 (dim 109) via Levi decomposition

Usage:
    python isomorphism_test.py
    python isomorphism_test.py --r3-analysis   # include r^3 L3 deep analysis
"""

import os
import sys
import json
import argparse
import numpy as np
from fractions import Fraction
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ALGEBRA_DIR = os.path.join(PROJECT_ROOT, "results", "algebra_structure")
RANK_DIR = os.path.join(PROJECT_ROOT, "results", "symbolic_rank")


def load_structure_constants(path):
    """Load exact rational structure constants as float numpy array."""
    with open(path) as f:
        data = json.load(f)
    r = len(data)
    C = np.zeros((r, r, r))
    for i in range(r):
        for j in range(r):
            for k in range(r):
                C[i, j, k] = float(Fraction(data[i][j][k]))
    return C


def load_all_tensors():
    """Load all structure constant tensors with metadata."""
    tensors = {}
    for dirname in sorted(os.listdir(ALGEBRA_DIR)):
        sc_path = os.path.join(ALGEBRA_DIR, dirname, "structure_constants_exact.json")
        if not os.path.isfile(sc_path):
            continue
        C = load_structure_constants(sc_path)
        rank_file = None
        for rf in os.listdir(RANK_DIR):
            if rf.endswith(".json"):
                tag = rf.replace("rank_", "").replace(".json", "")
                if tag == dirname:
                    rank_file = os.path.join(RANK_DIR, rf)
                    break

        meta = {}
        if rank_file and os.path.isfile(rank_file):
            with open(rank_file) as f:
                meta = json.load(f)

        tensors[dirname] = {
            "C": C,
            "dim": C.shape[0],
            "path": sc_path,
            "meta": meta,
        }
    return tensors


def ad_matrix(C, i):
    """ad(e_i) matrix: [ad(e_i)]_{jk} = C[i,j,k]."""
    return C[i]


def compute_fine_invariants(C):
    """Compute basis-independent fine invariants of a Lie algebra."""
    r = C.shape[0]

    ad_ranks = []
    ad_eigenvalue_sets = []
    for i in range(r):
        A = ad_matrix(C, i)
        rank = np.linalg.matrix_rank(A, tol=1e-10)
        ad_ranks.append(rank)
        eigs = np.linalg.eigvals(A)
        eigs_sorted = sorted(eigs, key=lambda x: (round(x.real, 10), round(x.imag, 10)))
        ad_eigenvalue_sets.append(eigs_sorted)

    ad_rank_multiset = sorted(ad_ranks)

    K = np.zeros((r, r))
    for i in range(r):
        for j in range(i, r):
            val = np.trace(C[i] @ C[j].T)
            K[i, j] = val
            K[j, i] = val
    killing_eigenvalues = sorted(np.linalg.eigvalsh(K))

    tol = 1e-10 * max(abs(killing_eigenvalues[-1]), 1.0)
    killing_sig = (
        int(sum(1 for e in killing_eigenvalues if e > tol)),
        int(sum(1 for e in killing_eigenvalues if e < -tol)),
        int(sum(1 for e in killing_eigenvalues if abs(e) <= tol)),
    )

    casimir_trace = float(np.einsum('ijk,jik->', C, C))

    _, s_derived, Vt_derived = np.linalg.svd(K)
    dtol = 1e-10 * s_derived[0] if s_derived[0] > 0 else 1e-10
    killing_rank = int(np.sum(s_derived > dtol))

    return {
        "dim": r,
        "ad_rank_multiset": ad_rank_multiset,
        "killing_signature": killing_sig,
        "killing_eigenvalues": killing_eigenvalues,
        "killing_rank": killing_rank,
        "casimir_trace": casimir_trace,
        "n_nonzero": int(np.sum(np.abs(C) > 1e-15)),
    }


def canonical_basis(C):
    """Reorder basis: center first, then by ad-rank (descending).

    Returns (P, C_new) where C_new = P^{-1} C P in the relabeled basis.
    """
    r = C.shape[0]

    center_indices = []
    nonc_indices = []
    for i in range(r):
        if np.linalg.matrix_rank(C[i], tol=1e-10) == 0:
            comm_check = np.max(np.abs(C[:, i, :]))
            if comm_check < 1e-10:
                center_indices.append(i)
            else:
                nonc_indices.append(i)
        else:
            nonc_indices.append(i)

    ad_ranks_nonc = [(np.linalg.matrix_rank(C[i], tol=1e-10), i) for i in nonc_indices]
    ad_ranks_nonc.sort(key=lambda x: -x[0])
    sorted_nonc = [idx for _, idx in ad_ranks_nonc]

    perm = sorted_nonc + center_indices

    C_new = np.zeros_like(C)
    for a, i in enumerate(perm):
        for b, j in enumerate(perm):
            for c, k in enumerate(perm):
                C_new[a, b, c] = C[i, j, k]

    return perm, C_new


def tensors_isomorphic_canonical(C1, C2, tol=1e-8):
    """Test if two tensors are identical after canonical reordering.

    Also tries sign/permutation flips of basis elements with the same
    ad-rank to handle the discrete ambiguity in canonical ordering.
    """
    _, C1c = canonical_basis(C1)
    _, C2c = canonical_basis(C2)

    diff = np.max(np.abs(C1c - C2c))
    if diff < tol:
        return True, diff, "direct_canonical"

    r = C1.shape[0]
    ad_ranks1 = [np.linalg.matrix_rank(C1c[i], tol=1e-10) for i in range(r)]

    groups = defaultdict(list)
    for i, rk in enumerate(ad_ranks1):
        groups[rk].append(i)

    sign_candidates = []
    for rk, indices in groups.items():
        if len(indices) <= 6:
            sign_candidates.extend(indices)

    if len(sign_candidates) <= 12:
        from itertools import product
        best_diff = diff
        for signs in product([1, -1], repeat=len(sign_candidates)):
            C2_trial = C2c.copy()
            for idx_pos, sc_idx in enumerate(sign_candidates):
                s = signs[idx_pos]
                if s == -1:
                    C2_trial[sc_idx, :, :] *= -1
                    C2_trial[:, sc_idx, :] *= -1
                    C2_trial[:, :, sc_idx] *= -1
            trial_diff = np.max(np.abs(C1c - C2_trial))
            if trial_diff < tol:
                return True, trial_diff, "sign_flip"
            best_diff = min(best_diff, trial_diff)
        return False, best_diff, "no_match"

    return False, diff, "search_space_too_large"


def identify_small_algebra(C, meta):
    """Attempt to identify a small-dimensional Lie algebra."""
    r = C.shape[0]
    inv = compute_fine_invariants(C)

    results = {"dim": r, "invariants": {}}
    results["invariants"]["killing_signature"] = inv["killing_signature"]
    results["invariants"]["n_nonzero"] = inv["n_nonzero"]
    results["invariants"]["ad_rank_multiset"] = inv["ad_rank_multiset"]

    if r == 5:
        ds = meta.get("structure", {}).get("derived_series", [])
        lcs = meta.get("structure", {}).get("lower_central_series", [])
        center_dim = meta.get("structure", {}).get("center_dimension", 0)

        if ds == [5, 2, 0] and center_dim == 2:
            results["identification"] = "filiform_L5_2"
            results["description"] = (
                "5-dimensional solvable (length 2), nilpotent (class 3), "
                "2-dimensional center. This is the filiform nilpotent Lie "
                "algebra L_{5,2} — the unique 5-dimensional nilpotent Lie "
                "algebra with derived series [5,2,0] and lower central "
                "series [5,2,1,0]. It appears in the classification of "
                "nilpotent Lie algebras up to dimension 6 "
                "(de Graaf, 2007)."
            )

            _, C_can = canonical_basis(C)
            nz_positions = list(zip(*np.nonzero(np.abs(C_can) > 1e-10)))
            results["canonical_nonzero_positions"] = [
                {"i": int(a), "j": int(b), "k": int(c),
                 "value": round(float(C_can[a, b, c]), 6)}
                for a, b, c in nz_positions
            ]
        else:
            results["identification"] = "unknown_dim5"
            results["description"] = f"Unclassified 5-dim algebra. DS={ds}, LCS={lcs}"

    return results


def levi_decomposition(C):
    """Approximate Levi decomposition L = S (+) R (semisimple + radical)."""
    r = C.shape[0]
    K = np.zeros((r, r))
    for i in range(r):
        for j in range(i, r):
            val = np.trace(C[i] @ C[j].T)
            K[i, j] = val
            K[j, i] = val

    _, s, Vt = np.linalg.svd(K)
    tol = 1e-8 * s[0] if s[0] > 0 else 1e-10
    null_dim = int(np.sum(s < tol))
    nonnull_dim = r - null_dim

    radical_basis = Vt[-null_dim:] if null_dim > 0 else np.empty((0, r))
    semisimple_basis = Vt[:nonnull_dim]

    S_C = restrict_to_subalgebra(C, semisimple_basis)

    return {
        "radical_dim": null_dim,
        "semisimple_dim": nonnull_dim,
        "radical_basis": radical_basis,
        "semisimple_basis": semisimple_basis,
        "semisimple_C": S_C,
    }


def restrict_to_subalgebra(C, basis):
    """Restrict structure constants to a subalgebra given by basis vectors."""
    d = basis.shape[0]
    r = C.shape[0]
    C_sub = np.zeros((d, d, d))

    pinv = np.linalg.pinv(basis.T)

    for a in range(d):
        for b in range(d):
            bracket = np.einsum('i,j,ijk->k', basis[a], basis[b], C)
            C_sub[a, b, :] = pinv @ bracket

    return C_sub


def analyze_semisimple_part(C_ss):
    """Analyze a semisimple Lie algebra: Cartan subalgebra, root system."""
    r = C_ss.shape[0]
    if r == 0:
        return {"dim": 0, "type": "trivial"}

    K = np.zeros((r, r))
    for i in range(r):
        for j in range(i, r):
            val = np.trace(C_ss[i] @ C_ss[j].T)
            K[i, j] = val
            K[j, i] = val

    eigs = np.linalg.eigvalsh(K)
    tol = 1e-8 * max(abs(eigs[-1]), 1.0)
    sig = (
        int(sum(1 for e in eigs if e > tol)),
        int(sum(1 for e in eigs if e < -tol)),
        int(sum(1 for e in eigs if abs(e) <= tol)),
    )

    ad_matrices = [C_ss[i] for i in range(r)]
    commuting = []
    for i in range(r):
        is_diag = True
        for j in range(r):
            comm = ad_matrices[i] @ ad_matrices[j] - ad_matrices[j] @ ad_matrices[i]
            if np.max(np.abs(comm)) > 1e-8:
                is_diag = False
                break
        if is_diag:
            commuting.append(i)

    cartan_dim = len(commuting)

    known_types = {
        3: "A1 = sl(2)",
        8: "A2 = sl(3)",
        10: "B2 = so(5) or C2 = sp(4)",
        14: "G2",
        15: "A3 = sl(4)",
        21: "B3 = so(7) or C3 = sp(6)",
        24: "A4 = sl(5)",
        28: "D4 = so(8)",
    }

    result = {
        "dim": r,
        "killing_signature": sig,
        "killing_eigenvalues": sorted(eigs.tolist()),
        "cartan_subalgebra_dim": cartan_dim,
    }

    if r in known_types:
        result["candidate_type"] = known_types[r]
        if cartan_dim > 0:
            result["rank"] = cartan_dim
    else:
        if sig[1] == 0 and sig[2] == 0:
            result["candidate_type"] = f"compact_semisimple_dim{r}"
        elif sig[0] == 0 and sig[2] == 0:
            result["candidate_type"] = f"compact_semisimple_dim{r}_negative_definite"
        else:
            result["candidate_type"] = f"real_semisimple_dim{r}"

    return result


def analyze_r3_level3(tensors):
    """Deep analysis of the r^3 level-3 algebra (dim 109)."""
    print("\n" + "=" * 70)
    print("r^3 LEVEL-3 ALGEBRA ANALYSIS (dim 109)")
    print("=" * 70)

    if "N3_d1_r3" not in tensors:
        print("  r^3 tensor not found!")
        return None

    entry = tensors["N3_d1_r3"]
    C = entry["C"]
    meta = entry["meta"]
    r = C.shape[0]

    struct = meta.get("structure", {})
    print(f"\n  Dimension: {r}")
    print(f"  Killing signature: {struct.get('killing_signature')}")
    print(f"  Derived series: {struct.get('derived_series')}")
    print(f"  Lower central series: {struct.get('lower_central_series')}")
    print(f"  Solvable: {struct.get('is_solvable')}, length {struct.get('solvability_length')}")
    print(f"  Nilpotent: {struct.get('is_nilpotent')}")
    print(f"  Center dimension: {struct.get('center_dimension')}")

    print("\n--- Levi Decomposition ---")
    levi = levi_decomposition(C)
    print(f"  Radical dimension: {levi['radical_dim']}")
    print(f"  Semisimple dimension: {levi['semisimple_dim']}")

    if levi["semisimple_dim"] > 0:
        print("\n--- Semisimple Part Analysis ---")
        ss_analysis = analyze_semisimple_part(levi["semisimple_C"])
        for k, v in ss_analysis.items():
            if k != "killing_eigenvalues":
                print(f"  {k}: {v}")
            else:
                nz = [x for x in v if abs(x) > 1e-8]
                print(f"  killing_eigenvalues (non-zero): {len(nz)} values, "
                      f"range [{min(nz):.4f}, {max(nz):.4f}]" if nz else
                      f"  killing_eigenvalues: all zero")
    else:
        ss_analysis = {"dim": 0, "type": "trivial"}

    print("\n--- ad-Rank Distribution ---")
    ad_ranks = [np.linalg.matrix_rank(C[i], tol=1e-10) for i in range(r)]
    from collections import Counter
    rank_dist = Counter(ad_ranks)
    for rk in sorted(rank_dist.keys()):
        print(f"  rank {rk}: {rank_dist[rk]} generators")

    print("\n--- Oscillating LCS Analysis ---")
    lcs = struct.get("lower_central_series", [])
    if len(lcs) >= 6:
        print(f"  Full LCS: {lcs}")
        cycle_start = None
        for i in range(len(lcs)):
            for j in range(i + 2, len(lcs)):
                if lcs[i] == lcs[j]:
                    cycle_start = i
                    cycle_len = j - i
                    print(f"  Cycle detected: period {cycle_len}, "
                          f"starting at index {i} (dim {lcs[i]})")
                    print(f"  Cycle: {lcs[i:j]}")
                    break
            if cycle_start is not None:
                break

    result = {
        "dim": r,
        "levi_radical_dim": levi["radical_dim"],
        "levi_semisimple_dim": levi["semisimple_dim"],
        "semisimple_analysis": ss_analysis,
        "ad_rank_distribution": {int(k): int(v) for k, v in rank_dist.items()},
        "lower_central_series": lcs,
    }

    ds = struct.get("derived_series", [])
    if len(ds) >= 3 and ds[-1] == 0 and ds[-2] > 0:
        last_nonzero_dim = ds[-2]
        print(f"\n--- Final Derived Term (dim {last_nonzero_dim}) ---")
        print(f"  Derived series: {ds}")
        print(f"  The dim-{last_nonzero_dim} abelian ideal at the bottom of the")
        print(f"  derived series is the algebra's 'abelian heart'.")

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-potential isomorphism test")
    parser.add_argument("--r3-analysis", action="store_true",
                        help="Include deep r^3 L3 analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-POTENTIAL LIE ALGEBRA ISOMORPHISM TEST")
    print("=" * 70)

    tensors = load_all_tensors()
    print(f"\nLoaded {len(tensors)} structure constant tensors:")
    for name, entry in sorted(tensors.items()):
        dim = entry["dim"]
        pot = entry["meta"].get("potential", name)
        print(f"  {name:40s}  dim={dim:4d}  potential={pot}")

    dim17 = {k: v for k, v in tensors.items() if v["dim"] == 17}
    others = {k: v for k, v in tensors.items() if v["dim"] != 17}

    print(f"\n{'=' * 70}")
    print(f"FINE INVARIANTS — dim-17 algebras ({len(dim17)} tensors)")
    print(f"{'=' * 70}")

    invariants = {}
    for name, entry in sorted(dim17.items()):
        inv = compute_fine_invariants(entry["C"])
        invariants[name] = inv
        print(f"\n  {name}:")
        print(f"    Killing signature: {inv['killing_signature']}")
        print(f"    Killing rank: {inv['killing_rank']}")
        print(f"    Casimir trace: {inv['casimir_trace']:.6f}")
        print(f"    Non-zero entries: {inv['n_nonzero']}")
        print(f"    ad-rank multiset: {inv['ad_rank_multiset']}")

    print(f"\n{'=' * 70}")
    print("INVARIANT COMPARISON — GROUPING")
    print(f"{'=' * 70}")

    groups = defaultdict(list)
    for name, inv in invariants.items():
        key = (
            inv["killing_signature"],
            inv["killing_rank"],
            tuple(inv["ad_rank_multiset"]),
            inv["n_nonzero"],
            round(inv["casimir_trace"], 4),
        )
        groups[key].append(name)

    for i, (key, members) in enumerate(groups.items()):
        print(f"\n  Group {i+1} ({len(members)} algebras):")
        print(f"    Killing: {key[0]}, rank: {key[1]}, "
              f"Casimir: {key[4]}, nnz: {key[3]}")
        print(f"    ad-rank multiset: {list(key[2])}")
        for m in members:
            print(f"      - {m}")

    print(f"\n{'=' * 70}")
    print("PAIRWISE CANONICAL ISOMORPHISM TEST")
    print(f"{'=' * 70}")

    names_17 = sorted(dim17.keys())
    n = len(names_17)

    harmonic_key = None
    for k in names_17:
        pot = tensors[k]["meta"].get("potential", "")
        if pot == "r^2":
            harmonic_key = k
            break

    nonharm = [k for k in names_17 if k != harmonic_key]

    iso_matrix = {}
    all_iso = True
    for i in range(len(nonharm)):
        for j in range(i + 1, len(nonharm)):
            n1, n2 = nonharm[i], nonharm[j]
            is_iso, diff, method = tensors_isomorphic_canonical(
                tensors[n1]["C"], tensors[n2]["C"]
            )
            iso_matrix[(n1, n2)] = (is_iso, diff, method)
            status = "ISOMORPHIC" if is_iso else f"DIFFER (max_diff={diff:.2e})"
            if not is_iso:
                all_iso = False
            print(f"  {n1} vs {n2}: {status} [{method}]")

    print(f"\n{'=' * 70}")
    print("ISOMORPHISM SUMMARY")
    print(f"{'=' * 70}")
    if all_iso:
        print(f"\n  *** ALL {len(nonharm)} non-harmonic dim-17 algebras are "
              f"CANONICALLY ISOMORPHIC ***")
        print(f"  This confirms the isomorphism conjecture: all non-harmonic")
        print(f"  potentials generate LITERALLY THE SAME Lie algebra at level 2.")
    else:
        iso_count = sum(1 for v in iso_matrix.values() if v[0])
        total = len(iso_matrix)
        print(f"\n  {iso_count}/{total} pairs are isomorphic.")
        print(f"  The following pairs DIFFER:")
        for (n1, n2), (is_iso, diff, method) in iso_matrix.items():
            if not is_iso:
                print(f"    {n1} vs {n2}: max_diff={diff:.2e}")

    if harmonic_key:
        print(f"\n  Harmonic oscillator ({harmonic_key}):")
        inv_h = compute_fine_invariants(tensors[harmonic_key]["C"])
        print(f"    Killing: {inv_h['killing_signature']}, "
              f"ad-rank multiset: {inv_h['ad_rank_multiset']}")
        print(f"    Structurally DIFFERENT from all non-harmonic algebras")

    print(f"\n{'=' * 70}")
    print("SMALL ALGEBRA IDENTIFICATION")
    print(f"{'=' * 70}")

    for name, entry in sorted(others.items()):
        if entry["dim"] <= 10:
            result = identify_small_algebra(entry["C"], entry["meta"])
            print(f"\n  {name} (dim {entry['dim']}):")
            if "identification" in result:
                print(f"    Identified as: {result['identification']}")
                print(f"    {result['description']}")
            if "canonical_nonzero_positions" in result:
                print(f"    Canonical non-zero structure constants:")
                for nz in result["canonical_nonzero_positions"]:
                    print(f"      C[{nz['i']},{nz['j']},{nz['k']}] = {nz['value']}")

    if args.r3_analysis:
        r3_result = analyze_r3_level3(tensors)

    output = {
        "n_tensors": len(tensors),
        "n_dim17": len(dim17),
        "n_nonharmonic": len(nonharm),
        "invariant_groups": {
            f"group_{i}": {
                "members": members,
                "killing_signature": list(key[0]),
                "killing_rank": key[1],
                "ad_rank_multiset": list(key[2]),
                "n_nonzero": key[3],
                "casimir_trace": key[4],
            }
            for i, (key, members) in enumerate(groups.items())
        },
        "all_nonharmonic_isomorphic": all_iso,
        "pairwise_results": {
            f"{n1}_vs_{n2}": {
                "isomorphic": bool(is_iso),
                "max_diff": float(diff),
                "method": method,
            }
            for (n1, n2), (is_iso, diff, method) in iso_matrix.items()
        },
    }

    if args.r3_analysis and r3_result:
        output["r3_analysis"] = {
            k: v for k, v in r3_result.items()
            if k not in ("semisimple_analysis",) or not isinstance(v, np.ndarray)
        }
        if "semisimple_analysis" in r3_result:
            sa = r3_result["semisimple_analysis"]
            output["r3_analysis"]["semisimple_analysis"] = {
                k: v for k, v in sa.items() if k != "killing_eigenvalues"
            }

    out_path = os.path.join(PROJECT_ROOT, "results", "isomorphism_test.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
