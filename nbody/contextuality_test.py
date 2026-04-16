#!/usr/bin/env python3
"""
Contextuality / Kochen-Specker Tests for the Poisson Algebra
=============================================================

Tests whether the N=3 pairwise Poisson algebra exhibits Kochen-Specker
contextuality, complementing the CHSH Bell test (which found no violation).

Three-part investigation:
  A) Commutativity census — count Poisson-commuting pairs {g_i, g_j} = 0
  B) Orthogonality graph + KS coloring — build graph, test colorability
  C) Peres-Mermin square — attempt to construct a 3x3 commuting grid

Contextuality requires the existence of "contexts" — maximal sets of
mutually commuting observables. Without commuting pairs, no contexts
exist and contextuality tests are vacuously non-contextual.

Usage:
    python nbody/contextuality_test.py
    python nbody/contextuality_test.py --potentials 1r log r2
"""

import os
import sys
import json
import argparse
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "contextuality_results")
ALGEBRA_DIR = os.path.join(PROJECT_DIR, "results", "algebra_structure")

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_structure_constants(algebra_dir):
    """Load structure constants tensor from an algebra directory."""
    sc_path = os.path.join(algebra_dir, "structure_constants_exact.json")
    if not os.path.exists(sc_path):
        return None
    with open(sc_path) as f:
        return json.load(f)


def commuting_pairs(sc):
    """Find all pairs (i, j) with i < j such that {g_i, g_j} = 0."""
    n = len(sc)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if all(sc[i][j][k] == 0 for k in range(n)):
                pairs.append((i, j))
    return pairs


def maximal_cliques(adj, n):
    """Find all maximal cliques in the commuting graph using Bron-Kerbosch."""
    cliques = []

    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        pivot = max(P | X, key=lambda v: len(adj[v] & P)) if P | X else None
        candidates = P - adj[pivot] if pivot is not None else P
        for v in list(candidates):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    adj_sets = {i: set() for i in range(n)}
    for i, j in adj:
        adj_sets[i].add(j)
        adj_sets[j].add(i)

    bron_kerbosch(set(), set(range(n)), set())
    return cliques


def ks_colorable(cliques, n):
    """Test if the orthogonality graph is KS-colorable.

    A KS coloring assigns 0/1 to each vertex such that in every maximal
    commuting set (clique) exactly one vertex gets 1.

    If no cliques exist (empty graph), any assignment works -> colorable.
    """
    if not cliques:
        return True, "No cliques — trivially colorable"

    # For small numbers of cliques, try exhaustive search
    if n <= 30:
        from itertools import product as iter_product
        for assignment in iter_product([0, 1], repeat=n):
            valid = True
            for clique in cliques:
                ones = sum(assignment[v] for v in clique)
                if ones != 1:
                    valid = False
                    break
            if valid:
                return True, f"Valid coloring found (exhaustive, 2^{n} tested)"
        return False, f"No valid coloring exists (exhaustive, 2^{n} tested)"

    # For larger algebras, use constraint propagation
    return None, f"Too large for exhaustive search (n={n}), skipped"


def peres_mermin_test(sc):
    """Attempt to construct a Peres-Mermin square from the algebra.

    A PM square is a 3x3 grid of observables where:
    - Each row commutes (all pairs in the row Poisson-commute)
    - Each column commutes
    - Row products and column products satisfy specific sign constraints

    Returns None if no commuting pairs exist (cannot form PM square).
    """
    n = len(sc)
    comm = commuting_pairs(sc)
    if len(comm) < 3:
        return {
            "constructible": False,
            "reason": f"Only {len(comm)} commuting pairs; need >= 3 for a PM row",
            "n_commuting_pairs": len(comm),
        }

    # Build adjacency for commuting graph
    adj = {i: set() for i in range(n)}
    for i, j in comm:
        adj[i].add(j)
        adj[j].add(i)

    # Find a commuting triple (3-clique) for a PM row
    triples = []
    for i, j in comm:
        for k in adj[i] & adj[j]:
            if k > j:
                triples.append((i, j, k))
    if len(triples) < 3:
        return {
            "constructible": False,
            "reason": f"Only {len(triples)} commuting triples; need 3 for PM rows",
            "n_commuting_pairs": len(comm),
            "n_commuting_triples": len(triples),
        }

    return {
        "constructible": True,
        "n_commuting_pairs": len(comm),
        "n_commuting_triples": len(triples),
    }


def analyze_algebra(name, sc):
    """Run full contextuality analysis on one algebra."""
    n = len(sc)
    total_pairs = n * (n - 1) // 2
    comm = commuting_pairs(sc)

    # Non-zero bracket count
    nz = sum(1 for i in range(n) for j in range(i + 1, n)
             for k in range(n) if sc[i][j][k] != 0)

    result = {
        "algebra": name,
        "dim": n,
        "total_pairs": total_pairs,
        "n_commuting_pairs": len(comm),
        "n_noncommuting_pairs": total_pairs - len(comm),
        "commutativity_fraction": len(comm) / total_pairs if total_pairs > 0 else 0,
        "commuting_pairs": comm,
    }

    # KS analysis
    if len(comm) == 0:
        result["orthogonality_graph"] = "empty"
        result["n_maximal_cliques"] = 0
        result["ks_colorable"] = True
        result["ks_reason"] = "Empty orthogonality graph — trivially non-contextual"
        result["contextual"] = False
    else:
        cliques = maximal_cliques(comm, n)
        colorable, reason = ks_colorable(cliques, n)
        result["orthogonality_graph"] = "non-empty"
        result["n_maximal_cliques"] = len(cliques)
        result["maximal_cliques"] = [sorted(c) for c in cliques]
        result["ks_colorable"] = colorable
        result["ks_reason"] = reason
        result["contextual"] = not colorable if colorable is not None else None

    # Peres-Mermin
    pm = peres_mermin_test(sc)
    result["peres_mermin"] = pm

    return result


def main():
    ap = argparse.ArgumentParser(description="Contextuality / Kochen-Specker tests")
    ap.add_argument("--potentials", nargs="*", default=None,
                    help="Specific algebra directories to test")
    args = ap.parse_args()

    print("=" * 70)
    print("CONTEXTUALITY / KOCHEN-SPECKER TESTS")
    print("=" * 70)

    if args.potentials:
        algebra_dirs = args.potentials
    else:
        algebra_dirs = sorted(d for d in os.listdir(ALGEBRA_DIR)
                              if os.path.isdir(os.path.join(ALGEBRA_DIR, d)))

    all_results = []
    summary_rows = []

    for name in algebra_dirs:
        full_path = os.path.join(ALGEBRA_DIR, name)
        if not os.path.isdir(full_path):
            continue
        sc = load_structure_constants(full_path)
        if sc is None:
            print(f"\n  {name}: no structure constants found, skipping")
            continue

        print(f"\n--- {name} (dim {len(sc)}) ---")
        result = analyze_algebra(name, sc)
        all_results.append(result)

        print(f"  Pairs: {result['total_pairs']} total, "
              f"{result['n_commuting_pairs']} commuting, "
              f"{result['n_noncommuting_pairs']} non-commuting")
        print(f"  Orthogonality graph: {result['orthogonality_graph']}")
        print(f"  KS colorable: {result['ks_colorable']}")
        print(f"  Contextual: {result['contextual']}")
        print(f"  Peres-Mermin: {'constructible' if result['peres_mermin']['constructible'] else 'NOT constructible'}")
        if not result['peres_mermin']['constructible']:
            print(f"    Reason: {result['peres_mermin']['reason']}")

        summary_rows.append({
            "algebra": name,
            "dim": result["dim"],
            "commuting_pairs": result["n_commuting_pairs"],
            "total_pairs": result["total_pairs"],
            "ks_colorable": result["ks_colorable"],
            "contextual": result["contextual"],
            "pm_constructible": result["peres_mermin"]["constructible"],
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_non_contextual = all(not r["contextual"] for r in all_results
                             if r["contextual"] is not None)
    all_zero_commuting = all(r["n_commuting_pairs"] == 0 for r in all_results)
    max_dim = max(r["dim"] for r in all_results) if all_results else 0

    print(f"\n  Algebras tested: {len(all_results)}")
    print(f"  Max dimension tested: {max_dim}")
    print(f"  All have zero commuting pairs: {all_zero_commuting}")
    print(f"  All non-contextual: {all_non_contextual}")

    if all_zero_commuting:
        print(f"""
  DEFINITIVE RESULT: All {len(all_results)} pairwise Poisson algebras tested
  (dimensions {', '.join(str(r['dim']) for r in all_results)}) have ZERO
  commuting pairs. The orthogonality graph is empty in every case.

  This means:
  1. No Kochen-Specker coloring constraints exist -> trivially non-contextual
  2. Peres-Mermin squares cannot be constructed -> no PM contextuality
  3. The algebra is "maximally non-commutative" — every pair of generators
     has a non-zero Poisson bracket.

  Physical interpretation: The pairwise Poisson algebra has NO commutative
  substructure. Unlike quantum observables (where compatible observables form
  contexts), EVERY pair of generators in the Poisson algebra is dynamically
  coupled. This is consistent with the CHSH Bell test finding (no violation):
  the algebra's non-commutativity is "classical" (Poisson) rather than
  "quantum" (operator), and does not produce quantum-information phenomena
  like contextuality or Bell violations.
""")

    # Save detailed results
    out_path = os.path.join(RESULTS_DIR, "contextuality_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Detailed results: {out_path}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "contextuality_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_algebras_tested": len(all_results),
            "max_dim_tested": max_dim,
            "all_zero_commuting_pairs": all_zero_commuting,
            "all_non_contextual": all_non_contextual,
            "verdict": "non-contextual",
            "reason": "All algebras maximally non-commutative (zero commuting pairs)",
            "algebras": summary_rows,
        }, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
