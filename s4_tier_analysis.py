#!/usr/bin/env python3
"""
S_4 Representation Decomposition for the N=4 Poisson Algebra.

Analogous to clebsch_gordan_analysis.py (S_3 for N=3), this script
decomposes the N=4 pairwise Poisson algebra under S_4 permutation
symmetry using Clebsch-Gordan rules.

N=4 observed dimension sequence: [6, 14, 62, 1260]
  Level 0:    6 generators (H_ij, C(4,2) = 6 pair Hamiltonians)
  Level 1:    8 new  (cum 14)
  Level 2:   48 new  (cum 62)
  Level 3: 1198 new  (cum 1260)

S_4 has 5 irreps:
  triv   (dim 1): trivial representation
  sign   (dim 1): sign/alternating representation
  std    (dim 3): standard representation
  sign_std (dim 3): sign x standard
  hook   (dim 2): the 2-dimensional irrep (partition [2,2])

Character table of S_4 (conjugacy classes: e, (12), (12)(34), (123), (1234)):
          e   (12)  (12)(34)  (123)  (1234)
  triv    1     1      1       1       1
  sign    1    -1      1       1      -1
  std     3     1     -1       0      -1
  sign_std 3   -1     -1       0       1
  hook    2     0      2      -1       0

Class sizes:       1     6      3       8       6
"""

import json
import os
import sys

# ==========================================================================
# S_4 Representation Theory
# ==========================================================================

# CG decomposition rules for S_4, derived from the character table.
# V x W decomposition: multiply characters pointwise, then decompose
# using inner product <chi_rho, chi> = (1/|G|) sum |C_i| chi_rho(C_i)* chi(C_i)

# Precomputed CG coefficients for S_4:
# Format: CG[i][j] = dict mapping irrep index -> multiplicity in V_i x V_j
# Irrep indices: 0=triv, 1=sign, 2=std, 3=sign_std, 4=hook

# Character table rows
IRREPS = ['triv', 'sign', 'std', 'sign_std', 'hook']
IRREP_DIMS = [1, 1, 3, 3, 2]
CLASS_SIZES = [1, 6, 3, 8, 6]
CHAR_TABLE = [
    [1,  1,  1,  1,  1],   # triv
    [1, -1,  1,  1, -1],   # sign
    [3,  1, -1,  0, -1],   # std
    [3, -1, -1,  0,  1],   # sign_std
    [2,  0,  2, -1,  0],   # hook
]
ORDER = 24


def _inner_product(chi1, chi2):
    """Inner product of two class functions on S_4."""
    s = sum(CLASS_SIZES[c] * chi1[c] * chi2[c] for c in range(5))
    return s // ORDER


def _decompose(chi):
    """Decompose a class function into irrep multiplicities."""
    return [_inner_product(CHAR_TABLE[i], chi) for i in range(5)]


def _tensor_chars(chi1, chi2):
    """Pointwise product of two characters."""
    return [chi1[c] * chi2[c] for c in range(5)]


def _exterior2_char(chi):
    """Character of Lambda^2(V) from character of V.
    chi_{Lambda^2}(g) = (chi(g)^2 - chi(g^2)) / 2
    Need chi evaluated on g^2 for each conjugacy class.
    """
    # g^2 mapping for conjugacy classes of S_4:
    #   e -> e,  (12) -> e,  (12)(34) -> e,  (123) -> (123)^2=(132) ~ (123),
    #   (1234) -> (12)(34)
    # So chi(g^2) = [chi(e), chi(e), chi(e), chi((123)), chi((12)(34))]
    chi_g2 = [chi[0], chi[0], chi[0], chi[3], chi[2]]
    return [(chi[c]**2 - chi_g2[c]) // 2 for c in range(5)]


def _sym2_char(chi):
    """Character of Sym^2(V) from character of V.
    chi_{Sym^2}(g) = (chi(g)^2 + chi(g^2)) / 2
    """
    chi_g2 = [chi[0], chi[0], chi[0], chi[3], chi[2]]
    return [(chi[c]**2 + chi_g2[c]) // 2 for c in range(5)]


class S4Rep:
    """S_4 representation as multiplicities of the 5 irreps."""
    def __init__(self, n_triv, n_sign, n_std, n_sign_std, n_hook, label=""):
        self.mults = [n_triv, n_sign, n_std, n_sign_std, n_hook]
        self.label = label

    @property
    def dim(self):
        return sum(m * d for m, d in zip(self.mults, IRREP_DIMS))

    def character(self):
        """Full character as a class function."""
        return [sum(self.mults[i] * CHAR_TABLE[i][c] for i in range(5))
                for c in range(5)]

    def tensor(self, other):
        chi = _tensor_chars(self.character(), other.character())
        m = _decompose(chi)
        return S4Rep(*m)

    def exterior2(self):
        chi = _exterior2_char(self.character())
        m = _decompose(chi)
        return S4Rep(*m)

    def sym2(self):
        chi = _sym2_char(self.character())
        m = _decompose(chi)
        return S4Rep(*m)

    def __add__(self, other):
        return S4Rep(*[a + b for a, b in zip(self.mults, other.mults)])

    def __repr__(self):
        parts = []
        for i, name in enumerate(IRREPS):
            if self.mults[i]:
                parts.append(f"{self.mults[i]}{name}")
            
        label = f" [{self.label}]" if self.label else ""
        return f"{' + '.join(parts) or '0'} (dim {self.dim}){label}"

    def table_row(self):
        return self.mults + [self.dim]


# ==========================================================================
# Edge representation of S_4
# ==========================================================================
# The 6 pair Hamiltonians H_{ij} (edges of K_4) form a representation of S_4.
# S_4 acts on {1,2,3,4}; it permutes the edges {ij}.
# The edge representation on K_4 has character:
#   e: 6 edges fixed -> chi = 6
#   (12): edges {13}↔{23}, {14}↔{24}, {12} fixed, {34} fixed -> chi = 2
#   (12)(34): {12}↔{12} no, {12}→{21}={12} fixed, {34}→{43}={34} fixed,
#             {13}↔{24}, {14}↔{23} -> chi = 2
#   (123): {12}→{23}→{13}→{12} cycle, {14}→{24}→{34}→{14} cycle -> chi = 0
#   (1234): {12}→{23}→{34}→{14}→{12} cycle of 4, {13}↔{24} -> chi = 0

EDGE_CHAR = [6, 2, 2, 0, 0]
EDGE_DECOMP = _decompose(EDGE_CHAR)

# ==========================================================================
# Main analysis
# ==========================================================================

def main():
    print("=" * 70)
    print("S_4 REPRESENTATION DECOMPOSITION OF THE N=4 POISSON ALGEBRA")
    print("=" * 70)

    # Verify edge representation
    print(f"\n--- Level 0: 6 pair Hamiltonians H_ij (edges of K_4) ---")
    L0 = S4Rep(*EDGE_DECOMP, label="level 0")
    print(f"  Edge character: {EDGE_CHAR}")
    print(f"  Decomposition: {L0}")
    assert L0.dim == 6, f"Edge rep should be dim 6, got {L0.dim}"

    # Level 1: brackets {H_ij, H_kl} for all pairs of edges
    # The Poisson bracket is S_4-equivariant, so {L0, L0} transforms as
    # Lambda^2(L0) (antisymmetric part of L0 x L0, since {f,g} = -{g,f})
    print(f"\n--- Level 1: {{L0, L0}} = Lambda^2(L0) ---")
    L1_candidates = L0.exterior2()
    L1_candidates.label = "L1 candidates"
    print(f"  Lambda^2(edge rep): {L1_candidates}")
    print(f"  Candidate count: {L1_candidates.dim} (observed: C(6,2)=15)")

    # The actual dimension at level 1 is 14 (cumulative) - 6 (level 0) = 8 new
    # So 15 candidates, 8 independent (7 vanish or are dependent)
    # The cumulative L0+L1 has dim 14
    print(f"  Observed new generators at L1: 8 (of 15 candidates)")
    print(f"  Cumulative through L1: 14")

    # Level 2: {L1, L0} + Lambda^2(L1)
    print(f"\n--- Level 2 candidates ---")
    L2_cross = L1_candidates.tensor(L0)
    L2_cross.label = "{L1, L0}"
    L2_self = L1_candidates.exterior2()
    L2_self.label = "Lambda^2(L1)"
    L2 = L2_cross + L2_self
    L2.label = "L2 candidates"
    print(f"  {{L1, L0}} = L1 x L0: {L2_cross}")
    print(f"  Lambda^2(L1): {L2_self}")
    print(f"  Total L2 candidates: {L2}")
    print(f"  Observed new generators at L2: 48 (of {L2.dim} candidates)")
    print(f"  Cumulative through L2: 62")

    # Level 3: {L2, L0} + {L2, L1} + Lambda^2(L2)
    print(f"\n--- Level 3 candidates ---")
    L3_20 = L2.tensor(L0)
    L3_20.label = "{L2, L0}"
    L3_21 = L2.tensor(L1_candidates)
    L3_21.label = "{L2, L1}"
    L3_22 = L2.exterior2()
    L3_22.label = "Lambda^2(L2)"
    L3 = L3_20 + L3_21 + L3_22
    L3.label = "L3 candidates"
    print(f"  {{L2, L0}}: {L3_20}")
    print(f"  {{L2, L1}}: {L3_21}")
    print(f"  Lambda^2(L2): {L3_22}")
    print(f"  Total L3 candidates: {L3}")

    F_total = L0 + L1_candidates + L2 + L3
    F_total.label = "full algebra candidates"
    print(f"\n  FULL ALGEBRA CANDIDATES: {F_total}")

    # ==========================================================================
    # Comparison with observed dimensions
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON WITH OBSERVED RANK SEQUENCE [6, 14, 62, 1260]")
    print("=" * 70)

    observed = {
        0: {"cum": 6, "new": 6},
        1: {"cum": 14, "new": 8},
        2: {"cum": 62, "new": 48},
        3: {"cum": 1260, "new": 1198},
    }

    candidates = {
        0: L0,
        1: L1_candidates,
        2: L2,
        3: L3,
    }

    print(f"\n  {'Level':>6s} {'Candidates':>12s} {'Observed new':>14s} "
          f"{'Observed cum':>14s} {'Syzygies':>10s}")
    cum_cand = 0
    for lv in range(4):
        cand = candidates[lv].dim
        cum_cand += cand
        obs = observed[lv]
        syz = cum_cand - obs["cum"]
        print(f"  {'L' + str(lv):>6s} {cand:>12d} {obs['new']:>14d} "
              f"{obs['cum']:>14d} {syz:>10d}")

    print(f"\n  Total candidates: {F_total.dim}")
    print(f"  Total independent: 1260")
    print(f"  Total syzygies: {F_total.dim - 1260}")

    # ==========================================================================
    # S_4 isotypic content of observed algebra
    # ==========================================================================
    print("\n" + "=" * 70)
    print("S_4 ISOTYPIC DECOMPOSITION BY BRACKET LEVEL")
    print("=" * 70)

    print(f"\n  {'Level':>8s} {'dim':>5s} {'triv':>6s} {'sign':>6s} "
          f"{'std':>6s} {'s-std':>6s} {'hook':>6s}")
    levels = [("L0", L0), ("L1", L1_candidates), ("L2", L2), ("L3", L3)]
    for label, rep in levels:
        m = rep.mults
        print(f"  {label:>8s} {rep.dim:>5d} {m[0]:>6d} {m[1]:>6d} "
              f"{m[2]:>6d} {m[3]:>6d} {m[4]:>6d}")
    m = F_total.mults
    print(f"  {'Total':>8s} {F_total.dim:>5d} {m[0]:>6d} {m[1]:>6d} "
          f"{m[2]:>6d} {m[3]:>6d} {m[4]:>6d}")

    print(f"\n  Generator contributions from each irrep type:")
    for i, name in enumerate(IRREPS):
        contrib = F_total.mults[i] * IRREP_DIMS[i]
        print(f"    {name:>8s}: {F_total.mults[i]:>4d} copies x dim {IRREP_DIMS[i]} "
              f"= {contrib:>5d} generators")

    # ==========================================================================
    # Predictions for tier structure
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TIER STRUCTURE PREDICTIONS")
    print("=" * 70)

    # For N=3: n_E = 52 = Tier 1 size. Each E-doublet contributes
    # one dynamically dominant direction.
    # For N=4: analogously, the irrep that "senses geometry" most should
    # dominate Tier 1.

    # The standard representation (dim 3) acts on the "relative positions"
    # of 4 bodies. By analogy with N=3 where E (dim 2, the standard of S_3)
    # dominated, we predict std (dim 3, the standard of S_4) dominates.

    n_std = F_total.mults[2]
    n_sign_std = F_total.mults[3]
    n_hook = F_total.mults[4]
    n_triv = F_total.mults[0]
    n_sign = F_total.mults[1]

    print(f"""
  Analogy with N=3 (S_3):
    S_3: n_E = 52 = Tier 1 size exactly
    S_3: n_A + n_Ap = 24 + 28 = 52 = remaining tiers (16 + 4 = 20 observed)

  S_4 candidate isotypic content:
    n_triv     = {n_triv:>5d}  (geometry-blind, lowest dynamical significance)
    n_sign     = {n_sign:>5d}  (parity-sensitive)
    n_std      = {n_std:>5d}  (standard: senses relative geometry)
    n_sign_std = {n_sign_std:>5d}  (sign x standard)
    n_hook     = {n_hook:>5d}  (hook: partition [2,2])

  Predictions (by analogy with S_3 tier structure):
    If std dominates Tier 1: expected Tier 1 size ~ n_std = {n_std}
    Each std copy contributes 3 generators, giving {3 * n_std} total std generators
    Fraction of algebra in std: {3 * n_std}/{F_total.dim} = {3*n_std/F_total.dim:.1%}
""")

    # E-fraction analysis (generalization of the 2/3 rule from N=3)
    print("  Irrep fraction analysis (generators / total candidates):")
    for i, name in enumerate(IRREPS):
        contrib = F_total.mults[i] * IRREP_DIMS[i]
        frac = contrib / F_total.dim if F_total.dim > 0 else 0
        print(f"    {name:>8s}: {contrib:>5d} / {F_total.dim} = {frac:.1%}")

    # ==========================================================================
    # Save results for dataset builder
    # ==========================================================================
    results_dir = os.path.join("results", "tier_decomposition")
    os.makedirs(results_dir, exist_ok=True)

    # S3 decomposition (from known values)
    s3_data = {
        "N": 3,
        "symmetry_group": "S3",
        "irreps": ["A", "Ap", "E"],
        "irrep_dims": [1, 1, 2],
        "levels": {
            "L0": {"dim": 3, "mults": [1, 0, 1]},
            "L1": {"dim": 3, "mults": [1, 0, 1]},
            "L2": {"dim": 12, "mults": [2, 2, 4]},
            "L3": {"dim": 138, "mults": [20, 26, 46]},
        },
        "total": {"dim": 156, "mults": [24, 28, 52]},
        "observed_rank": [3, 6, 17, 116],
        "observed_new": [3, 3, 11, 99],
        "tier_structure": [52, 44, 16, 4],
        "key_result": "n_E = 52 = Tier 1 size exactly",
    }

    # S4 decomposition
    s4_data = {
        "N": 4,
        "symmetry_group": "S4",
        "irreps": IRREPS,
        "irrep_dims": IRREP_DIMS,
        "levels": {},
        "total": {"dim": F_total.dim, "mults": F_total.mults},
        "observed_rank": [6, 14, 62, 1260],
        "observed_new": [6, 8, 48, 1198],
        "tier_structure": None,
    }

    for label, rep in levels:
        s4_data["levels"][label] = {"dim": rep.dim, "mults": rep.mults}

    combined = {"S3": s3_data, "S4": s4_data}
    out_path = os.path.join(results_dir, "s3_s4_decomposition.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # ==========================================================================
    # Comparison of S_3 and S_4
    # ==========================================================================
    print("\n" + "=" * 70)
    print("S_3 vs S_4 COMPARISON")
    print("=" * 70)

    print(f"""
  Property                          S_3 (N=3)       S_4 (N=4)
  ----------------------------------------------------------------
  Group order                       6               24
  Number of irreps                  3               5
  Pair Hamiltonians (L0)            3               6
  Total candidates                  156             {F_total.dim}
  Observed rank (L3)                116             1260
  Total syzygies                    {156-116}              {F_total.dim - 1260}
  Syzygy fraction                   {(156-116)/156:.1%}           {(F_total.dim-1260)/F_total.dim:.1%}

  Dominant irrep (L0)               E (dim 2)       {IRREPS[EDGE_DECOMP.index(max(EDGE_DECOMP))]} (dim {IRREP_DIMS[EDGE_DECOMP.index(max(EDGE_DECOMP))]})
  Dominant irrep fraction (total)   2/3 = 66.7%     {max(F_total.mults[i]*IRREP_DIMS[i] for i in range(5))}/{F_total.dim} = {max(F_total.mults[i]*IRREP_DIMS[i] for i in range(5))/F_total.dim:.1%}
""")

    # ==========================================================================
    # Final summary
    # ==========================================================================
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
  1. EDGE REPRESENTATION:
     The 6 pair Hamiltonians of N=4 decompose under S_4 as:
     {L0}
     (Compare N=3: A + E = 1 + 2 = 3)

  2. CG PREDICTION:
     Total candidates through L3: {F_total.dim}
     Observed independent: 1260
     Syzygies: {F_total.dim - 1260}

  3. S_4 ISOTYPIC CONTENT OF CANDIDATES:
     triv:     {F_total.mults[0]:>5d} copies ({F_total.mults[0]*1:>5d} generators)
     sign:     {F_total.mults[1]:>5d} copies ({F_total.mults[1]*1:>5d} generators)
     std:      {F_total.mults[2]:>5d} copies ({F_total.mults[2]*3:>5d} generators)
     sign_std: {F_total.mults[3]:>5d} copies ({F_total.mults[3]*3:>5d} generators)
     hook:     {F_total.mults[4]:>5d} copies ({F_total.mults[4]*2:>5d} generators)

  4. TIER PREDICTION (by S_3 analogy):
     If the standard representation dominates Tier 1 as E does for S_3,
     then Tier 1 size for N=4 ~ n_std = {n_std}.
     This needs verification against N=4 SVD tier data (not yet computed).
""")


if __name__ == "__main__":
    main()
