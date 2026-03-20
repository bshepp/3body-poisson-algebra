"""
Stability Atlas: Variable-Resolution Dimension Landscape
for the Pairwise Poisson Algebra of the Three-Body Problem

Brian Sheppard, February 2026

Maps the local SVD rank of the Poisson algebra generators across
configuration space at variable resolution.  Any structure that appears
-- rank drops, ridges, basins -- is evidence of non-trivial topology
in the dimension landscape.

Engine: uses exact symbolic Poisson brackets from exact_growth.py.
Generators are computed symbolically once and compiled to fast NumPy
evaluators; per-grid-point evaluation is pure numerical linear algebra.

The atlas operates in the reduced "shape space" of the three-body
problem: after removing centre-of-mass translation, rotation, and
scale, the space of triangle shapes is 2D, parameterised by
(mu, phi) = (side-length ratio r13/r12, angle at vertex 1).

USAGE:
    # Quick coarse scan
    python stability_atlas.py --resolution coarse --level 2

    # High resolution around Lagrange point
    python stability_atlas.py --resolution fine --level 3 --focus lagrange

    # Full adaptive scan
    python stability_atlas.py --resolution adaptive --level 3

    # Compare potentials
    python stability_atlas.py --potential 1/r --level 3
    python stability_atlas.py --potential 1/r2 --level 3
    python stability_atlas.py --potential harmonic --level 3
"""

import numpy as np
from numpy.linalg import svd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
import os
import sys
from pathlib import Path
from time import time

from sympy import Add

from exact_growth import (
    x1, y1, x2, y2, x3, y3,
    px1, py1, px2, py2, px3, py3,
    u12, u13, u23,
    Q_VARS, P_VARS, U_VARS, ALL_VARS,
    total_deriv, poisson_bracket, simplify_generator,
    lambdify_generators,
    T1, T2, T3,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AtlasConfig:
    """Configuration for a stability atlas computation."""
    
    # Masses
    masses: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    G: float = 1.0
    
    # Potential type: '1/r', '1/r2', 'harmonic'
    potential_type: str = '1/r'
    
    # Bracket level to compute (0, 1, 2, 3)
    max_level: int = 3
    
    # Resolution settings
    resolution: str = 'coarse'  # 'coarse', 'medium', 'fine', 'adaptive'
    
    # Grid sizes for each resolution
    grid_sizes: Dict[str, int] = field(default_factory=lambda: {
        'coarse': 20,
        'medium': 50,
        'fine': 100,
        'ultra': 200
    })
    
    # Number of phase-space samples per configuration point
    n_phase_samples: int = 200
    
    # Local ball radius for rank computation
    epsilon: float = 1e-2
    
    # SVD gap threshold
    svd_gap_threshold: float = 1e4
    
    # Adaptive refinement settings
    adaptive_refine_threshold: float = 0.1  # refine where rank changes by this fraction
    adaptive_max_depth: int = 4
    
    # Focus regions (named special configurations)
    focus: Optional[str] = None  # 'lagrange', 'euler', 'collision12', etc.
    
    # Charges for Coulomb potential (None = standard gravitational)
    charges: Optional[Tuple[float, float, float]] = None
    
    # Output
    output_dir: str = './atlas_output'
    save_raw: bool = True


# =============================================================================
# SHAPE SPACE PARAMETERIZATION
# =============================================================================

class ShapeSpace:
    """
    Parameterizes the reduced configuration space of three bodies.
    
    After removing center-of-mass (2D), rotation (1D), and scale (1D),
    the shape space is 2D. We use the parameterization:
    
        (mu, phi) where:
            mu  = r13 / r12  (ratio of two side lengths, in [0, inf))
            phi = angle at vertex 1 between sides r12 and r13, in [0, pi]
    
    This gives a half-plane that covers all triangle shapes exactly once
    (up to labeling symmetry).
    
    The special configurations have known coordinates:
        Lagrange equilateral:  mu = 1, phi = pi/3
        Euler collinear:       phi = 0 or phi = pi (degenerate)
        Isosceles:             mu = 1 (any phi)
        Near-collision 1-2:    mu -> inf (r13 >> r12) or scale r12 -> 0
        Near-collision 1-3:    mu -> 0
    """
    
    # Named special configurations
    SPECIAL_CONFIGS = {
        'lagrange': (1.0, np.pi / 3),
        'lagrange_obtuse': (1.0, 2 * np.pi / 3),
        'isosceles_right': (1.0, np.pi / 2),
        'euler_collinear': (0.5, np.pi),  # approximate — body 3 between 1 and 2
        'near_collision_12': (10.0, np.pi / 3),  # r13 >> r12
        'near_collision_13': (0.1, np.pi / 3),   # r13 << r12
        'symmetric_line': (1.0, np.pi),  # isoceles collinear
    }
    
    @staticmethod
    def shape_to_positions(mu: float, phi: float, scale: float = 1.0) -> np.ndarray:
        """
        Convert shape parameters to Cartesian positions.
        
        Places body 1 at origin, body 2 along x-axis at distance scale,
        body 3 at distance mu*scale from body 1 at angle phi.
        
        Returns: array of shape (3, 2) = [[x1,y1], [x2,y2], [x3,y3]]
        """
        positions = np.zeros((3, 2))
        positions[0] = [0.0, 0.0]              # body 1 at origin
        positions[1] = [scale, 0.0]             # body 2 on x-axis
        positions[2] = [mu * scale * np.cos(phi),  # body 3 at angle phi
                        mu * scale * np.sin(phi)]
        return positions
    
    @staticmethod
    def make_grid(resolution: str, grid_sizes: dict,
                  mu_range: Tuple[float, float] = (0.2, 3.0),
                  phi_range: Tuple[float, float] = (0.1, np.pi - 0.1)
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a grid over shape space."""
        n = grid_sizes.get(resolution, 50)
        mu_vals = np.linspace(mu_range[0], mu_range[1], n)
        phi_vals = np.linspace(phi_range[0], phi_range[1], n)
        return mu_vals, phi_vals
    
    @staticmethod
    def focus_region(name: str, width: float = 0.3
                     ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get a focused region around a special configuration."""
        if name not in ShapeSpace.SPECIAL_CONFIGS:
            raise ValueError(f"Unknown configuration: {name}. "
                           f"Known: {list(ShapeSpace.SPECIAL_CONFIGS.keys())}")
        mu0, phi0 = ShapeSpace.SPECIAL_CONFIGS[name]
        return ((max(0.1, mu0 - width), mu0 + width),
                (max(0.05, phi0 - width), min(np.pi - 0.05, phi0 + width)))


# =============================================================================
# POTENTIAL DEFINITIONS
# =============================================================================

class Potential:
    """
    Maps potential type strings to symbolic Hamiltonians and metadata.

    Uses the polynomial u_ij = 1/r_ij representation from exact_growth.py
    so all expressions stay polynomial and the existing chain-rule machinery
    handles differentiation exactly.
    """

    REGISTRY = {
        '1/r': {
            'name': 'Newtonian (1/r)',
            'integrable': False,
            'singular': True,
        },
        '1/r2': {
            'name': 'Calogero-Moser (1/r^2)',
            'integrable': '1D only',
            'singular': True,
        },
        'harmonic': {
            'name': 'Harmonic (r^2)',
            'integrable': True,
            'singular': False,
        },
        'coulomb': {
            'name': 'Coulomb (1/r with charges)',
            'integrable': False,
            'singular': True,
        },
    }

    @staticmethod
    def get_metadata(potential_type: str) -> dict:
        if potential_type.startswith('1/r^'):
            n = float(potential_type[4:])
            return {
                'name': f'1/r^{n}',
                'integrable': '1D only' if n == 2 else False,
                'singular': True,
            }
        if potential_type not in Potential.REGISTRY:
            raise ValueError(f"Unknown potential: {potential_type}. "
                             f"Known: {list(Potential.REGISTRY.keys())}")
        return Potential.REGISTRY[potential_type]

    @staticmethod
    def _potential_exponent(potential_type: str):
        """Return the exponent n for a 1/r^n potential, or None."""
        if potential_type == '1/r':
            return 1
        if potential_type == '1/r2':
            return 2
        if potential_type.startswith('1/r^'):
            return float(potential_type[4:])
        return None

    @staticmethod
    def get_symbolic_hamiltonians(potential_type: str, charges=None):
        """
        Return (H12, H13, H23) as SymPy expressions in the polynomial
        u_ij representation used by exact_growth.py.

        Accepts '1/r^n' for any real exponent n (e.g. '1/r^1.5').
        When charges=(q1,q2,q3) is provided with a 1/r^n potential,
        constructs H_ij = T_i + T_j + q_i*q_j*u_ij^n.
        """
        from sympy import Integer as _Int

        n = Potential._potential_exponent(potential_type)

        if charges is not None:
            if n is None:
                raise ValueError(
                    f"Charges not supported with potential '{potential_type}'")
            q1, q2, q3 = [_Int(c) if isinstance(c, int) else c
                          for c in charges]
            u12_n = u12 if n == 1 else u12**n
            u13_n = u13 if n == 1 else u13**n
            u23_n = u23 if n == 1 else u23**n
            return (T1 + T2 + q1*q2*u12_n,
                    T1 + T3 + q1*q3*u13_n,
                    T2 + T3 + q2*q3*u23_n)

        r12_sq = (x1 - x2)**2 + (y1 - y2)**2
        r13_sq = (x1 - x3)**2 + (y1 - y3)**2
        r23_sq = (x2 - x3)**2 + (y2 - y3)**2

        if potential_type == '1/r':
            return (T1 + T2 - u12,
                    T1 + T3 - u13,
                    T2 + T3 - u23)

        if potential_type == '1/r2':
            return (T1 + T2 - u12**2,
                    T1 + T3 - u13**2,
                    T2 + T3 - u23**2)

        if potential_type.startswith('1/r^'):
            from sympy import Rational, nsimplify
            n_sym = nsimplify(n, rational=False)
            return (T1 + T2 - u12**n_sym,
                    T1 + T3 - u13**n_sym,
                    T2 + T3 - u23**n_sym)

        if potential_type == 'harmonic':
            return (T1 + T2 + r12_sq,
                    T1 + T3 + r13_sq,
                    T2 + T3 + r23_sq)

        raise ValueError(f"No symbolic Hamiltonians for '{potential_type}'")


# =============================================================================
# CORE ALGEBRA COMPUTATION (exact symbolic engine)
# =============================================================================

class PoissonAlgebra:
    """
    Pre-computes the Poisson algebra generators symbolically via exact_growth.py,
    then evaluates them numerically at arbitrary phase-space points.

    All symbolic work (brackets, simplification, lambdification) happens once
    in __init__.  Per-grid-point evaluation is pure NumPy -- fast and exact.
    """

    def __init__(self, config: AtlasConfig):
        self.config = config
        self.pot_meta = Potential.get_metadata(config.potential_type)

        charges_label = ""
        if config.charges is not None:
            charges_label = f" charges=({','.join(f'{c:+g}' for c in config.charges)})"
            pot_short = config.potential_type.replace('/', '')
            self.pot_meta = {**self.pot_meta,
                             'name': f"Coulomb {config.potential_type}{charges_label}"}

        print(f"  Building symbolic generators for {self.pot_meta['name']}...")
        t0 = time()

        H12, H13, H23 = Potential.get_symbolic_hamiltonians(
            config.potential_type, charges=config.charges)

        all_exprs = []
        all_names = []
        all_levels = []
        computed_pairs = set()

        # -- Level 0 --
        for name, expr in [("H12", H12), ("H13", H13), ("H23", H23)]:
            all_exprs.append(expr)
            all_names.append(name)
            all_levels.append(0)
        for i in range(3):
            for j in range(i + 1, 3):
                computed_pairs.add(frozenset({i, j}))

        # -- Level 1 --
        for short, full, i, j in [("K1", "{H12,H13}", 0, 1),
                                   ("K2", "{H12,H23}", 0, 2),
                                   ("K3", "{H13,H23}", 1, 2)]:
            expr = simplify_generator(
                poisson_bracket(all_exprs[i], all_exprs[j]))
            all_exprs.append(expr)
            all_names.append(short)
            all_levels.append(1)

        # -- Levels 2+ --
        for level in range(2, config.max_level + 1):
            frontier = [i for i, lv in enumerate(all_levels) if lv == level - 1]
            n_existing = len(all_exprs)
            new_exprs = []
            new_names = []

            for i in frontier:
                for j in range(n_existing):
                    if i == j:
                        continue
                    pair = frozenset({i, j})
                    if pair in computed_pairs:
                        continue
                    computed_pairs.add(pair)
                    if all_exprs[i] == 0 or all_exprs[j] == 0:
                        continue
                    expr = simplify_generator(
                        poisson_bracket(all_exprs[i], all_exprs[j]))
                    new_exprs.append(expr)
                    new_names.append(f"{{{all_names[i]},{all_names[j]}}}")

            for expr, name in zip(new_exprs, new_names):
                all_exprs.append(expr)
                all_names.append(name)
                all_levels.append(level)

            n_zero = sum(1 for e in new_exprs if e == 0)
            print(f"    Level {level}: {len(new_exprs)} generators "
                  f"({n_zero} zero)")

        # Filter zeros for evaluation
        nonzero_mask = [i for i, e in enumerate(all_exprs) if e != 0]
        self._exprs = [all_exprs[i] for i in nonzero_mask]
        self._levels = [all_levels[i] for i in nonzero_mask]
        self._names = [all_names[i] for i in nonzero_mask]
        self._n_generators = len(self._exprs)

        print(f"    Total: {len(all_exprs)} generators, "
              f"{self._n_generators} non-zero")

        # Lambdify once
        print(f"    Compiling evaluator...", flush=True)
        self._evaluate = lambdify_generators(self._exprs)

        elapsed = time() - t0
        print(f"  Algebra ready ({elapsed:.1f}s)")

    # -----------------------------------------------------------------
    # Local phase-space sampling
    # -----------------------------------------------------------------
    def _sample_local(self, positions: np.ndarray,
                      n_samples: int, epsilon: float,
                      mom_range: float = 0.5,
                      min_sep: float = 0.1,
                      seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate phase-space samples localised around a configuration.

        Returns (Z_qp, Z_u) matching the format expected by
        lambdify_generators: Z_qp is (N, 12), Z_u is (N, 3).
        """
        rng = np.random.RandomState(seed)
        base_q = positions.flatten()  # (6,)

        base_r_min = self._base_min_sep(base_q)
        effective_min_sep = min(min_sep, 0.3 * base_r_min)

        Z_qp = np.zeros((n_samples, 12))
        Z_u = np.zeros((n_samples, 3))
        accepted = 0
        max_attempts = n_samples * 200

        for _ in range(max_attempts):
            if accepted >= n_samples:
                break

            q = base_q + rng.randn(6) * epsilon
            p = rng.randn(6) * mom_range

            dx12 = q[0] - q[2]; dy12 = q[1] - q[3]
            dx13 = q[0] - q[4]; dy13 = q[1] - q[5]
            dx23 = q[2] - q[4]; dy23 = q[3] - q[5]

            r12 = np.sqrt(dx12**2 + dy12**2)
            r13 = np.sqrt(dx13**2 + dy13**2)
            r23 = np.sqrt(dx23**2 + dy23**2)

            if min(r12, r13, r23) < effective_min_sep:
                continue

            Z_qp[accepted, :6] = q
            Z_qp[accepted, 6:] = p
            Z_u[accepted] = [1.0 / r12, 1.0 / r13, 1.0 / r23]
            accepted += 1

        if accepted < n_samples:
            Z_qp = Z_qp[:accepted]
            Z_u = Z_u[:accepted]

        return Z_qp, Z_u

    @staticmethod
    def _base_min_sep(base_q):
        """Minimum pairwise distance in a flat (6,) position vector."""
        dx12 = base_q[0] - base_q[2]; dy12 = base_q[1] - base_q[3]
        dx13 = base_q[0] - base_q[4]; dy13 = base_q[1] - base_q[5]
        dx23 = base_q[2] - base_q[4]; dy23 = base_q[3] - base_q[5]
        return min(np.sqrt(dx12**2 + dy12**2),
                   np.sqrt(dx13**2 + dy13**2),
                   np.sqrt(dx23**2 + dy23**2))

    # -----------------------------------------------------------------
    # Rank computation at a single configuration
    # -----------------------------------------------------------------
    def compute_rank_at_configuration(self, positions: np.ndarray,
                                       level: int,
                                       n_samples: Optional[int] = None,
                                       epsilon: Optional[float] = None
                                       ) -> Tuple[int, np.ndarray, dict]:
        """
        Evaluate the pre-compiled generators at local phase-space samples
        around *positions* and determine the SVD rank.

        Args:
            positions: (3, 2) array of body positions
            level: bracket level (0-3).  Only generators up to this level
                   are included in the SVD.
            n_samples: override config.n_phase_samples
            epsilon: override config.epsilon

        Returns:
            rank: integer local SVD rank
            singular_values: full array of singular values
            info: dict with diagnostics
        """
        if n_samples is None:
            n_samples = self.config.n_phase_samples
        if epsilon is None:
            epsilon = self.config.epsilon

        p = positions.reshape(3, 2)
        r12 = np.linalg.norm(p[0] - p[1])
        r13 = np.linalg.norm(p[0] - p[2])
        r23 = np.linalg.norm(p[1] - p[2])
        r_min = min(r12, r13, r23)
        epsilon = min(epsilon, 0.1 * r_min)

        Z_qp, Z_u = self._sample_local(positions, n_samples, epsilon)
        full_matrix = self._evaluate(Z_qp, Z_u)   # (N, n_generators)

        # Select generators up to the requested level
        col_mask = [i for i, lv in enumerate(self._levels) if lv <= level]
        if not col_mask:
            return 0, np.array([]), {'n_generators': 0, 'level': level}

        sub = full_matrix[:, col_mask]

        # Column-normalise for numerical stability
        norms = np.linalg.norm(sub, axis=0)
        norms[norms < 1e-15] = 1.0
        sub = sub / norms

        U, S, Vt = svd(sub, full_matrices=False)

        rank = self._rank_from_gap(S)

        info = {
            'n_samples': n_samples,
            'epsilon': epsilon,
            'level': level,
            'n_generators': len(col_mask),
            'singular_values': S,
            'max_gap_ratio': self._max_gap_ratio(S),
            'gap_location': rank,
        }
        return rank, S, info

    # -----------------------------------------------------------------
    # SVD gap helpers (quiet -- no printing)
    # -----------------------------------------------------------------
    def _rank_from_gap(self, singular_values: np.ndarray) -> int:
        if len(singular_values) <= 1:
            return len(singular_values)
        for k in range(len(singular_values) - 1):
            if singular_values[k + 1] < 1e-10:
                return k + 1
            ratio = singular_values[k] / singular_values[k + 1]
            if ratio > self.config.svd_gap_threshold:
                return k + 1
        return len(singular_values)

    def _max_gap_ratio(self, singular_values: np.ndarray) -> float:
        if len(singular_values) <= 1:
            return float('inf')
        max_ratio = 0.0
        for k in range(len(singular_values) - 1):
            if singular_values[k + 1] > 1e-15:
                ratio = singular_values[k] / singular_values[k + 1]
                max_ratio = max(max_ratio, ratio)
        return max_ratio

    # -----------------------------------------------------------------
    # Tier detection and adaptive epsilon
    # -----------------------------------------------------------------
    @staticmethod
    def _find_tiers(singular_values: np.ndarray, threshold: float = 10.0,
                    max_tiers: int = 8) -> list:
        """Find tier boundaries: indices where sv[k]/sv[k+1] > threshold.

        Returns list of (index, gap_ratio) sorted by index, capped at
        *max_tiers* entries.  The "cliff" (transition to machine-epsilon
        noise) is included if present.
        """
        tiers = []
        n = len(singular_values)
        for k in range(n - 1):
            if singular_values[k + 1] < 1e-15:
                tiers.append((k + 1, float('inf')))
                break
            ratio = singular_values[k] / singular_values[k + 1]
            if ratio >= threshold:
                tiers.append((k + 1, ratio))
        tiers.sort(key=lambda t: t[0])
        return tiers[:max_tiers]

    @staticmethod
    def _gap_score(singular_values: np.ndarray, tiers: list,
                   cliff_weight: float = 1.0,
                   tier_weight: float = 0.3) -> float:
        """Composite score rewarding a clean cliff and resolved internal tiers.

        score = cliff_weight * log10(cliff_gap)
              + tier_weight  * sum(log10(internal_gaps))

        The cliff is defined as the tier with the largest gap ratio.
        Internal tiers are everything else.  Infinite gaps are capped
        at 1e16 for scoring purposes.
        """
        if not tiers:
            return 0.0
        cap = 1e16

        best_idx = max(range(len(tiers)), key=lambda i: min(tiers[i][1], cap))
        cliff_gap = min(tiers[best_idx][1], cap)
        score = cliff_weight * np.log10(max(cliff_gap, 1.0))

        for i, (_, gap) in enumerate(tiers):
            if i == best_idx:
                continue
            score += tier_weight * np.log10(max(min(gap, cap), 1.0))
        return score

    def compute_adaptive_rank(self, positions: np.ndarray, level: int,
                              eps_range: Tuple[float, float] = (1e-4, 5e-3),
                              n_eps: int = 8,
                              n_samples: Optional[int] = None,
                              tier_threshold: float = 10.0,
                              max_tiers: int = 8,
                              early_exit_patience: int = 2
                              ) -> Tuple[int, np.ndarray, dict]:
        """Sweep epsilon geometrically and pick the scale that maximises
        multi-tier gap clarity.

        Returns (rank, singular_values, info) where *info* contains::

            optimal_eps      – chosen epsilon
            gap_score        – composite score at optimal eps
            tier_boundaries  – list of (index, gap_ratio)
            n_eps_tested     – how many epsilons were evaluated
            all_eps          – array of tested epsilons
            all_scores       – array of scores at each tested eps
        """
        eps_lo, eps_hi = eps_range
        epsilons = np.geomspace(eps_hi, eps_lo, n_eps)

        best_score = -1.0
        best_rank = 0
        best_svs = np.array([])
        best_eps = epsilons[0]
        best_tiers: list = []
        decline_count = 0
        tested_eps = []
        tested_scores = []

        for eps in epsilons:
            rank, svs, _ = self.compute_rank_at_configuration(
                positions, level, n_samples=n_samples, epsilon=eps)
            tiers = self._find_tiers(svs, threshold=tier_threshold,
                                     max_tiers=max_tiers)
            score = self._gap_score(svs, tiers)
            tested_eps.append(eps)
            tested_scores.append(score)

            if score > best_score:
                best_score = score
                best_rank = rank
                best_svs = svs
                best_eps = eps
                best_tiers = tiers
                decline_count = 0
            else:
                decline_count += 1
                if decline_count >= early_exit_patience:
                    break

        info = {
            'optimal_eps': best_eps,
            'gap_score': best_score,
            'tier_boundaries': best_tiers,
            'n_eps_tested': len(tested_eps),
            'all_eps': np.array(tested_eps),
            'all_scores': np.array(tested_scores),
            'n_generators': len(best_svs),
            'level': level,
            'max_gap_ratio': self._max_gap_ratio(best_svs),
        }
        return best_rank, best_svs, info


# =============================================================================
# ATLAS COMPUTATION
# =============================================================================

class StabilityAtlas:
    """
    Computes and stores the dimension landscape over shape space.
    
    The atlas is a 2D grid over (mu, phi) shape coordinates, with
    the local SVD rank d_loc(n; mu, phi) computed at each grid point.
    
    Supports:
        - Fixed-resolution uniform grids
        - Adaptive refinement around features (rank changes)
        - Focused high-resolution scans around special configurations
        - Multi-potential comparison
    """
    
    def __init__(self, config: AtlasConfig):
        self.config = config
        self.algebra = PoissonAlgebra(config)
        self.results = {}
        
    def compute_uniform(self, 
                        mu_range: Tuple[float, float] = (0.2, 3.0),
                        phi_range: Tuple[float, float] = (0.1, np.pi - 0.1)
                        ) -> dict:
        """
        Compute the atlas on a uniform grid.
        
        Returns dict with:
            'mu_vals': 1D array of mu grid values
            'phi_vals': 1D array of phi grid values
            'rank_map': 2D array of local ranks, shape (n_mu, n_phi)
            'gap_map': 2D array of max gap ratios
            'sv_map': dict mapping (i,j) to singular value arrays
        """
        mu_vals, phi_vals = ShapeSpace.make_grid(
            self.config.resolution, self.config.grid_sizes,
            mu_range, phi_range
        )
        
        n_mu = len(mu_vals)
        n_phi = len(phi_vals)
        
        rank_map = np.zeros((n_mu, n_phi), dtype=int)
        gap_map = np.zeros((n_mu, n_phi))
        sv_map = {}
        
        total = n_mu * n_phi
        computed = 0
        
        print(f"\nComputing stability atlas:")
        print(f"  Potential: {self.algebra.pot_meta['name']}")
        print(f"  Masses: {self.config.masses}")
        if self.config.charges is not None:
            print(f"  Charges: {self.config.charges}")
        print(f"  Level: {self.config.max_level}")
        print(f"  Grid: {n_mu} x {n_phi} = {total} points")
        print(f"  Samples per point: {self.config.n_phase_samples}")
        print(f"  Epsilon: {self.config.epsilon}")
        print()
        
        for i, mu in enumerate(mu_vals):
            for j, phi in enumerate(phi_vals):
                positions = ShapeSpace.shape_to_positions(mu, phi)
                
                try:
                    rank, svs, info = self.algebra.compute_rank_at_configuration(
                        positions, self.config.max_level
                    )
                    rank_map[i, j] = rank
                    gap_map[i, j] = info['max_gap_ratio']
                    sv_map[(i, j)] = svs[:20]  # store top 20 singular values
                except Exception as e:
                    rank_map[i, j] = -1  # mark failures
                    gap_map[i, j] = 0
                    print(f"  WARNING: Failed at (mu={mu:.3f}, phi={phi:.3f}): {e}")
                
                computed += 1
                if computed % max(1, total // 20) == 0:
                    pct = 100 * computed / total
                    print(f"  [{pct:5.1f}%] ({i},{j}) mu={mu:.3f} phi={phi:.3f} "
                          f"rank={rank_map[i,j]} gap={gap_map[i,j]:.1e}")
        
        result = {
            'mu_vals': mu_vals,
            'phi_vals': phi_vals,
            'rank_map': rank_map,
            'gap_map': gap_map,
            'sv_map': sv_map,
            'config': {
                'masses': self.config.masses,
                'potential': self.config.potential_type,
                'charges': self.config.charges,
                'level': self.config.max_level,
                'epsilon': self.config.epsilon,
                'n_samples': self.config.n_phase_samples,
                'resolution': self.config.resolution,
            }
        }
        
        self.results = result
        return result
    
    def compute_focused(self, focus_name: str, resolution: str = 'fine') -> dict:
        """Compute high-resolution atlas around a special configuration."""
        mu_range, phi_range = ShapeSpace.focus_region(focus_name)
        
        # Override resolution for focused scan
        old_res = self.config.resolution
        self.config.resolution = resolution
        
        result = self.compute_uniform(mu_range, phi_range)
        result['focus'] = focus_name
        result['focus_center'] = ShapeSpace.SPECIAL_CONFIGS[focus_name]
        
        self.config.resolution = old_res
        return result
    
    def compute_adaptive(self,
                         mu_range: Tuple[float, float] = (0.2, 3.0),
                         phi_range: Tuple[float, float] = (0.1, np.pi - 0.1),
                         depth: int = 0) -> dict:
        """
        Adaptive refinement: start coarse, refine where rank changes.
        
        Any region where neighboring grid points have different ranks
        gets subdivided and recomputed at higher resolution. This
        concentrates computational effort at the boundaries between
        rank regions — exactly where the interesting topology is.
        """
        # Start with coarse grid
        self.config.resolution = 'coarse'
        result = self.compute_uniform(mu_range, phi_range)
        
        if depth >= self.config.adaptive_max_depth:
            return result
        
        # Find regions where rank changes
        rank_map = result['rank_map']
        mu_vals = result['mu_vals']
        phi_vals = result['phi_vals']
        
        refinement_regions = []
        
        for i in range(rank_map.shape[0] - 1):
            for j in range(rank_map.shape[1] - 1):
                # Check if any neighbor has a different rank
                local_ranks = [
                    rank_map[i, j], rank_map[i+1, j],
                    rank_map[i, j+1], rank_map[i+1, j+1]
                ]
                if len(set(r for r in local_ranks if r >= 0)) > 1:
                    # Rank varies here — refine
                    refinement_regions.append((
                        (mu_vals[i], mu_vals[min(i+1, len(mu_vals)-1)]),
                        (phi_vals[j], phi_vals[min(j+1, len(phi_vals)-1)])
                    ))
        
        if refinement_regions:
            print(f"\n  Adaptive depth {depth}: found {len(refinement_regions)} "
                  f"regions to refine")
            
            result['refinements'] = []
            for mu_r, phi_r in refinement_regions:
                # Pad the region slightly
                pad_mu = (mu_r[1] - mu_r[0]) * 0.1
                pad_phi = (phi_r[1] - phi_r[0]) * 0.1
                
                sub_result = self.compute_adaptive(
                    (mu_r[0] - pad_mu, mu_r[1] + pad_mu),
                    (phi_r[0] - pad_phi, phi_r[1] + pad_phi),
                    depth + 1
                )
                result['refinements'].append(sub_result)
        
        return result
    
    def compute_epsilon_sweep(self, config_name: str,
                               epsilons: Optional[List[float]] = None) -> dict:
        """
        Sweep epsilon (local ball radius) at a fixed configuration.
        
        This tests how the local rank changes as you zoom in/out.
        A rank that changes with epsilon indicates the configuration
        is near a boundary between rank regions.
        
        This is the core test from Section 4.4.2 of the contractibility
        objection: if rank varies with epsilon at a Lagrange point,
        the landscape has features.
        """
        if epsilons is None:
            epsilons = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
        
        mu, phi = ShapeSpace.SPECIAL_CONFIGS[config_name]
        positions = ShapeSpace.shape_to_positions(mu, phi)
        
        ranks = []
        sv_data = []
        
        print(f"\nEpsilon sweep at {config_name} (mu={mu:.3f}, phi={phi:.3f}):")
        
        for eps in epsilons:
            rank, svs, info = self.algebra.compute_rank_at_configuration(
                positions, self.config.max_level,
                n_samples=self.config.n_phase_samples * 2,  # extra samples for precision
                epsilon=eps
            )
            ranks.append(rank)
            sv_data.append(svs)
            print(f"  eps={eps:.1e}  rank={rank}  gap={info['max_gap_ratio']:.1e}")
        
        return {
            'config_name': config_name,
            'mu': mu, 'phi': phi,
            'epsilons': epsilons,
            'ranks': ranks,
            'singular_values': sv_data,
            'level': self.config.max_level,
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

class AtlasVisualizer:
    """
    Generate visualizations of the stability atlas.
    
    Produces:
    1. Heat map of local rank over shape space
    2. Contour plot of SVD gap ratios
    3. Epsilon sweep plots at special configurations
    4. Comparison plots across potential types
    """
    
    @staticmethod
    def save_plot_data(result: dict, output_dir: str):
        """
        Save atlas data in a format that can be visualized with
        matplotlib, plotly, or any other tool.
        
        Also generates a self-contained HTML visualization using plotly CDN.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw data as numpy arrays
        np.save(os.path.join(output_dir, 'mu_vals.npy'), result['mu_vals'])
        np.save(os.path.join(output_dir, 'phi_vals.npy'), result['phi_vals'])
        np.save(os.path.join(output_dir, 'rank_map.npy'), result['rank_map'])
        np.save(os.path.join(output_dir, 'gap_map.npy'), result['gap_map'])
        
        # Save config
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(result['config'], f, indent=2)
        
        # Generate HTML visualization
        AtlasVisualizer._generate_html(result, output_dir)
        
        print(f"\nResults saved to {output_dir}/")
        print(f"  Open atlas_visualization.html in a browser to view.")
    
    @staticmethod
    def _generate_html(result: dict, output_dir: str):
        """Generate a self-contained HTML visualization."""
        
        mu_vals = result['mu_vals'].tolist()
        phi_vals = result['phi_vals'].tolist()
        rank_map = result['rank_map'].tolist()
        gap_map = result['gap_map'].tolist()
        config = result['config']
        
        # Mark special configurations
        specials = []
        for name, (mu, phi) in ShapeSpace.SPECIAL_CONFIGS.items():
            if (mu_vals[0] <= mu <= mu_vals[-1] and 
                phi_vals[0] <= phi <= phi_vals[-1]):
                specials.append({'name': name, 'mu': mu, 
                               'phi': phi, 'phi_deg': phi * 180 / np.pi})
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Stability Atlas — {config['potential']} potential</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Georgia', serif; background: #1a1a2e; color: #e0e0e0; 
               margin: 0; padding: 20px; }}
        h1 {{ color: #c9a96e; text-align: center; margin-bottom: 5px; }}
        h2 {{ color: #8899aa; text-align: center; font-weight: normal; margin-top: 5px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .plot {{ margin: 20px 0; background: #16213e; border-radius: 8px; padding: 10px; }}
        .info {{ background: #0f3460; border-radius: 8px; padding: 15px; margin: 20px 0;
                 font-family: monospace; font-size: 14px; line-height: 1.6; }}
        .info span.label {{ color: #c9a96e; }}
        .finding {{ background: #1a3a5c; border-left: 4px solid #c9a96e; padding: 15px; 
                    margin: 15px 0; border-radius: 0 8px 8px 0; }}
        .finding h3 {{ color: #c9a96e; margin-top: 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Stability Atlas</h1>
    <h2>Local Dimension Landscape of the Pairwise Poisson Algebra</h2>
    
    <div class="info">
        <span class="label">Potential:</span> {config['potential']}<br>
        <span class="label">Masses:</span> {config['masses']}<br>
        <span class="label">Bracket level:</span> {config['level']}<br>
        <span class="label">Local ball radius (ε):</span> {config['epsilon']}<br>
        <span class="label">Phase-space samples per point:</span> {config['n_samples']}<br>
        <span class="label">Grid resolution:</span> {config['resolution']}
    </div>
    
    <div class="finding">
        <h3>Reading the Atlas</h3>
        <p><strong>Low rank (dark)</strong> = stable configuration. Local approximate 
        conservation laws hold. The system's identity dynamics are constrained here.</p>
        <p><strong>High rank (bright)</strong> = unconstrained dynamics. Full algebraic 
        complexity accessible. This is the "hot mess" regime.</p>
        <p><strong>Rank transitions</strong> = boundaries between basins. Phase transition 
        territory. If these exist, the landscape has non-trivial topology — 
        it is not a flat surface.</p>
    </div>
    
    <div class="plot" id="rankPlot"></div>
    <div class="plot" id="gapPlot"></div>
    
    <script>
    var mu_vals = {json.dumps(mu_vals)};
    var phi_vals_rad = {json.dumps(phi_vals)};
    var phi_vals_deg = phi_vals_rad.map(x => x * 180 / Math.PI);
    var rank_map = {json.dumps(rank_map)};
    var gap_map = {json.dumps(gap_map)};
    var specials = {json.dumps(specials)};
    
    // Rank heatmap
    var rankTrace = {{
        z: rank_map,
        x: phi_vals_deg,
        y: mu_vals,
        type: 'heatmap',
        colorscale: [
            [0, '#0d1117'],
            [0.25, '#1a3a5c'],
            [0.5, '#2e6e8e'],
            [0.75, '#c9a96e'],
            [1, '#f5e6c8']
        ],
        colorbar: {{ title: 'Local Rank d_loc', titleside: 'right' }},
        hovertemplate: 'φ: %{{x:.1f}}°<br>μ: %{{y:.3f}}<br>rank: %{{z}}<extra></extra>'
    }};
    
    // Mark special configurations
    var specialTrace = {{
        x: specials.map(s => s.phi_deg),
        y: specials.map(s => s.mu),
        text: specials.map(s => s.name),
        mode: 'markers+text',
        type: 'scatter',
        marker: {{ size: 12, color: '#ff6b6b', symbol: 'diamond',
                   line: {{ width: 2, color: 'white' }} }},
        textposition: 'top center',
        textfont: {{ color: '#ff6b6b', size: 11 }},
        hovertemplate: '%{{text}}<br>φ: %{{x:.1f}}°<br>μ: %{{y:.3f}}<extra></extra>'
    }};
    
    var rankLayout = {{
        title: {{ text: 'Local SVD Rank over Shape Space',
                  font: {{ color: '#c9a96e', size: 18 }} }},
        xaxis: {{ title: 'φ (angle at vertex 1, degrees)', color: '#8899aa',
                  gridcolor: '#2a2a4a' }},
        yaxis: {{ title: 'μ = r₁₃/r₁₂ (side ratio)', color: '#8899aa',
                  gridcolor: '#2a2a4a' }},
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#0d1117',
        font: {{ color: '#e0e0e0' }},
        margin: {{ t: 60, b: 60, l: 60, r: 60 }}
    }};
    
    Plotly.newPlot('rankPlot', [rankTrace, specialTrace], rankLayout);
    
    // Gap ratio heatmap (log scale)
    var logGap = gap_map.map(row => row.map(v => v > 0 ? Math.log10(v) : 0));
    
    var gapTrace = {{
        z: logGap,
        x: phi_vals_deg,
        y: mu_vals,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: {{ title: 'log₁₀(gap ratio)', titleside: 'right' }},
        hovertemplate: 'φ: %{{x:.1f}}°<br>μ: %{{y:.3f}}<br>log₁₀(gap): %{{z:.1f}}<extra></extra>'
    }};
    
    var gapLayout = {{
        title: {{ text: 'SVD Gap Ratio (confidence of rank determination)',
                  font: {{ color: '#c9a96e', size: 18 }} }},
        xaxis: {{ title: 'φ (angle at vertex 1, degrees)', color: '#8899aa',
                  gridcolor: '#2a2a4a' }},
        yaxis: {{ title: 'μ = r₁₃/r₁₂ (side ratio)', color: '#8899aa',
                  gridcolor: '#2a2a4a' }},
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#0d1117',
        font: {{ color: '#e0e0e0' }},
        margin: {{ t: 60, b: 60, l: 60, r: 60 }}
    }};
    
    Plotly.newPlot('gapPlot', [gapTrace, specialTrace], gapLayout);
    </script>
    
    <div class="finding">
        <h3>What To Look For</h3>
        <p>If this atlas shows <strong>uniform rank everywhere</strong> — no features, 
        no variation — then the dimension landscape is flat and the mass invariance 
        may be measuring a contractible surface.</p>
        <p>If this atlas shows <strong>rank drops at special configurations</strong> 
        (Lagrange equilateral, Euler collinear) with full rank elsewhere — 
        the landscape has topology. The invariance is non-trivial. 
        The features are the stable configurations of the three-body problem, 
        detected purely from the algebra.</p>
        <p>Any structure at any resolution is compelling.</p>
    </div>
</div>
</body>
</html>"""
        
        with open(os.path.join(output_dir, 'atlas_visualization.html'), 'w',
                  encoding='utf-8') as f:
            f.write(html)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_atlas(potential_type='1/r', masses=(1.0, 1.0, 1.0), level=2,
              resolution='coarse', focus=None, epsilon=1e-2,
              n_samples=200, output_dir='./atlas_output', charges=None):
    """
    Main entry point for computing a stability atlas.
    
    Quick start:
        run_atlas()                           # coarse 1/r atlas, level 2
        run_atlas(level=3, resolution='medium')  # higher level, finer grid
        run_atlas(focus='lagrange')            # zoom on Lagrange point
        run_atlas(potential_type='harmonic')   # control case
        run_atlas(charges=(2, -1, -1))        # helium Coulomb
    """
    
    config = AtlasConfig(
        masses=masses,
        potential_type=potential_type,
        max_level=level,
        resolution=resolution,
        epsilon=epsilon,
        n_phase_samples=n_samples,
        output_dir=output_dir,
        focus=focus,
        charges=charges,
    )
    
    atlas = StabilityAtlas(config)
    
    if focus:
        result = atlas.compute_focused(focus, resolution='fine')
    elif resolution == 'adaptive':
        result = atlas.compute_adaptive()
    else:
        result = atlas.compute_uniform()
    
    # Save and visualize
    AtlasVisualizer.save_plot_data(result, output_dir)
    
    # Also do epsilon sweep at special configurations
    print("\n" + "="*60)
    print("EPSILON SWEEPS AT SPECIAL CONFIGURATIONS")
    print("="*60)
    
    for name in ['lagrange', 'euler_collinear', 'isosceles_right']:
        try:
            sweep = atlas.compute_epsilon_sweep(name)
            print(f"\n  {name}: ranks = {sweep['ranks']}")
            if len(set(sweep['ranks'])) > 1:
                print(f"  *** RANK VARIES WITH EPSILON — FEATURE DETECTED ***")
            else:
                print(f"  Rank stable at {sweep['ranks'][0]}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
    
    return atlas, result


def run_comparison(level=2, resolution='coarse'):
    """
    Run atlases for multiple potential types and compare.
    This is the decisive test for the contractibility objection.
    """
    potentials = ['1/r', '1/r2', 'harmonic']
    results = {}
    
    for pot in potentials:
        print(f"\n{'='*60}")
        print(f"  COMPUTING ATLAS FOR {pot} POTENTIAL")
        print(f"{'='*60}")
        
        output_dir = f'./atlas_output/{pot.replace("/", "_")}'
        atlas, result = run_atlas(
            potential_type=pot, level=level, resolution=resolution,
            output_dir=output_dir
        )
        results[pot] = result
    
    # Compare
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for pot, result in results.items():
        rank_map = result['rank_map']
        valid = rank_map[rank_map >= 0]
        print(f"\n  {pot}:")
        print(f"    Rank range: [{valid.min()}, {valid.max()}]")
        print(f"    Unique ranks: {sorted(set(valid.flatten()))}")
        print(f"    Rank varies: {'YES — FEATURES DETECTED' if valid.min() != valid.max() else 'No — uniform'}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Stability Atlas computation')
    parser.add_argument('--potential', default='1/r', 
                       choices=['1/r', '1/r2', 'harmonic'])
    parser.add_argument('--masses', nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument('--charges', nargs=3, type=int, default=None,
                       metavar=('Q1', 'Q2', 'Q3'),
                       help='Charges for Coulomb potential, e.g. 2 -1 -1')
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--resolution', default='coarse',
                       choices=['coarse', 'medium', 'fine', 'ultra', 'adaptive'])
    parser.add_argument('--focus', default=None,
                       choices=['lagrange', 'euler_collinear', 'isosceles_right',
                               'lagrange_obtuse', 'near_collision_12'])
    parser.add_argument('--epsilon', type=float, default=1e-2)
    parser.add_argument('--n-samples', type=int, default=200)
    parser.add_argument('--output', default=None,
                       help='Output directory (auto-named if not specified)')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison across potential types')
    
    args = parser.parse_args()
    
    charges = tuple(args.charges) if args.charges else None

    if args.output is not None:
        output_dir = args.output
    elif charges is not None:
        tag = "_".join(f"{c:+d}" for c in charges)
        output_dir = f'./atlas_output/coulomb_{tag}'
    else:
        output_dir = f'./atlas_output/{args.potential.replace("/", "_")}'

    if args.compare:
        run_comparison(level=args.level, resolution=args.resolution)
    else:
        run_atlas(
            potential_type=args.potential,
            masses=tuple(args.masses),
            level=args.level,
            resolution=args.resolution,
            focus=args.focus,
            epsilon=args.epsilon,
            n_samples=args.n_samples,
            output_dir=output_dir,
            charges=charges,
        )
