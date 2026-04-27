"""Validation tests for the shape-sphere atlas driver.

Run:  python -m pytest test_shape_sphere_atlas.py -v
Or:   python test_shape_sphere_atlas.py   (pure-stdlib runner)
"""

import math
import sys
import numpy as np

from shape_sphere_atlas import (
    SQRT3,
    sphere_to_positions,
    thetaphi_to_sphere,
)


# --------------------------------------------------------------------------- #
# 1. Landmarks                                                                #
# --------------------------------------------------------------------------- #

def _shape_coords(positions):
    """Same Jacobi → (s1,s2,s3) map as the JS viewer's muPhiToShapeSphere,
    but applied directly to a (3,2) positions array (any frame, COM-free or
    not — translation invariant).  Used here only for the round-trip check.
    """
    r1, r2, r3 = positions
    rho1 = r2 - r1
    rho2 = (2.0 / SQRT3) * (r3 - 0.5 * (r1 + r2))
    R2 = np.dot(rho1, rho1) + np.dot(rho2, rho2)
    w1 = np.dot(rho1, rho1) - np.dot(rho2, rho2)
    w2 = 2.0 * np.dot(rho1, rho2)
    w3 = 2.0 * (rho1[0] * rho2[1] - rho1[1] * rho2[0])
    return np.array([w1, w2, w3]) / R2


def test_pole_north_is_equilateral_L4():
    """s = (0,0,+1) must produce an equilateral triangle (L4)."""
    pos = sphere_to_positions(0.0, 0.0, 1.0)
    assert pos is not None
    r12 = np.linalg.norm(pos[1] - pos[0])
    r13 = np.linalg.norm(pos[2] - pos[0])
    r23 = np.linalg.norm(pos[2] - pos[1])
    assert math.isclose(r12, r13, rel_tol=1e-12)
    assert math.isclose(r12, r23, rel_tol=1e-12)
    # Orientation: r3 should be on the +y side (positive cross product).
    cross = (pos[1, 0] - pos[0, 0]) * (pos[2, 1] - pos[0, 1]) \
          - (pos[1, 1] - pos[0, 1]) * (pos[2, 0] - pos[0, 0])
    assert cross > 0


def test_pole_south_is_equilateral_L5():
    pos = sphere_to_positions(0.0, 0.0, -1.0)
    r12 = np.linalg.norm(pos[1] - pos[0])
    r13 = np.linalg.norm(pos[2] - pos[0])
    r23 = np.linalg.norm(pos[2] - pos[1])
    assert math.isclose(r12, r13, rel_tol=1e-12)
    assert math.isclose(r12, r23, rel_tol=1e-12)
    cross = (pos[1, 0] - pos[0, 0]) * (pos[2, 1] - pos[0, 1]) \
          - (pos[1, 1] - pos[0, 1]) * (pos[2, 0] - pos[0, 0])
    assert cross < 0


def test_binary_collisions_on_equator():
    """The three equator binary-collision points should make two bodies coincide."""
    # r2 = r3 at (s1,s2,s3) = (1/2, +√3/2, 0)
    pos = sphere_to_positions(0.5, SQRT3 / 2, 0.0)
    assert np.linalg.norm(pos[1] - pos[2]) < 1e-12
    # r1 = r3 at (1/2, -√3/2, 0)
    pos = sphere_to_positions(0.5, -SQRT3 / 2, 0.0)
    assert np.linalg.norm(pos[0] - pos[2]) < 1e-12


def test_binary_r1r2_is_degenerate():
    """The s1 = -1 cap is the r1=r2 collision; driver returns None."""
    assert sphere_to_positions(-1.0, 0.0, 0.0) is None
    assert sphere_to_positions(-1.0 + 1e-15, 0.0, 0.0) is None


def test_round_trip_jacobi():
    """sphere_to_positions ∘ Jacobi map = identity on a generic point."""
    rng = np.random.default_rng(20260426)
    for _ in range(50):
        # Sample a generic sphere point bounded away from s1 = -1.
        v = rng.standard_normal(3)
        v /= np.linalg.norm(v)
        if v[0] < -0.9:
            v[0] = -0.9
            v /= np.linalg.norm(v)
        s1, s2, s3 = v
        pos = sphere_to_positions(s1, s2, s3)
        assert pos is not None
        s_back = _shape_coords(pos)
        assert np.allclose(s_back, [s1, s2, s3], atol=1e-12), \
            f"Round-trip mismatch: {s_back} vs {[s1, s2, s3]}"


def test_thetaphi_to_sphere_matches_landmarks():
    s = thetaphi_to_sphere(0.0, 0.0)
    assert np.allclose(s, (0.0, 0.0, 1.0), atol=1e-12)
    s = thetaphi_to_sphere(np.pi, 0.0)
    assert np.allclose(s, (0.0, 0.0, -1.0), atol=1e-12)
    s = thetaphi_to_sphere(np.pi / 2, np.pi / 3)
    assert np.allclose(s, (0.5, SQRT3 / 2, 0.0), atol=1e-12)


def test_hyperradius_is_unity():
    """All recovered configurations should have R² = |ρ1|² + |ρ2|² = 1."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.standard_normal(3); v /= np.linalg.norm(v)
        if v[0] < -0.9:
            continue
        pos = sphere_to_positions(*v)
        rho1 = pos[1] - pos[0]
        rho2 = (2.0 / SQRT3) * (pos[2] - 0.5 * (pos[0] + pos[1]))
        R2 = np.dot(rho1, rho1) + np.dot(rho2, rho2)
        assert math.isclose(R2, 1.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Algebra-level test (slow — builds the symbolic engine, ~30 s)               #
# --------------------------------------------------------------------------- #

def test_generic_point_rank_matches_reference_sequence():
    """At a generic interior point of the shape sphere, the planar 3-body
    Poisson algebra at level 3 with potential 1/r must reach the published
    generic dimension 116 (the [3, 6, 17, 116] sequence).

    The Lagrange equilateral points (poles) and the binary-collision
    equator points all have additional symmetry → strict rank drops, so
    this test deliberately picks a generic point off all of them.
    """
    from stability_atlas import AtlasConfig, PoissonAlgebra
    cfg = AtlasConfig(potential_type='1/r', max_level=3,
                      n_phase_samples=400, epsilon=1e-3,
                      svd_gap_threshold=1e4)
    alg = PoissonAlgebra(cfg)
    # Generic point: away from poles, equator, and the symmetry meridians.
    s1, s2, s3 = thetaphi_to_sphere(0.7, 0.4)
    pos = sphere_to_positions(s1, s2, s3)
    rank, _, info = alg.compute_rank_at_configuration(pos, 3, epsilon=1e-3)
    assert rank == 116, f"Expected generic rank 116 for 1/r, got {rank}"
    assert info['max_gap_ratio'] > 1e6


def test_L4_rank_drops_at_lagrange_pole():
    """Sanity: at L4 (north pole, equilateral) the rank is strictly below
    the generic 116 due to the extra symmetry. Documents the well-known
    Lagrange-point rank drop.
    """
    from stability_atlas import AtlasConfig, PoissonAlgebra
    cfg = AtlasConfig(potential_type='1/r', max_level=3,
                      n_phase_samples=400, epsilon=1e-3,
                      svd_gap_threshold=1e4)
    alg = PoissonAlgebra(cfg)
    pos = sphere_to_positions(0.0, 0.0, 1.0)
    rank, _, _ = alg.compute_rank_at_configuration(pos, 3, epsilon=1e-3)
    assert rank < 116, f"Expected rank drop at L4, got generic rank {rank}"
    assert rank >= 100, f"Suspiciously low rank at L4: {rank}"


# --------------------------------------------------------------------------- #
# Stdlib runner so we don't require pytest                                    #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    fast = '--fast' in sys.argv
    fns = [v for k, v in globals().items() if k.startswith('test_')]
    if fast:
        fns = [f for f in fns if 'rank' not in f.__name__]
    fail = 0
    for f in fns:
        try:
            f()
            print(f"  PASS  {f.__name__}")
        except AssertionError as e:
            fail += 1
            print(f"  FAIL  {f.__name__}: {e}")
        except Exception as e:
            fail += 1
            print(f"  ERROR {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-fail}/{len(fns)} passed")
    sys.exit(1 if fail else 0)
