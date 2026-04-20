"""Regression tests for the core Poisson algebra engines."""
import os
import sys

# Ensure repo root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _seq_from_dims(dims, max_level):
    """Convert NBodyAlgebra.compute_growth's {level: dim} dict to a list."""
    return [dims[lv] for lv in range(max_level + 1)]


def test_sympy_version():
    import sympy
    parts = tuple(int(x) for x in sympy.__version__.split(".")[:3])
    assert parts >= (1, 13, 3), f"Need sympy>=1.13.3, got {sympy.__version__}"


def test_nbody_n3_levels_0_2():
    """Canonical N=3 d=2 1/r baseline: dimension sequence [3, 6, 17]."""
    from nbody.exact_growth_nbody import NBodyAlgebra
    alg = NBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r",
                       checkpoint_dir=None)
    dims = alg.compute_growth(max_level=2, n_samples=500, seed=42)
    seq = _seq_from_dims(dims, 2)
    assert seq == [3, 6, 17], f"Expected [3, 6, 17], got {seq}"


def test_three_body_algebra_d2_levels_0_1():
    """ThreeBodyAlgebra d=2 baseline: dimension sequence [3, 6]."""
    sys.path.insert(0, os.path.join(ROOT, "3d"))
    from exact_growth_nd import ThreeBodyAlgebra
    alg = ThreeBodyAlgebra(d_spatial=2, checkpoint_dir=None)
    dims = alg.compute_growth(max_level=1, n_samples=500, seed=42)
    seq = _seq_from_dims(dims, 1)
    assert seq == [3, 6], f"Expected [3, 6], got {seq}"


def test_planar_engine_levels_0_2():
    """Planar legacy engine via the canonical NBodyAlgebra path.

    The free function ``exact_growth.compute_exact_growth`` does not return
    its dimension sequence (prints only). We exercise the same simplify
    pipeline through ``NBodyAlgebra(n_bodies=3, d_spatial=2, potential='1/r')``
    which uses the same ``simplify_generator`` -> SVD code path.
    """
    from nbody.exact_growth_nbody import NBodyAlgebra
    alg = NBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r",
                       checkpoint_dir=None)
    dims = alg.compute_growth(max_level=2, n_samples=500, seed=42)
    seq = _seq_from_dims(dims, 2)
    assert seq == [3, 6, 17], f"Expected [3, 6, 17], got {seq}"


def test_schwarzschild_composite_levels_0_3():
    """Schwarzschild composite N=3 d=2 L<=3 -> [3, 6, 17, 116].

    Guards the simplify_generator patch from cancel -> together (April 2026).
    Cancel hangs/OOMs on this case; together completes in ~60-90s. Any future
    regression that puts cancel back in the hot loop will time out CI.
    Validation evidence: bench_flint/validation_summary.md.
    """
    import sympy as sp
    from nbody.exact_growth_nbody import NBodyAlgebra

    # Schwarzschild radial effective potential V_eff = -M*u + (L^2/2)*u^2 - M*L^2*u^3
    # with M=1, L=1 so the params are (-1, 1), (1/2, 2), (-1, 3).
    params = [
        (-sp.Integer(1), 1),
        (sp.Rational(1, 2), 2),
        (-sp.Integer(1), 3),
    ]
    alg = NBodyAlgebra(
        n_bodies=3, d_spatial=2, potential="composite",
        potential_params=params, checkpoint_dir=None,
    )
    # n_samples=500 gives a clean SVD gap at L=3 for this composite; lower
    # values (e.g. 300) hit the precision wall and report 115 instead of 116.
    # See bench_flint/diagnose_extreme_mass.json for the wider study.
    dims = alg.compute_growth(max_level=3, n_samples=500, seed=42)
    seq = _seq_from_dims(dims, 3)
    assert seq == [3, 6, 17, 116], (
        f"Expected [3, 6, 17, 116], got {seq}. "
        "If this fails after a simplify_generator change, see "
        "bench_flint/validation_summary.md and revert."
    )


if __name__ == "__main__":
    tests = [
        ("SymPy version", test_sympy_version),
        ("NBodyAlgebra N=3 d=2 1/r levels 0-2", test_nbody_n3_levels_0_2),
        ("Planar engine equivalent levels 0-2", test_planar_engine_levels_0_2),
        ("ThreeBodyAlgebra d=2 levels 0-1", test_three_body_algebra_d2_levels_0_1),
        ("Schwarzschild composite levels 0-3", test_schwarzschild_composite_levels_0_3),
    ]
    failed = 0
    for name, fn in tests:
        try:
            print(f"Running: {name} ...", end=" ", flush=True)
            fn()
            print("PASS")
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1
    sys.exit(failed)
