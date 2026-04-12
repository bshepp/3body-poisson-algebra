"""Regression tests for the core Poisson algebra engines."""
import sys
import os

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_sympy_version():
    import sympy
    parts = tuple(int(x) for x in sympy.__version__.split(".")[:3])
    assert parts >= (1, 13, 3), f"Need sympy>=1.13.3, got {sympy.__version__}"


def test_nbody_n3_levels_0_2():
    from nbody.exact_growth_nbody import NBodyAlgebra
    alg = NBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r",
                       checkpoint_dir=None)
    dims = alg.compute_exact_growth(max_level=2, n_samples=5000)
    assert dims == [3, 6, 17], f"Expected [3, 6, 17], got {dims}"


def test_planar_levels_0_2():
    from exact_growth import compute_exact_growth
    dims = compute_exact_growth(max_level=2, n_samples=5000)
    assert dims == [3, 6, 17], f"Expected [3, 6, 17], got {dims}"


def test_three_body_algebra_d2_levels_0_1():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "3d"))
    from exact_growth_nd import ThreeBodyAlgebra
    alg = ThreeBodyAlgebra(d_spatial=2, checkpoint_dir=None)
    dims = alg.compute_exact_growth(max_level=1, n_samples=5000)
    assert dims == [3, 6], f"Expected [3, 6], got {dims}"


if __name__ == "__main__":
    tests = [
        ("SymPy version", test_sympy_version),
        ("NBodyAlgebra N=3 levels 0-2", test_nbody_n3_levels_0_2),
        ("Planar engine levels 0-2", test_planar_levels_0_2),
        ("ThreeBodyAlgebra d=2 levels 0-1", test_three_body_algebra_d2_levels_0_1),
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
