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


def test_resume_pair_reconstruction():
    """E1 regression (exact_growth.py --resume pair reconstruction).

    Simulate levels [0, 0, 0, 1, 1, 1] (3 level-0 generators, 3 level-1
    generators) with start_level=2, i.e. a checkpoint that finished
    level 1 and is about to start level 2. A fresh, uninterrupted run
    reaching this same point has only ever marked the 3 level0-level0
    pairs as computed_pairs (level 1's own loop never touches
    computed_pairs -- see nbody/exact_growth_nbody.py:929-948, mirrored
    in exact_growth.py's level-1 block). The 9 {level0,level1} cross
    pairs must NOT be marked, since level 2's loop still needs to form
    those brackets. The old buggy condition
    (``all_levels[i] + all_levels[j] < start_level``) marks all 12
    pairs (3 level0-level0 + 9 level0-level1, since 0+1=1 < 2), wrongly
    skipping the cross brackets on resume.
    """
    from exact_growth import resume_pair_already_computed
    levels = [0, 0, 0, 1, 1, 1]
    start_level = 2

    reconstructed = set()
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            if resume_pair_already_computed(levels[i], levels[j], start_level):
                reconstructed.add(frozenset({i, j}))

    level0_pairs = {frozenset({i, j}) for i in range(3) for j in range(i + 1, 3)}
    cross_pairs = {frozenset({i, j}) for i in range(3) for j in range(3, 6)}

    assert reconstructed == level0_pairs, (
        f"Expected exactly the 3 level0-level0 pairs, got {reconstructed}"
    )
    assert not (reconstructed & cross_pairs), (
        "level0-level1 cross pairs must not be marked as already computed"
    )


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


def test_flat_func_big_add_factor():
    """Python 3.13 compiler-recursion regression (_make_flat_func).

    A CSE-reduced cancelled rational function typically has the shape
    coeff*(huge sum). The old _expr_to_chunked_lines only chunked
    top-level Adds, so the huge sum nested inside the Mul was emitted
    inline on one line, which Python 3.13's compiler recursion cap
    rejects (RecursionError during compilation; first seen on the
    level-3 generator {{K1,H12},{K1,K3}} on jaga). The fixed version
    hoists big-Add factors into chunked temp accumulators. This test
    builds a synthetic coeff*(2000-term sum), requires it to compile,
    and checks the numeric value against exact substitution.
    """
    import sympy as _sp
    from exact_growth import ALL_VARS, _make_flat_func

    x = _sp.Symbol("px1")
    y = _sp.Symbol("py1")
    big = _sp.Add(*[_sp.Integer(k + 1) * x ** (k % 3) * y for k in range(2000)],
                  evaluate=False)
    expr = _sp.Mul(x + 2, big, evaluate=False)

    f = _make_flat_func(expr, "_test_bigmul")
    args = [3.0 if str(v) == "px1" else (0.5 if str(v) == "py1" else 0.0)
            for v in ALL_VARS]
    got = float(f(*args))
    exact = float(expr.xreplace({x: _sp.Rational(3), y: _sp.Rational(1, 2)}))
    rel = abs(got - exact) / abs(exact)
    assert rel < 1e-9, f"flat func numeric mismatch: {got} vs {exact}"


if __name__ == "__main__":
    tests = [
        ("SymPy version", test_sympy_version),
        ("Resume pair reconstruction (E1)", test_resume_pair_reconstruction),
        ("Flat-func big-Add-factor chunking (py3.13)", test_flat_func_big_add_factor),
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
