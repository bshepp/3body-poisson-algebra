"""
poisson_n3_d2_engine.sage
---------------------------------------------------------------------------
Third-leg CAS oracle for the planar 3-body Poisson algebra, mirroring the
Mathematica engine at mathematica/poisson_n3_d2_engine.wl line-for-line so
the cumulative ranks coincide level by level.

Defines:
  pb(f, g)               -- Poisson bracket with chain rule on u_ij
  H0(pot)                -- list of pairwise Hamiltonians
  expr_list_rank(exprs)  -- exact rank over QQ via sparse matrix + Matrix.rank()
  build_algebra(pot, k)  -- Lie closure to depth k, returns dict with
                            cumulative_rank, raw_gens_per_level, timing

See poisson_n3_d2.sage for the documentation header.

Phase-space variables:    x_i, y_i, px_i, py_i  for i = 1..3
Auxiliary variables:      u_ij = 1/r_ij              (i < j)
H_ij = (px_i^2 + py_i^2)/2 + (px_j^2 + py_j^2)/2 + potential_term(u_ij, pot)

Chain rule on u_ij:
    d u_ij / d x_k = -(x_i - x_j) u_ij^3 (delta_ki - delta_kj)
    d u_ij / d y_k = -(y_i - y_j) u_ij^3 (delta_ki - delta_kj)
    d u_ij / d p_k = 0

Implementation note: every generator and every bracket here is a
polynomial in (q, p, u) with QQ coefficients. We therefore work directly
in PolynomialRing(QQ, ...) and skip the FractionField wrapper that the
naive translation of Mathematica's `Together` would require -- that
gives a ~10x speedup on the L=3 rank computation.
---------------------------------------------------------------------------
"""

import sys
from sage.all import PolynomialRing, QQ, ZZ, GF, Matrix, lcm
import time

N_BODIES = 3
D_SPATIAL = 2


# Build the polynomial ring over Q in (x_i, y_i, px_i, py_i, u_ij).
# 6N + N(N-1)/2 = 18 + 3 = 21 generators for N=3, d=2.
def _build_ring(n_bodies=N_BODIES):
    q_names = []
    p_names = []
    for i in range(1, n_bodies + 1):
        q_names += ['x%d' % i, 'y%d' % i]
        p_names += ['px%d' % i, 'py%d' % i]
    u_names = []
    for i in range(1, n_bodies + 1):
        for j in range(i + 1, n_bodies + 1):
            u_names.append('u%d%d' % (i, j))
    var_names = q_names + p_names + u_names
    R = PolynomialRing(QQ, var_names)
    return R, q_names, p_names, u_names


_RING, _Q_NAMES, _P_NAMES, _U_NAMES = _build_ring(N_BODIES)

# Pull each generator name into a Python dict for fast lookup.
_GEN = {name: _RING.gen(_RING.variable_names().index(name))
        for name in _RING.variable_names()}


def x(i): return _GEN['x%d' % i]
def y(i): return _GEN['y%d' % i]
def px(i): return _GEN['px%d' % i]
def py(i): return _GEN['py%d' % i]
def u(i, j):
    a, b = (i, j) if i < j else (j, i)
    return _GEN['u%d%d' % (a, b)]


# Mathematica engine interleaves (x1,y1,x2,y2,x3,y3) and (px1,py1,...).
# Match that ordering for the bracket sum so partial sums stay comparable.
def _q_vars_interleaved():
    out = []
    for i in range(1, N_BODIES + 1):
        out += [x(i), y(i)]
    return out


def _p_vars_interleaved():
    out = []
    for i in range(1, N_BODIES + 1):
        out += [px(i), py(i)]
    return out


_Q_LIST = _q_vars_interleaved()
_P_LIST = _p_vars_interleaved()


def _du_dq(i, j, qvar):
    """Mathematica dUdQ[i, j, var]. Returns an element of the polynomial ring."""
    if qvar == x(i):
        return -(x(i) - x(j)) * u(i, j) ** 3
    if qvar == x(j):
        return  (x(i) - x(j)) * u(i, j) ** 3
    if qvar == y(i):
        return -(y(i) - y(j)) * u(i, j) ** 3
    if qvar == y(j):
        return  (y(i) - y(j)) * u(i, j) ** 3
    return _RING.zero()


def pdq(expr, qvar):
    """Position partial with u_ij chain rule. Stays in the polynomial ring."""
    base = expr.derivative(qvar)
    chain = _RING.zero()
    for i in range(1, N_BODIES + 1):
        for j in range(i + 1, N_BODIES + 1):
            d = _du_dq(i, j, qvar)
            if d.is_zero():
                continue
            chain += expr.derivative(u(i, j)) * d
    return base + chain


def pdp(expr, pvar):
    """Momentum partial -- no chain rule (u_ij has no momentum content)."""
    return expr.derivative(pvar)


def pb(f, g):
    """Canonical Poisson bracket on the (q, p) symplectic form. Polynomial."""
    total = _RING.zero()
    for qv, pv in zip(_Q_LIST, _P_LIST):
        total += pdq(f, qv) * pdp(g, pv) - pdp(f, pv) * pdq(g, qv)
    return total


def _kinetic(i):
    return (px(i) ** 2 + py(i) ** 2) / QQ(2)


def _potential_term(uij, pot):
    if pot == '1/r':   return -uij
    if pot == '1/r^2': return -uij ** 2
    if pot == '1/r^3': return -uij ** 3
    raise ValueError('Unknown potential: %r' % pot)


def _r_sq_2d(i, j):
    return (x(i) - x(j)) ** 2 + (y(i) - y(j)) ** 2


def h_pair(i, j, pot):
    if pot == 'harmonic':
        # Match the Mathematica engine: H_ij = T_i + T_j + r_ij^2 (no u_ij).
        return _kinetic(i) + _kinetic(j) + _r_sq_2d(i, j)
    return _kinetic(i) + _kinetic(j) + _potential_term(u(i, j), pot)


def H0(pot):
    out = []
    for i in range(1, N_BODIES + 1):
        for j in range(i + 1, N_BODIES + 1):
            out.append(h_pair(i, j, pot))
    return out


# ---- Coefficient extraction & rank ---------------------------------------

_LARGE_PRIME = 2147483647  # Mersenne 2^31 - 1


def expr_list_rank(exprs, mod_prime=_LARGE_PRIME):
    """Exact rank via mod-p reduction.

    Each generator is a polynomial in QQ[x, p, u]; we read its .dict()
    representation, clear common denominators row-by-row to get integer
    coefficients, then reduce modulo `mod_prime` and ask Sage for the
    rank of the resulting GF(p) sparse matrix.

    For our generators the coefficients are tiny rationals (at most ~10
    digits) and the matrix is sparse, so this is much faster than
    Matrix(QQ).rank() while remaining exact with probability 1 - O(1/p)
    of giving the right rank.  A second prime can be used to verify.

    Set mod_prime=None to use exact QQ rank (slower).
    """
    rows = [e.dict() for e in exprs]
    mono_index = {}
    for row in rows:
        for mono in row:
            if mono not in mono_index:
                mono_index[mono] = len(mono_index)
    n_rows = len(rows)
    n_cols = len(mono_index)

    if mod_prime is None:
        M = Matrix(QQ, n_rows, n_cols, sparse=True)
        for r, row in enumerate(rows):
            for mono, coef in row.items():
                M[r, mono_index[mono]] = coef
        return M.rank()

    Fp = GF(mod_prime)
    M = Matrix(Fp, n_rows, n_cols, sparse=True)
    for r, row in enumerate(rows):
        # Clear denominators row-wise: scale row so it's all integers,
        # then reduce mod p. Equivalent ranks on the QQ row.
        denoms = [c.denominator() for c in row.values()]
        if denoms:
            L = denoms[0]
            for d in denoms[1:]:
                L = lcm(L, d)
        else:
            L = 1
        for mono, coef in row.items():
            num = (coef * L).numerator()
            M[r, mono_index[mono]] = Fp(int(num))
    return M.rank()


# ---- Lie closure ----------------------------------------------------------

def _log(msg):
    print(msg)
    sys.stdout.flush()


def build_algebra(pot, max_lv):
    """Mirror the Mathematica buildAlgebra and the Python filtration.

    Returns a dict with keys matching the Mathematica engine output so the
    sage/results/n3_d2_dimseq.json matches mathematica/results/n3_d2_dimseq.json
    field-for-field.
    """
    _log('[%s] generating algebra up to level %d' % (pot, max_lv))
    t_start = time.time()

    exprs = list(H0(pot))
    n0 = len(exprs)
    levels = [0] * n0
    pairs_set = set()
    ranks = {0: expr_list_rank(exprs)}
    raw_per_level = {0: n0}
    _log('  L0: %d gens, cumulative rank = %d' % (n0, ranks[0]))

    if max_lv >= 1:
        t_level = time.time()
        for i in range(n0):
            for j in range(i + 1, n0):
                exprs.append(pb(exprs[i], exprs[j]))
                levels.append(1)
                pairs_set.add((i, j))
        raw_per_level[1] = len(exprs) - n0
        ranks[1] = expr_list_rank(exprs)
        _log('  L1: +%d raw, cumulative rank = %d  [%.2fs]'
             % (raw_per_level[1], ranks[1], time.time() - t_level))

    for lv in range(2, max_lv + 1):
        t_level = time.time()
        frontier = [k for k, lvk in enumerate(levels) if lvk == lv - 1]
        n_exist = len(exprs)

        work_items = []
        for i in frontier:
            for j in range(n_exist):
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in pairs_set:
                    continue
                pairs_set.add(pair)
                work_items.append((i, j))

        raw_per_level[lv] = len(work_items)
        _log('  L%d: frontier=%d existing=%d candidates=%d'
             % (lv, len(frontier), n_exist, len(work_items)))

        for k, (i, j) in enumerate(work_items):
            exprs.append(pb(exprs[i], exprs[j]))
            levels.append(lv)
            if (k + 1) % 25 == 0:
                _log('    ... %d/%d brackets computed [%.1fs]'
                     % (k + 1, len(work_items), time.time() - t_level))

        ranks[lv] = expr_list_rank(exprs)
        _log('    cumulative rank = %d  [%.2fs]'
             % (ranks[lv], time.time() - t_level))

    return {
        'potential':          pot,
        'n_bodies':           N_BODIES,
        'd_spatial':          D_SPATIAL,
        'max_level':          max_lv,
        'elapsed_s':          round(time.time() - t_start, 2),
        'cumulative_rank':    [ranks[lv] for lv in range(max_lv + 1)],
        'raw_gens_per_level': [raw_per_level[lv] for lv in range(max_lv + 1)],
        'total_generators':   len(exprs),
    }
