"""Truncated multivariate Taylor jets over F_p, vectorized over points.

A jet of order m at a point z is the array of Taylor coefficients
f(z + eps) = sum_{|alpha| <= m} c_alpha eps^alpha, truncated to total
degree m. Every coefficient is stored as a length-S vector of residues
mod p (one entry per sample point), so all jet arithmetic is numpy
vector arithmetic across the whole point batch at once.

This is the kernel that lets nested Poisson-bracket trees be evaluated
numerically without ever constructing symbolic expressions: a bracket
consumes one derivative order, so a level-L tree needs only order-L
jets of the base Hamiltonians (a few hundred to a few thousand
coefficients), while the tree count can be in the millions.

All arithmetic is exact mod p (p < 2^31; uint64 intermediates).
"""
from itertools import combinations_with_replacement

import numpy as np


class JetContext:
    """Monomial tables, truncated-product tables, and derivative maps
    for jets in `nv` variables up to total order `max_order`."""

    def __init__(self, nv, max_order, prime):
        self.nv = nv
        self.max_order = max_order
        self.prime = prime
        self._p = np.uint64(prime)

        # monomial list per order: all exponent tuples with |alpha| <= m,
        # indexed consistently (order-m table is a prefix extension of
        # order-(m-1): same layout, graded by total degree).
        monos = [tuple([0] * nv)]
        for deg in range(1, max_order + 1):
            for c in combinations_with_replacement(range(nv), deg):
                e = [0] * nv
                for v in c:
                    e[v] += 1
                monos.append(tuple(e))
        self.monos = monos
        self.index = {m: i for i, m in enumerate(monos)}
        deg_of = [sum(m) for m in monos]
        # n_coeffs[m] = number of monomials of total degree <= m
        self.n_coeffs = [sum(1 for d in deg_of if d <= m)
                         for m in range(max_order + 1)]
        self.deg_of = deg_of

        # multiplication tables: for output order m, a list over ia of
        # (ia, ib_array, iout_array) covering all pairs with
        # deg(a)+deg(b) <= m. Built lazily per requested m.
        self._mult_tables = {}

        # derivative tables: for var v, arrays (i_from, i_to, factor)
        # with monomial(i_from) having e_v >= 1 and
        # monomial(i_to) = monomial(i_from) - e_v.
        self._deriv_tables = []
        for v in range(nv):
            frm, to, fac = [], [], []
            for i, mo in enumerate(monos):
                if mo[v] >= 1:
                    lower = list(mo)
                    lower[v] -= 1
                    frm.append(i)
                    to.append(self.index[tuple(lower)])
                    fac.append(mo[v])
            self._deriv_tables.append(
                (np.array(frm, dtype=np.intp),
                 np.array(to, dtype=np.intp),
                 np.array(fac, dtype=np.uint64)))

    def mult_table(self, m):
        """Truncated-product table for output order m."""
        if m in self._mult_tables:
            return self._mult_tables[m]
        nm = self.n_coeffs[m]
        rows = []
        for ia in range(nm):
            da = self.deg_of[ia]
            ibs, iouts = [], []
            for ib in range(nm):
                if da + self.deg_of[ib] > m:
                    continue
                s = tuple(x + y for x, y in
                          zip(self.monos[ia], self.monos[ib]))
                ibs.append(ib)
                iouts.append(self.index[s])
            rows.append((ia,
                         np.array(ibs, dtype=np.intp),
                         np.array(iouts, dtype=np.intp)))
        self._mult_tables[m] = rows
        return rows

    # ---- jet constructors -------------------------------------------

    def zero(self, m, S):
        return np.zeros((self.n_coeffs[m], S), dtype=np.uint64)

    def const(self, value_vec, m):
        """Jet of a constant-per-point value (S-vector)."""
        out = np.zeros((self.n_coeffs[m], len(value_vec)),
                       dtype=np.uint64)
        out[0] = value_vec % self._p
        return out

    def variable(self, v, value_vec, m):
        """Jet of coordinate v: value + eps_v."""
        out = self.const(value_vec, m)
        if m >= 1:
            e = [0] * self.nv
            e[v] = 1
            out[self.index[tuple(e)]] = 1
        return out

    # ---- jet arithmetic (arrays are (n_coeffs(m), S)) ---------------

    def add(self, a, b):
        n = min(a.shape[0], b.shape[0])
        out = a[:n] + b[:n]
        return out % self._p

    def sub(self, a, b):
        n = min(a.shape[0], b.shape[0])
        return (a[:n] + (self._p - b[:n])) % self._p

    def scale(self, a, k):
        return (a * np.uint64(k % self.prime)) % self._p

    def mult(self, a, b, m):
        """Truncated product to order m."""
        nm = self.n_coeffs[m]
        S = a.shape[1]
        out = np.zeros((nm, S), dtype=np.uint64)
        for ia, ibs, iouts in self.mult_table(m):
            if ia >= a.shape[0]:
                break
            av = a[ia]
            if not av.any():
                continue
            keep = ibs < b.shape[0]
            ib = ibs[keep]
            io = iouts[keep]
            out[io] = (out[io] + av * b[ib]) % self._p
        return out

    def deriv(self, a, v, m_out):
        """Partial derivative wrt var v, output truncated to m_out."""
        nm = self.n_coeffs[m_out]
        S = a.shape[1]
        out = np.zeros((nm, S), dtype=np.uint64)
        frm, to, fac = self._deriv_tables[v]
        keep = (frm < a.shape[0]) & (to < nm)
        f, t, c = frm[keep], to[keep], fac[keep]
        out[t] = (a[f] * c[:, None]) % self._p
        return out

    def truncate(self, a, m):
        return a[:self.n_coeffs[m]]
