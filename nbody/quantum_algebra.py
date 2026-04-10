#!/usr/bin/env python3
"""
Quantum Commutator Algebra Engine
==================================

Extends the classical Poisson algebra framework to quantum mechanics by
replacing {f,g} with the Moyal bracket [f,g]/(i*hbar), which for polynomial
observables terminates at finite order in hbar.

The Moyal bracket for canonical variables (q_k, p_k) is:

    [f, g] = sum_{n=1}^infty (i*hbar/2)^n / n! *
             sum_{s} (-1)^{|s|} (d^n f / prod dq_k^{s_k} dp_k^{n_k-s_k})
                               * (d^n g / prod dq_k^{n_k-s_k} dp_k^{s_k})

For single degree of freedom this simplifies to:
    [f, g] = sum_{n=1}^N (i*hbar)^n / n! *
             sum_{k=0}^n (-1)^k C(n,k) * (d^n f / dq^k dp^{n-k})
                                        * (d^n g / dq^{n-k} dp^k)

where N = min(deg_p(f) + deg_p(g), deg_q(f) + deg_q(g)).

For multiple degrees of freedom, the sum runs over multi-indices.

Usage
-----
    from quantum_algebra import QuantumNBodyAlgebra

    alg = QuantumNBodyAlgebra(n_bodies=3, d_spatial=2, potential="1/r")
    H12, H13, H23 = alg.hamiltonian_list
    comm = alg.commutator(H12, H13)  # includes hbar terms
"""

import os
import sys
from time import time
from itertools import combinations, product
from math import comb as binomial

import sympy as sp
from sympy import (Symbol, symbols, diff, Integer, Rational, cancel, expand,
                   I as sp_I, Add)

os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_growth_nbody import NBodyAlgebra, COORD_LABELS


hbar_sym = Symbol('hbar')


class QuantumNBodyAlgebra(NBodyAlgebra):
    """Quantum commutator algebra for the N-body problem.

    Extends NBodyAlgebra by replacing the Poisson bracket with the
    Moyal bracket (quantum commutator divided by i*hbar at leading order).

    The generators are stored as polynomials in q, p, u, hbar where hbar
    is a formal parameter. At hbar=0, everything reduces to the classical
    Poisson algebra.
    """

    def __init__(self, n_bodies=3, d_spatial=2, potential="1/r",
                 masses=None, charges=None, potential_params=None,
                 checkpoint_dir=None, external_potential=None):
        super().__init__(
            n_bodies=n_bodies, d_spatial=d_spatial, potential=potential,
            masses=masses, charges=charges, potential_params=potential_params,
            checkpoint_dir=checkpoint_dir, external_potential=external_potential)

        self.hbar = hbar_sym
        self.all_vars = list(self.all_vars) + [self.hbar]

    def _max_derivative_order(self, f, g):
        """Determine the maximum derivative order needed for the Moyal bracket.

        The Moyal series terminates when all higher p-derivatives vanish.
        For expressions polynomial in p, this is the sum of p-degrees.
        The q-derivatives (through chain rule with u) never terminate,
        but they only appear paired with p-derivatives in the Moyal formula,
        so the p-degree controls termination.
        """
        max_p = 0
        for p in self.p_vars:
            try:
                df = sp.degree(expand(f), p)
            except Exception:
                df = 0
            try:
                dg = sp.degree(expand(g), p)
            except Exception:
                dg = 0
            max_p = max(max_p, df + dg)

        return max_p

    def _moyal_bracket_term(self, f, g, n):
        """Compute the n-th order term of the Moyal star commutator.

        The star product is f*g = f * exp(i*hbar/2 * P) * g where
        P = sum_k (d/dq_k^L d/dp_k^R - d/dp_k^L d/dq_k^R).

        The n-th term of f*g is:
            (i*hbar/2)^n / n! * P^n(f,g)

        where P^n distributes n applications of the bidifferential
        operator. The star commutator [f,g] = f*g - g*f picks up
        only the terms where P^n is antisymmetric, i.e., the sign
        alternates: P^n(f,g) - P^n(g,f) = 2*P^n(f,g) when n is odd,
        and 0 when n is even.

        So [f,g] = 2 * sum_{m=0} (i*hbar/2)^{2m+1} / (2m+1)! * P^{2m+1}(f,g)
                 = sum_{m=0} i*hbar * (i*hbar/2)^{2m} / (2m+1)! * ... * 2 * P^{2m+1}(f,g)

        We compute the n-th star product term and return it.
        For the commutator, only odd n contributes (doubled).
        """
        n_dof = self.n_q

        P_n_fg = self._bidiff_power_n(f, g, n)

        if P_n_fg == 0:
            return Integer(0)

        prefactor = (sp_I * self.hbar / 2)**n / sp.factorial(n)
        return expand(prefactor * P_n_fg)

    def _bidiff_power_n(self, f, g, n):
        """Compute P^n(f,g) where P = sum_k (d_qk^L d_pk^R - d_pk^L d_qk^R).

        P^n distributes as a multinomial over the DOFs. Each application
        of P on DOF k does either (d_qk f)(d_pk g) or -(d_pk f)(d_qk g).

        P^n(f,g) = sum over sequences (k_1,...,k_n) in {1..K} and
                   (s_1,...,s_n) in {0,1}:
                   prod_j [if s_j=0: (d_q_{k_j} f_j)(d_p_{k_j} g_j)
                           if s_j=1: -(d_p_{k_j} f_j)(d_q_{k_j} g_j)]

        But derivatives accumulate on f and g independently, so we can
        organize by multi-index: for each DOF k, let a_k = # times
        (d_qk on f, d_pk on g) was chosen and b_k = # times
        (d_pk on f, d_qk on g) was chosen. Then a_k + b_k = n_k
        (times DOF k was selected) and sum_k n_k = n.

        P^n(f,g) = n! / prod_k(n_k!) * sum over (a,b) with a_k+b_k=n_k:
                   (-1)^{sum b_k} * prod_k C(n_k,a_k) *
                   (d^{sum a_k+b_k}_q,p f) * (d^{sum a_k+b_k}_q,p g)

        where f is differentiated a_k times by q_k and b_k times by p_k,
        and g is differentiated a_k times by p_k and b_k times by q_k.
        """
        n_dof = self.n_q

        result = Integer(0)

        for partition in self._partitions_of_n(n, n_dof):
            multinomial_coeff = sp.factorial(n)
            for n_k in partition:
                multinomial_coeff //= sp.factorial(n_k)

            for ab_choices in self._ab_choices(partition):
                sign = (-1) ** sum(b for _, b in ab_choices)
                binom_prod = 1
                for n_k, (a_k, b_k) in zip(partition, ab_choices):
                    binom_prod *= binomial(n_k, a_k)

                df = f
                dg = g
                for dof_idx, (a_k, b_k) in enumerate(ab_choices):
                    q_k = self.q_vars[dof_idx]
                    p_k = self.p_vars[dof_idx]
                    for _ in range(a_k):
                        df = self.total_deriv(df, q_k)
                        if df == 0:
                            break
                    if df == 0:
                        break
                    for _ in range(b_k):
                        df = diff(df, p_k)
                        if df == 0:
                            break
                    if df == 0:
                        break
                    for _ in range(a_k):
                        dg = diff(dg, p_k)
                        if dg == 0:
                            break
                    if dg == 0:
                        break
                    for _ in range(b_k):
                        dg = self.total_deriv(dg, q_k)
                        if dg == 0:
                            break
                    if dg == 0:
                        break

                if df == 0 or dg == 0:
                    continue

                result += sign * multinomial_coeff * binom_prod * df * dg

        return result

    def _partitions_of_n(self, n, k):
        """Generate all ways to distribute n among k bins (compositions)."""
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for rest in self._partitions_of_n(n - i, k - 1):
                yield (i,) + rest

    def _ab_choices(self, partition):
        """For each n_k in partition, generate all (a_k, b_k) with a_k+b_k=n_k."""
        if not partition:
            yield []
            return
        n_k = partition[0]
        rest = partition[1:]
        for a_k in range(n_k + 1):
            b_k = n_k - a_k
            for rest_choices in self._ab_choices(rest):
                yield [(a_k, b_k)] + rest_choices

    def _mixed_deriv(self, expr, q, nq, p, np_):
        """Compute d^(nq+np) expr / dq^nq dp^np using chain rule for q."""
        result = expr
        for _ in range(np_):
            result = diff(result, p)
            if result == 0:
                return Integer(0)
        for _ in range(nq):
            result = self.total_deriv(result, q)
            if result == 0:
                return Integer(0)
        return result

    def commutator(self, f, g):
        """Compute [f, g] / (i*hbar) via the Moyal star commutator.

        The star commutator [f,g] = f*g - g*f only has contributions
        from odd powers of the bidifferential operator (even powers
        are symmetric and cancel). So:

            [f,g] = 2 * sum_{m=0}^M (i*hbar/2)^{2m+1} / (2m+1)! * P^{2m+1}(f,g)

        Dividing by i*hbar gives:

            [f,g]/(i*hbar) = {f,g} + (hbar^2/24)*P^3(f,g) + ...

        which is real and reduces to the Poisson bracket at hbar=0.
        """
        max_n = self._max_derivative_order(f, g)

        if max_n == 0:
            return Integer(0)

        raw_comm = Integer(0)
        for n in range(1, max_n + 1, 2):
            term_n = self._moyal_bracket_term(f, g, n)
            if term_n != 0:
                raw_comm += 2 * term_n

        reduced = expand(raw_comm / (sp_I * self.hbar))
        return reduced

    def raw_commutator(self, f, g):
        """Compute the raw commutator [f, g] = f*g - g*f."""
        max_n = self._max_derivative_order(f, g)

        if max_n == 0:
            return Integer(0)

        result = Integer(0)
        for n in range(1, max_n + 1, 2):
            term_n = self._moyal_bracket_term(f, g, n)
            if term_n != 0:
                result += 2 * term_n

        return expand(result)

    def commutator_diffop(self, f, g, test_vars=None):
        """Cross-check: compute [f,g] via differential operator representation.

        Replace p_k -> -i*hbar * d/dq_k, apply f(g(phi)) - g(f(phi)) for a
        test monomial phi, then read off the operator.

        This only works for expressions polynomial in p (not involving u).
        Used for validation at low levels.
        """
        if test_vars is None:
            test_vars = self.q_vars

        phi = Integer(1)
        for q in test_vars:
            phi *= q

        def apply_op(op_expr, func):
            """Apply quantum operator to function by substituting p -> -i*hbar*d/dq."""
            expanded = expand(op_expr)
            terms = Add.make_args(expanded)
            result = Integer(0)
            for term in terms:
                coeff_part = term
                deriv_func = func
                for k, p_k in enumerate(self.p_vars):
                    q_k = self.q_vars[k]
                    p_deg = sp.degree(sp.Poly(term, p_k)) if term.has(p_k) else 0
                    if p_deg > 0:
                        coeff_part = coeff_part.subs(p_k, 1) / p_k**(p_deg - 1)
                        coeff_part = coeff_part.subs(p_k, 1)
                        for _ in range(p_deg):
                            deriv_func = diff(deriv_func, q_k)
                        coeff_part *= (-sp_I * self.hbar) ** p_deg
                result += expand(coeff_part * deriv_func)
            return result

        fg_phi = apply_op(f, apply_op(g, phi))
        gf_phi = apply_op(g, apply_op(f, phi))

        return expand(fg_phi - gf_phi)

    def simplify_generator(self, expr):
        """Cancel and collect terms, keeping hbar explicit."""
        return cancel(expand(expr))
