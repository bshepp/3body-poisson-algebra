"""Exact mod-p dense linear algebra via 16-bit-limb float64 GEMMs.

Same kernel as the 2026-07-29 N=5 L3 campaign (sz_rank), validated
against FLINT nmod_mat on the N=4 L<=3 evaluation matrix (rank 1260).
Every partial sum stays below 2^53, so BLAS matmuls are exact.
"""
import numpy as np


def matmul_modp(A, B, prime):
    if A.shape[0] == 0 or B.shape[1] == 0 or A.shape[1] == 0:
        return np.zeros((A.shape[0], B.shape[1]), dtype=np.uint64)
    p = np.uint64(prime)
    A0 = (A & np.uint64(0xFFFF)).astype(np.float64)
    A1 = (A >> np.uint64(16)).astype(np.float64)
    B0 = (B & np.uint64(0xFFFF)).astype(np.float64)
    B1 = (B >> np.uint64(16)).astype(np.float64)
    hh = np.mod(np.dot(A1, B1), float(prime)).astype(np.uint64)
    hl = np.mod(np.dot(A1, B0) + np.dot(A0, B1),
                float(prime)).astype(np.uint64)
    ll = np.mod(np.dot(A0, B0), float(prime)).astype(np.uint64)
    c32 = np.uint64((1 << 32) % prime)
    c16 = np.uint64((1 << 16) % prime)
    return (hh * c32 % p + hl * c16 % p + ll) % p


def _submod(A, B, prime):
    p = np.uint64(prime)
    return (A + (p - B)) % p


def _rref_base(M, prime):
    p = np.uint64(prime)
    rows, pivs = [], []
    for i in range(M.shape[0]):
        r = M[i].copy()
        while True:
            nz = np.nonzero(r)[0]
            if nz.size == 0:
                break
            lead = int(nz[0])
            hit = None
            for j, pc in enumerate(pivs):
                if pc == lead:
                    hit = j
                    break
            if hit is None:
                inv = np.uint64(pow(int(r[lead]), -1, prime))
                r = (r * inv) % p
                for j in range(len(rows)):
                    f = rows[j][lead]
                    if f:
                        rows[j] = _submod(rows[j], (r * f) % p, prime)
                rows.append(r)
                pivs.append(lead)
                break
            r = _submod(r, (rows[hit] * r[lead]) % p, prime)
    if not rows:
        return np.zeros((0, M.shape[1]), dtype=np.uint64), []
    order = np.argsort(pivs)
    return np.stack([rows[j] for j in order]), [pivs[j] for j in order]


def rref_modp(M, prime, base=64):
    m = M.shape[0]
    if m == 0:
        return M.copy(), []
    if m <= base:
        return _rref_base(M, prime)
    half = m // 2
    RA, pivA = rref_modp(M[:half], prime, base)
    B = M[half:]
    if pivA:
        B = _submod(B, matmul_modp(B[:, pivA], RA, prime), prime)
    B = B[np.any(B, axis=1)]
    RB, pivB = rref_modp(B, prime, base)
    if pivB and RA.shape[0]:
        RA = _submod(RA, matmul_modp(RA[:, pivB], RB, prime), prime)
    R = np.concatenate([RA, RB], axis=0)
    pivs = pivA + pivB
    order = np.argsort(pivs)
    return R[order], [pivs[j] for j in order]


class IncrementalBasis:
    """Reduced row basis over F_p; add row-blocks, read cumulative rank."""

    def __init__(self, prime):
        self.prime = prime
        self.basis = None
        self.pivs = []

    def add(self, rows):
        rows = rows[np.any(rows, axis=1)] if rows.size else rows
        if rows.shape[0] == 0:
            return self.rank
        if self.basis is not None and self.pivs:
            rows = _submod(rows,
                           matmul_modp(rows[:, self.pivs], self.basis,
                                       self.prime), self.prime)
            rows = rows[np.any(rows, axis=1)]
            if rows.shape[0] == 0:
                return self.rank
        RN, pivN = rref_modp(rows, self.prime)
        if self.basis is None:
            self.basis, self.pivs = RN, pivN
        elif RN.shape[0]:
            self.basis = _submod(
                self.basis,
                matmul_modp(self.basis[:, pivN], RN, self.prime),
                self.prime)
            self.basis = np.concatenate([self.basis, RN], axis=0)
            self.pivs = self.pivs + pivN
            order = np.argsort(self.pivs)
            self.basis = self.basis[order]
            self.pivs = [self.pivs[j] for j in order]
        return self.rank

    @property
    def rank(self):
        return 0 if self.basis is None else self.basis.shape[0]
