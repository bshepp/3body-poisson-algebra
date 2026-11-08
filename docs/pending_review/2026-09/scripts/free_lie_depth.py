"""Depth filtration of the free Lie algebra on k generators, realised inside the free
associative algebra (commutators). A_0 = gens, A_n = A_{n-1} + [A_{n-1}, A_{n-1}].
Reports dim A_n.  Rank is computed per multidegree (the free Lie algebra is multigraded)."""
import sys, itertools
from collections import defaultdict
from fractions import Fraction
k = int(sys.argv[1]); MAXD = int(sys.argv[2])
P = 2147483647
def mul(a, b):
    out = defaultdict(int)
    for wa, ca in a.items():
        for wb, cb in b.items():
            out[wa + wb] = (out[wa + wb] + ca * cb) % P
    return {w: c for w, c in out.items() if c}
def br(a, b):
    ab = mul(a, b); ba = mul(b, a)
    out = dict(ab)
    for w, c in ba.items():
        out[w] = (out.get(w, 0) - c) % P
    return {w: c for w, c in out.items() if c}
def mdeg(x):
    w = next(iter(x)); return tuple(w.count(i) for i in range(k))
def reduce_basis(elems):
    """Return a basis (list of elements) of span(elems), grouped by multidegree."""
    groups = defaultdict(list)
    for e in elems:
        if e: groups[mdeg(e)].append(e)
    basis = []
    for md, es in groups.items():
        piv = {}  # pivot word -> reduced vector
        kept = []
        for e in es:
            v = dict(e)
            changed = True
            while True:
                lead = next((w for w in sorted(v) if w in piv), None)
                if lead is None: break
                f = v[lead]; pv = piv[lead]
                for w, c in pv.items():
                    v[w] = (v.get(w, 0) - f * c) % P
                v = {w: c for w, c in v.items() if c}
            if v:
                lw = min(v); inv = pow(v[lw], -1, P)
                v = {w: c * inv % P for w, c in v.items()}
                # back-substitute into existing pivots is unnecessary for rank
                piv[lw] = v; kept.append(e)
        basis += kept
    return basis
gens = [{(i,): 1} for i in range(k)]
A = reduce_basis(gens); frontier = list(A)
print("depth 0:", len(A), flush=True)
for d in range(1, MAXD + 1):
    cands = []
    old = [x for x in A if x not in frontier]
    for i, x in enumerate(frontier):
        for y in frontier[i+1:]:
            cands.append(br(x, y))
        for y in old:
            cands.append(br(x, y))
    newA = reduce_basis(A + cands)
    frontier = newA[len(A):] if False else [x for x in newA if not any(x is y for y in A)]
    A = newA
    print(f"depth {d}: {len(A)}  (candidates {len(cands)})", flush=True)
from collections import Counter
print("free per-length:", sorted(Counter(len(next(iter(x))) for x in A).items()))
