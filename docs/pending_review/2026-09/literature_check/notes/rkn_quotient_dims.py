"""Dimensions of the graded pieces of the free Lie algebra L(A,B) and of the
quotient L(A,B)/<[B,[B,[B,A]]]> (the 'RKN' Lie algebra), computed inside the
free associative algebra over Q.  Independent check, not taken from literature.
"""
import sys
from fractions import Fraction
from itertools import product
from sympy import Matrix

GENS = ("A", "B")

def mul(p, q):
    r = {}
    for w1, c1 in p.items():
        for w2, c2 in q.items():
            w = w1 + w2
            r[w] = r.get(w, 0) + c1 * c2
    return {w: c for w, c in r.items() if c != 0}

def add(p, q, s=1):
    r = dict(p)
    for w, c in q.items():
        r[w] = r.get(w, 0) + s * c
    return {w: c for w, c in r.items() if c != 0}

def br(p, q):
    return add(mul(p, q), mul(q, p), -1)

def gen(x):
    return {x: 1}

def left_normed(word):
    # [x1,[x2,[...,[x_{n-1},x_n]]]]
    p = gen(word[-1])
    for x in reversed(word[:-1]):
        p = br(gen(x), p)
    return p

def rank(vectors, n):
    words = ["".join(w) for w in product(GENS, repeat=n)]
    idx = {w: i for i, w in enumerate(words)}
    if not vectors:
        return 0
    M = Matrix([[v.get(w, 0) for w in words] for v in vectors])
    return M.rank()

def main(nmax):
    R = br(gen("B"), br(gen("B"), br(gen("B"), gen("A"))))  # [B,[B,[B,A]]]
    print("n  dim L_n  dim I_n  dim (L/I)_n")
    for n in range(1, nmax + 1):
        Ln = [left_normed(w) for w in product(GENS, repeat=n)]
        dL = rank(Ln, n)
        if n < 4:
            dI = 0
        else:
            In = []
            for w in product(GENS, repeat=n - 4):
                p = R
                for x in reversed(w):
                    p = br(gen(x), p)
                In.append(p)
            dI = rank(In, n)
        print(n, dL, dI, dL - dI)
        sys.stdout.flush()

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
