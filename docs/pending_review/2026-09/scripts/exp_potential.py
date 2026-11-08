"""N=3, d=1 pairwise algebra for a pure exponential potential V(r)=exp(-r) (Toda/Morse-type),
on the chamber x1>x2>x3, where r_ij = x_i - x_j.  w_ij = exp(-(x_i-x_j)).
Everything is polynomial in (p, w). Physical relation: w13 = w12*w23."""
import sys, random
import sympy as sp
p = sp.symbols('p1:4'); w12, w13, w23 = W = sp.symbols('w12 w13 w23')
pairs = {(1,2): w12, (1,3): w13, (2,3): w23}
def dx(f, k):  # d/dx_k, with dw_ij/dx_i = -w_ij, dw_ij/dx_j = +w_ij
    out = 0
    for (i, j), w in pairs.items():
        if k == i: out += -w * sp.diff(f, w)
        if k == j: out += w * sp.diff(f, w)
    return out
def pb(f, g):
    return sp.expand(sum(dx(f, k) * sp.diff(g, p[k-1]) - sp.diff(f, p[k-1]) * dx(g, k) for k in (1, 2, 3)))
sign = float(sys.argv[2]) if len(sys.argv) > 2 else 1
H = [p[i-1]**2/2 + p[j-1]**2/2 + sp.Integer(int(sign)) * w for (i, j), w in pairs.items()]
MAXL = int(sys.argv[1])
levels = [H]; allg = list(H)
for L in range(1, MAXL + 1):
    fs = len(allg) - len(levels[-1]); new = []
    for i in range(len(allg)):
        for j in range(max(i+1, fs), len(allg)):
            b = pb(allg[i], allg[j])
            if b != 0: new.append(b)
    levels.append(new); allg += new
P = 2147483647; rng = random.Random(3)
V = list(p) + list(W)
polys = [sp.Poly(g, *V) for g in allg]
def ev(poly, pt):
    return int(poly.eval(dict(zip(V, pt)))) % P if False else sum(int(sp.Rational(c).p) * pow(int(sp.Rational(c).q), -1, P) * _mono(m, pt) for m, c in poly.terms()) % P
def _mono(m, pt):
    r = 1
    for e, v in zip(m, pt):
        if e: r = r * pow(v, e, P) % P
    return r
def rank(rows):
    M = [r[:] for r in rows]; rk = 0
    for c in range(len(M[0])):
        piv = next((i for i in range(rk, len(M)) if M[i][c]), None)
        if piv is None: continue
        M[rk], M[piv] = M[piv], M[rk]; inv = pow(M[rk][c], -1, P); M[rk] = [x*inv % P for x in M[rk]]
        for i in range(len(M)):
            if i != rk and M[i][c]:
                f = M[i][c]; M[i] = [(a - f*b) % P for a, b in zip(M[i], M[rk])]
        rk += 1
    return rk
S = int(sys.argv[3]) if len(sys.argv) > 3 else 300
def pts(physical):
    out = []
    for _ in range(S):
        pp = [rng.randrange(1, P) for _ in range(3)]
        if physical:
            e = [rng.randrange(1, P) for _ in range(3)]
            ww = [e[1]*pow(e[0], -1, P) % P, e[2]*pow(e[0], -1, P) % P, e[2]*pow(e[1], -1, P) % P]
        else:
            ww = [rng.randrange(1, P) for _ in range(3)]
        out.append(pp + ww)
    return out
cum = [sum(len(l) for l in levels[:k+1]) for k in range(len(levels))]
for phys in (False, True):
    PT = pts(phys)
    E = [[ev(q, pt) for pt in PT] for q in polys]
    print("physical" if phys else "formal  ", [rank(E[:c]) for c in cum], flush=True)
