"""Depth-filtration dims d(0..4) of the N=3 pairwise algebra (V=-g/r, generic masses), mod p.
d=1 (all orderings, i.e. the full phase space) or d=2 (planar, u=1/r on the variety).
Uses (i) the word-length grading (weights x:2, p:-1, u:-2 make every Lie word of length L
homogeneous of weight 1-3L, so the rank splits into length blocks) and (ii) the
derivative pipeline: {f,g}(pt) = sum_k Dq_k f(pt) dp_k g(pt) - dp_k f(pt) Dq_k g(pt),
done in exact GF(p) arithmetic instead of float64.
Rank of an evaluation matrix over GF(p) is a certified LOWER bound on the true dimension
(and equals it with high probability)."""
import sys, random, time
import numpy as np
from flint import nmod_mpoly_ctx, nmod_mat
D = int(sys.argv[1]); S = int(sys.argv[2]); P = 2147483647
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 17)
if D == 1:
    Qn = [['x1'],['x2'],['x3']]; Pn = [['p1'],['p2'],['p3']]
else:
    Qn = [['x1','y1'],['x2','y2'],['x3','y3']]; Pn = [['px1','py1'],['px2','py2'],['px3','py3']]
pairs = [(0,1),(0,2),(1,2)]; UN = {(0,1):'u12',(0,2):'u13',(1,2):'u23'}
names = tuple(sum(Qn,[]) + sum(Pn,[]) + list(UN.values()))
ctx = nmod_mpoly_ctx.get(names, modulus=P); G = dict(zip(names, ctx.gens()))
m = [rng.randrange(2,P) for _ in range(3)]; gc = [rng.randrange(2,P) for _ in range(3)]
T = [pow(2*m[i],-1,P)*sum(G[n]**2 for n in Pn[i]) for i in range(3)]
H = [T[i]+T[j]-gc[k]*G[UN[(i,j)]] for k,(i,j) in enumerate(pairs)]
def Dq(f,b,c):
    out = f.derivative(Qn[b][c])
    for (i,j),un in UN.items():
        if b in (i,j):
            fu = f.derivative(un)
            if fu.is_zero(): continue
            t = fu*(G[Qn[i][c]]-G[Qn[j][c]])*G[un]**3
            out = out - t if b == i else out + t
    return out
def pb(f,h):
    r = ctx.from_dict({})
    for b in range(3):
        for c in range(D):
            r += Dq(f,b,c)*h.derivative(Pn[b][c]) - f.derivative(Pn[b][c])*Dq(h,b,c)
    return r
def sqrt_mod(a):
    r = pow(a,(P+1)//4,P); return r if r*r % P == a % P else None
CH = [(1,1,1),(-1,1,1),(1,-1,-1),(1,1,-1),(-1,-1,1),(-1,-1,-1)]
pts = []
while len(pts) < S:
    q = [rng.randrange(P) for _ in range(3*D)]; p = [rng.randrange(P) for _ in range(3*D)]; us = []
    sg = rng.choice(CH)
    for s,(i,j) in zip(sg,pairs):
        if D == 1:
            d = (q[i]-q[j]) % P
            if d == 0: break
            us.append(s*pow(d,-1,P) % P)
        else:
            r2 = sum((q[D*i+c]-q[D*j+c])**2 for c in range(D)) % P
            r = sqrt_mod(r2) if r2 else None
            if not r: break
            us.append(pow(r,-1,P))
    if len(us) == 3: pts.append(q+p+us)
def ev(f): return np.array([int(f(*pt)) for pt in pts], dtype=np.int64)
def rank_rows(rows):
    if not rows: return 0
    return nmod_mat([[int(v) for v in r] for r in rows], P).rank()
# ---- symbolic levels 0..3 (homogeneous elements tagged with word length) ----
t0 = time.time()
A = [(h,1) for h in H]; frontier = list(A)
for lev in (1,2,3):
    cands = []
    for i,(f,lf) in enumerate(frontier):
        for (g,lg) in A:
            if any(g is ff for ff,_ in frontier[:i+1]): continue
            b = pb(f,g)
            if not b.is_zero(): cands.append((b,lf+lg))
    # select a basis per length block by evaluation rank
    newA = list(A); newF = []
    blocks = {}
    for f,l in A: blocks.setdefault(l,[]).append(ev(f))
    for b,l in cands:
        blk = blocks.setdefault(l,[]); r0 = rank_rows(blk) if blk else 0
        v = ev(b)
        if rank_rows(blk+[v]) > r0:
            blk.append(v); newA.append((b,l)); newF.append((b,l))
    A, frontier = newA, newF
    print(f"d({lev}) = {len(A)}   ({len(cands)} candidates, {time.time()-t0:.0f}s)", flush=True)
# ---- level 4 via exact derivative pipeline ----
print("evaluating derivatives of level<=3 basis ...", flush=True)
DQ = []; DP = []; VAL = []
for f,l in A:
    DQ.append([ev(Dq(f,b,c)) for b in range(3) for c in range(D)])
    DP.append([ev(f.derivative(Pn[b][c])) for b in range(3) for c in range(D)])
    VAL.append(ev(f))
print(f"  done ({time.time()-t0:.0f}s)", flush=True)
blocks = {}
for (f,l),v in zip(A,VAL): blocks.setdefault(l,[]).append(v)
nA = len(A); frontier_idx = [k for k in range(nA) if any(A[k][0] is ff for ff,_ in frontier)]
ncand = 0
for i in frontier_idx:
    for j in range(nA):
        if j in frontier_idx and j <= i: continue
        acc = np.zeros(S, dtype=np.int64)
        for k in range(3*D):
            acc = (acc + DQ[i][k]*DP[j][k] % P) % P
            acc = (acc - DP[i][k]*DQ[j][k] % P) % P
        if acc.any():
            blocks.setdefault(A[i][1]+A[j][1],[]).append(acc); ncand += 1
print(f"level-4 candidates (nonzero): {ncand}", flush=True)
tot = 0
for L in sorted(blocks):
    r = rank_rows(blocks[L]); tot += r
    flag = "  <-- SATURATED, increase S" if r >= S - 5 else ""
    print(f"  length {L:2d}: rows {len(blocks[L]):5d}  rank {r:5d}{flag}", flush=True)
print(f"d(4) = {tot}   (S={S}, {time.time()-t0:.0f}s)")
