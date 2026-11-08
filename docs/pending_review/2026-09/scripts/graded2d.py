"""Planar (d=2) N=3, V=-g/r, generic masses: word-length graded dims, formal vs physical, mod p."""
import sys, random, time
from flint import nmod_mpoly_ctx
P = 2147483647  # = 3 mod 4
names = ('x1','y1','x2','y2','x3','y3','px1','py1','px2','py2','px3','py3','u12','u13','u23')
ctx = nmod_mpoly_ctx.get(names, modulus=P)
G = dict(zip(names, ctx.gens()))
Q = [['x1','y1'],['x2','y2'],['x3','y3']]; PP = [['px1','py1'],['px2','py2'],['px3','py3']]
pairs = [(0,1),(0,2),(1,2)]; UN = {(0,1):'u12',(0,2):'u13',(1,2):'u23'}
rng = random.Random(5)
m = [rng.randrange(2,P) for _ in range(3)]; g = [rng.randrange(2,P) for _ in range(3)]
T = [pow(2*m[i],-1,P)*(G[PP[i][0]]**2+G[PP[i][1]]**2) for i in range(3)]
H = [T[i]+T[j]-g[k]*G[UN[(i,j)]] for k,(i,j) in enumerate(pairs)]
def Dq(f, b, c):
    out = f.derivative(Q[b][c])
    for (i,j),un in UN.items():
        if b in (i,j):
            d = G[Q[i][c]]-G[Q[j][c]]
            fu = f.derivative(un)
            if b == i: out -= fu*d*G[un]**3
            else: out += fu*d*G[un]**3
    return out
def pb(f,h):
    r = ctx.from_dict({})
    for b in range(3):
        for c in range(2):
            r += Dq(f,b,c)*h.derivative(PP[b][c]) - f.derivative(PP[b][c])*Dq(h,b,c)
    return r
class Basis:
    def __init__(s): s.piv={}; s.elems=[]
    def add(s, vec, elem):
        v=dict(vec)
        while v:
            lead=max(v)
            if lead not in s.piv: break
            f=v[lead]
            for k,c in s.piv[lead].items():
                nv=(v.get(k,0)-f*c)%P
                if nv: v[k]=nv
                else: v.pop(k,None)
        if not v: return False
        lead=max(v); inv=pow(v[lead],-1,P); s.piv[lead]={k:c*inv%P for k,c in v.items()}; s.elems.append(elem); return True
def vec(f): return {tuple(k):int(c) for k,c in f.to_dict().items()}
def sqrt_mod(a):
    r=pow(a,(P+1)//4,P); return r if r*r%P==a%P else None
NPTS=int(sys.argv[2]); pts=[]
while len(pts)<NPTS:
    q=[rng.randrange(P) for _ in range(6)]; p=[rng.randrange(P) for _ in range(6)]; us=[]
    for (i,j) in pairs:
        r2=((q[2*i]-q[2*j])**2+(q[2*i+1]-q[2*j+1])**2)%P; r=sqrt_mod(r2) if r2 else None
        if not r: break
        us.append(pow(r,-1,P))
    if len(us)==3: pts.append(q+p+us)
def evalvec(f): 
    out={}
    for i,pt in enumerate(pts):
        v=int(f(*pt))
        if v: out[i]=v
    return out
layer=H
for L in range(1,int(sys.argv[1])+1):
    t0=time.time()
    cands = H if L==1 else [pb(h,b) for b in layer for h in H]
    Bf=Basis(); Bp=Basis()
    for c in cands:
        if Bf.add(vec(c),c): Bp.add(evalvec(c),c)
    layer=Bf.elems
    print(f"L={L:2d} formal g_L={len(Bf.elems):6d} physical g_L={len(Bp.elems):6d} ({time.time()-t0:.1f}s)",flush=True)
