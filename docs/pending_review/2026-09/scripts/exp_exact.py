"""Exact rank over Q of the monomial-coefficient matrix for V = c*exp(-r), N=3, d=1, one ordering.
Variables p1..p3, w12,w13,w23 (w_ij = exp(-(x_i-x_j))). Formal rank (w free) is an UPPER bound
on the true function dimension; also computes the rank after imposing w13 = w12*w23 exactly."""
import sys, sympy as sp
from flint import fmpq_mat
p = sp.symbols('p1:4'); w12, w13, w23 = W = sp.symbols('w12 w13 w23')
pairs = {(1,2): w12, (1,3): w13, (2,3): w23}
masses = [sp.Rational(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1,1,1]
def dx(f,k):
    return sum((-w*sp.diff(f,w) if k==i else w*sp.diff(f,w) if k==j else 0) for (i,j),w in pairs.items())
def pb(f,g): return sp.expand(sum(dx(f,k)*sp.diff(g,p[k-1]) - sp.diff(f,p[k-1])*dx(g,k) for k in (1,2,3)))
H = [p[i-1]**2/(2*masses[i-1]) + p[j-1]**2/(2*masses[j-1]) + w for (i,j),w in pairs.items()]
levels=[H]; allg=list(H)
for L in (1,2,3):
    fs=len(allg)-len(levels[-1]); new=[]
    for i in range(len(allg)):
        for j in range(max(i+1,fs),len(allg)):
            b=pb(allg[i],allg[j])
            if b!=0: new.append(b)
    levels.append(new); allg+=new
cum=[sum(len(l) for l in levels[:k+1]) for k in range(4)]
def exact_ranks(exprs, gens):
    ds=[sp.Poly(e,*gens).as_dict() for e in exprs]
    mons=sorted({m for d in ds for m in d}); idx={m:i for i,m in enumerate(mons)}
    out=[]
    for c in cum:
        M=fmpq_mat(c,len(mons))
        for r,d in enumerate(ds[:c]):
            for m,v in d.items(): M[r,idx[m]]=sp.Rational(v).p if sp.Rational(v).q==1 else __import__('flint').fmpq(int(sp.Rational(v).p),int(sp.Rational(v).q))
        out.append(M.rank())
    return out, len(mons)
r,nm=exact_ranks(allg, list(p)+list(W))
print("masses",masses," formal exact Q ranks (w free):",r," monomials",nm)
phys=[sp.expand(e.subs(w13,w12*w23)) for e in allg]
r2,nm2=exact_ranks(phys, list(p)+[w12,w23])
print("                  exact Q ranks with w13=w12*w23 imposed:",r2," monomials",nm2)
