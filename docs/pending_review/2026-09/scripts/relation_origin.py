"""Length-4 relation vs Newton's third law. 1D, N=3, H_ij = p_i^2/2 + p_j^2/2 + V_ij.
u_ij = 1/(x_i - c x_j): du/dx_i = -u^2, du/dx_j = +c u^2 (exact). Evaluated mod p at points with
u = 1/(x_i - c x_j), so ranks are certified lower bounds; 18 is the free maximum."""
import sys, random
from flint import nmod_mpoly_ctx, nmod_mat
P = 2147483647
names = ('x1','x2','x3','p1','p2','p3','u12','u13','u23')
ctx = nmod_mpoly_ctx.get(names, modulus=P); G = dict(zip(names, ctx.gens()))
pairs = [(0,1),(0,2),(1,2)]; UN = {(0,1):'u12',(0,2):'u13',(1,2):'u23'}
mode = sys.argv[1]
c = 2 if mode == 'skew' else 1
X = ['x1','x2','x3']; PN = ['p1','p2','p3']
def Dx(f, k):
    out = f.derivative(X[k])
    for (i,j),un in UN.items():
        fu = f.derivative(un)
        if k == i: out -= fu*G[un]**2
        elif k == j: out += c*fu*G[un]**2
    return out
def pb(f, g):
    r = ctx.from_dict({})
    for k in range(3): r += Dx(f,k)*g.derivative(PN[k]) - f.derivative(PN[k])*Dx(g,k)
    return r
half = pow(2, -1, P)
def V(i, j):
    u = G[UN[(i,j)]]; xi, xj = G[X[i]], G[X[j]]
    return {'translation': u, 'skew': u, 'external': u + xi*xi, 'generic2': u + xi*xj*xj,
            'translation_poly': u + (xi-xj)**3}.get(mode, u)
R = random.Random(9)
def randquad():
    return sum((R.randrange(1,P)*G[PN[k]]*G[PN[l]] for k in range(3) for l in range(k,3)), ctx.from_dict({}))
def randpotx():
    return sum((R.randrange(1,P)*G[X[a]]**e1*G[X[b]]**e2 for a in range(3) for b in range(3) for e1,e2 in [(1,0),(2,1),(1,2),(3,0)]), ctx.from_dict({}))
if mode == 'randquad':      # random flat kinetic forms + random position-only potentials (no body structure)
    H = [randquad() + randpotx() + G[UN[pr]] for pr in pairs]
elif mode == 'curved':      # kinetic term with position-dependent coefficient: breaks {T,T}=0
    H = [half*G[PN[i]]**2 + half*G[PN[j]]**2 + G[X[i]]*G[PN[j]]**2 + V(i,j) for i,j in pairs]
elif mode == 'momentum_potential':   # 'potential' that depends on momentum: breaks {V,V}=0 / weight grading
    H = [half*G[PN[i]]**2 + half*G[PN[j]]**2 + V(i,j) + G[PN[i]]*G[X[j]] for i,j in pairs]
elif mode == 'nonlocal':   # V_12 also depends on the third body's position
    third = {(0,1):2,(0,2):1,(1,2):0}
    H = [half*G[PN[i]]**2 + half*G[PN[j]]**2 + V(i,j) + G[X[i]]*G[X[third[(i,j)]]]**2 for i,j in pairs]
elif mode == 'curved_local':  # T_i = (1 + x_i^2) p_i^2 / 2 : local to body i, but not flat
    T = [half*(1+G[X[k]]**2)*G[PN[k]]**2 for k in range(3)]
    H = [T[i] + T[j] + V(i,j) for i,j in pairs]
elif mode == 'local_generic':  # T_i = generic function of (x_i,p_i); V_ij generic in (x_i,x_j,p_i,p_j)-free
    T = [half*G[PN[k]]**2 + G[X[k]]*G[PN[k]]**3 + G[X[k]]**2*G[PN[k]] for k in range(3)]
    H = [T[i] + T[j] + V(i,j) + G[X[i]]*G[X[j]]**2 for i,j in pairs]
elif mode == 'cubic_kinetic':   # local, V standard, but T_k has a p^3 term
    T = [half*G[PN[k]]**2 + G[PN[k]]**3 for k in range(3)]
    H = [T[i] + T[j] + V(i,j) for i,j in pairs]
elif mode == 'quartic_kinetic':  # local, homogeneous degree 4 in p (still 'homogeneous', not 2)
    T = [G[PN[k]]**4 for k in range(3)]
    H = [T[i] + T[j] + V(i,j) for i,j in pairs]
elif mode == 'linear_plus_quadratic':  # T_k = p_k^2/2 + a*p_k (a magnetic-like shift, still quadratic-affine)
    T = [half*G[PN[k]]**2 + (k+2)*G[PN[k]] for k in range(3)]
    H = [T[i] + T[j] + V(i,j) for i,j in pairs]
else:
    H = [half*G[PN[i]]**2 + half*G[PN[j]]**2 + V(i,j) for i,j in pairs]
def rn(w):
    e = H[w[-1]]
    for ch in reversed(w[:-1]): e = pb(H[ch], e)
    return e
basis = [(0,0,0,1),(0,0,0,2),(0,0,1,2),(0,1,0,1),(0,1,0,2),(0,1,1,2),(0,2,0,2),(0,2,1,2),(1,0,0,2),
         (1,0,1,2),(1,1,0,1),(1,1,0,2),(1,1,1,2),(1,2,0,2),(1,2,1,2),(2,0,1,2),(2,2,0,2),(2,2,1,2)]
els = [rn(w) for w in basis]
rng = random.Random(4); pts = []
while len(pts) < 60:
    xs = [rng.randrange(P) for _ in range(3)]; ps = [rng.randrange(P) for _ in range(3)]
    ds = [(xs[i] - c*xs[j]) % P for i,j in pairs]
    if all(ds): pts.append(xs + ps + [pow(d, -1, P) for d in ds])
M = nmod_mat([[int(e(*pt)) for pt in pts] for e in els], P)
print(f"{mode:17s} rank of 18 length-4 words = {M.rank()}   (17 = relation holds, 18 = free)")
