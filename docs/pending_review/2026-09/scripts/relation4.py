"""Find the non-free relation among length-4 Lie words in H12,H13,H23 (a=H12,b=H13,c=H23)."""
import sys, itertools, random
import sympy as sp
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', '..', '..', 'nbody')); sys.setrecursionlimit(100000)
from exact_growth_nbody import NBodyAlgebra
D=int(sys.argv[1]); masses=None
if len(sys.argv)>2 and sys.argv[2]=='generic':
    m1,m2,m3=sp.symbols('m1 m2 m3',positive=True); masses={1:m1,2:m2,3:m3}
alg=NBodyAlgebra(n_bodies=3,d_spatial=D,potential="1/r",masses=masses)
Hs=alg.hamiltonian_list; lab='abc'
# free associative realisation for choosing a basis of right-normed words
def fmul(x,y):
    out={}
    for wa,ca in x.items():
        for wb,cb in y.items(): out[wa+wb]=out.get(wa+wb,0)+ca*cb
    return {k:v for k,v in out.items() if v}
def fbr(x,y):
    o=fmul(x,y)
    for k,v in fmul(y,x).items(): o[k]=o.get(k,0)-v
    return {k:v for k,v in o.items() if v}
def rn(word, gens, br):
    e=gens[word[-1]]
    for ch in reversed(word[:-1]): e=br(gens[ch],e)
    return e
fg={i:{(i,):1} for i in range(3)}
words=[w for w in itertools.product(range(3),repeat=4) if w[-1]!=w[-2]]
allmon=sorted({k for w in words for k in rn(w,fg,fbr)})
basis=[]; rows=[]
for w in words:
    v=rn(w,fg,fbr); r=[v.get(k,0) for k in allmon]
    if sp.Matrix(rows+[r]).rank()>len(rows): rows.append(r); basis.append(w)
print("free basis size:",len(basis))
pb=lambda f,g: alg.simplify_generator(alg.poisson_bracket(f,g))
phys=[sp.expand(rn(w,dict(enumerate(Hs)),pb)) for w in basis]
V=alg.all_vars
polys=[sp.Poly(sp.together(e).as_numer_denom()[0],*V) for e in phys]
dens=[sp.together(e).as_numer_denom()[1] for e in phys]
print("denominators:",set(dens))
mons=sorted({m for p in polys for m in p.as_dict()})
M=sp.Matrix([[p.as_dict().get(m,0) for m in mons] for p in polys]).T
ns=M.nullspace()
print("physical rank:",M.rank(),"  kernel dim:",len(ns))
for v in ns:
    v=v/ max(v, key=lambda t: 0 if t==0 else 1)
    terms=[(sp.nsimplify(sp.simplify(c)),''.join(lab[i] for i in w)) for c,w in zip(v,basis) if sp.simplify(c)!=0]
    print("relation:"," + ".join(f"({c})*[{w}]" for c,w in terms))
