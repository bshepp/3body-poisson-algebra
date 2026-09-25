"""Exact fiber ranks (order 0) and order<=1 jet ranks over fixed triangles, planar, V=-g/r, mod p.
order 0: rank of g|_{q=q0} as functions of p.   order<=1: rank of (g, d_q g)|_{q=q0}."""
import sys, os, random
from flint import nmod_mat
EQUAL = os.environ.get('EQUAL') == '1'
sys.argv = ['d4.py', '2', '400', '17']
src = open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'd4.py')).read()
if EQUAL:
    src = src.replace("m = [rng.randrange(2,P) for _ in range(3)]; gc = [rng.randrange(2,P) for _ in range(3)]", "m = [1,1,1]; gc = [1,1,1]")
src = src.replace('P = 2147483647', 'P = ' + os.environ.get('PRIME', '2147483647'))
exec(src.split("# ---- level 4 via exact derivative pipeline ----")[0])
def sqrtm(a):
    r = pow(a % P, (P+1)//4, P); assert r*r % P == a % P; return r
inv = lambda a: pow(a % P, -1, P)
h = inv(2)
s3h = None
try: s3h = sqrtm(3*inv(4))            # sqrt(3)/2 mod p
except AssertionError: pass
shapes = {  # positions mod p, and the three pairwise distances mod p (r12, r13, r23)
 'generic scalene 3-4-5':   ([0,0, 3,0, 0,4], [3,4,5]),
 'isosceles (5,5,6)':       ([0,0, 6,0, 3,4], [6,5,5]),
 'Euler collinear (1:2)':   ([0,0, 1,0, 3,0], [1,3,2]),
 'collinear midpoint':      ([0,0, 1,0, 2,0], [1,2,1]),
}
if s3h is not None:
    shapes['Lagrange equilateral'] = ([0,0, 1,0, h, s3h], [1,1,1])
rng2 = random.Random(5)
for name,(q,r) in shapes.items():
    qm = [v % P for v in q]; us = [inv(d) for d in r]
    pts = [qm + [rng2.randrange(P) for _ in range(6)] + us for _ in range(250)]
    rows0 = [[int(f(*pt)) for pt in pts] for f,l in A]
    r0 = nmod_mat(rows0, P).rank()
    rows1 = []
    for f,l in A:
        row = [int(f(*pt)) for pt in pts[:120]]
        for b in range(3):
            for c in range(2):
                df = Dq(f,b,c); row += [int(df(*pt)) for pt in pts[:120]]
        rows1.append(row)
    r1 = nmod_mat(rows1, P).rank()
    print(f"{'equal' if EQUAL else 'random'} masses | {name:24s} order-0 rank {r0:3d} | order<=1 rank {r1:3d}", flush=True)
