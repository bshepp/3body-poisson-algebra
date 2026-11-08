"""Compare formal rank (u_ij free) vs physical rank (u_ij = 1/r_ij on phase space), mod p."""
import sys, random, itertools
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', '..', '..', 'nbody'))
sys.setrecursionlimit(100000)
import sympy as sp
from exact_growth_nbody import NBodyAlgebra

N, D, MAXL = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
pot = sys.argv[4] if len(sys.argv) > 4 else "1/r"
P = 2147483647  # 2^31-1, = 3 mod 4
alg = NBodyAlgebra(n_bodies=N, d_spatial=D, potential=pot)
V = alg.all_vars
levels = [list(alg.hamiltonian_list)]
allg = list(levels[0])
for L in range(1, MAXL + 1):
    new = []
    prev_all = list(allg)
    frontier_start = len(prev_all) - len(levels[-1])
    for i in range(len(prev_all)):
        for j in range(max(i + 1, frontier_start), len(prev_all)):
            b = alg.simplify_generator(alg.poisson_bracket(prev_all[i], prev_all[j]))
            if b != 0:
                new.append(b)
    levels.append(new); allg += new
    print(f"level {L}: {len(new)} candidates", flush=True)

prep = [alg._prep_modp_eval(g, P) for g in allg]
nq = alg.n_q
rng = random.Random(1)

def sqrt_mod(a):
    r = pow(a, (P + 1) // 4, P)
    return r if r * r % P == a % P else None

def point(physical, chamber=None):
    while True:
        q = [rng.randrange(1, P) for _ in range(nq)]
        p = [rng.randrange(1, P) for _ in range(alg.n_p)]
        if not physical:
            u = [rng.randrange(1, P) for _ in alg.u_vars]
            return tuple(q + p + u)
        u = []; ok = True
        for (bi, bj) in alg.body_pairs:
            qi = q[(bi - 1) * D:(bi) * D]; qj = q[(bj - 1) * D:(bj) * D]
            if D == 1:
                diff = (qi[0] - qj[0]) % P
                if diff == 0: ok = False; break
                s = 1 if chamber is None else chamber[(bi, bj)]
                u.append(s * pow(diff, -1, P) % P)
            else:
                r2 = sum((a - b) ** 2 for a, b in zip(qi, qj)) % P
                r = sqrt_mod(r2) if r2 else None
                if not r: ok = False; break
                u.append(pow(r, -1, P))
        if ok:
            return tuple(q + p + u)

def rank_mod_p(rows):
    M = [list(r) for r in rows]; rk = 0; ncol = len(M[0]) if M else 0
    for c in range(ncol):
        piv = next((i for i in range(rk, len(M)) if M[i][c]), None)
        if piv is None: continue
        M[rk], M[piv] = M[piv], M[rk]
        inv = pow(M[rk][c], -1, P)
        M[rk] = [x * inv % P for x in M[rk]]
        for i in range(len(M)):
            if i != rk and M[i][c]:
                f = M[i][c]; M[i] = [(a - f * b) % P for a, b in zip(M[i], M[rk])]
        rk += 1
    return rk

def evalmat(pts):
    rows = []
    for (nt, dt) in prep:
        row = []
        for pt in pts:
            n = alg._eval_poly_modp(nt, pt, P); d = alg._eval_poly_modp(dt, pt, P)
            row.append(n * pow(d, -1, P) % P)
        rows.append(row)
    return rows

S = int(sys.argv[5]) if len(sys.argv) > 5 else 400
cum = [sum(len(l) for l in levels[:k + 1]) for k in range(len(levels))]
def cumranks(pts):
    E = evalmat(pts)
    return [rank_mod_p(E[:c]) for c in cum]
print("formal   (u free)        :", cumranks([point(False) for _ in range(S)]), flush=True)
if D == 1:
    ch = {pr: 1 for pr in alg.body_pairs}
    print("physical (one chamber)   :", cumranks([point(True, ch) for _ in range(S)]), flush=True)
    orders = list(itertools.permutations(range(1, N + 1)))
    pts = []
    for k in range(S):
        order = orders[k % len(orders)]
        pos = {b: order.index(b) for b in range(1, N + 1)}
        chs = {(bi, bj): (1 if pos[bi] < pos[bj] else -1) for (bi, bj) in alg.body_pairs}
        pts.append(point(True, chs))
    print("physical (all chambers)  :", cumranks(pts), flush=True)
else:
    print("physical (u=1/r)         :", cumranks([point(True) for _ in range(S)]), flush=True)
