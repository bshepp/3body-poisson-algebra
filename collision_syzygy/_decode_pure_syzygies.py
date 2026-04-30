"""
Decode and print the 23 PURE-CONSTANT (eps-independent) algebraic syzygies
found by the Pade interpolator over the (4,3) eps-stratum.

Each such row j gives an integer linear combination of the 156 generators that
vanishes identically on the stratum (and, since the stratum is generic enough
to detect all algebraic identities of the level-3 algebra, almost certainly
vanishes as a polynomial identity in the auxiliary phase-space variables --
a candidate "deep" Jacobi-type identity).

Sparsest first (1, 2, 3 nonzero pivots).
"""
import json
import pickle
from fractions import Fraction

D = json.load(open('collision_syzygy_pade.json'))
with open('checkpoints/level_3.pkl', 'rb') as f:
    ck = pickle.load(f)

names = ck['names']
levels = ck['levels']
assert len(names) == 156

pivots = D['pivot_set']
free_cols = D['free_columns']


def coef_str(num_list, sign_prefix=False):
    """num is a single 'p/q' string for constants."""
    f = Fraction(num_list[0])
    if f == 1:
        s = ""
    elif f == -1:
        s = "-"
    elif f.denominator == 1:
        s = f"{f.numerator}*"
    else:
        s = f"({f})*"
    return s


def render_row(j):
    row = D['results'][j]
    own = free_cols[j]
    parts = []
    # Own free col has coefficient +1
    parts.append(("+", "1", own))
    # Other free cols are zero (skip)
    # Pivots
    for p in pivots:
        r = row[p]
        if r['kind'] != 'rat':
            continue
        if all(c == "0/1" for c in r['num']):
            continue
        # Constant only (this is the "pure-constant" path)
        f = Fraction(r['num'][0])
        if f == 0:
            continue
        sign = "+" if f > 0 else "-"
        mag = abs(f)
        if mag == 1:
            mag_s = "1"
        else:
            mag_s = str(mag)
        parts.append((sign, mag_s, p))
    # Render
    s = ""
    for i, (sign, mag, idx) in enumerate(parts):
        nm = f"g[{idx}]={names[idx]}"
        if mag == "1":
            term = nm
        else:
            term = f"{mag}*{nm}"
        if i == 0:
            s += ("- " if sign == "-" else "") + term
        else:
            s += f" {sign} {term}"
    return s + "  =  0"


pure_rows = []
for j, row in enumerate(D['results']):
    if any(row[p]['kind'] == 'FAIL' for p in pivots):
        continue
    nz = sum(
        1 for p in pivots
        if row[p]['kind'] == 'rat' and any(c != "0/1" for c in row[p]['num'])
    )
    pure_rows.append((j, nz))

pure_rows.sort(key=lambda x: (x[1], x[0]))

print(f"# {len(pure_rows)} PURE-CONSTANT (eps-independent) algebraic syzygies")
print(f"#  format: integer combination of generators g[i] (with names) = 0")
print(f"#  generator levels: K_a (level 0), H_ab (level 1), {{K_a,H_bc}} etc.")
print()

for j, nz in pure_rows:
    own = free_cols[j]
    print(f"## row j={j:2d}   own free col = g[{own}] = {names[own]}   "
          f"(level {levels[own]})  --  {nz + 1} terms")
    print("    " + render_row(j))
    print()
