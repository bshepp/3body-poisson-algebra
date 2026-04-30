import json, pickle, sys
sys.path.insert(0, '.')

D = json.load(open('collision_syzygy_pade.json'))

# Need generator names -> import from checkpoint
import sympy as sp
from exact_growth import Q_VARS, P_VARS, U_VARS, ALL_VARS

# Load level-3 to get generator names
with open('checkpoints/level_3.pkl', 'rb') as f:
    ck = pickle.load(f)
print("Checkpoint keys:", list(ck.keys())[:10])

# Try common attribute names
gen_names = None
for key in ('names', 'generator_names', 'levels'):
    if key in ck:
        print(f"  {key}: type={type(ck[key]).__name__}")
        if key == 'names':
            gen_names = ck[key]
        if key == 'levels':
            lv = ck['levels']
            print(f"    len(levels)={len(lv)} sample={lv[:5]}")

# Look for exprs / expressions
for key in ('exprs', 'expressions', 'generators'):
    if key in ck:
        print(f"  {key}: len={len(ck[key])}")
