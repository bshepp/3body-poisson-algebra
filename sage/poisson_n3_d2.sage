"""
poisson_n3_d2.sage
---------------------------------------------------------------------------
Independent SageMath oracle for the planar 3-body Poisson algebra.

Reproduces the cumulative-rank dimension sequence
    [3, 6, 17, 116]
for both 1/r and 1/r^2 pairwise potentials at d=2, N=3.

Mirrors the Mathematica oracle (mathematica/poisson_n3_d2.wl) field-for-
field so the JSON outputs can be diffed.

Phase-space variables:    {x_i, y_i, px_i, py_i}, i = 1..3
Auxiliary variables:      u_ij = 1/r_ij                (i < j)
H_ij = (px_i^2 + py_i^2)/2 + (px_j^2 + py_j^2)/2 + potential_term(u_ij, pot)

Filtration (matches Python engine):
  Level 0: H_12, H_13, H_23
  Level 1: { {H_a, H_b} : a < b in level 0 }
  Level k: bracket each (level k-1) frontier element with each prior
           generator (level 0..k-1), modulo a frozenset-pair dedupe.

Run:    sage sage/poisson_n3_d2.sage
Output: sage/results/n3_d2_dimseq.json
---------------------------------------------------------------------------
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone

# Sage preprocesses .sage files into a temp directory before exec, so
# __file__ does not point at the source. Resolve the engine path by
# walking up from sys.argv[0] (the invocation path) and from cwd.
def _find_engine():
    candidates = []
    if sys.argv and sys.argv[0]:
        d = os.path.dirname(os.path.abspath(sys.argv[0]))
        candidates.append(os.path.join(d, 'poisson_n3_d2_engine.sage'))
    cwd = os.getcwd()
    candidates += [
        os.path.join(cwd, 'sage', 'poisson_n3_d2_engine.sage'),
        os.path.join(cwd, 'poisson_n3_d2_engine.sage'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        'Could not locate poisson_n3_d2_engine.sage; tried: %s' % candidates
    )

ENGINE = _find_engine()
HERE = os.path.dirname(ENGINE)
load(ENGINE)  # noqa: F821  -- provided by Sage runtime

MAX_LEVEL = 3
POTENTIALS = ['1/r', '1/r^2']

RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_FILE = os.path.join(RESULTS_DIR, 'n3_d2_dimseq.json')

import sage.version  # noqa: E402

print('=' * 60)
print('  SageMath oracle: planar 3-body Poisson algebra')
print('  Sage version: %s' % sage.version.version)
print('  Python: %s' % sys.version.split()[0])
print('  N=%d  d=%d  maxLevel=%d' % (N_BODIES, D_SPATIAL, MAX_LEVEL))
print('=' * 60)

results = {}
for pot in POTENTIALS:
    print()
    results[pot] = build_algebra(pot, MAX_LEVEL)

# ---- JSON output (schema matches mathematica/results/n3_d2_dimseq.json) ----

out = {
    'sage_version':   sage.version.version,
    'python_version': sys.version.split()[0],
    'system_id':      '%s-%s' % (platform.system(), platform.machine()),
    'timestamp_utc':  datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'),
    'n_bodies':       int(N_BODIES),
    'd_spatial':      int(D_SPATIAL),
    'max_level':      int(MAX_LEVEL),
    'results':        results,
}

def _json_default(o):
    # Sage's Integer / Rational / Real types -> plain Python.
    try:
        # Most Sage numeric types respond to int() cleanly
        return int(o)
    except (TypeError, ValueError):
        pass
    try:
        return float(o)
    except (TypeError, ValueError):
        pass
    return str(o)

with open(OUT_FILE, 'w') as fh:
    json.dump(out, fh, indent=2, default=_json_default)

print()
print('=' * 60)
print('  SUMMARY')
print('=' * 60)

expected = [3, 6, 17, 116]
for pot in POTENTIALS:
    r = results[pot]['cumulative_rank']
    exp = expected[:len(r)]
    tag = 'MATCH' if r == exp else 'MISMATCH'
    print('%s:  cumulative_rank = %s   expected %s   %s'
          % (pot, r, exp, tag))

print()
print('Results saved to: %s' % OUT_FILE)
