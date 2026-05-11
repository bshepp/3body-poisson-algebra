"""
poisson_n3_d2_harmonic.sage
---------------------------------------------------------------------------
Independent SageMath oracle for the harmonic (r^2) planar 3-body Poisson
algebra. Mirrors mathematica/poisson_n3_d2_harmonic.wl.

Reproduces the cumulative-rank dimension sequence
    [3, 6, 13, 15, 15]
at d=2, N=3 with H_ij = T_i + T_j + r_ij^2.

The harmonic potential closes at dimension 15 and is the structural
opposite of the singular potentials (1/r, 1/r^2, 1/r^3, log) which all
give [3, 6, 17, 116] and grow without bound.

Run:    sage sage/poisson_n3_d2_harmonic.sage
Output: sage/results/n3_d2_harmonic.json
---------------------------------------------------------------------------
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone

# See poisson_n3_d2.sage for why we resolve the engine path this way.
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
load(ENGINE)  # noqa: F821  -- Sage runtime

MAX_LEVEL = 4
POT       = 'harmonic'
EXPECTED  = [3, 6, 13, 15, 15]

RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_FILE = os.path.join(RESULTS_DIR, 'n3_d2_harmonic.json')

import sage.version  # noqa: E402

print('=' * 60)
print('  SageMath oracle: planar 3-body harmonic Poisson algebra')
print('  Sage version: %s' % sage.version.version)
print('  Python: %s' % sys.version.split()[0])
print('  N=%d  d=%d  potential=%s  maxLevel=%d'
      % (N_BODIES, D_SPATIAL, POT, MAX_LEVEL))
print('=' * 60)

print()
result = build_algebra(POT, MAX_LEVEL)

out = {
    'sage_version':   sage.version.version,
    'python_version': sys.version.split()[0],
    'system_id':      '%s-%s' % (platform.system(), platform.machine()),
    'timestamp_utc':  datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'),
    'n_bodies':       int(N_BODIES),
    'd_spatial':      int(D_SPATIAL),
    'potential':      POT,
    'max_level':      int(MAX_LEVEL),
    'expected':       EXPECTED,
    'result':         result,
}

def _json_default(o):
    try:
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
r = result['cumulative_rank']
exp = EXPECTED[:len(r)]
tag = 'MATCH' if r == exp else 'MISMATCH'
print('%s:  cumulative_rank = %s   expected %s   %s' % (POT, r, exp, tag))

print()
print('Results saved to: %s' % OUT_FILE)
