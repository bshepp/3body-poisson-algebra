"""Print a table of all saved neural algebra results."""
import json
import glob
import os

files = sorted(glob.glob("results/neural_algebras/*.json"))
print(f"{'Config':<55s} {'Dims':<25s} {'Time(s)':<10s}")
print("-" * 95)
rows = []
for f in files:
    d = json.load(open(f))
    cfg = (f"L={d['n_layers']} k={d['width']} {d['coupling_type']} "
           f"{d['loss_function']} {d['activation']}")
    dims = str(d['dimension_sequence'])
    t = d.get('computation_time_seconds', '?')
    rows.append((cfg, dims, t))
    print(f"{cfg:<55s} {dims:<25s} {t}")

print()
print("=" * 70)
print("GROUPED BY DIMENSION SEQUENCE (universality classes)")
print("=" * 70)
groups = {}
for cfg, dims, t in rows:
    groups.setdefault(dims, []).append(cfg)
for dims in sorted(groups.keys()):
    cfgs = groups[dims]
    print(f"\n{dims} ({len(cfgs)} configs):")
    for c in cfgs:
        print(f"  - {c}")
