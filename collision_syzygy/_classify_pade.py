import json
D = json.load(open('collision_syzygy_pade.json'))
rows = D['results']
n_free = len(rows)
pivots = D['pivot_set']
pure = []
mixed = []
for j, row in enumerate(rows):
    fails = sum(1 for p in pivots if row[p]['kind'] == 'FAIL')
    nonzero_pivots = sum(
        1 for p in pivots
        if row[p]['kind'] == 'rat' and any(c != '0/1' for c in row[p]['num'])
    )
    if fails == 0 and nonzero_pivots > 0:
        pure.append((j, nonzero_pivots))
    elif fails == 0 and nonzero_pivots == 0:
        # The vector has only its own free column = 1, all other entries 0.
        pure.append((j, 0))
    else:
        mixed.append((j, fails, nonzero_pivots))

print(f"PURE-CONSTANT rows (no eps-dependent failures): {len(pure)}")
for j, n in sorted(pure, key=lambda x: (x[1], x[0])):
    fc = D['free_columns'][j]
    print(f"  row j={j:2d}  free_col={fc:3d}  nonzero_pivots={n}")

print()
print(f"MIXED rows (have at least one eps-dependent failing entry): {len(mixed)}")
hist = {}
for _, f, _ in mixed:
    hist[f] = hist.get(f, 0) + 1
print("  failure-count histogram (#failing entries -> #rows):")
for k in sorted(hist):
    print(f"    {k:3d} failing entries: {hist[k]:3d} rows")
