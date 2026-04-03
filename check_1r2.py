#!/usr/bin/env python3
import numpy as np, json, os

for d in ['1r2', '1r2_q+2_-1_-1', 'atlas-1r2-q2m1m1']:
    path = os.path.join('aws_results', 'atlas_full', d)
    print(f'=== {d} ===')
    if os.path.isdir(path):
        print(f'  files: {os.listdir(path)}')
    else:
        print('  NOT FOUND')
        continue
    rm_path = os.path.join(path, 'rank_map.npy')
    cfg_path = os.path.join(path, 'config.json')
    sum_path = os.path.join(path, 'summary.json')
    if os.path.exists(rm_path):
        rm = np.load(rm_path)
        gm = np.load(os.path.join(path, 'gap_map.npy'))
        print(f'  shape: {rm.shape}, unique ranks: {sorted(set(rm.ravel().astype(int)))[:8]}')
    else:
        print('  no rank_map.npy')
    if os.path.exists(cfg_path):
        c = json.load(open(cfg_path))
        print(f'  masses={c.get("masses")}, charges={c.get("charges")}, potential={c.get("potential_type")}, res={c.get("resolution")}')
    else:
        print('  no config.json')
    if os.path.exists(sum_path):
        s = json.load(open(sum_path))
        print(f'  r116={s.get("rank_116_fraction")}, rows={s.get("rows_completed")}')
    else:
        print('  no summary.json')
    print()
