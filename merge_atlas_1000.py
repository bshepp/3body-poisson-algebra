"""
Selective merge of 1000x1000 atlas data.

Merges only fully complete blocks into a clean dataset.
Partial/questionable blocks are catalogued but excluded from the merged arrays.
"""

import os
import json
import numpy as np

OUTPUT_BASE = 'atlas_1000'
MU_RANGE = (0.05, 5.0)
PHI_RANGE = (0.05, np.pi - 0.05)
GRID_N = 1000


def merge_clean():
    pot_dir = '1_r'
    base = os.path.join(OUTPUT_BASE, pot_dir)
    merged_dir = os.path.join(base, 'merged_clean')
    os.makedirs(merged_dir, exist_ok=True)

    blocks = sorted([
        d for d in os.listdir(base)
        if d.startswith('block_') and os.path.isdir(os.path.join(base, d))
    ])

    mu_all = np.linspace(MU_RANGE[0], MU_RANGE[1], GRID_N)
    phi_all = np.linspace(PHI_RANGE[0], PHI_RANGE[1], GRID_N)

    clean_rank, clean_gap, clean_sv = [], [], []
    clean_mu_indices = []
    excluded = []

    for bdir in blocks:
        bpath = os.path.join(base, bdir)
        cp_file = os.path.join(bpath, 'checkpoint.json')
        cfg_file = os.path.join(bpath, 'config.json')

        if not os.path.exists(cp_file) or not os.path.exists(cfg_file):
            print(f"  SKIP  {bdir}: no checkpoint/config")
            excluded.append({'block': bdir, 'reason': 'no checkpoint/config'})
            continue

        with open(cp_file) as f:
            cp = json.load(f)
        with open(cfg_file) as f:
            cfg = json.load(f)

        expected = cfg['end_row'] - cfg['start_row']
        actual = cp['completed_rows']
        start = cfg['start_row']
        end = cfg['end_row']

        if actual < expected:
            print(f"  EXCL  {bdir}: incomplete ({actual}/{expected} rows, "
                  f"global {start}-{start+actual-1}) -- kept for analysis")
            excluded.append({
                'block': bdir,
                'reason': f'incomplete: {actual}/{expected} rows',
                'global_rows': f'{start}-{start+actual-1}',
                'mu_range': [float(mu_all[start]), float(mu_all[start+actual-1])],
            })
            continue

        r = np.load(os.path.join(bpath, 'rank_map.npy'))
        g = np.load(os.path.join(bpath, 'gap_map.npy'))
        s = np.load(os.path.join(bpath, 'sv_spectra.npy'))

        clean_rank.append(r)
        clean_gap.append(g)
        clean_sv.append(s)
        clean_mu_indices.extend(range(start, end))

        print(f"  OK    {bdir}: {actual} rows (global {start}-{end-1}), "
              f"mu=[{mu_all[start]:.4f}, {mu_all[end-1]:.4f}], "
              f"rank=[{r.min()}, {r.max()}]")

    rank_map = np.concatenate(clean_rank, axis=0)
    gap_map = np.concatenate(clean_gap, axis=0)
    sv_spectra = np.concatenate(clean_sv, axis=0)
    mu_clean = mu_all[clean_mu_indices]

    np.save(os.path.join(merged_dir, 'rank_map.npy'), rank_map)
    np.save(os.path.join(merged_dir, 'gap_map.npy'), gap_map)
    np.save(os.path.join(merged_dir, 'sv_spectra.npy'), sv_spectra)
    np.save(os.path.join(merged_dir, 'mu_vals.npy'), mu_clean)
    np.save(os.path.join(merged_dir, 'phi_vals.npy'), phi_all)

    missing_rows = sorted(set(range(GRID_N)) - set(clean_mu_indices))
    missing_mu = [(int(i), float(mu_all[i])) for i in missing_rows]

    manifest = {
        'potential': '1/r',
        'grid_n': GRID_N,
        'epsilon': 5e-3,
        'n_generators': int(sv_spectra.shape[2]),
        'clean_rows': len(clean_mu_indices),
        'missing_rows': len(missing_rows),
        'coverage_pct': round(100 * len(clean_mu_indices) / GRID_N, 1),
        'mu_range_clean': [float(mu_clean.min()), float(mu_clean.max())],
        'shape': list(rank_map.shape),
        'rank_range': [int(rank_map.min()), int(rank_map.max())],
        'gap_range': [float(gap_map.min()), float(gap_map.max())],
        'excluded_blocks': excluded,
        'missing_row_sample': missing_mu[:20],
    }

    with open(os.path.join(merged_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  CLEAN MERGE COMPLETE")
    print(f"  Shape: {rank_map.shape}  ({manifest['coverage_pct']}% coverage)")
    print(f"  Rows: {len(clean_mu_indices)}/{GRID_N} "
          f"(missing {len(missing_rows)} rows)")
    print(f"  mu range: [{mu_clean.min():.4f}, {mu_clean.max():.4f}]")
    print(f"  Rank range: [{rank_map.min()}, {rank_map.max()}]")
    print(f"  Gap range:  [{gap_map.min():.2e}, {gap_map.max():.2e}]")
    print(f"  Excluded blocks: {len(excluded)}")
    for ex in excluded:
        print(f"    - {ex['block']}: {ex['reason']}")
    print(f"  Output: {merged_dir}/")
    print(f"{'='*70}")

    return manifest


if __name__ == '__main__':
    merge_clean()
