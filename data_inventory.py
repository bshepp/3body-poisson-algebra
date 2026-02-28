"""Complete data inventory across all datasets."""

import os
import json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
HIRES = os.path.join(BASE, 'atlas_output_hires')
A1000 = os.path.join(BASE, 'atlas_1000')
RESULTS = os.path.join(BASE, 'results')


def scan_atlas_dir(path):
    """Read an atlas directory with rank_map, gap_map, sv_spectra."""
    info = {'path': os.path.relpath(path, BASE)}
    cp = os.path.join(path, 'checkpoint.json')
    cfg = os.path.join(path, 'config.json')

    if os.path.exists(cfg):
        with open(cfg) as f:
            info['config'] = json.load(f)
    if os.path.exists(cp):
        with open(cp) as f:
            info['checkpoint'] = json.load(f)

    rp = os.path.join(path, 'rank_map.npy')
    gp = os.path.join(path, 'gap_map.npy')
    sp = os.path.join(path, 'sv_spectra.npy')

    if os.path.exists(rp):
        r = np.load(rp)
        info['shape'] = list(r.shape)
        info['rank_range'] = [int(r.min()), int(r.max())]
        vals, counts = np.unique(r, return_counts=True)
        info['rank_distribution'] = {int(v): int(c) for v, c in zip(vals, counts)}
    if os.path.exists(gp):
        g = np.load(gp)
        valid = g[g > 0]
        if len(valid) > 0:
            info['gap_range'] = [float(valid.min()), float(valid.max())]
    if os.path.exists(sp):
        s = np.load(sp)
        info['sv_spectra_shape'] = list(s.shape)
        info['sv_spectra_size_mb'] = round(s.nbytes / 1e6, 1)
    return info


def scan_l4_dir(path):
    """Read a Level 4 results directory."""
    info = {'path': os.path.relpath(path, BASE)}
    rp = os.path.join(path, 'results.json')
    if os.path.exists(rp):
        with open(rp) as f:
            info['results'] = json.load(f)
    sp = os.path.join(path, 'svd_spectrum.npy')
    if os.path.exists(sp):
        s = np.load(sp)
        info['spectrum_shape'] = list(s.shape)
    gp = os.path.join(path, 'gap_ratios.npy')
    if os.path.exists(gp):
        g = np.load(gp)
        info['gap_ratios_shape'] = list(g.shape)
    info['has_analysis_png'] = os.path.exists(os.path.join(path, 'level4_analysis.png'))
    return info


def main():
    inventory = {'datasets': [], 'visualizations': []}

    # --- 100x100 hires atlas ---
    print("=" * 60)
    print("  DATA INVENTORY")
    print("=" * 60)

    print("\n--- 100x100 Hires Atlas ---")
    for pot_dir in ['1_r', '1_r2', 'harmonic']:
        path = os.path.join(HIRES, pot_dir)
        if os.path.exists(os.path.join(path, 'rank_map.npy')):
            info = scan_atlas_dir(path)
            info['type'] = '100x100_atlas'
            info['potential'] = pot_dir
            info['epsilon'] = '5e-3'
            inventory['datasets'].append(info)
            print(f"  {pot_dir} (eps=5e-3): shape={info.get('shape')}, "
                  f"rank={info.get('rank_range')}")

        for eps_dir in sorted(os.listdir(path)):
            eps_path = os.path.join(path, eps_dir)
            if eps_dir.startswith('eps_') and os.path.isdir(eps_path):
                if os.path.exists(os.path.join(eps_path, 'rank_map.npy')):
                    info = scan_atlas_dir(eps_path)
                    info['type'] = '100x100_multi_epsilon'
                    info['potential'] = pot_dir
                    info['epsilon'] = eps_dir.replace('eps_', '')
                    inventory['datasets'].append(info)
                    print(f"  {pot_dir} ({eps_dir}): shape={info.get('shape')}, "
                          f"rank={info.get('rank_range')}")

    # --- 1000x1000 atlas ---
    print("\n--- 1000x1000 Atlas ---")
    clean = os.path.join(A1000, '1_r', 'merged_clean')
    if os.path.exists(os.path.join(clean, 'rank_map.npy')):
        info = scan_atlas_dir(clean)
        info['type'] = '1000x1000_clean_merge'
        info['potential'] = '1_r'
        info['epsilon'] = '5e-3'
        inventory['datasets'].append(info)
        print(f"  merged_clean: shape={info.get('shape')}, "
              f"rank={info.get('rank_range')}, "
              f"sv_mb={info.get('sv_spectra_size_mb')}")

    excl = os.path.join(A1000, '1_r', 'block_0100_0200')
    if os.path.exists(os.path.join(excl, 'rank_map.npy')):
        info = scan_atlas_dir(excl)
        info['type'] = '1000x1000_excluded_block'
        info['potential'] = '1_r'
        info['epsilon'] = '5e-3'
        inventory['datasets'].append(info)
        cp = info.get('checkpoint', {})
        print(f"  block_0100 (excluded): {cp.get('completed_rows', '?')}/100 rows, "
              f"rank={info.get('rank_range')}")

    for bdir in sorted(os.listdir(os.path.join(A1000, '1_r'))):
        bpath = os.path.join(A1000, '1_r', bdir)
        if bdir.startswith('block_') and bdir != 'block_0100_0200' and os.path.isdir(bpath):
            if os.path.exists(os.path.join(bpath, 'rank_map.npy')):
                info = scan_atlas_dir(bpath)
                info['type'] = '1000x1000_block'
                info['potential'] = '1_r'
                cp = info.get('checkpoint', {})
                print(f"  {bdir}: {cp.get('completed_rows', '?')} rows, "
                      f"rank={info.get('rank_range')}")

    # --- Level 4 results ---
    print("\n--- Level 4 Results ---")
    l4_dirs = sorted([d for d in os.listdir(RESULTS) if d.startswith('level4_')])
    for d in l4_dirs:
        path = os.path.join(RESULTS, d)
        if os.path.isdir(path):
            info = scan_l4_dir(path)
            info['type'] = 'level4'
            inventory['datasets'].append(info)
            res = info.get('results', {})
            print(f"  {d}: rank={res.get('rank', '?')}, "
                  f"n_gens={res.get('n_generators', '?')}, "
                  f"gap={res.get('max_gap_ratio', '?')}")

    # --- Existing visualizations ---
    print("\n--- Existing Visualizations ---")
    viz_dirs = [
        (os.path.join(HIRES, 'sv_analysis'), 'sv_analysis'),
        (os.path.join(HIRES, 'multi_epsilon'), 'multi_epsilon_viz'),
        (os.path.join(A1000, 'viz'), 'atlas_1000_viz'),
    ]
    for vdir, label in viz_dirs:
        if os.path.exists(vdir):
            files = [f for f in os.listdir(vdir) if os.path.isfile(os.path.join(vdir, f))]
            inventory['visualizations'].append({
                'location': os.path.relpath(vdir, BASE),
                'label': label,
                'files': files,
                'count': len(files),
            })
            print(f"  {label}: {len(files)} files")
            for f in sorted(files):
                sz = os.path.getsize(os.path.join(vdir, f))
                print(f"    {f} ({sz/1024:.0f} KB)")

    hires_root = [f for f in os.listdir(HIRES)
                  if os.path.isfile(os.path.join(HIRES, f))
                  and (f.endswith('.png') or f.endswith('.html'))]
    if hires_root:
        inventory['visualizations'].append({
            'location': os.path.relpath(HIRES, BASE),
            'label': 'hires_root',
            'files': hires_root,
            'count': len(hires_root),
        })
        print(f"  hires_root: {len(hires_root)} files")

    # --- Summary ---
    n_atlas = sum(1 for d in inventory['datasets'] if 'atlas' in d.get('type', ''))
    n_l4 = sum(1 for d in inventory['datasets'] if d.get('type') == 'level4')
    n_viz = sum(v['count'] for v in inventory['visualizations'])
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {n_atlas} atlas datasets, {n_l4} Level 4 results, "
          f"{n_viz} visualization files")
    print(f"{'='*60}")

    out = os.path.join(BASE, 'data_inventory.json')
    with open(out, 'w') as f:
        json.dump(inventory, f, indent=2, default=str)
    print(f"\n  Saved to {out}")


if __name__ == '__main__':
    main()
