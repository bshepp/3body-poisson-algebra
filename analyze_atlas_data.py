"""Quick diagnostic of atlas_1000 data: clean merge + excluded block."""

import numpy as np
import json

MU_ALL = np.linspace(0.05, 5.0, 1000)
PHI_ALL = np.linspace(0.05, np.pi - 0.05, 1000)


def analyze_block(label, rank, gap, mu):
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')
    print(f'  Shape: {rank.shape}')
    print(f'  mu range: [{mu.min():.4f}, {mu.max():.4f}]')
    print(f'\n  Rank distribution:')
    vals, counts = np.unique(rank, return_counts=True)
    for v, c in zip(vals, counts):
        pct = 100 * c / rank.size
        print(f'    rank={v:4d}: {c:7d} points ({pct:.2f}%)')

    t_rows, _ = np.where(rank == -1)
    if len(t_rows) > 0:
        t_mu = mu[t_rows]
        print(f'\n  Timeouts (-1): {len(t_rows)} points')
        print(f'    mu range: [{t_mu.min():.4f}, {t_mu.max():.4f}]')

    drops = (rank > 0) & (rank < 116)
    d_rows, _ = np.where(drops)
    if len(d_rows) > 0:
        d_mu = mu[d_rows]
        print(f'\n  Rank drops (<116): {len(d_rows)} points')
        print(f'    Rank values: {np.unique(rank[drops])}')
        print(f'    mu range: [{d_mu.min():.4f}, {d_mu.max():.4f}]')

    excess = rank > 116
    e_rows, e_cols = np.where(excess)
    if len(e_rows) > 0:
        e_mu = mu[e_rows]
        print(f'\n  Rank excess (>116): {len(e_rows)} points')
        print(f'    Rank values: {np.unique(rank[excess])}')
        print(f'    mu range: [{e_mu.min():.4f}, {e_mu.max():.4f}]')

        anom_local = np.unique(e_rows)[:10]
        for ri in anom_local:
            row_ranks = rank[ri]
            unusual = row_ranks[row_ranks > 116]
            print(f'    Row {ri} (mu={mu[ri]:.4f}): '
                  f'{len(unusual)} anomalous points, '
                  f'ranks={np.unique(unusual)}, '
                  f'gap_range=[{gap[ri].min():.1e}, {gap[ri].max():.1e}]')

    mask116 = rank == 116
    if mask116.any():
        g116 = gap[mask116]
        print(f'\n  Gap ratio stats (rank=116 points):')
        print(f'    min={g116.min():.2e}  median={np.median(g116):.2e}  '
              f'max={g116.max():.2e}')
        print(f'    <100:  {(g116 < 100).sum()} points')
        print(f'    <1000: {(g116 < 1000).sum()} points')
        print(f'    >1e6:  {(g116 > 1e6).sum()} points')


def main():
    # Clean merged data
    clean = 'atlas_1000/1_r/merged_clean'
    r_clean = np.load(f'{clean}/rank_map.npy')
    g_clean = np.load(f'{clean}/gap_map.npy')
    mu_clean = np.load(f'{clean}/mu_vals.npy')
    analyze_block('CLEAN DATA (800 rows, mu ~1.04-5.0)', r_clean, g_clean, mu_clean)

    # Block 100 (excluded)
    bpath = 'atlas_1000/1_r/block_0100_0200'
    r100 = np.load(f'{bpath}/rank_map.npy')[:80]
    g100 = np.load(f'{bpath}/gap_map.npy')[:80]
    mu100 = MU_ALL[100:180]
    analyze_block('BLOCK 100 [EXCLUDED] (80 rows, mu ~0.55-0.94)', r100, g100, mu100)

    print(f'\n{"="*70}')
    print(f'  COVERAGE SUMMARY')
    print(f'{"="*70}')
    print(f'  Clean:    800 rows (mu 1.04-5.00)  -> merged_clean/')
    print(f'  Excluded:  80 rows (mu 0.55-0.94)  -> block_0100_0200/')
    print(f'  Missing:  120 rows (rows 0-99: no data, rows 180-199: lost)')
    print(f'  Total downloaded: 880/1000 rows (88%)')
    print(f'  Clean coverage:   800/1000 rows (80%)')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
