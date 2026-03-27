#!/usr/bin/env python3
"""Render triptych: 1/r^2 (left) | 1/r^-2 (center) | difference (right)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_full_atlas import load_config, _heatmap, OUT_DIR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator

os.makedirs(OUT_DIR, exist_ok=True)

mu2, phi2, rank2, gap2, cfg2, sum2 = load_config('1r2')
muh, phih, rankh, gaph, cfgh, sumh = load_config('1r-2')

log2 = np.log10(np.clip(gap2, 1.0, None))
logh = np.log10(np.clip(gaph, 1.0, None))

r116_2 = sum2.get('rank_116_fraction', 0)
r116_h = sumh.get('rank_116_fraction', 0)

if log2.shape != logh.shape:
    interp = RegularGridInterpolator((muh, phih), logh, method='linear',
        bounds_error=False, fill_value=np.nan)
    MU_g, PHI_g = np.meshgrid(mu2, phi2, indexing='ij')
    logh_r = interp(np.stack([MU_g.ravel(), PHI_g.ravel()], axis=-1)
                     ).reshape(log2.shape)
    diff = log2 - logh_r
else:
    diff = log2 - logh

fin2 = log2[np.isfinite(log2)]
vmin2, vmax2 = np.percentile(fin2, 1), np.percentile(fin2, 99)

finh = logh[np.isfinite(logh)]
vminh, vmaxh = np.percentile(finh, 1), np.percentile(finh, 99)

diff_finite = diff[np.isfinite(diff)]
diff_abs = np.nanmax(np.abs(diff_finite))
diff_lim = max(diff_abs, 0.1)

fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor='white')

lbl_gap = r'$\log_{10}$(gap ratio)'
lbl_dgap = r'$\Delta\;\log_{10}$(gap ratio)'

unique_ranks_2 = sum2.get('unique_ranks', [])
rank_str_2 = ', '.join(str(r) for r in unique_ranks_2)

unique_ranks_h = sumh.get('unique_ranks', [])
rank_str_h = ', '.join(str(r) for r in unique_ranks_h)

im0 = _heatmap(axes[0], phi2, mu2, log2, cm.inferno, vmin2, vmax2,
    '1/r\u00b2 (Calogero\u2013Moser)\nrank 116: {:.1%}  |  ranks: {}'.format(
        r116_2, rank_str_2))
fig.colorbar(im0, ax=axes[0], pad=0.02).set_label(lbl_gap, fontsize=10)

im1 = _heatmap(axes[1], phih, muh, logh, cm.inferno, vminh, vmaxh,
    '1/r\u207b\u00b2 (Harmonic)\nrank 116: {:.1%}  |  ranks: {}'.format(
        r116_h, rank_str_h))
fig.colorbar(im1, ax=axes[1], pad=0.02).set_label(lbl_gap, fontsize=10)

im2 = _heatmap(axes[2], phi2, mu2, diff, cm.RdBu_r, -diff_lim, diff_lim,
    'Difference\n(1/r\u00b2 minus 1/r\u207b\u00b2)')
fig.colorbar(im2, ax=axes[2], pad=0.02).set_label(lbl_dgap, fontsize=10)

fig.suptitle(
    'Atlas Comparison \u2014 1/r\u00b2 (Calogero\u2013Moser)  vs  '
    '1/r\u207b\u00b2 (Harmonic)',
    fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
fname = os.path.join(OUT_DIR, 'triptych_1r2_vs_1r-2.png')
plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', fname)
