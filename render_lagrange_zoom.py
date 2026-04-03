#!/usr/bin/env python3
"""Zoomed triptych around the Lagrange equilateral point: 1/r vs 1/r^3."""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

d = os.path.join('aws_results', 'atlas_full')

fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='white')

datasets = {}
for name in ['1r_q+1_+1_+1', '1r3']:
    mu = np.load(os.path.join(d, name, 'mu_vals.npy'))
    phi = np.load(os.path.join(d, name, 'phi_vals.npy'))
    gap = np.load(os.path.join(d, name, 'gap_map.npy'))
    datasets[name] = (mu, phi, np.log10(np.clip(gap, 1, None)))

mu, phi, _ = datasets['1r_q+1_+1_+1']
mu_mask = (mu >= 0.5) & (mu <= 1.8)
phi_mask = (phi >= np.radians(20)) & (phi <= np.radians(100))
mi = np.where(mu_mask)[0]
pi = np.where(phi_mask)[0]
sub_mu = mu[mi]
sub_phi = phi[pi]

vmin, vmax = 6.0, 10.0
labels = [('1r_q+1_+1_+1', '1/r (baseline)'), ('1r3', r'1/r$^3$')]

for idx, (name, label) in enumerate(labels):
    _, _, lg = datasets[name]
    sub = lg[np.ix_(mi, pi)]
    ax = axes[idx]
    im = ax.pcolormesh(sub_phi * 180 / np.pi, sub_mu, sub,
                       cmap=cm.inferno, vmin=vmin, vmax=vmax, shading='auto')
    ax.plot(60, 1.0, 'o', color='cyan', markersize=14, fillstyle='none',
            markeredgewidth=2)
    ax.annotate('Lagrange', (60, 1.0), textcoords='offset points', xytext=(10, 8),
                color='cyan', fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$\phi$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(label, fontsize=14)
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r'$\log_{10}$(gap ratio)', fontsize=10)

lg1 = datasets['1r_q+1_+1_+1'][2]
lg3 = datasets['1r3'][2]
diff = lg3 - lg1
sub_diff = diff[np.ix_(mi, pi)]
dlim = max(np.abs(sub_diff).max(), 0.5)

ax = axes[2]
im2 = ax.pcolormesh(sub_phi * 180 / np.pi, sub_mu, sub_diff,
                    cmap=cm.RdBu_r, vmin=-dlim, vmax=dlim, shading='auto')
ax.plot(60, 1.0, 'o', color='black', markersize=14, fillstyle='none',
        markeredgewidth=2)
ax.annotate('Lagrange', (60, 1.0), textcoords='offset points', xytext=(10, 8),
            color='black', fontsize=11, fontweight='bold')
ax.set_xlabel(r'$\phi$ (degrees)', fontsize=12)
ax.set_ylabel(r'$\mu$', fontsize=12)
ax.set_title(r'Difference (1/r$^3$ $-$ 1/r)', fontsize=14)
fig.colorbar(im2, ax=ax, pad=0.02).set_label(r'$\Delta\;\log_{10}$(gap)', fontsize=10)

fig.suptitle(r'Lagrange Region Zoom  —  1/r vs 1/r$^3$', fontsize=16,
             fontweight='bold', y=1.01)
plt.tight_layout()
out = os.path.join('aws_results', 'atlas_figures', 'lagrange_zoom_1r3.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved {out}')
