#!/usr/bin/env python3
"""Render standalone 1/r^2 gap ratio panel for preprint teaser figure."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HIRES_DIR = 'atlas_output_hires'
d = os.path.join(HIRES_DIR, '1_r2')
mu = np.load(os.path.join(d, 'mu_vals.npy'))
phi = np.load(os.path.join(d, 'phi_vals.npy'))
gap = np.load(os.path.join(d, 'gap_map.npy'))

lg = np.log10(np.clip(gap, 1, None))

phi_max = 170
phi_mask = phi * 180 / np.pi <= phi_max
phi_trim = phi[phi_mask]
lg_trim = lg[:, phi_mask]

STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(phi_trim * 180 / np.pi, mu, lg_trim,
                   cmap='inferno', shading='auto')

lx, ly = np.pi / 3 * 180 / np.pi, 1.0
ax.plot(lx, ly, '*', color='cyan', markersize=14,
        markeredgecolor='black', markeredgewidth=0.8, zorder=5)
ax.annotate('Lagrange', (lx, ly), textcoords='offset points',
            xytext=(10, -4), color='cyan', fontsize=10,
            fontweight='bold', path_effects=STROKE)

ax.set_xlim(phi_trim[0] * 180 / np.pi, phi_max)
ax.set_xlabel(r'$\phi$ (deg)', fontsize=13)
ax.set_ylabel(r'$\mu$', fontsize=13)
ax.set_title(r'$\log_{10}$(gap ratio) $-$ $1/r^2$ (Calogero-Moser)', fontsize=14)
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label(r'$\log_{10}$(gap ratio)', fontsize=12)

plt.tight_layout()
plt.savefig('fig_atlas_teaser.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved fig_atlas_teaser.png')
