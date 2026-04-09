#!/usr/bin/env python3
"""
Render the 1/r² (Calogero-Moser) knee-index map as a single-panel figure
for embedding on the research dashboard.

The knee index = SV index where the steepest consecutive log-drop occurs.
Source data: atlas_output_hires/1_r2/sv_spectra.npy (100×100 grid).
Output: website/assets/knee_index_1r2.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
HIRES_DIR = os.path.join(ROOT, 'atlas_output_hires', '1_r2')
OUT_PATH = os.path.join(SCRIPT_DIR, 'assets', 'knee_index_1r2.png')



# Load data
mu   = np.load(os.path.join(HIRES_DIR, 'mu_vals.npy'))
phi  = np.load(os.path.join(HIRES_DIR, 'phi_vals.npy'))
rank = np.load(os.path.join(HIRES_DIR, 'rank_map.npy'))
sv   = np.load(os.path.join(HIRES_DIR, 'sv_spectra.npy'))

phi_deg = np.degrees(phi)
mask = rank >= 116

# Knee index: index of steepest consecutive log-drop in SV spectrum
log_sv = np.log10(np.clip(sv[:, :, :116], 1e-30, None))
log_diff = -np.diff(log_sv, axis=2)           # shape (..., 115)
knee_idx = np.argmax(log_diff, axis=2).astype(float)
knee_idx[~mask] = np.nan

ki_valid = knee_idx[np.isfinite(knee_idx)]
vmin = np.percentile(ki_valid, 1)
vmax = np.percentile(ki_valid, 99)

# Render
fig, ax = plt.subplots(figsize=(8, 6.5), facecolor='#0a0c10')
ax.set_facecolor('#0a0c10')

im = ax.pcolormesh(phi_deg, mu, knee_idx, cmap='plasma',
                   vmin=vmin, vmax=vmax, shading='auto')

ax.set_xlim(phi_deg[0], phi_deg[-1])
ax.set_ylim(mu[0], mu[-1])

ax.set_xlabel(r'$\phi$ (deg)', fontsize=12, color='#c8ccd4')
ax.set_ylabel(r'$\mu$', fontsize=12, color='#c8ccd4')
ax.set_title('Knee Index — 1/r² (Calogero-Moser)\nSV index of steepest spectral drop',
             fontsize=13, fontweight='bold', color='#c8ccd4', pad=12)
ax.tick_params(colors='#7a8090')
for spine in ax.spines.values():
    spine.set_color('#2a3040')

cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label('SV index of steepest drop', fontsize=10, color='#c8ccd4')
cb.ax.tick_params(colors='#7a8090')
cb.outline.set_edgecolor('#2a3040')

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='#0a0c10')
plt.close()
print(f"Saved {OUT_PATH}  ({os.path.getsize(OUT_PATH)} bytes)")
