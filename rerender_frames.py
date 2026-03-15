"""Re-render atlas animation frames with adaptive colorscale."""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

OUT_DIR = 'atlas_animation'
STROKE = [pe.withStroke(linewidth=2.5, foreground='black')]

mu_vals = np.load(os.path.join(OUT_DIR, 'mu_vals.npy'))
phi_vals = np.load(os.path.join(OUT_DIR, 'phi_vals.npy'))
n_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

all_lg = []
for n in n_values:
    g = np.load(os.path.join(OUT_DIR, f'gap_map_n{n:.4f}.npy'))
    all_lg.append(np.log10(np.clip(g, 1, None)))

vmin = 0
vmax = max(lg.max() for lg in all_lg) * 1.05
print(f"Color range: [{vmin:.2f}, {vmax:.2f}]")

frame_paths = []
for idx, (n, lg) in enumerate(zip(n_values, all_lg)):
    phi_deg = phi_vals * 180 / np.pi
    phi_mask = phi_deg <= 170
    phi_trim = phi_deg[phi_mask]
    lg_trim = lg[:, phi_mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(phi_trim, mu_vals, lg_trim,
                       cmap='inferno', shading='auto',
                       vmin=vmin, vmax=vmax)

    ax.plot(60.0, 1.0, '*', color='cyan', markersize=14,
            markeredgecolor='black', markeredgewidth=0.8, zorder=5)
    ax.annotate('Lagrange', (60.0, 1.0), textcoords='offset points',
                xytext=(10, -4), color='cyan', fontsize=10,
                fontweight='bold', path_effects=STROKE)

    ax.set_xlim(phi_trim[0], 170)
    ax.set_xlabel(r'$\phi$ (deg)', fontsize=13)
    ax.set_ylabel(r'$\mu$', fontsize=13)

    n_str = f'{int(n)}' if n == int(n) else f'{n:.2f}'
    title = r'$\log_{10}$(gap ratio) $-$ $1/r^{' + n_str + r'}$'
    ax.set_title(title, fontsize=14)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'$\log_{10}$(gap ratio)', fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'frame_{idx:03d}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    frame_paths.append(path)
    print(f'Frame {idx}: n={n:.2f}, max_lg={lg.max():.2f}')

frames = [Image.open(p) for p in frame_paths]
forward = list(frames)
backward = list(reversed(frames[1:-1]))
all_frames = forward + backward
gif_path = os.path.join(OUT_DIR, 'atlas_sweep_1r_to_1r2.gif')
all_frames[0].save(gif_path, save_all=True,
                   append_images=all_frames[1:],
                   duration=500, loop=0)
print(f'\nGIF saved: {gif_path} ({len(all_frames)} frames)')
