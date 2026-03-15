#!/usr/bin/env python3
"""
Three-body orbit simulation for video animations.

Produces MP4 clips of:
  1. Chaotic three-body orbit (generic initial conditions)
  2. Lagrange equilateral orbit (stable, with perturbation)
  3. Figure-eight choreography

Usage:
    python video/orbit_sim.py                # all three
    python video/orbit_sim.py --scene chaos  # just the chaotic one
"""

import os
import argparse
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patheffects as pe

OUT_DIR = os.path.join('video', 'renders')
os.makedirs(OUT_DIR, exist_ok=True)

DARK_BG = '#0d1117'
COLORS = ['#58a6ff', '#f0c040', '#ff6b6b']
TRAIL_ALPHA = 0.4


def three_body_rhs(t, state, masses, G=1.0, softening=1e-4):
    """Right-hand side for planar 3-body equations of motion."""
    x = state[:6].reshape(3, 2)
    v = state[6:].reshape(3, 2)

    acc = np.zeros_like(x)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            rij = x[j] - x[i]
            dist = np.sqrt(np.sum(rij**2) + softening**2)
            acc[i] += G * masses[j] * rij / dist**3

    return np.concatenate([v.flatten(), acc.flatten()])


def simulate(x0, v0, masses, t_span, dt=0.001):
    """Integrate the three-body problem."""
    state0 = np.concatenate([x0.flatten(), v0.flatten()])
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(three_body_rhs, t_span, state0,
                    args=(masses,), t_eval=t_eval,
                    method='DOP853', rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[:6].reshape(3, 2, -1)


def animate_orbit(t, positions, masses, title, filename,
                  trail_length=500, fps=60, duration=10):
    """Create an animated orbit video."""
    n_frames = fps * duration
    step = max(1, len(t) // n_frames)
    frame_indices = list(range(0, len(t), step))[:n_frames]

    all_pos = positions.reshape(3, 2, -1)
    margin = 0.3
    xmin = all_pos[:, 0, :].min() - margin
    xmax = all_pos[:, 0, :].max() + margin
    ymin = all_pos[:, 1, :].min() - margin
    ymax = all_pos[:, 1, :].max() + margin

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    fig, ax = plt.subplots(figsize=(10.8, 10.8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_axis_off()

    ax.set_title(title, fontsize=20, color='white', pad=15,
                 fontfamily='serif')

    trails = []
    dots = []
    for i in range(3):
        trail, = ax.plot([], [], '-', color=COLORS[i],
                         alpha=TRAIL_ALPHA, linewidth=1.5)
        trails.append(trail)
        size = 80 * (masses[i] / max(masses))**0.5 + 40
        dot, = ax.plot([], [], 'o', color=COLORS[i], markersize=8,
                       markeredgecolor='white', markeredgewidth=0.5)
        dots.append(dot)

    for spine in ax.spines.values():
        spine.set_visible(False)

    def init():
        for trail in trails:
            trail.set_data([], [])
        for dot in dots:
            dot.set_data([], [])
        return trails + dots

    def update(frame_num):
        idx = frame_indices[frame_num]
        start = max(0, idx - trail_length)
        for i in range(3):
            trails[i].set_data(all_pos[i, 0, start:idx],
                               all_pos[i, 1, start:idx])
            dots[i].set_data([all_pos[i, 0, idx]], [all_pos[i, 1, idx]])
        return trails + dots

    writer = FFMpegWriter(fps=fps, bitrate=5000)
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(frame_indices), blit=True)

    out_path = os.path.join(OUT_DIR, filename)
    anim.save(out_path, writer=writer, dpi=150,
              savefig_kwargs={'facecolor': DARK_BG})
    plt.close()
    print(f'  Saved {out_path} ({len(frame_indices)} frames)')


def chaotic_orbit():
    """Generic chaotic three-body orbit."""
    masses = np.array([1.0, 1.0, 1.0])
    x0 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.8]])
    v0 = np.array([[0.35, 0.1], [-0.2, -0.15], [-0.15, 0.05]])

    v0 -= np.average(v0, axis=0, weights=masses)

    print('  Simulating chaotic orbit...')
    t, pos = simulate(x0, v0, masses, (0, 25), dt=0.0005)
    print(f'  {len(t)} timesteps')

    animate_orbit(t, pos, masses, 'Chaotic Three-Body Orbit',
                  'orbit_chaotic.mp4', trail_length=800, duration=12)


def lagrange_orbit():
    """Near-equilateral Lagrange orbit with small perturbation."""
    masses = np.array([1.0, 1.0, 1.0])
    R = 1.0
    angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    x0 = R * np.column_stack([np.cos(angles), np.sin(angles)])

    omega = np.sqrt(3 * 1.0 / R**3)
    v0 = omega * R * np.column_stack([-np.sin(angles), np.cos(angles)])
    v0[0] += np.array([0.05, 0.03])

    v0 -= np.average(v0, axis=0, weights=masses)

    print('  Simulating Lagrange orbit...')
    t, pos = simulate(x0, v0, masses, (0, 30), dt=0.0005)
    print(f'  {len(t)} timesteps')

    animate_orbit(t, pos, masses, 'Near-Lagrange Orbit (perturbed equilateral)',
                  'orbit_lagrange.mp4', trail_length=1200, duration=12)


def figure_eight():
    """Cris Moore's figure-eight choreography."""
    masses = np.array([1.0, 1.0, 1.0])
    x0 = np.array([
        [-0.97000436, 0.24308753],
        [0.97000436, -0.24308753],
        [0.0, 0.0]
    ])
    v0 = np.array([
        [0.4662036850, 0.4323657300],
        [0.4662036850, 0.4323657300],
        [-0.9324073700, -0.8647314600]
    ])

    print('  Simulating figure-eight orbit...')
    t, pos = simulate(x0, v0, masses, (0, 20), dt=0.0005)
    print(f'  {len(t)} timesteps')

    animate_orbit(t, pos, masses, 'Figure-Eight Choreography',
                  'orbit_figure_eight.mp4', trail_length=2000, duration=10)


def main():
    parser = argparse.ArgumentParser(description='Three-body orbit animations')
    parser.add_argument('--scene', type=str, default='all',
                        choices=['all', 'chaos', 'lagrange', 'eight'])
    args = parser.parse_args()

    print('=== Three-Body Orbit Simulator ===')
    print(f'Output: {OUT_DIR}/')
    print()

    if args.scene in ('all', 'chaos'):
        print('[1] Chaotic orbit...')
        chaotic_orbit()

    if args.scene in ('all', 'lagrange'):
        print('[2] Lagrange orbit...')
        lagrange_orbit()

    if args.scene in ('all', 'eight'):
        print('[3] Figure-eight...')
        figure_eight()

    print('\nDone.')


if __name__ == '__main__':
    main()
