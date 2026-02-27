#!/usr/bin/env python3
"""
High-Resolution Shape Sphere Atlas
====================================

100x100 grid scan at eps=5e-3 for all three potentials with:
  - Full singular value spectra saved at every grid point
  - Row-level checkpointing (crash loses at most 100 points)
  - Targeted epsilon sweeps at gap-ratio dip points
  - Interactive plotly + static matplotlib visualizations

Usage:
    python shape_sphere_hires.py                    # full run: scan + sweeps + viz
    python shape_sphere_hires.py scan               # grid scan only
    python shape_sphere_hires.py sweep              # targeted sweeps (needs scan data)
    python shape_sphere_hires.py viz                # visualizations only
    python shape_sphere_hires.py scan --potential 1/r  # single potential
"""

import os
import sys
import json
import argparse
import numpy as np
from numpy.linalg import svd
from time import time, strftime
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------------------------------------------------------------------
# Potential directory names (filesystem-safe)
# ---------------------------------------------------------------------------
POT_DIR = {'1/r': '1_r', '1/r2': '1_r2', 'harmonic': 'harmonic'}
POT_LABEL = {
    '1/r':      '1/r (Newton)',
    '1/r2':     '1/r² (Calogero-Moser)',
    'harmonic': 'r² (Harmonic)',
}
ALL_POTENTIALS = ['1/r', '1/r2', 'harmonic']

HIRES_DIR = 'atlas_output_hires'
GRID_N = 100
EPSILON = 5e-3
N_SAMPLES = 400
LEVEL = 3
MU_RANGE = (0.05, 5.0)
PHI_RANGE = (0.05, np.pi - 0.05)


# ===================================================================
# PHASE 1 — High-resolution grid scan with checkpointing
# ===================================================================

def make_output_dir(potential_type):
    d = os.path.join(HIRES_DIR, POT_DIR[potential_type])
    os.makedirs(d, exist_ok=True)
    return d


def load_checkpoint(out_dir):
    cp_file = os.path.join(out_dir, 'checkpoint.json')
    if os.path.exists(cp_file):
        with open(cp_file) as f:
            return json.load(f)
    return None


def save_checkpoint(out_dir, completed_rows, n_generators):
    cp_file = os.path.join(out_dir, 'checkpoint.json')
    with open(cp_file, 'w') as f:
        json.dump({
            'completed_rows': completed_rows,
            'n_generators': n_generators,
            'timestamp': strftime('%Y-%m-%d %H:%M:%S'),
        }, f)


def flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map, sv_spectra,
                  proximity_map=None):
    np.save(os.path.join(out_dir, 'mu_vals.npy'), mu_vals)
    np.save(os.path.join(out_dir, 'phi_vals.npy'), phi_vals)
    np.save(os.path.join(out_dir, 'rank_map.npy'), rank_map)
    np.save(os.path.join(out_dir, 'gap_map.npy'), gap_map)
    np.save(os.path.join(out_dir, 'sv_spectra.npy'), sv_spectra)
    if proximity_map is not None:
        np.save(os.path.join(out_dir, 'proximity_map.npy'), proximity_map)


def run_grid_scan(potential_type):
    """Run the 100x100 grid scan for one potential with row-level checkpointing."""
    from stability_atlas import AtlasConfig, PoissonAlgebra, ShapeSpace

    out_dir = make_output_dir(potential_type)
    label = POT_LABEL[potential_type]

    print(f"\n{'='*70}")
    print(f"  GRID SCAN: {label}")
    print(f"  Grid: {GRID_N}x{GRID_N} = {GRID_N**2} points")
    print(f"  Epsilon: {EPSILON}, Samples/point: {N_SAMPLES}, Level: {LEVEL}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*70}\n")

    # Build algebra (symbolic precomputation)
    config = AtlasConfig(
        potential_type=potential_type,
        max_level=LEVEL,
        n_phase_samples=N_SAMPLES,
        epsilon=EPSILON,
        svd_gap_threshold=1e4,
    )
    algebra = PoissonAlgebra(config)
    n_gen = algebra._n_generators

    # Grid coordinates
    mu_vals = np.linspace(MU_RANGE[0], MU_RANGE[1], GRID_N)
    phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], GRID_N)

    # Check for existing checkpoint
    cp = load_checkpoint(out_dir)
    start_row = 0

    if cp is not None and cp.get('n_generators') == n_gen:
        start_row = cp['completed_rows']
        rank_map = np.load(os.path.join(out_dir, 'rank_map.npy'))
        gap_map = np.load(os.path.join(out_dir, 'gap_map.npy'))
        sv_spectra = np.load(os.path.join(out_dir, 'sv_spectra.npy'))
        prox_path = os.path.join(out_dir, 'proximity_map.npy')
        if os.path.exists(prox_path):
            proximity_map = np.load(prox_path)
        else:
            proximity_map = np.zeros((GRID_N, GRID_N))
        print(f"  Resuming from row {start_row} ({start_row * GRID_N} points done)")
    else:
        rank_map = np.zeros((GRID_N, GRID_N), dtype=int)
        gap_map = np.zeros((GRID_N, GRID_N))
        sv_spectra = np.zeros((GRID_N, GRID_N, n_gen), dtype=np.float64)
        proximity_map = np.zeros((GRID_N, GRID_N))

    # Save grid coordinates immediately
    np.save(os.path.join(out_dir, 'mu_vals.npy'), mu_vals)
    np.save(os.path.join(out_dir, 'phi_vals.npy'), phi_vals)

    # Save config
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({
            'potential': potential_type,
            'grid_n': GRID_N,
            'epsilon': EPSILON,
            'n_samples': N_SAMPLES,
            'level': LEVEL,
            'n_generators': n_gen,
            'mu_range': list(MU_RANGE),
            'phi_range': list(PHI_RANGE),
        }, f, indent=2)

    total = GRID_N * GRID_N
    t_scan_start = time()

    for i in range(start_row, GRID_N):
        t_row_start = time()
        mu = mu_vals[i]

        for j in range(GRID_N):
            phi = phi_vals[j]
            positions = ShapeSpace.shape_to_positions(mu, phi)

            p = positions.reshape(3, 2)
            dists = [np.linalg.norm(p[0] - p[1]),
                     np.linalg.norm(p[0] - p[2]),
                     np.linalg.norm(p[1] - p[2])]
            proximity_map[i, j] = min(dists) / max(dists)

            try:
                rank, svs, info = algebra.compute_rank_at_configuration(
                    positions, LEVEL
                )
                rank_map[i, j] = rank
                gap_map[i, j] = info['max_gap_ratio']
                sv_spectra[i, j, :len(svs)] = svs
            except Exception as e:
                rank_map[i, j] = -1
                gap_map[i, j] = 0
                print(f"  WARN: Failed at ({i},{j}) mu={mu:.3f} "
                      f"phi={phi:.3f}: {e}")

        # Checkpoint after every row
        flush_arrays(out_dir, mu_vals, phi_vals, rank_map, gap_map, sv_spectra,
                     proximity_map)
        save_checkpoint(out_dir, i + 1, n_gen)

        done = (i + 1) * GRID_N
        elapsed = time() - t_scan_start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        row_time = time() - t_row_start

        ranks_this_row = rank_map[i, :]
        print(f"  Row {i+1:3d}/{GRID_N}  mu={mu:.3f}  "
              f"[{done:5d}/{total}]  "
              f"row={row_time:.1f}s  "
              f"ETA={remaining/60:.0f}m  "
              f"ranks=[{ranks_this_row.min()},{ranks_this_row.max()}]  "
              f"gap=[{gap_map[i,:].min():.1e},{gap_map[i,:].max():.1e}]",
              flush=True)

    total_time = time() - t_scan_start
    print(f"\n  Grid scan complete: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Rank range: [{rank_map.min()}, {rank_map.max()}]")
    print(f"  Gap range:  [{gap_map.min():.2e}, {gap_map.max():.2e}]")
    print(f"  Results in: {out_dir}/")

    return out_dir


# ===================================================================
# PHASE 2 — Targeted epsilon sweeps at gap-ratio dip points
# ===================================================================

def run_targeted_sweeps(potential_type, gap_threshold_log=4.0,
                        percentile_cutoff=5):
    """Identify gap-ratio dips and run epsilon sweeps at those points."""
    from stability_atlas import AtlasConfig, PoissonAlgebra, ShapeSpace

    out_dir = os.path.join(HIRES_DIR, POT_DIR[potential_type])
    label = POT_LABEL[potential_type]

    if not os.path.exists(os.path.join(out_dir, 'gap_map.npy')):
        print(f"  No grid data for {label}, skipping sweeps")
        return

    print(f"\n{'='*70}")
    print(f"  TARGETED EPSILON SWEEPS: {label}")
    print(f"{'='*70}\n")

    mu_vals = np.load(os.path.join(out_dir, 'mu_vals.npy'))
    phi_vals = np.load(os.path.join(out_dir, 'phi_vals.npy'))
    rank_map = np.load(os.path.join(out_dir, 'rank_map.npy'))
    gap_map = np.load(os.path.join(out_dir, 'gap_map.npy'))

    log_gap = np.log10(np.clip(gap_map, 1, None))

    # Identify targets: bottom percentile OR below absolute threshold
    pct_threshold = np.percentile(log_gap[gap_map > 0], percentile_cutoff)
    threshold = min(pct_threshold, gap_threshold_log)

    targets = np.argwhere(log_gap < threshold)
    # Also add known special configurations
    special_ij = []
    specials = {
        'lagrange': (1.0, np.pi / 3),
        'euler_collinear': (0.5, np.pi),
        'isosceles_right': (1.0, np.pi / 2),
        'lagrange_obtuse': (1.0, 2 * np.pi / 3),
    }
    for name, (mu_s, phi_s) in specials.items():
        i_near = np.argmin(np.abs(mu_vals - mu_s))
        j_near = np.argmin(np.abs(phi_vals - phi_s))
        special_ij.append((i_near, j_near, name))
        if not any(np.array_equal(t, [i_near, j_near]) for t in targets):
            targets = np.vstack([targets, [i_near, j_near]]) if len(targets) > 0 \
                else np.array([[i_near, j_near]])

    print(f"  Threshold: log10(gap) < {threshold:.1f}")
    print(f"  Targets from dips: {len(targets) - len(special_ij)}")
    print(f"  Targets from special configs: {len(special_ij)}")
    print(f"  Total sweep targets: {len(targets)}")

    if len(targets) == 0:
        print("  No targets found, skipping sweeps")
        return

    # Build algebra (reuse precomputation)
    config = AtlasConfig(
        potential_type=potential_type,
        max_level=LEVEL,
        n_phase_samples=N_SAMPLES,
        epsilon=EPSILON,
        svd_gap_threshold=1e4,
    )
    algebra = PoissonAlgebra(config)

    epsilons = [5e-3, 1e-3, 5e-4, 1e-4]
    sweep_results = []

    for idx, (i, j) in enumerate(targets):
        mu = mu_vals[i]
        phi = phi_vals[j]
        positions = ShapeSpace.shape_to_positions(mu, phi)

        # Check if this is a named special config
        name = None
        for si, sj, sname in special_ij:
            if si == i and sj == j:
                name = sname
                break

        ranks_at_eps = {}
        gaps_at_eps = {}
        svs_at_eps = {}

        for eps in epsilons:
            try:
                rank, svs, info = algebra.compute_rank_at_configuration(
                    positions, LEVEL, epsilon=eps
                )
                ranks_at_eps[str(eps)] = int(rank)
                gaps_at_eps[str(eps)] = float(info['max_gap_ratio'])
                svs_at_eps[str(eps)] = svs.tolist()
            except Exception as e:
                ranks_at_eps[str(eps)] = -1
                gaps_at_eps[str(eps)] = 0
                svs_at_eps[str(eps)] = []

        entry = {
            'i': int(i), 'j': int(j),
            'mu': float(mu), 'phi': float(phi),
            'phi_deg': float(phi * 180 / np.pi),
            'name': name,
            'base_rank': int(rank_map[i, j]),
            'base_gap': float(gap_map[i, j]),
            'ranks': ranks_at_eps,
            'gaps': gaps_at_eps,
        }
        sweep_results.append(entry)

        varies = len(set(ranks_at_eps.values())) > 1
        marker = " ***" if varies else ""
        label_str = f" ({name})" if name else ""
        print(f"  [{idx+1:3d}/{len(targets)}] "
              f"mu={mu:.3f} phi={phi*180/np.pi:.1f}deg{label_str}  "
              f"ranks={list(ranks_at_eps.values())}{marker}")

    # Save sweep results
    sweep_file = os.path.join(out_dir, 'epsilon_sweep_targets.json')
    with open(sweep_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Sweep results saved to {sweep_file}")


# ===================================================================
# PHASE 3 — Visualizations
# ===================================================================

def mu_phi_to_shape_sphere(mu, phi):
    """Map (mu, phi) to Dragt shape sphere coordinates (s1, s2, s3)."""
    w2_sq = mu**2 - mu * np.cos(phi) + 0.25
    N = 1.0 + w2_sq
    s1 = (1.0 - w2_sq) / N
    s2 = 2.0 * (mu * np.cos(phi) - 0.5) / N
    s3 = 2.0 * mu * np.sin(phi) / N
    return s1, s2, s3


def build_interactive_viz(potentials=None):
    """Build interactive plotly HTML with all potentials and multiple coloring modes."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if potentials is None:
        potentials = [p for p in ALL_POTENTIALS
                      if os.path.exists(os.path.join(HIRES_DIR, POT_DIR[p], 'gap_map.npy'))]

    if not potentials:
        print("  No hi-res data found, skipping interactive viz")
        return

    fig = go.Figure()
    buttons_potential = []
    trace_groups = {}
    trace_idx = 0

    for pot_idx, pot in enumerate(potentials):
        out_dir = os.path.join(HIRES_DIR, POT_DIR[pot])
        mu = np.load(os.path.join(out_dir, 'mu_vals.npy'))
        phi = np.load(os.path.join(out_dir, 'phi_vals.npy'))
        rank_map = np.load(os.path.join(out_dir, 'rank_map.npy'))
        gap_map = np.load(os.path.join(out_dir, 'gap_map.npy'))
        sv_path = os.path.join(out_dir, 'sv_spectra.npy')
        sv_spectra = np.load(sv_path) if os.path.exists(sv_path) else None

        MU, PHI = np.meshgrid(mu, phi, indexing='ij')
        S1, S2, S3 = mu_phi_to_shape_sphere(MU, PHI)

        log_gap = np.log10(np.clip(gap_map, 1.0, None))
        hover = np.empty(MU.shape, dtype=object)
        for i in range(MU.shape[0]):
            for j in range(MU.shape[1]):
                hover[i, j] = (
                    f"mu={MU[i,j]:.3f}, phi={PHI[i,j]*180/np.pi:.1f}deg<br>"
                    f"rank={rank_map[i,j]}, gap={gap_map[i,j]:.2e}"
                )

        visible_default = (pot_idx == 0)
        start_idx = trace_idx

        # --- Surface: log10(gap ratio) ---
        fig.add_trace(go.Surface(
            x=S1, y=S2, z=S3,
            surfacecolor=log_gap,
            colorscale='Inferno',
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorbar=dict(title="log10(gap)", x=1.02, len=0.5, y=0.75),
            visible=visible_default,
            name=f'{POT_LABEL[pot]} — gap ratio',
        ))
        trace_idx += 1

        # --- Surface: rank ---
        fig.add_trace(go.Surface(
            x=S1, y=S2, z=S3,
            surfacecolor=rank_map.astype(float),
            colorscale='Viridis',
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorbar=dict(title="rank", x=1.02, len=0.5, y=0.75),
            visible=False,
            name=f'{POT_LABEL[pot]} — rank',
        ))
        trace_idx += 1

        # --- Surface: sv ratio (sv[K-2]/sv[K-1] near the gap) ---
        if sv_spectra is not None:
            max_rank = int(rank_map.max())
            if max_rank > 1 and max_rank < sv_spectra.shape[2]:
                sv_ratio = np.zeros_like(gap_map)
                for i in range(sv_spectra.shape[0]):
                    for j in range(sv_spectra.shape[1]):
                        svs = sv_spectra[i, j]
                        k = max_rank - 1
                        if k + 1 < len(svs) and svs[k + 1] > 1e-15:
                            sv_ratio[i, j] = np.log10(svs[k] / svs[k + 1])
                        else:
                            sv_ratio[i, j] = np.log10(svs[k] / 1e-15) if svs[k] > 0 else 0

                fig.add_trace(go.Surface(
                    x=S1, y=S2, z=S3,
                    surfacecolor=sv_ratio,
                    colorscale='Plasma',
                    customdata=hover,
                    hovertemplate="%{customdata}<extra></extra>",
                    colorbar=dict(title=f"log10(sv[{max_rank-1}]/sv[{max_rank}])",
                                  x=1.02, len=0.5, y=0.75),
                    visible=False,
                    name=f'{POT_LABEL[pot]} — SV ratio at gap',
                ))
                trace_idx += 1

                # --- Surface: the K-th singular value itself ---
                sv_at_gap = np.log10(np.clip(sv_spectra[:, :, max_rank - 1], 1e-20, None))
                fig.add_trace(go.Surface(
                    x=S1, y=S2, z=S3,
                    surfacecolor=sv_at_gap,
                    colorscale='Magma',
                    customdata=hover,
                    hovertemplate="%{customdata}<extra></extra>",
                    colorbar=dict(title=f"log10(sv[{max_rank}])",
                                  x=1.02, len=0.5, y=0.75),
                    visible=False,
                    name=f'{POT_LABEL[pot]} — SV at gap boundary',
                ))
                trace_idx += 1

        # Mirror to lower hemisphere (dimmer)
        fig.add_trace(go.Surface(
            x=S1, y=S2, z=-S3,
            surfacecolor=log_gap,
            colorscale='Inferno',
            opacity=0.4,
            showscale=False,
            hoverinfo='skip',
            visible=visible_default,
            name=f'{POT_LABEL[pot]} — mirror',
        ))
        trace_idx += 1

        end_idx = trace_idx
        trace_groups[pot] = (start_idx, end_idx)

        # Special configuration markers
        specials = {
            'Lagrange': (1.0, np.pi / 3),
            'Euler': (0.5, np.pi),
            'Isosceles-90': (1.0, np.pi / 2),
        }
        for sname, (ms, ps) in specials.items():
            ss1, ss2, ss3 = mu_phi_to_shape_sphere(ms, ps)
            i_near = np.argmin(np.abs(mu - ms))
            j_near = np.argmin(np.abs(phi - ps))
            r = rank_map[i_near, j_near]
            g = gap_map[i_near, j_near]
            fig.add_trace(go.Scatter3d(
                x=[ss1], y=[ss2], z=[ss3],
                mode='markers+text',
                marker=dict(size=6, color='cyan', symbol='diamond',
                            line=dict(width=1, color='white')),
                text=[sname],
                textposition='top center',
                textfont=dict(size=10, color='cyan'),
                hovertext=f"{sname}<br>mu={ms:.3f}, phi={ps*180/np.pi:.1f}deg<br>"
                          f"rank={r}, gap={g:.2e}",
                hoverinfo='text',
                visible=visible_default,
                showlegend=False,
            ))
            trace_idx += 1

        trace_groups[pot] = (start_idx, trace_idx)

    # Equator (collinear great circle)
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
        z=np.zeros(200).tolist(),
        mode='lines',
        line=dict(color='rgba(255,255,255,0.3)', width=2),
        hoverinfo='skip', showlegend=False, visible=True,
    ))
    equator_idx = trace_idx
    trace_idx += 1

    # Build potential-switching buttons
    for pot in potentials:
        start, end = trace_groups[pot]
        vis = [False] * trace_idx
        # Show first surface (gap ratio) and mirror and markers
        vis[start] = True  # gap ratio surface
        # Mirror is always the one before markers start
        for ti in range(start, end):
            name = fig.data[ti].name if hasattr(fig.data[ti], 'name') else ''
            if 'mirror' in str(name) or isinstance(fig.data[ti], go.Scatter3d):
                vis[ti] = True
        vis[equator_idx] = True

        buttons_potential.append(dict(
            label=POT_LABEL[pot],
            method='update',
            args=[{'visible': vis}],
        ))

    # Build coloring mode buttons (for first potential)
    color_buttons = []
    if potentials:
        pot0 = potentials[0]
        s0, e0 = trace_groups[pot0]
        surface_traces = [i for i in range(s0, e0)
                          if isinstance(fig.data[i], go.Surface)
                          and 'mirror' not in str(getattr(fig.data[i], 'name', ''))]
        for surf_i in surface_traces:
            surf_name = fig.data[surf_i].name
            vis = [False] * trace_idx
            vis[surf_i] = True
            # Keep mirror and markers visible
            for ti in range(s0, e0):
                nm = str(getattr(fig.data[ti], 'name', ''))
                if 'mirror' in nm or isinstance(fig.data[ti], go.Scatter3d):
                    vis[ti] = True
            vis[equator_idx] = True
            short = surf_name.split('—')[-1].strip() if '—' in surf_name else surf_name
            color_buttons.append(dict(
                label=short,
                method='update',
                args=[{'visible': vis}],
            ))

    fig.update_layout(
        title=dict(text='Gap Ratio Landscape on the Shape Sphere',
                   font=dict(size=18)),
        scene=dict(
            xaxis_title='s₁', yaxis_title='s₂', zaxis_title='s₃',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=-1.2, z=0.8)),
        ),
        updatemenus=[
            dict(
                type='dropdown', direction='down',
                x=0.02, y=0.98, xanchor='left', yanchor='top',
                buttons=buttons_potential,
                showactive=True,
                bgcolor='rgba(30,30,30,0.8)',
                font=dict(color='white'),
            ),
            dict(
                type='dropdown', direction='down',
                x=0.25, y=0.98, xanchor='left', yanchor='top',
                buttons=color_buttons,
                showactive=True,
                bgcolor='rgba(30,30,30,0.8)',
                font=dict(color='white'),
            ) if color_buttons else {},
        ],
        paper_bgcolor='rgb(20,20,30)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=50, b=0),
        width=1200, height=800,
    )

    html_path = os.path.join(HIRES_DIR, 'shape_sphere_interactive.html')
    fig.write_html(html_path, include_plotlyjs=True)
    print(f"  Interactive visualization saved to {html_path}")
    return html_path


def build_static_figures(potentials=None):
    """Generate publication-quality static matplotlib figures from hi-res data."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib import cm, colors
    from scipy.interpolate import RegularGridInterpolator

    if potentials is None:
        potentials = [p for p in ALL_POTENTIALS
                      if os.path.exists(os.path.join(HIRES_DIR, POT_DIR[p], 'gap_map.npy'))]

    if not potentials:
        print("  No hi-res data found, skipping static figures")
        return

    for pot in potentials:
        out_dir = os.path.join(HIRES_DIR, POT_DIR[pot])
        mu = np.load(os.path.join(out_dir, 'mu_vals.npy'))
        phi = np.load(os.path.join(out_dir, 'phi_vals.npy'))
        rank_map = np.load(os.path.join(out_dir, 'rank_map.npy'))
        gap_map = np.load(os.path.join(out_dir, 'gap_map.npy'))

        log_gap = np.log10(np.clip(gap_map, 1.0, None))
        MU, PHI = np.meshgrid(mu, phi, indexing='ij')
        S1, S2, S3 = mu_phi_to_shape_sphere(MU, PHI)

        valid = np.isfinite(log_gap)
        vmin = np.nanpercentile(log_gap[valid], 1)
        vmax = np.nanpercentile(log_gap[valid], 99)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = cm.inferno
        FC = cmap_obj(norm(np.where(valid, log_gap, vmin)))

        fig = plt.figure(figsize=(18, 8), facecolor='white')

        # Flat heatmap
        ax1 = fig.add_subplot(121)
        im = ax1.pcolormesh(phi * 180 / np.pi, mu, log_gap,
                            cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
        specials = {
            'Lagrange': (1.0, 60), 'Euler': (0.5, 180),
            'Isos-90': (1.0, 90), 'Lagrange-obt': (1.0, 120),
        }
        for name, (m, p) in specials.items():
            if mu[0] <= m <= mu[-1] and phi[0]*180/np.pi <= p <= phi[-1]*180/np.pi:
                ax1.plot(p, m, '*', color='cyan', markersize=14,
                         markeredgecolor='white', markeredgewidth=0.5)
                ax1.annotate(name, (p, m), textcoords='offset points',
                             xytext=(8, 5), color='white', fontsize=9,
                             fontweight='bold',
                             path_effects=[pe.withStroke(linewidth=2,
                                                         foreground='black')])
        ax1.set_xlabel(r'$\phi$ (degrees)', fontsize=13)
        ax1.set_ylabel(r'$\mu = r_{13}/r_{12}$', fontsize=13)
        ax1.set_title(r'Shape parameter space $(\mu, \phi)$', fontsize=14)
        cb1 = fig.colorbar(im, ax=ax1, pad=0.02)
        cb1.set_label(r'$\log_{10}$(gap ratio)', fontsize=12)

        # 3D sphere
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(S1, S2, S3, facecolors=FC,
                         rstride=1, cstride=1, shade=False,
                         antialiased=True, alpha=0.95)
        FC_m = FC.copy()
        FC_m[:, :, 3] = 0.4
        ax2.plot_surface(S1, S2, -S3, facecolors=FC_m,
                         rstride=1, cstride=1, shade=False, antialiased=True)
        theta = np.linspace(0, 2*np.pi, 300)
        ax2.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
                 'k-', alpha=0.25, linewidth=0.8)

        for name, (ms, ps_deg) in specials.items():
            ps = ps_deg * np.pi / 180
            s1, s2, s3 = mu_phi_to_shape_sphere(ms, ps)
            ax2.scatter([s1], [s2], [s3], color='cyan', s=60, zorder=10,
                        edgecolors='white', linewidth=1, depthshade=False)

        ax2.set_xlabel('$s_1$', fontsize=11)
        ax2.set_ylabel('$s_2$', fontsize=11)
        ax2.set_zlabel('$s_3$', fontsize=11)
        ax2.set_title('Shape sphere $S^2$', fontsize=14)
        ax2.view_init(elev=30, azim=-55)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cb2 = fig.colorbar(sm, ax=ax2, shrink=0.55, pad=0.08)
        cb2.set_label(r'$\log_{10}$(gap ratio)', fontsize=12)

        fig.suptitle(
            f'Gap Ratio Landscape — {POT_LABEL[pot]} (100x100, eps={EPSILON})',
            fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        fname = os.path.join(HIRES_DIR, f'shape_sphere_hires_{POT_DIR[pot]}.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved {fname}")

    # Comparison figure
    if len(potentials) >= 2:
        fig = plt.figure(figsize=(7 * len(potentials), 7), facecolor='white')
        for idx, pot in enumerate(potentials):
            out_dir = os.path.join(HIRES_DIR, POT_DIR[pot])
            mu = np.load(os.path.join(out_dir, 'mu_vals.npy'))
            phi = np.load(os.path.join(out_dir, 'phi_vals.npy'))
            gap_map = np.load(os.path.join(out_dir, 'gap_map.npy'))
            log_gap = np.log10(np.clip(gap_map, 1.0, None))
            MU, PHI = np.meshgrid(mu, phi, indexing='ij')
            S1, S2, S3 = mu_phi_to_shape_sphere(MU, PHI)

            valid = np.isfinite(log_gap)
            vmin = np.nanpercentile(log_gap[valid], 1)
            vmax = np.nanpercentile(log_gap[valid], 99)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            FC = cm.inferno(norm(np.where(valid, log_gap, vmin)))

            ax = fig.add_subplot(1, len(potentials), idx + 1, projection='3d')
            ax.plot_surface(S1, S2, S3, facecolors=FC,
                            rstride=1, cstride=1, shade=False,
                            antialiased=True, alpha=0.95)
            FC_m = FC.copy()
            FC_m[:, :, 3] = 0.4
            ax.plot_surface(S1, S2, -S3, facecolors=FC_m,
                            rstride=1, cstride=1, shade=False, antialiased=True)
            theta = np.linspace(0, 2*np.pi, 300)
            ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
                    'k-', alpha=0.25, linewidth=0.8)

            s1l, s2l, s3l = mu_phi_to_shape_sphere(1.0, np.pi/3)
            ax.scatter([s1l], [s2l], [s3l], color='cyan', s=60, zorder=10,
                       edgecolors='white', linewidth=1, depthshade=False)

            ax.set_xlabel('$s_1$', fontsize=10, labelpad=-2)
            ax.set_ylabel('$s_2$', fontsize=10, labelpad=-2)
            ax.set_zlabel('$s_3$', fontsize=10, labelpad=-2)
            ax.set_title(POT_LABEL[pot], fontsize=13, pad=10)
            ax.view_init(elev=30, azim=-55)

            sm = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
            cb.set_label(r'$\log_{10}$(gap)', fontsize=10)

        fig.suptitle(
            'Shape Sphere Comparison (100x100, eps={})'.format(EPSILON),
            fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = os.path.join(HIRES_DIR, 'shape_sphere_hires_comparison.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved {fname}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='High-Resolution Shape Sphere Atlas')
    parser.add_argument('mode', nargs='?', default='all',
                        choices=['all', 'scan', 'sweep', 'viz'],
                        help='Which phase(s) to run')
    parser.add_argument('--potential', type=str, default=None,
                        choices=['1/r', '1/r2', 'harmonic'],
                        help='Run only one potential (default: all three)')
    args = parser.parse_args()

    pots = [args.potential] if args.potential else ALL_POTENTIALS

    if args.mode in ('all', 'scan'):
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: HIGH-RESOLUTION GRID SCAN")
        print(f"# Potentials: {pots}")
        print(f"# Grid: {GRID_N}x{GRID_N}, eps={EPSILON}, "
              f"samples={N_SAMPLES}, level={LEVEL}")
        print(f"{'#'*70}")
        for pot in pots:
            run_grid_scan(pot)

    if args.mode in ('all', 'sweep'):
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: TARGETED EPSILON SWEEPS")
        print(f"{'#'*70}")
        for pot in pots:
            run_targeted_sweeps(pot)

    if args.mode in ('all', 'viz'):
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: VISUALIZATIONS")
        print(f"{'#'*70}")
        avail = [p for p in pots
                 if os.path.exists(os.path.join(HIRES_DIR, POT_DIR[p], 'gap_map.npy'))]
        if avail:
            print("\n  Building static figures...")
            build_static_figures(avail)
            print("\n  Building interactive visualization...")
            build_interactive_viz(avail)
        else:
            print("  No data available yet. Run 'scan' first.")

    print(f"\n{'#'*70}")
    print(f"# COMPLETE")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
