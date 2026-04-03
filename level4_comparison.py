#!/usr/bin/env python3
"""
Level-4 Comparison — Phase 1 Item 1.5
======================================

Compares dim(L₄) lower bounds across all computed configurations:
  A) Bar chart of d(4) at max sample count per config
  B) Convergence curves: d(4) vs n_samples
  C) Level-by-level stacked bar chart

Data from results/level4_*/results.json and aws_results/level4_*/results.json.
Output goes to spectral_depth/.
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'spectral_depth'
os.makedirs(OUT_DIR, exist_ok=True)

# Directories to search for level4 results
SEARCH_DIRS = ['results', 'aws_results']


def discover_results():
    """Find all level4 results.json files and parse them."""
    records = []
    for base in SEARCH_DIRS:
        pattern = os.path.join(base, 'level4_*', 'results.json')
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Parse config type from directory name
            dirname = os.path.basename(os.path.dirname(path))
            # e.g. "level4_global_200000" -> config="global", n_samples=200000
            parts = dirname.replace('level4_', '').rsplit('_', 1)
            if len(parts) == 2:
                config_type, n_str = parts
                try:
                    n_samples = int(n_str)
                except ValueError:
                    config_type = dirname.replace('level4_', '')
                    n_samples = data.get('n_samples', 0)
            else:
                config_type = parts[0]
                n_samples = data.get('n_samples', 0)

            records.append({
                'config': config_type,
                'n_samples': n_samples,
                'd4': data.get('d4_lower_bound', data.get('dims', [0]*5)[4] if 'dims' in data else 0),
                'definitive': data.get('definitive_gap', False),
                'boundary_gap': data.get('boundary_gap_ratio', 0),
                'dims': data.get('dims', data.get('level_dims', {})),
                'path': path,
            })

    # Deduplicate: same config + n_samples → keep first found
    seen = set()
    deduped = []
    for r in records:
        key = (r['config'], r['n_samples'])
        if key not in seen and r['d4'] > 0:
            seen.add(key)
            deduped.append(r)
    return deduped


def fig_bar_chart(records):
    """Figure A: d(4) at max sample count per config."""
    # Group by config, take max n_samples
    best = {}
    for r in records:
        key = r['config']
        if key not in best or r['n_samples'] > best[key]['n_samples']:
            best[key] = r

    # Sort by d4 descending
    configs = sorted(best.values(), key=lambda x: -x['d4'])

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    x = np.arange(len(configs))
    colors = ['#2ecc71' if c['definitive'] else '#e74c3c' for c in configs]
    bars = ax.bar(x, [c['d4'] for c in configs], color=colors, edgecolor='black', linewidth=0.5)

    # Annotate
    for i, c in enumerate(configs):
        ax.text(i, c['d4'] + 50, f"{c['d4']}", ha='center', fontsize=10, fontweight='bold')
        ax.text(i, c['d4'] * 0.5, f"n={c['n_samples']//1000}K",
                ha='center', fontsize=8, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([c['config'].replace('_', ' ').title() for c in configs], fontsize=11)
    ax.set_ylabel('dim(L₄) lower bound', fontsize=12)
    ax.set_title('Level-4 Dimension Lower Bounds by Configuration Type',
                 fontsize=14, fontweight='bold')

    # Legend for definitive/not
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='Definitive gap'),
                       Patch(facecolor='#e74c3c', edgecolor='black', label='Not definitive')]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = 'level4_comparison_chart.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


def fig_convergence(records):
    """Figure B: d(4) vs n_samples, one line per config."""
    # Group by config type
    by_config = {}
    for r in records:
        by_config.setdefault(r['config'], []).append(r)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    colors_map = {'global': '#e41a1c', 'lagrange': '#377eb8',
                  'euler': '#4daf4a', 'scalene': '#984ea3'}
    markers_map = {'global': 'o', 'lagrange': 's', 'euler': '^', 'scalene': 'D'}

    for config, recs in sorted(by_config.items()):
        recs = sorted(recs, key=lambda x: x['n_samples'])
        ns = [r['n_samples'] for r in recs]
        d4s = [r['d4'] for r in recs]
        defs = [r['definitive'] for r in recs]

        c = colors_map.get(config, '#999999')
        m = markers_map.get(config, 'o')
        ax.plot(ns, d4s, '-', color=c, marker=m, markersize=7,
                linewidth=2, label=config.title())

        # Mark definitive points with a filled marker, open for not definitive
        for n, d, defin in zip(ns, d4s, defs):
            if defin:
                ax.plot(n, d, marker=m, markersize=10, color=c,
                        markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of phase-space samples', fontsize=12)
    ax.set_ylabel('dim(L₄) lower bound', fontsize=12)
    ax.set_title('Level-4 Convergence: dim(L₄) vs Sample Count',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Power-law fit for global data
    global_recs = sorted(by_config.get('global', []), key=lambda x: x['n_samples'])
    if len(global_recs) >= 3:
        ns = np.array([r['n_samples'] for r in global_recs])
        d4s = np.array([r['d4'] for r in global_recs])
        # Fit log(d4) = a * log(n) + b
        coeffs = np.polyfit(np.log10(ns), np.log10(d4s), 1)
        n_extrap = np.logspace(np.log10(ns.min()), np.log10(ns.max() * 5), 100)
        d4_fit = 10 ** (coeffs[0] * np.log10(n_extrap) + coeffs[1])
        ax.plot(n_extrap, d4_fit, '--', color='#e41a1c', alpha=0.4, linewidth=1.5,
                label=f'Global fit: d₄ ∝ n^{{{coeffs[0]:.2f}}}')
        ax.legend(fontsize=10)

    plt.tight_layout()
    fname = 'level4_convergence_curves.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


def fig_stacked_levels(records):
    """Figure C: stacked bar chart of level dims for best run per config."""
    best = {}
    for r in records:
        key = r['config']
        if key not in best or r['n_samples'] > best[key]['n_samples']:
            best[key] = r

    configs = sorted(best.values(), key=lambda x: -x['d4'])

    # Extract per-level dims
    level_data = []
    for c in configs:
        dims_raw = c['dims']
        if isinstance(dims_raw, dict):
            dims = [dims_raw.get(str(i), 0) for i in range(5)]
        elif isinstance(dims_raw, list) and len(dims_raw) >= 5:
            dims = dims_raw[:5]
        else:
            dims = [3, 6, 17, 116, c['d4']]
        # Convert to NEW generators per level
        new_gens = [dims[0]] + [dims[i] - dims[i-1] for i in range(1, len(dims))]
        level_data.append(new_gens)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    x = np.arange(len(configs))
    level_colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']
    level_labels = ['L₀ (3)', 'L₁ (+3)', 'L₂ (+11)', 'L₃ (new)', 'L₄ (new)']

    bottom = np.zeros(len(configs))
    for lvl in range(5):
        vals = [ld[lvl] for ld in level_data]
        ax.bar(x, vals, bottom=bottom, color=level_colors[lvl],
               edgecolor='black', linewidth=0.3, label=level_labels[lvl])
        bottom += vals

    # Annotate total dim at top
    for i, c in enumerate(configs):
        ax.text(i, bottom[i] + 50, f'{c["d4"]}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['config'].title()}\n({c['n_samples']//1000}K)"
                        for c in configs], fontsize=10)
    ax.set_ylabel('Cumulative dimension', fontsize=12)
    ax.set_title('Level-by-Level Growth: Stacked Dimensions per Config',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = 'level4_stacked_levels.png'
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {fname}")


if __name__ == '__main__':
    print("\n--- 1.5 Level-4 Comparison ---")
    records = discover_results()
    print(f"  Found {len(records)} level-4 result files")
    for r in sorted(records, key=lambda x: (x['config'], x['n_samples'])):
        gap_str = '✓' if r['definitive'] else '✗'
        print(f"    {r['config']:12s} n={r['n_samples']:>7d}  d₄={r['d4']:>5d}  {gap_str}  {r['path']}")

    fig_bar_chart(records)
    fig_convergence(records)
    fig_stacked_levels(records)

    print(f"\nAll output in {OUT_DIR}/")
