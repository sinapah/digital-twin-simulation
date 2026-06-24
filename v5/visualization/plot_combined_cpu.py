#!/usr/bin/env python3
"""
Combined CPU usage plot: 9 lines (3 edges × 3 runs: v4, v5 kde, v5 wgan)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

RUNS = {
    'v4 (real)':   '/home/ubuntu/digital-twin-simulation/v4/outputs',
    'v5 KDE':      '/home/ubuntu/digital-twin-simulation/v5/outputs/kde',
    'v5 WGAN':     '/home/ubuntu/digital-twin-simulation/v5/outputs/wgan',
}

COLORS = {'v4 (real)': '#1f77b4', 'v5 KDE': '#2ca02c', 'v5 WGAN': '#d62728'}
STYLES = {0: '-', 1: '--', 2: ':'}

fig, ax = plt.subplots(figsize=(14, 8))

for run_label, run_dir in RUNS.items():
    for edge_id in range(3):
        path = os.path.join(run_dir, f'edge_{edge_id}_metrics.csv')
        df = pd.read_csv(path)
        ax.plot(
            df['round'], df['cpu_avg'],
            color=COLORS[run_label],
            linestyle=STYLES[edge_id],
            linewidth=2,
            label=f'{run_label} — Edge {edge_id}',
            alpha=0.85,
        )

ax.set_xlabel('Round', fontsize=12)
ax.set_ylabel('CPU Usage (%)', fontsize=12)
ax.set_title('CPU Usage per Edge: V4 (real) vs V5 KDE vs V5 WGAN', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'combined_cpu_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved combined_cpu_comparison.png")
plt.close()