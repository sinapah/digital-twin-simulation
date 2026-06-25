#!/usr/bin/env python3
"""
Combined CPU usage plot: 4 miniplots (2x2 grid).
v5 Baseline explicitly moved to the top right. Y-axis locked between 170% and 210%.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Capture the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Rebuild the directory layout relative to the script's home location
RUNS = {
    'v4 (real)':   os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'v4', 'outputs')),
    'v5 Baseline': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'baseline')),
    'v5 KDE':      os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'kde')),
    'v5 WGAN':     os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'wgan')),
}

# Explicit positions in the 2x2 grid: (row, column)
GRID_POSITIONS = {
    'v4 (real)':   (0, 0),  # Top Left
    'v5 Baseline': (0, 1),  # Top Right
    'v5 KDE':      (1, 0),  # Bottom Left
    'v5 WGAN':     (1, 1),  # Bottom Right
}

COLORS = {
    'v4 (real)':   '#1f77b4', 
    'v5 Baseline': '#7f7f7f',
    'v5 KDE':      '#2ca02c', 
    'v5 WGAN':     '#d62728',
}
STYLES = {0: '-', 1: '--', 2: ':'}

# Create a 2x2 grid of subplots sharing axes scales
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

# Process each experiment configuration
for run_label, run_dir in RUNS.items():
    row, col = GRID_POSITIONS[run_label]
    ax = axes[row, col]  
    
    for edge_id in range(3):
        path = os.path.join(run_dir, f'edge_{edge_id}_metrics.csv')
        
        try:
            df = pd.read_csv(path)
            
            # Plot metrics directly without filtering or checking for outages
            ax.plot(
                df['round'], df['cpu_avg'],
                color=COLORS[run_label],
                linestyle=STYLES[edge_id],
                linewidth=2,
                label=f'Edge {edge_id}',
                alpha=0.85,
            )
                    
        except FileNotFoundError:
            print(f"Warning: Missing data for {run_label} (Checked path: {path})")

    # Customize individual miniplot parameters
    ax.set_title(run_label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    
    # Set the Y-axis range strictly from 170 to 210
    ax.set_ylim(170, 210)

# Set unified global labels across outer framework
fig.supxlabel('Round', fontsize=13)
fig.supylabel('CPU Usage (%)', fontsize=13)
fig.suptitle('CPU Usage Comparison across Edge Nodes', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# --- THE MISSING SAVE BLOCK ---
save_path = os.path.join(SCRIPT_DIR, 'stacked_cpu_plots.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Success! Saved graphic to: {save_path}")
plt.close()