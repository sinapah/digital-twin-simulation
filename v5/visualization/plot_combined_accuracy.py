#!/usr/bin/env python3
"""
Combined Accuracy usage plot: 4 miniplots (2x2 grid).
v5 Baseline explicitly moved to the top right. Y-axis locked between 0.0 and 1.05.
Uses solid lines with distinct, clear colors per Edge agent.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Capture the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Rebuild the directory layout relative to the script's home location
RUNS = {
    'Real system':   os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'v4', 'outputs')),
    'DT with no outage': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'baseline')),
    'DT with KDE delays':      os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'kde')),
    'DT with WGAN delays':     os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'wgan')),
}

# Explicit positions in the 2x2 grid: (row, column)
GRID_POSITIONS = {
    'Real system':   (0, 0),  # Top Left
    'DT with no outage': (0, 1),  # Top Right
    'DT with KDE delays':      (1, 0),  # Bottom Left
    'DT with WGAN delays':     (1, 1),  # Bottom Right
}

# 3. Distinct color mapping per Edge node (solid lines across all graphs)
COLORS = {
    0: '#1f77b4',  # Deep Blue
    1: '#e377c2',  # Pink / Magenta
    2: '#2ca02c'   # Forest Green
}

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
            
            # Plot accuracy directly using specific colors for each edge agent
            ax.plot(
                df['round'], df['accuracy'],
                color=COLORS[edge_id],
                linestyle='-',          # Strictly solid lines for all edges
                linewidth=2.5,          # Slightly thicker for optimal visibility
                label=f'Edge {edge_id}',
                alpha=0.85,
            )
                    
        except FileNotFoundError:
            print(f"Warning: Missing data for {run_label} (Checked path: {path})")

    # Customize individual miniplot parameters
    ax.set_title(run_label, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='lower right')  # Lower right is usually ideal for tracking upward accuracy lines
    
    # Set the Y-axis range strictly from 0.0 to 1.05 to clearly monitor bounds
    ax.set_ylim(-0.05, 1.05)

# Set unified global labels across outer framework
fig.supxlabel('Round', fontsize=13)
fig.supylabel('Accuracy', fontsize=13)
fig.suptitle('Accuracy Comparison across Edge Nodes', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# Save the unified graphic canvas
save_path = os.path.join(SCRIPT_DIR, 'stacked_accuracy_plots.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Success! Saved graphic to: {save_path}")

plt.show()