#!/usr/bin/env python3
"""
Plot accuracy for a single experiment configuration.
Displays the accuracy of Edge 0, Edge 1, and Edge 2 on one figure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Available experiment directories
RUNS = {
    'Training Accuracy in the Real System': os.path.abspath(
        os.path.join(SCRIPT_DIR, '..', '..', 'v4', 'outputs')
    ),
    'Training Accuracy in Digital Twin': os.path.abspath(
        os.path.join(SCRIPT_DIR, '..', 'outputs', 'baseline')
    ),
    'Training Accuracy in Digital Twin with KDE delays': os.path.abspath(
        os.path.join(SCRIPT_DIR, '..', 'outputs', 'kde')
    ),
    'Training Accuracy in Digital Twin with WGAN delays': os.path.abspath(
        os.path.join(SCRIPT_DIR, '..', 'outputs', 'wgan')
    ),
}

# ------------------------------------------------------------------
# Select which experiment to plot
# ------------------------------------------------------------------
RUN_LABEL = 'Training Accuracy in the Real System'
RUN_DIR = RUNS[RUN_LABEL]

# Edge colors
COLORS = {
    0: '#1f77b4',  # Blue
    1: '#e377c2',  # Pink
    2: '#2ca02c',  # Green
}

plt.figure(figsize=(8, 6))

# Plot accuracy for each edge
for edge_id in range(3):
    path = os.path.join(RUN_DIR, f'edge_{edge_id}_metrics.csv')

    try:
        df = pd.read_csv(path)

        plt.plot(
            df['round'],
            df['accuracy'],
            color=COLORS[edge_id],
            linewidth=2.5,
            label=f'Edge {edge_id}',
            alpha=0.9,
        )

    except FileNotFoundError:
        print(f"Warning: Missing file: {path}")

# Figure formatting
plt.xlabel('Round', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(RUN_LABEL, fontsize=14, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)

# Accuracy limits
plt.ylim(-0.05, 1.05)

plt.tight_layout()

# Save figure
save_path = os.path.join(SCRIPT_DIR, 'real_system_accuracy_plot.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"Success! Saved plot to: {save_path}")

plt.show()