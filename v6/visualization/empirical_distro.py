#!/usr/bin/env python3
"""
Combined ECDF of Interarrival Delays plot: 4 miniplots (2x2 grid).
Plots Edge 0, Edge 1, and Edge 2 across v4, v5 Baseline, v5 KDE, and v5 WGAN.
Uses unique solid colors for each edge node to differentiate them easily.
"""

import csv
import matplotlib.pyplot as plt
import os

# 1. Capture the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Map experiment labels to their respective folder directories
RUNS = {
    'Real system':   os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'v4', 'outputs')),
    'DT with no outage': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'baseline')),
    'DT with KDE delays':      os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'kde')),
    'DT with WGAN delays':     os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'outputs', 'wgan')),
}

# 3. Define grid positions to map runs to specific quadrants
GRID_POSITIONS = {
    'Real system':   (0, 0),  # Top Left
    'DT with no outage': (0, 1),  # Top Right
    'DT with KDE delays':      (1, 0),  # Bottom Left
    'DT with WGAN delays':     (1, 1),  # Bottom Right
}

# 4. Distinct solid color mapping per Edge node
COLORS = {
    0: '#1f77b4',  # Deep Blue
    1: '#e377c2',  # Distinct Pink/Magenta
    2: '#2ca02c'   # Forest Green
}

def get_empirical_distribution(file_path):
    delays = []
    
    try:
        with open(file_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    delay = float(row['interarrival_delay'])
                    if delay > 0.0:  # Skip initialization/reset rows
                        delays.append(delay)
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        return [], []
                
    delays.sort()
    n = len(delays)
    if n == 0:
        return [], []
    
    x_coords = []
    y_coords = []
    for index, delay in enumerate(delays):
        probability = (index + 1) / n
        x_coords.append(delay)
        y_coords.append(probability)
        
    return x_coords, y_coords


# Create a 2x2 grid sharing both X and Y axis scales for easy comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

# Process each run configuration
for run_label, run_dir in RUNS.items():
    row, col = GRID_POSITIONS[run_label]
    ax = axes[row, col]  
    
    for edge_id in range(3):
        path = os.path.join(run_dir, f'edge_{edge_id}_arrivals.csv')
        
        x, y = get_empirical_distribution(path)
        
        if x and y:
            ax.plot(
                x, y, 
                color=COLORS[edge_id],
                linestyle='-',          # Kept strictly solid as requested
                drawstyle='steps-post', 
                linewidth=2.5,          # Slightly thicker for clearer color rendering
                label=f'Edge {edge_id}',
                alpha=0.85
            )
        else:
            print(f"Warning: Missing or empty data for {run_label} (Checked path: {path})")

    # Customize the quadrant plot parameters
    ax.set_title(run_label, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='lower right')

# Set layout parameters and unified outer label frames
fig.supxlabel('Interarrival Delay (seconds)', fontsize=13)
fig.supylabel('F(x) - Cumulative Probability', fontsize=13)
fig.suptitle('Empirical Cumulative Distribution Function (ECDF) of Interarrival Delays', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# Save the unified graphic canvas
plot_save_path = os.path.join(SCRIPT_DIR, 'stacked_ecdf_plots.png')
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"\nGrid plot successfully saved to image: {plot_save_path}")

# Render to user interface
plt.show()