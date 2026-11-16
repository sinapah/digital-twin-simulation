#!/usr/bin/env python3
"""
Empirical Cumulative Distribution Function (ECDF) of Interarrival Delays
for the Baseline experiment only.
"""

import csv
import matplotlib.pyplot as plt
import os

# Directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Baseline output directory
BASELINE_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', 'outputs', 'baseline')
)

# Color mapping for each edge
COLORS = {
    0: '#1f77b4',  # Blue
    1: '#e377c2',  # Pink
    2: '#2ca02c'   # Green
}


def get_empirical_distribution(file_path):
    delays = []

    try:
        with open(file_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    delay = float(row['interarrival_delay'])
                    if delay > 0:
                        delays.append(delay)
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        return [], []

    delays.sort()
    n = len(delays)

    if n == 0:
        return [], []

    x = delays
    y = [(i + 1) / n for i in range(n)]

    return x, y


# Create a single figure
plt.figure(figsize=(8, 6))

for edge_id in range(3):
    path = os.path.join(BASELINE_DIR, f'edge_{edge_id}_arrivals.csv')

    x, y = get_empirical_distribution(path)

    if x:
        plt.plot(
            x,
            y,
            drawstyle='steps-post',
            linewidth=2.5,
            color=COLORS[edge_id],
            label=f'Edge {edge_id}',
            alpha=0.85,
        )
    else:
        print(f"Warning: Missing data for Edge {edge_id}")

plt.title("Baseline ECDF of Interarrival Delays", fontsize=14, fontweight='bold')
plt.xlabel("Interarrival Delay (seconds)")
plt.ylabel("F(x) - Cumulative Probability")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()

plot_save_path = os.path.join(SCRIPT_DIR, "baseline_ecdf.png")
plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {plot_save_path}")

plt.show()