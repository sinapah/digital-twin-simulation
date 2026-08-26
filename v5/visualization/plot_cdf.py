#!/usr/bin/env python3
"""
Empirical Cumulative Distribution Function (ECDF) + Gamma Fit
for Interarrival Delays (single scenario).
"""

import csv
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

scenario = "wgan"

# Directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory
BASELINE_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', 'outputs', scenario)
)

# Colors for edges
COLORS = {
    0: '#1f77b4',
    1: '#e377c2',
    2: '#2ca02c'
}

# -----------------------------------------------------
# Load data
# -----------------------------------------------------

def get_interarrival_times(file_path):
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
        return np.array([])

    return np.array(delays)

# -----------------------------------------------------
# Plot
# -----------------------------------------------------

plt.figure(figsize=(8, 6))

for edge_id in range(3):
    path = os.path.join(BASELINE_DIR, f'edge_{edge_id}_arrivals.csv')

    delays = get_interarrival_times(path)

    if len(delays) == 0:
        print(f"Warning: Missing data for Edge {edge_id}")
        continue

    # -------------------------
    # ECDF
    # -------------------------
    x = np.sort(delays)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.plot(
        x,
        y,
        drawstyle='steps-post',
        linewidth=2.5,
        color=COLORS[edge_id],
        label=f'Edge {edge_id} ECDF',
        alpha=0.85
    )

    # -------------------------
    # Gamma fitting
    # -------------------------
    shape, loc, scale = stats.gamma.fit(delays, floc=0)

    # KS test
    ks_stat, p_value = stats.kstest(
        delays,
        'gamma',
        args=(shape, loc, scale)
    )

    print(f"\n{'='*60}")
    print(f"Edge {edge_id}")
    print(f"{'='*60}")
    print(f"Shape (k): {shape:.4f}")
    print(f"Scale (θ): {scale:.6f}")
    print(f"KS statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # -------------------------
    # Gamma CDF overlay
    # -------------------------
    x_fit = np.linspace(0, max(delays), 400)
    gamma_cdf = stats.gamma.cdf(x_fit, shape, loc, scale)

    plt.plot(
        x_fit,
        gamma_cdf,
        '--',
        color=COLORS[edge_id],
        linewidth=2,
        alpha=0.7
    )

# -----------------------------------------------------
# Figure formatting
# -----------------------------------------------------

plt.title(
    f"ECDF + Gamma Fit of Interarrival Delays ({scenario.upper()})",
    fontsize=14,
    fontweight='bold'
)

plt.xlabel("Interarrival Delay (seconds)")
plt.ylabel("Cumulative Distribution")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()

# -----------------------------------------------------
# Save
# -----------------------------------------------------

plot_save_path = os.path.join(SCRIPT_DIR, f"{scenario}_ecdf_gamma.pdf")
plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")

print(f"\nPlot saved to: {plot_save_path}")

plt.show()