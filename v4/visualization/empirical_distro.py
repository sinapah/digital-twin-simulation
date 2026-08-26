#!/usr/bin/env python3
"""
Plot the empirical cumulative distribution function (ECDF) of
interarrival delays for each edge and fit a Gamma distribution.

For each edge, the script:
1. Loads interarrival delays.
2. Plots the empirical CDF.
3. Fits a Gamma distribution.
4. Performs a Kolmogorov-Smirnov goodness-of-fit test.
5. Prints the fitted Gamma parameters and KS test results.
6. Overlays the fitted Gamma CDF on the ECDF.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

COLORS = {
    "Edge 0": "#1f77b4",  # Blue
    "Edge 1": "#e377c2",  # Pink
    "Edge 2": "#2ca02c",  # Green
}


def get_interarrival_times(file_path):
    """Read interarrival delays from CSV."""
    delays = []

    with open(file_path, mode="r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                delay = float(row["interarrival_delay"])

                # Ignore initialization row
                if delay > 0:
                    delays.append(delay)

            except (ValueError, KeyError):
                continue

    return np.array(delays)


# -----------------------------------------------------
# Paths
# -----------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")

files = {
    "Edge 0": os.path.join(OUTPUT_DIR, "edge_0_arrivals.csv"),
    "Edge 1": os.path.join(OUTPUT_DIR, "edge_1_arrivals.csv"),
    "Edge 2": os.path.join(OUTPUT_DIR, "edge_2_arrivals.csv"),
}

# -----------------------------------------------------
# Plot
# -----------------------------------------------------

plt.figure(figsize=(10, 6))

for edge_name, file_path in files.items():

    try:
        delays = get_interarrival_times(file_path)

        if len(delays) == 0:
            print(f"No valid data found for {edge_name}")
            continue

        # -------------------------
        # Empirical CDF
        # -------------------------

        x = np.sort(delays)
        y = np.arange(1, len(x) + 1) / len(x)

        plt.plot(
            x,
            y,
            drawstyle="steps-post",
            linewidth=2,
            color=COLORS[edge_name],
            label=f"{edge_name} ECDF",
        )

        # -------------------------
        # Fit Gamma distribution
        # -------------------------

        # Fix location at zero since interarrival times are positive
        shape, loc, scale = stats.gamma.fit(delays, floc=0)

        # -------------------------
        # Goodness-of-fit (KS test)
        # -------------------------

        ks_statistic, p_value = stats.kstest(
            delays,
            "gamma",
            args=(shape, loc, scale),
        )

        print(f"\n{'=' * 50}")
        print(edge_name)
        print(f"{'=' * 50}")
        print(f"Shape parameter (k): {shape:.6f}")
        print(f"Scale parameter (θ): {scale:.6f}")
        print(f"Location parameter : {loc:.6f}")
        print(f"KS statistic       : {ks_statistic:.6f}")
        print(f"P-value            : {p_value:.6f}")

        if p_value > 0.05:
            print("Conclusion: Gamma distribution is NOT rejected.")
        else:
            print("Conclusion: Gamma distribution is rejected.")

        # -------------------------
        # Plot fitted Gamma CDF
        # -------------------------

        x_fit = np.linspace(0, max(delays), 500)

        gamma_cdf = stats.gamma.cdf(
            x_fit,
            shape,
            loc,
            scale,
        )

        plt.plot(
            x_fit,
            gamma_cdf,
            "--",
            linewidth=2,
            color=COLORS[edge_name],
            alpha=0.8,
            label=f"{edge_name} Gamma Fit",
        )

    except FileNotFoundError:
        print(f"Could not find file: {file_path}")

# -----------------------------------------------------
# Figure formatting
# -----------------------------------------------------

plt.title(
    "Empirical Distribution of Interarrival Delays in the Physical System",
    fontsize=14,
    fontweight="bold",
)

plt.xlabel("Interarrival Delay (seconds)", fontsize=12)
plt.ylabel("Cumulative Distribution", fontsize=12)

plt.xlim(left=0)
plt.ylim(-0.02, 1.02)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()

# -----------------------------------------------------
# Save figure
# -----------------------------------------------------

plot_save_path = os.path.join(SCRIPT_DIR, "ecdf_gamma_fit.pdf")
plt.savefig(plot_save_path, bbox_inches="tight")

print(f"\nPlot successfully saved to: {plot_save_path}")

plt.show()