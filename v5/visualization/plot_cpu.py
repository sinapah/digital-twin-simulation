#!/usr/bin/env python3
"""
Visualization script for federated learning CPU usage across edges.
Reads CSV metrics files and plots CPU usage for each edge over rounds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def load_edge_metrics(scenario="baseline"):
    outputs_dir=f"../outputs/{scenario}"
    """Load metrics CSV files for all edges."""
    edge_data = {}

    # Find all edge metrics files
    pattern = os.path.join(outputs_dir, "edge_*_metrics.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No metrics files found in {outputs_dir}")

    for file_path in files:
        # Extract edge ID from filename
        filename = os.path.basename(file_path)
        edge_id = int(filename.split("_")[1])  # edge_0_metrics.csv -> 0

        # Load CSV
        df = pd.read_csv(file_path)
        edge_data[edge_id] = df

        print(f"Loaded edge {edge_id}: {len(df)} rounds")

    return edge_data


def plot_cpu_usage(edge_data, save_path="cpu_usage.png"):
    """Plot average CPU usage for each edge over rounds."""
    plt.figure(figsize=(12, 8))

    COLORS = {
        "Edge 0": "#1f77b4",  # Blue
        "Edge 1": "#e377c2",  # Pink
        "Edge 2": "#2ca02c",  # Green
    }

    for edge_id, df in sorted(edge_data.items()):
        label = f"Edge {edge_id}"
        color = COLORS.get(label, plt.cm.tab10(edge_id))

        plt.plot(
            df["round"],
            df["cpu_avg"],
            color=color,
            linewidth=2.5,
            label=label,
        )

    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Average CPU Usage (%)", fontsize=12)
    plt.title(
        "Average CPU Usage Across Edges In the Digital Twin",
        fontsize=14,
        fontweight="bold",
    )

    plt.ylim(170, 210)
    plt.yticks(range(170, 211, 10))

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.close()


def main():
    """Main function."""
    print("Loading edge metrics...")
    edge_data = load_edge_metrics()

    if not edge_data:
        print("No edge data found!")
        return

    print(f"Found data for {len(edge_data)} edges: {sorted(edge_data.keys())}")

    # Save plot in the current directory
    viz_dir = os.path.dirname(__file__)
    os.makedirs(viz_dir, exist_ok=True)

    print("Creating CPU usage plot...")
    plot_cpu_usage(
        edge_data,
        save_path=os.path.join(viz_dir, "cpu_usage_baseline.pdf"),
    )

    print("Done!")


if __name__ == "__main__":
    main()