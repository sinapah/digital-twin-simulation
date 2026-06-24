#!/usr/bin/env python3
"""
Visualization script for digital twin simulation results.
Hardcoded to automatically overlay v5/kde, v5/wgan, and v4 (Real System)
CPU metrics onto a single unified plot with vertical outage indicators.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def load_edge_metrics(outputs_dir):
    edge_data = {}
    pattern = os.path.join(outputs_dir, 'edge_*_metrics.csv')
    files = glob.glob(pattern)

    if not files:
        print(f"Warning: No metrics files found in {outputs_dir}")
        return {}

    for file_path in files:
        filename = os.path.basename(file_path)
        edge_id = int(filename.split('_')[1])
        df = pd.read_csv(file_path)
        edge_data[edge_id] = df
        print(f"Loaded edge {edge_id} from {outputs_dir.split('/')[-2:]}")

    return edge_data


def get_outage_periods(datasets_list):
    """Scans multiple simulation datasets to map out unique outage windows per edge."""
    periods = {}
    for dataset in datasets_list:
        for edge_id, df in dataset.items():
            if 'is_outage' not in df.columns:
                continue
            outage_rounds = df[df['is_outage'] == 1]['round']
            if not outage_rounds.empty:
                start = outage_rounds.min()
                end = outage_rounds.max()
                # Store or update the wider outage range if frameworks vary slightly
                if edge_id in periods:
                    periods[edge_id] = (min(periods[edge_id][0], start), max(periods[edge_id][1], end))
                else:
                    periods[edge_id] = (start, end)
    return periods


def main():
    # 1. Define paths directly as requested
    kde_dir = '../outputs/kde'
    wgan_dir = '../outputs/wgan'
    v4_dir = '../../v4/outputs'

    # 2. Load all datasets
    print("Loading datasets...")
    kde_data = load_edge_metrics(kde_dir)
    wgan_data = load_edge_metrics(wgan_dir)
    v4_data = load_edge_metrics(v4_dir)

    # 3. Initialize the plot
    _, ax = plt.subplots(figsize=(14, 8))

    # Base color palette mapped to Edge IDs
    # Edge 0 = Blue, Edge 1 = Green, Edge 2 = Red
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    # --- Plot KDE (Solid lines) ---
    for edge_id, df in kde_data.items():
        color = colors[edge_id % len(colors)]
        ax.plot(df['round'], df['cpu_avg'], color=color, linestyle='-', linewidth=2,
                label=f'v5 KDE | Edge {edge_id}', alpha=0.9)

    # --- Plot WGAN (Dash-Dot lines) ---
    for edge_id, df in wgan_data.items():
        color = colors[edge_id % len(colors)]
        ax.plot(df['round'], df['cpu_avg'], color=color, linestyle='-.', linewidth=2,
                label=f'v5 WGAN | Edge {edge_id}', alpha=0.9)

    # --- Plot V4 Real System (Dashed lines) ---
    for edge_id, df in v4_data.items():
        color = colors[edge_id % len(colors)]
        ax.plot(df['round'], df['cpu_avg'], color=color, linestyle='--', linewidth=2.5,
                label=f'v4 Real System | Edge {edge_id}', alpha=0.5)

    # --- Overlay Outages ---
    # Scans both active frameworks for outage flags
    outage_periods = get_outage_periods([kde_data, wgan_data])
    for edge_id, (start, end) in outage_periods.items():
        color = colors[edge_id % len(colors)]
        # Draw translucent vertical band background
        ax.axvspan(start, end, alpha=0.10, color=color, zorder=0)
        
        # Draw a subtle top marker text identifying the outage
        mid = (start + end) / 2
        ax.annotate(f'Edge {edge_id}\nOutage', xy=(mid, ax.get_ylim()[1] * 0.90),
                    ha='center', fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))

    # 4. Styling and adjustments
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('CPU Usage (%)', fontsize=12)
    ax.set_title('Unified CPU Usage Comparison with Outage Overlay\nv5 Frameworks (KDE & WGAN) vs. v4 Real System Baseline',
                 fontsize=14, fontweight='bold')
    
    # Place the legend to the right so it stays clean and legible
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    ax.grid(True, alpha=0.3)
    
    # Save output in the script's directory
    viz_dir = os.path.dirname(__file__)
    save_path = os.path.join(viz_dir, 'unified_cpu_comparison_with_outages.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Unified plot with outages saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()