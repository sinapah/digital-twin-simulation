#!/usr/bin/env python3
"""
Visualization script for digital twin simulation results.
Plots CPU usage, loss, and accuracy across edges with outage overlay.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def load_edge_metrics(outputs_dir='../outputs'):
    edge_data = {}
    pattern = os.path.join(outputs_dir, 'edge_*_metrics.csv')
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No metrics files found in {outputs_dir}")

    for file_path in files:
        filename = os.path.basename(file_path)
        edge_id = int(filename.split('_')[1])
        df = pd.read_csv(file_path)
        edge_data[edge_id] = df
        print(f"Loaded edge {edge_id}: {len(df)} rounds")

    return edge_data


def get_outage_periods(edge_data):
    periods = {}
    for edge_id, df in edge_data.items():
        if 'is_outage' not in df.columns:
            continue
        outage_rounds = df[df['is_outage'] == 1]['round']
        if not outage_rounds.empty:
            start = outage_rounds.min()
            end = outage_rounds.max()
            periods[edge_id] = (start, end)
    return periods


def plot_cpu_with_outages(edge_data, save_path='cpu_usage_with_outages.png'):
    _, ax = plt.subplots(figsize=(14, 8))

    colors = ['#1f77b4', '#2ca02c', '#d62728']
    outage_periods = get_outage_periods(edge_data)

    for edge_id, df in edge_data.items():
        color = colors[edge_id % len(colors)]
        ax.plot(df['round'], df['cpu_avg'], color=color, linewidth=2.5,
                label=f'Edge {edge_id} (Avg)', alpha=0.85)
        ax.fill_between(df['round'], df['cpu_avg'], df['cpu_peak'],
                        color=color, alpha=0.1)

    for edge_id, (start, end) in outage_periods.items():
        color = colors[edge_id % len(colors)]
        ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)
        mid = (start + end) / 2
        ax.annotate(f'Edge {edge_id}\nOutage', xy=(mid, ax.get_ylim()[1] * 0.95),
                    ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('CPU Usage (%)', fontsize=12)
    ax.set_title('CPU Usage Across Edges During Federated Learning\n(Shaded bands = outage periods)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"CPU plot saved to {save_path}")
    plt.close()


def plot_training_metrics(edge_data, save_path='training_metrics.png'):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#1f77b4', '#2ca02c', '#d62728']
    outage_periods = get_outage_periods(edge_data)

    for edge_id, df in edge_data.items():
        color = colors[edge_id % len(colors)]
        ax1.plot(df['round'], df['loss'], color=color, linewidth=2,
                 label=f'Edge {edge_id}', alpha=0.85)
        ax2.plot(df['round'], df['accuracy'], color=color, linewidth=2,
                 label=f'Edge {edge_id}', alpha=0.85)

        if edge_id in outage_periods:
            start, end = outage_periods[edge_id]
            for ax in [ax1, ax2]:
                ax.axvspan(start, end, alpha=0.12, color=color, zorder=0)

    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Federated Learning Metrics (shaded = outage periods)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to {save_path}")
    plt.close()


def plot_samples_per_round(edge_data, save_path='samples_per_round.png'):
    _, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#2ca02c', '#d62728']
    outage_periods = get_outage_periods(edge_data)

    for edge_id, df in edge_data.items():
        color = colors[edge_id % len(colors)]
        is_outage = df['is_outage'] == 1 if 'is_outage' in df.columns else None
        normal = df[~is_outage] if is_outage is not None else df
        outage = df[is_outage] if is_outage is not None else pd.DataFrame()

        ax.plot(df['round'], df['samples_trained'], color=color, linewidth=1.5,
                alpha=0.4, linestyle=':')
        ax.scatter(normal['round'], normal['samples_trained'],
                   color=color, s=20, label=f'Edge {edge_id} (Normal)', alpha=0.7)
        if not outage.empty:
            ax.scatter(outage['round'], outage['samples_trained'],
                       color=color, marker='x', s=40,
                       label=f'Edge {edge_id} (Outage)', alpha=0.9)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Samples Trained', fontsize=12)
    ax.set_title('Training Samples Per Round (x = outage period)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Samples plot saved to {save_path}")
    plt.close()


def main():
    print("Loading edge metrics...")
    edge_data = load_edge_metrics()

    if not edge_data:
        print("No edge data found!")
        return

    print(f"Found data for {len(edge_data)} edges: {sorted(edge_data.keys())}")

    outage_periods = get_outage_periods(edge_data)
    if outage_periods:
        print("Outage periods detected:")
        for eid, (s, e) in outage_periods.items():
            print(f"  Edge {eid}: rounds {s}-{e}")

    viz_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(viz_dir, exist_ok=True)

    plot_cpu_with_outages(edge_data, save_path=os.path.join(viz_dir, 'cpu_usage_with_outages.png'))
    plot_training_metrics(edge_data, save_path=os.path.join(viz_dir, 'training_metrics.png'))
    plot_samples_per_round(edge_data, save_path=os.path.join(viz_dir, 'samples_per_round.png'))

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()