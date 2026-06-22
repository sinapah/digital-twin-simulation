#!/usr/bin/env python3
"""
Visualization script for federated learning CPU usage across edges.
Reads CSV metrics files and plots CPU usage for each edge over rounds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def load_edge_metrics(outputs_dir='../outputs'):
    """Load metrics CSV files for all edges."""
    edge_data = {}
    
    # Find all edge metrics files
    pattern = os.path.join(outputs_dir, 'edge_*_metrics.csv')
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No metrics files found in {outputs_dir}")
    
    for file_path in files:
        # Extract edge ID from filename
        filename = os.path.basename(file_path)
        edge_id = int(filename.split('_')[1])  # edge_0_metrics.csv -> 0
        
        # Load CSV
        df = pd.read_csv(file_path)
        edge_data[edge_id] = df
        
        print(f"Loaded edge {edge_id}: {len(df)} rounds")
    
    return edge_data

def plot_cpu_usage(edge_data, save_path='cpu_usage.png'):
    """Plot CPU usage for each edge over rounds."""
    plt.figure(figsize=(12, 8))
    
    # Define colors and labels for each edge
    colors = ['blue', 'green', 'red']
    labels = ['Edge 0', 'Edge 1', 'Edge 2']
    
    for edge_id, df in edge_data.items():
        if edge_id >= len(colors):
            # Handle more edges if needed
            color = plt.cm.tab10(edge_id)
            label = f'Edge {edge_id}'
        else:
            color = colors[edge_id]
            label = labels[edge_id]
        
        # Plot CPU average usage
        plt.plot(df['round'], df['cpu_avg'], 
                color=color, linewidth=2, label=f'{label} (Avg)', alpha=0.8)
        
        # Optionally plot CPU peak usage as lighter line
        plt.plot(df['round'], df['cpu_peak'], 
                color=color, linewidth=1, linestyle='--', 
                label=f'{label} (Peak)', alpha=0.5)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.title('CPU Usage Across Edges During Federated Learning', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    # Also show the plot (if running interactively)
    # plt.show()
    
    plt.close()

def plot_cpu_comparison(edge_data, save_path='cpu_comparison.png'):
    """Create a comparison plot showing all edges on same axis."""
    plt.figure(figsize=(14, 8))
    
    # Define colors for each edge
    colors = ['blue', 'green', 'red']
    
    for edge_id, df in edge_data.items():
        if edge_id >= len(colors):
            color = plt.cm.tab10(edge_id)
        else:
            color = colors[edge_id]
        
        plt.plot(df['round'], df['cpu_avg'], 
                color=color, linewidth=2.5, label=f'Edge {edge_id}')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Average CPU Usage (%)', fontsize=12)
    plt.title('CPU Usage Comparison Across Edges', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    
    plt.close()

def main():
    """Main function to load data and create plots."""
    print("Loading edge metrics...")
    edge_data = load_edge_metrics()
    
    if not edge_data:
        print("No edge data found!")
        return
    
    print(f"Found data for {len(edge_data)} edges: {sorted(edge_data.keys())}")
    
    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create plots
    print("\nCreating CPU usage plots...")
    plot_cpu_usage(edge_data, save_path=os.path.join(viz_dir, 'cpu_usage_detailed.png'))
    plot_cpu_comparison(edge_data, save_path=os.path.join(viz_dir, 'cpu_usage_comparison.png'))
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()