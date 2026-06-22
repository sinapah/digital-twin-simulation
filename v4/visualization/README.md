# Visualization Directory

This directory contains scripts for visualizing federated learning metrics.

## Files

- `plot_cpu_usage.py`: Python script that reads CSV metrics files from the outputs directory and creates plots showing CPU usage across edges.

## Generated Plots

When run, the script creates two PNG files:
1. `cpu_usage_detailed.png`: Shows both average and peak CPU usage for each edge (solid lines for average, dashed for peak)
2. `cpu_usage_comparison.png`: Shows average CPU usage for all edges on the same plot for easy comparison

## Usage

To generate the plots:

```bash
# From the visualization directory:
../venv/bin/python3 plot_cpu_usage.py

# Or from the v4 directory:
./venv/bin/python3 visualization/plot_cpu_usage.py
```

## Requirements

- Python 3.x
- pandas
- matplotlib

These are already installed in the v4 virtual environment.