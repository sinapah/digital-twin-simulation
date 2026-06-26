import csv
import os
import matplotlib.pyplot as plt

def get_empirical_distribution(file_path):
    delays = []
    
    with open(file_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                delay = float(row['interarrival_delay'])
                if delay > 0.0:  # Skip the initialization row
                    delays.append(delay)
            except (ValueError, KeyError):
                continue
                
    delays.sort()
    n = len(delays)
    if n == 0:
        return [], []
    
    # Separate into x (delays) and y (probabilities) arrays for plotting
    x_coords = []
    y_coords = []
    for index, delay in enumerate(delays):
        probability = (index + 1) / n
        x_coords.append(delay)
        y_coords.append(probability)
        
    return x_coords, y_coords

# 1. Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')

files = {
    'Edge 0': os.path.join(OUTPUT_DIR, 'edge_0_arrivals.csv'),
    'Edge 1': os.path.join(OUTPUT_DIR, 'edge_1_arrivals.csv'),
    'Edge 2': os.path.join(OUTPUT_DIR, 'edge_2_arrivals.csv')
}

# 2. Plotting setup
plt.figure(figsize=(10, 6))

# 3. Process each file and add it to the plot
for edge_name, file_path in files.items():
    try:
        x, y = get_empirical_distribution(file_path)
        
        if x and y:
            # drawstyle='steps-post' accurately represents the discrete nature of an ECDF
            plt.plot(x, y, label=edge_name, drawstyle='steps-post', linewidth=2)
            print(f"Added {edge_name} to the plot.")
        else:
            print(f"Warning: No valid data found for {edge_name}")
            
    except FileNotFoundError:
        print(f"Error: Could not find file at '{file_path}'")

# 4. Customize and show the plot
plt.title('Real System - Empirical Cumulative Distribution Function (ECDF) of Interarrival Delays', fontsize=14)
plt.xlabel('Interarrival Delay (seconds)', fontsize=12)
plt.ylabel('F(x) - Cumulative Probability', fontsize=12)
plt.ylim(-0.05, 1.05)  # Padding to clearly see the 0 and 1 bounds
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Network Edge')

# Save the plot in the visualizations folder
plot_save_path = os.path.join(SCRIPT_DIR, 'ecdf_plot.png')
plt.savefig(plot_save_path, dpi=300)
print(f"\nPlot successfully saved as image: {plot_save_path}")

# Display the window
plt.show()