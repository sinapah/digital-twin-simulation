import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp

# =========================
# 1. LOAD INTERARRIVAL DATA
# =========================
df = pd.read_csv("interarrival_log.csv")

delays = df["interarrival"].dropna().values
delays = delays[delays > 1e-6]

print(f"Loaded {len(delays)} real samples")

# =========================
# 2. FIT KDE
# =========================
kde = gaussian_kde(delays)

# =========================
# 3. GENERATE SYNTHETIC DATA
# =========================
NUM_SYNTHETIC = len(delays)  # match size
synthetic = kde.resample(NUM_SYNTHETIC).flatten()

# =========================
# 4. APPLY CONSTRAINTS
# =========================
synthetic = synthetic[synthetic > 0]

upper_bound = np.percentile(delays, 99.5)
synthetic = np.clip(synthetic, 0, upper_bound)

print(f"Synthetic samples: {len(synthetic)}")

# =========================
# 5. CONVERT TO MICROSECONDS FOR PLOTTING
# =========================
delays_us = delays * 1e6
synthetic_us = synthetic * 1e6

# =========================
# 6. HISTOGRAM COMPARISON
# =========================
plt.figure(figsize=(8,5))

plt.hist(delays_us, bins=50, density=True, alpha=0.6, label="Original")
plt.hist(synthetic_us, bins=50, density=True, alpha=0.6, label="KDE Synthetic")

plt.title("Histogram Comparison (Interarrival Delays)")
plt.xlabel("Delay (µs)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("histogram_comparison.png", dpi=300)
plt.close()

# =========================
# 7. CDF COMPARISON
# =========================
def plot_cdf(data, label):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label)

plt.figure(figsize=(8,5))

plot_cdf(delays_us, "Original")
plot_cdf(synthetic_us, "KDE Synthetic")

plt.title("CDF Comparison (Interarrival Delays)")
plt.xlabel("Delay (µs)")
plt.ylabel("CDF")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cdf_comparison.png", dpi=300)
plt.close()

# =========================
# 8. KS TEST
# =========================
ks_stat, p_value = ks_2samp(delays, synthetic)

print("\nKS Test Result:")
print(f"KS Statistic: {ks_stat:.6f}")
print(f"P-value: {p_value:.6e}")

# =========================
# 9. SAVE SYNTHETIC DATA
# =========================
np.savetxt("synthetic_interarrival_kde.csv", synthetic, delimiter=",")

print("\nSaved:")
print("- histogram_comparison.png")
print("- cdf_comparison.png")
print("- synthetic_interarrival_kde.csv")