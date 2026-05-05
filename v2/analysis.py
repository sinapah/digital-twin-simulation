import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("interarrival_log.csv")

# Drop NaNs and very small noise values
delays = df["interarrival"].dropna().values
delays = delays[delays > 1e-6]

print(f"Loaded {len(delays)} samples")

# =========================
# FIT DISTRIBUTIONS (in seconds)
# =========================
results = []

# Normal
mu, sigma = stats.norm.fit(delays)
ks, p = stats.kstest(delays, 'norm', args=(mu, sigma))
results.append(("Normal", ks, p, (mu, sigma)))

# Exponential
loc, scale = stats.expon.fit(delays)
ks, p = stats.kstest(delays, 'expon', args=(loc, scale))
results.append(("Exponential", ks, p, (loc, scale)))

# Gamma
shape, loc, scale = stats.gamma.fit(delays)
ks, p = stats.kstest(delays, 'gamma', args=(shape, loc, scale))
results.append(("Gamma", ks, p, (shape, loc, scale)))

# Beta (requires scaling)
min_d, max_d = delays.min(), delays.max()
scaled = (delays - min_d) / (max_d - min_d + 1e-9)

a, b, loc, scale = stats.beta.fit(scaled)
ks, p = stats.kstest(scaled, 'beta', args=(a, b, loc, scale))
results.append(("Beta", ks, p, (a, b, loc, scale)))

# =========================
# PRINT RESULTS
# =========================
print(f"\n{'Distribution':<15} {'KS Stat':<12} {'P-value'}")
print("-" * 40)

for name, ks, p, _ in results:
    print(f"{name:<15} {ks:<12.6f} {p:.6e}")

# =========================
# CONVERT TO MICROSECONDS
# =========================
delays_us = delays * 1e6

# x-axis for plotting (in µs)
x_us = np.linspace(delays_us.min(), delays_us.max(), 500)

# Convert back to seconds for PDF evaluation
x_sec = x_us / 1e6

# =========================
# PLOT
# =========================
plt.figure(figsize=(8,5))

# Histogram (µs)
plt.hist(delays_us, bins=50, density=True, alpha=0.5, label="Data")

# Overlay fits
for name, _, _, params in results:
    if name == "Normal":
        y = stats.norm.pdf(x_sec, *params)
    elif name == "Exponential":
        y = stats.expon.pdf(x_sec, *params)
    elif name == "Gamma":
        y = stats.gamma.pdf(x_sec, *params)
    elif name == "Beta":
        x_scaled = (x_sec - min_d) / (max_d - min_d + 1e-9)
        y = stats.beta.pdf(x_scaled, *params) / (max_d - min_d + 1e-9)
    else:
        continue

    plt.plot(x_us, y, label=name)

# Labels
plt.title("Interarrival Delay Distribution Fit")
plt.xlabel("Delay (µs)")
plt.ylabel("Density")
plt.legend()
plt.grid()

# Optional zoom (removes extreme tails)
low, high = np.percentile(delays_us, [1, 99])
plt.xlim(low, high)

# Save plot
plt.tight_layout()
plt.savefig("distribution_fit.png", dpi=300)
plt.show()