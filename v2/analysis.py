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
# FIT DISTRIBUTIONS
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

# Beta (requires scaling to [0,1])
min_d, max_d = delays.min(), delays.max()
scaled = (delays - min_d) / (max_d - min_d + 1e-9)

a, b, loc, scale = stats.beta.fit(scaled)
ks, p = stats.kstest(scaled, 'beta', args=(a, b, loc, scale))
results.append(("Beta", ks, p, (a, b, loc, scale)))

# =========================
# PRINT RESULTS
# =========================
print(f"\n{'Distribution':<15} {'KS Stat':<12} {'P-value'}")
print("-"*40)

for name, ks, p, _ in results:
    print(f"{name:<15} {ks:<12.6f} {p:.6e}")

# =========================
# PLOT
# =========================
x = np.linspace(delays.min(), delays.max(), 500)

plt.figure(figsize=(8,5))

# Histogram
plt.hist(delays, bins=50, density=True, alpha=0.5, label="Data")

# Overlay fits
for name, _, _, params in results:
    if name == "Normal":
        y = stats.norm.pdf(x, *params)
    elif name == "Exponential":
        y = stats.expon.pdf(x, *params)
    elif name == "Gamma":
        y = stats.gamma.pdf(x, *params)
    elif name == "Beta":
        # Scale x to [0,1] for Beta
        x_scaled = (x - min_d) / (max_d - min_d + 1e-9)
        y = stats.beta.pdf(x_scaled, *params) / (max_d - min_d + 1e-9)
    else:
        continue

    plt.plot(x, y, label=name)

plt.title("Interarrival Delay Distribution Fit")
plt.xlabel("Delay (s)")
plt.ylabel("Density")
plt.legend()
plt.grid()

# Save plot
plt.tight_layout()
plt.savefig("distribution_fit.png", dpi=300)
plt.show()