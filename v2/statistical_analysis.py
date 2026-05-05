import numpy as np
import pandas as pd
from scipy import stats

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("interarrival_log.csv")

delays = df["interarrival"].dropna().values
delays = delays[delays > 1e-6]

print(f"Loaded {len(delays)} samples")

# =========================
# FIT + KS TEST
# =========================
results = []

# Normal
mu, sigma = stats.norm.fit(delays)
ks, p = stats.kstest(delays, 'norm', args=(mu, sigma))
results.append(("Normal", ks, p))

# Exponential
loc, scale = stats.expon.fit(delays)
ks, p = stats.kstest(delays, 'expon', args=(loc, scale))
results.append(("Exponential", ks, p))

# Gamma
shape, loc, scale = stats.gamma.fit(delays)
ks, p = stats.kstest(delays, 'gamma', args=(shape, loc, scale))
results.append(("Gamma", ks, p))

# Beta (requires scaling to [0,1])
min_d, max_d = delays.min(), delays.max()
scaled = (delays - min_d) / (max_d - min_d + 1e-9)

a, b, loc, scale = stats.beta.fit(scaled)
ks, p = stats.kstest(scaled, 'beta', args=(a, b, loc, scale))
results.append(("Beta", ks, p))

# =========================
# PRINT TABLE
# =========================
print(f"\n{'Distribution':<15} {'KS Value':<12} {'P-value'}")
print("-" * 45)

for name, ks, p in results:
    print(f"{name:<15} {ks:<12.6f} {p:.6e}")