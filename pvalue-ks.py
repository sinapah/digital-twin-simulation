import numpy as np
from scipy import stats

real_delays = np.array([
    0.12, 0.15, 0.09, 0.18, 0.22, 0.13, 0.11, 0.20, 0.25,
    0.14, 0.10, 0.16, 0.19, 0.23, 0.30, 0.27, 0.21,
    0.17, 0.15, 0.12, 0.08, 0.11, 0.18, 0.24, 0.29,
    0.26, 0.22, 0.20, 0.16, 0.14, 0.13, 0.10, 0.09,
    0.11, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25,
    0.28, 0.31, 0.27, 0.24, 0.22, 0.20, 0.18,
    0.16, 0.14, 0.12
])

results = []

# Normal
mu, sigma = stats.norm.fit(real_delays)
ks_stat, p_value = stats.kstest(real_delays, 'norm', args=(mu, sigma))
results.append(("Normal", ks_stat, p_value))

# Exponential
loc, scale = stats.expon.fit(real_delays)
ks_stat, p_value = stats.kstest(real_delays, 'expon', args=(loc, scale))
results.append(("Exponential", ks_stat, p_value))

# Gamma
shape, loc, scale = stats.gamma.fit(real_delays)
ks_stat, p_value = stats.kstest(real_delays, 'gamma', args=(shape, loc, scale))
results.append(("Gamma", ks_stat, p_value))

# Beta
a, b, loc, scale = stats.beta.fit(real_delays)
ks_stat, p_value = stats.kstest(real_delays, 'beta', args=(a, b, loc, scale))
results.append(("Beta", ks_stat, p_value))

# Poisson
# Poisson is discrete, so first scale the data
scaled_data = np.round(real_delays * 100).astype(int)
lam = np.mean(scaled_data)

ks_stat, p_value = stats.kstest(
    scaled_data,
    lambda x: stats.poisson.cdf(x, lam)
)
results.append(("Poisson", ks_stat, p_value))

# Print nicely
print(f"{'Distribution':<15} {'KS Statistic':<15} {'P-value'}")
print("-" * 45)

for dist, ks, p in results:
    print(f"{dist:<15} {ks:<15.6f} {p:.6e}")
