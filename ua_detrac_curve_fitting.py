import numpy as np
from scipy import stats

# --------------------------------------------
# STEP 1: SIMULATE UA-DETRAC FRAME TIMESTAMPS
# --------------------------------------------
FPS = 25
N_FRAMES = 500

# Ideal timestamps
timestamps = np.arange(N_FRAMES) / FPS

# Add realistic jitter (simulate capture/network variation)
np.random.seed(42)
jitter = np.random.normal(0, 0.002, size=N_FRAMES)

timestamps_jittered = timestamps + jitter

# --------------------------------------------
# STEP 2: COMPUTE INTERARRIVAL DELAYS
# --------------------------------------------
real_delays = np.diff(timestamps_jittered)

# Ensure positivity
real_delays = np.clip(real_delays, 1e-6, None)

# --------------------------------------------
# STEP 3: FIT DISTRIBUTIONS (YOUR ORIGINAL CODE)
# --------------------------------------------
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

# Poisson (discrete approximation)
scaled_data = np.round(real_delays * 1000).astype(int)
lam = np.mean(scaled_data)

ks_stat, p_value = stats.kstest(
    scaled_data,
    lambda x: stats.poisson.cdf(x, lam)
)
results.append(("Poisson", ks_stat, p_value))

# --------------------------------------------
# STEP 4: PRINT RESULTS
# --------------------------------------------
print(f"{'Distribution':<15} {'KS Statistic':<15} {'P-value'}")
print("-" * 45)

for dist, ks, p in results:
    print(f"{dist:<15} {ks:<15.6f} {p:.6e}")
