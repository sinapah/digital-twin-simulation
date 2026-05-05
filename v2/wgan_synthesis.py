import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp

# =========================================================
# 1. LOAD REAL DATA
# =========================================================
df = pd.read_csv("interarrival_log.csv")

delays = df["interarrival"].dropna().values
delays = delays[delays > 1e-6]

print(f"Loaded {len(delays)} samples")

# =========================================================
# 2. NORMALIZE
# =========================================================
mean = delays.mean()
std = delays.std()
normalized = (delays - mean) / std

tensor_data = torch.tensor(normalized, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# 3. MODELS
# =========================================================
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(1, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 1))
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# 4. HYPERPARAMETERS
# =========================================================
z_dim = 16
lr = 2e-4
n_epochs = 50   # MUCH lower now (large dataset)
critic_iters = 3
ema_decay = 0.999

gen = Generator(z_dim).to(device)
crit = Critic().to(device)

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(0.5, 0.9))

ema_gen = Generator(z_dim).to(device)
ema_gen.load_state_dict(gen.state_dict())

# =========================================================
# 5. TRAINING
# =========================================================
print("Training WGAN...")

for epoch in range(n_epochs):
    for real, in dataloader:
        real = real.to(device)

        # slight noise
        real = real + 0.01 * torch.randn_like(real)

        # ----- Critic -----
        for _ in range(critic_iters):
            z = torch.randn(real.size(0), z_dim, device=device)
            fake = gen(z).detach()

            loss_crit = -(torch.mean(crit(real)) - torch.mean(crit(fake)))

            crit_opt.zero_grad()
            loss_crit.backward()
            crit_opt.step()

        # ----- Generator -----
        z = torch.randn(real.size(0), z_dim, device=device)
        fake = gen(z)
        loss_gen = -torch.mean(crit(fake))

        gen_opt.zero_grad()
        loss_gen.backward()
        gen_opt.step()

        # EMA update
        with torch.no_grad():
            for p_ema, p in zip(ema_gen.parameters(), gen.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

    print(f"Epoch {epoch+1}/{n_epochs} | C: {loss_crit.item():.4f} | G: {loss_gen.item():.4f}")

print("Training Complete\n")

# =========================================================
# 6. GENERATE SYNTHETIC DATA
# =========================================================
with torch.no_grad():
    z = torch.randn(len(delays), z_dim, device=device)
    generated = ema_gen(z).cpu().numpy().flatten()

# denormalize
generated = generated * std + mean

# enforce constraints
generated = generated[generated > 0]
upper = np.percentile(delays, 99.5)
generated = np.clip(generated, 0, upper)

print(f"Generated samples: {len(generated)}")

# =========================================================
# 7. KS TEST
# =========================================================
ks_stat, p_val = ks_2samp(delays, generated)

print("\nKS Test:")
print(f"KS: {ks_stat:.6f}")
print(f"P-value: {p_val:.6e}")

# =========================================================
# 8. CONVERT TO MICROSECONDS
# =========================================================
real_us = delays * 1e6
gen_us = generated * 1e6

# =========================================================
# 9. HISTOGRAM
# =========================================================
plt.figure(figsize=(8,5))
plt.hist(real_us, bins=50, density=True, alpha=0.6, label="Real")
plt.hist(gen_us, bins=50, density=True, alpha=0.6, label="WGAN")
plt.title("Histogram Comparison (WGAN)")
plt.xlabel("Delay (µs)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wgan_histogram.png", dpi=300)
plt.close()

# =========================================================
# 10. CDF
# =========================================================
def plot_cdf(data, label):
    s = np.sort(data)
    cdf = np.arange(1, len(s)+1) / len(s)
    plt.plot(s, cdf, label=label)

plt.figure(figsize=(8,5))
plot_cdf(real_us, "Real")
plot_cdf(gen_us, "WGAN")
plt.title("CDF Comparison (WGAN)")
plt.xlabel("Delay (µs)")
plt.ylabel("CDF")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wgan_cdf.png", dpi=300)
plt.close()

# =========================================================
# 11. SAVE
# =========================================================
np.savetxt("synthetic_interarrival_wgan.csv", generated, delimiter=",")

print("\nSaved:")
print("- wgan_histogram.png")
print("- wgan_cdf.png")
print("- synthetic_interarrival_wgan.csv")