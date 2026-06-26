#!/usr/bin/env python3
"""
fit_delays.py — Fit KDE and WGAN models to baseline interarrival delays.

Usage:
    python3 fit_delays.py [--arrivals-dir outputs/baseline] [--out-dir ../v2] [--samples 100000]

Reads:  outputs/baseline/edge_*_arrivals.csv  (produced by baseline mode)
Writes: ../v2/synthetic_interarrival_kde.csv
        ../v2/synthetic_interarrival_wgan.csv
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_delays(arrivals_dir: str) -> np.ndarray:
    """Load and concatenate interarrival_delay values from all edge CSVs."""
    pattern = os.path.join(arrivals_dir, 'edge_*_arrivals.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No arrival CSVs found in {arrivals_dir}")

    delays = []
    for f in files:
        df = pd.read_csv(f)
        if 'interarrival_delay' not in df.columns:
            print(f"  Skipping {f}: no interarrival_delay column")
            continue
        d = df['interarrival_delay'].dropna().values
        d = d[d > 0]          # drop zero first-arrival
        d = d[d < np.percentile(d, 99)]  # drop extreme outliers (round gaps)
        delays.append(d)
        print(f"  Loaded {len(d)} delays from {os.path.basename(f)}")

    if not delays:
        raise ValueError("No valid delay data found")

    all_delays = np.concatenate(delays)
    print(f"  Total: {len(all_delays)} delay samples "
          f"(min={all_delays.min():.4f}s, "
          f"mean={all_delays.mean():.4f}s, "
          f"max={all_delays.max():.4f}s)")
    return all_delays


def fit_and_sample_kde(delays: np.ndarray, n_samples: int) -> np.ndarray:
    """KDE using scipy if available, otherwise numpy histogram resampling."""
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(delays, bw_method='silverman')
        samples = kde.resample(n_samples).flatten()
        samples = np.clip(samples, 0, None)
        print(f"  KDE fitted (scipy gaussian_kde, silverman bandwidth), "
              f"sampled {n_samples} values")
    except ImportError:
        # Fallback: histogram-based resampling
        hist, edges = np.histogram(delays, bins=200, density=True)
        cdf = np.cumsum(hist * np.diff(edges))
        cdf = np.concatenate([[0], cdf / cdf[-1]])
        u = np.random.uniform(0, 1, n_samples)
        samples = np.interp(u, cdf, edges)
        print(f"  KDE fitted (histogram fallback), sampled {n_samples} values")
    return samples


def fit_and_sample_wgan(delays: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Lightweight conditional WGAN-GP in pure PyTorch.
    Generator maps noise → delay; Critic scores real vs fake.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("  PyTorch not available — falling back to KDE for WGAN output")
        return fit_and_sample_kde(delays, n_samples)

    LATENT = 32
    HIDDEN = 64
    EPOCHS = 300
    BATCH = 256
    N_CRITIC = 5
    GP_LAMBDA = 10

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(LATENT, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                nn.Linear(HIDDEN, 1), nn.Softplus()
            )
        def forward(self, z):
            return self.net(z)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, HIDDEN), nn.LeakyReLU(0.2),
                nn.Linear(HIDDEN, HIDDEN), nn.LeakyReLU(0.2),
                nn.Linear(HIDDEN, 1)
            )
        def forward(self, x):
            return self.net(x)

    def gradient_penalty(critic, real, fake):
        alpha = torch.rand(real.size(0), 1)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = critic(interp)
        grads = torch.autograd.grad(d_interp, interp,
                                    grad_outputs=torch.ones_like(d_interp),
                                    create_graph=True)[0]
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    data = torch.FloatTensor(delays).unsqueeze(1)
    G = Generator()
    C = Critic()
    opt_g = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_c = optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(EPOCHS):
        idx = torch.randperm(len(data))[:BATCH]
        real = data[idx]
        for _ in range(N_CRITIC):
            z = torch.randn(BATCH, LATENT)
            fake = G(z).detach()
            loss_c = -(C(real).mean() - C(fake).mean()) + GP_LAMBDA * gradient_penalty(C, real, fake)
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()
        z = torch.randn(BATCH, LATENT)
        loss_g = -C(G(z)).mean()
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if (epoch + 1) % 100 == 0:
            print(f"  WGAN epoch {epoch+1}/{EPOCHS} — "
                  f"C loss: {loss_c.item():.4f}, G loss: {loss_g.item():.4f}")

    with torch.no_grad():
        z = torch.randn(n_samples, LATENT)
        samples = G(z).squeeze().numpy()

    print(f"  WGAN trained and sampled {n_samples} values")
    return samples


def write_csv(samples: np.ndarray, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(samples, columns=['interarrival_delay']).to_csv(
        path, index=False, header=False)
    print(f"  Saved {len(samples)} samples → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fit KDE & WGAN to baseline delays')
    parser.add_argument('--arrivals-dir', type=str, default='outputs/baseline',
                        help='Directory with edge_*_arrivals.csv (default: outputs/baseline)')
    parser.add_argument('--out-dir', type=str, default='../v2',
                        help='Directory to write synthetic delay CSVs (default: ../v2)')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Number of synthetic samples to generate (default: 100000)')
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"Delay Model Fitting from Baseline Arrivals")
    print(f"{'='*55}")

    print(f"\n[1] Loading baseline delays from {args.arrivals_dir}...")
    delays = load_delays(args.arrivals_dir)

    print(f"\n[2] Fitting KDE and sampling {args.samples} values...")
    kde_samples = fit_and_sample_kde(delays, args.samples)
    kde_path = os.path.join(args.out_dir, 'synthetic_interarrival_kde.csv')
    write_csv(kde_samples, kde_path)

    print(f"\n[3] Training WGAN and sampling {args.samples} values...")
    wgan_samples = fit_and_sample_wgan(delays, args.samples)
    wgan_path = os.path.join(args.out_dir, 'synthetic_interarrival_wgan.csv')
    write_csv(wgan_samples, wgan_path)

    print(f"\n{'='*55}")
    print(f"Done. Files written:")
    print(f"  {kde_path}")
    print(f"  {wgan_path}")
    print(f"Now run:")
    print(f"  python3 simulator.py --mode kde --rounds 100")
    print(f"  python3 simulator.py --mode wgan --rounds 100")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
