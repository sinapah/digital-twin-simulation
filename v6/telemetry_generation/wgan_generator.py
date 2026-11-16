import random
import numpy as np
from typing import Dict, List, Optional

from .base import TelemetryGenerator, TelemetrySample
from .kde_generator import KDEGenerator


class WGANGenerator(TelemetryGenerator):
    LATENT = 16
    HIDDEN = 64
    EPOCHS = 200
    BATCH = 128
    N_CRITIC = 5
    GP_LAMBDA = 10
    NUM_FIELDS = 14

    def __init__(self):
        self.generator_net = None
        self.critic_net = None
        self.fitted = False
        self.fallback = KDEGenerator()

    def _build_networks(self):
        import torch
        import torch.nn as nn

        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(WGANGenerator.LATENT, WGANGenerator.HIDDEN),
                    nn.ReLU(),
                    nn.Linear(WGANGenerator.HIDDEN, WGANGenerator.HIDDEN),
                    nn.ReLU(),
                    nn.Linear(WGANGenerator.HIDDEN, WGANGenerator.NUM_FIELDS),
                )

            def forward(self, z):
                return self.net(z)

        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(WGANGenerator.NUM_FIELDS, WGANGenerator.HIDDEN),
                    nn.LeakyReLU(0.2),
                    nn.Linear(WGANGenerator.HIDDEN, WGANGenerator.HIDDEN),
                    nn.LeakyReLU(0.2),
                    nn.Linear(WGANGenerator.HIDDEN, 1),
                )

            def forward(self, x):
                return self.net(x)

        self.generator_net = Generator()
        self.critic_net = Critic()

    def fit(self, history: List[Dict]) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        if len(history) < self.BATCH:
            self.fallback.fit(history)
            return

        def gradient_penalty(critic, real, fake):
            alpha = torch.rand(real.size(0), 1)
            interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
            d_interp = critic(interp)
            grads = torch.autograd.grad(
                d_interp, interp,
                grad_outputs=torch.ones_like(d_interp),
                create_graph=True,
            )[0]
            return ((grads.norm(2, dim=1) - 1) ** 2).mean()

        fields = [
            'cpu_util', 'cpu_avg', 'cpu_peak', 'memory_util', 'memory_rss',
            'queue_length', 'frame_arrival_rate', 'inference_latency',
            'loss', 'accuracy', 'samples_trained', 'training_time',
            'communication_delay', 'training_progress',
        ]

        self._field_names = fields
        self._field_means = {}
        self._field_stds = {}

        data_list = []
        for f in fields:
            values = [r.get(f, 0.0) for r in history]
            vals = np.array(values, dtype=np.float32)
            mean = float(np.mean(vals))
            std = max(float(np.std(vals)), 1e-6)
            self._field_means[f] = mean
            self._field_stds[f] = std
            data_list.append((vals - mean) / std)

        data = np.stack(data_list, axis=1)
        data_t = torch.FloatTensor(data)

        self._build_networks()
        G = self.generator_net
        C = self.critic_net
        opt_g = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
        opt_c = optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))

        for epoch in range(self.EPOCHS):
            idx = torch.randperm(len(data_t))[:self.BATCH]
            real = data_t[idx]
            for _ in range(self.N_CRITIC):
                z = torch.randn(self.BATCH, self.LATENT)
                fake = G(z).detach()
                loss_c = (
                    -(C(real).mean() - C(fake).mean())
                    + self.GP_LAMBDA * gradient_penalty(C, real, fake)
                )
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
            z = torch.randn(self.BATCH, self.LATENT)
            loss_g = -C(G(z)).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        self.generator_net = G
        self.fitted = True
        print(f"[WGANGenerator] Trained on {len(data_t)} samples for {self.EPOCHS} epochs")

    def generate(self, edge_id: int, context: dict) -> TelemetrySample:
        import torch

        if not self.fitted or self.generator_net is None:
            return self.fallback.generate(edge_id, context)

        last_states = context.get('last_states', [])
        if last_states and not self.fitted:
            self.fit(last_states)

        if not self.fitted or self.generator_net is None:
            return self.fallback.generate(edge_id, context)

        try:
            G = self.generator_net
            G.eval()
            with torch.no_grad():
                z = torch.randn(1, self.LATENT)
                sample = G(z).squeeze().numpy()

            fields = [
                'cpu_util', 'cpu_avg', 'cpu_peak', 'memory_util', 'memory_rss',
                'queue_length', 'frame_arrival_rate', 'inference_latency',
                'loss', 'accuracy', 'samples_trained', 'training_time',
                'communication_delay', 'training_progress',
            ]

            result = {}
            for i, f in enumerate(fields):
                val = sample[i]
                mean = self._field_means.get(f, 0.0)
                std = self._field_stds.get(f, 1.0)
                result[f] = float(val * std + mean)

            safe_clamp = {
                'cpu_util': (0.0, 100.0),
                'cpu_avg': (0.0, 100.0),
                'cpu_peak': (0.0, 100.0),
                'memory_util': (0.0, 100.0),
                'memory_rss': (0.0, None),
                'queue_length': (0.0, None),
                'frame_arrival_rate': (0.0, None),
                'inference_latency': (0.0, None),
                'loss': (0.0, None),
                'accuracy': (0.0, 1.0),
                'samples_trained': (0.0, None),
                'training_time': (0.0, None),
                'communication_delay': (0.0, None),
                'training_progress': (0.0, None),
            }

            for f, (lo, hi) in safe_clamp.items():
                val = result[f]
                if hi is not None:
                    val = min(val, hi)
                val = max(val, lo)
                result[f] = val

            return TelemetrySample(
                cpu_util=result['cpu_util'],
                cpu_avg=result['cpu_avg'],
                cpu_peak=result['cpu_peak'],
                memory_util=result['memory_util'],
                memory_rss=result['memory_rss'],
                queue_length=int(result['queue_length']),
                frame_arrival_rate=result['frame_arrival_rate'],
                inference_latency=result['inference_latency'],
                loss=result['loss'],
                accuracy=result['accuracy'],
                samples_trained=int(result['samples_trained']),
                training_time=result['training_time'],
                communication_delay=result['communication_delay'],
                training_progress=result['training_progress'],
            )
        except Exception as e:
            print(f"[WGANGenerator] Error: {e}, falling back to KDE")
            return self.fallback.generate(edge_id, context)
