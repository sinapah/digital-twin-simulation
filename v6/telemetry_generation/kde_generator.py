import random
import numpy as np
from typing import Dict, List, Optional

from .base import TelemetryGenerator, TelemetrySample


class KDEGenerator(TelemetryGenerator):
    FIELD_RANGES = {
        'cpu_util': (0.0, 100.0),
        'cpu_avg': (0.0, 100.0),
        'cpu_peak': (0.0, 100.0),
        'memory_util': (0.0, 100.0),
        'memory_rss': (0.0, 8 * 1024 ** 3),
        'queue_length': (0, 500),
        'frame_arrival_rate': (0.0, 500.0),
        'inference_latency': (0.0, 500.0),
        'loss': (0.0, 5.0),
        'accuracy': (0.0, 1.0),
        'samples_trained': (0, 5000),
        'training_time': (0.0, 120.0),
        'communication_delay': (0.0, 30.0),
        'training_progress': (0.0, 10.0),
    }

    FIELD_DEFAULTS = {
        'cpu_util': 50.0,
        'cpu_avg': 50.0,
        'cpu_peak': 60.0,
        'memory_util': 40.0,
        'memory_rss': 2 * 1024 ** 3,
        'queue_length': 10,
        'frame_arrival_rate': 100.0,
        'inference_latency': 50.0,
        'loss': 1.0,
        'accuracy': 0.5,
        'samples_trained': 100,
        'training_time': 30.0,
        'communication_delay': 0.5,
        'training_progress': 3.0,
    }

    def __init__(self):
        self.kde_models: Dict[int, Dict[str, object]] = {}
        self.global_means: Dict[str, float] = {}
        self.global_stds: Dict[str, float] = {}
        self.fitted = False

    def fit(self, history: List[Dict]) -> None:
        if not history:
            return
        try:
            from scipy.stats import gaussian_kde
            for field in self.FIELD_RANGES:
                values = [r.get(field, self.FIELD_DEFAULTS[field])
                          for r in history]
                values = np.array(values, dtype=float)
                self.global_means[field] = float(np.mean(values))
                self.global_stds[field] = max(float(np.std(values)), 1e-6)
                try:
                    kde = gaussian_kde(values, bw_method='silverman')
                    self.kde_models.setdefault(-1, {})[field] = kde
                except Exception:
                    pass
            self.fitted = True
        except ImportError:
            hist_values = []
            for field in self.FIELD_RANGES:
                values = [r.get(field, self.FIELD_DEFAULTS[field])
                          for r in history]
                hist_values.append(np.mean(values) if values else self.FIELD_DEFAULTS[field])
            self.global_means = {f: hist_values[i] for i, f in enumerate(self.FIELD_RANGES)}
            self.fitted = True

    def _sample_field(self, field: str) -> float:
        lo, hi = self.FIELD_RANGES[field]
        default = self.FIELD_DEFAULTS[field]

        if self.fitted and -1 in self.kde_models and field in self.kde_models[-1]:
            try:
                kde = self.kde_models[-1][field]
                sample = float(kde.resample(1)[0, 0])
                return np.clip(sample, lo, hi).item()
            except Exception:
                pass

        if field in self.global_means:
            mean = self.global_means[field]
            std = self.global_stds.get(field, mean * 0.2)
            sample = np.random.normal(mean, max(std, 1e-6))
            return np.clip(sample, lo, hi).item()

        if isinstance(default, float):
            jitter = default * 0.2 * (random.random() - 0.5) * 2
            return np.clip(default + jitter, lo, hi).item()
        return float(default)

    def generate(self, edge_id: int, context: dict) -> TelemetrySample:
        last_states = context.get('last_states', [])
        if last_states:
            self.fit(last_states)

        field_values = {}
        for field in self.FIELD_RANGES:
            field_values[field] = self._sample_field(field)

        return TelemetrySample(
            cpu_util=field_values['cpu_util'],
            cpu_avg=field_values['cpu_avg'],
            cpu_peak=field_values['cpu_peak'],
            memory_util=field_values['memory_util'],
            memory_rss=field_values['memory_rss'],
            queue_length=int(field_values['queue_length']),
            frame_arrival_rate=field_values['frame_arrival_rate'],
            inference_latency=field_values['inference_latency'],
            loss=field_values['loss'],
            accuracy=field_values['accuracy'],
            samples_trained=int(field_values['samples_trained']),
            training_time=field_values['training_time'],
            communication_delay=field_values['communication_delay'],
            training_progress=field_values['training_progress'],
        )
