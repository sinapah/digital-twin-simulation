from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TelemetrySample:
    cpu_util: float = 0.0
    cpu_avg: float = 0.0
    cpu_peak: float = 0.0
    memory_util: float = 0.0
    memory_rss: float = 0.0
    queue_length: int = 0
    frame_arrival_rate: float = 0.0
    inference_latency: float = 0.0
    loss: float = 0.0
    accuracy: float = 0.0
    samples_trained: int = 0
    training_time: float = 0.0
    communication_delay: float = 0.0
    training_progress: float = 0.0


class TelemetryGenerator(ABC):
    @abstractmethod
    def generate(self, edge_id: int, context: dict) -> TelemetrySample:
        ...

    @abstractmethod
    def fit(self, history: List[Dict]) -> None:
        ...

    def name(self) -> str:
        return self.__class__.__name__
