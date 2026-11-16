from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EdgeState:
    edge_id: int
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
    is_outage: bool = False
    is_synthetic: bool = False
    timestamp: float = 0.0
    round_number: int = -1


@dataclass
class GlobalState:
    round_number: int = -1
    num_edges_connected: int = 0
    avg_cpu: float = 0.0
    avg_memory: float = 0.0
    avg_accuracy: float = 0.0
    avg_loss: float = 0.0
    fl_rounds_completed: int = 0
    aggregation_delay: float = 0.0
    client_participation_rate: float = 1.0
    num_participants: int = 0


@dataclass
class PredictedOutcome:
    latency: float = 0.0
    queue_length: float = 0.0
    resource_utilization: float = 0.0
    fl_throughput: float = 0.0
    accuracy: float = 0.0
    loss: float = 0.0


@dataclass
class Recommendation:
    action_id: str = ''
    description: str = ''
    predicted_outcome: PredictedOutcome = field(default_factory=PredictedOutcome)
    confidence: float = 0.0


class StateRepresentation:
    def __init__(self, num_edges: int = 3):
        self.num_edges = num_edges
        self.edges: List[EdgeState] = [EdgeState(edge_id=i) for i in range(num_edges)]
        self.global_state = GlobalState()

    def update_edge(self, edge_id: int, **kwargs):
        if 0 <= edge_id < self.num_edges:
            for k, v in kwargs.items():
                if hasattr(self.edges[edge_id], k):
                    setattr(self.edges[edge_id], k, v)

    def update_global(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.global_state, k):
                setattr(self.global_state, k, v)

    def get_edge(self, edge_id: int) -> EdgeState:
        if 0 <= edge_id < self.num_edges:
            return self.edges[edge_id]
        return EdgeState(edge_id=edge_id)

    def recompute_global(self):
        connected = [e for e in self.edges if not e.is_outage]
        self.global_state.num_edges_connected = len(connected)
        self.global_state.avg_cpu = (
            sum(e.cpu_util for e in connected) / len(connected) if connected else 0.0
        )
        self.global_state.avg_memory = (
            sum(e.memory_util for e in connected) / len(connected) if connected else 0.0
        )
        self.global_state.avg_accuracy = (
            sum(e.accuracy for e in connected) / len(connected) if connected else 0.0
        )
        self.global_state.avg_loss = (
            sum(e.loss for e in connected) / len(connected) if connected else 0.0
        )
        return self.global_state
