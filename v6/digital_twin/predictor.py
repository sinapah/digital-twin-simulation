from typing import List, Optional

from .state_representation import (
    EdgeState, GlobalState, PredictedOutcome, Recommendation
)
from .history_repository import HistoryRepository


class Predictor:
    def __init__(self, history: HistoryRepository):
        self.history = history

    def what_if(self, current_global: GlobalState,
                proposed: dict) -> PredictedOutcome:
        base_accuracy = current_global.avg_accuracy
        base_cpu = current_global.avg_cpu
        base_participants = current_global.num_participants

        epochs = proposed.get('local_epochs', 3)
        batch_size = proposed.get('batch_size', 64)
        sampling_rate = proposed.get('frame_sampling_rate', 1.0)
        aggregation_interval = proposed.get('aggregation_interval', 1)

        epoch_ratio = epochs / 3.0
        batch_ratio = 64.0 / max(batch_size, 1)
        sampling_ratio = max(sampling_rate, 0.1)

        latency = 1.0 * epoch_ratio * batch_ratio / sampling_ratio
        queue_length = 10.0 * epoch_ratio / batch_ratio
        resource_util = min(100.0, base_cpu * epoch_ratio * batch_ratio / sampling_ratio)
        fl_throughput = base_participants / max(aggregation_interval, 1) * sampling_ratio
        accuracy = min(1.0, base_accuracy + 0.02 * (epoch_ratio - 1.0))

        return PredictedOutcome(
            latency=round(latency, 4),
            queue_length=round(queue_length, 2),
            resource_utilization=round(resource_util, 2),
            fl_throughput=round(fl_throughput, 2),
            accuracy=round(accuracy, 4),
            loss=round(max(0.0, 1.0 - accuracy), 4),
        )

    def recommend(self, current_global: GlobalState) -> List[Recommendation]:
        candidates = [
            {'local_epochs': 1, 'batch_size': 128, 'frame_sampling_rate': 0.5},
            {'local_epochs': 3, 'batch_size': 64, 'frame_sampling_rate': 1.0},
            {'local_epochs': 5, 'batch_size': 32, 'frame_sampling_rate': 1.0},
            {'local_epochs': 2, 'batch_size': 64, 'frame_sampling_rate': 0.75},
            {'local_epochs': 4, 'batch_size': 64, 'frame_sampling_rate': 1.0},
        ]
        results = []
        for i, cand in enumerate(candidates):
            outcome = self.what_if(current_global, cand)
            desc = (f"epochs={cand['local_epochs']}, batch={cand['batch_size']}, "
                    f"sampling={cand['frame_sampling_rate']}")
            results.append(Recommendation(
                action_id=f'action_{i}',
                description=desc,
                predicted_outcome=outcome,
                confidence=0.7,
            ))
        results.sort(key=lambda r: r.predicted_outcome.accuracy, reverse=True)
        return results[:3]
