from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ControlAction:
    local_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    frame_sampling_rate: Optional[float] = None
    aggregation_interval: Optional[int] = None
    client_participation: Optional[List[bool]] = None
    max_crops_per_round: Optional[int] = None
    sender_fps: Optional[int] = None

    def to_dict(self) -> dict:
        d = {}
        if self.local_epochs is not None:
            d['local_epochs'] = self.local_epochs
        if self.batch_size is not None:
            d['batch_size'] = self.batch_size
        if self.learning_rate is not None:
            d['learning_rate'] = self.learning_rate
        if self.frame_sampling_rate is not None:
            d['frame_sampling_rate'] = self.frame_sampling_rate
        if self.aggregation_interval is not None:
            d['aggregation_interval'] = self.aggregation_interval
        if self.client_participation is not None:
            d['client_participation'] = self.client_participation
        if self.max_crops_per_round is not None:
            d['max_crops_per_round'] = self.max_crops_per_round
        if self.sender_fps is not None:
            d['sender_fps'] = self.sender_fps
        return d

    @staticmethod
    def from_dict(d: dict) -> 'ControlAction':
        return ControlAction(
            local_epochs=d.get('local_epochs'),
            batch_size=d.get('batch_size'),
            learning_rate=d.get('learning_rate'),
            frame_sampling_rate=d.get('frame_sampling_rate'),
            aggregation_interval=d.get('aggregation_interval'),
            client_participation=d.get('client_participation'),
            max_crops_per_round=d.get('max_crops_per_round'),
            sender_fps=d.get('sender_fps'),
        )

    @staticmethod
    def default() -> 'ControlAction':
        return ControlAction(
            local_epochs=3,
            batch_size=64,
            learning_rate=1e-3,
            frame_sampling_rate=1.0,
            aggregation_interval=1,
            client_participation=[True, True, True],
            max_crops_per_round=0,
            sender_fps=500,
        )


@dataclass
class AppliedAction:
    action: ControlAction = field(default_factory=ControlAction)
    round_number: int = -1
    source: str = ''
    predicted_accuracy: float = 0.0
    actual_accuracy: float = 0.0
