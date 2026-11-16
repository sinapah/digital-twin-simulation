import csv
import os
import time
from typing import List, Optional


class TelemetryRecord:
    def __init__(self, edge_id: int, round_number: int, fields: dict,
                 is_synthetic: bool = False, timestamp: float = None):
        self.edge_id = edge_id
        self.round_number = round_number
        self.fields = dict(fields)
        self.is_synthetic = is_synthetic
        self.timestamp = timestamp or time.time()

    def __repr__(self):
        return (f"TelemetryRecord(edge={self.edge_id}, round={self.round_number}, "
                f"synthetic={self.is_synthetic})")


HISTORY_FIELDS = [
    'edge_id', 'round_number', 'timestamp', 'is_synthetic',
    'cpu_util', 'cpu_avg', 'cpu_peak', 'memory_util', 'memory_rss',
    'queue_length', 'frame_arrival_rate', 'inference_latency',
    'loss', 'accuracy', 'samples_trained', 'training_time',
    'communication_delay', 'training_progress',
]


class HistoryRepository:
    def __init__(self, output_dir: str = 'outputs/dt/history'):
        self.output_dir = output_dir
        self.records: List[TelemetryRecord] = []
        self._flushed_rounds = set()

    def append(self, record: TelemetryRecord):
        self.records.append(record)

    def get_recent(self, edge_id: int, n: int = 10) -> List[TelemetryRecord]:
        matching = [r for r in self.records if r.edge_id == edge_id]
        return matching[-n:]

    def get_all(self, edge_id: int) -> List[TelemetryRecord]:
        return [r for r in self.records if r.edge_id == edge_id]

    def get_range(self, round_start: int, round_end: int) -> List[TelemetryRecord]:
        return [
            r for r in self.records
            if round_start <= r.round_number <= round_end
        ]

    def get_recent_global(self, n: int = 10) -> List[TelemetryRecord]:
        return self.records[-n:]

    def get_ground_truth(self, edge_id: int) -> List[TelemetryRecord]:
        return [r for r in self.records
                if r.edge_id == edge_id and not r.is_synthetic]

    def flush(self, round_number: int = None):
        if round_number is not None and round_number in self._flushed_rounds:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'telemetry_history.csv')
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            if write_header:
                w.writeheader()
            for r in self.records:
                row = {k: r.fields.get(k, '') for k in HISTORY_FIELDS}
                row['edge_id'] = r.edge_id
                row['round_number'] = r.round_number
                row['timestamp'] = r.timestamp
                row['is_synthetic'] = int(r.is_synthetic)
                w.writerow(row)
        if round_number is not None:
            self._flushed_rounds.add(round_number)

    def clear(self):
        self.records.clear()
        self._flushed_rounds.clear()
