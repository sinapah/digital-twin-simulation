import csv
import os
import time
from typing import Dict, List, Optional

from .metrics import system_metrics, fl_metrics, dt_metrics, synthetic_telemetry_metrics


class EvaluationLogger:
    def __init__(self, output_dir: str = 'outputs/dt'):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)

        self.system_log = []
        self.fl_log = []
        self.dt_log = []
        self.synthetic_log = []
        self.prediction_error_log = []
        self.recovery_log = []

    def log_system_metrics(self, global_state, edge_states: list):
        sm = system_metrics(edge_states)
        sm['round'] = global_state.round_number
        sm['timestamp'] = time.time()
        self.system_log.append(sm)

    def log_fl_metrics(self, global_state):
        fm = fl_metrics(global_state)
        fm['round'] = global_state.round_number
        fm['timestamp'] = time.time()
        self.fl_log.append(fm)

    def log_dt_metrics(self, state_representation):
        dm = dt_metrics(state_representation)
        dm['round'] = state_representation.global_state.round_number
        dm['timestamp'] = time.time()
        self.dt_log.append(dm)

    def log_synthetic_telemetry(self, edge_id: int, round_num: int,
                                 synthetic: dict, ground_truth: Optional[dict]):
        record = {
            'edge_id': edge_id,
            'round': round_num,
            'timestamp': time.time(),
        }
        for k, v in synthetic.items():
            record[f'syn_{k}'] = v
        if ground_truth:
            metrics = synthetic_telemetry_metrics(synthetic, ground_truth)
            record.update(metrics)
            for k, v in ground_truth.items():
                record[f'actual_{k}'] = v
        self.synthetic_log.append(record)

    def log_prediction_error(self, edge_id: int, round_num: int,
                              predicted: dict, actual: dict):
        metrics = synthetic_telemetry_metrics(predicted, actual)
        record = {
            'edge_id': edge_id,
            'round': round_num,
            'timestamp': time.time(),
            **metrics,
        }
        self.prediction_error_log.append(record)

    def log_outage_recovery(self, edge_id: int, outage_duration: float,
                             recovery_time: float):
        record = {
            'edge_id': edge_id,
            'outage_duration': outage_duration,
            'recovery_time': recovery_time,
            'timestamp': time.time(),
        }
        self.recovery_log.append(record)

    def _write_csv(self, filename: str, records: List[dict]):
        if not records:
            return
        path = os.path.join(self.output_dir, 'metrics', filename)
        fieldnames = list(records[0].keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(records)

    def flush(self):
        self._write_csv('system_metrics.csv', self.system_log)
        self._write_csv('fl_metrics.csv', self.fl_log)
        self._write_csv('dt_metrics.csv', self.dt_log)
        self._write_csv('synthetic_telemetry.csv', self.synthetic_log)
        self._write_csv('prediction_errors.csv', self.prediction_error_log)
        self._write_csv('outage_recovery.csv', self.recovery_log)
