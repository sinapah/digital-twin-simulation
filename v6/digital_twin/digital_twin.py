import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .state_representation import StateRepresentation
from .history_repository import HistoryRepository, TelemetryRecord
from .outage_detector import OutageDetector, OutageStatus
from .telemetry_collector import TelemetryCollector
from .predictor import Predictor
from telemetry_generation.base import TelemetryGenerator
from telemetry_generation.kde_generator import KDEGenerator
from telemetry_generation.wgan_generator import WGANGenerator
from evaluation.logger import EvaluationLogger


DT_PORT = 6000
ORCHESTRATOR_PORT = 6100
NUM_EDGES = 3


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


TELEMETRY_FIELD_MAP = {
    'cpu_util': 'cpu_current',
    'cpu_avg': 'cpu_avg',
    'cpu_peak': 'cpu_peak',
    'memory_util': 'memory_util',
    'memory_rss': 'memory_rss',
    'queue_length': 'queue_length',
    'frame_arrival_rate': 'frame_arrival_rate',
    'inference_latency': 'inference_latency',
    'loss': 'loss',
    'accuracy': 'accuracy',
    'samples_trained': 'samples_trained',
    'training_time': 'training_time',
    'communication_delay': 'communication_delay',
    'training_progress': 'training_progress',
}


class DigitalTwin:
    def __init__(self, num_edges: int = NUM_EDGES,
                 telemetry_port: int = DT_PORT,
                 output_dir: str = 'outputs/dt',
                 synthesis_mode: str = 'none'):
        self.num_edges = num_edges
        self.telemetry_port = telemetry_port
        self.output_dir = output_dir
        self.synthesis_mode = synthesis_mode

        self.state = StateRepresentation(num_edges=num_edges)
        self.history = HistoryRepository(output_dir=os.path.join(output_dir, 'history'))
        self.detector = OutageDetector(timeout_seconds=30.0)
        self.collector = TelemetryCollector(port=telemetry_port)
        self.predictor = Predictor(self.history)
        self.logger = EvaluationLogger(output_dir=output_dir)

        self.generator: TelemetryGenerator = None
        if synthesis_mode == 'kde':
            self.generator = KDEGenerator()
        elif synthesis_mode == 'wgan':
            self.generator = WGANGenerator()

        self.telemetry_count = 0
        self.synthetic_count = 0
        self.outage_count = 0
        self.recovery_count = 0

        self.orchestrator_host = '127.0.0.1'
        self.orchestrator_port = ORCHESTRATOR_PORT
        self.orchestrator_conn = None

        self._orchestrator_recommendations = []

    def connect_orchestrator(self):
        import socket
        try:
            self.orchestrator_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.orchestrator_conn.settimeout(5.0)
            self.orchestrator_conn.connect((self.orchestrator_host, self.orchestrator_port))
            flush_print(f"[DT] Connected to orchestrator at "
                        f"{self.orchestrator_host}:{self.orchestrator_port}")
        except Exception as e:
            flush_print(f"[DT] Orchestrator not available: {e}")
            self.orchestrator_conn = None

    def send_recommendations(self, recommendations: list):
        if self.orchestrator_conn is None:
            return
        from utils.tcp_comm import send_msg
        try:
            recs = []
            for r in recommendations:
                recs.append({
                    'action_id': r.action_id,
                    'description': r.description,
                    'predicted_outcome': {
                        'latency': r.predicted_outcome.latency,
                        'queue_length': r.predicted_outcome.queue_length,
                        'resource_utilization': r.predicted_outcome.resource_utilization,
                        'fl_throughput': r.predicted_outcome.fl_throughput,
                        'accuracy': r.predicted_outcome.accuracy,
                        'loss': r.predicted_outcome.loss,
                    },
                    'confidence': r.confidence,
                })
            send_msg(self.orchestrator_conn, {
                'type': 'RECOMMENDATIONS',
                'round': self.state.global_state.round_number,
                'recommendations': recs,
            })
        except Exception as e:
            flush_print(f"[DT] Error sending recommendations: {e}")

    def on_telemetry(self, msg: dict):
        msg_type = msg.get('type')
        if msg_type == 'TELEMETRY_UPDATE':
            self._handle_telemetry_update(msg)
        elif msg_type == 'ORCHESTRATOR_ACTION':
            self._handle_orchestrator_action(msg)
        elif msg_type == 'SHUTDOWN':
            self.collector.stop()

    def _handle_telemetry_update(self, msg: dict):
        round_num = msg.get('round', -1)
        fl_metrics = msg.get('fl_metrics', {})
        edge_statuses = msg.get('edge_status', {})

        self.state.global_state.round_number = round_num
        self.state.global_state.fl_rounds_completed = round_num + 1
        self.state.global_state.aggregation_delay = fl_metrics.get('aggregation_delay', 0.0)
        self.state.global_state.num_participants = fl_metrics.get('num_participants', 0)

        for edge_id_str, status in edge_statuses.items():
            edge_id = int(edge_id_str)
            status_outage = self.detector.check(edge_id)

            if status_outage.just_recovered:
                self.recovery_count += 1
                self._log_recovery_metrics(edge_id)

            if status_outage.is_outage and self.generator is not None:
                self.outage_count += 1
                last_states = self.history.get_recent(edge_id, 5)
                context = {
                    'edge_id': edge_id,
                    'round_number': round_num,
                    'last_states': [r.fields for r in last_states],
                    'global_state': {
                        'avg_cpu': self.state.global_state.avg_cpu,
                        'avg_accuracy': self.state.global_state.avg_accuracy,
                    }
                }
                synthetic = self.generator.generate(edge_id, context)
                fields = {}
                for state_key, telemetry_key in TELEMETRY_FIELD_MAP.items():
                    val = getattr(synthetic, state_key, None)
                    if val is not None:
                        fields[state_key] = val

                self.state.update_edge(edge_id, is_outage=True, is_synthetic=True,
                                       round_number=round_num, timestamp=time.time(),
                                       **fields)
                record = TelemetryRecord(edge_id, round_num, fields,
                                         is_synthetic=True)
                self.history.append(record)
                self.synthetic_count += 1

                self.logger.log_synthetic_telemetry(
                    edge_id=edge_id, round_num=round_num,
                    synthetic=fields, ground_truth=None,
                )
            else:
                fields = {}
                for state_key, telemetry_key in TELEMETRY_FIELD_MAP.items():
                    val = status.get(telemetry_key)
                    if val is not None:
                        fields[state_key] = val

                self.state.update_edge(edge_id, is_outage=False, is_synthetic=False,
                                       round_number=round_num, timestamp=time.time(),
                                       **fields)
                self.detector.record_telemetry(edge_id, status)
                record = TelemetryRecord(edge_id, round_num, fields,
                                         is_synthetic=False)
                self.history.append(record)
                self.telemetry_count += 1

                last_synthetic = [r for r in self.history.get_recent(edge_id, 3)
                                  if r.is_synthetic]
                if last_synthetic:
                    syn = last_synthetic[-1]
                    self.logger.log_prediction_error(
                        edge_id=edge_id, round_num=round_num,
                        predicted=syn.fields, actual=fields,
                    )

            self.detector.record_telemetry(edge_id, status)

        self.state.recompute_global()
        self.logger.log_system_metrics(
            global_state=self.state.global_state,
            edge_states=self.state.edges,
        )

        self.history.flush(round_number=round_num)

        if round_num >= 0 and round_num % 5 == 0:
            recs = self.predictor.recommend(self.state.global_state)
            self._orchestrator_recommendations = recs
            self.send_recommendations(recs)

        if round_num % 10 == 0:
            flush_print(
                f"[DT] Round {round_num}: "
                f"telemetry={self.telemetry_count}, "
                f"synthetic={self.synthetic_count}, "
                f"outages={self.outage_count}, "
                f"recoveries={self.recovery_count}"
            )
            gs = self.state.global_state
            flush_print(
                f"[DT] Global: cpu={gs.avg_cpu:.1f}%, "
                f"mem={gs.avg_memory:.1f}%, "
                f"acc={gs.avg_accuracy:.4f}"
            )

    def _handle_orchestrator_action(self, msg: dict):
        flush_print(f"[DT] Received action acknowledgement: {msg.get('action_id', 'N/A')}")

    def _log_recovery_metrics(self, edge_id: int):
        info = self.detector.get_recovery_info(edge_id)
        if info:
            self.logger.log_outage_recovery(
                edge_id=edge_id,
                outage_duration=info.get('duration', 0.0),
                recovery_time=info.get('recovery_time', time.time()),
            )

    def start(self):
        flush_print(f"[DT] Starting Digital Twin (synthesis={self.synthesis_mode})")
        os.makedirs(self.output_dir, exist_ok=True)
        self.collector.on_telemetry(self.on_telemetry)
        self.collector.start()
        self.connect_orchestrator()

    def stop(self):
        flush_print(f"[DT] Stopping Digital Twin")
        self.logger.flush()
        self.history.flush()
        self.collector.stop()
        flush_print(f"[DT] Summary: telemetry={self.telemetry_count}, "
                    f"synthetic={self.synthetic_count}, "
                    f"outages={self.outage_count}, "
                    f"recoveries={self.recovery_count}")

    def get_recommendations(self):
        return list(self._orchestrator_recommendations)
