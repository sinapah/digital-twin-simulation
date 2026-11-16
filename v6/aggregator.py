import socket
import threading
import torch
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.simple_cnn import SimpleCNN
from utils.tcp_comm import send_msg, recv_msg, send_weights, recv_weights, fedavg


AGGREGATOR_PORT = 5000
NUM_EDGES = 3
NUM_ROUNDS = 100
DT_PORT = 6000
ORCHESTRATOR_CONTROL_PORT = 6200


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class Aggregator:
    def __init__(self, host: str = '127.0.0.1', port: int = AGGREGATOR_PORT,
                 num_rounds: int = NUM_ROUNDS,
                 dt_host: str = '127.0.0.1', dt_port: int = DT_PORT,
                 orchestrator_control_port: int = ORCHESTRATOR_CONTROL_PORT):
        self.host = host
        self.port = port
        self.num_rounds = num_rounds
        self.dt_host = dt_host
        self.dt_port = dt_port
        self.orchestrator_control_port = orchestrator_control_port

        self.model = SimpleCNN(num_classes=4)
        self.edge_connections = {}
        self.edge_weights = {}
        self.edge_status = {}
        self.lock = threading.Lock()
        self.current_round = 0
        self.server = None
        self.running = True

        self.dt_client = None
        self.dt_lock = threading.Lock()

        self.current_control = {}
        self.control_lock = threading.Lock()

        self.control_server = None
        self.control_thread = None

        self.round_telemetry_log = []

        flush_print(f"[Aggregator] Initialized on {host}:{port}")
        flush_print(f"[Aggregator] DT forwarding to {dt_host}:{dt_port}")
        flush_print(f"[Aggregator] Orchestrator control on port {orchestrator_control_port}")

    def _connect_dt(self):
        try:
            self.dt_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dt_client.settimeout(10.0)
            self.dt_client.connect((self.dt_host, self.dt_port))
            flush_print(f"[Aggregator] Connected to DT at {self.dt_host}:{self.dt_port}")
        except Exception as e:
            flush_print(f"[Aggregator] DT not available: {e}")
            self.dt_client = None

    def _forward_telemetry_to_dt(self):
        if self.dt_client is None:
            return
        with self.lock:
            status_copy = dict(self.edge_status)

        round_start_time = getattr(self, '_round_start_time', time.time())
        round_end_time = time.time()

        fl_metrics = {
            'aggregation_delay': round(round_end_time - round_start_time, 4),
            'num_participants': len(status_copy),
            'round_duration': round(round_end_time - round_start_time, 4),
        }

        msg = {
            'type': 'TELEMETRY_UPDATE',
            'round': self.current_round,
            'edge_status': {str(k): v for k, v in status_copy.items()},
            'fl_metrics': fl_metrics,
        }

        try:
            send_msg(self.dt_client, msg)
        except Exception as e:
            flush_print(f"[Aggregator] Error forwarding telemetry: {e}")
            with self.dt_lock:
                self.dt_client = None

    def _start_control_server(self):
        self.control_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_server.bind(('127.0.0.1', self.orchestrator_control_port))
        self.control_server.listen(5)
        self.control_thread = threading.Thread(target=self._control_accept_loop, daemon=True)
        self.control_thread.start()
        flush_print(f"[Aggregator] Control server on port {self.orchestrator_control_port}")

    def _control_accept_loop(self):
        while self.running:
            try:
                conn, addr = self.control_server.accept()
                flush_print(f"[Aggregator] Orchestrator connected on control channel "
                            f"from {addr}")
                t = threading.Thread(target=self._handle_orchestrator,
                                     args=(conn,), daemon=True)
                t.start()
            except Exception:
                if self.running:
                    time.sleep(0.5)

    def _handle_orchestrator(self, conn: socket.socket):
        while self.running:
            try:
                data = recv_msg(conn)
                if data is None:
                    break
                if data.get('type') == 'CONTROL_ACTION':
                    action = data.get('action', {})
                    with self.control_lock:
                        self.current_control = action
                    send_msg(conn, {'type': 'CONTROL_ACK', 'round': data.get('round', -1)})
                    flush_print(f"[Aggregator] Received control action: {action}")
            except Exception:
                break
        try:
            conn.close()
        except Exception:
            pass

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        flush_print(f"[Aggregator] Listening on {self.host}:{self.port}")

        for _ in range(NUM_EDGES):
            conn, addr = self.server.accept()
            data = recv_msg(conn)
            if data and data.get('type') == 'CONNECT':
                edge_id = data['edge_id']
                with self.lock:
                    self.edge_connections[edge_id] = conn
                flush_print(f"[Aggregator] Edge {edge_id} connected from {addr}")
                send_msg(conn, {'type': 'CONNECTED', 'edge_id': edge_id})

        flush_print(f"[Aggregator] All {NUM_EDGES} edges connected")
        self._connect_dt()
        self._start_control_server()

    def _send_round_start(self, edge_id, round_num, is_outage):
        conn = self.edge_connections[edge_id]
        with self.control_lock:
            control_copy = dict(self.current_control)
        try:
            msg = {'type': 'ROUND_START', 'round': round_num, 'is_outage': is_outage}
            if control_copy:
                msg['control'] = control_copy
            send_msg(conn, msg)
        except Exception as e:
            flush_print(f"[Aggregator] Error sending ROUND_START to Edge {edge_id}: {e}")

    def _recv_round_complete(self, edge_id):
        conn = self.edge_connections[edge_id]
        try:
            data = recv_msg(conn)
            if data and data.get('type') == 'ROUND_COMPLETE':
                return data
        except Exception as e:
            flush_print(f"[Aggregator] Error receiving ROUND_COMPLETE from Edge {edge_id}: {e}")
        return None

    def _handle_edge_round(self, edge_id: int):
        conn = self.edge_connections[edge_id]
        try:
            data = recv_msg(conn)
            if data and data.get('type') == 'WEIGHTS_UPLOAD':
                weights = recv_weights(conn)
                with self.lock:
                    self.edge_weights[edge_id] = weights

                data = recv_msg(conn)
                if data and data.get('type') == 'STATUS':
                    with self.lock:
                        self.edge_status[edge_id] = data

                flush_print(f"[Aggregator] Received weights + status from Edge {edge_id}")
        except Exception as e:
            flush_print(f"[Aggregator] Error receiving from Edge {edge_id}: {e}")

    def _send_weights_to_edge(self, edge_id: int, aggregated_weights: dict):
        conn = self.edge_connections[edge_id]
        try:
            send_msg(conn, {'type': 'WEIGHTS_UPDATE', 'round': self.current_round})
            send_weights(conn, aggregated_weights)
            flush_print(f"[Aggregator] Sent updated weights to Edge {edge_id}")
        except Exception as e:
            flush_print(f"[Aggregator] Error sending to Edge {edge_id}: {e}")

    def run_round(self, round_num: int, outage_flags=None):
        self.current_round = round_num
        self.edge_weights = {}
        self.edge_status = {}

        flush_print(f"\n[Aggregator] === Round {round_num} ===")

        if outage_flags is None:
            outage_flags = [False] * NUM_EDGES

        self._round_start_time = time.time()

        start_threads = []
        for edge_id in range(NUM_EDGES):
            t = threading.Thread(target=self._send_round_start,
                                 args=(edge_id, round_num, outage_flags[edge_id]))
            start_threads.append(t)
            t.start()
        for t in start_threads:
            t.join(timeout=30)

        recv_threads = []
        for edge_id in range(NUM_EDGES):
            t = threading.Thread(target=self._handle_edge_round, args=(edge_id,))
            recv_threads.append(t)
            t.start()

        for t in recv_threads:
            t.join(timeout=120)

        if len(self.edge_weights) < NUM_EDGES:
            flush_print(f"[Aggregator] WARNING: Only received weights from "
                        f"{len(self.edge_weights)}/{NUM_EDGES} edges")
            missing = set(range(NUM_EDGES)) - set(self.edge_weights.keys())
            flush_print(f"[Aggregator] Missing edges: {missing}")
            self._forward_telemetry_to_dt()
            return self.edge_status

        weights_list = [self.edge_weights[i] for i in range(NUM_EDGES)]
        aggregated = fedavg(weights_list)
        self.model.load_state_dict(aggregated)

        avg_loss = sum(self.edge_status.get(i, {}).get('loss', 0) for i in range(NUM_EDGES)) / NUM_EDGES
        avg_acc = sum(self.edge_status.get(i, {}).get('accuracy', 0) for i in range(NUM_EDGES)) / NUM_EDGES
        flush_print(f"[Aggregator] FedAvg complete. Avg loss: {avg_loss:.4f}, Avg acc: {avg_acc:.4f}")

        send_threads = []
        for edge_id in range(NUM_EDGES):
            t = threading.Thread(target=self._send_weights_to_edge, args=(edge_id, aggregated))
            send_threads.append(t)
            t.start()

        for t in send_threads:
            t.join(timeout=60)

        complete_threads = []
        for edge_id in range(NUM_EDGES):
            t = threading.Thread(target=self._recv_round_complete, args=(edge_id,))
            complete_threads.append(t)
            t.start()

        for t in complete_threads:
            t.join(timeout=30)

        self._forward_telemetry_to_dt()

        return self.edge_status

    def shutdown(self):
        self.running = False
        for edge_id, conn in self.edge_connections.items():
            try:
                conn.close()
            except:
                pass
        if self.dt_client:
            try:
                self.dt_client.close()
            except:
                pass
        if self.control_server:
            try:
                self.control_server.close()
            except:
                pass
        if self.server:
            self.server.close()
        flush_print(f"[Aggregator] Shutdown complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V6 Aggregator with DT Forwarding')
    parser.add_argument('--port', type=int, default=AGGREGATOR_PORT)
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS)
    parser.add_argument('--dt-port', type=int, default=DT_PORT)
    parser.add_argument('--orchestrator-control-port', type=int,
                        default=ORCHESTRATOR_CONTROL_PORT)
    args = parser.parse_args()

    agg = Aggregator(
        port=args.port,
        num_rounds=args.rounds,
        dt_port=args.dt_port,
        orchestrator_control_port=args.orchestrator_control_port,
    )
    agg.start()

    for r in range(args.rounds):
        agg.run_round(r)

    flush_print(f"\n[Aggregator] All {args.rounds} rounds complete. Shutting down.")
    agg.shutdown()


if __name__ == '__main__':
    main()
