import socket
import threading
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.simple_cnn import SimpleCNN
from utils.tcp_comm import send_msg, recv_msg, send_weights, recv_weights, fedavg


AGGREGATOR_PORT = 5000
NUM_EDGES = 3
NUM_ROUNDS = 100


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class Aggregator:
    def __init__(self, host: str = '127.0.0.1', port: int = AGGREGATOR_PORT,
                 num_rounds: int = NUM_ROUNDS):
        self.host = host
        self.port = port
        self.num_rounds = num_rounds
        self.model = SimpleCNN(num_classes=4)
        self.edge_connections = {}
        self.edge_weights = {}
        self.edge_status = {}
        self.lock = threading.Lock()
        self.current_round = 0
        self.server = None
        self.running = True

        flush_print(f"[Aggregator] Initialized on {host}:{port}")

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

    def run_round(self, round_num: int):
        self.current_round = round_num
        self.edge_weights = {}
        self.edge_status = {}

        flush_print(f"\n[Aggregator] === Round {round_num} ===")

        threads = []
        for edge_id in range(NUM_EDGES):
            t = threading.Thread(target=self._handle_edge_round, args=(edge_id,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=120)

        if len(self.edge_weights) < NUM_EDGES:
            flush_print(f"[Aggregator] WARNING: Only received weights from "
                        f"{len(self.edge_weights)}/{NUM_EDGES} edges")
            missing = set(range(NUM_EDGES)) - set(self.edge_weights.keys())
            flush_print(f"[Aggregator] Missing edges: {missing}")
            return

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

    def shutdown(self):
        self.running = False
        for edge_id, conn in self.edge_connections.items():
            try:
                conn.close()
            except:
                pass
        if self.server:
            self.server.close()
        flush_print(f"[Aggregator] Shutdown complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Aggregator for Federated Learning Digital Twin')
    parser.add_argument('--port', type=int, default=AGGREGATOR_PORT)
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS)
    args = parser.parse_args()

    agg = Aggregator(port=args.port, num_rounds=args.rounds)
    agg.start()

    for r in range(args.rounds):
        agg.run_round(r)

    flush_print(f"\n[Aggregator] All {args.rounds} rounds complete. Shutting down.")
    agg.shutdown()


if __name__ == '__main__':
    main()