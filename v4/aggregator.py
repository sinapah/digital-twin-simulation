# =========================================================
# V4 Aggregator - Federated Learning Coordinator
# =========================================================
# Runs on the aggregator VM
# Receives weights from all edge VMs, performs FedAvg, broadcasts updated weights
# =========================================================

import argparse
import socket
import threading
import torch
import json
import time
import sys
import io
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

DEFAULT_PORT = 5000
NUM_EDGES = 3
AGGREGATION_INTERVAL = 5
DEFAULT_ROUNDS = 100


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def send_msg(sock: socket.socket, msg_dict: dict):
    data = json.dumps(msg_dict).encode()
    size_prefix = len(data).to_bytes(8, 'big')
    sock.sendall(size_prefix + data)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    size_data = b''
    while len(size_data) < 8:
        chunk = sock.recv(8 - len(size_data))
        if not chunk:
            return None
        size_data += chunk
    msg_size = int.from_bytes(size_data, 'big')
    msg_data = b''
    while len(msg_data) < msg_size:
        chunk = sock.recv(min(65536, msg_size - len(msg_data)))
        if not chunk:
            return None
        msg_data += chunk
    return json.loads(msg_data.decode())


def send_weights(sock: socket.socket, weights: dict):
    buf = io.BytesIO()
    torch.save(weights, buf)
    data = buf.getvalue()
    size_prefix = len(data).to_bytes(8, 'big')
    sock.sendall(size_prefix + data)


def recv_weights(sock: socket.socket) -> Optional[dict]:
    size_data = b''
    while len(size_data) < 8:
        chunk = sock.recv(8 - len(size_data))
        if not chunk:
            return None
        size_data += chunk
    msg_size = int.from_bytes(size_data, 'big')
    msg_data = b''
    while len(msg_data) < msg_size:
        chunk = sock.recv(min(65536, msg_size - len(msg_data)))
        if not chunk:
            return None
        msg_data += chunk
    buf = io.BytesIO(msg_data)
    return torch.load(buf, weights_only=False)


@dataclass
class EdgeConnection:
    edge_id: int
    sock: socket.socket
    addr: Tuple[str, int]
    last_heartbeat: float
    weights_received: bool
    local_weights: Optional[Dict] = None


class Aggregator:

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.edges: Dict[int, EdgeConnection] = {}
        self.lock = threading.Lock()
        self.round_complete = threading.Event()
        self.global_weights = None
        self.round_count = 0
        self.metrics = []
        self.message_threads = {}
        flush_print(f"[Aggregator] Initialized on port {port}")
        flush_print(f"[Aggregator] Waiting for {NUM_EDGES} edge VMs to connect...")

    def start(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('0.0.0.0', self.port))
        server_sock.listen(5)
        flush_print(f"[Aggregator] Listening on port {self.port}")

        accept_thread = threading.Thread(target=self._accept_connections, args=(server_sock,))
        accept_thread.daemon = True
        accept_thread.start()

        self._aggregation_loop()

    def _accept_connections(self, server_sock: socket.socket):
        connected_count = 0
        while connected_count < NUM_EDGES:
            sock, addr = server_sock.accept()
            flush_print(f"[Aggregator] Connection from {addr}")

            data = sock.recv(1024).decode()
            msg = json.loads(data)

            if msg['type'] == 'CONNECT':
                edge_id = msg['edge_id']
                with self.lock:
                    self.edges[edge_id] = EdgeConnection(
                        edge_id=edge_id, sock=sock, addr=addr,
                        last_heartbeat=time.time(), weights_received=False
                    )
                connected_count += 1
                flush_print(f"[Aggregator] Edge {edge_id} connected ({connected_count}/{NUM_EDGES})")

                response = json.dumps({'type': 'CONNECTED', 'round': 0})
                sock.sendall(response.encode())

                message_thread = threading.Thread(target=self._handle_edge_messages, args=(edge_id,))
                message_thread.daemon = True
                message_thread.start()
                self.message_threads[edge_id] = message_thread

    def _handle_edge_messages(self, edge_id: int):
        edge = self.edges[edge_id]
        sock = edge.sock
        sock.settimeout(5.0)

        while True:
            try:
                msg = recv_msg(sock)
                if msg is None:
                    flush_print(f"[Aggregator] Edge {edge_id} disconnected")
                    return

                with self.lock:
                    edge = self.edges.get(edge_id)
                    if edge:
                        self.handle_edge_message(edge, sock, msg)
            except socket.timeout:
                time.sleep(0.01)
            except Exception as e:
                flush_print(f"[Aggregator] Error handling messages from edge {edge_id}: {e}")
                time.sleep(1)

    def _aggregation_loop(self):
        while True:
            if len(self.edges) < NUM_EDGES:
                time.sleep(0.1)
                continue

            flush_print(f"\n[Aggregator] Starting round {self.round_count}")
            self.round_complete.clear()

            self._broadcast_round_start()
            self._wait_for_weights()
            self._aggregate_weights()
            self._broadcast_weights()

            self.round_count += 1
            flush_print(f"[Aggregator] Round {self.round_count - 1} complete")
            
            if self.round_count >= DEFAULT_ROUNDS:
                flush_print(f"[Aggregator] Completed {DEFAULT_ROUNDS} rounds, shutting down...")
                break

            with self.lock:
                for edge in self.edges.values():
                    edge.weights_received = False

    def _broadcast_round_start(self):
        msg = {'type': 'ROUND_START', 'round': self.round_count}
        with self.lock:
            for edge in self.edges.values():
                try:
                    send_msg(edge.sock, msg)
                except Exception as e:
                    flush_print(f"[Aggregator] Error sending to edge {edge.edge_id}: {e}")

    def _wait_for_weights(self):
        timeout = 300
        start_time = time.time()
        while True:
            with self.lock:
                all_received = all(edge.weights_received for edge in self.edges.values())
            if all_received:
                break
            if time.time() - start_time > timeout:
                flush_print(f"[Aggregator] Timeout waiting for weights after {timeout}s")
                return
            time.sleep(0.1)

    def _aggregate_weights(self):
        flush_print("[Aggregator] Aggregating weights...")
        with self.lock:
            edge_weights = []
            for edge_id in sorted(self.edges.keys()):
                edge = self.edges[edge_id]
                if hasattr(edge, 'local_weights') and edge.local_weights is not None:
                    edge_weights.append(edge.local_weights)
            if not edge_weights:
                flush_print("[Aggregator] No weights to aggregate")
                return
            aggregated = {}
            for key in edge_weights[0].keys():
                stacked = torch.stack([w[key] for w in edge_weights])
                aggregated[key] = stacked.mean(dim=0)
            self.global_weights = aggregated
        flush_print(f"[Aggregator] Aggregated weights from {len(edge_weights)} edges")

    def _broadcast_weights(self):
        if self.global_weights is None:
            flush_print("[Aggregator] No global weights to broadcast")
            return
        flush_print("[Aggregator] Broadcasting updated weights...")
        with self.lock:
            for edge in self.edges.values():
                try:
                    send_msg(edge.sock, {'type': 'WEIGHTS_UPDATE', 'round': self.round_count})
                    send_weights(edge.sock, self.global_weights)
                except Exception as e:
                    flush_print(f"[Aggregator] Error sending to edge {edge.edge_id}: {e}")

    def handle_edge_message(self, edge: EdgeConnection, sock: socket.socket, data: dict):
        msg_type = data['type']
        if msg_type == 'WEIGHTS_UPLOAD':
            flush_print(f"[Aggregator] Received WEIGHTS_UPLOAD from edge {edge.edge_id}")
            weights = recv_weights(sock)
            if weights:
                edge.local_weights = weights
                edge.weights_received = True
                edge.last_heartbeat = time.time()
                flush_print(f"[Aggregator] Received weights from edge {edge.edge_id}")
        elif msg_type == 'STATUS':
            self.metrics.append({
                'edge_id': edge.edge_id, 'round': self.round_count,
                'timestamp': time.time(), **data
            })
        elif msg_type == 'HEARTBEAT':
            edge.last_heartbeat = time.time()


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Aggregator')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help=f'Port to listen on (default: {DEFAULT_PORT})')
    args = parser.parse_args()
    aggregator = Aggregator(port=args.port)
    aggregator.start()


if __name__ == '__main__':
    main()