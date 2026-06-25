import socket
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time
import os
import csv
import psutil
import sys
import io
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.simple_cnn import SimpleCNN
from utils.tcp_comm import send_msg, recv_msg, send_weights, recv_weights


AGGREGATOR_PORT = 5000
DEFAULT_UDP_PORT = 7000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOCAL_EPOCHS = 3
IMG_SIZE = 64


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ── Sender control protocol (8-byte prefix, same as aggregator TCP) ──────────

def send_ctrl(sock, msg_dict):
    data = json.dumps(msg_dict).encode()
    sock.sendall(len(data).to_bytes(8, 'big') + data)


def recv_ctrl(sock) -> Optional[dict]:
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


# ── UDP crop receiver ─────────────────────────────────────────────────────────

class UDPCropReceiver:
    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.sock.settimeout(5.0)
        self.buffer = []      # list of (image_tensor, label_tensor)
        self.arrivals = []    # per-crop arrival log
        self.complete = False
        self.last_ts = None
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.buffer = []
            self.arrivals = []
            self.complete = False
            self.last_ts = None

    def receive_all(self):
        while not self.complete:
            try:
                data, addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue

            meta_len = int.from_bytes(data[:4], 'big')
            meta = json.loads(data[4:4 + meta_len].decode())

            if meta.get('t') == 'END':
                self.complete = True
                break

            crop_bytes = data[4 + meta_len:]
            crop_np = torch.frombuffer(bytearray(crop_bytes), dtype=torch.uint8) \
                           .reshape(3, IMG_SIZE, IMG_SIZE).float() / 255.0
            label = torch.tensor(meta['l'], dtype=torch.long)

            now = time.time()
            delay = (now - self.last_ts) if self.last_ts is not None else 0.0
            self.last_ts = now

            with self.lock:
                self.buffer.append((crop_np, label))
                self.arrivals.append({
                    'seq_num': meta['sn'],
                    'folder': meta.get('f', ''),
                    'crop_index': meta.get('ci', 0),
                    'label': meta.get('l', -1),
                    'arrival_timestamp': now,
                    'interarrival_delay': delay,
                })

    def write_arrival_log(self, edge_id: int, output_dir: str, round_num: int):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'edge_{edge_id}_arrivals.csv')
        with self.lock:
            rows = list(self.arrivals)
        if not rows:
            return
        mode = 'w' if round_num == 0 else 'a'
        write_header = round_num == 0
        with open(path, mode, newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'seq_num', 'folder', 'crop_index', 'label',
                'arrival_timestamp', 'interarrival_delay'
            ])
            if write_header:
                w.writeheader()
            w.writerows(rows)
        flush_print(f"[Edge {edge_id}] Arrival log: {len(rows)} crops logged to {path}")

    def get_buffer_size(self):
        with self.lock:
            return len(self.buffer)


# ── Edge Agent ────────────────────────────────────────────────────────────────

class EdgeAgent:
    def __init__(self, edge_id: int, aggregator_host: str = '127.0.0.1',
                 aggregator_port: int = AGGREGATOR_PORT,
                 udp_port: int = DEFAULT_UDP_PORT,
                 output_dir: str = 'outputs'):
        self.edge_id = edge_id
        self.aggregator_host = aggregator_host
        self.aggregator_port = aggregator_port
        self.udp_port = udp_port
        self.sender_control_port = udp_port + 1
        self.output_dir = output_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN(num_classes=4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        self.round_count = 0
        self.sock = None
        self.sender_conn = None
        self.metrics_log = []
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)
        self.udp_receiver = UDPCropReceiver(udp_port)

        flush_print(f"[Edge {edge_id}] Initialized on {self.device}")
        flush_print(f"[Edge {edge_id}] UDP port: {udp_port}, "
                    f"sender control: {self.sender_control_port}")

    def connect_to_sender(self):
        self.sender_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sender_conn.settimeout(60.0)
        for attempt in range(30):
            try:
                self.sender_conn.connect(('127.0.0.1', self.sender_control_port))
                flush_print(f"[Edge {self.edge_id}] Connected to sender control "
                            f"on port {self.sender_control_port}")
                return
            except ConnectionRefusedError:
                if attempt == 0:
                    flush_print(f"[Edge {self.edge_id}] Waiting for sender on "
                                f"port {self.sender_control_port}...")
                time.sleep(2)
                self.sender_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sender_conn.settimeout(60.0)
        raise ConnectionRefusedError(
            f"Sender not available on port {self.sender_control_port} after 60s")

    def connect_to_aggregator(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(300.0)
        flush_print(f"[Edge {self.edge_id}] Connecting to aggregator at "
                    f"{self.aggregator_host}:{self.aggregator_port}")
        self.sock.connect((self.aggregator_host, self.aggregator_port))
        send_msg(self.sock, {'type': 'CONNECT', 'edge_id': self.edge_id})
        response = recv_msg(self.sock)
        if response and response.get('type') == 'CONNECTED':
            flush_print(f"[Edge {self.edge_id}] Connected to aggregator")

    def _request_udp_data(self, round_num: int, is_outage: bool):
        self.udp_receiver.reset()
        send_ctrl(self.sender_conn, {
            'type': 'REQUEST_DATA',
            'round': round_num,
            'udp_port': self.udp_port,
            'is_outage': is_outage,
        })
        ack = recv_ctrl(self.sender_conn)
        if ack:
            flush_print(f"[Edge {self.edge_id}] Sender ack: "
                        f"{ack.get('total_crops', 0)} crops for round {round_num}")
        self.udp_receiver.receive_all()
        flush_print(f"[Edge {self.edge_id}] Received "
                    f"{self.udp_receiver.get_buffer_size()} crops via UDP")

    def train_local(self) -> Dict:
        count = self.udp_receiver.get_buffer_size()
        if count == 0:
            flush_print(f"[Edge {self.edge_id}] No samples to train on")
            return {'loss': 0, 'accuracy': 0, 'cpu_avg': 0,
                    'cpu_peak': 0, 'samples_trained': 0}

        with self.udp_receiver.lock:
            images = torch.stack([item[0] for item in self.udp_receiver.buffer])
            labels = torch.stack([item[1] for item in self.udp_receiver.buffer])

        # Compute per-class weights (inverse frequency, skip absent classes)
        class_counts = torch.bincount(labels, minlength=4).float()
        present = class_counts > 0
        class_weights = torch.ones(4)
        class_weights[present] = 1.0 / class_counts[present]
        class_weights[present] = (class_weights[present] /
                                   class_weights[present].sum() *
                                   present.sum().float())
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        flush_print(f"[Edge {self.edge_id}] Class dist: " +
                    " ".join(f"{c}:{int(n)}" for c, n in
                             zip(['car', 'van', 'bus', 'other'], class_counts.tolist())))

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        flush_print(f"[Edge {self.edge_id}] Training on {count} samples "
                    f"({len(loader)} batches)")

        self.model.train()
        cpu_samples = []
        total_loss = 0
        total = 0
        all_preds_list = []
        all_targets_list = []

        for epoch in range(LOCAL_EPOCHS):
            for data, target in loader:
                cpu_samples.append(self.process.cpu_percent(interval=None))
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds_list.append(pred.cpu())
                all_targets_list.append(target.cpu())
                total += target.size(0)

        cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        cpu_peak = max(cpu_samples) if cpu_samples else 0.0
        avg_loss = total_loss / len(loader) if loader else 0

        # Balanced accuracy: mean per-class recall
        all_preds = torch.cat(all_preds_list)
        all_targets = torch.cat(all_targets_list)
        per_class_recall = []
        for c in range(4):
            mask = all_targets == c
            if mask.sum() > 0:
                per_class_recall.append(
                    (all_preds[mask] == c).float().mean().item())
        balanced_acc = (sum(per_class_recall) / len(per_class_recall)
                        if per_class_recall else 0.0)

        flush_print(f"[Edge {self.edge_id}] Training complete. "
                    f"Loss: {avg_loss:.4f}, Balanced Acc: {balanced_acc:.4f}, "
                    f"Samples: {total}, CPU avg: {cpu_avg:.1f}%")

        return {'loss': avg_loss, 'accuracy': balanced_acc,
                'cpu_avg': cpu_avg, 'cpu_peak': cpu_peak, 'samples_trained': total}

    def get_weights(self) -> Dict:
        return {k: v.cpu().detach() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict):
        self.model.load_state_dict(weights)

    def run_round(self, round_num: int, is_outage: bool = False) -> Dict:
        status = "OUTAGE" if is_outage else "NORMAL"
        flush_print(f"\n[Edge {self.edge_id}] Round {round_num} [{status}]")

        self._request_udp_data(round_num, is_outage)
        self.udp_receiver.write_arrival_log(
            self.edge_id, self.output_dir, round_num)

        metrics = self.train_local()

        send_msg(self.sock, {'type': 'WEIGHTS_UPLOAD', 'round': round_num})
        send_weights(self.sock, self.get_weights())
        send_msg(self.sock, {'type': 'STATUS', 'round': round_num,
                             'edge_id': self.edge_id, **metrics})

        metrics['round'] = round_num
        metrics['edge_id'] = self.edge_id
        metrics['timestamp'] = time.time()
        metrics['is_outage'] = int(is_outage)
        self.metrics_log.append(metrics)
        self._flush_metrics_csv(metrics)

        try:
            data = recv_msg(self.sock)
            if data and data.get('type') == 'WEIGHTS_UPDATE':
                weights = recv_weights(self.sock)
                self.set_weights(weights)
                flush_print(f"[Edge {self.edge_id}] Received updated weights")
        except (ConnectionResetError, socket.timeout):
            flush_print(f"[Edge {self.edge_id}] Aggregator connection closed")
            raise SystemExit(0)
        except Exception as e:
            flush_print(f"[Edge {self.edge_id}] Error receiving weights: {e}")

        self.round_count += 1
        return metrics

    def _flush_metrics_csv(self, metrics):
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f'edge_{self.edge_id}_metrics.csv')
        fieldnames = ['round', 'edge_id', 'timestamp', 'loss', 'accuracy',
                      'cpu_avg', 'cpu_peak', 'samples_trained', 'is_outage']
        mode = 'w' if self.round_count == 0 else 'a'
        write_header = self.round_count == 0
        with open(filepath, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    def shutdown(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        flush_print(f"[Edge {self.edge_id}] Shutdown")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='V5 Edge Agent with UDP Crop Reception')
    parser.add_argument('--edge-id', type=int, required=True)
    parser.add_argument('--aggregator-host', type=str, default='127.0.0.1')
    parser.add_argument('--aggregator-port', type=int, default=AGGREGATOR_PORT)
    parser.add_argument('--udp-port', type=int, default=DEFAULT_UDP_PORT)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--rounds', type=int, default=100)
    args = parser.parse_args()

    agent = EdgeAgent(
        edge_id=args.edge_id,
        aggregator_host=args.aggregator_host,
        aggregator_port=args.aggregator_port,
        udp_port=args.udp_port,
        output_dir=args.output_dir,
    )
    agent.connect_to_sender()
    agent.connect_to_aggregator()

    flush_print(f"[Edge {args.edge_id}] Waiting for ROUND_START...")

    while True:
        try:
            data = recv_msg(agent.sock)
        except (ConnectionResetError, socket.timeout, OSError):
            flush_print(f"[Edge {args.edge_id}] Aggregator disconnected")
            break
        if data is None:
            break

        if data.get('type') == 'ROUND_START':
            round_num = data['round']
            is_outage = data.get('is_outage', False)
            metrics = agent.run_round(round_num, is_outage)
            send_msg(agent.sock, {'type': 'ROUND_COMPLETE'})

        elif data.get('type') == 'SHUTDOWN':
            break

    agent.shutdown()


if __name__ == '__main__':
    main()
