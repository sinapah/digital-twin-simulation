import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import time
import os
import csv
import psutil
import sys
import io
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.simple_cnn import SimpleCNN
from utils.tcp_comm import send_msg, recv_msg, send_weights, recv_weights


AGGREGATOR_PORT = 5000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOCAL_EPOCHS = 3


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class ImageDataset(Dataset):
    def __init__(self, samples: List):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return img, label


class EdgeAgent:
    def __init__(self, edge_id: int, aggregator_host: str = '127.0.0.1',
                 aggregator_port: int = AGGREGATOR_PORT):
        self.edge_id = edge_id
        self.aggregator_host = aggregator_host
        self.aggregator_port = aggregator_port

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN(num_classes=4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        self.round_count = 0
        self.sock = None
        self.metrics_log = []
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)

        flush_print(f"[Edge {edge_id}] Initialized on {self.device}")

    def connect_to_aggregator(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(120.0)

        flush_print(f"[Edge {self.edge_id}] Connecting to aggregator at "
                    f"{self.aggregator_host}:{self.aggregator_port}")
        self.sock.connect((self.aggregator_host, self.aggregator_port))

        send_msg(self.sock, {'type': 'CONNECT', 'edge_id': self.edge_id})
        response = recv_msg(self.sock)
        if response and response.get('type') == 'CONNECTED':
            flush_print(f"[Edge {self.edge_id}] Connected to aggregator")

    def train_local(self, samples: List) -> Dict:
        if not samples:
            flush_print(f"[Edge {self.edge_id}] No samples to train on")
            return {'loss': 0, 'accuracy': 0, 'cpu_avg': 0, 'cpu_peak': 0, 'samples_trained': 0}

        dataset = ImageDataset(samples)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        flush_print(f"[Edge {self.edge_id}] Training on {len(samples)} samples "
                    f"({len(loader)} batches)")

        self.model.train()
        cpu_samples = []
        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(LOCAL_EPOCHS):
            for batch_idx, (data, target) in enumerate(loader):
                cpu_samples.append(self.process.cpu_percent(interval=None))
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
        cpu_peak = max(cpu_samples) if cpu_samples else 0.0
        samples_trained = total
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0

        flush_print(f"[Edge {self.edge_id}] Training complete. "
                    f"Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
                    f"Samples: {samples_trained}, CPU avg: {cpu_avg:.1f}%")

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'cpu_avg': cpu_avg,
            'cpu_peak': cpu_peak,
            'samples_trained': samples_trained
        }

    def get_weights(self) -> Dict:
        return {k: v.cpu().detach() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict):
        self.model.load_state_dict(weights)

    def run_round(self, samples: List, is_outage: bool = False) -> Dict:
        status = "OUTAGE" if is_outage else "NORMAL"
        flush_print(f"\n[Edge {self.edge_id}] Round {self.round_count} [{status}]")

        metrics = self.train_local(samples)

        send_msg(self.sock, {'type': 'WEIGHTS_UPLOAD', 'round': self.round_count})
        send_weights(self.sock, self.get_weights())

        status_msg = {
            'type': 'STATUS',
            'round': self.round_count,
            'edge_id': self.edge_id,
            **metrics
        }
        send_msg(self.sock, status_msg)

        metrics['round'] = self.round_count
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
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'edge_{self.edge_id}_metrics.csv')

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
    parser = argparse.ArgumentParser(description='Edge Agent for Digital Twin Simulation')
    parser.add_argument('--edge-id', type=int, required=True)
    parser.add_argument('--aggregator-host', type=str, default='127.0.0.1')
    parser.add_argument('--aggregator-port', type=int, default=AGGREGATOR_PORT)
    args = parser.parse_args()

    agent = EdgeAgent(
        edge_id=args.edge_id,
        aggregator_host=args.aggregator_host,
        aggregator_port=args.aggregator_port
    )
    agent.connect_to_aggregator()

    flush_print(f"[Edge {args.edge_id}] Ready. Waiting for simulator to provide data...")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        agent.shutdown()


if __name__ == '__main__':
    main()