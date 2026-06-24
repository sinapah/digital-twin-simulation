import argparse
import socket
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import json
import time
import os
import csv
import psutil
import sys
import io
from typing import Dict, List, Tuple, Optional


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


DEFAULT_AGGREGATOR_PORT = 5000
DEFAULT_SENDER_PORT = 6000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOCAL_EPOCHS = 3
IMG_SIZE = 64
DEFAULT_IMAGE_DIR = '../DETRAC-Images/DETRAC-Images'
DEFAULT_ANNOTATION_DIR = '../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class UDPCropReceiver:
    def __init__(self, port: int):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.sock.settimeout(1.0)
        self.buffer = []
        self.arrivals = []
        self.complete = False
        self.last_ts = None
        self.lock = threading.Lock()

    def receive_all(self):
        flush_print(f"[UDPReceiver] Listening on port {self.port}")
        while not self.complete:
            try:
                data, addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue

            meta_len = int.from_bytes(data[:4], 'big')
            meta = json.loads(data[4:4 + meta_len].decode())

            if meta.get('t') == 'END':
                total = meta.get('tc', 0)
                flush_print(f"[UDPReceiver] Stream complete: {total} crops expected")
                self.complete = True
                break

            crop_bytes = data[4 + meta_len:]
            crop_np = torch.frombuffer(crop_bytes, dtype=torch.uint8).clone().reshape(3, IMG_SIZE, IMG_SIZE).float() / 255.0
            label = torch.tensor(meta['l'], dtype=torch.long)

            now = time.time()
            delay = (now - self.last_ts) if self.last_ts is not None else 0.0
            self.last_ts = now

            with self.lock:
                self.buffer.append((crop_np, label))
                self.arrivals.append({
                    'seq_num': meta['sn'],
                    'folder': meta.get('f', ''),
                    'frame_num': meta.get('fn', 0),
                    'crop_index': meta.get('ci', 0),
                    'arrival_timestamp': now,
                    'interarrival_delay': delay
                })

        self.sock.close()

    def write_arrival_log(self, edge_id: int):
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'edge_{edge_id}_arrivals.csv')
        with self.lock:
            rows = list(self.arrivals)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['seq_num', 'folder', 'frame_num',
                                              'crop_index', 'arrival_timestamp',
                                              'interarrival_delay'])
            w.writeheader()
            w.writerows(rows)
        flush_print(f"[Edge {edge_id}] Arrival log saved to {path} ({len(rows)} arrivals)")

    def get_buffer_size(self):
        with self.lock:
            return len(self.buffer)


class EdgeAgent:
    def __init__(self, edge_id: int, intersection_indices: List[int],
                 aggregator_host: str, aggregator_port: int, sender_port: int,
                 image_dir: str = DEFAULT_IMAGE_DIR,
                 annotation_dir: str = DEFAULT_ANNOTATION_DIR,
                 max_frames_per_video: int = 50):
        self.edge_id = edge_id
        self.intersection_indices = intersection_indices
        self.aggregator_host = aggregator_host
        self.aggregator_port = aggregator_port
        self.sender_port = sender_port
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.max_frames_per_video = max_frames_per_video

        self.model = SimpleCNN(num_classes=4).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        self.data_loader = None
        self.round_count = 0
        self.connected = False
        self.metrics_log = []
        self.udp_receiver = UDPCropReceiver(sender_port)

        flush_print(f"[Edge {edge_id}] Initialized for intersections {intersection_indices}")
        flush_print(f"[Edge {edge_id}] Device: {device}")

    def _load_data_from_udp(self):
        flush_print(f"[Edge {self.edge_id}] Waiting for UDP crop stream on port {self.sender_port}...")
        self.udp_receiver.receive_all()

        if self.udp_receiver.get_buffer_size() < 10:
            flush_print(f"[Edge {self.edge_id}] Too few crops via UDP ({self.udp_receiver.get_buffer_size()}), falling back to filesystem")
            self.udp_receiver.write_arrival_log(self.edge_id)
            self._load_data_from_filesystem()
            return

        flush_print(f"[Edge {self.edge_id}] Received {self.udp_receiver.get_buffer_size()} crops via UDP")

        with self.udp_receiver.lock:
            images = torch.stack([item[0] for item in self.udp_receiver.buffer])
            labels = torch.stack([item[1] for item in self.udp_receiver.buffer])

        dataset = TensorDataset(images, labels)
        self.data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        flush_print(f"[Edge {self.edge_id}] DataLoader created with {len(self.data_loader)} batches")

        self.udp_receiver.write_arrival_log(self.edge_id)

    def _load_data_from_filesystem(self):
        flush_print(f"[Edge {self.edge_id}] Loading data from filesystem")
        from utils.detrac_loader import DETRACLoader, DETRACDataset
        from torchvision import transforms

        loader = DETRACLoader(self.image_dir, self.annotation_dir)
        all_folders = loader.get_video_folders()

        annotated = [
            f for f in all_folders
            if os.path.exists(os.path.join(self.annotation_dir, f"{f}.xml"))
        ]

        selected_folders = []
        for idx in self.intersection_indices:
            if idx < len(annotated):
                selected_folders.append(annotated[idx])

        if not selected_folders:
            flush_print(f"[Edge {self.edge_id}] No annotated folders for indices {self.intersection_indices}, using dummy data")
            self._create_dummy_data_loader()
            return

        flush_print(f"[Edge {self.edge_id}] Loading {len(selected_folders)} annotated folders: {selected_folders}")

        transform = transforms.Compose([transforms.ToTensor()])
        dataset = DETRACDataset(loader, selected_folders, max_frames=self.max_frames_per_video, transform=transform)

        if len(dataset) == 0:
            flush_print(f"[Edge {self.edge_id}] No crops in annotated folders, using dummy data")
            self._create_dummy_data_loader()
            return

        images = torch.stack([dataset[i][0] for i in range(len(dataset))])
        labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        self.data_loader = DataLoader(TensorDataset(images, labels), batch_size=BATCH_SIZE, shuffle=True)
        flush_print(f"[Edge {self.edge_id}] Loaded {len(dataset)} crops from filesystem")

    def connect_to_aggregator(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(60.0)
        try:
            flush_print(f"[Edge {self.edge_id}] Connecting to aggregator at {self.aggregator_host}:{self.aggregator_port}")
            self.sock.connect((self.aggregator_host, self.aggregator_port))
            msg = json.dumps({'type': 'CONNECT', 'edge_id': self.edge_id})
            self.sock.sendall(msg.encode())
            response = self.sock.recv(1024).decode()
            data = json.loads(response)
            if data['type'] == 'CONNECTED':
                self.connected = True
                flush_print(f"[Edge {self.edge_id}] Aggregator confirmed connection")
        except Exception as e:
            flush_print(f"[Edge {self.edge_id}] Failed to connect: {e}")
            raise

    def train_local(self) -> Dict:
        if self.data_loader is None:
            flush_print(f"[Edge {self.edge_id}] No data available for training")
            return {'loss': 0, 'accuracy': 0, 'cpu_avg': 0, 'cpu_peak': 0, 'samples_trained': 0}

        self.model.train()
        cpu_samples = []
        proc = psutil.Process()
        proc.cpu_percent(interval=None)
        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(LOCAL_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                cpu_samples.append(proc.cpu_percent(interval=None))
                data, target = data.to(device), target.to(device)
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
        avg_loss = total_loss / len(self.data_loader) if len(self.data_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0

        flush_print(f"[Edge {self.edge_id}] Training complete. Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Samples: {samples_trained}")
        return {'loss': avg_loss, 'accuracy': accuracy, 'cpu_avg': cpu_avg, 'cpu_peak': cpu_peak, 'samples_trained': samples_trained}

    def _create_dummy_data_loader(self):
        from torch.utils.data import Dataset, DataLoader
        class DummyDataset(Dataset):
            def __init__(self, n=100):
                self.n = n
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                return torch.randn(3, IMG_SIZE, IMG_SIZE), torch.randint(0, 4, (1,)).item()
        self.data_loader = DataLoader(DummyDataset(100), batch_size=BATCH_SIZE, shuffle=True)

    def get_weights(self) -> Dict:
        return {k: v.cpu().detach() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict):
        self.model.load_state_dict(weights)

    def send_weights_to_aggregator(self):
        try:
            send_msg(self.sock, {'type': 'WEIGHTS_UPLOAD', 'round': self.round_count})
            send_weights(self.sock, self.get_weights())
            flush_print(f"[Edge {self.edge_id}] Sent weights to aggregator")
        except Exception as e:
            flush_print(f"[Edge {self.edge_id}] Failed to send weights: {e}")

    def receive_weights_from_aggregator(self):
        try:
            data = recv_msg(self.sock)
            if data and data.get('type') == 'WEIGHTS_UPDATE':
                weights = recv_weights(self.sock)
                self.set_weights(weights)
                flush_print(f"[Edge {self.edge_id}] Received updated weights")
        except ConnectionResetError:
            flush_print(f"[Edge {self.edge_id}] Aggregator disconnected")
            raise SystemExit(0)
        except Exception as e:
            flush_print(f"[Edge {self.edge_id}] Failed to receive weights: {e}")

    def send_status(self, metrics: Dict):
        msg = {'type': 'STATUS', 'round': self.round_count, 'edge_id': self.edge_id, **metrics}
        try:
            send_msg(self.sock, msg)
        except Exception as e:
            flush_print(f"[Edge {self.edge_id}] Failed to send status: {e}")

    def run_round(self):
        print(f"\n[Edge {self.edge_id}] Starting round {self.round_count}")
        metrics = self.train_local()
        self.send_weights_to_aggregator()
        self.send_status(metrics)
        flush_print(f"[Edge {self.edge_id}] CPU avg: {metrics.get('cpu_avg', 0):.1f}%, peak: {metrics.get('cpu_peak', 0):.1f}%")

        metrics['round'] = self.round_count
        metrics['edge_id'] = self.edge_id
        metrics['timestamp'] = time.time()
        self.metrics_log.append(metrics)
        self._flush_metrics_csv(metrics)

        self.receive_weights_from_aggregator()
        self.round_count += 1
        flush_print(f"[Edge {self.edge_id}] Round {self.round_count - 1} complete")

    def _flush_metrics_csv(self, metrics):
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'edge_{self.edge_id}_metrics.csv')
        fieldnames = ['round', 'edge_id', 'timestamp', 'loss', 'accuracy',
                      'cpu_avg', 'cpu_peak', 'samples_trained']
        mode = 'w' if self.round_count == 0 else 'a'
        write_header = self.round_count == 0
        with open(filepath, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    def main_loop(self):
        self._load_data_from_udp()
        self.connect_to_aggregator()

        flush_print(f"[Edge {self.edge_id}] Waiting for ROUND_START from aggregator...")

        while True:
            try:
                msg = recv_msg(self.sock)
                if msg is None:
                    flush_print(f"[Edge {self.edge_id}] Aggregator disconnected")
                    break
                if msg['type'] == 'ROUND_START':
                    flush_print(f"[Edge {self.edge_id}] Received ROUND_START")
                    self.run_round()
            except SystemExit:
                raise
            except Exception as e:
                flush_print(f"[Edge {self.edge_id}] Loop error: {e}")
                time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description='Edge Agent with UDP Crop Reception')
    parser.add_argument('--edge-id', type=int, required=True)
    parser.add_argument('--intersection-indices', type=int, nargs='+', required=True)
    parser.add_argument('--aggregator-host', type=str, required=True)
    parser.add_argument('--aggregator-port', type=int, default=DEFAULT_AGGREGATOR_PORT)
    parser.add_argument('--sender-port', type=int, default=DEFAULT_SENDER_PORT)
    parser.add_argument('--image-dir', type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--annotation-dir', type=str, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument('--max-frames-per-video', type=int, default=50)

    args = parser.parse_args()

    edge = EdgeAgent(
        edge_id=args.edge_id,
        intersection_indices=args.intersection_indices,
        aggregator_host=args.aggregator_host,
        aggregator_port=args.aggregator_port,
        sender_port=args.sender_port,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        max_frames_per_video=args.max_frames_per_video,
    )
    edge.main_loop()


if __name__ == '__main__':
    main()
