# =========================================================
# V4 Edge Agent - Local Training with Federated Aggregation
# =========================================================
# Runs on each edge VM
# Receives images from sender, trains locally, participates in federated learning
# =========================================================

import argparse
import socket
import threading
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
from dataclasses import dataclass, field


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def send_msg(sock: socket.socket, msg_dict: dict):
    """Send a small dict as JSON with an 8-byte size prefix"""
    data = json.dumps(msg_dict).encode()
    size_prefix = len(data).to_bytes(8, 'big')
    sock.sendall(size_prefix + data)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    """Receive a size-prefixed JSON message"""
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
    """Send model weights as binary using torch.save"""
    buf = io.BytesIO()
    torch.save(weights, buf)
    data = buf.getvalue()
    size_prefix = len(data).to_bytes(8, 'big')
    sock.sendall(size_prefix + data)


def recv_weights(sock: socket.socket) -> Optional[dict]:
    """Receive model weights as binary"""
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

# =========================================================
# CONFIG
# =========================================================
DEFAULT_AGGREGATOR_PORT = 5000
DEFAULT_SENDER_PORT = 6000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOCAL_EPOCHS = 3
IMG_SIZE = 64


class SimpleCNN(nn.Module):
    """Simple CNN for vehicle classification"""
    
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
        x = self.pool(torch.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(torch.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = self.pool(torch.relu(self.conv3(x)))  # 16x16 -> 8x8
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DETRACDataset(Dataset):
    """UA-DETRAC dataset for vehicle classification"""
    
    def __init__(self, image_dir: str, annotation_dir: str, 
                 intersection_indices: List[int], max_frames: int = 50):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.intersection_indices = intersection_indices
        self.max_frames = max_frames
        self.transforms = None
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from UA-DETRAC format"""
        # TODO: Implement UA-DETRAC loading
        # For now, create placeholder
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx):
        # TODO: Implement actual item loading
        pass


class EdgeAgent:
    """Edge device that trains locally and participates in federated learning"""
    
    def __init__(self, edge_id: int, intersection_indices: List[int],
                 aggregator_host: str, aggregator_port: int, sender_port: int,
                 image_dir: str = '../DETRAC-Images/DETRAC-Images',
                 annotation_dir: str = '../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML',
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
        
        self.dataset = None
        self.data_loader = None
        self.round_count = 0
        self.connected = False
        self.metrics_log = []
        
        self.flush_print(f"[Edge {edge_id}] Initialized for intersections {intersection_indices}")
        self.flush_print(f"[Edge {edge_id}] Device: {device}")
        self.flush_print(f"[Edge {edge_id}] Image dir: {image_dir}")
        self.flush_print(f"[Edge {edge_id}] Annotation dir: {annotation_dir}")
        
        self._load_data()
    
    def _load_data(self):
        """Load real UA-DETRAC data from mounted directory"""
        from utils.detrac_loader import DETRACLoader, DETRACDataset
        from torchvision import transforms
        
        loader = DETRACLoader(self.image_dir, self.annotation_dir)
        all_folders = loader.get_video_folders()
        
        if not all_folders:
            self.flush_print(f"[Edge {self.edge_id}] WARNING: No UA-DETRAC folders found at {self.image_dir}")
            self.flush_print(f"[Edge {self.edge_id}] Falling back to dummy data")
            self._create_dummy_data_loader()
            return
        
        selected_folders = []
        for idx in self.intersection_indices:
            if idx < len(all_folders):
                selected_folders.append(all_folders[idx])
        
        self.flush_print(f"[Edge {self.edge_id}] Loading {len(selected_folders)} UA-DETRAC folders: {selected_folders}")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.dataset = DETRACDataset(
            loader=loader,
            video_folders=selected_folders,
            max_frames=self.max_frames_per_video,
            transform=transform
        )
        self.flush_print(f"[Edge {self.edge_id}] Loaded {len(self.dataset)} vehicle crops from UA-DETRAC")
        
        if len(self.dataset) == 0:
            self.flush_print(f"[Edge {self.edge_id}] No crops found in UA-DETRAC folders, falling back to dummy data")
            self._create_dummy_data_loader()
            return
        
        from torch.utils.data import DataLoader
        self.data_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.flush_print(f"[Edge {self.edge_id}] DataLoader created with {len(self.data_loader)} batches")
    
    def connect_to_aggregator(self):
        """Connect to aggregator and register"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set a longer timeout for socket operations given large weight transfers
        self.sock.settimeout(60.0)  # 60 second timeout for weight transfer
        
        try:
            self.flush_print(f"[Edge {self.edge_id}] Connecting to aggregator at {self.aggregator_host}:{self.aggregator_port}")
            self.sock.connect((self.aggregator_host, self.aggregator_port))
            self.flush_print(f"[Edge {self.edge_id}] Connected to aggregator")
            
            # Send connection message
            msg = json.dumps({
                'type': 'CONNECT',
                'edge_id': self.edge_id
            })
            self.sock.sendall(msg.encode())
            
            # Wait for confirmation
            response = self.sock.recv(1024).decode()
            data = json.loads(response)
            
            if data['type'] == 'CONNECTED':
                self.connected = True
                self.flush_print(f"[Edge {self.edge_id}] Aggregator confirmed connection")
            
        except Exception as e:
            self.flush_print(f"[Edge {self.edge_id}] Failed to connect: {e}")
            raise
    
    def start_data_receiver(self):
        """Start receiving images from sender"""
        receiver_thread = threading.Thread(target=self._receive_images)
        receiver_thread.daemon = True
        receiver_thread.start()
        self.flush_print(f"[Edge {self.edge_id}] Started data receiver on port {self.sender_port}")
    
    def _receive_images(self):
        """Receive images from sender"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.sender_port))
        server.listen(1)
        
        self.flush_print(f"[Edge {self.edge_id}] Waiting for sender on port {self.sender_port}")
        
        while True:
            conn, addr = server.accept()
            self.flush_print(f"[Edge {self.edge_id}] Sender connected from {addr}")
            
            # Receive image batch
            threading.Thread(target=self._handle_image_batch, args=(conn,)).start()
    
    def _handle_image_batch(self, conn: socket.socket):
        """Handle incoming image batch"""
        # TODO: Implement image batch handling
        # For now, just receive and acknowledge
        try:
            data = conn.recv(4096)
            msg = json.loads(data.decode())
            
            if msg['type'] == 'IMAGE_BATCH':
                # Process images
                # TODO: Add to dataset
                pass
            
            conn.sendall(json.dumps({'type': 'ACK'}).encode())
        except Exception as e:
            self.flush_print(f"[Edge {self.edge_id}] Error handling batch: {e}")
    
    def train_local(self) -> Dict:
        """Train on local data for one round"""
        if self.data_loader is None:
            self.flush_print(f"[Edge {self.edge_id}] No data available for training, returning empty metrics")
            return {
                'loss': 0,
                'accuracy': 0,
                'cpu_avg': 0,
                'cpu_peak': 0,
                'samples_trained': 0
            }
        
        self.flush_print(f"[Edge {self.edge_id}] Starting local training with {len(self.data_loader.dataset)} samples")
        self.model.train()
        
        cpu_samples = []
        process = psutil.Process()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for epoch in range(LOCAL_EPOCHS):
            self.flush_print(f"[Edge {self.edge_id}] Epoch {epoch+1}/{LOCAL_EPOCHS}")
            for batch_idx, (data, target) in enumerate(self.data_loader):
                cpu_samples.append(process.cpu_percent(interval=None))
                
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
        
        self.flush_print(f"[Edge {self.edge_id}] Local training complete. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Samples: {samples_trained}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'cpu_avg': cpu_avg,
            'cpu_peak': cpu_peak,
            'samples_trained': samples_trained
        }
    
    def _create_dummy_data_loader(self):
        """Create a dummy data loader for testing"""
        # For now, create synthetic data to avoid dependency on UA-DETRAC
        # This allows the system to run without actual data
        from torch.utils.data import Dataset, DataLoader
        import torch
        
        class DummyDataset(Dataset):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Create random image data
                img = torch.randn(3, IMG_SIZE, IMG_SIZE)
                label = torch.randint(0, 4, (1,)).item()
                return img, label
        
        self.dataset = DummyDataset(num_samples=100)
        self.data_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    def flush_print(self, *args):
        """Print with flush to ensure output appears immediately"""
        print(*args)
        sys.stdout.flush()
    
    def get_weights(self) -> Dict:
        """Get current model weights"""
        return {k: v.cpu().detach() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights: Dict):
        """Set model weights from aggregator"""
        self.model.load_state_dict(weights)
    
    def send_weights_to_aggregator(self):
        """Upload local weights to aggregator using binary serialization"""
        try:
            weights = self.get_weights()
            # Send metadata as small JSON
            send_msg(self.sock, {'type': 'WEIGHTS_UPLOAD', 'round': self.round_count})
            # Send weights as binary
            send_weights(self.sock, weights)
            self.flush_print(f"[Edge {self.edge_id}] Sent weights to aggregator")
        except Exception as e:
            self.flush_print(f"[Edge {self.edge_id}] Failed to send weights: {e}")
    
    def receive_weights_from_aggregator(self):
        """Receive updated weights from aggregator"""
        try:
            data = recv_msg(self.sock)
            if data and data.get('type') == 'WEIGHTS_UPDATE':
                weights = recv_weights(self.sock)
                self.set_weights(weights)
                self.flush_print(f"[Edge {self.edge_id}] Received updated weights")
        except ConnectionResetError:
            self.flush_print(f"[Edge {self.edge_id}] Aggregator connection closed (round complete)")
            raise SystemExit(0)
        except Exception as e:
            self.flush_print(f"[Edge {self.edge_id}] Failed to receive weights: {e}")
    
    def send_status(self, metrics: Dict):
        """Send training status to aggregator"""
        msg = {
            'type': 'STATUS',
            'round': self.round_count,
            'edge_id': self.edge_id,
            **metrics
        }
        try:
            send_msg(self.sock, msg)
        except Exception as e:
            self.flush_print(f"[Edge {self.edge_id}] Failed to send status: {e}")
    
    def run_round(self):
        """Execute one federated learning round"""
        print(f"\n[Edge {self.edge_id}] Starting round {self.round_count}")
        
        # Train locally
        metrics = self.train_local()
        
        # Send weights to aggregator
        self.send_weights_to_aggregator()
        
        # Send status
        self.send_status(metrics)
        
        # Print CPU metrics
        self.flush_print(f"[Edge {self.edge_id}] CPU avg: {metrics.get('cpu_avg', 0):.1f}%, peak: {metrics.get('cpu_peak', 0):.1f}%")
        
        # Log metrics to CSV
        metrics['round'] = self.round_count
        metrics['edge_id'] = self.edge_id
        metrics['timestamp'] = time.time()
        self.metrics_log.append(metrics)
        self._flush_metrics_csv(metrics)
        
        # Wait for updated weights
        self.receive_weights_from_aggregator()
        
        self.round_count += 1
        self.flush_print(f"[Edge {self.edge_id}] Round {self.round_count - 1} complete")
    
    def _flush_metrics_csv(self, metrics):
        """Write metrics to CSV file"""
        import csv
        import os
        
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
        """Main federated learning loop"""
        self.connect_to_aggregator()
        self.start_data_receiver()
        
        flush_print(f"[Edge {self.edge_id}] Waiting for aggregator messages...")
        
        while True:
            try:
                msg = recv_msg(self.sock)
                if msg is None:
                    flush_print(f"[Edge {self.edge_id}] Aggregator disconnected")
                    break
                
                if msg['type'] == 'ROUND_START':
                    flush_print(f"[Edge {self.edge_id}] Received ROUND_START")
                    self.run_round()
                
            except SystemExit as e:
                self.flush_print(f"[Edge {self.edge_id}] Exiting with code {e.code}")
                raise
            except Exception as e:
                print(f"[Edge {self.edge_id}] Loop error: {e}")
                time.sleep(0.1)


# =========================================================
# MAIN
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='Edge Agent for Federated Learning')
    parser.add_argument('--edge-id', type=int, required=True, help='Edge device ID')
    parser.add_argument('--intersection-indices', type=int, nargs='+', required=True,
                       help='Intersection indices assigned to this edge')
    parser.add_argument('--aggregator-host', type=str, required=True,
                       help='Aggregator VM IP address')
    parser.add_argument('--aggregator-port', type=int, default=DEFAULT_AGGREGATOR_PORT,
                       help=f'Aggregator port (default: {DEFAULT_AGGREGATOR_PORT})')
    parser.add_argument('--sender-port', type=int, default=DEFAULT_SENDER_PORT,
                       help=f'Port to receive images from sender (default: {DEFAULT_SENDER_PORT})')
    parser.add_argument('--image-dir', type=str, default='../DETRAC-Images/DETRAC-Images',
                       help='Path to UA-DETRAC images')
    parser.add_argument('--annotation-dir', type=str, default='../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML',
                       help='Path to UA-DETRAC annotations')
    parser.add_argument('--max-frames-per-video', type=int, default=50,
                       help='Maximum frames per video')
    
    args = parser.parse_args()
    
    edge = EdgeAgent(
        edge_id=args.edge_id,
        intersection_indices=args.intersection_indices,
        aggregator_host=args.aggregator_host,
        aggregator_port=args.aggregator_port,
        sender_port=args.sender_port,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        max_frames_per_video=args.max_frames_per_video
    )
    
    edge.main_loop()


if __name__ == '__main__':
    main()
