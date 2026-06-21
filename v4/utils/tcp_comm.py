# =========================================================
# V4 Utilities - Shared Components
# =========================================================

import socket
import json
import torch
import threading
from typing import Dict, Any

# =========================================================
# TCP Communication Utilities
# =========================================================

def create_tcp_server(host: str = '0.0.0.0', port: int = 5000) -> socket.socket:
    """Create a TCP server socket"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    return server


def create_tcp_client(host: str, port: int) -> socket.socket:
    """Create a TCP client socket"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return sock


def send_json(sock: socket.socket, data: Dict[str, Any]):
    """Send JSON message over TCP"""
    msg = json.dumps(data)
    sock.sendall(msg.encode())


def recv_json(sock: socket.socket, bufsize: int = 4096) -> Dict[str, Any]:
    """Receive JSON message over TCP"""
    data = sock.recv(bufsize).decode()
    return json.loads(data)


def send_tensor(sock: socket.socket, tensor: torch.Tensor):
    """Send a PyTorch tensor over TCP"""
    # Serialize tensor to bytes
    buffer = torch.buffer(tensor)
    
    # Send header with size
    header = json.dumps({
        'type': 'TENSOR',
        'size': len(buffer),
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype)
    })
    sock.sendall(header.encode())
    
    # Send tensor data
    sock.sendall(buffer)


def recv_tensor(sock: socket.socket) -> torch.Tensor:
    """Receive a PyTorch tensor over TCP"""
    # Receive header
    header = sock.recv(4096).decode()
    data = json.loads(header)
    
    # Receive tensor data
    buffer = sock.recv(data['size'])
    
    # Deserialize
    tensor = torch.frombuffer(buffer, dtype=torch.float32)
    tensor = tensor.view(data['shape'])
    
    return tensor


# =========================================================
# Thread-safe message queue
# =========================================================

class MessageQueue:
    """Thread-safe queue for messages"""
    
    def __init__(self, maxsize: int = 100):
        self.queue = []
        self.lock = threading.Lock()
        self.maxsize = maxsize
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
    
    def put(self, item: Any, timeout: float = None) -> bool:
        """Add item to queue"""
        with self.not_full:
            if self.maxsize > 0:
                if not self.not_full.wait_for(lambda: len(self.queue) < self.maxsize, timeout):
                    return False
            self.queue.append(item)
            self.not_empty.notify()
            return True
    
    def get(self, timeout: float = None) -> Any:
        """Remove and return item from queue"""
        with self.not_empty:
            if not self.not_empty.wait_for(lambda: len(self.queue) > 0, timeout):
                raise TimeoutError("Queue timeout")
            item = self.queue.pop(0)
            self.not_full.notify()
            return item
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return len(self.queue) == 0
    
    def size(self) -> int:
        """Get queue size"""
        with self.lock:
            return len(self.queue)


# =========================================================
# Metrics Collector
# =========================================================

class MetricsCollector:
    """Collect and report training metrics"""
    
    def __init__(self):
        self.metrics = []
        self.lock = threading.Lock()
    
    def record(self, data: Dict[str, Any]):
        """Record a metric"""
        with self.lock:
            data['timestamp'] = time.time()
            self.metrics.append(data)
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get all metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def clear(self):
        """Clear metrics"""
        with self.lock:
            self.metrics.clear()


# =========================================================
# Federated Learning Helpers
# =========================================================

def fedavg(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Perform FedAvg aggregation"""
    if not weights_list:
        raise ValueError("Empty weights list")
    
    aggregated = {}
    for key in weights_list[0].keys():
        stacked = torch.stack([w[key] for w in weights_list])
        aggregated[key] = stacked.mean(dim=0)
    
    return aggregated


def get_model_size(model) -> int:
    """Get model size in bytes"""
    size = 0
    for param in model.parameters():
        size += param.numel() * param.element_size()
    return size


def serialize_weights(weights: Dict[str, torch.Tensor]) -> bytes:
    """Serialize model weights to bytes"""
    buffer = torch.buffer(weights)
    return buffer


def deserialize_weights(buffer: bytes, model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deserialize model weights from bytes"""
    # TODO: Implement proper deserialization
    pass
