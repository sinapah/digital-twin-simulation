import socket
import json
import torch
import io
from typing import Dict, Any, Optional, List


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


def fedavg(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not weights_list:
        raise ValueError("Empty weights list")
    aggregated = {}
    for key in weights_list[0].keys():
        stacked = torch.stack([w[key] for w in weights_list])
        aggregated[key] = stacked.mean(dim=0)
    return aggregated