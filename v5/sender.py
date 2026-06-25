import argparse
import socket
import os
import json
import time
import csv
import random
import sys
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.detrac_loader import DETRACLoader
from camera_simulator import CameraSimulator, NUM_EDGES, VIDEOS_PER_EDGE, HISTORICAL_FOLDERS


DEFAULT_UDP_PORT = 7000
DEFAULT_FPS = 25
DEFAULT_NUM_ROUNDS = 100
CROP_SIZE = 64


def send_msg(sock, msg_dict):
    data = json.dumps(msg_dict).encode()
    sock.sendall(len(data).to_bytes(8, 'big') + data)


def recv_msg(sock):
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


class V5Sender:
    def __init__(self, edge_id: int, delay_model: str = 'kde',
                 udp_port: int = DEFAULT_UDP_PORT,
                 fps: int = DEFAULT_FPS,
                 max_frames_per_video: int = 50,
                 num_rounds: int = DEFAULT_NUM_ROUNDS,
                 max_crops_per_round: int = 0):
        self.edge_id = edge_id
        self.delay_model = delay_model
        self.udp_port = udp_port
        self.control_port = udp_port + 1
        self.fps = fps
        self.max_frames_per_video = max_frames_per_video
        self.num_rounds = num_rounds
        self.max_crops_per_round = max_crops_per_round
        self.seq_num = 0
        self.send_log = []

        print(f"[V5Sender {edge_id}] Initializing with {delay_model.upper()} delay model")

        self.cam = CameraSimulator(
            delay_model=delay_model,
            max_frames_per_video=max_frames_per_video,
        )

        self.normal_crops = self._load_crops(is_outage=False)
        self.outage_crops = self._load_crops(is_outage=True)

        print(f"[V5Sender {edge_id}] Normal data: {len(self.normal_crops)} crops")
        print(f"[V5Sender {edge_id}] Outage data (historical): {len(self.outage_crops)} crops")
        print(f"[V5Sender {edge_id}] UDP port: {udp_port}, control port: {self.control_port}")

    def _load_crops(self, is_outage: bool) -> List:
        from utils.detrac_loader import DETRACDataset
        from torchvision import transforms

        if is_outage:
            folders = [self.cam.historical_folders[self.edge_id]]
        else:
            folders = self.cam.edge_folders[self.edge_id]

        transform = transforms.Compose([transforms.ToTensor()])
        dataset = DETRACDataset(
            loader=self.cam.detrac_loader,
            video_folders=folders,
            max_frames=self.max_frames_per_video,
            transform=transform,
        )

        crops = []
        for i in range(len(dataset)):
            img_tensor, label = dataset[i]
            # Convert tensor back to bytes for UDP transmission
            crop_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            # Store as (C, H, W) bytes for consistency with V4
            crop_chw = img_tensor.mul(255).byte().numpy()
            crops.append((crop_chw.tobytes(), int(label),
                          folders[0] if is_outage else 'normal', 0, i))

        source = 'historical' if is_outage else 'normal'
        print(f"[V5Sender {self.edge_id}] Loaded {len(crops)} {source} crops from {folders}")
        return crops

    def _get_round_crops(self, is_outage: bool) -> List:
        crops = list(self.outage_crops if is_outage else self.normal_crops)
        random.shuffle(crops)
        if self.max_crops_per_round > 0:
            crops = crops[:self.max_crops_per_round]
        return crops

    def _build_udp_packet(self, crop_bytes: bytes, seq_num: int,
                          folder: str, frame_num: int, crop_index: int,
                          label: int) -> bytes:
        meta = {
            'sn': seq_num, 'ed': self.edge_id, 'f': folder,
            'fn': frame_num, 'ci': crop_index, 'l': label,
            'ts': time.time()
        }
        meta_bytes = json.dumps(meta).encode()
        return len(meta_bytes).to_bytes(4, 'big') + meta_bytes + crop_bytes

    def _stream_round(self, round_num: int, udp_port: int, is_outage: bool):
        host = '127.0.0.1'
        crops = self._get_round_crops(is_outage)
        start_time = time.time()
        source = 'OUTAGE/historical' if is_outage else 'NORMAL'

        print(f"[V5Sender {self.edge_id}] Round {round_num} [{source}]: "
              f"streaming {len(crops)} crops to {host}:{udp_port}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for ci, (crop_bytes, label, folder, frame_num, crop_idx) in enumerate(crops):
                seq = self.seq_num
                self.seq_num += 1
                packet = self._build_udp_packet(
                    crop_bytes, seq, folder, frame_num, crop_idx, label
                )
                sock.sendto(packet, (host, udp_port))

                self.send_log.append({
                    'round': round_num,
                    'seq_num': seq,
                    'is_outage': int(is_outage),
                    'folder': folder,
                    'crop_index': crop_idx,
                    'label': label,
                    'send_timestamp': time.time()
                })

                elapsed = time.time() - start_time
                expected = elapsed * self.fps
                if (ci + 1) > expected:
                    time.sleep(((ci + 1) - expected) / self.fps)

            end_meta = {'t': 'END', 'tc': len(crops), 'round': round_num,
                        'sn': self.seq_num, 'ed': self.edge_id,
                        'f': '', 'fn': 0, 'ci': 0, 'l': 0, 'ts': time.time()}
            end_bytes = json.dumps(end_meta).encode()
            end_packet = len(end_bytes).to_bytes(4, 'big') + end_bytes
            sock.sendto(end_packet, (host, udp_port))

            elapsed = time.time() - start_time
            print(f"[V5Sender {self.edge_id}] Round {round_num} done: "
                  f"{len(crops)} crops in {elapsed:.2f}s")
        except Exception as e:
            print(f"[V5Sender {self.edge_id}] Error streaming round {round_num}: {e}")
        finally:
            sock.close()

    def _write_send_log(self):
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'v5_sender_{self.edge_id}_send_log.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'round', 'seq_num', 'is_outage', 'folder',
                'crop_index', 'label', 'send_timestamp'
            ])
            w.writeheader()
            w.writerows(self.send_log)
        print(f"[V5Sender {self.edge_id}] Send log saved to {path}")

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.control_port))
        server.listen(1)
        print(f"[V5Sender {self.edge_id}] Listening for control on port {self.control_port}")

        while True:
            conn, addr = server.accept()
            print(f"[V5Sender {self.edge_id}] Edge connected from {addr}")

            while True:
                msg = recv_msg(conn)
                if msg is None:
                    print(f"[V5Sender {self.edge_id}] Edge disconnected")
                    self._write_send_log()
                    break

                if msg['type'] == 'REQUEST_DATA':
                    round_num = msg['round']
                    udp_port = msg.get('udp_port', self.udp_port)
                    is_outage = msg.get('is_outage', False)
                    crops = self._get_round_crops(is_outage)
                    send_msg(conn, {'type': 'ACK', 'round': round_num,
                                    'total_crops': len(crops), 'is_outage': is_outage})
                    self._stream_round(round_num, udp_port, is_outage)

                elif msg['type'] == 'SHUTDOWN':
                    print(f"[V5Sender {self.edge_id}] Shutting down")
                    self._write_send_log()
                    conn.close()
                    server.close()
                    return

            conn.close()

        server.close()


def main():
    parser = argparse.ArgumentParser(description='V5 UDP Crop Sender')
    parser.add_argument('--edge-id', type=int, required=True)
    parser.add_argument('--delay-model', type=str, default='kde',
                        choices=['kde', 'wgan'])
    parser.add_argument('--udp-port', type=int, default=DEFAULT_UDP_PORT)
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS)
    parser.add_argument('--max-frames', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument('--max-crops-per-round', type=int, default=0)
    args = parser.parse_args()

    sender = V5Sender(
        edge_id=args.edge_id,
        delay_model=args.delay_model,
        udp_port=args.udp_port,
        fps=args.fps,
        max_frames_per_video=args.max_frames,
        num_rounds=args.rounds,
        max_crops_per_round=args.max_crops_per_round,
    )
    sender.serve()


if __name__ == '__main__':
    main()
