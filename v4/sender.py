import argparse
import socket
import os
import json
import time
import csv
import threading
import random
from typing import List, Tuple
from PIL import Image
import numpy as np


DEFAULT_TARGET_PORT = 6000
DEFAULT_FPS = 25
CROP_SIZE = 64
DEFAULT_NUM_ROUNDS = 100


def get_intersection_folders(base_dir: str, indices: List[int]) -> List[str]:
    all_folders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    return [all_folders[idx] for idx in indices if idx < len(all_folders)]


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


class Sender:
    def __init__(self, edge_id: int, intersection_indices: List[int],
                 target_host: str, target_port: int, fps: int = DEFAULT_FPS,
                 max_frames_per_video: int = 50,
                 num_rounds: int = DEFAULT_NUM_ROUNDS,
                 image_dir: str = '../DETRAC-Images/DETRAC-Images',
                 annotation_dir: str = '../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML',
                 max_crops_per_round: int = 0):
        self.edge_id = edge_id
        self.intersection_indices = intersection_indices
        self.target_host = target_host
        self.target_port = target_port
        self.fps = fps
        self.max_frames_per_video = max_frames_per_video
        self.num_rounds = num_rounds
        self.max_crops_per_round = max_crops_per_round
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.control_port = target_port + 1

        raw_folders = get_intersection_folders(image_dir, intersection_indices)
        self.intersection_folders = [
            f for f in raw_folders
            if os.path.exists(os.path.join(annotation_dir, f"{f}.xml"))
        ]
        skipped = len(raw_folders) - len(self.intersection_folders)

        self.round_data = [[] for _ in range(num_rounds)]
        self.seq_num = 0
        self.start_time = None
        self.send_log = []

        print(f"[Sender {edge_id}] Initialized")
        print(f"[Sender {edge_id}] Intersection indices: {intersection_indices}")
        print(f"[Sender {edge_id}] Folders with annotations: {self.intersection_folders}")
        if skipped:
            print(f"[Sender {edge_id}] Skipped {skipped} folders without annotations")
        print(f"[Sender {edge_id}] Target: {target_host}:{target_port} UDP")
        print(f"[Sender {edge_id}] Control: port {self.control_port} TCP")
        print(f"[Sender {edge_id}] FPS target: {fps}")
        print(f"[Sender {edge_id}] Rounds: {num_rounds}")

        self._preload_and_partition()

    def load_frame(self, folder: str, frame_num: int) -> np.ndarray:
        img_path = os.path.join(self.image_dir, folder, f"img{frame_num:05d}.jpg")
        if not os.path.exists(img_path):
            return None
        return np.array(Image.open(img_path).convert('RGB'))

    def extract_crops(self, frame: np.ndarray, bboxes: List[Tuple],
                      target_size: int = CROP_SIZE) -> List[np.ndarray]:
        crops = []
        for (x, y, w, h) in bboxes:
            margin = int(max(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crop = np.array(Image.fromarray(crop).resize((target_size, target_size)))
                crops.append(crop)
        return crops

    def load_annotations(self, folder: str) -> dict:
        import xml.etree.ElementTree as ET
        ann_path = os.path.join(self.annotation_dir, f"{folder}.xml")
        if not os.path.exists(ann_path):
            return {}
        tree = ET.parse(ann_path)
        root = tree.getroot()
        frames = {}
        for frame in root.findall('.//frame'):
            frame_num = int(frame.get('num'))
            bboxes = []
            for target in frame.findall('.//target'):
                bbox = target.find('box')
                if bbox is not None:
                    x = int(float(bbox.get('left')))
                    y = int(float(bbox.get('top')))
                    w = int(float(bbox.get('width')))
                    h = int(float(bbox.get('height')))
                    vehicle_type = target.find('attribute').get('vehicle_type')
                    label = {'car': 0, 'van': 1, 'bus': 2}.get(vehicle_type, 3)
                    bboxes.append((x, y, w, h, label))
            if bboxes:
                frames[frame_num] = bboxes
        return frames

    def _preload_and_partition(self):
        all_crops = []
        total_folders = len(self.intersection_folders)
        for fi, folder in enumerate(self.intersection_folders):
            print(f"[Sender {self.edge_id}] Pre-loading {folder} ({fi+1}/{total_folders})...")
            annotations = self.load_annotations(folder)
            if not annotations:
                continue
            frame_keys = sorted(annotations.keys())
            for frame_num in frame_keys:
                if frame_num > self.max_frames_per_video:
                    break
                frame = self.load_frame(folder, frame_num)
                if frame is None:
                    continue
                bboxes = [(x, y, w, h) for (x, y, w, h, _) in annotations[frame_num]]
                labels = [lbl for (_, _, _, _, lbl) in annotations[frame_num]]
                crops = self.extract_crops(frame, bboxes)
                for ci, crop in enumerate(crops):
                    all_crops.append((crop.tobytes(), labels[ci], folder, frame_num, ci))

        for r in range(self.num_rounds):
            random.shuffle(all_crops)
            subset = all_crops
            if self.max_crops_per_round > 0 and self.max_crops_per_round < len(subset):
                subset = all_crops[:self.max_crops_per_round]
            self.round_data[r] = list(subset)

        total = len(all_crops)
        used = self.max_crops_per_round if self.max_crops_per_round > 0 else total
        print(f"[Sender {self.edge_id}] Pre-loaded {total} crops from {total_folders} folders, "
              f"{used} per round (reshuffled)")

    def build_udp_packet(self, crop_bytes: bytes, seq_num: int,
                         folder: str, frame_num: int, crop_index: int,
                         label: int) -> bytes:
        meta = {'sn': seq_num, 'ed': self.edge_id, 'f': folder,
                'fn': frame_num, 'ci': crop_index, 'l': label, 'ts': time.time()}
        meta_bytes = json.dumps(meta).encode()
        return len(meta_bytes).to_bytes(4, 'big') + meta_bytes + crop_bytes

    def _stream_round(self, round_num: int, udp_port: int):
        host = '127.0.0.1'
        crops = self.round_data[round_num]
        start_time = time.time()

        print(f"[Sender {self.edge_id}] Streaming round {round_num}: {len(crops)} crops to {host}:{udp_port}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for ci, (crop_bytes, label, folder, frame_num, crop_idx) in enumerate(crops):
                seq = self.seq_num
                self.seq_num += 1
                packet = self.build_udp_packet(crop_bytes, seq, folder, frame_num, crop_idx, label)
                sock.sendto(packet, (host, udp_port))

                self.send_log.append({
                    'round': round_num,
                    'seq_num': seq,
                    'folder': folder,
                    'frame_num': frame_num,
                    'crop_index': crop_idx,
                    'label': label,
                    'send_timestamp': time.time()
                })

                elapsed = time.time() - start_time
                expected = elapsed * self.fps
                if (ci + 1) > expected:
                    time.sleep(((ci + 1) - expected) / self.fps)

            end_packet = self.build_udp_packet(b'', self.seq_num, '', 0, 0, 0)
            meta = json.loads(end_packet[4:4 + int.from_bytes(end_packet[:4], 'big')])
            meta['t'] = 'END'
            meta['tc'] = len(crops)
            meta['round'] = round_num
            meta_bytes = json.dumps(meta).encode()
            end_packet = len(meta_bytes).to_bytes(4, 'big') + meta_bytes
            sock.sendto(end_packet, (host, udp_port))
        except Exception as e:
            print(f"[Sender {self.edge_id}] Error streaming round {round_num}: {e}")
        finally:
            sock.close()

        elapsed = time.time() - start_time
        print(f"[Sender {self.edge_id}] Round {round_num} done: {len(crops)} crops in {elapsed:.2f}s")

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.control_port))
        server.listen(1)
        print(f"[Sender {self.edge_id}] Listening for control on port {self.control_port}")

        while True:
            conn, addr = server.accept()
            print(f"[Sender {self.edge_id}] Edge connected from {addr}")

            while True:
                msg = recv_msg(conn)
                if msg is None:
                    print(f"[Sender {self.edge_id}] Edge disconnected")
                    self._write_send_log()
                    break

                if msg['type'] == 'REQUEST_DATA':
                    round_num = msg['round']
                    udp_port = msg.get('udp_port', self.target_port)
                    send_msg(conn, {'type': 'ACK', 'round': round_num,
                                    'total_crops': len(self.round_data[round_num])})
                    self._stream_round(round_num, udp_port)

                elif msg['type'] == 'SHUTDOWN':
                    print(f"[Sender {self.edge_id}] Shutting down")
                    self._write_send_log()
                    conn.close()
                    server.close()
                    return

            conn.close()

        server.close()


    def _write_send_log(self):
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f'sender_{self.edge_id}_send_log.csv')
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['round', 'seq_num', 'folder', 'frame_num',
                                              'crop_index', 'label', 'send_timestamp'])
            w.writeheader()
            w.writerows(self.send_log)
        print(f"[Sender {self.edge_id}] Send log saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description='UA-DETRAC Crop Sender (UDP, round-by-round)')
    parser.add_argument('--edge-id', type=int, required=True)
    parser.add_argument('--intersection-indices', type=int, nargs='+', required=True)
    parser.add_argument('--target-host', type=str, required=True)
    parser.add_argument('--target-port', type=int, default=DEFAULT_TARGET_PORT)
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS)
    parser.add_argument('--max-frames-per-video', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument('--image-dir', type=str, default='../DETRAC-Images/DETRAC-Images')
    parser.add_argument('--annotation-dir', type=str,
                        default='../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML')
    parser.add_argument('--max-crops-per-round', type=int, default=0,
                        help='Limit crops per round (0 = unlimited, default: 0)')

    args = parser.parse_args()

    sender = Sender(
        edge_id=args.edge_id,
        intersection_indices=args.intersection_indices,
        target_host=args.target_host,
        target_port=args.target_port,
        fps=args.fps,
        max_frames_per_video=args.max_frames_per_video,
        num_rounds=args.rounds,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        max_crops_per_round=args.max_crops_per_round,
    )
    sender.serve()


if __name__ == '__main__':
    main()