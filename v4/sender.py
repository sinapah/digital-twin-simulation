import argparse
import socket
import os
import json
import time
import csv
from typing import List, Tuple
from PIL import Image
import numpy as np


DEFAULT_TARGET_PORT = 6000
DEFAULT_FPS = 25
CROP_SIZE = 64


def get_intersection_folders(base_dir: str, indices: List[int]) -> List[str]:
    all_folders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    return [all_folders[idx] for idx in indices if idx < len(all_folders)]


class Sender:
    def __init__(self, edge_id: int, intersection_indices: List[int],
                 target_host: str, target_port: int, fps: int = DEFAULT_FPS,
                 max_frames_per_video: int = 50,
                 image_dir: str = '../DETRAC-Images/DETRAC-Images',
                 annotation_dir: str = '../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML'):
        self.edge_id = edge_id
        self.intersection_indices = intersection_indices
        self.target_host = target_host
        self.target_port = target_port
        self.fps = fps
        self.max_frames_per_video = max_frames_per_video
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        raw_folders = get_intersection_folders(image_dir, intersection_indices)
        self.intersection_folders = [
            f for f in raw_folders
            if os.path.exists(os.path.join(annotation_dir, f"{f}.xml"))
        ]
        skipped = len(raw_folders) - len(self.intersection_folders)
        self.frame_interval = 1.0 / fps
        self.seq_num = 0
        self.start_time = None
        self.send_log = []

        print(f"[Sender {edge_id}] Initialized")
        print(f"[Sender {edge_id}] Intersection indices: {intersection_indices}")
        print(f"[Sender {edge_id}] Folders with annotations: {self.intersection_folders}")
        if skipped:
            print(f"[Sender {edge_id}] Skipped {skipped} folders without annotations")
        print(f"[Sender {edge_id}] Target: {target_host}:{target_port} UDP")
        print(f"[Sender {edge_id}] FPS target: {fps}")

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

    def build_udp_packet(self, crop_bytes: bytes, seq_num: int,
                         folder: str, frame_num: int, crop_index: int,
                         label: int) -> bytes:
        meta = {
            'sn': seq_num,
            'ed': self.edge_id,
            'f': folder,
            'fn': frame_num,
            'ci': crop_index,
            'l': label,
            'ts': time.time()
        }
        meta_bytes = json.dumps(meta).encode()
        return len(meta_bytes).to_bytes(4, 'big') + meta_bytes + crop_bytes

    def run(self):
        self.start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        expected_crops = 0

        print(f"\n[Sender {self.edge_id}] Starting UDP stream...")

        for folder in self.intersection_folders:
            print(f"[Sender {self.edge_id}] Processing {folder}...")
            annotations = self.load_annotations(folder)
            if not annotations:
                print(f"[Sender {self.edge_id}] No annotations for {folder}, skipping")
                continue

            for frame_num in sorted(annotations.keys()):
                if frame_num > self.max_frames_per_video:
                    break

                frame = self.load_frame(folder, frame_num)
                if frame is None:
                    continue

                bboxes = [(x, y, w, h) for (x, y, w, h, label) in annotations[frame_num]]
                labels = [label for (x, y, w, h, label) in annotations[frame_num]]
                crops = self.extract_crops(frame, bboxes)

                for ci, crop in enumerate(crops):
                    crop_bytes = crop.tobytes()
                    packet = self.build_udp_packet(
                        crop_bytes, self.seq_num,
                        folder, frame_num, ci, labels[ci]
                    )
                    sock.sendto(packet, (self.target_host, self.target_port))

                    self.send_log.append({
                        'seq_num': self.seq_num,
                        'folder': folder,
                        'frame_num': frame_num,
                        'crop_index': ci,
                        'label': labels[ci],
                        'send_timestamp': time.time()
                    })
                    self.seq_num += 1
                    expected_crops += 1

                    elapsed = time.time() - self.start_time
                    expected_frames = elapsed * self.fps
                    if expected_crops > expected_frames:
                        sleep_time = (expected_crops - expected_frames) / self.fps
                        time.sleep(sleep_time)

        end_packet = self.build_udp_packet(b'', self.seq_num, '', 0, 0, 0)
        meta = json.loads(end_packet[4:4 + int.from_bytes(end_packet[:4], 'big')])
        meta['t'] = 'END'
        meta['tc'] = expected_crops
        meta_bytes = json.dumps(meta).encode()
        end_packet = len(meta_bytes).to_bytes(4, 'big') + meta_bytes
        sock.sendto(end_packet, (self.target_host, self.target_port))

        elapsed = time.time() - self.start_time
        print(f"\n[Sender {self.edge_id}] Complete!")
        print(f"[Sender {self.edge_id}] Sent {expected_crops} crops in {elapsed:.2f}s")
        print(f"[Sender {self.edge_id}] Effective rate: {expected_crops / elapsed:.2f} crops/s")

        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f'sender_{self.edge_id}_send_log.csv')
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['seq_num', 'folder', 'frame_num',
                                              'crop_index', 'label', 'send_timestamp'])
            w.writeheader()
            w.writerows(self.send_log)
        print(f"[Sender {self.edge_id}] Send log saved to {log_path}")

        sock.close()


def main():
    parser = argparse.ArgumentParser(description='UA-DETRAC Crop Sender (UDP)')
    parser.add_argument('--edge-id', type=int, required=True)
    parser.add_argument('--intersection-indices', type=int, nargs='+', required=True)
    parser.add_argument('--target-host', type=str, required=True)
    parser.add_argument('--target-port', type=int, default=DEFAULT_TARGET_PORT)
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS)
    parser.add_argument('--max-frames-per-video', type=int, default=50)
    parser.add_argument('--image-dir', type=str, default='../DETRAC-Images/DETRAC-Images')
    parser.add_argument('--annotation-dir', type=str,
                        default='../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML')

    args = parser.parse_args()

    sender = Sender(
        edge_id=args.edge_id,
        intersection_indices=args.intersection_indices,
        target_host=args.target_host,
        target_port=args.target_port,
        fps=args.fps,
        max_frames_per_video=args.max_frames_per_video,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
    )
    sender.run()


if __name__ == '__main__':
    main()