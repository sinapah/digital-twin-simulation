# =========================================================
# V4 Sender - UA-DETRAC Image Streamer
# =========================================================
# Runs on each edge VM
# Streams images from assigned UA-DETRAC intersections to edge agent
# =========================================================

import argparse
import socket
import threading
import os
import json
import time
from typing import List, Tuple
from PIL import Image
import numpy as np

# =========================================================
# CONFIG
# =========================================================
DEFAULT_TARGET_PORT = 6000
DEFAULT_FPS = 25


def get_intersection_folders(base_dir: str, indices: List[int]) -> List[str]:
    """Get UA-DETRAC intersection folders for given indices"""
    all_folders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])
    
    folders = []
    for idx in indices:
        if idx < len(all_folders):
            folders.append(all_folders[idx])
    
    return folders


def load_frames_from_folder(folder_path: str, max_frames: int) -> List[np.ndarray]:
    """Load frames from a UA-DETRAC image folder"""
    frames = []
    
    # UA-DETRAC images are typically named img_0001.jpg, img_0002.jpg, etc.
    img_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    
    for img_file in img_files[:max_frames]:
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('RGB')
        frames.append(np.array(img))
    
    return frames


class Sender:
    """Streams UA-DETRAC images to edge agent"""
    
    def __init__(self, edge_id: int, intersection_indices: List[int],
                 target_host: str, target_port: int, fps: int = DEFAULT_FPS,
                 max_frames_per_video: int = 50):
        self.edge_id = edge_id
        self.intersection_indices = intersection_indices
        self.target_host = target_host
        self.target_port = target_port
        self.fps = fps
        self.max_frames_per_video = max_frames_per_video
        
        self.image_dir = '../DETRAC-Images/DETRAC-Images'
        self.annotation_dir = '../DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML'
        
        self.intersection_folders = get_intersection_folders(
            self.image_dir, intersection_indices
        )
        
        self.frame_interval = 1.0 / fps
        self.total_frames_sent = 0
        self.start_time = None
        
        print(f"[Sender {edge_id}] Initialized")
        print(f"[Sender {edge_id}] Intersections: {intersection_indices}")
        print(f"[Sender {edge_id}] Folders: {self.intersection_folders}")
        print(f"[Sender {edge_id}] Target: {target_host}:{target_port}")
    
    def connect_to_edge(self) -> socket.socket:
        """Connect to edge agent"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.target_host, self.target_port))
        print(f"[Sender {self.edge_id}] Connected to edge agent")
        return sock
    
    def send_image_batch(self, sock: socket.socket, images: List[np.ndarray], 
                         labels: List[int], frame_ids: List[int]) -> bool:
        """Send a batch of images to edge agent"""
        try:
            # Create batch message
            batch = {
                'type': 'IMAGE_BATCH',
                'edge_id': self.edge_id,
                'frame_ids': frame_ids,
                'batch_size': len(images),
                'data': []  # TODO: Serialize images
            }
            
            # Send header
            header = json.dumps(batch)
            sock.sendall(header.encode())
            
            # TODO: Send image data
            # For now, just acknowledge
            return True
            
        except Exception as e:
            print(f"[Sender {self.edge_id}] Error sending batch: {e}")
            return False
    
    def send_intersection_data(self, sock: socket.socket, folder: str):
        """Send all data from one intersection folder"""
        print(f"[Sender {self.edge_id}] Sending data from {folder}")
        
        folder_path = os.path.join(self.image_dir, folder)
        frames = load_frames_from_folder(folder_path, self.max_frames_per_video)
        
        if not frames:
            print(f"[Sender {self.edge_id}] No frames found in {folder}")
            return
        
        print(f"[Sender {self.edge_id}] Loaded {len(frames)} frames from {folder}")
        
        # Send in batches
        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_ids = list(range(i, min(i + batch_size, len(frames))))
            
            success = self.send_image_batch(
                sock, batch_frames, [0] * len(batch_frames), batch_ids
            )
            
            if not success:
                break
            
            self.total_frames_sent += len(batch_frames)
            
            # Rate limiting
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                expected_frames = elapsed * self.fps
                if self.total_frames_sent > expected_frames:
                    sleep_time = (self.total_frames_sent - expected_frames) / self.fps
                    time.sleep(sleep_time)
    
    def run(self):
        """Main sender loop"""
        self.start_time = time.time()
        
        print(f"\n[Sender {self.edge_id}] Starting image stream...")
        print(f"[Sender {self.edge_id}]Streaming {len(self.intersection_folders)} intersections")
        
        for folder in self.intersection_folders:
            sock = self.connect_to_edge()
            self.send_intersection_data(sock, folder)
            sock.close()
        
        elapsed = time.time() - self.start_time
        print(f"\n[Sender {self.edge_id}] Complete!")
        print(f"[Sender {self.edge_id}] Sent {self.total_frames_sent} frames in {elapsed:.2f}s")
        print(f"[Sender {self.edge_id}] Effective rate: {self.total_frames_sent/elapsed:.2f} FPS")


def main():
    parser = argparse.ArgumentParser(description='UA-DETRAC Image Sender')
    parser.add_argument('--edge-id', type=int, required=True, help='Edge device ID')
    parser.add_argument('--intersection-indices', type=int, nargs='+', required=True,
                       help='Intersection indices to stream')
    parser.add_argument('--target-host', type=str, required=True,
                       help='Edge VM IP address')
    parser.add_argument('--target-port', type=int, default=DEFAULT_TARGET_PORT,
                       help=f'Edge port (default: {DEFAULT_TARGET_PORT})')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                       help=f'Frames per second (default: {DEFAULT_FPS})')
    parser.add_argument('--max-frames-per-video', type=int, default=50,
                       help='Maximum frames per video')
    
    args = parser.parse_args()
    
    sender = Sender(
        edge_id=args.edge_id,
        intersection_indices=args.intersection_indices,
        target_host=args.target_host,
        target_port=args.target_port,
        fps=args.fps,
        max_frames_per_video=args.max_frames_per_video
    )
    
    sender.run()


if __name__ == '__main__':
    main()
