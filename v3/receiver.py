import argparse
import csv
import os
import random
import socket
import threading
import time
from collections import defaultdict
from typing import Dict, List

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
V2_DIR = os.path.join(REPO_ROOT, "v2")


class DelaySampler:
    def __init__(self, mode: str, fixed_delay: float):
        self.mode = mode
        self.fixed_delay = fixed_delay
        self.synthetic = {}
        for name in ("kde", "wgan"):
            paths = [
                os.path.join(BASE_DIR, f"synthetic_interarrival_{name}.csv"),
                os.path.join(V2_DIR, f"synthetic_interarrival_{name}.csv"),
            ]
            path = next((candidate for candidate in paths if os.path.exists(candidate)), None)
            if path is not None:
                values = pd.read_csv(path, header=None).values.flatten()
                values = values[values > 1e-6]
                self.synthetic[name] = values.astype(float)
        if mode in ("kde", "wgan") and mode not in self.synthetic:
            raise FileNotFoundError(
                f"Could not find synthetic_interarrival_{mode}.csv in v3 or v2."
            )

    def sample(self) -> float:
        if self.mode == "fixed":
            return max(self.fixed_delay, 0.0)
        if self.mode in self.synthetic and len(self.synthetic[self.mode]) > 0:
            return max(float(random.choice(self.synthetic[self.mode])), 1e-6)
        return 0.0


class Receiver:
    def __init__(self, args):
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((args.host, args.port))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_seen: Dict[str, float] = {}
        self.prev_sender_ts: Dict[str, float] = {}
        self.prev_global_ts = None
        self.outage_active: Dict[str, bool] = defaultdict(bool)
        self.fallback_stops: Dict[str, threading.Event] = {}
        self.partial_frames = {}
        self.queue_length = 0
        self.delay_sampler = DelaySampler(args.fallback_mode, args.fixed_fallback_delay)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        self.output_file = open(args.output, "w", newline="")
        self.writer = csv.writer(self.output_file)
        self.writer.writerow(
            [
                "timestamp",
                "sender_id",
                "interarrival_sender",
                "interarrival_global",
                "event",
                "source",
                "queue_length",
                "frame_key",
            ]
        )

    def close(self) -> None:
        self.stop_event.set()
        for event in self.fallback_stops.values():
            event.set()
        self.output_file.close()

    def write_event(
        self,
        timestamp: float,
        sender_id: str,
        interarrival_sender,
        interarrival_global,
        event: str,
        source: str,
        frame_key: str,
    ) -> None:
        with self.lock:
            self.writer.writerow(
                [
                    f"{timestamp:.9f}",
                    sender_id,
                    "" if interarrival_sender is None else f"{interarrival_sender:.9f}",
                    "" if interarrival_global is None else f"{interarrival_global:.9f}",
                    event,
                    source,
                    self.queue_length,
                    frame_key,
                ]
            )
            self.output_file.flush()

    def parse_packet(self, packet: bytes):
        try:
            header, chunk = packet.split(b"||", 1)
            parts = header.decode().split("|")
            if len(parts) not in (7, 9):
                raise ValueError("unexpected header field count")
            if len(parts) == 7:
                sender_id, seq, folder_id, frame_id, chunk_id, total_chunks, send_ts = parts
                sample_id = 0
                label = -1
            else:
                (
                    sender_id,
                    seq,
                    folder_id,
                    frame_id,
                    sample_id,
                    label,
                    chunk_id,
                    total_chunks,
                    send_ts,
                ) = parts
            return {
                "sender_id": sender_id,
                "seq": int(seq),
                "folder_id": int(folder_id),
                "frame_id": int(frame_id),
                "sample_id": int(sample_id),
                "label": int(label),
                "chunk_id": int(chunk_id),
                "total_chunks": int(total_chunks),
                "send_ts": float(send_ts),
                "chunk": chunk,
            }
        except Exception as exc:
            raise ValueError(f"Invalid packet header: {exc}") from exc

    def receive_forever(self) -> None:
        print(f"Receiver listening on {self.args.host}:{self.args.port}", flush=True)
        monitor = threading.Thread(target=self.monitor_outages, daemon=True)
        monitor.start()

        while not self.stop_event.is_set():
            packet, _addr = self.sock.recvfrom(self.args.recv_bytes)
            now = time.perf_counter()
            parsed = self.parse_packet(packet)
            sender_id = parsed["sender_id"]
            frame_key = f"{sender_id}:{parsed['seq']}:{parsed['folder_id']}:{parsed['frame_id']}"

            interarrival_sender = (
                now - self.prev_sender_ts[sender_id]
                if sender_id in self.prev_sender_ts
                else None
            )
            interarrival_global = (
                now - self.prev_global_ts if self.prev_global_ts is not None else None
            )
            self.prev_sender_ts[sender_id] = now
            self.prev_global_ts = now
            self.last_seen[sender_id] = now

            if self.outage_active[sender_id]:
                self.end_outage(sender_id, now)

            completed = self.add_chunk(frame_key, parsed)
            if completed:
                self.queue_length += 1
                self.write_event(
                    timestamp=now,
                    sender_id=sender_id,
                    interarrival_sender=interarrival_sender,
                    interarrival_global=interarrival_global,
                    event="frame_received",
                    source="live",
                    frame_key=frame_key,
                )
                self.queue_length = max(self.queue_length - 1, 0)

    def add_chunk(self, frame_key: str, parsed) -> bool:
        entry = self.partial_frames.setdefault(
            frame_key,
            {
                "total": parsed["total_chunks"],
                "chunks": {},
            },
        )
        entry["chunks"][parsed["chunk_id"]] = parsed["chunk"]
        complete = len(entry["chunks"]) == entry["total"]
        if complete:
            del self.partial_frames[frame_key]
        return complete

    def monitor_outages(self) -> None:
        while not self.stop_event.is_set():
            now = time.perf_counter()
            for sender_id, last_seen in list(self.last_seen.items()):
                if not self.outage_active[sender_id] and now - last_seen > self.args.outage_timeout:
                    self.start_outage(sender_id, now)
            time.sleep(self.args.monitor_interval)

    def start_outage(self, sender_id: str, timestamp: float) -> None:
        self.outage_active[sender_id] = True
        self.write_event(
            timestamp=timestamp,
            sender_id=sender_id,
            interarrival_sender=None,
            interarrival_global=None,
            event="outage_start",
            source="control",
            frame_key="",
        )
        print(f"[outage_start] sender={sender_id}", flush=True)
        if self.args.fallback_mode != "none":
            stop = threading.Event()
            self.fallback_stops[sender_id] = stop
            thread = threading.Thread(
                target=self.inject_fallback,
                args=(sender_id, stop),
                daemon=True,
            )
            thread.start()

    def end_outage(self, sender_id: str, timestamp: float) -> None:
        self.outage_active[sender_id] = False
        stop = self.fallback_stops.pop(sender_id, None)
        if stop is not None:
            stop.set()
        self.write_event(
            timestamp=timestamp,
            sender_id=sender_id,
            interarrival_sender=None,
            interarrival_global=None,
            event="outage_end",
            source="control",
            frame_key="",
        )
        print(f"[outage_end] sender={sender_id}", flush=True)

    def inject_fallback(self, sender_id: str, stop_event: threading.Event) -> None:
        counter = 0
        while not self.stop_event.is_set() and not stop_event.is_set():
            delay = self.delay_sampler.sample()
            if delay > 0:
                stop_event.wait(delay)
            else:
                stop_event.wait(self.args.min_fallback_interval)
            if stop_event.is_set():
                break
            now = time.perf_counter()
            frame_key = f"{sender_id}:fallback:{counter}"
            self.queue_length += 1
            self.write_event(
                timestamp=now,
                sender_id=sender_id,
                interarrival_sender=None,
                interarrival_global=None,
                event="fallback_frame",
                source=f"fallback_{self.args.fallback_mode}",
                frame_key=frame_key,
            )
            self.queue_length = max(self.queue_length - 1, 0)
            counter += 1


def parse_args():
    parser = argparse.ArgumentParser(description="v3 receiver with outage detection")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--recv-bytes", type=int, default=4096)
    parser.add_argument("--outage-timeout", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=0.1)
    parser.add_argument(
        "--fallback-mode",
        choices=("none", "fixed", "kde", "wgan"),
        default="none",
    )
    parser.add_argument("--fixed-fallback-delay", type=float, default=0.0)
    parser.add_argument("--min-fallback-interval", type=float, default=0.01)
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "outputs", "receiver_interarrival_log.csv"),
    )
    return parser.parse_args()


def main() -> None:
    receiver = Receiver(parse_args())
    try:
        receiver.receive_forever()
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
