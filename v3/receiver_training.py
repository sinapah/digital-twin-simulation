import argparse
import csv
import io
import os
import queue
import random
import socket
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from training_simulation import (
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    OUTPUT_DIR,
    DelaySampler,
    ResourceSampler,
    SimpleCNN,
    SimulationConfig,
    build_datasets,
    compute_class_weights,
    ensure_dirs,
    evaluate,
    set_seed,
)


class LiveReceiver:
    def __init__(self, args, transform):
        self.args = args
        self.transform = transform
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((args.host, args.port))
        self.stop_event = threading.Event()
        self.last_seen: Dict[str, float] = {}
        self.prev_sender_ts: Dict[str, float] = {}
        self.prev_global_ts: Optional[float] = None
        self.outage_active: Dict[str, bool] = defaultdict(bool)
        self.partial_samples = {}
        self.queues = {
            sender_id: queue.Queue(maxsize=args.queue_size)
            for sender_id in args.expected_senders.split(",")
            if sender_id
        }
        self.lock = threading.Lock()
        self.events = []

    def start(self):
        thread = threading.Thread(target=self.receive_loop, daemon=True)
        thread.start()
        monitor = threading.Thread(target=self.monitor_outages, daemon=True)
        monitor.start()
        return thread

    def receive_loop(self):
        print(f"Receiver/trainer listening on {self.args.host}:{self.args.port}", flush=True)
        while not self.stop_event.is_set():
            packet, _addr = self.sock.recvfrom(self.args.recv_bytes)
            now = time.perf_counter()
            parsed = self.parse_packet(packet)
            sender_id = parsed["sender_id"]
            self.last_seen[sender_id] = now

            if self.outage_active[sender_id]:
                self.outage_active[sender_id] = False
                self.record_event(now, sender_id, "outage_end", "control")

            sample = self.add_chunk(parsed)
            if sample is not None:
                self.enqueue_sample(sender_id, sample)

    def parse_packet(self, packet: bytes):
        header, chunk = packet.split(b"||", 1)
        parts = header.decode().split("|")
        if len(parts) != 9:
            raise ValueError(
                "receiver_training.py requires labeled crop packets from sender.py "
                "in default --mode crops format"
            )
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
        now = time.perf_counter()
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
            "interarrival_sender": interarrival_sender,
            "interarrival_global": interarrival_global,
            "chunk": chunk,
        }

    def add_chunk(self, parsed):
        key = (
            parsed["sender_id"],
            parsed["seq"],
            parsed["folder_id"],
            parsed["frame_id"],
            parsed["sample_id"],
        )
        entry = self.partial_samples.setdefault(
            key,
            {
                "total": parsed["total_chunks"],
                "chunks": {},
                "label": parsed["label"],
                "sender_id": parsed["sender_id"],
                "interarrival_sender": parsed["interarrival_sender"],
                "interarrival_global": parsed["interarrival_global"],
            },
        )
        entry["chunks"][parsed["chunk_id"]] = parsed["chunk"]
        if len(entry["chunks"]) != entry["total"]:
            return None

        payload = b"".join(entry["chunks"][i] for i in range(entry["total"]))
        del self.partial_samples[key]
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        tensor = self.transform(image)
        return {
            "x": tensor,
            "y": entry["label"],
            "source": "live",
            "sender_id": entry["sender_id"],
            "received_at": time.perf_counter(),
            "interarrival_sender": entry["interarrival_sender"],
            "interarrival_global": entry["interarrival_global"],
        }

    def enqueue_sample(self, sender_id: str, sample) -> None:
        if sender_id not in self.queues:
            self.queues[sender_id] = queue.Queue(maxsize=self.args.queue_size)
        try:
            self.queues[sender_id].put_nowait(sample)
            self.record_event(time.perf_counter(), sender_id, "sample_received", "live")
        except queue.Full:
            self.record_event(time.perf_counter(), sender_id, "sample_dropped_queue_full", "live")

    def monitor_outages(self):
        while not self.stop_event.is_set():
            now = time.perf_counter()
            for sender_id, last_seen in list(self.last_seen.items()):
                if (
                    not self.outage_active[sender_id]
                    and now - last_seen > self.args.outage_timeout
                ):
                    self.outage_active[sender_id] = True
                    self.record_event(now, sender_id, "outage_start", "control")
            time.sleep(self.args.monitor_interval)

    def drain(self, sender_id: str, max_samples: int):
        drained = []
        sample_queue = self.queues.setdefault(sender_id, queue.Queue(maxsize=self.args.queue_size))
        for _ in range(max_samples):
            try:
                drained.append(sample_queue.get_nowait())
            except queue.Empty:
                break
        return drained

    def discard_queued(self) -> int:
        discarded = 0
        for sample_queue in self.queues.values():
            while True:
                try:
                    sample_queue.get_nowait()
                    discarded += 1
                except queue.Empty:
                    break
        return discarded

    def queue_lengths(self):
        return {sender_id: sample_queue.qsize() for sender_id, sample_queue in self.queues.items()}

    def record_event(self, timestamp, sender_id, event, source):
        with self.lock:
            self.events.append(
                {
                    "timestamp": timestamp,
                    "sender_id": sender_id,
                    "event": event,
                    "source": source,
                    "queue_length": self.queues.get(sender_id).qsize()
                    if sender_id in self.queues
                    else 0,
                }
            )

    def stop(self):
        self.stop_event.set()
        self.sock.close()


class ClassBalancedReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.samples_by_class = {cls: [] for cls in range(NUM_CLASSES)}
        self.total = 0

    def add_many(self, samples) -> None:
        for sample in samples:
            label = int(sample["y"])
            bucket = self.samples_by_class[label]
            bucket.append(sample)
            self.total += 1
            while self.total > self.max_size:
                self._drop_oldest()

    def _drop_oldest(self) -> None:
        non_empty = [
            (label, bucket)
            for label, bucket in self.samples_by_class.items()
            if bucket
        ]
        if not non_empty:
            return
        label, bucket = max(non_empty, key=lambda item: len(item[1]))
        bucket.pop(0)
        self.total -= 1

    def label_counts(self) -> List[int]:
        return [len(self.samples_by_class[cls]) for cls in range(NUM_CLASSES)]

    def sample_balanced(self, count: int) -> List[dict]:
        available_classes = [
            cls for cls, bucket in self.samples_by_class.items() if bucket
        ]
        if not available_classes:
            return []

        selected = []
        per_class = max(1, count // len(available_classes))
        for cls in available_classes:
            bucket = self.samples_by_class[cls]
            take = min(per_class, len(bucket))
            selected.extend(random.sample(bucket, take))

        while len(selected) < count:
            cls = random.choice(available_classes)
            selected.append(random.choice(self.samples_by_class[cls]))

        random.shuffle(selected)
        return selected[:count]


def compute_capped_class_weights(datasets, max_weight: float) -> torch.Tensor:
    base = compute_class_weights(datasets)
    if max_weight > 0:
        base = torch.clamp(base, max=max_weight)
    return base


def fallback_samples(
    dataset,
    count: int,
    start_index: int,
    sender_id: str,
    delay_sampler: DelaySampler,
    fallback_mode: str,
    time_scale: float,
):
    samples = []
    if len(dataset) == 0:
        return samples, start_index, 0.0

    total_delay = 0.0
    strategy = "fixed" if fallback_mode == "fixed" else fallback_mode
    for _ in range(count):
        delay = delay_sampler.sample(strategy) * time_scale
        if delay > 0:
            time.sleep(delay)
        x, y = dataset[start_index % len(dataset)]
        samples.append(
            {
                "x": x,
                "y": y,
                "source": f"fallback_{fallback_mode}",
                "sender_id": sender_id,
                "received_at": time.perf_counter(),
            }
        )
        total_delay += delay
        start_index += 1
    return samples, start_index, total_delay


def train_batch(model, optimizer, loss_fn, samples):
    if not samples:
        return 0.0, float("nan")
    x = torch.stack([sample["x"] for sample in samples]).to(DEVICE)
    y = torch.tensor([sample["y"] for sample in samples], dtype=torch.long).to(DEVICE)
    start = time.time()
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    return time.time() - start, float(loss.item())


def evaluate_with_distribution(model, loader):
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    pred_counts = [0] * NUM_CLASSES

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            preds = model(x).argmax(1)
            for cls in range(NUM_CLASSES):
                target_mask = y == cls
                pred_counts[cls] += (preds == cls).sum().item()
                class_correct[cls] += (preds[target_mask] == cls).sum().item()
                class_total[cls] += target_mask.sum().item()

    total = sum(class_total)
    correct = sum(class_correct)
    per_class = [
        class_correct[cls] / class_total[cls] if class_total[cls] > 0 else float("nan")
        for cls in range(NUM_CLASSES)
    ]
    valid_per_class = [value for value in per_class if not np.isnan(value)]
    balanced_accuracy = float(np.mean(valid_per_class)) if valid_per_class else 0.0
    return (
        correct / total if total > 0 else 0.0,
        balanced_accuracy,
        per_class,
        pred_counts,
        class_total,
    )


def write_events(events: List[dict], path: str) -> None:
    if not events:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "sender_id", "event", "source", "queue_length"],
        )
        writer.writeheader()
        writer.writerows(events)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train on labeled UDP crops at the receiver, with local fallback during outages."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--recv-bytes", type=int, default=4096)
    parser.add_argument("--expected-senders", default="camera0,camera1,camera2")
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--outage-timeout", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=0.1)
    parser.add_argument("--fallback-mode", choices=("none", "fixed", "kde", "wgan"), default="kde")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--samples-per-sender-per-round", type=int, default=32)
    parser.add_argument("--round-collect-seconds", type=float, default=2.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--allow-empty-rounds", action="store_true")
    parser.add_argument(
        "--keep-stale-queue",
        action="store_true",
        help="Do not discard queued samples before each collection window. Useful for throughput stress tests, but not for outage semantics.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-batches-per-round", type=int, default=4)
    parser.add_argument("--replay-buffer-size", type=int, default=10000)
    parser.add_argument("--max-class-weight", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--videos-per-agent", type=int, default=1)
    parser.add_argument("--historical-videos-per-agent", type=int, default=5)
    parser.add_argument("--test-videos", type=int, default=10)
    parser.add_argument("--max-frames-per-video", type=int, default=50)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)

    config = SimulationConfig(
        videos_per_agent=args.videos_per_agent,
        historical_videos_per_agent=args.historical_videos_per_agent,
        test_videos=args.test_videos,
        max_frames_per_video=args.max_frames_per_video,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        time_scale=args.time_scale,
        seed=args.seed,
    )
    _live_datasets, historical_datasets, test_dataset = build_datasets(config)
    class_weights = compute_capped_class_weights(historical_datasets, args.max_class_weight)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    delay_sampler = DelaySampler(config)

    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
        ]
    )
    receiver = LiveReceiver(args, transform)
    receiver.start()

    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    sender_ids = [sender for sender in args.expected_senders.split(",") if sender]
    fallback_indices = {sender_id: 0 for sender_id in sender_ids}
    replay_buffer = ClassBalancedReplayBuffer(args.replay_buffer_size)
    metrics = []

    try:
        print("Waiting for live sender data...", flush=True)
        startup_deadline = time.time() + args.startup_timeout
        while (
            not any(receiver.queue_lengths().values())
            and time.time() < startup_deadline
        ):
            time.sleep(0.25)
        if not any(receiver.queue_lengths().values()):
            raise RuntimeError(
                "No live samples arrived before startup timeout. "
                "Check that sender processes are running, TARGET_HOST points to this VM, "
                "the port is open, and sender dependencies are installed. "
                "Use --allow-empty-rounds only for debugging receiver startup."
            )
        for round_number in range(1, args.rounds + 1):
            with ResourceSampler() as resources:
                round_start = time.time()
                stale_discarded = 0 if args.keep_stale_queue else receiver.discard_queued()
                time.sleep(args.round_collect_seconds)

                round_samples = []
                fallback_delay = 0.0
                for sender_index, sender_id in enumerate(sender_ids):
                    live = receiver.drain(sender_id, args.samples_per_sender_per_round)
                    if live:
                        round_samples.extend(live)
                        continue

                    if receiver.outage_active[sender_id] and args.fallback_mode != "none":
                        fallback, next_index, delay = fallback_samples(
                            historical_datasets[sender_index % len(historical_datasets)],
                            args.samples_per_sender_per_round,
                            fallback_indices[sender_id],
                            sender_id,
                            delay_sampler,
                            args.fallback_mode,
                            args.time_scale,
                        )
                        fallback_indices[sender_id] = next_index
                        fallback_delay += delay
                        round_samples.extend(fallback)

                random.shuffle(round_samples)
                replay_buffer.add_many(round_samples)
                train_sample_count = min(
                    replay_buffer.total,
                    max(args.batch_size, args.batch_size * args.train_batches_per_round),
                )
                train_samples = replay_buffer.sample_balanced(train_sample_count)
                compute_time = 0.0
                batch_losses = []
                for start in range(0, len(train_samples), args.batch_size):
                    batch_time, batch_loss = train_batch(
                        model,
                        optimizer,
                        loss_fn,
                        train_samples[start : start + args.batch_size],
                    )
                    compute_time += batch_time
                    if not np.isnan(batch_loss):
                        batch_losses.append(batch_loss)

                elapsed = time.time() - round_start
                queues = receiver.queue_lengths()
                live_count = sum(1 for sample in round_samples if sample["source"] == "live")
                fallback_count = len(round_samples) - live_count
                if not train_samples and not args.allow_empty_rounds:
                    raise RuntimeError(
                        f"Round {round_number} had zero training samples. "
                        "Sender data is not reaching the receiver and no fallback samples "
                        "were available. Refusing to write misleading accuracy metrics."
                    )
            resource = resources.summary()
            (
                frame_acc,
                balanced_accuracy,
                per_class,
                pred_counts,
                test_label_counts,
            ) = evaluate_with_distribution(model, test_loader)
            train_label_counts = [0] * NUM_CLASSES
            for sample in train_samples:
                train_label_counts[int(sample["y"])] += 1
            train_live_count = sum(1 for sample in train_samples if sample["source"] == "live")
            train_fallback_count = len(train_samples) - train_live_count
            received_label_counts = [0] * NUM_CLASSES
            for sample in round_samples:
                received_label_counts[int(sample["y"])] += 1
            buffer_label_counts = replay_buffer.label_counts()
            row = {
                "round": round_number,
                "frame_accuracy": frame_acc,
                "balanced_accuracy": balanced_accuracy,
                "samples_processed": len(train_samples),
                "samples_trained": len(train_samples),
                "samples_received": len(round_samples),
                "stale_samples_discarded": stale_discarded,
                "live_samples": live_count,
                "live_samples_trained": train_live_count,
                "fallback_samples": fallback_count,
                "fallback_samples_trained": train_fallback_count,
                "fallback_delay": fallback_delay,
                "compute_time": compute_time,
                "train_loss_avg": float(np.mean(batch_losses)) if batch_losses else float("nan"),
                "round_wall_time": elapsed,
                "samples_trained_per_second": len(train_samples) / elapsed if elapsed > 0 else 0,
                "images_processed_per_second": len(round_samples) / elapsed if elapsed > 0 else 0,
                "replay_buffer_size": replay_buffer.total,
                "queue_length_avg": float(np.mean(list(queues.values()))) if queues else 0.0,
                "queue_length_max": max(queues.values()) if queues else 0,
                "outage_active": any(receiver.outage_active.values()),
                "resource_scope": "receive_fallback_train_excludes_evaluation",
                **resource,
            }
            for cls in range(NUM_CLASSES):
                row[f"acc_{CLASS_NAMES[cls]}"] = per_class[cls]
                row[f"pred_{CLASS_NAMES[cls]}"] = pred_counts[cls]
                row[f"test_{CLASS_NAMES[cls]}"] = test_label_counts[cls]
                row[f"train_{CLASS_NAMES[cls]}"] = train_label_counts[cls]
                row[f"received_{CLASS_NAMES[cls]}"] = received_label_counts[cls]
                row[f"buffer_{CLASS_NAMES[cls]}"] = buffer_label_counts[cls]
            metrics.append(row)
            print(
                f"Round {round_number}: acc={frame_acc*100:.2f}% "
                f"balanced={balanced_accuracy*100:.2f}% "
                f"received={len(round_samples)} trained={len(train_samples)} "
                f"live={live_count} fallback={fallback_count} "
                f"stale_discarded={stale_discarded} "
                f"cpu_avg={row['cpu_avg']:.1f}% "
                f"host_cpu_avg={row['cpu_avg_host_percent']:.1f}%",
                flush=True,
            )
    finally:
        receiver.stop()

    metrics_path = os.path.join(OUTPUT_DIR, "receiver_training_metrics.csv")
    events_path = os.path.join(OUTPUT_DIR, "receiver_training_events.csv")
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    write_events(receiver.events, events_path)
    print(f"Saved {metrics_path}")
    print(f"Saved {events_path}")


if __name__ == "__main__":
    main()
