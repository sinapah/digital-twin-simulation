import argparse
import copy
import os
import random
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import psutil
except ImportError as exc:
    raise ImportError(
        "v3 resource metrics require psutil. Install dependencies with "
        "`pip install -r v3/requirements.txt`."
    ) from exc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
V2_DIR = os.path.join(REPO_ROOT, "v2")
DETRAC_IMAGES_DIR = os.path.join(REPO_ROOT, "DETRAC-Images", "DETRAC-Images")
DETRAC_ANNOT_DIR = os.path.join(
    REPO_ROOT,
    "DETRAC-Train-Annotations-XML",
    "DETRAC-Train-Annotations-XML",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")

VEHICLE_CLASSES = {"car": 0, "van": 1, "bus": 2, "others": 3}
CLASS_NAMES = {v: k for k, v in VEHICLE_CLASSES.items()}
NUM_CLASSES = len(VEHICLE_CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class SimulationConfig:
    num_agents: int = 3
    videos_per_agent: int = 10
    historical_videos_per_agent: int = 5
    test_videos: int = 10
    max_frames_per_video: int = 50
    rounds: int = 100
    local_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-3
    img_size: int = 64
    architecture: str = "peer_to_peer"
    upload_mean: float = 0.3
    upload_std: float = 0.1
    download_mean: float = 0.2
    download_std: float = 0.05
    upload_concurrency: int = 1
    download_concurrency: int = 1
    outage_start_round: int = 20
    outage_end_round: int = 60
    outage_agents: Tuple[int, ...] = (0,)
    fixed_replay_delay: float = 0.0
    time_scale: float = 1.0
    seed: int = 42
    track_eval_interval: int = 10


@dataclass(frozen=True)
class Scenario:
    name: str
    fallback_strategy: str
    delay_strategy: str
    enable_outage: bool


SCENARIOS = {
    "baseline_live": Scenario(
        name="baseline_live",
        fallback_strategy="live",
        delay_strategy="live",
        enable_outage=False,
    ),
    "outage_no_fallback": Scenario(
        name="outage_no_fallback",
        fallback_strategy="none",
        delay_strategy="none",
        enable_outage=True,
    ),
    "outage_replay_fixed": Scenario(
        name="outage_replay_fixed",
        fallback_strategy="historical",
        delay_strategy="fixed",
        enable_outage=True,
    ),
    "outage_replay_kde": Scenario(
        name="outage_replay_kde",
        fallback_strategy="historical",
        delay_strategy="kde",
        enable_outage=True,
    ),
    "outage_replay_wgan": Scenario(
        name="outage_replay_wgan",
        fallback_strategy="historical",
        delay_strategy="wgan",
        enable_outage=True,
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)


class DelaySampler:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.synthetic = {
            "kde": self._load_synthetic_delays("kde"),
            "wgan": self._load_synthetic_delays("wgan"),
        }
        self.live_delays = self.synthetic["wgan"]

    def _load_synthetic_delays(self, model: str) -> np.ndarray:
        candidates = [
            os.path.join(BASE_DIR, f"synthetic_interarrival_{model}.csv"),
            os.path.join(V2_DIR, f"synthetic_interarrival_{model}.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                delays = pd.read_csv(path, header=None).values.flatten()
                delays = delays[delays > 1e-6]
                if len(delays) == 0:
                    raise RuntimeError(f"No valid positive delays in {path}")
                return delays.astype(float)
        raise FileNotFoundError(
            f"Could not find synthetic_interarrival_{model}.csv in v3 or v2."
        )

    def sample(self, strategy: str) -> float:
        if strategy in ("none", "no_data"):
            return 0.0
        if strategy == "fixed":
            return max(self.config.fixed_replay_delay, 0.0)
        if strategy == "live":
            return max(float(random.choice(self.live_delays)), 1e-6)
        if strategy in self.synthetic:
            return max(float(random.choice(self.synthetic[strategy])), 1e-6)
        raise ValueError(f"Unknown delay strategy: {strategy}")


class ResourceSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._record()

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._record()
            time.sleep(self.interval)

    def _record(self) -> None:
        self.cpu_samples.append(self.process.cpu_percent(interval=None))
        self.memory_samples.append(self.process.memory_info().rss / (1024 * 1024))

    def summary(self) -> Dict[str, float]:
        cpu = np.array(self.cpu_samples, dtype=float) if self.cpu_samples else np.array([0.0])
        mem = (
            np.array(self.memory_samples, dtype=float)
            if self.memory_samples
            else np.array([0.0])
        )
        cpu_count = psutil.cpu_count() or 1
        cpu_avg = float(np.mean(cpu))
        cpu_peak = float(np.max(cpu))
        return {
            "cpu_avg": cpu_avg,
            "cpu_peak": cpu_peak,
            "cpu_var": float(np.var(cpu)),
            "cpu_avg_host_percent": cpu_avg / cpu_count,
            "cpu_peak_host_percent": cpu_peak / cpu_count,
            "memory_avg_mb": float(np.mean(mem)),
            "memory_peak_mb": float(np.max(mem)),
        }


class DETRACDataset(Dataset):
    def __init__(
        self,
        video_dirs: Sequence[str],
        annotation_dir: str,
        transform=None,
        max_frames_per_video: int = 50,
    ):
        self.samples: List[Tuple[str, Tuple[float, float, float, float], int, str, int]] = []
        self.video_metadata: Dict[str, Dict[str, str]] = {}
        self.transform = transform

        for video_dir in video_dirs:
            seq_name = os.path.basename(video_dir)
            xml_path = os.path.join(annotation_dir, f"{seq_name}.xml")
            if not os.path.exists(xml_path):
                print(f"Missing annotation: {xml_path}")
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            seq_attr = root.find("sequence_attribute")
            self.video_metadata[seq_name] = {
                "weather": (
                    seq_attr.get("sence_weather", "unknown")
                    if seq_attr is not None
                    else "unknown"
                ),
                "camera_state": (
                    seq_attr.get("camera_state", "unknown")
                    if seq_attr is not None
                    else "unknown"
                ),
            }

            frames = root.findall("frame")
            step = max(1, len(frames) // max_frames_per_video)
            sampled = frames[::step][:max_frames_per_video]

            for frame_elem in sampled:
                frame_num = int(frame_elem.get("num"))
                img_path = os.path.join(video_dir, f"img{frame_num:05d}.jpg")
                if not os.path.exists(img_path):
                    continue

                for target in frame_elem.findall(".//target"):
                    box_elem = target.find("box")
                    attr_elem = target.find("attribute")
                    if box_elem is None or attr_elem is None:
                        continue

                    track_id = int(target.get("id", -1))
                    vtype = attr_elem.get("vehicle_type", "others")
                    label = VEHICLE_CLASSES.get(vtype, 3)
                    left = float(box_elem.get("left"))
                    top = float(box_elem.get("top"))
                    width = float(box_elem.get("width"))
                    height = float(box_elem.get("height"))

                    self.samples.append(
                        (img_path, (left, top, width, height), label, seq_name, track_id)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (left, top, width, height), label, _video_id, _track_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size

        x1 = max(0, int(left))
        y1 = max(0, int(top))
        x2 = min(iw, int(left + width))
        y2 = min(ih, int(top + height))
        crop = img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else img

        if self.transform:
            crop = self.transform(crop)

        return crop, label

    def get_track_index_groups(self):
        groups = defaultdict(list)
        for idx, (_, _, _, video_id, track_id) in enumerate(self.samples):
            groups[(video_id, track_id)].append(idx)
        return dict(groups)

    def get_condition_index_groups(self, condition_key: str):
        groups = defaultdict(list)
        for idx, (_, _, _, video_id, _) in enumerate(self.samples):
            cond = self.video_metadata.get(video_id, {}).get(condition_key, "unknown")
            groups[cond].append(idx)
        return dict(groups)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class AgentRoundMetrics:
    agent_id: int
    data_source: str
    outage_active: bool
    samples_processed: int
    live_samples: int
    fallback_samples: int
    ingestion_wait: float
    compute_time: float
    upload_queue_wait: float
    upload_transfer_time: float
    logical_queue_avg: float
    logical_queue_max: int


class EdgeAgent:
    def __init__(
        self,
        agent_id: int,
        live_dataset: DETRACDataset,
        historical_dataset: DETRACDataset,
        class_weights: Optional[torch.Tensor],
        config: SimulationConfig,
        delay_sampler: DelaySampler,
        upload_semaphore: threading.Semaphore,
    ):
        self.id = agent_id
        self.live_dataset = live_dataset
        self.historical_dataset = historical_dataset
        self.class_weights = class_weights
        self.config = config
        self.delay_sampler = delay_sampler
        self.upload_semaphore = upload_semaphore
        self.model = SimpleCNN().to(DEVICE)

    def is_outage(self, round_number: int, scenario: Scenario) -> bool:
        if not scenario.enable_outage:
            return False
        return (
            self.id in self.config.outage_agents
            and self.config.outage_start_round <= round_number <= self.config.outage_end_round
        )

    def _select_dataset(
        self, round_number: int, scenario: Scenario
    ) -> Tuple[Optional[DETRACDataset], str, str]:
        outage = self.is_outage(round_number, scenario)
        if not outage:
            return self.live_dataset, "live", "live"
        if scenario.fallback_strategy == "none":
            return None, "none", "none"
        if scenario.fallback_strategy == "historical":
            return self.historical_dataset, "fallback", scenario.delay_strategy
        raise ValueError(f"Unknown fallback strategy: {scenario.fallback_strategy}")

    def _upload_transfer_latency(self) -> float:
        scale = {"centralized": 1.0, "regional": 0.6, "peer_to_peer": 0.4}
        factor = scale.get(self.config.architecture, 1.0)
        delay = max(np.random.normal(self.config.upload_mean * factor, self.config.upload_std), 0)
        return delay * self.config.time_scale

    def train(self, global_weights, round_number: int, scenario: Scenario):
        self.model.load_state_dict(global_weights)
        dataset, data_source, delay_strategy = self._select_dataset(round_number, scenario)
        outage_active = self.is_outage(round_number, scenario)

        if dataset is None or len(dataset) == 0:
            return self.model.state_dict(), AgentRoundMetrics(
                agent_id=self.id,
                data_source=data_source,
                outage_active=outage_active,
                samples_processed=0,
                live_samples=0,
                fallback_samples=0,
                ingestion_wait=0.0,
                compute_time=0.0,
                upload_queue_wait=0.0,
                upload_transfer_time=0.0,
                logical_queue_avg=0.0,
                logical_queue_max=0,
            )

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(DEVICE) if self.class_weights is not None else None
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        ingestion_total = 0.0
        compute_total = 0.0
        samples_processed = 0
        queue_lengths: List[int] = []

        self.model.train()
        for _ in range(self.config.local_epochs):
            for x, y in loader:
                sampled_delay = self.delay_sampler.sample(delay_strategy)
                sleep_delay = sampled_delay * self.config.time_scale
                if sleep_delay > 0:
                    time.sleep(sleep_delay)
                ingestion_total += sleep_delay

                if delay_strategy == "fixed" and sampled_delay == 0:
                    queue_lengths.append(int(x.size(0)))
                else:
                    queue_lengths.append(1)

                t0 = time.time()
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad()
                preds = self.model(x)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                compute_total += time.time() - t0
                samples_processed += int(x.size(0))

        queue_start = time.time()
        self.upload_semaphore.acquire()
        upload_queue_wait = time.time() - queue_start
        try:
            upload_transfer = self._upload_transfer_latency()
            if upload_transfer > 0:
                time.sleep(upload_transfer)
        finally:
            self.upload_semaphore.release()

        live_samples = samples_processed if data_source == "live" else 0
        fallback_samples = samples_processed if data_source == "fallback" else 0
        metrics = AgentRoundMetrics(
            agent_id=self.id,
            data_source=data_source,
            outage_active=outage_active,
            samples_processed=samples_processed,
            live_samples=live_samples,
            fallback_samples=fallback_samples,
            ingestion_wait=ingestion_total,
            compute_time=compute_total,
            upload_queue_wait=upload_queue_wait,
            upload_transfer_time=upload_transfer,
            logical_queue_avg=float(np.mean(queue_lengths)) if queue_lengths else 0.0,
            logical_queue_max=max(queue_lengths) if queue_lengths else 0,
        )
        return self.model.state_dict(), metrics

    def receive_model(self, global_weights, download_semaphore: threading.Semaphore) -> float:
        queue_start = time.time()
        download_semaphore.acquire()
        queue_wait = time.time() - queue_start
        try:
            transfer = max(np.random.normal(self.config.download_mean, self.config.download_std), 0)
            transfer *= self.config.time_scale
            if transfer > 0:
                time.sleep(transfer)
        finally:
            download_semaphore.release()
        self.model.load_state_dict(global_weights)
        return queue_wait + transfer


def aggregate_models(models, weights):
    positive = [(model, weight) for model, weight in zip(models, weights) if weight > 0]
    if not positive:
        global_model = SimpleCNN().to(DEVICE)
        global_model.load_state_dict(models[0])
        return global_model

    global_model = SimpleCNN().to(DEVICE)
    total = sum(weight for _, weight in positive)
    agg = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_model.state_dict().items()
    }

    for model_state, weight in positive:
        scaled = weight / total
        for k in agg:
            agg[k] += scaled * model_state[k].float()

    ref = global_model.state_dict()
    agg = {k: agg[k].to(ref[k].dtype) for k in agg}
    global_model.load_state_dict(agg)
    return global_model


def evaluate(model, loader):
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            preds = model(x).argmax(1)
            for cls in range(NUM_CLASSES):
                mask = y == cls
                class_correct[cls] += (preds[mask] == cls).sum().item()
                class_total[cls] += mask.sum().item()

    total = sum(class_total)
    correct = sum(class_correct)
    frame_acc = correct / total if total > 0 else 0.0
    per_class = [
        class_correct[cls] / class_total[cls] if class_total[cls] > 0 else float("nan")
        for cls in range(NUM_CLASSES)
    ]
    return frame_acc, per_class


def evaluate_per_track(model, dataset):
    model.eval()
    track_groups = dataset.get_track_index_groups()
    correct = 0
    total = 0

    with torch.no_grad():
        for (_video_id, _track_id), indices in track_groups.items():
            true_label = dataset.samples[indices[0]][2]
            crops = torch.stack([dataset[i][0] for i in indices]).to(DEVICE)
            preds = model(crops).argmax(1)
            vote = torch.mode(preds).values.item()
            correct += int(vote == true_label)
            total += 1

    return correct / total if total > 0 else 0.0


def evaluate_per_condition(model, dataset):
    model.eval()
    results = {}
    for condition_key in ("weather", "camera_state"):
        cond_groups = dataset.get_condition_index_groups(condition_key)
        cond_acc = {}
        with torch.no_grad():
            for cond_value, indices in cond_groups.items():
                correct = 0
                total = 0
                for batch_start in range(0, len(indices), 128):
                    batch = [dataset[i] for i in indices[batch_start : batch_start + 128]]
                    x = torch.stack([sample[0] for sample in batch]).to(DEVICE)
                    y = torch.tensor([sample[1] for sample in batch]).to(DEVICE)
                    preds = model(x).argmax(1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                cond_acc[cond_value] = correct / total if total > 0 else float("nan")
        results[condition_key] = cond_acc
    return results


def list_annotated_videos() -> List[str]:
    videos = sorted(
        os.path.join(DETRAC_IMAGES_DIR, name)
        for name in os.listdir(DETRAC_IMAGES_DIR)
        if os.path.isdir(os.path.join(DETRAC_IMAGES_DIR, name))
        and os.path.exists(os.path.join(DETRAC_ANNOT_DIR, f"{name}.xml"))
    )
    if not videos:
        raise RuntimeError("No annotated DETRAC videos found.")
    return videos


def build_transforms(config: SimulationConfig):
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, test_transform


def build_datasets(config: SimulationConfig):
    videos = list_annotated_videos()
    total_needed = (
        config.num_agents * config.videos_per_agent
        + config.num_agents * config.historical_videos_per_agent
        + config.test_videos
    )
    if len(videos) < total_needed:
        raise RuntimeError(
            "Not enough annotated videos for disjoint live/history/test splits: "
            f"need {total_needed}, found {len(videos)}"
        )

    rng = np.random.default_rng(config.seed)
    shuffled = [videos[i] for i in rng.permutation(len(videos))]
    train_transform, test_transform = build_transforms(config)

    cursor = 0
    live_splits = []
    for _ in range(config.num_agents):
        live_splits.append(shuffled[cursor : cursor + config.videos_per_agent])
        cursor += config.videos_per_agent

    historical_splits = []
    for _ in range(config.num_agents):
        historical_splits.append(
            shuffled[cursor : cursor + config.historical_videos_per_agent]
        )
        cursor += config.historical_videos_per_agent

    test_split = shuffled[cursor : cursor + config.test_videos]

    live_datasets = [
        DETRACDataset(
            split,
            DETRAC_ANNOT_DIR,
            transform=train_transform,
            max_frames_per_video=config.max_frames_per_video,
        )
        for split in live_splits
    ]
    historical_datasets = [
        DETRACDataset(
            split,
            DETRAC_ANNOT_DIR,
            transform=train_transform,
            max_frames_per_video=config.max_frames_per_video,
        )
        for split in historical_splits
    ]
    test_dataset = DETRACDataset(
        test_split,
        DETRAC_ANNOT_DIR,
        transform=test_transform,
        max_frames_per_video=config.max_frames_per_video,
    )
    return live_datasets, historical_datasets, test_dataset


def compute_class_weights(datasets: Sequence[DETRACDataset]) -> torch.Tensor:
    label_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for dataset in datasets:
        for sample in dataset.samples:
            label_counts[sample[2]] += 1

    return torch.tensor(
        label_counts.sum() / (NUM_CLASSES * label_counts + 1e-6),
        dtype=torch.float32,
    )


def scenario_from_name(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Options: {', '.join(SCENARIOS)}")
    return SCENARIOS[name]


def run_scenario(
    scenario: Scenario,
    config: SimulationConfig,
    live_datasets: Sequence[DETRACDataset],
    historical_datasets: Sequence[DETRACDataset],
    test_dataset: DETRACDataset,
    delay_sampler: DelaySampler,
):
    set_seed(config.seed)
    upload_semaphore = threading.Semaphore(config.upload_concurrency)
    download_semaphore = threading.Semaphore(config.download_concurrency)
    class_weights = compute_class_weights(live_datasets)
    test_loader = DataLoader(test_dataset, batch_size=128)

    agents = [
        EdgeAgent(
            agent_id=i,
            live_dataset=live_datasets[i],
            historical_dataset=historical_datasets[i],
            class_weights=class_weights,
            config=config,
            delay_sampler=delay_sampler,
            upload_semaphore=upload_semaphore,
        )
        for i in range(config.num_agents)
    ]
    global_model = SimpleCNN().to(DEVICE)
    rows = []
    track_rows = []
    scenario_start = time.time()

    print(f"\nRunning scenario: {scenario.name}")
    print(
        "Outage window: "
        f"rounds {config.outage_start_round}-{config.outage_end_round}, "
        f"agents {list(config.outage_agents)}"
    )

    for round_number in range(1, config.rounds + 1):
        round_start = time.time()
        global_snapshot = copy.deepcopy(global_model.state_dict())

        with ResourceSampler() as resource_sampler:
            with ThreadPoolExecutor(max_workers=config.num_agents) as executor:
                futures = [
                    executor.submit(
                        agent.train,
                        copy.deepcopy(global_snapshot),
                        round_number,
                        scenario,
                    )
                    for agent in agents
                ]
                results = [future.result() for future in futures]

            updates = [result[0] for result in results]
            agent_metrics = [result[1] for result in results]
            processed_weights = [metric.samples_processed for metric in agent_metrics]
            global_model = aggregate_models(updates, processed_weights)

            agg_snapshot = copy.deepcopy(global_model.state_dict())
            with ThreadPoolExecutor(max_workers=config.num_agents) as executor:
                downloads = [
                    executor.submit(agent.receive_model, copy.deepcopy(agg_snapshot), download_semaphore)
                    for agent in agents
                ]
                download_times = [future.result() for future in downloads]

        round_wall_time = time.time() - round_start
        frame_acc, per_class = evaluate(global_model, test_loader)
        resource = resource_sampler.summary()

        samples_processed = sum(metric.samples_processed for metric in agent_metrics)
        live_samples = sum(metric.live_samples for metric in agent_metrics)
        fallback_samples = sum(metric.fallback_samples for metric in agent_metrics)
        images_per_second = samples_processed / round_wall_time if round_wall_time > 0 else 0.0
        outage_active = any(metric.outage_active for metric in agent_metrics)

        row = {
            "scenario": scenario.name,
            "fallback_strategy": scenario.fallback_strategy,
            "delay_strategy": scenario.delay_strategy,
            "round": round_number,
            "outage_active": outage_active,
            "round_wall_time": round_wall_time,
            "frame_accuracy": frame_acc,
            "samples_processed": samples_processed,
            "live_samples": live_samples,
            "fallback_samples": fallback_samples,
            "images_processed_per_second": images_per_second,
            "mean_ingestion_wait": float(np.mean([m.ingestion_wait for m in agent_metrics])),
            "mean_compute_time": float(np.mean([m.compute_time for m in agent_metrics])),
            "mean_upload_queue": float(np.mean([m.upload_queue_wait for m in agent_metrics])),
            "mean_upload_xfer": float(np.mean([m.upload_transfer_time for m in agent_metrics])),
            "mean_download_time": float(np.mean(download_times)),
            "queue_length_avg": float(np.mean([m.logical_queue_avg for m in agent_metrics])),
            "queue_length_max": max(m.logical_queue_max for m in agent_metrics),
            **resource,
        }
        for cls in range(NUM_CLASSES):
            row[f"acc_{CLASS_NAMES[cls]}"] = per_class[cls]
        rows.append(row)

        if (
            round_number % config.track_eval_interval == 0
            or round_number == config.rounds
        ):
            track_acc = evaluate_per_track(global_model, test_dataset)
            condition_results = evaluate_per_condition(global_model, test_dataset)
            track_rows.append(
                {
                    "scenario": scenario.name,
                    "round": round_number,
                    "track_accuracy": track_acc,
                    "condition_results": repr(condition_results),
                }
            )
            print(
                f"  Round {round_number}: frame={frame_acc*100:.2f}% "
                f"track={track_acc*100:.2f}% samples={samples_processed}"
            )
        else:
            print(
                f"  Round {round_number}: frame={frame_acc*100:.2f}% "
                f"samples={samples_processed} fallback={fallback_samples}"
            )

    metrics_df = pd.DataFrame(rows)
    track_df = pd.DataFrame(track_rows)
    total_time = time.time() - scenario_start
    final_track_acc = evaluate_per_track(global_model, test_dataset)
    summary = summarize_scenario(metrics_df, scenario, total_time, final_track_acc)
    return metrics_df, track_df, summary


def summarize_scenario(
    metrics_df: pd.DataFrame,
    scenario: Scenario,
    total_time: float,
    final_track_acc: float,
) -> Dict[str, float]:
    final_row = metrics_df.iloc[-1]
    return {
        "scenario": scenario.name,
        "fallback_strategy": scenario.fallback_strategy,
        "delay_strategy": scenario.delay_strategy,
        "rounds_completed": int(metrics_df["round"].max()),
        "final_frame_accuracy": float(final_row["frame_accuracy"]),
        "final_track_accuracy": float(final_track_acc),
        "accuracy_variance": float(metrics_df["frame_accuracy"].var(ddof=0)),
        "avg_cpu": float(metrics_df["cpu_avg"].mean()),
        "peak_cpu": float(metrics_df["cpu_peak"].max()),
        "avg_memory_mb": float(metrics_df["memory_avg_mb"].mean()),
        "peak_memory_mb": float(metrics_df["memory_peak_mb"].max()),
        "avg_queue_length": float(metrics_df["queue_length_avg"].mean()),
        "max_queue_length": int(metrics_df["queue_length_max"].max()),
        "avg_throughput": float(metrics_df["images_processed_per_second"].mean()),
        "samples_processed": int(metrics_df["samples_processed"].sum()),
        "fallback_samples": int(metrics_df["fallback_samples"].sum()),
        "effective_training_duration": float(metrics_df["round_wall_time"].sum()),
        "total_runtime": total_time,
    }


def save_scenario_outputs(metrics_df: pd.DataFrame, track_df: pd.DataFrame, scenario_name: str) -> None:
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{scenario_name}.csv")
    track_path = os.path.join(OUTPUT_DIR, f"track_domain_{scenario_name}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    track_df.to_csv(track_path, index=False)
    print(f"Saved {metrics_path}")
    print(f"Saved {track_path}")


def plot_scenario(metrics_df: pd.DataFrame, scenario_name: str, config: SimulationConfig) -> None:
    rounds = metrics_df["round"]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, metrics_df["frame_accuracy"], marker="o")
    if metrics_df["outage_active"].any():
        plt.axvspan(
            config.outage_start_round,
            config.outage_end_round,
            alpha=0.15,
            color="red",
            label="Outage window",
        )
        plt.legend()
    plt.title(f"Frame Accuracy - {scenario_name}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"accuracy_{scenario_name}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.stackplot(
        rounds,
        metrics_df["mean_ingestion_wait"],
        metrics_df["mean_compute_time"],
        metrics_df["mean_upload_queue"],
        metrics_df["mean_upload_xfer"],
        labels=["Ingestion", "Compute", "Upload queue", "Upload xfer"],
        alpha=0.8,
    )
    plt.title(f"Mean Time Breakdown - {scenario_name}")
    plt.xlabel("Round")
    plt.ylabel("Seconds")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"time_breakdown_{scenario_name}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, metrics_df["cpu_avg"], label="CPU avg")
    plt.plot(rounds, metrics_df["cpu_peak"], label="CPU peak", alpha=0.7)
    plt.title(f"CPU Utilization - {scenario_name}")
    plt.xlabel("Round")
    plt.ylabel("Percent")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"cpu_{scenario_name}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, metrics_df["queue_length_avg"], label="Average")
    plt.plot(rounds, metrics_df["queue_length_max"], label="Max")
    plt.title(f"Logical Queue Length - {scenario_name}")
    plt.xlabel("Round")
    plt.ylabel("Queued samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"queue_{scenario_name}.png"), dpi=300)
    plt.close()


def plot_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["scenario"], summary_df["final_frame_accuracy"] * 100)
    plt.title("Final Frame Accuracy by Scenario")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "summary_final_accuracy.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["scenario"], summary_df["avg_cpu"])
    plt.title("Average CPU Utilization by Scenario")
    plt.ylabel("CPU (%)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "summary_cpu.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["scenario"], summary_df["avg_queue_length"])
    plt.title("Average Logical Queue Length by Scenario")
    plt.ylabel("Samples")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "summary_queue.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["scenario"], summary_df["avg_throughput"])
    plt.title("Average Throughput by Scenario")
    plt.ylabel("Images processed per second")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "summary_throughput.png"), dpi=300)
    plt.close()


def parse_agents(value: str) -> Tuple[int, ...]:
    if not value:
        return tuple()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="v3 federated training simulation with camera outages and fallback replay."
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS.keys()),
        help="Comma-separated scenario names to run.",
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--videos-per-agent", type=int, default=10)
    parser.add_argument("--historical-videos-per-agent", type=int, default=5)
    parser.add_argument("--test-videos", type=int, default=10)
    parser.add_argument("--max-frames-per-video", type=int, default=50)
    parser.add_argument("--outage-start-round", type=int, default=20)
    parser.add_argument("--outage-end-round", type=int, default=60)
    parser.add_argument("--outage-agents", default="0")
    parser.add_argument("--fixed-replay-delay", type=float, default=0.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track-eval-interval", type=int, default=10)
    parser.add_argument("--upload-concurrency", type=int, default=1)
    parser.add_argument("--download-concurrency", type=int, default=1)
    return parser.parse_args()


def build_config(args) -> SimulationConfig:
    return SimulationConfig(
        videos_per_agent=args.videos_per_agent,
        historical_videos_per_agent=args.historical_videos_per_agent,
        test_videos=args.test_videos,
        max_frames_per_video=args.max_frames_per_video,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        outage_start_round=args.outage_start_round,
        outage_end_round=args.outage_end_round,
        outage_agents=parse_agents(args.outage_agents),
        fixed_replay_delay=args.fixed_replay_delay,
        time_scale=args.time_scale,
        seed=args.seed,
        track_eval_interval=args.track_eval_interval,
        upload_concurrency=args.upload_concurrency,
        download_concurrency=args.download_concurrency,
    )


def selected_scenarios(value: str) -> List[Scenario]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    return [scenario_from_name(name) for name in names]


def main() -> None:
    args = parse_args()
    config = build_config(args)
    scenarios = selected_scenarios(args.scenarios)
    ensure_dirs()
    set_seed(config.seed)

    print("Building disjoint live, historical, and test datasets...")
    live_datasets, historical_datasets, test_dataset = build_datasets(config)
    for i, dataset in enumerate(live_datasets):
        print(f"  Agent {i} live samples:       {len(dataset)}")
    for i, dataset in enumerate(historical_datasets):
        print(f"  Agent {i} historical samples: {len(dataset)}")
    print(f"  Test samples:                 {len(test_dataset)}")

    delay_sampler = DelaySampler(config)
    summaries = []
    for scenario in scenarios:
        metrics_df, track_df, summary = run_scenario(
            scenario,
            config,
            live_datasets,
            historical_datasets,
            test_dataset,
            delay_sampler,
        )
        save_scenario_outputs(metrics_df, track_df, scenario.name)
        plot_scenario(metrics_df, scenario.name, config)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(OUTPUT_DIR, "scenario_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    plot_summary(summary_df)
    print(f"\nSaved {summary_path}")
    print("Saved v3 visualizations in v3/visualizations/")


if __name__ == "__main__":
    main()
