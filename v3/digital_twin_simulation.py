import argparse
import copy
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training_simulation import (
    BASE_DIR,
    CLASS_NAMES,
    DETRAC_ANNOT_DIR,
    DETRAC_IMAGES_DIR,
    DEVICE,
    DETRACDataset,
    NUM_CLASSES,
    DelaySampler,
    ResourceSampler,
    SimpleCNN,
    SimulationConfig,
    aggregate_models,
    build_transforms,
    compute_class_weights,
    evaluate,
    list_annotated_videos,
    set_seed,
)


DIGITAL_TWIN_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "digital_twin")


@dataclass(frozen=True)
class IntersectionSpec:
    id: str
    role: str
    edge: Optional[str]
    sequences: Tuple[str, ...]


@dataclass(frozen=True)
class DigitalTwinManifest:
    edges: Tuple[str, ...]
    intersections: Tuple[IntersectionSpec, ...]
    evaluation_sequences: Tuple[str, ...]


@dataclass(frozen=True)
class DigitalTwinScenario:
    name: str
    fallback_mode: str
    enable_outage: bool


SCENARIOS = {
    "baseline_live": DigitalTwinScenario("baseline_live", "none", False),
    "outage_no_fallback": DigitalTwinScenario("outage_no_fallback", "none", True),
    "outage_replay_fixed": DigitalTwinScenario("outage_replay_fixed", "fixed", True),
    "outage_replay_kde": DigitalTwinScenario("outage_replay_kde", "kde", True),
    "outage_replay_wgan": DigitalTwinScenario("outage_replay_wgan", "wgan", True),
}


class CyclicDatasetReader:
    def __init__(self, dataset: DETRACDataset, source: str, sensor_id: str):
        self.dataset = dataset
        self.source = source
        self.sensor_id = sensor_id
        self.index = 0

    def next_sample(self, received_at: float, replay_delay: float = 0.0) -> Optional[dict]:
        if len(self.dataset) == 0:
            return None
        x, y = self.dataset[self.index % len(self.dataset)]
        self.index += 1
        return {
            "x": x,
            "y": y,
            "source": self.source,
            "sensor_id": self.sensor_id,
            "received_at": received_at,
            "replay_delay": replay_delay,
        }


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.samples: List[dict] = []

    def add_many(self, samples: Sequence[dict]) -> None:
        if not samples:
            return
        self.samples.extend(samples)
        overflow = len(self.samples) - self.max_size
        if overflow > 0:
            del self.samples[:overflow]

    def sample(self, count: int) -> List[dict]:
        if count <= 0 or not self.samples:
            return []
        if len(self.samples) <= count:
            return list(self.samples)
        return random.sample(self.samples, count)

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class SensorRoundResult:
    edge_id: str
    sensor_id: str
    outage_active: bool
    live_samples: int
    fallback_samples: int
    queue_fill_seconds: float
    queue_fill_complete: bool


@dataclass
class EdgeRoundResult:
    edge_id: str
    model_state: Dict[str, torch.Tensor]
    samples_received: int
    samples_trained: int
    live_samples: int
    fallback_samples: int
    train_loss_avg: float
    compute_time: float
    queue_fill_seconds_avg: float
    queue_fill_seconds_max: float
    queue_fill_complete_sensors: int
    replay_buffer_size: int
    sensor_results: List[SensorRoundResult]


class SensorTwin:
    def __init__(
        self,
        intersection: IntersectionSpec,
        live_reader: CyclicDatasetReader,
        fallback_readers: Sequence[CyclicDatasetReader],
        delay_sampler: DelaySampler,
        config: SimulationConfig,
        samples_per_sensor_per_round: int,
        collect_seconds: float,
    ):
        self.intersection = intersection
        self.live_reader = live_reader
        self.fallback_readers = list(fallback_readers)
        self.delay_sampler = delay_sampler
        self.config = config
        self.samples_per_sensor_per_round = samples_per_sensor_per_round
        self.collect_seconds = collect_seconds
        self.fallback_index = 0

    def is_outage(
        self,
        scenario: DigitalTwinScenario,
        round_number: int,
        outage_intersections: Sequence[str],
        outage_start_round: int,
        outage_end_round: int,
    ) -> bool:
        if not scenario.enable_outage:
            return False
        return (
            self.intersection.id in outage_intersections
            and outage_start_round <= round_number <= outage_end_round
        )

    def collect(
        self,
        scenario: DigitalTwinScenario,
        round_number: int,
        outage_intersections: Sequence[str],
        outage_start_round: int,
        outage_end_round: int,
    ) -> Tuple[List[dict], SensorRoundResult]:
        outage_active = self.is_outage(
            scenario,
            round_number,
            outage_intersections,
            outage_start_round,
            outage_end_round,
        )
        if outage_active and scenario.fallback_mode == "none":
            result = SensorRoundResult(
                edge_id=self.intersection.edge or "",
                sensor_id=self.intersection.id,
                outage_active=True,
                live_samples=0,
                fallback_samples=0,
                queue_fill_seconds=float("nan"),
                queue_fill_complete=False,
            )
            return [], result

        source = "live"
        strategy = "live"
        reader = self.live_reader
        if outage_active:
            source = f"fallback_{scenario.fallback_mode}"
            strategy = "fixed" if scenario.fallback_mode == "fixed" else scenario.fallback_mode
            if not self.fallback_readers:
                result = SensorRoundResult(
                    edge_id=self.intersection.edge or "",
                    sensor_id=self.intersection.id,
                    outage_active=True,
                    live_samples=0,
                    fallback_samples=0,
                    queue_fill_seconds=float("nan"),
                    queue_fill_complete=False,
                )
                return [], result
            reader = self.fallback_readers[self.fallback_index % len(self.fallback_readers)]

        elapsed = 0.0
        samples = []
        for _ in range(self.samples_per_sensor_per_round):
            delay = self.delay_sampler.sample(strategy) * self.config.time_scale
            if self.config.time_scale == 0:
                delay = 0.0
            if delay > 0 and elapsed + delay > self.collect_seconds:
                break
            elapsed += max(delay, 0.0)
            sample = reader.next_sample(received_at=elapsed, replay_delay=delay if outage_active else 0.0)
            if sample is None:
                break
            if outage_active:
                sample["source"] = source
            samples.append(sample)
            if outage_active:
                self.fallback_index += 1

        live_count = sum(1 for sample in samples if sample["source"] == "live")
        fallback_count = len(samples) - live_count
        result = SensorRoundResult(
            edge_id=self.intersection.edge or "",
            sensor_id=self.intersection.id,
            outage_active=outage_active,
            live_samples=live_count,
            fallback_samples=fallback_count,
            queue_fill_seconds=elapsed if samples else float("nan"),
            queue_fill_complete=len(samples) >= self.samples_per_sensor_per_round,
        )
        return samples, result


class EdgeNode:
    def __init__(
        self,
        edge_id: str,
        sensors: Sequence[SensorTwin],
        config: SimulationConfig,
        class_weights: torch.Tensor,
        replay_buffer_size: int,
        train_batches_per_round: int,
    ):
        self.edge_id = edge_id
        self.sensors = list(sensors)
        self.config = config
        self.class_weights = class_weights
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.train_batches_per_round = train_batches_per_round
        self.model = SimpleCNN().to(DEVICE)

    def train_round(
        self,
        global_state: Dict[str, torch.Tensor],
        scenario: DigitalTwinScenario,
        round_number: int,
        outage_intersections: Sequence[str],
        outage_start_round: int,
        outage_end_round: int,
    ) -> EdgeRoundResult:
        self.model.load_state_dict(global_state)
        sensor_results = []
        round_samples = []
        for sensor in self.sensors:
            samples, result = sensor.collect(
                scenario,
                round_number,
                outage_intersections,
                outage_start_round,
                outage_end_round,
            )
            round_samples.extend(samples)
            sensor_results.append(result)

        random.shuffle(round_samples)
        self.replay_buffer.add_many(round_samples)
        train_count = min(
            len(self.replay_buffer),
            self.config.batch_size * self.train_batches_per_round,
        )
        train_samples = self.replay_buffer.sample(train_count)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(DEVICE))
        compute_time = 0.0
        losses = []
        self.model.train()
        for start in range(0, len(train_samples), self.config.batch_size):
            batch = train_samples[start : start + self.config.batch_size]
            if not batch:
                continue
            x = torch.stack([sample["x"] for sample in batch]).to(DEVICE)
            y = torch.tensor([sample["y"] for sample in batch], dtype=torch.long).to(DEVICE)
            t0 = time.time()
            optimizer.zero_grad()
            loss = loss_fn(self.model(x), y)
            loss.backward()
            optimizer.step()
            compute_time += time.time() - t0
            losses.append(float(loss.item()))

        fill_times = [
            result.queue_fill_seconds
            for result in sensor_results
            if not np.isnan(result.queue_fill_seconds)
        ]
        live_count = sum(1 for sample in round_samples if sample["source"] == "live")
        fallback_count = len(round_samples) - live_count
        return EdgeRoundResult(
            edge_id=self.edge_id,
            model_state=copy.deepcopy(self.model.state_dict()),
            samples_received=len(round_samples),
            samples_trained=len(train_samples),
            live_samples=live_count,
            fallback_samples=fallback_count,
            train_loss_avg=float(np.mean(losses)) if losses else float("nan"),
            compute_time=compute_time,
            queue_fill_seconds_avg=float(np.mean(fill_times)) if fill_times else float("nan"),
            queue_fill_seconds_max=float(np.max(fill_times)) if fill_times else float("nan"),
            queue_fill_complete_sensors=sum(
                result.queue_fill_complete for result in sensor_results
            ),
            replay_buffer_size=len(self.replay_buffer),
            sensor_results=sensor_results,
        )


def load_manifest(path: str) -> DigitalTwinManifest:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    intersections = tuple(
        IntersectionSpec(
            id=item["id"],
            role=item["role"],
            edge=item.get("edge"),
            sequences=tuple(item["sequences"]),
        )
        for item in raw["intersections"]
    )
    return DigitalTwinManifest(
        edges=tuple(raw["edges"]),
        intersections=intersections,
        evaluation_sequences=tuple(raw.get("evaluation_sequences", [])),
    )


def sequence_paths(sequence_names: Iterable[str]) -> List[str]:
    return [os.path.join(DETRAC_IMAGES_DIR, name) for name in sequence_names]


def validate_manifest(manifest: DigitalTwinManifest) -> None:
    if len(manifest.intersections) != 24:
        raise ValueError(f"Digital twin manifest must define 24 intersections, found {len(manifest.intersections)}")
    ids = [intersection.id for intersection in manifest.intersections]
    if len(ids) != len(set(ids)):
        raise ValueError("Intersection IDs must be unique")

    live = [intersection for intersection in manifest.intersections if intersection.role == "live"]
    fallback = [
        intersection
        for intersection in manifest.intersections
        if intersection.role == "historical_fallback"
    ]
    if len(live) != 21:
        raise ValueError(f"Expected 21 live intersections, found {len(live)}")
    if len(fallback) != 3:
        raise ValueError(f"Expected 3 historical fallback intersections, found {len(fallback)}")

    live_by_edge = {edge: 0 for edge in manifest.edges}
    for intersection in live:
        if intersection.edge not in live_by_edge:
            raise ValueError(f"Live intersection {intersection.id} has unknown edge {intersection.edge}")
        live_by_edge[intersection.edge] += 1
    for edge, count in live_by_edge.items():
        if count != 7:
            raise ValueError(f"Expected edge {edge} to have 7 live intersections, found {count}")

    all_sequences = []
    for intersection in manifest.intersections:
        all_sequences.extend(intersection.sequences)
    all_sequences.extend(manifest.evaluation_sequences)
    if len(all_sequences) != len(set(all_sequences)):
        raise ValueError("Manifest live/fallback/evaluation sequences must be disjoint")

    annotated = {os.path.basename(path) for path in list_annotated_videos()}
    missing = [sequence for sequence in all_sequences if sequence not in annotated]
    if missing:
        raise ValueError(f"Manifest references sequences without annotations: {missing}")


def build_dataset(sequence_names: Sequence[str], transform, max_frames_per_video: int) -> DETRACDataset:
    return DETRACDataset(
        sequence_paths(sequence_names),
        DETRAC_ANNOT_DIR,
        transform=transform,
        max_frames_per_video=max_frames_per_video,
    )


def build_edges(
    manifest: DigitalTwinManifest,
    config: SimulationConfig,
    args,
    delay_sampler: DelaySampler,
) -> Tuple[List[EdgeNode], DETRACDataset]:
    train_transform, test_transform = build_transforms(config)
    fallback_readers = []
    fallback_datasets = []
    for intersection in manifest.intersections:
        if intersection.role != "historical_fallback":
            continue
        dataset = build_dataset(
            intersection.sequences,
            train_transform,
            config.max_frames_per_video,
        )
        fallback_datasets.append(dataset)
        fallback_readers.append(
            CyclicDatasetReader(dataset, "fallback", intersection.id)
        )

    live_datasets = []
    edge_sensors: Dict[str, List[SensorTwin]] = {edge: [] for edge in manifest.edges}
    for intersection in manifest.intersections:
        if intersection.role != "live":
            continue
        dataset = build_dataset(
            intersection.sequences,
            train_transform,
            config.max_frames_per_video,
        )
        live_datasets.append(dataset)
        live_reader = CyclicDatasetReader(dataset, "live", intersection.id)
        sensor = SensorTwin(
            intersection=intersection,
            live_reader=live_reader,
            fallback_readers=fallback_readers,
            delay_sampler=delay_sampler,
            config=config,
            samples_per_sensor_per_round=args.samples_per_sensor_per_round,
            collect_seconds=args.round_collect_seconds,
        )
        edge_sensors[intersection.edge or ""].append(sensor)

    class_weights = compute_class_weights(live_datasets + fallback_datasets)
    edges = [
        EdgeNode(
            edge_id=edge,
            sensors=edge_sensors[edge],
            config=config,
            class_weights=class_weights,
            replay_buffer_size=args.replay_buffer_size,
            train_batches_per_round=args.train_batches_per_round,
        )
        for edge in manifest.edges
    ]

    test_dataset = build_dataset(
        manifest.evaluation_sequences,
        test_transform,
        config.max_frames_per_video,
    )
    return edges, test_dataset


def evaluate_with_balance(model: SimpleCNN, loader: DataLoader) -> Tuple[float, float, List[float]]:
    frame_acc, per_class = evaluate(model, loader)
    valid = [value for value in per_class if not np.isnan(value)]
    balanced = float(np.mean(valid)) if valid else 0.0
    return frame_acc, balanced, per_class


def parse_intersections(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def selected_scenarios(value: str) -> List[DigitalTwinScenario]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Options: {', '.join(SCENARIOS)}")
    return [SCENARIOS[name] for name in names]


def write_rows(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_scenario(
    scenario: DigitalTwinScenario,
    manifest: DigitalTwinManifest,
    config: SimulationConfig,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(config.seed)
    delay_sampler = DelaySampler(config)
    edges, test_dataset = build_edges(manifest, config, args, delay_sampler)
    test_loader = DataLoader(test_dataset, batch_size=128)
    global_model = SimpleCNN().to(DEVICE)
    global_rows = []
    edge_rows = []
    sensor_rows = []
    scenario_start = time.time()
    outage_intersections = parse_intersections(args.outage_intersections)

    print(f"\nRunning digital twin scenario: {scenario.name}")
    print(
        "Outage intersections: "
        f"{list(outage_intersections)} rounds {args.outage_start_round}-{args.outage_end_round}"
    )

    for round_number in range(1, config.rounds + 1):
        round_start = time.time()
        global_snapshot = copy.deepcopy(global_model.state_dict())
        with ResourceSampler() as resources:
            edge_results = [
                edge.train_round(
                    global_snapshot,
                    scenario,
                    round_number,
                    outage_intersections,
                    args.outage_start_round,
                    args.outage_end_round,
                )
                for edge in edges
            ]
            weights = [result.samples_trained for result in edge_results]
            global_model = aggregate_models([result.model_state for result in edge_results], weights)
            global_state = copy.deepcopy(global_model.state_dict())
            for edge in edges:
                edge.model.load_state_dict(global_state)

        resource = resources.summary()
        frame_acc, balanced_acc, per_class = evaluate_with_balance(global_model, test_loader)
        round_wall_time = time.time() - round_start
        samples_received = sum(result.samples_received for result in edge_results)
        samples_trained = sum(result.samples_trained for result in edge_results)
        live_samples = sum(result.live_samples for result in edge_results)
        fallback_samples = sum(result.fallback_samples for result in edge_results)
        fill_times = [
            result.queue_fill_seconds_max
            for result in edge_results
            if not np.isnan(result.queue_fill_seconds_max)
        ]
        outage_active = any(
            sensor.outage_active
            for result in edge_results
            for sensor in result.sensor_results
        )
        global_row = {
            "scenario": scenario.name,
            "round": round_number,
            "outage_active": outage_active,
            "frame_accuracy": frame_acc,
            "balanced_accuracy": balanced_acc,
            "samples_received": samples_received,
            "samples_trained": samples_trained,
            "live_samples": live_samples,
            "fallback_samples": fallback_samples,
            "round_wall_time": round_wall_time,
            "images_received_per_second": samples_received / round_wall_time if round_wall_time > 0 else 0.0,
            "samples_trained_per_second": samples_trained / round_wall_time if round_wall_time > 0 else 0.0,
            "queue_fill_seconds_max": float(np.max(fill_times)) if fill_times else float("nan"),
            "queue_fill_complete_edges": sum(
                result.queue_fill_complete_sensors == len(result.sensor_results)
                for result in edge_results
            ),
            "effective_training_duration": time.time() - scenario_start,
            **resource,
        }
        for cls in range(NUM_CLASSES):
            global_row[f"acc_{CLASS_NAMES[cls]}"] = per_class[cls]
        global_rows.append(global_row)

        for result in edge_results:
            edge_rows.append(
                {
                    "scenario": scenario.name,
                    "round": round_number,
                    "edge_id": result.edge_id,
                    "samples_received": result.samples_received,
                    "samples_trained": result.samples_trained,
                    "live_samples": result.live_samples,
                    "fallback_samples": result.fallback_samples,
                    "train_loss_avg": result.train_loss_avg,
                    "compute_time": result.compute_time,
                    "queue_fill_seconds_avg": result.queue_fill_seconds_avg,
                    "queue_fill_seconds_max": result.queue_fill_seconds_max,
                    "queue_fill_complete_sensors": result.queue_fill_complete_sensors,
                    "sensor_count": len(result.sensor_results),
                    "replay_buffer_size": result.replay_buffer_size,
                }
            )
            for sensor in result.sensor_results:
                sensor_rows.append(
                    {
                        "scenario": scenario.name,
                        "round": round_number,
                        "edge_id": result.edge_id,
                        "sensor_id": sensor.sensor_id,
                        "outage_active": sensor.outage_active,
                        "live_samples": sensor.live_samples,
                        "fallback_samples": sensor.fallback_samples,
                        "samples_received": sensor.live_samples + sensor.fallback_samples,
                        "queue_fill_seconds": sensor.queue_fill_seconds,
                        "queue_fill_complete": sensor.queue_fill_complete,
                    }
                )

        print(
            f"Round {round_number}: acc={frame_acc*100:.2f}% "
            f"balanced={balanced_acc*100:.2f}% "
            f"received={samples_received} trained={samples_trained} "
            f"live={live_samples} fallback={fallback_samples} "
            f"queue_fill_max={global_row['queue_fill_seconds_max']:.2f}s "
            f"cpu_avg={global_row['cpu_avg']:.1f}%"
        )

    return pd.DataFrame(global_rows), pd.DataFrame(edge_rows), pd.DataFrame(sensor_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fully simulated v3 digital twin with sensors, edge nodes, outages, fallback, and federated learning."
    )
    parser.add_argument(
        "--manifest",
        default=os.path.join(BASE_DIR, "digital_twin_manifest.json"),
    )
    parser.add_argument(
        "--scenarios",
        default="baseline_live,outage_no_fallback,outage_replay_fixed,outage_replay_kde,outage_replay_wgan",
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-batches-per-round", type=int, default=4)
    parser.add_argument("--samples-per-sensor-per-round", type=int, default=16)
    parser.add_argument("--round-collect-seconds", type=float, default=2.0)
    parser.add_argument("--replay-buffer-size", type=int, default=10000)
    parser.add_argument("--max-frames-per-video", type=int, default=50)
    parser.add_argument("--outage-intersections", default="intersection_00")
    parser.add_argument("--outage-start-round", type=int, default=20)
    parser.add_argument("--outage-end-round", type=int, default=60)
    parser.add_argument("--fixed-replay-delay", type=float, default=0.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    scenarios = selected_scenarios(args.scenarios)
    config = SimulationConfig(
        rounds=args.rounds,
        batch_size=args.batch_size,
        max_frames_per_video=args.max_frames_per_video,
        fixed_replay_delay=args.fixed_replay_delay,
        time_scale=args.time_scale,
        seed=args.seed,
    )
    os.makedirs(DIGITAL_TWIN_OUTPUT_DIR, exist_ok=True)

    for scenario in scenarios:
        global_df, edge_df, sensor_df = run_scenario(scenario, manifest, config, args)
        global_path = os.path.join(DIGITAL_TWIN_OUTPUT_DIR, f"global_metrics_{scenario.name}.csv")
        edge_path = os.path.join(DIGITAL_TWIN_OUTPUT_DIR, f"edge_metrics_{scenario.name}.csv")
        sensor_path = os.path.join(DIGITAL_TWIN_OUTPUT_DIR, f"sensor_metrics_{scenario.name}.csv")
        global_df.to_csv(global_path, index=False)
        edge_df.to_csv(edge_path, index=False)
        sensor_df.to_csv(sensor_path, index=False)
        print(f"Saved {global_path}")
        print(f"Saved {edge_path}")
        print(f"Saved {sensor_path}")


if __name__ == "__main__":
    main()
