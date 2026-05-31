# =========================================================
# INSTALL (run once)
# =========================================================
# !pip install torch torchvision matplotlib pandas pillow

# =========================================================
# IMPORTS
# =========================================================
import copy
import time
import random
import os
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# =========================================================
# CONFIG
# =========================================================
NUM_AGENTS = 3
VIDEOS_PER_AGENT = 10       # each agent trains on 10 MVI sequences
TEST_VIDEOS = 10            # held-out videos for evaluation
MAX_FRAMES_PER_VIDEO = 50   # sample up to N frames per sequence (speed vs. fidelity)
ROUNDS = 100
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3        # Adam with a lower LR converges more stably than SGD 0.01
IMG_SIZE = 64               # crop resize target

ARCHITECTURE = "peer_to_peer"
DELAY_MODEL = "wgan"   # Options: "kde" or "wgan"

UPLOAD_MEAN = 0.3
UPLOAD_STD = 0.1

DOWNLOAD_MEAN = 0.2
DOWNLOAD_STD = 0.05

# Maximum concurrent uploads to the aggregator.
# Set to 1 to model a single shared uplink (realistic contention).
# Increase to model parallel channels.
UPLOAD_CONCURRENCY = 1
DOWNLOAD_CONCURRENCY = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETRAC_IMAGES_DIR = os.path.join(BASE_DIR, "..", "DETRAC-Images", "DETRAC-Images")
DETRAC_ANNOT_DIR  = os.path.join(BASE_DIR, "..", "DETRAC-Train-Annotations-XML",
                                 "DETRAC-Train-Annotations-XML")

# =========================================================
# VEHICLE CLASS MAP  (car / van / bus / others)
# =========================================================
VEHICLE_CLASSES = {"car": 0, "van": 1, "bus": 2, "others": 3}
NUM_CLASSES = len(VEHICLE_CLASSES)

# =========================================================
# LOAD SYNTHETIC DELAYS
# =========================================================
delay_file = os.path.join(BASE_DIR, f"synthetic_interarrival_{DELAY_MODEL}.csv")

synthetic_delays = pd.read_csv(
    delay_file,
    header=None
).values.flatten()

# Remove invalid values
synthetic_delays = synthetic_delays[
    synthetic_delays > 1e-6
]

print(f"Loaded {len(synthetic_delays)} synthetic delays from {delay_file}")

# =========================================================
# SHARED CHANNEL SEMAPHORES
# Agents compete for a shared uplink/downlink to the aggregator.
# The semaphore serialises concurrent uploads/downloads, introducing
# realistic queuing delay when multiple agents finish at the same time.
# =========================================================
upload_semaphore   = threading.Semaphore(UPLOAD_CONCURRENCY)
download_semaphore = threading.Semaphore(DOWNLOAD_CONCURRENCY)

# =========================================================
# DELAY SAMPLER
# =========================================================
def sample_delay():
    """
    Randomly sample one synthesized delay
    """
    d = random.choice(synthetic_delays)

    # numerical safety
    return max(float(d), 1e-6)

# =========================================================
# DETRAC DATASET
# =========================================================
class DETRACDataset(Dataset):
    """
    Crops individual vehicle bounding boxes from DETRAC frames and
    returns (image_tensor, vehicle_class_index) pairs.

    vehicle_type → label: car=0, van=1, bus=2, others=3

    Each sample internally stores (img_path, box, label, video_id, track_id)
    so that temporal aggregation and per-condition evaluation can group
    samples by vehicle track or by scene metadata without touching the
    standard (x, y) DataLoader interface.

    Entire video sequences are assigned to either training agents or the
    test set at the caller level — this class never mixes them.  This
    enforces video-level separation: the model cannot benefit from
    near-duplicate frames belonging to the same recording appearing in
    both splits.
    """

    def __init__(self, video_dirs, annotation_dir, transform=None,
                 max_frames_per_video=MAX_FRAMES_PER_VIDEO):

        # Each entry: (img_path, (left,top,w,h), label, video_id, track_id)
        self.samples = []
        # video_id -> {"weather": str, "camera_state": str}
        self.video_metadata = {}
        self.transform = transform

        for video_dir in video_dirs:
            seq_name = os.path.basename(video_dir)
            xml_path = os.path.join(annotation_dir, f"{seq_name}.xml")

            if not os.path.exists(xml_path):
                print(f"Missing annotation: {xml_path}")
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Parse scene-level metadata for domain-shift evaluation
            seq_attr = root.find("sequence_attribute")
            self.video_metadata[seq_name] = {
                "weather":      seq_attr.get("sence_weather", "unknown") if seq_attr is not None else "unknown",
                "camera_state": seq_attr.get("camera_state",  "unknown") if seq_attr is not None else "unknown",
            }

            frames = root.findall("frame")

            # Evenly sub-sample frames to cap dataset size per video
            step    = max(1, len(frames) // max_frames_per_video)
            sampled = frames[::step][:max_frames_per_video]

            for frame_elem in sampled:
                frame_num = int(frame_elem.get("num"))
                img_path  = os.path.join(video_dir, f"img{frame_num:05d}.jpg")

                if not os.path.exists(img_path):
                    continue

                for target in frame_elem.findall(".//target"):
                    box_elem  = target.find("box")
                    attr_elem = target.find("attribute")

                    if box_elem is None or attr_elem is None:
                        continue

                    track_id = int(target.get("id", -1))
                    vtype    = attr_elem.get("vehicle_type", "others")
                    label    = VEHICLE_CLASSES.get(vtype, 3)

                    left   = float(box_elem.get("left"))
                    top    = float(box_elem.get("top"))
                    width  = float(box_elem.get("width"))
                    height = float(box_elem.get("height"))

                    self.samples.append(
                        (img_path, (left, top, width, height), label, seq_name, track_id)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (left, top, w, h), label, _video_id, _track_id = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size

        x1 = max(0, int(left))
        y1 = max(0, int(top))
        x2 = min(iw, int(left + w))
        y2 = min(ih, int(top + h))

        crop = img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else img

        if self.transform:
            crop = self.transform(crop)

        return crop, label

    # ------------------------------------------------------------------
    # GROUPING HELPERS  (used by evaluation functions, not DataLoader)
    # ------------------------------------------------------------------

    def get_track_index_groups(self):
        """
        Returns {(video_id, track_id): [sample_indices]}.

        Used for temporal aggregation: each vehicle track is observed
        across multiple frames; majority-voting over the track's predictions
        reflects how a real traffic system would classify a vehicle.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, (_, _, _, video_id, track_id) in enumerate(self.samples):
            groups[(video_id, track_id)].append(idx)
        return dict(groups)

    def get_condition_index_groups(self, condition_key):
        """
        Returns {condition_value: [sample_indices]}.

        condition_key is "weather" or "camera_state".
        Used for domain-shift evaluation: accuracy is reported separately
        for each environmental condition (sunny / night / rainy / cloudy,
        or stable / unstable camera).
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, (_, _, _, video_id, _) in enumerate(self.samples):
            cond = self.video_metadata.get(video_id, {}).get(condition_key, "unknown")
            groups[cond].append(idx)
        return dict(groups)

# =========================================================
# CNN MODEL  (64×64 input → 4 vehicle classes)
#
# Improvements over the original 2-layer network:
#   - Three conv blocks instead of two, giving a deeper feature hierarchy
#   - BatchNorm after each conv: normalises activations across the batch,
#     stabilising training and reducing sensitivity to initialisation
#   - Dropout before the final classifier: regularises the fully-connected
#     head and reduces over-fitting to the limited training videos
# =========================================================
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            # Block 1 — 64 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2 — 32 → 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3 — 16 → 8
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

# =========================================================
# EDGE AGENT
# =========================================================
class EdgeAgent:
    """
    Models a single edge node in the distributed system.

    Timing is separated into three conceptually distinct phases per round:

      1. Ingestion  — interarrival delays drawn from the synthetic distribution
                      gate when each batch of frames *arrives* at the node.
                      This is network/sensor time, not compute time.

      2. Compute    — forward + backward pass; pure CPU/GPU time with no
                      artificial sleep, so it reflects actual workload.

      3. Upload     — model update sent to the aggregator through a *shared*
                      semaphore-guarded channel.  Agents that finish training
                      concurrently must queue for the uplink, introducing
                      realistic contention and back-pressure.

    Agents run in parallel (ThreadPoolExecutor in the simulation loop), so
    round wall-clock time is determined by the slowest agent (the straggler),
    not the sum of all agents — matching how real distributed systems behave.
    """

    def __init__(self, agent_id, dataset, class_weights=None):

        self.id           = agent_id
        self.data         = dataset
        self.model        = SimpleCNN().to(DEVICE)
        # Inverse-frequency weights passed in at construction time.
        # Stored as a CPU tensor; moved to DEVICE inside train().
        self.class_weights = class_weights

        # Per-round time breakdowns (one entry per round)
        self.ingestion_wait_times  = []   # total data-arrival wait
        self.compute_times         = []   # total forward+backward time
        self.upload_queue_waits    = []   # time waiting for shared uplink
        self.upload_transfer_times = []   # actual transfer latency
        self.download_times        = []   # download (queue wait + transfer)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _upload_transfer_latency(self):
        """Sample one-way upload transfer time based on architecture."""
        scale = {"centralized": 1.0, "regional": 0.6, "peer_to_peer": 0.4}
        factor = scale.get(ARCHITECTURE, 1.0)
        return max(np.random.normal(UPLOAD_MEAN * factor, UPLOAD_STD), 0)

    # ------------------------------------------------------------------
    # TRAIN (called concurrently from ThreadPoolExecutor)
    # ------------------------------------------------------------------
    def train(self, global_weights):
        """
        Load global weights, train locally, then upload the update.

        global_weights must already be a deepcopy so threads do not share
        mutable tensor state.
        """
        self.model.load_state_dict(global_weights)

        # Adam adapts the learning rate per parameter, which converges more
        # stably than SGD across heterogeneous federated agents whose local
        # data distributions can differ significantly between videos.
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # Weighted loss: rare classes (bus, others) get higher weight so
        # that the model does not trivially optimise for majority classes.
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(DEVICE) if self.class_weights is not None else None
        )
        loader    = DataLoader(self.data, batch_size=BATCH_SIZE, shuffle=True)

        self.model.train()

        ingestion_total = 0.0
        compute_total   = 0.0

        # -----------------------------------
        # PHASE 1 + 2: INGESTION + COMPUTE
        # -----------------------------------
        for _ in range(LOCAL_EPOCHS):

            for x, y in loader:

                # Interarrival delay: models the time between consecutive
                # batches of frames arriving at the edge node over the
                # network.  The node cannot start processing until the
                # data has arrived, so the sleep precedes the compute step.
                arrival_delay = sample_delay()
                time.sleep(arrival_delay)
                ingestion_total += arrival_delay

                # Compute: forward + backward — no sleep here so that
                # actual GPU/CPU time is measured independently.
                t0 = time.time()

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                optimizer.zero_grad()
                preds = self.model(x)
                loss  = loss_fn(preds, y)
                loss.backward()
                optimizer.step()

                compute_total += time.time() - t0

        self.ingestion_wait_times.append(ingestion_total)
        self.compute_times.append(compute_total)

        # -----------------------------------
        # PHASE 3: UPLOAD (contended uplink)
        # -----------------------------------
        # Wait for the shared uplink to become free.  Multiple agents
        # finishing at the same time will queue here, producing realistic
        # queuing delays that affect round completion time.
        queue_start = time.time()
        upload_semaphore.acquire()
        queue_wait = time.time() - queue_start

        try:
            transfer = self._upload_transfer_latency()
            time.sleep(transfer)
        finally:
            upload_semaphore.release()

        self.upload_queue_waits.append(queue_wait)
        self.upload_transfer_times.append(transfer)

        return self.model.state_dict()

    # ------------------------------------------------------------------
    # RECEIVE GLOBAL MODEL (called concurrently from ThreadPoolExecutor)
    # ------------------------------------------------------------------
    def receive_model(self, global_weights):
        """
        Download the aggregated model through the shared downlink.

        global_weights must already be a deepcopy so threads do not share
        mutable tensor state.
        """
        queue_start = time.time()
        download_semaphore.acquire()
        queue_wait = time.time() - queue_start

        try:
            transfer = max(np.random.normal(DOWNLOAD_MEAN, DOWNLOAD_STD), 0)
            time.sleep(transfer)
        finally:
            download_semaphore.release()

        self.download_times.append(queue_wait + transfer)
        self.model.load_state_dict(global_weights)

# =========================================================
# FEDERATED AGGREGATION  (dataset-size weighted)
# =========================================================
def aggregate_models(models, dataset_sizes):
    """
    Weighted FedAvg: each agent's update is weighted by its dataset size.

    Uniform averaging gives equal weight to all agents regardless of how
    much data they trained on, which can skew the global model toward
    agents with small or unrepresentative datasets.  Weighting by size
    ensures that agents with larger, more representative video sets
    contribute proportionally more to the aggregated model.
    """
    global_model = SimpleCNN().to(DEVICE)

    total = sum(dataset_sizes)

    agg = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_model.state_dict().items()
    }

    for m, sz in zip(models, dataset_sizes):
        weight = sz / total
        for k in agg:
            agg[k] += weight * m[k].float()

    # Restore original dtypes before loading (e.g. BatchNorm's
    # num_batches_tracked is a Long and must stay that way).
    ref = global_model.state_dict()
    agg = {k: agg[k].to(ref[k].dtype) for k in agg}

    global_model.load_state_dict(agg)

    return global_model

# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, loader):
    """
    Frame-level accuracy and per-class accuracy.

    Returns (overall_accuracy, [acc_per_class]).
    Reporting per-class accuracy prevents class imbalance from masking
    poor minority-class performance behind a high aggregate number.
    """
    model.eval()

    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x).argmax(1)

            for c in range(NUM_CLASSES):
                mask = (y == c)
                class_correct[c] += (preds[mask] == c).sum().item()
                class_total[c]   += mask.sum().item()

    total   = sum(class_total)
    correct = sum(class_correct)

    frame_acc = correct / total if total > 0 else 0.0
    per_class = [
        class_correct[c] / class_total[c] if class_total[c] > 0 else float("nan")
        for c in range(NUM_CLASSES)
    ]

    return frame_acc, per_class


def evaluate_per_track(model, dataset):
    """
    Temporal aggregation via majority vote over each vehicle track.

    A track is one vehicle observed across multiple frames (same target ID
    within a video).  The model's prediction for each frame crop is recorded;
    the majority vote is the final prediction for that vehicle.

    This reflects how a real traffic perception system operates: decisions
    about a vehicle are made over time, not from a single instant.
    It also reveals whether the model is temporally consistent — a model
    that is right on average but inconsistent frame-to-frame would show
    a large gap between frame accuracy and track accuracy.
    """
    model.eval()

    track_groups = dataset.get_track_index_groups()
    correct = 0
    total   = 0

    with torch.no_grad():

        for (_video_id, _track_id), indices in track_groups.items():

            # All crops of the same track share one true label
            true_label = dataset.samples[indices[0]][2]

            # Batch inference over all frames of this track
            crops = torch.stack([dataset[i][0] for i in indices]).to(DEVICE)
            preds = model(crops).argmax(1)

            vote = torch.mode(preds).values.item()
            correct += int(vote == true_label)
            total   += 1

    return correct / total if total > 0 else 0.0


def evaluate_per_condition(model, dataset):
    """
    Frame-level accuracy broken down by scene condition.

    Reports accuracy separately for each weather condition (sunny, night,
    rainy, cloudy) and camera stability (stable, unstable).

    A model that achieves high overall accuracy but degrades sharply on
    night or rainy sequences is not robust for real-world deployment.
    This breakdown makes domain shift visible rather than averaged away.
    """
    model.eval()

    results = {}

    for condition_key in ("weather", "camera_state"):

        cond_groups = dataset.get_condition_index_groups(condition_key)
        cond_acc    = {}

        with torch.no_grad():

            for cond_value, indices in cond_groups.items():

                correct = 0
                total   = 0

                for batch_start in range(0, len(indices), 128):

                    batch = [dataset[i] for i in indices[batch_start:batch_start + 128]]
                    x = torch.stack([s[0] for s in batch]).to(DEVICE)
                    y = torch.tensor([s[1] for s in batch]).to(DEVICE)

                    preds = model(x).argmax(1)
                    correct += (preds == y).sum().item()
                    total   += y.size(0)

                cond_acc[cond_value] = correct / total if total > 0 else float("nan")

        results[condition_key] = cond_acc

    return results

# =========================================================
# DATASET  —  DETRAC vehicle crops
# =========================================================
# Training transform includes augmentation to improve generalisation
# to different lighting, camera angles, and weather conditions.
# Test transform applies only geometric normalisation (no augmentation)
# so evaluation is deterministic and comparable across rounds.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Collect all MVI folders that have a matching annotation XML
all_video_dirs = sorted([
    os.path.join(DETRAC_IMAGES_DIR, d)
    for d in os.listdir(DETRAC_IMAGES_DIR)
    if os.path.isdir(os.path.join(DETRAC_IMAGES_DIR, d))
    and os.path.exists(os.path.join(DETRAC_ANNOT_DIR, d + ".xml"))
])

total_needed = NUM_AGENTS * VIDEOS_PER_AGENT + TEST_VIDEOS
if len(all_video_dirs) < total_needed:
    raise RuntimeError(
        f"Not enough annotated videos: need {total_needed}, found {len(all_video_dirs)}"
    )

# Shuffle once for reproducibility, then carve out splits
rng = np.random.default_rng(42)
shuffled = [all_video_dirs[i] for i in rng.permutation(len(all_video_dirs))]

agent_video_splits = [
    shuffled[i * VIDEOS_PER_AGENT:(i + 1) * VIDEOS_PER_AGENT]
    for i in range(NUM_AGENTS)
]
test_video_dirs = shuffled[NUM_AGENTS * VIDEOS_PER_AGENT:
                           NUM_AGENTS * VIDEOS_PER_AGENT + TEST_VIDEOS]

print("Building agent datasets (parsing XML annotations & indexing crops)...")
agent_datasets = []
for i, video_dirs in enumerate(agent_video_splits):
    ds = DETRACDataset(video_dirs, DETRAC_ANNOT_DIR, transform=train_transform)
    print(f"  Agent {i}: {len(video_dirs)} videos → {len(ds)} crop samples")
    agent_datasets.append(ds)

print("Building test dataset...")
test_dataset = DETRACDataset(test_video_dirs, DETRAC_ANNOT_DIR, transform=test_transform)
print(f"  Test set: {len(test_video_dirs)} videos → {len(test_dataset)} crop samples")

test_loader = DataLoader(test_dataset, batch_size=128)

# =========================================================
# CLASS WEIGHTS  (address class imbalance)
# Count label frequencies across all training data and compute
# inverse-frequency weights.  This prevents the model from trivially
# collapsing to predict "car" (the dominant class) and ignoring buses
# and vans, which are the minority but equally important in traffic systems.
# =========================================================
label_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
for ds in agent_datasets:
    for sample in ds.samples:
        label_counts[sample[2]] += 1

class_weights = torch.tensor(
    label_counts.sum() / (NUM_CLASSES * label_counts + 1e-6),
    dtype=torch.float32
)

CLASS_NAMES = {v: k for k, v in VEHICLE_CLASSES.items()}
print("\nClass distribution across all training agents:")
for c in range(NUM_CLASSES):
    print(f"  {CLASS_NAMES[c]:8s}: {int(label_counts[c]):6d} samples  "
          f"(weight = {class_weights[c]:.3f})")

# =========================================================
# CREATE AGENTS
# =========================================================
agents = [
    EdgeAgent(i, agent_datasets[i], class_weights=class_weights)
    for i in range(NUM_AGENTS)
]

global_model = SimpleCNN().to(DEVICE)

# =========================================================
# METRICS
# =========================================================
round_accuracies         = []
round_per_class_acc      = {CLASS_NAMES[c]: [] for c in range(NUM_CLASSES)}
round_track_acc          = []   # NaN except every TRACK_EVAL_INTERVAL rounds
round_wall_times         = []
round_mean_ingestion     = []
round_mean_compute       = []
round_mean_upload_queue  = []
round_mean_upload_xfer   = []
round_mean_download      = []

TRACK_EVAL_INTERVAL = 10   # per-track + per-condition eval frequency

# =========================================================
# RUN SIMULATION
# =========================================================
print(f"\nRunning Simulation")
print(f"Architecture:   {ARCHITECTURE}")
print(f"Delay Model:    {DELAY_MODEL.upper()}")
print(f"Upload slots:   {UPLOAD_CONCURRENCY}  (shared uplink concurrency)")
print(f"Download slots: {DOWNLOAD_CONCURRENCY}  (shared downlink concurrency)")

simulation_start = time.time()

for r in range(ROUNDS):

    print(f"\nRound {r+1}")

    round_start = time.time()

    # -----------------------------------
    # LOCAL TRAINING  (all agents run concurrently)
    # Deepcopy ensures each thread gets its own tensor state — no
    # shared-mutable-state race conditions between threads.
    # -----------------------------------
    global_snapshot = copy.deepcopy(global_model.state_dict())

    with ThreadPoolExecutor(max_workers=NUM_AGENTS) as executor:
        futures = [
            executor.submit(agent.train, copy.deepcopy(global_snapshot))
            for agent in agents
        ]
        updates = [f.result() for f in futures]

    # Wall-clock round time is set by the slowest (straggler) agent,
    # which is how real federated systems behave.
    round_wall_time = time.time() - round_start

    # -----------------------------------
    # FEDERATED AGGREGATION
    # -----------------------------------
    global_model = aggregate_models(updates, [len(a.data) for a in agents])

    # -----------------------------------
    # DISTRIBUTE GLOBAL MODEL  (all agents download concurrently)
    # -----------------------------------
    agg_snapshot = copy.deepcopy(global_model.state_dict())

    with ThreadPoolExecutor(max_workers=NUM_AGENTS) as executor:
        futures = [
            executor.submit(agent.receive_model, copy.deepcopy(agg_snapshot))
            for agent in agents
        ]
        for f in futures:
            f.result()

    # -----------------------------------
    # EVALUATE
    # -----------------------------------
    frame_acc, per_class = evaluate(global_model, test_loader)

    round_accuracies.append(frame_acc)
    for c in range(NUM_CLASSES):
        round_per_class_acc[CLASS_NAMES[c]].append(per_class[c])

    round_wall_times.append(round_wall_time)
    round_mean_ingestion.append(
        np.mean([a.ingestion_wait_times[-1] for a in agents])
    )
    round_mean_compute.append(
        np.mean([a.compute_times[-1] for a in agents])
    )
    round_mean_upload_queue.append(
        np.mean([a.upload_queue_waits[-1] for a in agents])
    )
    round_mean_upload_xfer.append(
        np.mean([a.upload_transfer_times[-1] for a in agents])
    )
    round_mean_download.append(
        np.mean([a.download_times[-1] for a in agents])
    )

    # Per-track (temporal aggregation) and per-condition (domain shift)
    # are computed every TRACK_EVAL_INTERVAL rounds — they are heavier
    # because they iterate over the test dataset sample-by-sample.
    if (r + 1) % TRACK_EVAL_INTERVAL == 0 or r == ROUNDS - 1:
        track_acc       = evaluate_per_track(global_model, test_dataset)
        condition_results = evaluate_per_condition(global_model, test_dataset)
        round_track_acc.append((r + 1, track_acc))
        print(f"  [Track eval]  Track-level accuracy: {track_acc*100:.2f}%")
        for ckey, cdict in condition_results.items():
            cline = "  ".join(f"{k}={v*100:.1f}%" for k, v in sorted(cdict.items()))
            print(f"  [Domain: {ckey}]  {cline}")
    else:
        pass  # skip heavy evals this round

    print(f"  Frame accuracy:      {frame_acc*100:.2f}%")
    per_cls_str = "  ".join(
        f"{CLASS_NAMES[c]}={per_class[c]*100:.1f}%" if not np.isnan(per_class[c])
        else f"{CLASS_NAMES[c]}=n/a"
        for c in range(NUM_CLASSES)
    )
    print(f"  Per-class:           {per_cls_str}")
    print(f"  Round wall time:     {round_wall_time:.3f}s")
    print(f"  Avg ingestion wait:  {round_mean_ingestion[-1]:.3f}s")
    print(f"  Avg compute time:    {round_mean_compute[-1]:.3f}s")
    print(f"  Avg upload queue:    {round_mean_upload_queue[-1]:.3f}s")
    print(f"  Avg upload transfer: {round_mean_upload_xfer[-1]:.3f}s")

# =========================================================
# FINAL REPORT
# =========================================================
total_time = time.time() - simulation_start

print(f"\nSimulation Complete")
print(f"Total Time: {total_time:.2f}s")

# Final per-track and per-condition evaluation
final_track_acc       = evaluate_per_track(global_model, test_dataset)
final_condition_results = evaluate_per_condition(global_model, test_dataset)

print(f"\nFinal Track-Level Accuracy (majority vote): {final_track_acc*100:.2f}%")
for ckey, cdict in final_condition_results.items():
    print(f"\nDomain shift — {ckey}:")
    for cval, cacc in sorted(cdict.items()):
        bar = "█" * int(cacc * 20) if not np.isnan(cacc) else ""
        print(f"  {cval:12s}: {cacc*100:5.1f}%  {bar}")

# =========================================================
# SAVE METRICS
# =========================================================
metrics_df = pd.DataFrame({
    "round":               np.arange(1, ROUNDS + 1),
    "frame_accuracy":      round_accuracies,
    "round_wall_time":     round_wall_times,
    "mean_ingestion_wait": round_mean_ingestion,
    "mean_compute_time":   round_mean_compute,
    "mean_upload_queue":   round_mean_upload_queue,
    "mean_upload_xfer":    round_mean_upload_xfer,
    "mean_download_time":  round_mean_download,
    **{f"acc_{cls}": round_per_class_acc[cls] for cls in round_per_class_acc},
})

metrics_df.to_csv(
    f"simulation_metrics_{DELAY_MODEL}.csv",
    index=False
)

# =========================================================
# PLOT: FRAME ACCURACY
# =========================================================
plt.figure(figsize=(7, 5))
plt.plot(round_accuracies, marker='o')
plt.title(f"Global Frame Accuracy per Round ({DELAY_MODEL.upper()} Delays)")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./visualizations/accuracy_{DELAY_MODEL}.png", dpi=300)
plt.close()

# =========================================================
# PLOT: PER-CLASS ACCURACY
# Visualises whether minority classes (bus, others) are improving
# alongside the dominant class (car).
# =========================================================
plt.figure(figsize=(9, 5))
rounds = np.arange(1, ROUNDS + 1)
for cls_name in round_per_class_acc:
    vals = round_per_class_acc[cls_name]
    plt.plot(rounds, vals, marker='.', label=cls_name)
plt.title(f"Per-Class Accuracy per Round ({DELAY_MODEL.upper()} Delays)")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./visualizations/per_class_accuracy_{DELAY_MODEL}.png", dpi=300)
plt.close()

# =========================================================
# PLOT: FINAL DOMAIN-SHIFT BAR CHARTS
# =========================================================
for ckey, cdict in final_condition_results.items():
    labels = sorted(cdict.keys())
    values = [cdict[l] * 100 if not np.isnan(cdict[l]) else 0 for l in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title(f"Final Accuracy by {ckey.replace('_', ' ').title()} ({DELAY_MODEL.upper()})")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"./visualizations/domain_shift_{ckey}_{DELAY_MODEL}.png", dpi=300)
    plt.close()

# =========================================================
# PLOT: ROUND WALL TIME
# =========================================================
plt.figure(figsize=(7, 5))
plt.plot(round_wall_times, marker='o')
plt.title(f"Round Wall-Clock Time ({DELAY_MODEL.upper()} Delays)")
plt.xlabel("Round")
plt.ylabel("Time (s)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./visualizations/round_time_{DELAY_MODEL}.png", dpi=300)
plt.close()

# =========================================================
# PLOT: TIME BREAKDOWN (stacked area)
# Shows how ingestion wait, compute, upload queue, and
# upload transfer each contribute to total round time,
# making KDE vs WGAN differences interpretable.
# =========================================================
rounds = np.arange(1, ROUNDS + 1)

plt.figure(figsize=(9, 5))
plt.stackplot(
    rounds,
    round_mean_ingestion,
    round_mean_compute,
    round_mean_upload_queue,
    round_mean_upload_xfer,
    labels=["Ingestion wait", "Compute", "Upload queue", "Upload transfer"],
    alpha=0.8,
)
plt.title(f"Mean Per-Agent Time Breakdown ({DELAY_MODEL.upper()} Delays)")
plt.xlabel("Round")
plt.ylabel("Time (s)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./visualizations/time_breakdown_{DELAY_MODEL}.png", dpi=300)
plt.close()

# =========================================================
# PLOT: DELAY DISTRIBUTION
# =========================================================
plt.figure(figsize=(7, 5))
plt.hist(synthetic_delays * 1e6, bins=50, alpha=0.7)
plt.title(f"Synthesized Delay Distribution ({DELAY_MODEL.upper()})")
plt.xlabel("Delay (µs)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./visualizations/delay_distribution_{DELAY_MODEL}.png", dpi=300)
plt.close()

print("\nSaved:")
print(f"- simulation_metrics_{DELAY_MODEL}.csv")
print(f"- accuracy_{DELAY_MODEL}.png")
print(f"- per_class_accuracy_{DELAY_MODEL}.png")
print(f"- domain_shift_weather_{DELAY_MODEL}.png")
print(f"- domain_shift_camera_state_{DELAY_MODEL}.png")
print(f"- round_time_{DELAY_MODEL}.png")
print(f"- time_breakdown_{DELAY_MODEL}.png")
print(f"- delay_distribution_{DELAY_MODEL}.png")