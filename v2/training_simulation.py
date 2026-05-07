# =========================================================
# INSTALL (run once)
# =========================================================
# !pip install torch torchvision matplotlib pandas

# =========================================================
# IMPORTS
# =========================================================
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# =========================================================
# CONFIG
# =========================================================
NUM_AGENTS = 3
ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 64

ARCHITECTURE = "peer_to_peer"
DELAY_MODEL = "kde"   # Options: "kde" or "wgan"

UPLOAD_MEAN = 0.3
UPLOAD_STD = 0.1

DOWNLOAD_MEAN = 0.2
DOWNLOAD_STD = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# LOAD SYNTHETIC DELAYS
# =========================================================
delay_file = f"{os.path.dirname(__file__)}/synthetic_interarrival_{DELAY_MODEL}.csv"

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
# CNN MODEL
# =========================================================
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),

            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# EDGE AGENT
# =========================================================
class EdgeAgent:

    def __init__(self, agent_id, dataset):

        self.id = agent_id
        self.data = dataset

        self.model = SimpleCNN().to(DEVICE)

        self.processing_delays = []
        self.upload_times = []
        self.download_times = []
        self.round_times = []

    def train(self, global_weights=None):

        if global_weights is not None:
            self.model.load_state_dict(global_weights)

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.01
        )

        loss_fn = nn.CrossEntropyLoss()

        loader = DataLoader(
            self.data,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        self.model.train()

        start = time.time()

        # -----------------------------------
        # LOCAL TRAINING
        # -----------------------------------
        for _ in range(LOCAL_EPOCHS):

            for x, y in loader:

                # -----------------------------------
                # SYNTHESIZED INTERARRIVAL DELAY
                # -----------------------------------
                delay = sample_delay()

                self.processing_delays.append(delay)

                # Simulate realistic timing
                time.sleep(delay)

                # -----------------------------------
                # TRAIN STEP
                # -----------------------------------
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                optimizer.zero_grad()

                preds = self.model(x)

                loss = loss_fn(preds, y)

                loss.backward()

                optimizer.step()

        # -----------------------------------
        # UPLOAD LATENCY
        # -----------------------------------
        if ARCHITECTURE == "centralized":
            upload_latency = np.random.normal(
                UPLOAD_MEAN,
                UPLOAD_STD
            )

        elif ARCHITECTURE == "regional":
            upload_latency = np.random.normal(
                UPLOAD_MEAN * 0.6,
                UPLOAD_STD
            )

        elif ARCHITECTURE == "peer_to_peer":
            upload_latency = np.random.normal(
                UPLOAD_MEAN * 0.4,
                UPLOAD_STD
            )

        else:
            upload_latency = 0

        upload_latency = max(upload_latency, 0)

        time.sleep(upload_latency)

        self.upload_times.append(upload_latency)

        total_time = time.time() - start
        self.round_times.append(total_time)

        return self.model.state_dict()

    # =====================================================
    # RECEIVE GLOBAL MODEL
    # =====================================================
    def receive_model(self, global_weights):

        latency = np.random.normal(
            DOWNLOAD_MEAN,
            DOWNLOAD_STD
        )

        latency = max(latency, 0)

        time.sleep(latency)

        self.download_times.append(latency)

        self.model.load_state_dict(global_weights)

# =========================================================
# FEDERATED AGGREGATION
# =========================================================
def aggregate_models(models):

    global_model = SimpleCNN().to(DEVICE)

    agg = {
        k: torch.zeros_like(v)
        for k, v in global_model.state_dict().items()
    }

    for m in models:

        for k in agg:
            agg[k] += m[k]

    for k in agg:
        agg[k] /= len(models)

    global_model.load_state_dict(agg)

    return global_model

# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x).argmax(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

# =========================================================
# DATASET
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=128
)

# =========================================================
# SPLIT DATA ACROSS AGENTS
# =========================================================
indices = np.random.permutation(len(train_data))

sizes = np.random.multinomial(
    len(indices),
    [1 / NUM_AGENTS] * NUM_AGENTS
)

splits = []

start = 0

for sz in sizes:

    splits.append(
        Subset(
            train_data,
            indices[start:start+sz]
        )
    )

    start += sz

# =========================================================
# CREATE AGENTS
# =========================================================
agents = [
    EdgeAgent(i, splits[i])
    for i in range(NUM_AGENTS)
]

global_model = SimpleCNN().to(DEVICE)

# =========================================================
# METRICS
# =========================================================
round_accuracies = []
round_total_times = []

# =========================================================
# RUN SIMULATION
# =========================================================
print(f"\nRunning Simulation")
print(f"Architecture: {ARCHITECTURE}")
print(f"Delay Model: {DELAY_MODEL.upper()}")

simulation_start = time.time()

for r in range(ROUNDS):

    print(f"\nRound {r+1}")

    round_start = time.time()

    # -----------------------------------
    # LOCAL TRAINING
    # -----------------------------------
    updates = []

    for agent in agents:

        update = agent.train(
            global_model.state_dict()
        )

        updates.append(update)

    # -----------------------------------
    # FEDERATED AGGREGATION
    # -----------------------------------
    global_model = aggregate_models(updates)

    # -----------------------------------
    # DISTRIBUTE GLOBAL MODEL
    # -----------------------------------
    for agent in agents:

        agent.receive_model(
            global_model.state_dict()
        )

    # -----------------------------------
    # EVALUATE
    # -----------------------------------
    acc = evaluate(global_model, test_loader)

    round_time = time.time() - round_start

    round_accuracies.append(acc)
    round_total_times.append(round_time)

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Round Time: {round_time:.2f}s")

# =========================================================
# FINAL REPORT
# =========================================================
total_time = time.time() - simulation_start

print(f"\nSimulation Complete")
print(f"Total Time: {total_time:.2f}s")

# =========================================================
# SAVE METRICS
# =========================================================
metrics_df = pd.DataFrame({
    "round": np.arange(1, ROUNDS+1),
    "accuracy": round_accuracies,
    "round_time": round_total_times
})

metrics_df.to_csv(
    f"simulation_metrics_{DELAY_MODEL}.csv",
    index=False
)

# =========================================================
# PLOT: ACCURACY
# =========================================================
plt.figure(figsize=(7,5))

plt.plot(
    round_accuracies,
    marker='o'
)

plt.title(
    f"Global Accuracy per Round ({DELAY_MODEL.upper()} Delays)"
)

plt.xlabel("Round")
plt.ylabel("Accuracy")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"accuracy_{DELAY_MODEL}.png",
    dpi=300
)

plt.close()

# =========================================================
# PLOT: ROUND TIME
# =========================================================
plt.figure(figsize=(7,5))

plt.plot(
    round_total_times,
    marker='o'
)

plt.title(
    f"Round Completion Time ({DELAY_MODEL.upper()} Delays)"
)

plt.xlabel("Round")
plt.ylabel("Time (s)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"round_time_{DELAY_MODEL}.png",
    dpi=300
)

plt.close()

# =========================================================
# PLOT: DELAY DISTRIBUTION
# =========================================================
plt.figure(figsize=(7,5))

plt.hist(
    synthetic_delays * 1e6,
    bins=50,
    alpha=0.7
)

plt.title(
    f"Synthesized Delay Distribution ({DELAY_MODEL.upper()})"
)

plt.xlabel("Delay (µs)")
plt.ylabel("Frequency")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"delay_distribution_{DELAY_MODEL}.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print(f"- simulation_metrics_{DELAY_MODEL}.csv")
print(f"- accuracy_{DELAY_MODEL}.png")
print(f"- round_time_{DELAY_MODEL}.png")
print(f"- delay_distribution_{DELAY_MODEL}.png")