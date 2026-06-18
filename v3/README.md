# V3 Camera-Outage Fallback Experiment

V3 extends the V2 federated UA-DETRAC training simulation with camera outages, historical fallback replay, synthetic interarrival delays, resource metrics, and a Makefile for running three sender processes plus one receiver.

There are three related workflows:

1. **Live receiver-side training**: senders stream labeled vehicle crops to the receiver VM, and training happens on the receiver. If a sender goes silent, the receiver uses local historical samples paced by fixed/KDE/WGAN synthetic delays.
2. **Offline training simulation**: runs locally from DETRAC files and synthetic delay CSVs. This is useful for fast repeatable comparison of `baseline_live`, `outage_no_fallback`, `outage_replay_fixed`, `outage_replay_kde`, and `outage_replay_wgan` without running UDP senders.
3. **Fully simulated digital twin**: simulates 24 conceptual UA-DETRAC intersections as sensors, three edge devices with seven live intersections each, three held-out historical fallback intersections, sensor outages, edge-local training, and federated learning between the edge devices.

## Setup

Run commands from the repository root unless noted.

```bash
cd v3
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If you already created a `v3/venv` or `v3/.venv` before `opencv-python-headless` was added, run the install command again inside that environment:

```bash
pip install -r requirements.txt
```

Expected data layout:

```text
DETRAC-Images/DETRAC-Images/
DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/
v2/synthetic_interarrival_kde.csv
v2/synthetic_interarrival_wgan.csv
```

## Run the live receiver-side training experiment

This is the main end-to-end experiment path:

```text
sender.py processes on sender VM  --->  receiver_training.py on receiver VM
                                      training happens here
```

The sender transmits labeled UA-DETRAC vehicle crops. The receiver trains from those live samples. When a sender stops transmitting long enough to exceed `OUTAGE_TIMEOUT`, the receiver marks that sender as being in outage. If fallback is enabled, the receiver reads local historical DETRAC samples and injects them into the same per-sender training queue used by live samples, paced by fixed/KDE/WGAN synthetic delays.

### Receiver/training VM

On the receiver VM:

```bash
cd v3
make train-receiver FALLBACK_MODE=kde ROUNDS=100
```

The Makefile automatically uses `v3/venv/bin/python` if it exists, then `v3/.venv/bin/python`, then `python3`.

By default, the online receiver reserves the first 60 sorted UA-DETRAC folders for live sender traffic, uses 10 later folders for evaluation, and uses the last 20 sorted UA-DETRAC folders as the receiver-side fallback repository. Override those counts with:

```bash
make train-receiver \
  FALLBACK_MODE=kde \
  ROUNDS=100 \
  LIVE_RESERVED_VIDEO_COUNT=60 \
  FALLBACK_VIDEO_COUNT=20 \
  TIME_SCALE=1.0
```

`FALLBACK_MODE` can be:

- `none`: train only on live samples; missing sender data is not replaced.
- `fixed`: use local historical samples with fixed/immediate replay timing.
- `kde`: use local historical samples paced by KDE synthetic delays.
- `wgan`: use local historical samples paced by WGAN synthetic delays.

For short demos where KDE/WGAN delays make fallback arrivals too sparse to see in every round, lower `TIME_SCALE` or increase `ROUND_COLLECT_SECONDS`, for example:

```bash
make train-receiver FALLBACK_MODE=kde ROUNDS=100 TIME_SCALE=0.1 ROUND_COLLECT_SECONDS=5
```

Receiver-side training outputs:

```text
v3/outputs/receiver_training_metrics.csv
v3/outputs/receiver_training_events.csv
```

### Sender VM

On the sender VM:

```bash
cd v3
make senders TARGET_HOST=<receiver-vm-ip>
```

This starts three camera sender processes. By default, each sender streams 20 sorted UA-DETRAC folders:

- `camera0`: first 20 folders, starting at index `0`.
- `camera1`: second 20 folders, starting at index `20`.
- `camera2`: third 20 folders, starting at index `40`.

By default, `camera0` pauses during elapsed-time windows `20:35,55:70`; `camera1` and `camera2` stream continuously.

Override the outage windows like this:

```bash
make senders \
  TARGET_HOST=<receiver-vm-ip> \
  SENDER0_OUTAGE_WINDOWS=10:20,40:50 \
  SENDER1_OUTAGE_WINDOWS=30:45 \
  SENDER2_OUTAGE_WINDOWS=
```

### Receiver resource usage

Receiver resource usage is measured inside `receiver_training.py` with `psutil.Process(os.getpid())` during every training round.

The metrics include:

- `cpu_avg`: mean CPU percent sampled during the round.
- `cpu_peak`: maximum CPU percent sampled during the round.
- `cpu_var`: variance of CPU percent samples.
- `memory_avg_mb`: mean receiver process RSS memory in MiB.
- `memory_peak_mb`: maximum receiver process RSS memory in MiB.

`psutil` process CPU percent is measured relative to one CPU core, so values above `100` are possible on multicore machines. The metrics CSV also includes `cpu_avg_host_percent` and `cpu_peak_host_percent`, which divide by host CPU count to show approximate share of total host CPU capacity.

The resource scope is `receive_fallback_train_excludes_evaluation`: UDP receiving, outage monitoring, fallback replay, and model training are included; test-set evaluation is excluded so CPU does not look busy when no training samples were processed.

The CSV keeps `samples_processed` for compatibility, but it means samples trained in that round. Newer columns use clearer names: `samples_trained`, `live_samples_trained`, and `fallback_samples_trained`.

Overall `frame_accuracy` can look constant when the test set is class-imbalanced. For example, if the model predicts only `car`, frame accuracy may equal the fraction of test samples that are cars even though minority classes are failing. Use these diagnostic columns to interpret results:

- `balanced_accuracy`: mean per-class recall; this is less sensitive to class imbalance.
- `acc_car`, `acc_van`, `acc_bus`, `acc_others`: per-class accuracy.
- `pred_car`, `pred_van`, `pred_bus`, `pred_others`: model prediction counts on the test set.
- `test_car`, `test_van`, `test_bus`, `test_others`: test label counts.
- `train_car`, `train_van`, `train_bus`, `train_others`: labels actually trained in that round.
- `train_loss_avg`: average training loss for the round.
- `stale_samples_discarded`: queued samples dropped before the collection window so old backlog does not hide an outage.
- `samples_received`: live plus fallback samples received in that round.
- `samples_trained`: samples actually used for optimization in that round.
- `replay_buffer_size`: cumulative receiver-side replay buffer size.
- `received_*`: class counts received in the current round.
- `buffer_*`: class counts available in the replay buffer.

Receiver-side training uses a class-balanced replay buffer by default. Incoming live/fallback samples are added to the buffer, then training batches are sampled across available classes so the model does not only learn the majority `car` class. Tune this behavior with:

```bash
make train-receiver \
  FALLBACK_MODE=kde \
  ROUNDS=100 \
  TRAIN_BATCHES_PER_ROUND=8 \
  REPLAY_BUFFER_SIZE=20000 \
  MAX_CLASS_WEIGHT=5.0
```

Or run the Python script directly:

```bash
python3 receiver_training.py \
  --fallback-mode kde \
  --train-batches-per-round 8 \
  --replay-buffer-size 20000 \
  --max-class-weight 5.0
```

If no live or fallback samples reach a training round, `receiver_training.py` now fails instead of writing misleading constant accuracy rows. Use `--allow-empty-rounds` only when debugging receiver startup.

## Run the fully simulated digital twin

The digital twin is a controlled local simulation, not a VM/UDP workflow. It treats senders as simulated sensors/cameras and receivers as edge devices. Three edge devices each receive seven live intersections and federate their models after local training rounds. Three intersections are reserved as historical fallback data.

The split is defined in:

```text
v3/digital_twin_manifest.json
```

The design notes are in:

```text
v3/digital_twin_plan.md
```

Run all default scenarios:

```bash
cd v3
python3 digital_twin_simulation.py
```

Run a single fast smoke scenario:

```bash
python3 digital_twin_simulation.py \
  --scenarios outage_replay_kde \
  --rounds 1 \
  --max-frames-per-video 1 \
  --samples-per-sensor-per-round 1 \
  --batch-size 4 \
  --train-batches-per-round 1 \
  --round-collect-seconds 0 \
  --time-scale 0 \
  --outage-start-round 1 \
  --outage-end-round 1 \
  --outage-intersections intersection_00
```

Common options:

- `--scenarios baseline_live,outage_replay_kde`
- `--outage-intersections intersection_00,intersection_07`
- `--outage-start-round 20 --outage-end-round 60`
- `--samples-per-sensor-per-round 16`
- `--round-collect-seconds 2`
- `--time-scale 0` for fast validation without delay sleeps

Digital twin outputs are written to:

```text
v3/outputs/digital_twin/
```

Each scenario writes:

- `global_metrics_<scenario>.csv`
- `edge_metrics_<scenario>.csv`
- `sensor_metrics_<scenario>.csv`

## Run the offline training simulation

From `v3`:

```bash
python3 training_simulation.py
```

This runs all default scenarios for 100 rounds without requiring sender/receiver VMs:

- `baseline_live`
- `outage_no_fallback`
- `outage_replay_fixed`
- `outage_replay_kde`
- `outage_replay_wgan`

The default outage window is rounds `20` through `60` for agent `0`.

### Quick smoke test

Use this for a fast sanity check:

```bash
python3 training_simulation.py \
  --scenarios outage_replay_kde \
  --rounds 1 \
  --local-epochs 1 \
  --batch-size 8 \
  --videos-per-agent 1 \
  --historical-videos-per-agent 1 \
  --test-videos 1 \
  --max-frames-per-video 1 \
  --outage-start-round 1 \
  --outage-end-round 1 \
  --outage-agents 0 \
  --time-scale 0 \
  --track-eval-interval 1
```

### Useful simulation options

```bash
python3 training_simulation.py --help
```

Common options:

- `--scenarios baseline_live,outage_replay_wgan`
- `--rounds 20`
- `--outage-start-round 5 --outage-end-round 15`
- `--outage-agents 0,1`
- `--time-scale 0` to disable sleeps during fast validation

Simulation outputs are written to:

```text
v3/outputs/
v3/visualizations/
```

## Run the logging-only live sender/receiver demo

Use this only when you want to stream UA-DETRAC samples over UDP and observe outage detection/fallback logging without training. Historical image replay for training is implemented in `receiver_training.py`; this logging-only receiver records synthetic fallback arrival events.

### Receiver VM

On the receiver VM:

```bash
cd v3
make receiver FALLBACK_MODE=kde
```

`FALLBACK_MODE` can be:

- `none`
- `fixed`
- `kde`
- `wgan`

The logging-only receiver writes logs to:

```text
v3/logs/receiver.log
v3/outputs/receiver_interarrival_log.csv
```

### Sender VM

On the sender VM:

```bash
cd v3
make senders TARGET_HOST=<receiver-vm-ip>
```

This starts three sender processes:

- `camera0`: first 20 sorted UA-DETRAC folders; includes scheduled outage windows by default.
- `camera1`: second 20 sorted UA-DETRAC folders; continuous stream by default.
- `camera2`: third 20 sorted UA-DETRAC folders; continuous stream by default.

Default sender outage windows are:

```text
20:35,55:70
```

These are elapsed seconds from sender startup. During those windows the sender process stays alive but pauses transmission.

Override them like this:

```bash
make senders \
  TARGET_HOST=<receiver-vm-ip> \
  SENDER0_OUTAGE_WINDOWS=10:20,40:50 \
  SENDER1_OUTAGE_WINDOWS=30:45 \
  SENDER2_OUTAGE_WINDOWS=
```

### One-VM local demo

For a local quick run from `v3`:

```bash
make smoke-senders
```

This starts the receiver, runs all three senders briefly, then stops only the PIDs recorded by the Makefile.

## Stop and clean up

From `v3`:

```bash
make stop
make clean-logs
```

`make stop` only terminates processes whose PIDs are recorded under `v3/runtime/`.

## Experiment duration

Training simulation duration depends on scenario count, rounds, local epochs, frame cap, and hardware. Defaults are intentionally full-size:

```text
5 scenarios x 100 rounds x 3 local epochs
```

The live sender demo duration depends on selected sequence length and FPS. Defaults use:

```text
SEQUENCE_COUNT=20
FPS=25
SENDER0_OUTAGE_WINDOWS=20:35,55:70
SENDER1_OUTAGE_WINDOWS=
SENDER2_OUTAGE_WINDOWS=
```

So sender runtime is approximately the selected frame count divided by FPS, plus outage pause time for `camera0`.
