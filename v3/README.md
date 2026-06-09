# V3 Camera-Outage Fallback Experiment

V3 extends the V2 federated UA-DETRAC training simulation with camera outages, historical fallback replay, synthetic interarrival delays, resource metrics, and a Makefile for running three sender processes plus one receiver.

There are two related workflows:

1. **Live receiver-side training**: senders stream labeled vehicle crops to the receiver VM, and training happens on the receiver. If a sender goes silent, the receiver uses local historical samples paced by fixed/KDE/WGAN synthetic delays.
2. **Offline training simulation**: runs locally from DETRAC files and synthetic delay CSVs. This is useful for fast repeatable comparison of `baseline_live`, `outage_no_fallback`, `outage_replay_fixed`, `outage_replay_kde`, and `outage_replay_wgan` without running UDP senders.

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

The sender transmits labeled UA-DETRAC vehicle crops. The receiver trains from those live samples. When a sender stops transmitting long enough to exceed `OUTAGE_TIMEOUT`, the receiver marks that sender as being in outage. If fallback is enabled, the receiver reads local historical DETRAC samples and injects them into training with synthetic delays.

### Receiver/training VM

On the receiver VM:

```bash
cd v3
make train-receiver FALLBACK_MODE=kde ROUNDS=100
```

The Makefile automatically uses `v3/venv/bin/python` if it exists, then `v3/.venv/bin/python`, then `python3`.

`FALLBACK_MODE` can be:

- `none`: train only on live samples; missing sender data is not replaced.
- `fixed`: use local historical samples with fixed/immediate replay timing.
- `kde`: use local historical samples paced by KDE synthetic delays.
- `wgan`: use local historical samples paced by WGAN synthetic delays.

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

This starts three camera sender processes. By default, `camera0` pauses during elapsed-time windows `20:35,55:70`; `camera1` and `camera2` stream continuously.

Override the outage windows like this:

```bash
make senders TARGET_HOST=<receiver-vm-ip> OUTAGE_WINDOWS=10:20,40:50
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

Use this only when you want to stream UA-DETRAC samples over UDP and observe outage detection/fallback logging without training.

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

- `camera0`: includes scheduled outage windows by default.
- `camera1`: continuous stream.
- `camera2`: continuous stream.

Default sender outage windows are:

```text
20:35,55:70
```

These are elapsed seconds from sender startup. During those windows the sender process stays alive but pauses transmission.

Override them like this:

```bash
make senders TARGET_HOST=<receiver-vm-ip> OUTAGE_WINDOWS=10:20,40:50
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
SEQUENCE_COUNT=10
FPS=25
OUTAGE_WINDOWS=20:35,55:70
```

So sender runtime is approximately the selected frame count divided by FPS, plus outage pause time for `camera0`.
