# V6: Digital Twin for Federated Edge Learning

V6 extends V5 with a digital twin. The twin collects edge telemetry, detects outages, and synthesizes missing telemetry with KDE or WGAN generators. An orchestrator can send control actions back to the edges (local epochs, batch size, learning rate, sampling rate, aggregation interval, client participation, sender FPS). Evaluation metrics and history are logged to `outputs/dt/`.

## Contents

- `simulator.py`: runs an experiment, optionally with the digital twin enabled (`--dt-mode none|kde|wgan`).
- `aggregator.py`, `edge_agent.py`, `sender.py`, `camera_simulator.py`: same roles as in V5, with twin hooks added.
- `digital_twin/`: telemetry collector, history repository, outage detector, predictor, state representation.
- `telemetry_generation/`: KDE and WGAN telemetry generators.
- `orchestrator/`: control actions and the orchestrator server.
- `evaluation/`: metrics and logging.

## Setup

As in V5, `run.sh` uses the virtual environment at `../v4/venv`:

```bash
../v4/venv/bin/pip install -r requirements.txt
```

Delay models must be fitted before the `kde` and `wgan` modes can run. Run `./run.sh baseline` and then `./run.sh fit` first.

## Run

Without the digital twin:

```bash
cd v6
./run.sh baseline 100
./run.sh fit
./run.sh kde 100
./run.sh wgan 100
```

With the digital twin:

```bash
./run.sh dt-baseline 100   # twin in observation mode
./run.sh dt-kde 100        # twin with KDE telemetry synthesis
./run.sh dt-wgan 100       # twin with WGAN telemetry synthesis
```

Results: edge metrics in `outputs/<mode>/`, twin metrics in `outputs/dt/metrics/`, and twin history in `outputs/dt/history/`.

Remote senders (on each sender host, run `./run.sh sender0 --mode kde --fps 500 --rounds 100`, then sender1 and sender2; on the simulator host, pass the three sender addresses):

```bash
./run.sh dt-kde 100 <host0> <host1> <host2>
```
