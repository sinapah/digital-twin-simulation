# V5: Simulator-Driven Federated Learning

V5 turns the distributed system from V4 into a simulator. `simulator.py` launches the aggregator, three edge agents, and three camera senders as subprocesses. Edges train a `SimpleCNN` on UA-DETRAC crops. Frame arrivals are paced by delays from `baseline` (measured), `kde`, or `wgan` (synthetic). Weights move over TCP in binary form and are combined with FedAvg.

## Contents

- `simulator.py`: runs one full experiment in a chosen mode.
- `aggregator.py`: FedAvg server.
- `edge_agent.py`: edge training and metric logging.
- `sender.py` / `camera_simulator.py`: UDP frame streaming with `baseline`, `kde`, or `wgan` delays.
- `fit_delays.py`: fits KDE and WGAN models to the baseline arrival logs.
- `visualization/`: plotting scripts for the metrics.

## Setup

`run.sh` expects a virtual environment at `../v4/venv`, which is created by `v4/setup.sh`. Then install the V5 requirements into it:

```bash
../v4/venv/bin/pip install -r requirements.txt
```

UA-DETRAC images must be at `DETRAC-Images/DETRAC-Images` at the repository root.

## Run

Local run (all processes on one machine):

```bash
cd v5
./run.sh baseline 100   # measure real delays
./run.sh fit            # fit KDE/WGAN from the baseline logs
./run.sh kde 100        # simulate with KDE delays
./run.sh wgan 100       # simulate with WGAN delays
```

Metrics are written to `outputs/<mode>/`.

Distributed run (senders on separate hosts):

```bash
# On each sender host
./run.sh sender0 --mode kde --fps 25 --rounds 100   # also sender1, sender2

# On the simulator host
./run.sh kde 100 <host0> <host1> <host2>
```
