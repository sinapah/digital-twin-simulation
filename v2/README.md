# V2: Interarrival Capture and Federated Training Simulation

V2 is the first stage of the project. It measures real UDP frame interarrival times between a sender and a receiver, fits synthetic delay models (KDE and WGAN), and runs a federated learning simulation on UA-DETRAC vehicle crops that uses those delays.

## Contents

- `sender.py` / `receiver.py`: send UA-DETRAC frames over UDP and log interarrival times to `interarrival_log.csv`.
- `kde_synthesis.py` / `wgan_synthesis.py`: fit KDE and WGAN models to the log and write `synthetic_interarrival_{kde,wgan}.csv`.
- `analysis.py`, `statistical_analysis.py`: summary statistics and distribution checks.
- `training_simulation.py`: 3-agent peer-to-peer federated training with simulated upload/download delays. Set `DELAY_MODEL` (`"kde"` or `"wgan"`) in its CONFIG block.

## Setup

```bash
cd v2
pip install -r requirements.txt
```

UA-DETRAC images and annotations must be in `DETRAC-Images/DETRAC-Images` and `DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML` at the repository root.

## Run

1. Capture interarrival times (edit `TARGET_IP` in `sender.py` first):

   ```bash
   python receiver.py          # on the receiver host
   python sender.py            # on the sender host
   ```

2. Fit the synthetic delay models:

   ```bash
   python kde_synthesis.py
   python wgan_synthesis.py
   ```

3. Run the training simulation:

   ```bash
   python training_simulation.py
   ```
