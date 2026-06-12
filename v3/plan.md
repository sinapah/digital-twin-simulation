# Implementation Plan: v3 Camera-Outage Fallback Simulation

## Problem and approach

The v3 hypothesis proposes extending the existing UA-DETRAC federated-learning simulation with camera outages, receiver-side fallback replay, synthetic interarrival delays, and resource/continuity metrics. The implementation should preserve the working v2 experiment and create a new `v3/training_simulation.py` derived from `v2/training_simulation.py`.

The plan is to keep v2's core pieces--DETRAC crop dataset, SimpleCNN, federated EdgeAgent training, KDE/WGAN delay sampling, FedAvg aggregation, and accuracy/domain-shift evaluation--then add scenario-driven data availability behavior. Each experiment scenario will control whether an agent has live data, no replacement data, naive historical replay, KDE-paced replay, or WGAN-paced replay during configured outage windows.

## Current state observed

- `v2/training_simulation.py` is a single top-level script that:
  - Builds agent/test DETRAC datasets from annotated video folders.
  - Trains three `EdgeAgent` instances in parallel using FedAvg.
  - Uses synthetic interarrival delay CSVs for ingestion sleeps.
  - Records accuracy, per-class accuracy, track/domain-shift evaluation, and timing breakdowns.
  - Saves CSV metrics and plots for one selected `DELAY_MODEL`.
- `v2/sender.py` and `v2/receiver.py` collect real UDP interarrival traces, but the training simulation already consumes generated delay CSVs directly.
- `v2/sender.py` currently runs one sender loop over selected folders, with hard-coded target IP/port and no sender identity beyond folder/frame metadata in the packet header.
- `v2/receiver.py` currently logs global packet interarrival time only; it does not reassemble images, track camera-specific last-seen times, or detect outages per sender.
- `v3/hypothesis.md` asks for outage scenarios, fallback historical replay, synthetic-delay replay, CPU/memory/queue/throughput metrics, and comparisons against no-fallback and naive replay baselines.

## Implementation todos

1. Create the v3 simulation baseline
   - Copy `v2/training_simulation.py` to `v3/training_simulation.py`.
   - Add `v3/requirements.txt` derived from v2 and include any resource-metric dependency needed for CPU/memory sampling, likely `psutil`.
   - Ensure output paths write under `v3/outputs/` and `v3/visualizations/` so v2 artifacts are untouched.

2. Refactor configuration and execution
   - Replace scattered global constants with a small experiment configuration section or dataclass-like structures.
   - Add scenario definitions for:
     - `baseline_live`
     - `outage_no_fallback`
     - `outage_replay_fixed`
     - `outage_replay_kde`
     - `outage_replay_wgan`
   - Add outage schedule configuration by round and agent, e.g. which cameras fail and for which rounds.
   - Keep quick-run knobs for validation, such as low rounds and low max frames per video.

3. Add VM orchestration for live sender/receiver experiments
   - Add a `v3/Makefile` as the main operator entrypoint for repeatable VM runs.
   - Include targets for setup, receiver startup, three sender startup, individual sender startup, log cleanup, and stopping only processes started by the Makefile.
   - Back the Makefile with small shell or Python launch helpers if needed so three sender processes can run concurrently with distinct sender IDs, sequence assignments, ports or shared-port identities, and per-sender logs.
   - Avoid destructive process cleanup; track PIDs in a v3 runtime directory and terminate those specific PIDs.

4. Build non-leaking live, historical, and test splits
   - Extend the existing video-level split so each agent has live-training videos and a separate historical replay repository.
   - Keep the test videos disjoint from both live and historical replay data.
   - If the local DETRAC set has insufficient videos for all desired splits, fail loudly with a clear error instead of silently reusing test data.

5. Implement outage/fallback data scheduling inside `EdgeAgent`
   - Add a per-agent data-source policy that decides which loader to use for each round:
     - normal live dataset outside outages
     - reduced/no data for `outage_no_fallback`
     - historical replay for fallback scenarios
   - Track whether samples came from live or fallback sources.
   - For no-fallback outages, skip unavailable camera data while allowing other agents to continue.
   - For replay scenarios, use historical samples as the missing camera's replacement stream.

6. Implement intermittent stoppages for live sender/receiver flows
   - Extend the v3 sender so each process accepts `--sender-id`, assigned sequence paths, target host/port, FPS, and outage windows.
   - Model intermittent camera stoppages by pausing transmission inside configured outage windows while keeping the sender process alive and logging the start/end of each outage.
   - Include sender identity in every packet header so the receiver can maintain per-camera state instead of relying only on IP address.
   - Extend the receiver to track `last_seen` by sender ID, declare an outage after a configurable timeout, and emit outage start/end events.
   - In no-fallback mode, the receiver records the missing stream and lets downstream training see reduced data.
   - In fallback modes, the receiver injects historical images for the missing sender into the same queue used for live arrivals, paced by the selected delay strategy.

7. Implement delay strategies for fallback replay
   - Preserve current synthetic delay loading for KDE/WGAN.
   - Add explicit delay samplers:
     - measured/current live delay behavior
     - no-fallback/no arrivals
     - fixed or immediate replay delay for naive replay
     - KDE synthetic delays
     - WGAN synthetic delays
   - Apply the selected sampler to fallback ingestion timing so bursty naive replay and paced synthetic replay differ in resource behavior.

8. Add resource, queue, throughput, and continuity metrics
   - Add a lightweight metrics collector around each round and scenario.
   - Record CPU average, CPU peak, CPU variance, memory average, memory peak, logical queue length average/max, images received/processed per second, samples processed, rounds completed, and effective training duration.
   - Preserve existing timing metrics: round wall time, ingestion wait, compute time, upload queue, upload transfer, and download time.
   - Include live/fallback sample counts per round so data continuity is visible.

9. Run all v3 scenarios and save comparable outputs
   - Add an experiment runner that executes every configured scenario and writes one metrics CSV per scenario.
   - Save a combined summary CSV comparing final accuracy, track accuracy, resource metrics, throughput, queue length, samples processed, and total runtime.
   - Include scenario and delay strategy columns in every metrics row.

10. Add visualizations for v3 hypotheses
   - Keep v2 plots for accuracy, per-class accuracy, round time, and timing breakdown.
   - Add comparison plots for CPU utilization, memory usage, queue length, throughput, samples processed, and final accuracy across fallback strategies.
   - Add a plot that highlights outage windows so accuracy/resource changes can be interpreted against camera failures.

11. Validate the implementation
   - Run syntax/import checks for the new v3 script.
   - Run a small deterministic smoke experiment with reduced rounds and frame counts.
   - Confirm generated CSVs contain the expected scenario, resource, continuity, and accuracy columns.
   - Exercise Makefile dry-run/help targets and, where safe, a short three-sender local run against the receiver.
   - Confirm v2 files and outputs remain unchanged unless explicitly requested.

## Notes and considerations

- A Makefile is appropriate here because the live experiment requires repeatably starting coordinated receiver and sender processes with consistent arguments, logs, and cleanup.
- The first implementation should be simulation-first for training behavior because the requested base file is `v2/training_simulation.py`; the sender/receiver orchestration should support trace collection and end-to-end demonstration without blocking the simulation work.
- Intermittent stoppages should be deterministic and schedule-driven, not implemented by killing sender processes. Pausing transmission keeps process supervision simple and makes outage windows reproducible.
- The outage model should be deterministic by default for reproducibility, with all random seeds set consistently.
- Historical replay must not draw from the held-out test set.
- The v3 script should avoid silently fabricating successful metrics if an outage leaves an agent with no data; it should record zero processed samples for that source and continue only where the scenario semantics allow it.
- If resource metrics use `psutil`, requirements must be updated and the code should surface an import error with an installation hint rather than swallowing it.
