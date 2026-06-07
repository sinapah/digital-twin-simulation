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

3. Build non-leaking live, historical, and test splits
   - Extend the existing video-level split so each agent has live-training videos and a separate historical replay repository.
   - Keep the test videos disjoint from both live and historical replay data.
   - If the local DETRAC set has insufficient videos for all desired splits, fail loudly with a clear error instead of silently reusing test data.

4. Implement outage/fallback data scheduling inside `EdgeAgent`
   - Add a per-agent data-source policy that decides which loader to use for each round:
     - normal live dataset outside outages
     - reduced/no data for `outage_no_fallback`
     - historical replay for fallback scenarios
   - Track whether samples came from live or fallback sources.
   - For no-fallback outages, skip unavailable camera data while allowing other agents to continue.
   - For replay scenarios, use historical samples as the missing camera's replacement stream.

5. Implement delay strategies for fallback replay
   - Preserve current synthetic delay loading for KDE/WGAN.
   - Add explicit delay samplers:
     - measured/current live delay behavior
     - no-fallback/no arrivals
     - fixed or immediate replay delay for naive replay
     - KDE synthetic delays
     - WGAN synthetic delays
   - Apply the selected sampler to fallback ingestion timing so bursty naive replay and paced synthetic replay differ in resource behavior.

6. Add resource, queue, throughput, and continuity metrics
   - Add a lightweight metrics collector around each round and scenario.
   - Record CPU average, CPU peak, CPU variance, memory average, memory peak, logical queue length average/max, images received/processed per second, samples processed, rounds completed, and effective training duration.
   - Preserve existing timing metrics: round wall time, ingestion wait, compute time, upload queue, upload transfer, and download time.
   - Include live/fallback sample counts per round so data continuity is visible.

7. Run all v3 scenarios and save comparable outputs
   - Add an experiment runner that executes every configured scenario and writes one metrics CSV per scenario.
   - Save a combined summary CSV comparing final accuracy, track accuracy, resource metrics, throughput, queue length, samples processed, and total runtime.
   - Include scenario and delay strategy columns in every metrics row.

8. Add visualizations for v3 hypotheses
   - Keep v2 plots for accuracy, per-class accuracy, round time, and timing breakdown.
   - Add comparison plots for CPU utilization, memory usage, queue length, throughput, samples processed, and final accuracy across fallback strategies.
   - Add a plot that highlights outage windows so accuracy/resource changes can be interpreted against camera failures.

9. Validate the implementation
   - Run syntax/import checks for the new v3 script.
   - Run a small deterministic smoke experiment with reduced rounds and frame counts.
   - Confirm generated CSVs contain the expected scenario, resource, continuity, and accuracy columns.
   - Confirm v2 files and outputs remain unchanged unless explicitly requested.

## Notes and considerations

- The first implementation should be simulation-first, because the requested base file is `v2/training_simulation.py`; the UDP sender/receiver can remain as trace-generation tools unless later work requires live VM orchestration.
- The outage model should be deterministic by default for reproducibility, with all random seeds set consistently.
- Historical replay must not draw from the held-out test set.
- The v3 script should avoid silently fabricating successful metrics if an outage leaves an agent with no data; it should record zero processed samples for that source and continue only where the scenario semantics allow it.
- If resource metrics use `psutil`, requirements must be updated and the code should surface an import error with an installation hint rather than swallowing it.
