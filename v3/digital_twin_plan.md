# V3 Fully Simulated Digital Twin Plan

## Goal

Extend v3 from a sender/receiver outage experiment into a fully simulated digital twin of a distributed traffic-camera learning system.

In this digital twin:

- **Sensors/cameras** simulate UA-DETRAC intersections producing image streams.
- **Edge devices** receive images from their assigned sensors and train local models.
- **Three edge devices** federate with each other using model aggregation.
- **Outages** occur at the sensor/intersection level.
- **Historical fallback data** is used when live sensor data is unavailable.

## Proposed UA-DETRAC split

UA-DETRAC is treated as a 24-intersection source.

| Role | Count | Purpose |
|---|---:|---|
| Edge 0 live intersections | 7 | Sensors assigned to edge 0 |
| Edge 1 live intersections | 7 | Sensors assigned to edge 1 |
| Edge 2 live intersections | 7 | Sensors assigned to edge 2 |
| Historical fallback intersections | 3 | Reserved fallback repository for outages |

This 7/7/7/3 split is a reasonable design because it keeps most intersections active in the live system while reserving a disjoint historical pool for fallback behavior.

## Important dataset note

The current code operates on UA-DETRAC sequence folders. The local dataset has annotated sequence folders, but the current implementation does not expose an explicit 24-intersection mapping. The digital twin should therefore introduce a manifest that maps conceptual intersections to one or more UA-DETRAC sequence folders.

The manifest should make these assignments explicit:

- intersection ID
- assigned edge
- role: live, historical fallback, or test/evaluation
- underlying UA-DETRAC sequence folders

Live, fallback, and test/evaluation folders must be disjoint.

## Planned architecture

```text
sensor/intersection streams
        |
        v
edge-local queues and replay buffers
        |
        v
edge-local model training
        |
        v
federated aggregation between edges
        |
        v
global model and metrics
```

Each edge should own:

- assigned sensors/intersections
- an edge-local sample queue
- a replay buffer
- a local model and optimizer
- outage and fallback state
- per-edge metrics

## Outage behavior

Outages should be configured per sensor/intersection. During an outage:

1. The affected live sensor stops producing samples.
2. The owning edge detects or is told that the sensor is unavailable.
3. Depending on the scenario, the edge either:
   - receives no replacement data,
   - replays historical data at a fixed/immediate rate,
   - replays historical data paced by KDE synthetic delays,
   - replays historical data paced by WGAN synthetic delays.

Fallback samples should enter the same edge-local queue as live samples so queue length, queue fill time, throughput, CPU, and training behavior remain comparable.

## Core experiment scenarios

| Scenario | Outage | Fallback |
|---|---|---|
| Normal live baseline | No | No |
| Outage no fallback | Yes | No |
| Outage fixed fallback | Yes | Fixed/immediate historical replay |
| Outage KDE fallback | Yes | KDE-paced historical replay |
| Outage WGAN fallback | Yes | WGAN-paced historical replay |

## Metrics

The digital twin should record:

- global model accuracy and balanced accuracy
- per-edge samples received and trained
- per-edge live samples and fallback samples
- per-edge queue length and queue-fill time
- per-sensor outage status and sample counts
- CPU average, CPU peak, CPU variance
- memory average and peak
- throughput in samples/images per second
- training loss and convergence stability

## Implementation direction

Add a new digital twin simulation entrypoint rather than replacing the current online VM-oriented workflow. A likely file name is:

```text
v3/digital_twin_simulation.py
```

Reusable pieces from existing v3 code:

- `DETRACDataset`
- `SimpleCNN`
- synthetic delay loading and sampling
- resource sampling
- FedAvg aggregation
- accuracy and per-class evaluation helpers

New pieces needed:

- intersection manifest
- sensor simulation
- edge-node abstraction
- edge-local queue and replay buffer logic
- per-sensor outage scheduler
- digital-twin-specific metrics and outputs
