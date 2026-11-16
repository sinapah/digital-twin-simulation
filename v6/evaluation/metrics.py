import numpy as np
from typing import List, Dict, Optional


def system_metrics(edge_states: list) -> dict:
    cpus = [e.cpu_util for e in edge_states if not e.is_outage]
    mems = [e.memory_util for e in edge_states if not e.is_outage]
    queues = [e.queue_length for e in edge_states]
    latencies = [e.inference_latency for e in edge_states if not e.is_outage]

    return {
        'avg_cpu': float(np.mean(cpus)) if cpus else 0.0,
        'max_cpu': float(np.max(cpus)) if cpus else 0.0,
        'min_cpu': float(np.min(cpus)) if cpus else 0.0,
        'cpu_std': float(np.std(cpus)) if cpus else 0.0,
        'avg_memory': float(np.mean(mems)) if mems else 0.0,
        'avg_queue_length': float(np.mean(queues)) if queues else 0.0,
        'avg_latency': float(np.mean(latencies)) if latencies else 0.0,
        'resource_balance': float(np.std(cpus)) if cpus else 0.0,
    }


def fl_metrics(global_state) -> dict:
    return {
        'rounds_completed': global_state.fl_rounds_completed,
        'aggregation_delay': global_state.aggregation_delay,
        'client_participation_rate': global_state.client_participation_rate,
        'num_participants': global_state.num_participants,
        'num_edges_connected': global_state.num_edges_connected,
        'avg_accuracy': global_state.avg_accuracy,
        'avg_loss': global_state.avg_loss,
    }


def dt_metrics(state_representation) -> dict:
    gs = state_representation.global_state
    synthetic_edges = sum(1 for e in state_representation.edges if e.is_synthetic)
    outage_edges = sum(1 for e in state_representation.edges if e.is_outage)
    return {
        'global_cpu_estimation': gs.avg_cpu,
        'global_accuracy_estimation': gs.avg_accuracy,
        'synthetic_edges': synthetic_edges,
        'outage_edges': outage_edges,
        'prediction_accuracy': gs.avg_accuracy,
    }


def synthetic_telemetry_metrics(predicted: dict, actual: dict) -> dict:
    metrics = {}
    common_keys = set(predicted.keys()) & set(actual.keys())
    if not common_keys:
        return {'mae': 0.0, 'rmse': 0.0, 'wasserstein': 0.0}

    pred_vals = np.array([predicted[k] for k in common_keys], dtype=float)
    actual_vals = np.array([actual[k] for k in common_keys], dtype=float)

    mae = float(np.mean(np.abs(pred_vals - actual_vals)))
    rmse = float(np.sqrt(np.mean((pred_vals - actual_vals) ** 2)))

    try:
        from scipy.stats import wasserstein_distance
        ws = float(wasserstein_distance(pred_vals, actual_vals))
    except ImportError:
        ws = 0.0

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'wasserstein': round(ws, 4),
    }
