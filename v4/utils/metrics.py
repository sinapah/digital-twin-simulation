# =========================================================
# V4 Utilities - Metrics Collection
# =========================================================
# Provides utilities for collecting and reporting training metrics
# =========================================================

import time
import psutil
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TrainingMetrics:
    """Training metrics for one round"""
    round: int
    timestamp: float
    edge_id: int
    
    # Timing
    round_duration: float = 0.0
    ingestion_time: float = 0.0
    training_time: float = 0.0
    
    # Loss and accuracy
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    
    # Per-class metrics
    acc_car: float = 0.0
    acc_van: float = 0.0
    acc_bus: float = 0.0
    acc_others: float = 0.0
    
    # Sample counts
    samples_trained: int = 0
    live_samples: int = 0
    fallback_samples: int = 0
    
    # Resource usage
    cpu_avg: float = 0.0
    cpu_peak: float = 0.0
    memory_avg_mb: float = 0.0
    memory_peak_mb: float = 0.0
    
    # Queue metrics
    queue_length: int = 0
    queue_fill_time: float = 0.0
    
    # Throughput
    samples_per_second: float = 0.0


class MetricsCollector:
    """Collects and aggregates training metrics"""
    
    def __init__(self):
        self.metrics: List[TrainingMetrics] = []
        self.lock = None  # Will be initialized in thread
    
    def start_round(self, edge_id: int, round_num: int) -> TrainingMetrics:
        """Start collecting metrics for a round"""
        return TrainingMetrics(
            round=round_num,
            timestamp=time.time(),
            edge_id=edge_id
        )
    
    def record_resource_usage(self, metrics: TrainingMetrics):
        """Record current CPU and memory usage"""
        process = psutil.Process()
        
        metrics.cpu_avg = process.cpu_percent(interval=0.1)
        metrics.memory_avg_mb = process.memory_info().rss / (1024 * 1024)
    
    def finalize_round(self, metrics: TrainingMetrics, 
                      samples_processed: int,
                      training_duration: float):
        """Finalize metrics for a round"""
        metrics.samples_trained = samples_processed
        metrics.round_duration = time.time() - metrics.timestamp
        metrics.training_time = training_duration
        metrics.samples_per_second = samples_processed / max(training_duration, 0.001)
    
    def add_metrics(self, metrics: TrainingMetrics):
        """Add metrics to collection"""
        self.metrics.append(metrics)
    
    def get_metrics_df(self) -> 'pd.DataFrame':
        """Convert metrics to pandas DataFrame"""
        import pandas as pd
        
        data = []
        for m in self.metrics:
            data.append({
                'round': m.round,
                'timestamp': m.timestamp,
                'edge_id': m.edge_id,
                'round_duration': m.round_duration,
                'training_time': m.training_time,
                'train_loss': m.train_loss,
                'train_accuracy': m.train_accuracy,
                'balanced_accuracy': m.balanced_accuracy,
                'acc_car': m.acc_car,
                'acc_van': m.acc_van,
                'acc_bus': m.acc_bus,
                'acc_others': m.acc_others,
                'samples_trained': m.samples_trained,
                'live_samples': m.live_samples,
                'fallback_samples': m.fallback_samples,
                'cpu_avg': m.cpu_avg,
                'cpu_peak': m.cpu_peak,
                'memory_avg_mb': m.memory_avg_mb,
                'memory_peak_mb': m.memory_peak_mb,
                'queue_length': m.queue_length,
                'queue_fill_time': m.queue_fill_time,
                'samples_per_second': m.samples_per_second
            })
        
        return pd.DataFrame(data)
    
    def save_metrics(self, filepath: str):
        """Save metrics to CSV file"""
        df = self.get_metrics_df()
        df.to_csv(filepath, index=False)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics:
            return {}
        
        df = self.get_metrics_df()
        
        return {
            'total_rounds': len(self.metrics),
            'final_accuracy': df['train_accuracy'].iloc[-1] if len(df) > 0 else 0,
            'avg_samples_per_round': df['samples_trained'].mean(),
            'avg_training_time': df['training_time'].mean(),
            'avg_cpu': df['cpu_avg'].mean(),
            'avg_memory_mb': df['memory_avg_mb'].mean()
        }


class PerClassAccumulator:
    """Accumulates per-class metrics"""
    
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.correct = [0] * num_classes
        self.total = [0] * num_classes
        self.predictions = [0] * num_classes
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with new predictions"""
        pred_classes = predictions.argmax(dim=1)
        
        for pred, target in zip(pred_classes, targets):
            self.predictions[pred.item()] += 1
            self.total[target.item()] += 1
            if pred == target:
                self.correct[target.item()] += 1
    
    def get_accuracy(self, class_idx: int) -> float:
        """Get accuracy for a class"""
        if self.total[class_idx] == 0:
            return 0.0
        return self.correct[class_idx] / self.total[class_idx]
    
    def get_balanced_accuracy(self) -> float:
        """Get balanced accuracy (mean per-class accuracy)"""
        accuracies = [self.get_accuracy(i) for i in range(self.num_classes)]
        return sum(accuracies) / len(accuracies)
    
    def get_metrics_dict(self) -> Dict[str, float]:
        """Get metrics as dictionary"""
        return {
            'acc_car': self.get_accuracy(0),
            'acc_van': self.get_accuracy(1),
            'acc_bus': self.get_accuracy(2),
            'acc_others': self.get_accuracy(3),
            'pred_car': self.predictions[0],
            'pred_van': self.predictions[1],
            'pred_bus': self.predictions[2],
            'pred_others': self.predictions[3],
            'total_car': self.total[0],
            'total_van': self.total[1],
            'total_bus': self.total[2],
            'total_others': self.total[3]
        }
