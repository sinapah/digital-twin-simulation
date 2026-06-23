import threading
import time
import sys
import os
import random
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregator import Aggregator
from edge_agent import EdgeAgent
from camera_simulator import CameraSimulator, NUM_EDGES, VIDEOS_PER_EDGE


AGGREGATOR_PORT = 5000
NUM_ROUNDS = 100


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class Simulator:
    def __init__(self, delay_model: str = 'kde', num_rounds: int = NUM_ROUNDS):
        self.delay_model = delay_model
        self.num_rounds = num_rounds
        self.camera = CameraSimulator(delay_model=delay_model)
        self.aggregator = Aggregator(port=AGGREGATOR_PORT, num_rounds=num_rounds)
        self.edges = []
        self.outage_periods = []

        flush_print(f"\n{'='*60}")
        flush_print(f"Digital Twin Federated Learning Simulator")
        flush_print(f"{'='*60}")
        flush_print(f"Delay model: {delay_model.upper()}")
        flush_print(f"Rounds: {num_rounds}")
        flush_print(f"Edges: {NUM_EDGES}")
        flush_print(f"Videos per edge: {VIDEOS_PER_EDGE}")
        flush_print(f"{'='*60}\n")

    def _run_aggregator(self):
        self.aggregator.start()

        flush_print(f"[Simulator] Aggregator started, edges connected")

        for r in range(self.num_rounds):
            self.aggregator.run_round(r)

        flush_print(f"\n[Simulator] All {self.num_rounds} rounds complete. Shutting down aggregator.")
        self.aggregator.shutdown()

    def _run_edge(self, edge_id: int):
        agent = EdgeAgent(edge_id=edge_id, aggregator_port=AGGREGATOR_PORT)
        agent.connect_to_aggregator()
        self.edges.append(agent)

    def simulate_arrivals(self, samples: List, delays: List[float]):
        for (img, label), delay in zip(samples, delays):
            if delay > 0:
                time.sleep(delay)

    def run(self):
        self.outage_periods = self.camera.schedule_outages(self.num_rounds)

        agg_thread = threading.Thread(target=self._run_aggregator, daemon=True)
        agg_thread.start()
        time.sleep(0.5)

        edge_threads = []
        for i in range(NUM_EDGES):
            t = threading.Thread(target=self._run_edge, args=(i,))
            edge_threads.append(t)
            t.start()
            time.sleep(0.2)

        for t in edge_threads:
            t.join(timeout=10)

        flush_print(f"\n[Simulator] All edges connected. Starting training...\n")

        for r in range(self.num_rounds):
            flush_print(f"\n[Simulator] {'='*50}")
            flush_print(f"[Simulator] ROUND {r}/{self.num_rounds - 1}")
            flush_print(f"[Simulator] {'='*50}")

            round_samples = {}
            round_info = {}

            for edge_id in range(NUM_EDGES):
                samples, is_outage = self.camera.get_data_for_edge(
                    edge_id, r, self.outage_periods
                )
                round_samples[edge_id] = samples
                round_info[edge_id] = is_outage

                self.simulate_arrivals(
                    samples,
                    [self.camera.sample_delay() for _ in samples]
                )

            edge_results = [None] * NUM_EDGES

            def run_edge_round(eid):
                edge_results[eid] = self.edges[eid].run_round(
                    round_samples[eid], round_info[eid]
                )

            round_threads = []
            for edge_id in range(NUM_EDGES):
                t = threading.Thread(target=run_edge_round, args=(edge_id,))
                round_threads.append(t)
                t.start()
            for t in round_threads:
                t.join()

            flush_print(f"\n[Simulator] Round {r} summary:")
            for edge_id, metrics in enumerate(edge_results):
                status = "OUTAGE" if round_info[edge_id] else "NORMAL"
                flush_print(f"  Edge {edge_id} [{status}]: "
                            f"loss={metrics.get('loss', 0):.4f}, "
                            f"acc={metrics.get('accuracy', 0):.4f}, "
                            f"samples={metrics.get('samples_trained', 0)}, "
                            f"cpu={metrics.get('cpu_avg', 0):.1f}%")

            time.sleep(0.5)

        flush_print(f"\n[Simulator] {'='*60}")
        flush_print(f"[Simulator] SIMULATION COMPLETE")
        flush_print(f"[Simulator] {'='*60}")

        for agent in self.edges:
            agent.shutdown()

        agg_thread.join(timeout=5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Digital Twin Federated Learning Simulator')
    parser.add_argument('--delay-model', type=str, default='kde',
                        choices=['kde', 'wgan'],
                        help='Interarrival delay model (kde or wgan)')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS,
                        help='Number of FL rounds')
    parser.add_argument('--max-frames', type=int, default=50,
                        help='Max frames per video (default: 50)')
    args = parser.parse_args()

    sim = Simulator(delay_model=args.delay_model, num_rounds=args.rounds)
    sim.camera.max_frames_per_video = args.max_frames
    sim.run()


if __name__ == '__main__':
    main()