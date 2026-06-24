import subprocess
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregator import Aggregator
from camera_simulator import CameraSimulator, NUM_EDGES, VIDEOS_PER_EDGE


AGGREGATOR_PORT = 5000
NUM_ROUNDS = 100


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class Simulator:
    def __init__(self, delay_model: str = 'kde', num_rounds: int = NUM_ROUNDS,
                 max_frames: int = 50):
        self.delay_model = delay_model
        self.num_rounds = num_rounds
        self.max_frames = max_frames
        self.camera = CameraSimulator(delay_model=delay_model)
        self.aggregator = Aggregator(port=AGGREGATOR_PORT, num_rounds=num_rounds)
        self.edge_processes = []
        self.outage_periods = []
        self.output_dir = f'outputs/{delay_model}'

        flush_print(f"\n{'='*60}")
        flush_print(f"Digital Twin Federated Learning Simulator (subprocess edges)")
        flush_print(f"{'='*60}")
        flush_print(f"Delay model: {delay_model.upper()}")
        flush_print(f"Rounds: {num_rounds}")
        flush_print(f"Edges: {NUM_EDGES}")
        flush_print(f"Videos per edge: {VIDEOS_PER_EDGE}")
        flush_print(f"Max frames per video: {max_frames}")
        flush_print(f"{'='*60}\n")

    def _run_aggregator(self):
        self.aggregator.start()
        flush_print(f"[Simulator] Aggregator started")

        for r in range(self.num_rounds):
            outage_flags = [
                (s <= r <= e) for (s, e) in self.outage_periods
            ]
            edge_status = self.aggregator.run_round(r, outage_flags)

            flush_print(f"\n[Simulator] Round {r} summary:")
            for eid in range(NUM_EDGES):
                status = "OUTAGE" if outage_flags[eid] else "NORMAL"
                s = edge_status.get(eid, {})
                flush_print(f"  Edge {eid} [{status}]: "
                            f"loss={s.get('loss', 0):.4f}, "
                            f"acc={s.get('accuracy', 0):.4f}, "
                            f"samples={s.get('samples_trained', 0)}, "
                            f"cpu={s.get('cpu_avg', 0):.1f}%")

        flush_print(f"\n[Simulator] All {self.num_rounds} rounds complete. Shutting down.")
        self.aggregator.shutdown()

    def run(self):
        self.camera.max_frames_per_video = self.max_frames
        self.outage_periods = self.camera.schedule_outages(self.num_rounds)

        agg_thread = threading.Thread(target=self._run_aggregator, daemon=True)
        agg_thread.start()
        time.sleep(0.5)

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edge_agent.py')
        venv_python = sys.executable

        for i in range(NUM_EDGES):
            cores = f"{i * 2},{i * 2 + 1}"
            flush_print(f"[Simulator] Edge {i} pinned to cores {cores}")
            proc = subprocess.Popen(
                ['taskset', '-c', cores, venv_python, script,
                 '--edge-id', str(i),
                 '--aggregator-port', str(AGGREGATOR_PORT),
                 '--delay-model', self.delay_model,
                 '--max-frames', str(self.max_frames),
                 '--output-dir', self.output_dir,
                 '--rounds', str(self.num_rounds)],
                stdout=sys.stdout, stderr=sys.stderr
            )
            self.edge_processes.append(proc)
            time.sleep(0.3)

        flush_print(f"\n[Simulator] All {NUM_EDGES} edge subprocesses started\n")

        for proc in self.edge_processes:
            proc.wait()

        flush_print(f"\n[Simulator] All edge subprocesses finished")

        agg_thread.join(timeout=5)

        flush_print(f"\n[Simulator] {'='*60}")
        flush_print(f"[Simulator] SIMULATION COMPLETE")
        flush_print(f"[Simulator] Metrics saved to {self.output_dir}/")
        flush_print(f"[Simulator] {'='*60}")


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

    sim = Simulator(
        delay_model=args.delay_model,
        num_rounds=args.rounds,
        max_frames=args.max_frames,
    )
    sim.run()


if __name__ == '__main__':
    main()
