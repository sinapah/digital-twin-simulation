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
BASE_UDP_PORT = 7000   # Edge i listens on 7000+i*2, sender control on 7001+i*2


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class Simulator:
    def __init__(self, delay_model: str = 'kde', num_rounds: int = NUM_ROUNDS,
                 max_frames: int = 50, max_crops_per_round: int = 0,
                 sender_fps: int = 500, mode: str = 'kde'):
        self.delay_model = delay_model if mode != 'baseline' else 'kde'
        self.mode = mode
        self.num_rounds = num_rounds
        self.max_frames = max_frames
        self.max_crops_per_round = max_crops_per_round
        self.sender_fps = sender_fps
        self.camera = CameraSimulator(delay_model=self.delay_model)
        self.aggregator = Aggregator(port=AGGREGATOR_PORT, num_rounds=num_rounds)
        self.edge_processes = []
        self.sender_processes = []
        self.outage_periods = []
        self.output_dir = f'outputs/{mode}'

        flush_print(f"\n{'='*60}")
        flush_print(f"Digital Twin Federated Learning Simulator")
        flush_print(f"{'='*60}")
        flush_print(f"Mode: {mode.upper()}")
        flush_print(f"Delay model: {self.delay_model.upper()}")
        flush_print(f"Rounds: {num_rounds}")
        flush_print(f"Edges: {NUM_EDGES}")
        flush_print(f"Videos per edge: {VIDEOS_PER_EDGE}")
        flush_print(f"Max frames per video: {max_frames}")
        if max_crops_per_round:
            flush_print(f"Max crops per round: {max_crops_per_round}")
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
        self.outage_periods = self.camera.schedule_outages(
            self.num_rounds, baseline=(self.mode == 'baseline')
        )

        agg_thread = threading.Thread(target=self._run_aggregator, daemon=True)
        agg_thread.start()
        time.sleep(0.5)

        v5_dir = os.path.dirname(os.path.abspath(__file__))
        sender_script = os.path.join(v5_dir, 'sender.py')
        edge_script = os.path.join(v5_dir, 'edge_agent.py')
        venv_python = sys.executable

        # Start senders first (they pre-load data, then wait for edge connections)
        for i in range(NUM_EDGES):
            udp_port = BASE_UDP_PORT + i * 2
            sender_cores = f"{6 + i},{7 + i}"   # cores 6-8 for senders
            flush_print(f"[Simulator] Sender {i} on cores {sender_cores}, "
                        f"UDP {udp_port}, control {udp_port + 1}")
            extra = []
            if self.max_crops_per_round:
                extra = ['--max-crops-per-round', str(self.max_crops_per_round)]
            proc = subprocess.Popen(
                ['taskset', '-c', sender_cores, venv_python, sender_script,
                 '--edge-id', str(i),
                 '--delay-model', self.delay_model,
                 '--udp-port', str(udp_port),
                 '--fps', str(self.sender_fps),
                 '--max-frames', str(self.max_frames),
                 '--rounds', str(self.num_rounds)] + extra,
                stdout=sys.stdout, stderr=sys.stderr,
                cwd=v5_dir,
            )
            self.sender_processes.append(proc)
            time.sleep(0.5)

        flush_print(f"[Simulator] Senders starting, edges will retry until ready...")
        time.sleep(2)

        # Start edges (each connects to its sender, then the aggregator)
        for i in range(NUM_EDGES):
            udp_port = BASE_UDP_PORT + i * 2
            edge_cores = f"{i * 2},{i * 2 + 1}"
            flush_print(f"[Simulator] Edge {i} on cores {edge_cores}, "
                        f"UDP port {udp_port}")
            proc = subprocess.Popen(
                ['taskset', '-c', edge_cores, venv_python, edge_script,
                 '--edge-id', str(i),
                 '--aggregator-port', str(AGGREGATOR_PORT),
                 '--udp-port', str(udp_port),
                 '--output-dir', self.output_dir,
                 '--rounds', str(self.num_rounds)],
                stdout=sys.stdout, stderr=sys.stderr,
                cwd=v5_dir,
            )
            self.edge_processes.append(proc)
            time.sleep(0.3)

        flush_print(f"\n[Simulator] All processes started\n")

        for proc in self.edge_processes:
            proc.wait()

        flush_print(f"\n[Simulator] All edge subprocesses finished")

        for proc in self.sender_processes:
            proc.terminate()

        agg_thread.join(timeout=5)

        flush_print(f"\n[Simulator] {'='*60}")
        flush_print(f"[Simulator] SIMULATION COMPLETE")
        flush_print(f"[Simulator] Metrics saved to {self.output_dir}/")
        flush_print(f"[Simulator] {'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Digital Twin Federated Learning Simulator')
    parser.add_argument('--mode', type=str, default='kde',
                        choices=['baseline', 'kde', 'wgan'],
                        help='baseline=no outages (collect real delays); '
                             'kde/wgan=use fitted synthetic delays with outages')
    parser.add_argument('--delay-model', type=str, default=None,
                        choices=['kde', 'wgan'],
                        help='Override delay model (default: matches --mode)')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS)
    parser.add_argument('--max-frames', type=int, default=50)
    parser.add_argument('--max-crops-per-round', type=int, default=0,
                        help='Limit crops per round per edge (0 = unlimited)')
    parser.add_argument('--sender-fps', type=int, default=500)
    args = parser.parse_args()

    delay_model = args.delay_model or ('kde' if args.mode in ('baseline', 'kde') else 'wgan')

    sim = Simulator(
        delay_model=delay_model,
        num_rounds=args.rounds,
        max_frames=args.max_frames,
        max_crops_per_round=args.max_crops_per_round,
        sender_fps=args.sender_fps,
        mode=args.mode,
    )
    sim.run()


if __name__ == '__main__':
    main()
