import subprocess
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregator import Aggregator
from camera_simulator import CameraSimulator, NUM_EDGES, VIDEOS_PER_EDGE
from digital_twin.digital_twin import DigitalTwin
from orchestrator.orchestrator import Orchestrator


AGGREGATOR_PORT = 5000
NUM_ROUNDS = 100
BASE_UDP_PORT = 7000
DT_TELEMETRY_PORT = 6000
ORCHESTRATOR_PORT = 6100
AGGREGATOR_CONTROL_PORT = 6200


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class Simulator:
    def __init__(self, delay_model: str = 'kde', num_rounds: int = NUM_ROUNDS,
                 max_frames: int = 50, max_crops_per_round: int = 0,
                 sender_fps: int = 500, mode: str = 'kde',
                 sender_hosts: list = None,
                 dt_mode: str = 'none'):
        self.delay_model = delay_model if mode != 'baseline' else 'kde'
        self.mode = mode
        self.dt_mode = dt_mode
        self.num_rounds = num_rounds
        self.max_frames = max_frames
        self.max_crops_per_round = max_crops_per_round
        self.sender_fps = sender_fps
        self.sender_hosts = sender_hosts

        self.camera = CameraSimulator(delay_model=self.delay_model)
        self.aggregator = Aggregator(
            port=AGGREGATOR_PORT,
            num_rounds=num_rounds,
            dt_port=DT_TELEMETRY_PORT,
            orchestrator_control_port=AGGREGATOR_CONTROL_PORT,
        )
        self.edge_processes = []
        self.sender_processes = []
        self.outage_periods = []
        self.output_dir = f'outputs/{mode}'

        self.digital_twin = None
        self.orchestrator = None

        flush_print(f"\n{'='*60}")
        flush_print(f"V6 Digital Twin Federated Learning Simulator")
        flush_print(f"{'='*60}")
        flush_print(f"Mode: {mode.upper()}")
        flush_print(f"Digital Twin: {dt_mode.upper()}")
        flush_print(f"Delay model: {self.delay_model.upper()}")
        flush_print(f"Rounds: {num_rounds}")
        flush_print(f"Edges: {NUM_EDGES}")
        flush_print(f"Videos per edge: {VIDEOS_PER_EDGE}")
        flush_print(f"Max frames per video: {max_frames}")
        if max_crops_per_round:
            flush_print(f"Max crops per round: {max_crops_per_round}")
        if sender_hosts:
            flush_print(f"Sender VMs: {sender_hosts}")
        else:
            flush_print(f"Senders: local subprocesses")
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

        if self.dt_mode != 'none':
            self.digital_twin = DigitalTwin(
                num_edges=NUM_EDGES,
                telemetry_port=DT_TELEMETRY_PORT,
                output_dir=f'outputs/dt',
                synthesis_mode=self.dt_mode if self.dt_mode in ('kde', 'wgan') else 'none',
            )
            self.digital_twin.start()
            time.sleep(0.5)

            self.orchestrator = Orchestrator(
                port=ORCHESTRATOR_PORT,
                aggregator_control_port=AGGREGATOR_CONTROL_PORT,
                output_dir='outputs/dt',
            )
            self.orchestrator.start()
            time.sleep(0.5)

        agg_thread = threading.Thread(target=self._run_aggregator, daemon=True)
        agg_thread.start()
        time.sleep(1.0)

        v6_dir = os.path.dirname(os.path.abspath(__file__))
        sender_script = os.path.join(v6_dir, 'sender.py')
        edge_script = os.path.join(v6_dir, 'edge_agent.py')
        venv_python = sys.executable

        if self.sender_hosts:
            flush_print(f"[Simulator] Using remote senders. Start each sender VM:")
            for i, host in enumerate(self.sender_hosts):
                udp_port = BASE_UDP_PORT + i * 2
                flush_print(f"  Sender {i} on {host}: "
                            f"./run.sh sender{i} --mode {self.mode} "
                            f"--udp-port {udp_port} --fps {self.sender_fps}")
            flush_print(f"[Simulator] Waiting for all edges to connect to their senders...\n")
        else:
            for i in range(NUM_EDGES):
                udp_port = BASE_UDP_PORT + i * 2
                sender_cores = f"{6 + i},{7 + i}"
                flush_print(f"[Simulator] Sender {i} on cores {sender_cores}, "
                            f"UDP {udp_port}, control {udp_port + 1}")
                extra = []
                if self.max_crops_per_round:
                    extra = ['--max-crops-per-round', str(self.max_crops_per_round)]
                proc = subprocess.Popen(
                    ['taskset', '-c', sender_cores, venv_python, sender_script,
                     '--edge-id', str(i),
                     '--mode', self.mode,
                     '--delay-model', self.delay_model,
                     '--udp-port', str(udp_port),
                     '--fps', str(self.sender_fps),
                     '--max-frames', str(self.max_frames),
                     '--rounds', str(self.num_rounds)] + extra,
                    stdout=sys.stdout, stderr=sys.stderr,
                    cwd=v6_dir,
                )
                self.sender_processes.append(proc)
                time.sleep(0.5)

            flush_print(f"[Simulator] Senders starting, edges will retry until ready...")
            time.sleep(2)

        for i in range(NUM_EDGES):
            udp_port = BASE_UDP_PORT + i * 2
            edge_cores = f"{i * 2},{i * 2 + 1}"
            sender_host = self.sender_hosts[i] if self.sender_hosts else '127.0.0.1'
            flush_print(f"[Simulator] Edge {i} on cores {edge_cores}, "
                        f"UDP port {udp_port}, sender at {sender_host}")
            proc = subprocess.Popen(
                ['taskset', '-c', edge_cores, venv_python, edge_script,
                 '--edge-id', str(i),
                 '--aggregator-port', str(AGGREGATOR_PORT),
                 '--udp-port', str(udp_port),
                 '--sender-host', sender_host,
                 '--output-dir', self.output_dir,
                 '--rounds', str(self.num_rounds)],
                stdout=sys.stdout, stderr=sys.stderr,
                cwd=v6_dir,
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

        if self.orchestrator:
            self.orchestrator.stop()
        if self.digital_twin:
            self.digital_twin.stop()

        flush_print(f"\n[Simulator] {'='*60}")
        flush_print(f"[Simulator] SIMULATION COMPLETE")
        flush_print(f"[Simulator] Metrics saved to {self.output_dir}/")
        if self.dt_mode != 'none':
            flush_print(f"[Simulator] DT metrics saved to outputs/dt/")
        flush_print(f"[Simulator] {'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='V6 Digital Twin Federated Learning Simulator')
    parser.add_argument('--mode', type=str, default='kde',
                        choices=['baseline', 'kde', 'wgan'],
                        help='baseline=no outages; kde/wgan=use fitted delays')
    parser.add_argument('--delay-model', type=str, default=None,
                        choices=['kde', 'wgan'],
                        help='Override delay model')
    parser.add_argument('--dt-mode', type=str, default='none',
                        choices=['none', 'kde', 'wgan'],
                        help='Digital Twin synthesis mode: none=disabled, '
                             'kde=KDE synthesis, wgan=WGAN synthesis')
    parser.add_argument('--rounds', type=int, default=NUM_ROUNDS)
    parser.add_argument('--max-frames', type=int, default=50)
    parser.add_argument('--max-crops-per-round', type=int, default=0)
    parser.add_argument('--sender-fps', type=int, default=500)
    parser.add_argument('--sender-hosts', type=str, nargs=3, default=None,
                        metavar=('HOST0', 'HOST1', 'HOST2'),
                        help='IPs of 3 remote sender VMs')
    args = parser.parse_args()

    delay_model = args.delay_model or ('kde' if args.mode in ('baseline', 'kde') else 'wgan')

    sim = Simulator(
        delay_model=delay_model,
        num_rounds=args.rounds,
        max_frames=args.max_frames,
        max_crops_per_round=args.max_crops_per_round,
        sender_fps=args.sender_fps,
        mode=args.mode,
        sender_hosts=args.sender_hosts,
        dt_mode=args.dt_mode,
    )
    sim.run()


if __name__ == '__main__':
    main()
