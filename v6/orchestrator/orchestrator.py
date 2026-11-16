import json
import socket
import struct
import threading
import time
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .control_actions import ControlAction, AppliedAction


ORCHESTRATOR_PORT = 6100
AGGREGATOR_CONTROL_PORT = 6200


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError('Connection closed')
        data += chunk
    return data


def send_msg(sock, msg_dict):
    data = json.dumps(msg_dict).encode()
    sock.sendall(len(data).to_bytes(8, 'big') + data)


def recv_msg(sock):
    try:
        size_data = recv_exact(sock, 8)
        msg_size = struct.unpack('>Q', size_data)[0]
        msg_data = recv_exact(sock, msg_size)
        return json.loads(msg_data.decode())
    except Exception:
        return None


class Orchestrator:
    def __init__(self, port: int = ORCHESTRATOR_PORT,
                 aggregator_control_port: int = AGGREGATOR_CONTROL_PORT,
                 output_dir: str = 'outputs/dt'):
        self.port = port
        self.aggregator_control_port = aggregator_control_port
        self.output_dir = output_dir
        self.running = False
        self.server = None
        self.server_thread = None
        self.aggregator_conn = None
        self.dt_conn = None
        self.applied_actions: list = []
        self.current_action = ControlAction.default()
        self.dt_recommendations = []

    def start(self):
        self.running = True
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', self.port))
        self.server.listen(5)
        self.server_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.server_thread.start()
        flush_print(f"[Orchestrator] Listening for DT on port {self.port}")
        self._connect_aggregator()

    def _connect_aggregator(self):
        try:
            self.aggregator_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.aggregator_conn.settimeout(10.0)
            self.aggregator_conn.connect(('127.0.0.1', self.aggregator_control_port))
            flush_print(f"[Orchestrator] Connected to aggregator control on port "
                        f"{self.aggregator_control_port}")
        except Exception as e:
            flush_print(f"[Orchestrator] Aggregator not available: {e}")
            self.aggregator_conn = None

    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.server.accept()
                flush_print(f"[Orchestrator] DT connected from {addr}")
                self.dt_conn = conn
                t = threading.Thread(target=self._handle_dt, args=(conn,), daemon=True)
                t.start()
            except Exception:
                if self.running:
                    time.sleep(0.5)

    def _handle_dt(self, conn: socket.socket):
        while self.running:
            try:
                msg = recv_msg(conn)
                if msg is None:
                    break
                if msg.get('type') == 'RECOMMENDATIONS':
                    self._on_recommendations(msg)
                elif msg.get('type') == 'SHUTDOWN':
                    break
            except Exception:
                break
        try:
            conn.close()
        except Exception:
            pass

    def _on_recommendations(self, msg: dict):
        recs = msg.get('recommendations', [])
        round_num = msg.get('round', -1)
        flush_print(f"[Orchestrator] Received {len(recs)} recommendations "
                    f"for round {round_num}")
        self.dt_recommendations = recs

        if recs:
            best = recs[0]
            action = self._recommendation_to_action(best)
            self.current_action = action
            self._apply_action(action, round_num)

    def _recommendation_to_action(self, rec: dict) -> ControlAction:
        parts = rec.get('description', '').split(', ')
        action = ControlAction()
        for part in parts:
            if 'epochs=' in part:
                action.local_epochs = int(part.split('=')[1])
            elif 'batch=' in part:
                action.batch_size = int(part.split('=')[1])
            elif 'sampling=' in part:
                action.frame_sampling_rate = float(part.split('=')[1])
        return action

    def _apply_action(self, action: ControlAction, round_num: int):
        if self.aggregator_conn is None:
            self._connect_aggregator()
        if self.aggregator_conn is None:
            return

        try:
            send_msg(self.aggregator_conn, {
                'type': 'CONTROL_ACTION',
                'round': round_num,
                'action': action.to_dict(),
            })
            ack = recv_msg(self.aggregator_conn)
            if ack:
                self.applied_actions.append(AppliedAction(
                    action=action,
                    round_number=round_num,
                    source='dt_recommendation',
                    predicted_accuracy=0.0,
                ))
                flush_print(f"[Orchestrator] Applied control action at round {round_num}: "
                            f"epochs={action.local_epochs}, batch={action.batch_size}, "
                            f"sampling={action.frame_sampling_rate}")
        except Exception as e:
            flush_print(f"[Orchestrator] Error applying action: {e}")
            self.aggregator_conn = None

    def _write_action_log(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'orchestrator_actions.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['round', 'source', 'local_epochs', 'batch_size',
                        'frame_sampling_rate', 'predicted_accuracy', 'actual_accuracy'])
            for a in self.applied_actions:
                w.writerow([
                    a.round_number, a.source,
                    a.action.local_epochs or '',
                    a.action.batch_size or '',
                    a.action.frame_sampling_rate or '',
                    a.predicted_accuracy, a.actual_accuracy,
                ])
        flush_print(f"[Orchestrator] Action log saved to {path}")

    def stop(self):
        self.running = False
        self._write_action_log()
        if self.dt_conn:
            try:
                send_msg(self.dt_conn, {'type': 'SHUTDOWN'})
                self.dt_conn.close()
            except Exception:
                pass
        if self.aggregator_conn:
            try:
                self.aggregator_conn.close()
            except Exception:
                pass
        if self.server:
            try:
                self.server.close()
            except Exception:
                pass
        flush_print(f"[Orchestrator] Stopped")
