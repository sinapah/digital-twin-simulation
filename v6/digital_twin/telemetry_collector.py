import json
import socket
import struct
import threading
import time
from typing import Optional


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError('Connection closed')
        data += chunk
    return data


class TelemetryCollector:
    def __init__(self, host: str = '127.0.0.1', port: int = 6000):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
        self.callback = None
        self.connections = []

    def on_telemetry(self, callback):
        self.callback = callback

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.running = True
        self.server_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.server_thread.start()
        print(f"[TelemetryCollector] Listening for telemetry on {self.host}:{self.port}")

    def stop(self):
        self.running = False
        for c in self.connections:
            try:
                c.close()
            except Exception:
                pass
        if self.server:
            try:
                self.server.close()
            except Exception:
                pass
        print("[TelemetryCollector] Stopped")

    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.server.accept()
                print(f"[TelemetryCollector] Aggregator connected from {addr}")
                self.connections.append(conn)
                t = threading.Thread(target=self._handle_connection,
                                     args=(conn,), daemon=True)
                t.start()
            except Exception:
                if self.running:
                    time.sleep(0.5)

    def _handle_connection(self, conn: socket.socket):
        buf = b''
        while self.running:
            try:
                size_data = recv_exact(conn, 8)
                msg_size = struct.unpack('>Q', size_data)[0]
                msg_data = recv_exact(conn, msg_size)
                msg = json.loads(msg_data.decode())
                if self.callback:
                    self.callback(msg)
            except (ConnectionError, json.JSONDecodeError):
                print("[TelemetryCollector] Aggregator disconnected")
                break
        try:
            conn.close()
        except Exception:
            pass
