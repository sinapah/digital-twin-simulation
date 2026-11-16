import time
from typing import Dict, Tuple


class OutageStatus:
    def __init__(self, is_outage: bool = False, duration: float = 0.0,
                 just_recovered: bool = False):
        self.is_outage = is_outage
        self.duration = duration
        self.just_recovered = just_recovered


class OutageDetector:
    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout = timeout_seconds
        self.last_seen: Dict[int, float] = {}
        self.outage_start: Dict[int, float] = {}
        self.was_in_outage: Dict[int, bool] = {}
        self.recovery_telemetry: Dict[int, dict] = {}

    def record_telemetry(self, edge_id: int, telemetry: dict = None):
        now = time.time()
        self.last_seen[edge_id] = now
        was = self.was_in_outage.get(edge_id, False)
        self.was_in_outage[edge_id] = False
        if was and edge_id in self.outage_start:
            duration = now - self.outage_start.pop(edge_id, now)
            self.recovery_telemetry[edge_id] = {
                'duration': duration,
                'recovery_time': now,
                'actual_telemetry': telemetry,
            }
        if edge_id in self.outage_start:
            del self.outage_start[edge_id]

    def check(self, edge_id: int, current_timestamp: float = None) -> OutageStatus:
        now = current_timestamp or time.time()
        last = self.last_seen.get(edge_id)
        if last is None:
            return OutageStatus(is_outage=True, duration=0.0)

        elapsed = now - last
        if elapsed > self.timeout and not self.was_in_outage.get(edge_id, False):
            self.outage_start[edge_id] = last
            self.was_in_outage[edge_id] = True
            return OutageStatus(is_outage=True, duration=elapsed)

        if elapsed > self.timeout:
            return OutageStatus(is_outage=True, duration=now - self.outage_start.get(edge_id, last))

        just_rec = self.recovery_telemetry.pop(edge_id, None)
        if just_rec is not None:
            return OutageStatus(is_outage=False, duration=0.0, just_recovered=True)

        return OutageStatus(is_outage=False, duration=0.0)

    def get_recovery_info(self, edge_id: int) -> dict:
        return self.recovery_telemetry.pop(edge_id, {})
