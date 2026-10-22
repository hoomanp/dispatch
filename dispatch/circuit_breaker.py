import time
from typing import List


class BudgetExceededError(Exception):
    """Raised when an agent workflow exceeds its maximum allowed dollar ceiling."""
    pass


class CircuitOpenError(Exception):
    """Raised when the primary provider circuit is OPEN and failover is triggered."""
    pass


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold_pct: float = 10.0,
        window_size: int = 20,
        recovery_timeout_sec: float = 30.0,
    ):
        self.failure_threshold_pct = failure_threshold_pct
        self.window_size = window_size
        self.recovery_timeout_sec = recovery_timeout_sec
        self.history: List[bool] = []
        self.state: str = "CLOSED"
        self.last_trip_time: float = 0.0

    def record_call(self, success: bool, latency_ms: float = 150.0, max_latency_sla_ms: float = 2000.0) -> None:
        effective_success = success and (latency_ms <= max_latency_sla_ms)
        self.history.append(effective_success)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) >= 5:
            failures = self.history.count(False)
            failure_rate_pct = (failures / len(self.history)) * 100.0
            if failure_rate_pct >= self.failure_threshold_pct:
                self.state = "OPEN"
                self.last_trip_time = time.time()
            elif self.state == "HALF-OPEN" and effective_success:
                self.state = "CLOSED"

    def allow_primary_request(self) -> bool:
        if self.state == "OPEN":
            if (time.time() - self.last_trip_time) >= self.recovery_timeout_sec:
                self.state = "HALF-OPEN"
                return True
            return False
        return True
