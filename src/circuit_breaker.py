"""
DISPATCH Sub-25ms Circuit Breaker & Failover Engine (Module 3).

Enforces low-latency resilience and runaway cost containment:
1. Sliding-window health tracking across provider response codes (HTTP 429, 500, 502, 503, 529)
   and performance degradation (TTFT > 2000ms).
2. Fast circuit tripping: transitions to OPEN when failure rate exceeds 10% over 20 requests.
3. Zero-wait fallback failover: reroutes immediately to warm backup providers when tripped.
4. Hard workflow blast-radius containment: enforces max_cost_per_workflow ($1.50 default)
   to terminate runaway tool/agent retry loops.
"""

import collections
import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("dispatch.circuit_breaker")


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class BudgetExceededError(Exception):
    """Raised when an agent workflow exceeds its cumulative financial ceiling."""
    pass


class CircuitBreakerTrippedError(Exception):
    """Raised when all primary and fallback providers have tripped circuits."""
    pass


@dataclass
class ExecutionOutcome:
    status_code: int
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    is_failure: bool = False


class ProviderHealthTracker:
    """Sliding-window health and degradation monitor for a single provider/model endpoint."""

    def __init__(
        self,
        provider_id: str,
        window_size: int = 20,
        failure_threshold_rate: float = 0.10,
        recovery_timeout_seconds: float = 10.0,
        ttft_sla_ms: float = 2000.0,
    ):
        self.provider_id = provider_id
        self.window_size = window_size
        self.failure_threshold_rate = failure_threshold_rate
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.ttft_sla_ms = ttft_sla_ms

        self.state = CircuitState.CLOSED
        self._history: Deque[ExecutionOutcome] = collections.deque(maxlen=window_size)
        self._last_state_change: float = time.time()
        self._half_open_probes: int = 0
        self._lock = threading.Lock()

    def record_call(self, outcome: ExecutionOutcome) -> None:
        """Record provider call outcome and update circuit state."""
        # Classify failure: HTTP 429, 500, 502, 503, 529, or TTFT > 2000ms
        is_fail = (
            outcome.is_failure
            or outcome.status_code in (429, 500, 502, 503, 529)
            or (outcome.ttft_ms > self.ttft_sla_ms)
        )
        outcome.is_failure = is_fail

        with self._lock:
            self._history.append(outcome)
            now = time.time()

            if self.state == CircuitState.HALF_OPEN:
                if is_fail:
                    # Probe failed, trip back to OPEN
                    self.state = CircuitState.OPEN
                    self._last_state_change = now
                    logger.warning("Provider '%s' half-open probe failed -> OPEN", self.provider_id)
                else:
                    self._half_open_probes += 1
                    if self._half_open_probes >= 3:
                        self.state = CircuitState.CLOSED
                        self._history.clear()
                        self._last_state_change = now
                        logger.info("Provider '%s' recovered -> CLOSED", self.provider_id)
                return

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (now - self._last_state_change) >= self.recovery_timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_probes = 0
                    self._last_state_change = now
                    logger.info("Provider '%s' recovery timeout elapsed -> HALF_OPEN", self.provider_id)
                return

            # Circuit is CLOSED: evaluate sliding window failure rate
            if len(self._history) >= self.window_size:
                failures = sum(1 for call in self._history if call.is_failure)
                failure_rate = failures / len(self._history)
                if failure_rate > self.failure_threshold_rate:
                    self.state = CircuitState.OPEN
                    self._last_state_change = now
                    logger.warning(
                        "Circuit TRIPPED for provider '%s': %.1f%% failures over %d calls",
                        self.provider_id, failure_rate * 100, len(self._history)
                    )

    def is_available(self) -> bool:
        """Check if provider is available to take traffic (CLOSED or ready for HALF_OPEN probe)."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if (time.time() - self._last_state_change) >= self.recovery_timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_probes = 0
                    self._last_state_change = time.time()
                    return True
                return False
            if self.state == CircuitState.HALF_OPEN:
                return self._half_open_probes < 3
            return False

    @property
    def current_failure_rate(self) -> float:
        with self._lock:
            if not self._history:
                return 0.0
            failures = sum(1 for call in self._history if call.is_failure)
            return failures / len(self._history)


class CircuitBreaker:
    """Multi-provider failover router and per-workflow budget sentinel."""

    def __init__(
        self,
        providers: Optional[List[str]] = None,
        fallback_map: Optional[Dict[str, List[str]]] = None,
        default_max_cost_per_workflow: float = 1.50,
        window_size: int = 20,
        failure_threshold_rate: float = 0.10,
    ):
        self.default_max_cost_per_workflow = default_max_cost_per_workflow
        self.fallback_map = fallback_map or {
            "frontier-primary": ["frontier-fallback", "compact-primary"],
            "compact-primary": ["compact-fallback", "frontier-primary"],
        }
        self._trackers: Dict[str, ProviderHealthTracker] = {}
        all_providers = set(providers or ["frontier-primary", "frontier-fallback", "compact-primary", "compact-fallback"])
        for p in all_providers:
            self._trackers[p] = ProviderHealthTracker(
                provider_id=p,
                window_size=window_size,
                failure_threshold_rate=failure_threshold_rate,
            )

        # Per-workflow financial tracker: workflow_id -> cumulative_cost_usd
        self._workflow_spend: Dict[str, float] = collections.defaultdict(float)
        self._lock = threading.Lock()

    def get_tracker(self, provider_id: str) -> ProviderHealthTracker:
        if provider_id not in self._trackers:
            with self._lock:
                if provider_id not in self._trackers:
                    self._trackers[provider_id] = ProviderHealthTracker(provider_id=provider_id)
        return self._trackers[provider_id]

    def resolve_provider(self, target_provider: str) -> Tuple[str, bool]:
        """Resolve target provider; if tripped, failover to the first healthy fallback.

        Returns:
            Tuple of (effective_provider_id: str, is_failover: bool)
        """
        primary_tracker = self.get_tracker(target_provider)
        if primary_tracker.is_available():
            return target_provider, False

        # Target provider is tripped -> search fallback chain
        fallbacks = self.fallback_map.get(target_provider, [])
        for fb in fallbacks:
            fb_tracker = self.get_tracker(fb)
            if fb_tracker.is_available():
                logger.warning(
                    "Primary provider '%s' circuit is %s. Failing over to '%s'",
                    target_provider, primary_tracker.state.value, fb
                )
                return fb, True

        raise CircuitBreakerTrippedError(
            f"All providers in fallback chain for '{target_provider}' are TRIPPED"
        )

    def check_workflow_budget(
        self,
        workflow_id: str,
        estimated_step_cost: float = 0.0,
        max_budget_usd: Optional[float] = None,
    ) -> None:
        """Enforce hard ceiling on cumulative workflow cost to prevent runaway loops."""
        budget_limit = max_budget_usd or self.default_max_cost_per_workflow
        with self._lock:
            current_spend = self._workflow_spend[workflow_id]
            projected = current_spend + estimated_step_cost
            if projected > budget_limit:
                raise BudgetExceededError(
                    f"Workflow '{workflow_id}' breached financial ceiling: "
                    f"projected ${projected:.4f} > limit ${budget_limit:.2f}"
                )

    def record_step_spend(self, workflow_id: str, cost_usd: float) -> float:
        """Record step spend and return cumulative spend for workflow."""
        with self._lock:
            self._workflow_spend[workflow_id] += cost_usd
            return self._workflow_spend[workflow_id]

    def get_workflow_spend(self, workflow_id: str) -> float:
        with self._lock:
            return self._workflow_spend.get(workflow_id, 0.0)

    def clear_workflow(self, workflow_id: str) -> None:
        with self._lock:
            self._workflow_spend.pop(workflow_id, None)
