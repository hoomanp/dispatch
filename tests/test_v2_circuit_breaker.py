"""
Unit tests for DISPATCH Module 3: Sub-25ms Circuit Breaker & Failover.
"""

import time
import pytest

from dispatch import (
    BudgetExceededError,
    CircuitBreaker,
    CircuitBreakerTrippedError,
    CircuitState,
    ExecutionOutcome,
    ProviderHealthTracker,
)


def test_circuit_trips_to_open_on_failure_rate_exceeding_10_percent():
    tracker = ProviderHealthTracker(
        provider_id="frontier-primary",
        window_size=20,
        failure_threshold_rate=0.10,
    )

    # 17 successes
    for _ in range(17):
        tracker.record_call(ExecutionOutcome(status_code=200, ttft_ms=100.0))
        assert tracker.state == CircuitState.CLOSED

    # 3 failures (HTTP 500, HTTP 429, and TTFT > 2000ms SLA breach)
    # 3/20 = 15% > 10% threshold -> trips to OPEN
    tracker.record_call(ExecutionOutcome(status_code=500))
    tracker.record_call(ExecutionOutcome(status_code=429))
    tracker.record_call(ExecutionOutcome(status_code=200, ttft_ms=2500.0))  # TTFT breach

    assert tracker.state == CircuitState.OPEN
    assert tracker.is_available() is False


def test_automatic_failover_to_warm_fallback():
    cb = CircuitBreaker(
        providers=["frontier-primary", "frontier-fallback"],
        fallback_map={"frontier-primary": ["frontier-fallback"]},
        window_size=20,
        failure_threshold_rate=0.10,
    )

    # Initially routes to primary
    p, is_fb = cb.resolve_provider("frontier-primary")
    assert p == "frontier-primary"
    assert is_fb is False

    # Trip primary circuit
    primary_tracker = cb.get_tracker("frontier-primary")
    for _ in range(15):
        primary_tracker.record_call(ExecutionOutcome(status_code=200))
    for _ in range(5):
        primary_tracker.record_call(ExecutionOutcome(status_code=503))

    assert primary_tracker.state == CircuitState.OPEN

    # Subsequent resolution immediately reroutes to fallback
    resolved, is_fb = cb.resolve_provider("frontier-primary")
    assert resolved == "frontier-fallback"
    assert is_fb is True


def test_all_providers_tripped_raises_error():
    cb = CircuitBreaker(
        providers=["primary", "fallback"],
        fallback_map={"primary": ["fallback"]},
        window_size=20,
    )

    # Trip both
    for name in ("primary", "fallback"):
        tracker = cb.get_tracker(name)
        for _ in range(20):
            tracker.record_call(ExecutionOutcome(status_code=500))

    with pytest.raises(CircuitBreakerTrippedError):
        cb.resolve_provider("primary")


def test_workflow_budget_enforcement():
    cb = CircuitBreaker(default_max_cost_per_workflow=1.50)
    wf_id = "agent-wf-runaway-01"

    # Step 1: $0.50 (cumulative $0.50)
    cb.check_workflow_budget(wf_id, estimated_step_cost=0.50)
    cb.record_step_spend(wf_id, 0.50)

    # Step 2: $0.50 (cumulative $1.00)
    cb.check_workflow_budget(wf_id, estimated_step_cost=0.50)
    cb.record_step_spend(wf_id, 0.50)

    # Step 3: $0.60 projected (cumulative would be $1.60 > $1.50)
    # Must raise BudgetExceededError and terminate loop
    with pytest.raises(BudgetExceededError) as exc_info:
        cb.check_workflow_budget(wf_id, estimated_step_cost=0.60)

    assert "breached financial ceiling" in str(exc_info.value)
