import pytest
from dispatch.circuit_breaker import BudgetExceededError, CircuitBreaker
from dispatch.router import GatewayClient


def test_circuit_trips_open_on_error_rate():
    cb = CircuitBreaker(failure_threshold_pct=20.0, window_size=10)
    for _ in range(8):
        cb.record_call(success=True, latency_ms=120.0)
    for _ in range(2):
        cb.record_call(success=False, latency_ms=2500.0)

    assert cb.state == "OPEN"
    assert cb.allow_primary_request() is False


def test_workflow_budget_cap_enforced():
    client = GatewayClient(tenant_id="finops-test")
    steps = [{"prefix_tokens": 20000, "dynamic_input_tokens": 2000, "output_tokens": 1000} for _ in range(10)]
    with pytest.raises(BudgetExceededError):
        client.execute_workflow(
            workflow_id="over-budget-wf",
            system_prefix="Policy",
            tools=[],
            session_context={},
            steps=steps,
            policy_mode="naive_unstratified",
            max_workflow_budget_usd=0.50,
        )
