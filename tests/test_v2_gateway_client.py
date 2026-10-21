"""
Unit tests for DISPATCH GatewayClient SDK and end-to-end integration.
"""

import pytest

from dispatch import GatewayClient, PriorityTier, QuotaPreemptedError


def test_gateway_client_paved_road_execution():
    SYSTEM_POLICY = "You are a cloud operations assistant. Format output as JSON."
    MCP_TOOL_SCHEMAS = [
        {"name": "get_metrics", "parameters": {"service": "payments"}},
        {"name": "check_logs", "parameters": {"severity": "error"}},
    ]
    turn_history = [{"role": "user", "content": "Analyze payments service degradation"}]

    client = GatewayClient(tenant_id="sec-ops-prod", priority=PriorityTier.TIER_0_CRITICAL)
    response = client.execute_step(
        workflow_id="inc-8841",
        system_prefix=SYSTEM_POLICY,
        tools=MCP_TOOL_SCHEMAS,
        messages=turn_history,
        max_workflow_budget_usd=1.00,
    )

    assert response.status == "SUCCESS"
    assert response.workflow_id == "inc-8841"
    assert response.tenant_id == "sec-ops-prod"
    assert response.prefix_hash is not None
    assert response.cost_usd > 0.0
    assert response.gateway_overhead_ms > 0.0


def test_gateway_client_multi_turn_cache_reuse():
    client = GatewayClient(tenant_id="cloud-ops", priority=PriorityTier.TIER_1_INTERNAL)

    system_prefix = "Always adhere to company security policy."
    tools = [{"name": "fetch_trace"}]

    # Step 1: Initial turn
    r1 = client.execute_step(
        workflow_id="wf-cache-test",
        system_prefix=system_prefix,
        tools=tools,
        messages=[{"role": "user", "content": "Step 1"}],
        step_index=1,
        policy="prefix_cached_frontier",
    )
    # Cold write on step 1
    assert r1.cache_hit is False

    # Step 2: Next turn with warm prefix
    r2 = client.execute_step(
        workflow_id="wf-cache-test",
        system_prefix=system_prefix,
        tools=tools,
        messages=[
            {"role": "user", "content": "Step 1"},
            {"role": "assistant", "content": "Result 1"},
            {"role": "user", "content": "Step 2"},
        ],
        step_index=2,
        policy="prefix_cached_frontier",
    )
    # Step 2 should hit the warm cache!
    assert r2.cache_hit is True
    assert r2.prefix_hash == r1.prefix_hash
