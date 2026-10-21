"""
Unit tests for DISPATCH Module 4: FinOps Chargeback & Overhead Telemetry.
"""

import pytest

from dispatch import FinOpsRecord, TelemetryCollector


def test_mandatory_telemetry_headers_validation():
    tc = TelemetryCollector()

    headers = {
        "X-Tenant-BU": "sec-ops-prod",
        "X-Workflow-Type": "incident-mitigation",
        "X-Environment": "production",
    }
    validated = tc.validate_headers(headers)
    assert validated["business_unit"] == "sec-ops-prod"
    assert validated["workflow_type"] == "incident-mitigation"
    assert validated["environment"] == "production"

    # Default fallback when headers missing
    fallback = tc.validate_headers({})
    assert fallback["business_unit"] == "default-bu"
    assert fallback["environment"] == "production"


def test_finops_record_emission_and_aggregates():
    tc = TelemetryCollector()

    # Step 1: Cold start
    tc.record_turn(FinOpsRecord(
        tenant_id="tenant-alpha",
        workflow_id="wf-101",
        uncached_tokens=22_000,
        cached_read_tokens=0,
        output_tokens=1_000,
        effective_cost_usd=0.128,
        cache_hit_ratio=0.0,
        gateway_overhead_ms=8.5,
        model_tier="frontier",
    ))

    # Steps 2-10: Warm reads (9 steps)
    for _ in range(9):
        tc.record_turn(FinOpsRecord(
            tenant_id="tenant-alpha",
            workflow_id="wf-101",
            uncached_tokens=2_000,
            cached_read_tokens=20_000,
            output_tokens=1_000,
            effective_cost_usd=0.032,
            cache_hit_ratio=0.909,
            gateway_overhead_ms=7.8,
            model_tier="frontier",
        ))

    summary = tc.get_finops_summary("tenant-alpha")
    assert summary["total_records"] == 10
    assert summary["total_workflows"] == 1
    # Total cost = 0.128 + 9 * 0.032 = 0.416
    assert abs(summary["effective_cost_per_workflow"] - 0.416) < 1e-3
    assert summary["tier_distribution"]["frontier_pct"] == 100.0


def test_gateway_overhead_target_under_15ms():
    tc = TelemetryCollector()
    samples = [4.2, 5.1, 7.3, 8.2, 9.1, 10.5, 11.4, 12.0, 13.5]
    for s in samples:
        tc.record_turn(FinOpsRecord(
            tenant_id="t1",
            workflow_id="w1",
            uncached_tokens=1000,
            cached_read_tokens=1000,
            output_tokens=500,
            effective_cost_usd=0.01,
            cache_hit_ratio=0.5,
            gateway_overhead_ms=s,
        ))

    p99 = tc.get_percentile_overhead(99.0)
    assert p99 < 15.0
