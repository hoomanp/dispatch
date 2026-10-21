"""
Unit tests for DISPATCH Module 2: Hierarchical Multi-Tenant Token Buckets.
"""

import time
import pytest

from dispatch import (
    HierarchicalQuotaManager,
    PriorityTier,
    QuotaExceededError,
    QuotaPreemptedError,
)


def test_normal_token_consumption():
    qm = HierarchicalQuotaManager(global_tpm_capacity=100_000, preemption_threshold=0.80)
    qm.register_tenant("prod-payments", tpm_limit=50_000, rpm_limit=100, priority=PriorityTier.TIER_0_CRITICAL)

    allowed, reason = qm.check_and_consume("prod-payments", tokens=5_000)
    assert allowed is True
    assert reason == "OK"
    assert qm.total_admitted == 1


def test_noisy_neighbor_tier2_preemption_at_80_percent():
    """When pool utilization exceeds 80%, TIER_2_BATCH requests are preempted."""
    qm = HierarchicalQuotaManager(global_tpm_capacity=100_000, preemption_threshold=0.80)
    qm.register_tenant("analytics-bu", tpm_limit=100_000, priority=PriorityTier.TIER_2_BATCH)
    qm.register_tenant("prod-auth", tpm_limit=100_000, priority=PriorityTier.TIER_0_CRITICAL)

    # 1. Fill pool to 75% (75,000 tokens)
    allowed, _ = qm.check_and_consume("prod-auth", tokens=75_000)
    assert allowed is True

    # 2. Batch request for 10,000 tokens would push pool to 85,000 (85% > 80%)
    # Must be preempted immediately with "HTTP 429: Preempted by Tier-0"
    allowed_batch, reason_batch = qm.check_and_consume("analytics-bu", tokens=10_000, priority=PriorityTier.TIER_2_BATCH)
    assert allowed_batch is False
    assert "Preempted by Tier-0" in reason_batch
    assert qm.tier2_preemptions == 1

    # 3. But a critical Tier-0 request at 75% pool CAN still proceed up to capacity
    allowed_t0, reason_t0 = qm.check_and_consume("prod-auth", tokens=15_000, priority=PriorityTier.TIER_0_CRITICAL)
    assert allowed_t0 is True
    assert reason_t0 == "OK"


def test_tenant_tpm_quota_limit():
    qm = HierarchicalQuotaManager(global_tpm_capacity=500_000)
    qm.register_tenant("test-team", tpm_limit=10_000, rpm_limit=50)

    # First call: 8,000 tokens (allowed)
    ok1, _ = qm.check_and_consume("test-team", tokens=8_000)
    assert ok1 is True

    # Second call: 3,000 tokens (would reach 11,000 > 10,000 limit -> rejected)
    ok2, reason2 = qm.check_and_consume("test-team", tokens=3_000)
    assert ok2 is False
    assert "TPM limit exceeded" in reason2


def test_sliding_window_expiration():
    qm = HierarchicalQuotaManager(global_tpm_capacity=100_000, window_seconds=1.0)
    qm.register_tenant("burst-team", tpm_limit=5_000)

    now = 1000.0
    ok, _ = qm.check_and_consume("burst-team", tokens=5_000, now=now)
    assert ok is True

    # Next immediate check: rejected
    ok_imm, _ = qm.check_and_consume("burst-team", tokens=1_000, now=now + 0.1)
    assert ok_imm is False

    # After 1.1s window expires: allowed again
    ok_after, _ = qm.check_and_consume("burst-team", tokens=2_000, now=now + 1.2)
    assert ok_after is True
