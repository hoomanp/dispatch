import pytest
from dispatch.quota import PriorityTier, QuotaManager, QuotaPreemptedError


def test_tier2_preempted_above_80_percent_while_tier0_passes():
    qm = QuotaManager(global_tpm_limit=10_000_000)

    # At 85% pool utilization, Tier-0 Critical passes while Tier-2 Batch is shed
    assert qm.check_and_consume(PriorityTier.TIER_0_CRITICAL, 25000, 85.0) is True
    with pytest.raises(QuotaPreemptedError):
        qm.check_and_consume(PriorityTier.TIER_2_BATCH, 25000, 85.0)

    # At 50% pool utilization, Tier-2 Batch is permitted to burst into idle capacity
    assert qm.check_and_consume(PriorityTier.TIER_2_BATCH, 25000, 50.0) is True


def test_cryptographic_tenant_cache_isolation():
    key_tenant_a = QuotaManager.compute_tenant_cache_key("secret-A", "tenant-a", "frontier", "hash123")
    key_tenant_b = QuotaManager.compute_tenant_cache_key("secret-B", "tenant-b", "frontier", "hash123")
    assert key_tenant_a != key_tenant_b
