import hashlib
import hmac
from enum import IntEnum


class PriorityTier(IntEnum):
    TIER_0_CRITICAL = 0   # Customer-facing voice/transaction workflows
    TIER_1_INTERNAL = 1   # Internal SRE, CI evaluations, operational queues
    TIER_2_BATCH = 2      # Bulk backfills, offline analytics, sandbox jobs


class QuotaPreemptedError(Exception):
    """Raised when a lower-priority workload is shed to protect Tier-0 SLAs."""
    pass


class QuotaManager:
    def __init__(self, global_tpm_limit: int = 10_000_000):
        self.global_tpm_limit = global_tpm_limit
        self.tier_0_reserve_tpm = int(global_tpm_limit * 0.60)
        self.tier_2_preemption_threshold_pct = 80.0
        self.tier_1_preemption_threshold_pct = 92.0

    def check_and_consume(
        self,
        priority: PriorityTier,
        requested_tokens: int,
        current_global_utilization_pct: float,
    ) -> bool:
        if requested_tokens <= 0:
            raise ValueError("requested_tokens must be positive")

        if priority == PriorityTier.TIER_0_CRITICAL:
            if current_global_utilization_pct >= 100.0:
                raise QuotaPreemptedError("HTTP 429: Global pool 100% exhausted.")
            return True

        if priority == PriorityTier.TIER_1_INTERNAL:
            if current_global_utilization_pct >= self.tier_1_preemption_threshold_pct:
                raise QuotaPreemptedError(
                    f"HTTP 429: Tier-1 request throttled at {current_global_utilization_pct:.1f}% utilization."
                )
            return True

        if priority == PriorityTier.TIER_2_BATCH:
            if current_global_utilization_pct >= self.tier_2_preemption_threshold_pct:
                raise QuotaPreemptedError(
                    f"HTTP 429: Tier-2 batch job preempted at {current_global_utilization_pct:.1f}% utilization to protect Tier-0 SLA."
                )
            return True

        return True

    @staticmethod
    def compute_tenant_cache_key(tenant_secret: str, tenant_id: str, model_id: str, routing_key: str) -> str:
        """
        Salts cache keys with a tenant-specific HMAC secret so cross-tenant cache bleed
        is mathematically impossible under HIPAA, PCI-DSS, and SOC 2 boundaries.
        """
        message = f"{tenant_id}:{model_id}:{routing_key}".encode("utf-8")
        return hmac.new(tenant_secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
