"""
DISPATCH Hierarchical Multi-Tenant Token Buckets (Module 2).

Enforces per-business-unit Token-Per-Minute (TPM) and Request-Per-Minute (RPM) quotas
with priority-tiered preemption to eliminate noisy-neighbor degradation.

Priority Tiers:
  - TIER_0_CRITICAL: Customer-facing production agents (highest priority, protected).
  - TIER_1_INTERNAL: Internal employee workflows.
  - TIER_2_BATCH: Offline analytics and backfill jobs (lowest priority, preemptible).

Preemption Rule:
  When global pool utilization exceeds 80% (pool_usage >= 0.80 * global_capacity),
  all TIER_2_BATCH requests are immediately preempted with:
  "HTTP 429: Preempted by Tier-0"
"""

import collections
import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dispatch.quota")


class PriorityTier(str, enum.Enum):
    TIER_0_CRITICAL = "TIER_0_CRITICAL"
    TIER_1_INTERNAL = "TIER_1_INTERNAL"
    TIER_2_BATCH = "TIER_2_BATCH"


class QuotaExceededError(Exception):
    """Raised when tenant or tier quota is exhausted."""
    pass


class QuotaPreemptedError(QuotaExceededError):
    """Raised when lower-priority requests are preempted by high-priority load."""
    pass


@dataclass
class TenantQuotaConfig:
    tenant_id: str
    tpm_limit: int = 100_000
    rpm_limit: int = 300
    priority: PriorityTier = PriorityTier.TIER_1_INTERNAL


@dataclass
class UsageWindow:
    """Sliding 60-second window tracker for tokens and requests."""
    records: collections.deque = field(default_factory=collections.deque)  # (timestamp, tokens)

    def prune(self, now: float, window_seconds: float = 60.0) -> None:
        cutoff = now - window_seconds
        while self.records and self.records[0][0] <= cutoff:
            self.records.popleft()

    def add(self, now: float, tokens: int) -> None:
        self.records.append((now, tokens))

    def current_usage(self, now: float, window_seconds: float = 60.0) -> Tuple[int, int]:
        """Returns (tokens_used, requests_count) in the last window_seconds."""
        self.prune(now, window_seconds)
        tokens_sum = sum(rec[1] for rec in self.records)
        return tokens_sum, len(self.records)


class HierarchicalQuotaManager:
    """Hierarchical token-bucket quota manager with Tier-0 preemption protection."""

    def __init__(
        self,
        global_tpm_capacity: int = 200_000,
        preemption_threshold: float = 0.80,
        window_seconds: float = 60.0,
        default_tenant_tpm: int = 1_000_000,
        default_tenant_rpm: int = 10_000,
    ):
        self.global_tpm_capacity = global_tpm_capacity
        self.preemption_threshold = preemption_threshold
        self.window_seconds = window_seconds
        self.default_tenant_tpm = default_tenant_tpm
        self.default_tenant_rpm = default_tenant_rpm
        self._lock = threading.Lock()

        # Global usage tracker
        self._global_usage = UsageWindow()

        # Per-tenant usage trackers: tenant_id -> UsageWindow
        self._tenant_usage: Dict[str, UsageWindow] = {}

        # Tenant configs: tenant_id -> TenantQuotaConfig
        self._tenant_configs: Dict[str, TenantQuotaConfig] = {}

        # Telemetry counters
        self.total_admitted: int = 0
        self.tier2_preemptions: int = 0
        self.quota_rejections: int = 0

    def register_tenant(
        self,
        tenant_id: str,
        tpm_limit: int = 100_000,
        rpm_limit: int = 300,
        priority: PriorityTier = PriorityTier.TIER_1_INTERNAL,
    ) -> TenantQuotaConfig:
        with self._lock:
            cfg = TenantQuotaConfig(
                tenant_id=tenant_id,
                tpm_limit=tpm_limit,
                rpm_limit=rpm_limit,
                priority=priority,
            )
            self._tenant_configs[tenant_id] = cfg
            if tenant_id not in self._tenant_usage:
                self._tenant_usage[tenant_id] = UsageWindow()
            return cfg

    def get_global_utilization(self, now: Optional[float] = None) -> float:
        """Returns current global pool utilization fraction [0.0, 1.0+]."""
        now = now if now is not None else time.time()
        with self._lock:
            used_tokens, _ = self._global_usage.current_usage(now, self.window_seconds)
            return used_tokens / max(1, self.global_tpm_capacity)

    def check_and_consume(
        self,
        tenant_id: str,
        tokens: int,
        priority: Optional[PriorityTier] = None,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Check admission under hierarchical quotas and preemption rules.

        Returns:
            Tuple of (allowed: bool, reason_message: str)
        """
        now = now if now is not None else time.time()

        with self._lock:
            cfg = self._tenant_configs.get(
                tenant_id,
                TenantQuotaConfig(
                    tenant_id=tenant_id,
                    tpm_limit=self.default_tenant_tpm,
                    rpm_limit=self.default_tenant_rpm,
                    priority=priority or PriorityTier.TIER_1_INTERNAL,
                ),
            )
            effective_priority = priority or cfg.priority

            # 1. Check Global Preemption Threshold for TIER_2_BATCH
            global_used_tokens, _ = self._global_usage.current_usage(now, self.window_seconds)
            utilization = (global_used_tokens + tokens) / max(1, self.global_tpm_capacity)

            if effective_priority == PriorityTier.TIER_2_BATCH:
                # Preempt immediately if pool utilization crosses 80%
                if utilization >= self.preemption_threshold:
                    self.tier2_preemptions += 1
                    err_msg = "HTTP 429: Preempted by Tier-0"
                    logger.warning(
                        "Batch request preempted for tenant '%s' at %.1f%% global utilization",
                        tenant_id, utilization * 100
                    )
                    return False, err_msg

            # 2. Check Tenant-Level Quota Limits
            tenant_window = self._tenant_usage.setdefault(tenant_id, UsageWindow())
            tenant_tokens, tenant_reqs = tenant_window.current_usage(now, self.window_seconds)

            if (tenant_tokens + tokens) > cfg.tpm_limit:
                self.quota_rejections += 1
                return False, f"HTTP 429: Tenant '{tenant_id}' TPM limit exceeded ({tenant_tokens + tokens} > {cfg.tpm_limit})"

            if (tenant_reqs + 1) > cfg.rpm_limit:
                self.quota_rejections += 1
                return False, f"HTTP 429: Tenant '{tenant_id}' RPM limit exceeded ({tenant_reqs + 1} > {cfg.rpm_limit})"

            # 3. Check Global Capacity Cap (Tier-0 and Tier-1 can burst up to 100% capacity)
            if (global_used_tokens + tokens) > self.global_tpm_capacity and effective_priority != PriorityTier.TIER_0_CRITICAL:
                self.quota_rejections += 1
                return False, "HTTP 429: Global inference pool capacity saturated"

            # Admitted: Record consumption in global and tenant sliding windows
            self._global_usage.add(now, tokens)
            tenant_window.add(now, tokens)
            self.total_admitted += 1
            return True, "OK"

    def reset(self) -> None:
        """Reset all windows and counters (useful for test and benchmarking resets)."""
        with self._lock:
            self._global_usage = UsageWindow()
            self._tenant_usage.clear()
            self.total_admitted = 0
            self.tier2_preemptions = 0
            self.quota_rejections = 0
