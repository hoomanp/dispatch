import time
from typing import Any, Dict, List
from .canonicalizer import canonicalize_payload
from .circuit_breaker import BudgetExceededError, CircuitBreaker
from .quota import PriorityTier, QuotaManager
from .telemetry import FinOpsLedger

# Representative Enterprise Pricing ($ per 1,000,000 tokens)
PRICING_TABLE = {
    "frontier": {
        "input_uncached": 4.00,
        "cache_write": 5.00,
        "cache_read": 0.20,
        "output": 20.00,
    },
    "compact": {
        "input_uncached": 0.30,
        "cache_write": 0.35,
        "cache_read": 0.03,
        "output": 1.20,
    },
}


class GatewayClient:
    def __init__(
        self,
        tenant_id: str,
        priority: PriorityTier = PriorityTier.TIER_0_CRITICAL,
        tenant_secret: str = "enterprise-default-hmac-key",
    ):
        self.tenant_id = tenant_id
        self.priority = priority
        self.tenant_secret = tenant_secret
        self.quota_manager = QuotaManager()
        self.circuit_breaker = CircuitBreaker()
        self.ledger = FinOpsLedger()

    def execute_workflow(
        self,
        workflow_id: str,
        system_prefix: str,
        tools: List[Dict[str, Any]],
        session_context: Dict[str, Any],
        steps: List[Dict[str, Any]],
        policy_mode: str = "dispatch_full",
        global_utilization_pct: float = 45.0,
        max_workflow_budget_usd: float = 1.50,
    ) -> Dict[str, Any]:
        """
        Executes a multi-step agent workflow under one of three gateway policies:
        - 'naive_unstratified': Uncached frontier on every step (round-robin, no prefix reuse).
        - 'sticky_frontier': Canonicalized 3-block prefix + sticky routing on Frontier tier.
        - 'dispatch_full': Canonicalized prefix + capability routing (70% compact / 30% frontier) + quota guard.
        """
        total_cost_usd = 0.0
        total_input_tokens = 0
        total_cached_read_tokens = 0
        step_overheads_ms: List[float] = []
        routing_key = ""

        for idx, step in enumerate(steps):
            t0 = time.perf_counter()

            prefix_tokens = int(step.get("prefix_tokens", 20000))
            dynamic_input_tokens = int(step.get("dynamic_input_tokens", 2000))
            output_tokens = int(step.get("output_tokens", 1000))
            complexity = str(step.get("complexity", "compact"))

            # 1. Hierarchical Quota Check
            self.quota_manager.check_and_consume(
                priority=self.priority,
                requested_tokens=prefix_tokens + dynamic_input_tokens + output_tokens,
                current_global_utilization_pct=global_utilization_pct,
            )

            # 2. Canonicalize Payload & Compute Tenant-Isolated Sticky Key
            canonical = canonicalize_payload(
                system_prefix=system_prefix,
                tools=tools,
                session_context=session_context,
                dynamic_turns=[{"role": "user", "content": step.get("prompt", "step")}],
            )
            routing_key = self.quota_manager.compute_tenant_cache_key(
                self.tenant_secret, self.tenant_id, policy_mode, canonical["sticky_routing_key"]
            )

            # 3. Select Model Tier & Compute Step Economics
            if policy_mode == "naive_unstratified":
                rates = PRICING_TABLE["frontier"]
                step_cost = (
                    ((prefix_tokens + dynamic_input_tokens) * rates["input_uncached"])
                    + (output_tokens * rates["output"])
                ) / 1_000_000.0
                cached_tokens_this_step = 0

            elif policy_mode == "sticky_frontier":
                rates = PRICING_TABLE["frontier"]
                if idx == 0:
                    # Step 1 writes the 20,000-token prefix to KV cache
                    step_cost = (
                        (prefix_tokens * rates["cache_write"])
                        + (dynamic_input_tokens * rates["input_uncached"])
                        + (output_tokens * rates["output"])
                    ) / 1_000_000.0
                    cached_tokens_this_step = 0
                else:
                    # Steps 2..N read prefix from KV cache at $0.20/M
                    step_cost = (
                        (prefix_tokens * rates["cache_read"])
                        + (dynamic_input_tokens * rates["input_uncached"])
                        + (output_tokens * rates["output"])
                    ) / 1_000_000.0
                    cached_tokens_this_step = prefix_tokens

            elif policy_mode == "dispatch_full":
                # Circuit breaker check: fallback to compact if primary frontier is degraded
                primary_healthy = self.circuit_breaker.allow_primary_request()
                tier = "frontier" if (complexity == "frontier" and primary_healthy) else "compact"
                rates = PRICING_TABLE[tier]

                if idx == 0:
                    step_cost = (
                        (prefix_tokens * rates["cache_write"])
                        + (dynamic_input_tokens * rates["input_uncached"])
                        + (output_tokens * rates["output"])
                    ) / 1_000_000.0
                    cached_tokens_this_step = 0
                else:
                    step_cost = (
                        (prefix_tokens * rates["cache_read"])
                        + (dynamic_input_tokens * rates["input_uncached"])
                        + (output_tokens * rates["output"])
                    ) / 1_000_000.0
                    cached_tokens_this_step = prefix_tokens
            else:
                raise ValueError(f"Unknown policy_mode: {policy_mode}")

            total_cost_usd += step_cost
            total_input_tokens += prefix_tokens + dynamic_input_tokens
            total_cached_read_tokens += cached_tokens_this_step

            if total_cost_usd > max_workflow_budget_usd:
                raise BudgetExceededError(
                    f"Workflow {workflow_id} cost ${total_cost_usd:.3f} exceeded cap ${max_workflow_budget_usd:.2f}"
                )

            overhead_ms = (time.perf_counter() - t0) * 1000.0
            step_overheads_ms.append(overhead_ms)

        cache_hit_ratio = (
            (total_cached_read_tokens / (len(steps) * 20000.0)) * 100.0 if steps else 0.0
        )

        summary = {
            "tenant_id": self.tenant_id,
            "workflow_id": workflow_id,
            "policy_mode": policy_mode,
            "routing_key": routing_key,
            "total_cost_usd": round(total_cost_usd, 4),
            "prefix_cache_hit_ratio_pct": round(cache_hit_ratio, 1),
            "max_gateway_overhead_ms": round(max(step_overheads_ms), 3),
        }
        self.ledger.emit(summary)
        return summary
