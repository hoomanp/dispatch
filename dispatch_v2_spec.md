# Agent Execution Spec: DISPATCH v2 — Enterprise LLM Inference Control Plane

**Owner:** Hooman Parta (HP)  
**Target Role Alignment:** L8+ Director / VP of Engineering (AI Platform, Cloud Infrastructure & Security Governance)  
**Repository:** github.com/hoomanp/dispatch  

---

## 1. Objective & Context

Build or update the `dispatch` reference repository so that it demonstrates four executive platform pillars with zero external API spend required for verification (`make eval` runs deterministically against cached/simulated provider telemetry):

1. **3-Block Prompt Canonicalization:** Preserves provider KV cache hit rates across multi-turn agent loops by recursively sorting MCP tool schemas and stripping dynamic timestamps and UUIDs from system headers.
2. **Hierarchical Multi-Tenant Token Buckets:** Isolates business-unit quotas and preempts Tier-2 batch jobs when Tier-0 production traffic spikes above 80% global pool capacity.
3. **Sub-25ms Circuit Breaking & Failover:** Detects upstream provider HTTP 429/5xx errors or tail-latency spikes and fails over to a warm secondary tier while enforcing hard per-workflow dollar caps.
4. **FinOps Chargeback Telemetry:** Emits structured unit-cost attribution records (`cost_per_completed_workflow`, `prefix_cache_hit_ratio`, and `gateway_overhead_ms` p99).

---

## 2. Target Repository Layout

```
dispatch/
├── Makefile
├── requirements.txt
├── README.md
├── THREAT_MODEL.md
├── DECISION_LOG.md
├── dispatch_v2_spec.md
├── data/
│   └── golden_workflows.json      # Auto-generated 50 multi-step workflows (10 steps each)
├── dispatch/
│   ├── __init__.py
│   ├── canonicalizer.py           # 3-Block prompt stratifier & JSON key sorter
│   ├── quota.py                   # Hierarchical token bucket with Tier-0 preemption
│   ├── circuit_breaker.py         # Sliding-window error/latency tracker & failover
│   ├── router.py                  # Capability-aware router + sticky session hashing
│   └── telemetry.py               # FinOps chargeback ledger & p99 overhead timer
├── tests/
│   ├── test_canonicalizer.py
│   ├── test_quota_preemption.py
│   └── test_circuit_breaker.py
└── eval.py                        # Reproduces README benchmark table via `make eval`
```

---

## 3. Core Modules & Implementation Highlights

- **`dispatch/canonicalizer.py`:** Deterministic 3-block prompt contract (`block_1_immutable`, `block_2_session`, `block_3_dynamic`, `sticky_routing_key`).
- **`dispatch/quota.py`:** Hierarchical token bucket with `PriorityTier` (`TIER_0_CRITICAL`, `TIER_1_INTERNAL`, `TIER_2_BATCH`) and HMAC tenant cache isolation.
- **`dispatch/circuit_breaker.py`:** Provider health tracker with sliding window and `BudgetExceededError` enforcement.
- **`dispatch/telemetry.py`:** Append-only JSONL `FinOpsLedger`.
- **`dispatch/router.py`:** `GatewayClient` coordinating canonicalization, quota preemption, capability tiering, and FinOps emission.
- **`eval.py`:** 50 simulated 10-step workflows producing the executive comparison table across `naive_unstratified`, `sticky_frontier`, and `dispatch_full`.
