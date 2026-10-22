# DISPATCH v2: Enterprise Multi-Tenant Threat Model & Security Architecture

**Owner:** Hooman Parta (HP)  
**Governance Scope:** Multi-Tenant LLM Gateway, FinOps Control Plane & Agent Autonomy Boundaries  
**Compliance Standards:** HIPAA, PCI-DSS Level 1, SOC 2 Type II, NIST 800-53  

---

## 1. Executive Threat Overview

When foundation model inference transitions from isolated departmental prototypes to a unified enterprise control plane, four critical security and reliability threat vectors emerge:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             EXTERNAL THREAT SURFACE                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
[Threat 1: Cross-Tenant           [Threat 2: Noisy-Neighbor       [Threat 3: Runaway Spend
 Cache & Context Bleed]            Quota Starvation]               & Retry Storms]
     │                                 │                                 │
     ▼                                 ▼                                 ▼
[HMAC-SHA256 Key Salting]         [80% Tier-2 Preemption]         [Sub-25ms Circuit Breaker]
```

---

## 2. Threat Analysis & Defensive Mitigations

### Threat 1: Cross-Tenant Context Contamination & Cache Bleed
* **Attack Scenario:** In a shared inference environment, Tenant B submits a prompt structurally or semantically close to a prior query from Tenant A. A global cache lookup returns cached completions containing Tenant A's proprietary financial data, customer records, or API credentials.
* **Impact:** Critical compliance violation under HIPAA/PCI-DSS; loss of data sovereignty.
* **Control / Defensive Measure:**
  - **Cryptographic Namespace-Keyed Salting:** Local and provider prefix cache keys are salted with tenant-specific HMAC secrets:
    $$\text{CacheKey} = \text{HMAC-SHA256}(\text{TenantSecret}, \; \text{TenantID} \parallel \text{ModelID} \parallel \text{RoutingKey})$$
  - A cache hit across distinct tenant boundaries is mathematically impossible, even with byte-identical prompts.

### Threat 2: Noisy-Neighbor Quota Exhaustion (Denial of Service)
* **Attack Scenario:** An unthrottled offline analytics job or bulk embedding backfill spikes token consumption, consuming 100% of the shared foundation model Tokens Per Minute (TPM) quota. Incoming customer-facing Tier-0 transaction agents experience HTTP 429 outages.
* **Impact:** Revenue loss, SLA penalties, customer service disruption.
* **Control / Defensive Measure:**
  - **Hierarchical Token Buckets with Automated Preemption:**
    - `TIER_0_CRITICAL`: Customer-facing agents (guaranteed 60% pool reserve).
    - `TIER_1_INTERNAL`: Operational and CI/CD queues.
    - `TIER_2_BATCH`: Offline analytics and bulk backfills.
  - The instant global pool utilization crosses **80%**, all `TIER_2_BATCH` requests are preemptively dropped with `HTTP 429: Preempted by Tier-0`, guaranteeing 100% SLA preservation for Tier-0 services.

### Threat 3: Runaway Agent Spend & Unbounded Retry Storms
* **Attack Scenario:** A downstream microservice API schema changes, causing an autonomous operations agent to fail its tool call and enter an infinite retry loop. The agent consumes thousands of dollars of frontier model quota within minutes.
* **Impact:** Gross margin erosion, downstream microservice overload.
* **Control / Defensive Measure:**
  - **Sub-25ms Circuit Breaker & Hard Financial Caps:**
    - Sliding-window tracking (20 requests) of HTTP 429/5xx errors and tail latency (TTFT > 2000ms).
    - Tripping to `OPEN` when error rates exceed 10%, instantly rerouting to warm fallback models.
    - Hard `max_workflow_budget_usd` ceiling ($1.50 default): terminates agent execution immediately if projected cost exceeds budget.

### Threat 4: Ephemeral Header Leakage & KV Cache Invalidation
* **Attack Scenario:** Developers inject timestamps (`2026-10-21T12:00:00Z`), session UUIDs, or randomly ordered JSON tool schemas into system headers, invalidating transformer KV caches from token 0.
* **Impact:** 159% increase in inference token spend, increased time-to-first-token (TTFT).
* **Control / Defensive Measure:**
  - **3-Block Prompt Canonicalization:**
    - Block 1 (Immutable): Regex-strips ephemeral timestamps and UUIDs; recursively sorts MCP JSON keys (`json.dumps(sort_keys=True)`).
    - Block 2 (Session): Pinned tenant and workspace context (300s TTL).
    - Block 3 (Dynamic): Tool outputs and turns appended strictly at the tail.
