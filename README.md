# DISPATCH: Multi-Tenant LLM Inference Gateway & FinOps Control Plane

DISPATCH is an enterprise reference inference gateway designed to control unit economics, preserve provider KV cache locality, isolate multi-tenant quotas, and enforce deterministic circuit breaking for tool-using AI agents.

## The Production Problem

When enterprises scale from a single AI pilot to 40+ product engineering teams sharing foundation model endpoints, infrastructure margins and reliability degrade across three predictable fault lines:
1. **Prefix Cache Invalidation:** Developers inject dynamic timestamps, session UUIDs, or unsorted JSON tool schemas into prompt headers, invalidating transformer KV caches from token 0 and increasing multi-step workflow costs by up to 159%.
2. **Noisy-Neighbor Quota Exhaustion:** A single offline batch job from an internal analytics team can saturate shared cloud TPM (Tokens Per Minute) quotas, causing HTTP 429 outages for Tier-0 customer-facing services.
3. **Unbounded Retry Storms:** Minor tool-schema regressions cause autonomous agents to enter retry loops, multiplying token spend and exhausting downstream microservice pools.

```
Incoming Agent Request (Tenant Header + Priority Tier + Payload)
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. HIERARCHICAL QUOTA ENGINE (src/quota.py)                                 │
│    • Enforces per-tenant TPM budgets & per-workflow dollar caps             │
│    • Preempts Tier-2 batch traffic when global pool utilization >= 80%      │
└──────┬──────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. 3-BLOCK PROMPT CANONICALIZER (src/canonicalizer.py)                      │
│    • Block 1 (Immutable): Sorts MCP JSON keys; strips dynamic headers       │
│    • Block 2 (Session): Pins tenant context inside 300s TTL window          │
│    • Block 3 (Dynamic): Appends tool outputs strictly at prompt tail        │
└──────┬──────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. CAPABILITY ROUTER & CIRCUIT BREAKER (src/circuit_breaker.py)             │
│    • Sticky hashing on sha256(Block 1 + Block 2) for KV cache locality      │
│    • Tier 1 (Compact): Structured extraction & single-tool dispatch         │
│    • Tier 2 (Frontier): Multi-step synthesis & high-ambiguity planning      │
│    • Failover: Trips <25ms on upstream 429/5xx to warm secondary model      │
└──────┬──────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. FINOPS TELEMETRY LEDGER (src/telemetry.py)                               │
│    • Emits cost_per_completed_workflow, cache_hit_ratio, gateway_p99_ms     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Benchmark Results (`make eval`)

Evaluated across 50 representative 10-step operations agent workflows (20,000-token system/tool prefix, 2,000-token dynamic input, 1,000-token output per step) under standard frontier pricing ($4.00/M input, $20.00/M output, $5.00/M cache write, $0.20/M cache read) and compact tier pricing ($0.30/M input, $1.20/M output):

| Gateway Policy | Cost / 10-Step Workflow | Prefix Cache Hit Ratio | Gateway Overhead (p99) | Tier-0 SLA Preservation Under Batch Flood |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Unstratified (Round-Robin)** | $1.080 | 0.0% | 4.2 ms | 42.0% (Failed HTTP 429) |
| **Canonicalized Prefix + Sticky Frontier** | $0.416 (-61.5%) | 90.0% | 8.7 ms | 42.0% (Failed HTTP 429) |
| **DISPATCH Control Plane (Full)** | **$0.184 (-83.0%)** | **90.0%** | **11.4 ms** | **100.0% (Tier-2 Preempted)** |

## Developer Ergonomics (Paved-Road Adoption)

Product teams onboard without modifying agent business logic by passing standard requests through the DISPATCH SDK wrapper:

```python
from dispatch import GatewayClient, PriorityTier

client = GatewayClient(tenant_id="sec-ops-prod", priority=PriorityTier.TIER_0_CRITICAL)
response = client.execute_step(
    workflow_id="inc-8841",
    system_prefix=SYSTEM_POLICY,
    tools=MCP_TOOL_SCHEMAS,
    messages=turn_history,
    max_workflow_budget_usd=1.00
)
```

## Directory Structure

```
├── Makefile                           # Automation targets (make eval, make test)
├── eval.py                            # 50-workflow benchmark evaluation harness
├── dispatch/                          # Public SDK interface
│   └── __init__.py                    # Re-exports GatewayClient & PriorityTier
├── src/                               # Core Control Plane Modules
│   ├── canonicalizer.py               # 3-Block Prompt Canonicalizer & Consistent Hashing
│   ├── quota.py                       # Hierarchical Multi-Tenant Token Buckets & Preemption
│   ├── circuit_breaker.py             # Sub-25ms Circuit Breaker, Failover & Budget Sentinel
│   ├── telemetry.py                   # FinOps Chargeback & Gateway Overhead Isolation
│   ├── gateway.py                     # Central Control Plane Inference Gateway Engine
│   └── client.py                      # GatewayClient SDK wrapper
└── tests/                             # Comprehensive test suites
    ├── test_v2_canonicalizer.py
    ├── test_v2_quota.py
    ├── test_v2_circuit_breaker.py
    ├── test_v2_telemetry.py
    └── test_v2_gateway_client.py
```

## Core Control Plane Modules

### 1. 3-Block Prompt Canonicalizer (`src/canonicalizer.py`)
- **Block 1 (Static):** Sorts MCP JSON tool schemas recursively; regex-strips non-deterministic timestamps, time-of-day tokens, and UUIDs.
- **Block 2 (Session):** Pinned tenant and workspace context with a 300s TTL; isolated with cryptographic HMAC salts (`HMAC-SHA256(secret, tenant_id || model_id || prompt)`).
- **Block 3 (Dynamic):** Variable tool outputs and latest conversation turns appended strictly at the tail.
- **Consistent Hashing Router:** Maps `sha256(Block 1 + Block 2)` deterministically to provider replicas to maximize warm KV cache hits.

### 2. Hierarchical Multi-Tenant Token Buckets (`src/quota.py`)
- Per-business-unit TPM and RPM quotas with three priority tiers: `TIER_0_CRITICAL`, `TIER_1_INTERNAL`, and `TIER_2_BATCH`.
- When global inference pool utilization reaches $\ge 80\%$, automatically preempts `TIER_2_BATCH` requests (`HTTP 429: Preempted by Tier-0`) to ensure 100% SLA preservation for Tier-0 production traffic.

### 3. Sub-25ms Circuit Breaker & Failover (`src/circuit_breaker.py`)
- Tracks a sliding 20-request window of provider errors (HTTP 429, 500, 502, 503, 529, and TTFT > 2000ms).
- Trips to `OPEN` when failure rate exceeds 10%, instantly rerouting traffic to warm fallback providers.
- Hard workflow blast-radius budget ceiling ($1.50 default) to immediately terminate runaway agent loops.

### 4. FinOps Chargeback & Overhead Telemetry (`src/telemetry.py`)
- Isolates internal gateway processing overhead (`time.perf_counter()`, target p99 $< 15$ ms) from upstream provider latency.
- Emits structured FinOps records and validates mandatory enterprise chargeback headers (`X-Tenant-BU`, `X-Workflow-Type`, `X-Environment`).

---

## CapEx / OpEx Crossover Model: Managed APIs vs. Dedicated GPU Fleets

Evaluating self-hosted GPU clusters solely on peak theoretical token throughput ignores diurnal traffic troughs and SRE operational overhead.

Consider an enterprise workload processing **100 million input tokens and 20 million output tokens per day** (3.0B input and 600M output tokens per month):

| Cost Factor | Option A: Managed Multi-Tier Cloud Endpoints (Serverless API + Prefix Caching) | Option B: Self-Hosted 8xH100 GPU Node (Reserved Cloud Instance) |
| :--- | :--- | :--- |
| **Compute / Hardware** | Capability mix: 70% compact ($0.30/M in, $1.20/M out), 30% frontier with 80% cache hit ($0.96/M in, $20.00/M out) | 8 × H100 (80GB HBM3) at $2.20/GPU-hr on 1-yr reserved contract (730 hrs/mo) = **$12,848 / mo** |
| **Input Token Spend** | (2.1B × $0.30) + (0.9B × $0.96) = $630 + $864 = **$1,494 / mo** | Included in reserved instance |
| **Output Token Spend** | (420M × $1.20) + (180M × $20.00) = $504 + $3,600 = **$4,104 / mo** | Included in reserved instance |
| **Storage & Egress** | Included | Auxiliary control plane & storage = **$1,400 / mo** |
| **Platform / SRE Allocation** | **$0** (provider-managed availability & elasticity) | 0.5 FTE fully loaded (model patching, CUDA, autoscaling, on-call) = **$10,500 / mo** |
| **Total Monthly TCO** | **$5,598 / month** (or $12,000 / mo uncached baseline) | **$24,748 / month** |

---

## Mathematical Canary Governance: The Blast Radius Index ($BRI$)

For agentic workloads, platform SRE teams can gate CI/CD canary deployments using a composite Blast Radius Index ($BRI$) computed over OpenTelemetry span links during a 5% traffic shadow or canary window:

$$BRI = \sum_{s \in \text{Services}} w_s \cdot \Delta \text{Err}_s + \alpha \cdot \max\left(0, \frac{C_{\text{canary}} - C_{\text{baseline}}}{C_{\text{baseline}}}\right) + \beta \cdot \frac{R_{\text{tool}}}{R_{\text{max}}}$$

Where:
* $w_s$ is the criticality weight of downstream microservice $s$ ($w = 5.0$ for Tier-0 payment/auth services; $w = 1.0$ for internal reporting).
* $\Delta \text{Err}_s$ is the absolute increase in 4xx/5xx error rates on service $s$ during the canary window.
* $C_{\text{canary}}$ and $C_{\text{baseline}}$ represent mean token cost per workflow in canary vs. production baseline ($\alpha = 2.0$).
* $R_{\text{tool}}$ is the observed P95 tool-retry count per workflow against maximum allowed retry budget $R_{\text{max}}$ ($\beta = 1.5$).
* **Stop-Ship Threshold:** $BRI \ge 1.0$ automatically halts rollout, shifts 100% of traffic back to the pinned version artifact, and attaches offending trace IDs to the pull request.

---

## Executive FinOps Attribution & Unit-Cost Chargeback

| FinOps Metric | Formula / Definition | Executive Action Trigger |
| :--- | :--- | :--- |
| **Effective Cost per Completed Workflow** | $\frac{\text{Total Tenant Token Spend}}{\text{Successful Workflow Completions}}$ | Identifies prompt bloat or retry loops introduced by a specific release. |
| **Prefix Cache Efficiency Ratio** | $\frac{\text{Cached Read Tokens}}{\text{Total Input Tokens}}$ | Flags teams violating the 3-block prompt contract (target $\ge 75\%$). |
| **Tier Routing Distribution** | $\% \text{ Compact Tier vs. } \% \text{ Frontier Tier}$ | Highlights opportunities to distill repetitive tasks onto 15x cheaper models. |
| **Unattributed Shadow Spend** | $\text{Blocked Direct Egress Attempts}$ | Measures platform adoption vs. rogue API key usage. |

---

## Verification Commands

- **Run Benchmark Evaluation:**
  ```bash
  make eval
  ```
- **Run Complete Unit Test Suite:**
  ```bash
  make test
  ```
