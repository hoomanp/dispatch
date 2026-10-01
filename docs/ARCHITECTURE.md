# DISPATCH Architecture

## Design principle: thin brain on a battle-tested engine

DISPATCH deliberately does **not** implement provider integrations, retries,
cooldowns, rate-limit handling, or spend tracking. [LiteLLM](https://github.com/BerriAI/litellm)
does all of that at production scale. DISPATCH adds only what LiteLLM doesn't
have an opinion on:

1. **Which model group should this task use?** (task classification + tier chains)
2. **Is the budget healthy enough for that tier?** (budget gating)
3. **What context should the model see?** (memory assembly + document compression)

If a feature can be expressed as LiteLLM config, it goes in
`config/litellm_config.yaml`, not in Python.

## Request flow

```
client request (model="auto")
  │
  ├─ classify task            coding | reasoning | fast_chat | long_context
  │                           multilingual | math | quality | general
  ├─ check budget             LiteLLM spend API → tier 2/3 gates
  ├─ assemble context         (only if session_id given)
  │    ├─ HOT   last N turns from in-process session
  │    ├─ WARM  similar past-session summaries (Qdrant)
  │    ├─ COLD  relevant facts (SQLite FTS5 ∪ Qdrant vectors)
  │    └─ DOCS  map-reduce compression if over model limit
  ├─ select model group       first allowed group in the task's tier chain
  └─ forward to LiteLLM       which handles caching, fallback, retry, spend
```

## The four tiers

| Tier | What | Marginal cost | When used |
|---|---|---|---|
| 0 | Your hardware (vLLM / Ollama / LM Studio) | $0 | Always first |
| 1 | Free API quotas (Groq, Gemini free, OpenRouter free) | $0 | Local busy/absent |
| 2 | Budget APIs (DeepSeek, Gemini Flash, Haiku, 4o-mini) | ~$0.1–1 /M tokens | Gated by daily soft limit |
| 3 | Premium APIs (Sonnet, GPT-4o, Gemini Pro, Opus) | ~$1–15 /M tokens | Quality tasks, gated at 90% of monthly budget |

Hard rule: at 100% of `MONTHLY_BUDGET_USD`, everything routes to tier 0.
LiteLLM's own `max_budget` acts as a second, independent stop.

## The three cache layers

They are **independent and additive** — one request can benefit from all three.

**Layer 1 — response cache.** Full request → stored response. Redis for
exact SHA match, Qdrant for semantic similarity (cosine ≥ 0.92, embeddings
from the local `nomic-embed-text` model, so cache lookups cost nothing).
Hit = zero inference.

**Layer 2 — provider KV / prompt caching.** The provider stores KV tensors
for a stable prompt *prefix*; you still get fresh output, but the prefill is
skipped. Anthropic needs explicit `cache_control` breakpoints — LiteLLM
auto-injects them on system messages (`cache_control_injection_points` in the
config). OpenAI, Gemini 2.5, and DeepSeek cache automatically. Typical
savings: 50–90% of input cost. The rule that makes it work: **static content
first, dynamic content last** — one changed byte at the front invalidates
the prefix.

**Layer 3 — vLLM prefix cache.** `--enable-prefix-caching` lets concurrent
agents that share a system prompt reuse KV blocks on your own GPU. First
request pays full prefill; the rest are nearly free.

## Memory model

Inspired by the hot/warm/cold framing common in agent-memory literature:

- **Working (hot, 0ms)** — the active session's turns, held in-process.
- **Episodic (warm, ~50ms)** — every turn persisted to SQLite; session
  summaries embedded into Qdrant so future sessions can retrieve them by
  similarity.
- **Semantic (cold, ~200ms)** — when a session closes, a small local model
  (`phi4-mini`) summarizes it and extracts durable facts
  (preference / entity / skill / constraint / goal), stored in SQLite with an
  FTS5 index and embedded into Qdrant. Retrieval is the union of keyword
  (FTS) and vector search.
- **Procedural** — agent prompts and config in SQLite (table exists; wiring
  it into routing is an open TODO).

Everything runs on local models, so memory maintenance costs $0 and no
conversation data leaves your machines.

## Context beyond any window

`ContextPipeline` implements hierarchical map-reduce:

1. Chunk documents on paragraph boundaries (~6k tokens each).
2. Map: compress every chunk in parallel with the local model.
3. If the concatenated maps still exceed the target limit, reduce recursively
   (depth-capped).

A 10M-token corpus compresses to fit a 100k window in a few passes, at zero
API cost. Compression is lossy by design — the map instruction preserves
facts, decisions, and identifiers relevant to the query.

## Graceful degradation

Every external dependency failing is handled:

- Qdrant down → vector features return empty; FTS still works.
- Ollama down → embeddings/compression degrade (truncation fallback);
  routing unaffected.
- LiteLLM spend API unreachable → permissive on cheap tiers, LiteLLM's own
  `max_budget` remains the backstop.
- A provider failing → LiteLLM cooldown + fallback chains.

## Known limitations

- Task classification is regex-based. It's fast and free; it is not smart.
  Pass `task` explicitly for anything unusual.
- Fact extraction quality depends on the small local model; facts are capped
  at 20/session and deduplicated by content hash, but contradiction
  resolution between old and new facts is not implemented.
- Budget state polls LiteLLM's global spend endpoint; per-day accounting is
  approximate.

## Protocol adapters

OpenAI chat-completions is the internal format. `adapters.py` translates at
the edge so the Anthropic SDK (`/v1/messages`) and Gemini SDK
(`generateContent`) connect natively; both endpoints construct an internal
request and call the same `chat()` pipeline — routing, memory, budget, and
caching behave identically regardless of which protocol a client speaks.
Model-name mapping is family-based (opus→premium-best, sonnet→premium-balanced,
haiku→budget-claude, gemini-pro→premium-gemini, gemini-flash→budget-fast);
unknown names route as `auto`. Adapters are text-only by design — streaming
and tool-use translation live in LiteLLM's native endpoints, not here.

## Autonomy boundary (control plane)

`autonomy_boundary.py` implements the control-plane patterns from the
enterprise agent architecture this project follows: policies, approvals, and
escalations live separately from inference execution. Every request is
stamped at the boundary with an audit ID, residency tag, and data
classification. Classification drives tier selection automatically —
`regulated` data is physically incapable of reaching a cloud tier — and
blast-radius caps (per-request cost, document size) bound worst-case damage.
Denials are never silent: each produces an escalation record with the full
audit stamp, and the metric `dispatch_boundary_denials_total` makes policy
friction visible on the Grafana dashboard. The `evaluate()` call is the
single choke point; adding a new gated action class is one line of YAML.

## Observability

Three-stack discipline, implemented incrementally: (1) metrics now —
Prometheus counters/histograms for requests, latency, tokens, spend, memory,
and boundary denials, with a provisioned Grafana dashboard; (2) structured
request logging — LiteLLM's spend logs and UI cover the prompt/response
stack; (3) continuous evaluation — evals/harness.py runs a golden set against
live traffic with a pass-rate threshold; routing-only mode makes it CI-safe,
and full mode (including a regulated-classification sovereignty check)
runs on cron against production.

## Agent trust ledger

`trust_ledger.py` implements a Beta-distribution score per `agent_id`:
alpha accumulates good evidence (boundary allows, eval passes, human
approvals), beta accumulates bad evidence (denials, eval failures, human
rejections — weighted 3-6x heavier than the corresponding good event, so
trust is slow to earn and fast to lose). Evidence decays toward the
uninformative prior on a 14-day half-life, so ancient history stops
dominating a score without requiring an active reset.

The score maps to a trust tier (`probation` / `standard` / `trusted`) that
`AutonomyBoundary.evaluate()` applies as a second, independent narrowing
pass **after** the classification-driven tier ceiling — trust can force
approval on a tier the classification would otherwise allow autonomously,
and can widen the per-request cost cap for well-established agents, but
neither ever exceeds what the base policy and `hard_max_cost_per_request_usd`
permit. Sovereignty always wins: a maximally trusted agent still cannot
route regulated data to a cloud tier.

`AutonomyBoundary` takes its `TrustLedger` as a constructor parameter,
defaulting to the module singleton in production. This is deliberate: an
earlier version resolved the ledger via a bare module-level name inside
`evaluate()`, which worked correctly in a single production process but
broke under test — `importlib.reload()` mutates a module's namespace in
place, so any test file reloading `trust_ledger.py` later in the same
pytest session would silently redirect *already-constructed*
`AutonomyBoundary` instances from earlier test files to a different trust
database. Explicit injection fixes this for both testability and
production correctness: what an instance trusts is fixed at construction,
not resolved dynamically against shared mutable state. The test suite
proves the fix by running in three different file orderings (normal,
reversed, shuffled) and asserting identical pass counts in each.
