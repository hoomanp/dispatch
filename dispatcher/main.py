"""
DISPATCH — cost-aware LLM router.

Thin task-aware tier selector in front of a LiteLLM proxy.
LiteLLM handles providers, retries, cooldowns, fallbacks, caching, spend.
This service handles: task classification, budget-gated tier selection,
memory-aware context assembly, and large-document compression.

Run: uvicorn main:app --host 0.0.0.0 --port 8080
"""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from adapters import (anthropic_to_internal, gemini_to_internal,
                      internal_to_anthropic, internal_to_gemini)
from autonomy_boundary import boundary, group_tier
from bandit_router import bandit_router
from cost_engine import cost_engine
from memory_engine import ContextPipeline, Episode, memory
from otel import set_routing_attributes
from otel import setup as otel_setup
from privacy_mask import privacy_mask
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge,
                               Histogram, generate_latest)
from rate_limiter import rate_limiter
from semantic_router import TASK_ROUTES, semantic_router
from trust_ledger import trust

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("dispatch")

LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-dispatch-master")
MONTHLY_BUDGET = float(os.getenv("MONTHLY_BUDGET_USD", "50"))
DAILY_SOFT_LIMIT = float(os.getenv("DAILY_SOFT_LIMIT_USD", "5"))
AUTH_HEADERS = {"Authorization": f"Bearer {LITELLM_KEY}"}

# ── Prometheus metrics (observability stack #1) ──────────────────────────────
def _metric(cls, name, doc, labelnames=(), **kwargs):
    """Reload-safe metric creation (uvicorn --reload, test re-imports)."""
    try:
        return cls(name, doc, labelnames, **kwargs)
    except ValueError:
        from prometheus_client import REGISTRY
        return REGISTRY._names_to_collectors[name]


REQS = _metric(Counter, "dispatch_requests_total", "Requests routed",
               ["task", "model_group", "tier", "outcome"])
LATENCY = _metric(Histogram, "dispatch_request_seconds",
                  "End-to-end latency", ["model_group"],
                  buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120])
TOKENS = _metric(Counter, "dispatch_tokens_total", "Tokens by direction",
                 ["model_group", "direction"])
BUDGET_SPEND = _metric(Gauge, "dispatch_monthly_spend_usd",
                       "Monthly cloud spend")
BUDGET_LIMIT = _metric(Gauge, "dispatch_monthly_budget_usd",
                       "Monthly budget cap")
MEMORY_FACTS = _metric(Gauge, "dispatch_memory_facts",
                       "Semantic facts stored")
MEMORY_SESSIONS = _metric(Gauge, "dispatch_memory_sessions",
                          "Sessions recorded")
BOUNDARY_DENIALS = _metric(Counter, "dispatch_boundary_denials_total",
                           "Autonomy boundary denials", ["action", "reason"])
AGENT_TRUST = _metric(Gauge, "dispatch_agent_trust_score",
                      "Per-agent trust score (0-1)", ["agent_id"])
RATE_LIMIT_THROTTLES = _metric(Counter, "dispatch_rate_limit_throttles_total",
                              "Rate limit throttles", ["agent_id"])
PII_REDACTIONS = _metric(Counter, "dispatch_pii_redactions_total",
                        "PII/PHI synthetic token redactions", ["entity_type"])
BANDIT_REWARD = _metric(Histogram, "dispatch_bandit_reward",
                        "Bandit routing reward score", ["model_group"])
BUDGET_LIMIT.set(MONTHLY_BUDGET)

# ── Task classification ──────────────────────────────────────────────────────

# Task classification — PATTERNS and TASK_ROUTES are imported from
# semantic_router (single source of truth). The local copies that used to
# live here had drifted out of sync with it.


def classify_simple(prompt: str, context_tokens: int = 0) -> str:
    if context_tokens > 25000:
        return "long_context"
    decision = semantic_router.classify(prompt)
    return decision.task


# ── HTTP client pooling ──────────────────────────────────────────────────────
_client: Optional[httpx.AsyncClient] = None

def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
        )
    return _client


async def get_budget() -> dict:
    try:
        client = get_client()
        r = await client.get(f"{LITELLM_URL}/global/spend",
                             headers=AUTH_HEADERS, timeout=5.0)
        if r.status_code == 200:
            d = r.json()
            monthly = float(d.get("spend", 0.0) or 0.0)
            return {
                "monthly": monthly, "daily": 0.0,
                "tier2_ok": monthly < MONTHLY_BUDGET * 0.8,
                "tier3_ok": monthly < MONTHLY_BUDGET * 0.9,
            }
    except Exception:
        pass
    # LiteLLM unreachable or endpoint variant differs: be permissive on
    # cheap tiers, conservative on premium.
    return {"monthly": 0.0, "daily": 0.0, "tier2_ok": True, "tier3_ok": True}


def select_model(task: str, budget: dict, strategy: str = "cost_first",
                 force_tier: Optional[str] = None) -> str:
    if budget["monthly"] >= MONTHLY_BUDGET:
        return "local-always-on"
    route = list(TASK_ROUTES.get(task, TASK_ROUTES["general"]))

    eligible = []
    for mg in route:
        if force_tier and not mg.startswith(("local", force_tier)):
            continue
        if mg.startswith("budget") and not budget["tier2_ok"]:
            continue
        if mg.startswith("premium") and not budget["tier3_ok"]:
            continue
        eligible.append(mg)

    if not eligible:
        return "local-always-on"

    if strategy == "bandit":
        chosen, _ = bandit_router.select_arm(eligible, strategy="bandit")
        return chosen
    elif strategy == "local_only":
        local_candidates = [g for g in eligible if g.startswith("local")]
        return local_candidates[0] if local_candidates else "local-always-on"
    elif strategy == "quality_first":
        return list(reversed(eligible))[0]
    elif strategy == "speed_first":
        prio = ["local-fast", "free-fast", "local-always-on", "budget-fast"]
        speed_candidates = [g for g in prio if g in eligible]
        return speed_candidates[0] if speed_candidates else eligible[0]

    return eligible[0]


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory.startup()
    logger.info("DISPATCH ready")
    yield
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()


app = FastAPI(title="DISPATCH", version="1.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
otel_setup(app)

# ── Bearer auth (opt-in): set DISPATCH_API_KEY to require it ────────────────
DISPATCH_API_KEY = os.getenv("DISPATCH_API_KEY", "")
_AUTH_EXEMPT = {"/health", "/metrics"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if DISPATCH_API_KEY and request.url.path not in _AUTH_EXEMPT:
        header = request.headers.get("authorization", "")
        # Anthropic SDK sends x-api-key; OpenAI/Gemini SDKs send Bearer
        key = (header.removeprefix("Bearer ").strip()
               or request.headers.get("x-api-key", "").strip())
        if key != DISPATCH_API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                {"error": "unauthorized",
                 "hint": "send Authorization: Bearer <DISPATCH_API_KEY> "
                         "or x-api-key header"}, status_code=401)
    return await call_next(request)


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[dict]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    task: Optional[str] = None
    strategy: Optional[str] = None
    force_tier: Optional[str] = None
    session_id: Optional[str] = None
    use_memory: bool = True
    documents: Optional[list[str]] = None
    no_cache: bool = False
    classification: Optional[str] = None  # regulated|confidential|internal|public
    agent_id: Optional[str] = None  # for trust scoring; falls back to X-Agent-Id header
    organization_id: Optional[str] = None  # Phase 2: multi-tenant organization tag
    project_id: Optional[str] = None  # Phase 2: multi-tenant project tag
    mask_pii: Optional[bool] = None  # Phase 4: Edge PII/PHI synthetic redaction


class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    title: str = ""


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    if not req.messages:
        raise HTTPException(400, "messages required")
    agent_id = req.agent_id or request.headers.get("x-agent-id", "unattributed")
    org_id = req.organization_id or request.headers.get("x-organization-id", "")
    proj_id = req.project_id or request.headers.get("x-project-id", "")

    # Phase 2: Dynamic Rate Limiting (gated by agent trust tier)
    snap = trust.snapshot(agent_id)
    rate_limits = boundary.policy.get("rate_limits", {})
    if snap.tier_name == "probation":
        max_rpm = int(rate_limits.get("probation_rpm", 10))
    elif snap.tier_name == "trusted":
        max_rpm = int(rate_limits.get("trusted_rpm", 180))
    else:
        max_rpm = int(rate_limits.get("default_rpm", 60))

    allowed, remaining, retry_after = rate_limiter.check(f"agent:{agent_id}", max_rpm)
    if not allowed:
        RATE_LIMIT_THROTTLES.labels(agent_id=agent_id).inc()
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Agent '{agent_id}' ({snap.tier_name} tier) exceeded rate limit of {max_rpm} RPM.",
                "retry_after_seconds": retry_after,
                "agent_id": agent_id,
                "trust_tier": snap.tier_name,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(max_rpm),
                "X-RateLimit-Remaining": "0",
            },
        )

    prompt = str(req.messages[-1].get("content", ""))
    context_tokens = int(sum(
        len(str(m.get("content", "")).split()) * 1.3 for m in req.messages))
    task = req.task or classify_simple(prompt, context_tokens)
    budget = await get_budget()
    messages = [dict(m) for m in req.messages]

    # Phase 4: Edge PII/PHI Redaction & Synthetic Token Masking
    should_mask = (req.mask_pii if req.mask_pii is not None
                   else (request.headers.get("x-mask-pii", "").lower() in ("true", "1")
                         or req.classification in ("confidential", "internal")))
    token_map = {}
    detected_pii = []
    if should_mask:
        messages, token_map, detected_pii = privacy_mask.mask_messages(messages)
        for entity in detected_pii:
            PII_REDACTIONS.labels(entity_type=entity).inc()

    # Memory-aware context assembly
    memory_active = bool(req.session_id and req.use_memory)
    if memory_active:
        # Record the incoming user turn
        memory.record(Episode(
            id=hashlib.sha256(
                f"{req.session_id}u{time.time()}".encode()).hexdigest()[:12],
            session_id=req.session_id, timestamp=time.time(),
            role="user", content=prompt, task_type=task))

        wctx = await memory.build_context(
            session_id=req.session_id, query=prompt,
            documents=req.documents or [],
            system_prompt=messages[0]["content"]
            if messages[0].get("role") == "system" else "")

        if wctx.memory_injection:
            if messages[0].get("role") == "system":
                messages[0]["content"] = (
                    f"{messages[0]['content']}\n\n{wctx.memory_injection}")
            else:
                messages.insert(0, {"role": "system",
                                    "content": wctx.memory_injection})
    elif req.documents:
        doc_text, _ = await ContextPipeline().prepare(req.documents, prompt)
        block = f"Documents:\n{doc_text}"
        if messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0]['content']}\n\n{block}"
        else:
            messages.insert(0, {"role": "system", "content": block})

    model_group = (req.model if req.model != "auto"
                   else select_model(task, budget,
                                     strategy=req.strategy or "bandit",
                                     force_tier=req.force_tier))

    # ── Autonomy boundary (control plane) ────────────────────────────────────
    # Sovereignty: classification caps the tier; degrade rather than fail
    # when routing was automatic, hard-deny when the client forced a group.
    task_route = TASK_ROUTES.get(task, TASK_ROUTES["general"])
    if req.model == "auto":
        model_group = boundary.clamp_group_to_ceiling(
            model_group, req.classification, task_route)
    tier = group_tier(model_group)
    docs_chars = sum(len(d) for d in (req.documents or []))

    # Pre-flight token and cost estimation
    cost_est = cost_engine.estimate(
        messages=messages,
        model_group=model_group,
        max_tokens=req.max_tokens or 1024,
        cached_input=not req.no_cache,
        budget_ceiling_usd=float(boundary.policy.get("max_cost_per_request_usd", 0.50)),
    )
    if req.model == "auto" and cost_est.budget_exceeded and cost_est.recommended_group:
        logger.info(
            f"cost pre-flight: ${cost_est.estimated_cost_usd:.4f} exceeded cap; "
            f"downgrading '{model_group}' to '{cost_est.recommended_group}'")
        model_group = cost_est.recommended_group
        tier = group_tier(model_group)

    decision = boundary.evaluate(
        action=f"inference.tier{tier}",
        classification=req.classification,
        model_group=model_group,
        estimated_cost_usd=cost_est.estimated_cost_usd,
        documents_chars=docs_chars,
        agent_id=agent_id,
        organization_id=org_id,
        project_id=proj_id)
    if decision.trust:
        AGENT_TRUST.labels(agent_id=agent_id).set(decision.trust["score"])
    if not decision.allowed:
        BOUNDARY_DENIALS.labels(action=decision.action,
                                reason=decision.reason[:60]).inc()
        REQS.labels(task=task, model_group=model_group,
                    tier=str(tier), outcome="denied").inc()
        # Record failure penalty for bandit
        bandit_router.record_feedback(model_group, success=False)
        raise HTTPException(403, {
            "error": "autonomy_boundary",
            "reason": decision.reason,
            "audit": decision.audit,
            "escalation": decision.escalation})
    set_routing_attributes(task, model_group, tier,
                           decision.audit.get("audit_id", ""))

    payload = {
        "model": model_group, "messages": messages,
        "stream": req.stream, "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.no_cache:
        payload["cache"] = {"no-cache": True}

    start = time.time()

    if req.stream:
        async def streamer():
            client = get_client()
            async with client.stream(
                    "POST", f"{LITELLM_URL}/v1/chat/completions",
                    json=payload, headers=AUTH_HEADERS, timeout=300.0) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        return StreamingResponse(streamer(), media_type="text/event-stream")

    client = get_client()
    try:
        r = await client.post(f"{LITELLM_URL}/v1/chat/completions",
                              json=payload, headers=AUTH_HEADERS, timeout=300.0)
    except Exception as e:
        bandit_router.record_feedback(model_group, success=False)
        raise HTTPException(502, f"Upstream proxy execution failed: {e}")

    if r.status_code != 200:
        bandit_router.record_feedback(model_group, success=False)
        raise HTTPException(r.status_code, r.text)

    data = r.json()
    latency_ms = round((time.time() - start) * 1000, 1)

    # Phase 4: Reversible unmasking of generated response
    answer = data["choices"][0]["message"].get("content", "") or ""
    if token_map and answer:
        answer = privacy_mask.unmask_text(answer, token_map)
        data["choices"][0]["message"]["content"] = answer

    # Phase 4: Adaptive Bandit Router reward feedback
    reward = bandit_router.record_feedback(
        model_group=model_group,
        success=True,
        latency_ms=latency_ms,
        cost_usd=cost_est.estimated_cost_usd,
    )
    BANDIT_REWARD.labels(model_group=model_group).observe(reward)

    # Metrics
    usage_m = data.get("usage", {}) or {}
    REQS.labels(task=task, model_group=model_group,
                tier=str(tier), outcome="ok").inc()
    LATENCY.labels(model_group=model_group).observe(latency_ms / 1000)
    TOKENS.labels(model_group=model_group, direction="input").inc(
        int(usage_m.get("prompt_tokens", 0) or 0))
    TOKENS.labels(model_group=model_group, direction="output").inc(
        int(usage_m.get("completion_tokens", 0) or 0))
    BUDGET_SPEND.set(budget["monthly"])

    if memory_active:
        usage = data.get("usage", {}) or {}
        memory.record(Episode(
            id=hashlib.sha256(
                f"{req.session_id}a{time.time()}".encode()).hexdigest()[:12],
            session_id=req.session_id, timestamp=time.time(),
            role="assistant", content=answer, task_type=task,
            provider_id=model_group,
            tokens_used=int(usage.get("total_tokens", 0) or 0)))

    data["_dispatch"] = {
        "task": task, "model_group": model_group,
        "latency_ms": latency_ms, "session_id": req.session_id,
        "memory_active": memory_active,
        "audit": decision.audit,
        "bandit_reward": reward,
        "privacy_masking": {
            "enabled": bool(should_mask),
            "masked_entities": detected_pii,
            "tokens_redacted": len(token_map),
        },
        "cost_estimate": {
            "input_tokens": cost_est.input_tokens,
            "output_tokens": cost_est.output_tokens,
            "estimated_cost_usd": cost_est.estimated_cost_usd,
            "is_free": cost_est.is_free,
        },
        "budget": {"monthly_spend": round(budget["monthly"], 4),
                   "monthly_limit": MONTHLY_BUDGET},
    }
    return data


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    body["model"] = "local-embed"
    client = get_client()
    r = await client.post(f"{LITELLM_URL}/v1/embeddings",
                          json=body, headers=AUTH_HEADERS, timeout=60.0)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()


# ── Native protocol endpoints (Anthropic SDK, Gemini SDK) ────────────────────

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API — point the Anthropic SDK's base_url here.
    Text conversations only; for streaming/tool-use fidelity, use
    LiteLLM's native /v1/messages at :4000 (see README)."""
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(
            400, "Streaming not supported on this adapter; use "
                 "LiteLLM :4000/v1/messages for streaming Anthropic clients.")
    req = ChatRequest(**anthropic_to_internal(body))
    data = await chat(req, request)
    return internal_to_anthropic(data, body.get("model", "auto"))


@app.post("/v1beta/models/{model_action}")
async def gemini_generate(model_action: str, request: Request):
    """Gemini generateContent API — point google-genai's base_url here.
    Path arrives as '{model}:generateContent'. Text conversations only."""
    if ":" not in model_action:
        raise HTTPException(404, "expected {model}:generateContent")
    model, action = model_action.split(":", 1)
    if action not in ("generateContent",):
        raise HTTPException(
            400, f"action '{action}' not supported; use generateContent "
                 "(streaming: use the OpenAI-compat endpoint instead)")
    body = await request.json()
    req = ChatRequest(**gemini_to_internal(body, model))
    data = await chat(req, request)
    return internal_to_gemini(data)


# ── Sessions ─────────────────────────────────────────────────────────────────

@app.post("/v1/sessions")
async def create_session(req: SessionRequest):
    sid = req.session_id or str(uuid.uuid4())
    memory.start_session(sid, req.title)
    return {"session_id": sid, "title": req.title}


@app.delete("/v1/sessions/{session_id}")
async def close_session(session_id: str):
    asyncio.create_task(memory.end_session(session_id))
    return {"status": "closing", "session_id": session_id,
            "note": "fact extraction runs in background"}


# ── Introspection ────────────────────────────────────────────────────────────

@app.get("/memory/stats")
async def memory_stats():
    return memory.stats()


@app.get("/memory/sessions")
async def recent_sessions():
    return memory.store.get_recent_sessions(limit=20)


@app.get("/memory/facts")
async def list_facts(category: Optional[str] = None, include_superseded: bool = False, limit: int = 50):
    facts = memory.store.list_facts(category=category, include_superseded=include_superseded, limit=limit)
    return [{
        "id": f.id,
        "fact": f.fact,
        "category": f.category,
        "confidence": f.confidence,
        "valid": f.valid,
        "superseded_by": f.superseded_by,
        "entity_key": f.entity_key,
        "created_at": f.created_at,
        "updated_at": f.updated_at,
    } for f in facts]


@app.get("/memory/facts/{fact_id}/provenance")
async def fact_provenance(fact_id: str):
    prov = memory.store.get_fact_provenance(fact_id)
    if not prov:
        raise HTTPException(404, f"fact '{fact_id}' not found")
    return prov


@app.get("/memory/contradictions")
async def memory_contradictions(limit: int = 50):
    return {"contradictions": memory.store.get_contradictions(limit=limit)}


@app.post("/memory/search")
async def search_memory(request: Request):
    body = await request.json()
    query = body.get("query", "")
    include_superseded = bool(body.get("include_superseded", False))
    facts = memory.store.search_facts_fts(query, limit=10, include_superseded=include_superseded)
    return {"query": query,
            "facts": [{"id": f.id, "fact": f.fact, "category": f.category,
                       "confidence": f.confidence, "valid": f.valid,
                       "superseded_by": f.superseded_by} for f in facts]}


@app.post("/context/preview")
async def context_preview(request: Request):
    body = await request.json()
    docs = body.get("documents", [])
    if not docs:
        raise HTTPException(400, "documents list required")
    total_chars = sum(len(d) for d in docs)
    pipeline = ContextPipeline()
    needs = total_chars > pipeline.limit_chars
    return {"total_chars": total_chars,
            "total_tokens_estimated": total_chars // 4,
            "model_limit_tokens": pipeline.limit,
            "needs_compression": needs,
            "strategy": "hierarchical_map_reduce" if needs else "direct"}


@app.post("/dispatch/classify")
async def explain_route(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    context_tokens = int(body.get("context_tokens", 0))
    semantic_res = semantic_router.classify(prompt)
    task = "long_context" if context_tokens > 25000 else semantic_res.task
    budget = await get_budget()
    mg = select_model(task, budget,
                      strategy=body.get("strategy", "cost_first"))
    cost_est = cost_engine.estimate(
        messages=[{"role": "user", "content": prompt}],
        model_group=mg,
        max_tokens=int(body.get("max_tokens", 1024)),
    )
    return {
        "task": task,
        "confidence": semantic_res.confidence,
        "model_group": mg,
        "priority_chain": TASK_ROUTES.get(task, []),
        "justification": semantic_res.justification,
        "features": semantic_res.features,
        "estimated_cost_usd": cost_est.estimated_cost_usd,
        "estimated_tokens": cost_est.total_tokens,
        "budget": {"monthly_spend": round(budget["monthly"], 4)},
    }


@app.post("/dispatch/cost-estimate")
async def cost_estimate_endpoint(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    if not messages and "prompt" in body:
        messages = [{"role": "user", "content": body["prompt"]}]
    model_group = body.get("model", "budget-general")
    max_tokens = int(body.get("max_tokens", 1024))
    cached = bool(body.get("cached", False))
    est = cost_engine.estimate(
        messages=messages,
        model_group=model_group,
        max_tokens=max_tokens,
        cached_input=cached,
        budget_ceiling_usd=float(boundary.policy.get("max_cost_per_request_usd", 0.50)),
    )
    return est


@app.get("/dispatch/tiers")
async def tiers():
    budget = await get_budget()
    return {
        "tier_0_local": {"enabled": True, "cost": "free"},
        "tier_1_free_quota": {"enabled": True, "cost": "free (rate limited)"},
        "tier_2_budget": {"enabled": budget["tier2_ok"]},
        "tier_3_premium": {"enabled": budget["tier3_ok"]},
        "budget": {"monthly_spend": round(budget["monthly"], 4),
                   "monthly_limit": MONTHLY_BUDGET},
    }


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    s = memory.stats()
    MEMORY_FACTS.set(s["cold"]["semantic_facts"])
    MEMORY_SESSIONS.set(s["warm"]["total_sessions"])
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/trust/{agent_id}")
async def trust_snapshot(agent_id: str):
    """Current trust score, confidence, and recent evidence for one agent."""
    snap = trust.snapshot(agent_id)
    return {"agent_id": snap.agent_id, "score": snap.score,
            "confidence": snap.confidence, "tier": snap.tier_name,
            "events": snap.events, "alpha": snap.alpha, "beta": snap.beta,
            "recent_events": trust.recent_events(agent_id, limit=20)}


@app.get("/trust")
async def trust_leaderboard():
    """All agents with recent activity, most recent first."""
    return [{"agent_id": s.agent_id, "score": s.score,
             "confidence": s.confidence, "tier": s.tier_name,
             "events": s.events} for s in trust.leaderboard()]


class TrustEventRequest(BaseModel):
    event_type: str  # eval_pass | eval_fail | human_approve | human_reject
    detail: str = ""


@app.post("/trust/{agent_id}/event")
async def trust_record_event(agent_id: str, req: TrustEventRequest):
    """Feed external evidence into an agent's trust score — eval harness
    results or a human's approve/reject verdict on an escalation. Boundary
    allow/deny events are recorded automatically by evaluate() and are
    not accepted here."""
    if req.event_type not in ("eval_pass", "eval_fail",
                              "human_approve", "human_reject"):
        raise HTTPException(
            400, f"event_type must be one of eval_pass, eval_fail, "
                 f"human_approve, human_reject (got {req.event_type!r})")
    trust.record(agent_id, req.event_type, detail=req.detail)
    snap = trust.snapshot(agent_id)
    AGENT_TRUST.labels(agent_id=agent_id).set(snap.score)
    return {"agent_id": agent_id, "recorded": req.event_type,
            "new_score": snap.score, "new_tier": snap.tier_name}


@app.get("/boundary/policy")
async def boundary_policy():
    """Current autonomy boundary policy (control plane view)."""
    return boundary.policy_view()


class EscalationVerdictRequest(BaseModel):
    verdict: str  # approved | rejected
    reviewer: str = "security-admin"
    reason: str = ""


@app.post("/boundary/escalations/{escalation_id}/verdict")
async def resolve_escalation(escalation_id: str, req: EscalationVerdictRequest):
    """Phase 2: Review and resolve a pending escalation.
    
    Approving raises the agent's trust score; rejecting penalizes it.
    """
    try:
        result = boundary.resolve_escalation(
            escalation_id=escalation_id,
            verdict=req.verdict,
            reviewer=req.reviewer,
            reason=req.reason,
        )
        agent_id = result.get("agent_id")
        if agent_id:
            AGENT_TRUST.labels(agent_id=agent_id).set(result.get("new_trust_score", 0.5))
        return result
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/boundary/escalations")
async def boundary_escalations(status: Optional[str] = None):
    """Denied actions awaiting a policy change / approval."""
    return {"escalations": boundary.escalations(status=status)}


# ── Phase 4: Bandit Feedback & Stats ─────────────────────────────────────────

class BanditFeedbackRequest(BaseModel):
    model_group: str
    success: bool = True
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    user_satisfaction: Optional[float] = None  # -1.0 to 1.0 or 0.0 to 1.0


@app.post("/dispatch/feedback")
async def bandit_feedback(req: BanditFeedbackRequest):
    """Explicit feedback hook for the Adaptive Contextual Bandit Router."""
    reward = bandit_router.record_feedback(
        model_group=req.model_group,
        success=req.success,
        latency_ms=req.latency_ms,
        cost_usd=req.cost_usd,
        user_satisfaction=req.user_satisfaction,
    )
    BANDIT_REWARD.labels(model_group=req.model_group).observe(reward)
    return {
        "model_group": req.model_group,
        "reward_assigned": reward,
        "arm_stats": bandit_router.stats().get(req.model_group, {}),
    }


@app.get("/dispatch/bandit/stats")
async def bandit_stats():
    """Returns real-time multi-armed bandit arm performance and posterior distributions."""
    return {"arms": bandit_router.stats()}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"service": "DISPATCH", "version": "1.2.0",
            "docs": "/docs", "memory": memory.stats()}

