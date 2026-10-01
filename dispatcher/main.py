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
import re
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
from memory_engine import ContextPipeline, Episode, memory
from otel import set_routing_attributes
from otel import setup as otel_setup
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge,
                               Histogram, generate_latest)
from rate_limiter import rate_limiter
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
BUDGET_LIMIT.set(MONTHLY_BUDGET)

# ── Task classification ──────────────────────────────────────────────────────

PATTERNS = {
    "coding": re.compile(
        r"\b(code|function|class|def |implement|refactor|debug|bug|script|"
        r"python|typescript|javascript|rust|golang|sql|api|endpoint|"
        r"algorithm|regex|dockerfile)\b", re.I),
    "reasoning": re.compile(
        r"\b(analyze|reason|explain why|compare|evaluate|pros.?cons|"
        r"trade.?off|architecture|design|strategy|step.?by.?step|decision)\b",
        re.I),
    "math": re.compile(
        r"\b(calculate|compute|solve|equation|derivative|integral|proof|"
        r"theorem|probability|statistics|matrix)\b", re.I),
    "long_context": re.compile(
        r"\b(entire|whole|full|complete|all of|summarize this|"
        r"analyze this|the following file)\b", re.I),
    "multilingual": re.compile(
        r"\b(translate|in (spanish|french|german|chinese|japanese|arabic|"
        r"persian|korean|portuguese|italian|russian))\b", re.I),
    "quality": re.compile(
        r"\b(best possible|highest quality|most accurate|critical|"
        r"production.?ready|thorough)\b", re.I),
}

TASK_ROUTES = {
    "coding":       ["local-fast", "free-fast", "budget-general", "premium-balanced"],
    "reasoning":    ["local-fast", "budget-reasoning", "free-fast", "premium-balanced"],
    "fast_chat":    ["local-always-on", "local-mobile", "free-fast", "budget-fast"],
    "long_context": ["free-longcontext", "budget-fast", "budget-multilingual", "premium-gemini"],
    "multilingual": ["budget-multilingual", "free-longcontext", "budget-fast", "premium-gemini"],
    "math":         ["budget-reasoning", "free-fast", "premium-balanced", "premium-openai"],
    "quality":      ["premium-balanced", "premium-openai", "premium-gemini", "premium-best"],
    "embedding":    ["local-embed"],
    "general":      ["local-always-on", "free-general", "free-fast", "budget-general"],
}


def classify_simple(prompt: str, context_tokens: int = 0) -> str:
    if context_tokens > 25000:
        return "long_context"
    scores = {t: len(p.findall(prompt))
              for t, p in PATTERNS.items() if p.search(prompt)}
    if scores:
        return max(scores, key=scores.get)
    return "fast_chat" if len(prompt) < 80 else "general"


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
    if strategy == "local_only":
        force_tier = "local"
    elif strategy == "quality_first":
        route = list(reversed(route))
    elif strategy == "speed_first":
        prio = ["local-fast", "free-fast", "local-always-on", "budget-fast"]
        route = [g for g in prio if g in route] + \
                [g for g in route if g not in prio]
    for mg in route:
        if force_tier and not mg.startswith(("local", force_tier)):
            continue
        if mg.startswith("budget") and not budget["tier2_ok"]:
            continue
        if mg.startswith("premium") and not budget["tier3_ok"]:
            continue
        return mg
    return "local-always-on"


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
                                     strategy=req.strategy or "cost_first",
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
    decision = boundary.evaluate(
        action=f"inference.tier{tier}",
        classification=req.classification,
        model_group=model_group,
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
    r = await client.post(f"{LITELLM_URL}/v1/chat/completions",
                          json=payload, headers=AUTH_HEADERS, timeout=300.0)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    latency_ms = round((time.time() - start) * 1000, 1)

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
        answer = data["choices"][0]["message"].get("content", "") or ""
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
async def list_facts(category: Optional[str] = None, limit: int = 50):
    facts = memory.store.list_facts(category=category, limit=limit)
    return [{"fact": f.fact, "category": f.category,
             "confidence": f.confidence} for f in facts]


@app.post("/memory/search")
async def search_memory(request: Request):
    body = await request.json()
    query = body.get("query", "")
    facts = memory.store.search_facts_fts(query, limit=10)
    return {"query": query,
            "facts": [{"fact": f.fact, "category": f.category,
                       "confidence": f.confidence} for f in facts]}


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
    task = classify_simple(prompt, context_tokens)
    budget = await get_budget()
    mg = select_model(task, budget,
                      strategy=body.get("strategy", "cost_first"))
    return {"task": task, "model_group": mg,
            "priority_chain": TASK_ROUTES.get(task, []),
            "budget": {"monthly_spend": round(budget["monthly"], 4)}}


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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"service": "DISPATCH", "version": "1.1.0",
            "docs": "/docs", "memory": memory.stats()}
