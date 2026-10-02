"""
Protocol adapters — let Anthropic SDK and Google Gemini SDK clients
talk to DISPATCH natively, in their own wire formats.

OpenAI format is DISPATCH's internal lingua franca. These adapters
translate at the edge:

  Anthropic /v1/messages          <->  OpenAI chat completions
  Gemini    /v1beta/...:generateContent  <->  OpenAI chat completions

Scope (deliberate): text conversations. Tool-use blocks, images, and
streaming are NOT translated here — clients needing full-fidelity
Anthropic streaming/tools should target LiteLLM's native /v1/messages
endpoint at :4000 instead (documented in README).
"""

import time
import uuid


# ── Model name mapping ────────────────────────────────────────────────────────
# Native SDK clients send real provider model names. Map the well-known
# families onto DISPATCH model groups; anything unrecognized routes as auto.

def map_model_to_group(model: str) -> str:
    m = (model or "").lower().strip()
    known_groups = {
        "local-fast", "local-always-on", "local-mobile", "local-embed",
        "free-fast", "free-general", "free-longcontext",
        "budget-general", "budget-reasoning", "budget-fast", "budget-claude",
        "budget-openai", "budget-multilingual",
        "premium-balanced", "premium-openai", "premium-gemini", "premium-best",
    }
    if m in known_groups:
        return m

    # Strip provider prefixes if given (e.g. openrouter/anthropic/claude-3.7-sonnet)
    cleaned = m
    for prefix in ("openrouter/", "openai/", "anthropic/", "gemini/", "deepseek/", "groq/", "ollama/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    if "opus" in cleaned:
        return "premium-best"
    if "sonnet" in cleaned or "claude-3.7" in cleaned or "claude-3-7" in cleaned:
        return "premium-balanced"
    if "haiku" in cleaned:
        return "budget-claude"
    if "gemini" in cleaned and "pro" in cleaned:
        return "premium-gemini"
    if "gemini" in cleaned:  # flash / flash-lite
        return "budget-fast"
    if "o3-mini" in cleaned:
        return "budget-reasoning"
    if "gpt-4o-mini" in cleaned:
        return "budget-openai"
    if "gpt" in cleaned or cleaned.startswith("o1") or cleaned.startswith("o3"):
        return "premium-openai"
    if "deepseek" in cleaned:
        return "budget-reasoning" if ("reason" in cleaned or "r1" in cleaned) else "budget-general"
    if "qwen" in cleaned and ("72b" in cleaned or "coder" in cleaned):
        return "budget-multilingual" if "72b" in cleaned else "local-fast"
    if "qwen" in cleaned:
        return "local-fast"
    if "llama-3.3" in cleaned or "llama-3.1" in cleaned or "llama" in cleaned:
        return "free-fast"
    if "mistral" in cleaned:
        return "free-general"
    return "auto"


# ── Anthropic /v1/messages ────────────────────────────────────────────────────

def _flatten_content(content) -> str:
    """Anthropic content can be a string or a list of typed blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text")
    return str(content)


def anthropic_to_internal(body: dict) -> dict:
    """Anthropic messages request -> ChatRequest-shaped dict."""
    messages = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _flatten_content(system)})
    for m in body.get("messages", []):
        messages.append({"role": m.get("role", "user"),
                         "content": _flatten_content(m.get("content", ""))})
    return {
        "model": map_model_to_group(body.get("model", "auto")),
        "messages": messages,
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_tokens", 2048),
        "stream": False,  # adapter scope: non-streaming
        # DISPATCH extensions pass straight through if the client sent them
        "task": body.get("task"),
        "strategy": body.get("strategy"),
        "force_tier": body.get("force_tier"),
        "session_id": body.get("session_id"),
        "use_memory": body.get("use_memory", True),
        "documents": body.get("documents"),
        "no_cache": body.get("no_cache", False),
    }


def internal_to_anthropic(data: dict, requested_model: str) -> dict:
    """OpenAI-format response -> Anthropic messages response."""
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "") or ""
    usage = data.get("usage") or {}
    finish = choice.get("finish_reason", "stop")
    stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(
        finish, "end_turn")
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
        "_dispatch": data.get("_dispatch", {}),
    }


# ── Gemini :generateContent ───────────────────────────────────────────────────

def _gemini_parts_text(parts) -> str:
    if not isinstance(parts, list):
        return str(parts)
    return "\n".join(p.get("text", "") for p in parts
                     if isinstance(p, dict) and "text" in p)


def gemini_to_internal(body: dict, model: str) -> dict:
    """Gemini generateContent request -> ChatRequest-shaped dict."""
    messages = []
    sys_inst = body.get("systemInstruction") or body.get("system_instruction")
    if sys_inst:
        messages.append({"role": "system",
                         "content": _gemini_parts_text(sys_inst.get("parts", []))})
    for c in body.get("contents", []):
        role = "assistant" if c.get("role") == "model" else "user"
        messages.append({"role": role,
                         "content": _gemini_parts_text(c.get("parts", []))})
    gen = body.get("generationConfig") or body.get("generation_config") or {}
    return {
        "model": map_model_to_group(model),
        "messages": messages,
        "temperature": gen.get("temperature", 0.7),
        "max_tokens": gen.get("maxOutputTokens", gen.get("max_output_tokens", 2048)),
        "stream": False,
        "task": body.get("task"),
        "strategy": body.get("strategy"),
        "force_tier": body.get("force_tier"),
        "session_id": body.get("session_id"),
        "use_memory": body.get("use_memory", True),
        "documents": body.get("documents"),
        "no_cache": body.get("no_cache", False),
    }


def internal_to_gemini(data: dict) -> dict:
    """OpenAI-format response -> Gemini generateContent response."""
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "") or ""
    usage = data.get("usage") or {}
    finish = choice.get("finish_reason", "stop")
    finish_reason = {"stop": "STOP", "length": "MAX_TOKENS"}.get(finish, "STOP")
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": finish_reason,
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
        "modelVersion": data.get("model", ""),
        "_dispatch": data.get("_dispatch", {}),
    }
