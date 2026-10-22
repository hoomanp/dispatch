import os
import time
from typing import Dict, Any, Tuple
from src.cache import cached_call

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

FRONTIER_MODEL = "claude-3-5-sonnet-20241022"
COMMODITY_MODEL = "gemini-1.5-flash"

def _call_anthropic(model: str, prompt: str) -> Dict[str, Any]:
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    t0 = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    return {
        "text": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": latency_ms,
        "model": model,
        "cached": False
    }

def _call_gemini(model: str, prompt: str) -> Dict[str, Any]:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    usage = response.usage_metadata
    return {
        "text": response.text,
        "input_tokens": usage.prompt_token_count if usage else 0,
        "output_tokens": usage.candidates_token_count if usage else 0,
        "latency_ms": latency_ms,
        "model": model,
        "cached": False
    }

def execute_model_call(model: str, prompt: str) -> Dict[str, Any]:
    """Wraps provider calls in disk caching."""
    if "claude" in model:
        fn = _call_anthropic
    else:
        fn = _call_gemini
    return cached_call(model, prompt, fn)

def route_always_frontier(prompt: str) -> Tuple[Dict[str, Any], str]:
    target = FRONTIER_MODEL
    res = execute_model_call(target, prompt)
    return res, target

def route_heuristic(prompt: str) -> Tuple[Dict[str, Any], str]:
    tokens = prompt.split()
    if len(tokens) < 60 and not any(k in prompt.lower() for k in ["contract", "refund", "legal", "clause"]):
        target = COMMODITY_MODEL
    else:
        target = FRONTIER_MODEL
    res = execute_model_call(target, prompt)
    return res, target

def route_dispatch(prompt: str) -> Tuple[Dict[str, Any], str]:
    high_ambiguity_keywords = ["dispute", "cancel", "refund", "terms", "escalate", "policy exception"]
    requires_frontier = any(k in prompt.lower() for k in high_ambiguity_keywords) or len(prompt.split()) > 100
    
    target = FRONTIER_MODEL if requires_frontier else COMMODITY_MODEL
    res = execute_model_call(target, prompt)
    
    # Automated fallback circuit
    if target == COMMODITY_MODEL and len(res.get("text", "").split()) < 15:
        target = FRONTIER_MODEL
        res = execute_model_call(target, prompt)
        
    return res, target
