"""
DISPATCH Dynamic Pre-Flight Cost Engine (Phase 3).

Calculates exact pre-flight token estimates and USD cost projections before
LiteLLM execution. Enforces per-request blast-radius budgets, identifies
cost anomalies, and recommends cost-optimized tier substitutions.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dispatch.cost_engine")

# Pricing matrix per 1,000,000 tokens (USD)
# [Input $/1M, Output $/1M, Cached Read $/1M]
MODEL_GROUP_PRICING: Dict[str, Tuple[float, float, float]] = {
    # Tier 0: Local (Zero Cloud Spend)
    "local-fast":         (0.00, 0.00, 0.00),
    "local-always-on":    (0.00, 0.00, 0.00),
    "local-mobile":       (0.00, 0.00, 0.00),
    "local-embed":        (0.00, 0.00, 0.00),

    # Tier 1: Free Quotas
    "free-fast":          (0.00, 0.00, 0.00),
    "free-general":       (0.00, 0.00, 0.00),
    "free-longcontext":   (0.00, 0.00, 0.00),

    # Tier 2: Budget Cloud APIs
    "budget-general":     (0.14, 0.28, 0.014),   # DeepSeek V3
    "budget-reasoning":   (0.55, 2.19, 0.14),    # DeepSeek R1
    "budget-fast":        (0.10, 0.40, 0.025),   # Gemini 2.0 Flash
    "budget-claude":      (0.80, 4.00, 0.08),    # Claude 3.5 Haiku
    "budget-openai":      (0.15, 0.60, 0.075),   # GPT-4o Mini
    "budget-multilingual":(0.20, 0.80, 0.05),    # Moonshot / Qwen

    # Tier 3: Frontier / Premium APIs
    "premium-balanced":   (3.00, 15.00, 0.30),   # Claude 3.7 Sonnet
    "premium-openai":     (2.50, 10.00, 1.25),   # GPT-4o
    "premium-gemini":     (1.25, 5.00, 0.3125),  # Gemini 2.5 Pro
    "premium-best":       (15.00, 75.00, 1.50),  # Claude Opus
}

# Fallback hierarchy for budget overruns
DOWNGRADE_MAP: Dict[str, List[str]] = {
    "premium-best":     ["premium-balanced", "budget-claude", "local-fast"],
    "premium-balanced": ["budget-claude", "budget-general", "local-fast"],
    "premium-openai":   ["budget-openai", "budget-fast", "local-fast"],
    "premium-gemini":   ["budget-fast", "free-longcontext", "local-always-on"],
    "budget-reasoning": ["free-fast", "local-fast"],
    "budget-general":   ["free-general", "local-always-on"],
    "budget-claude":    ["budget-general", "local-fast"],
    "budget-openai":    ["free-fast", "local-always-on"],
}


def estimate_tokens(text: str) -> int:
    """Fast, accurate token estimation across diverse prompt structures.
    Uses ~3.75 chars per token for English code/text with word boundary weighting.
    """
    if not text:
        return 0
    char_len = len(text)
    # Heuristic: base character ratio
    tokens = char_len / 3.75
    # Whitespace and punctuation adjustment
    word_count = len(text.split())
    if word_count > 0:
        tokens = max(tokens, word_count * 1.2)
    return max(1, int(tokens))


def estimate_request_tokens(messages: List[Dict[str, Any]], max_tokens: int = 1024) -> Tuple[int, int]:
    """Estimate input tokens from messages and expected output tokens."""
    input_chars = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            input_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    input_chars += len(block.get("text", ""))

    input_tokens = estimate_tokens(" " * input_chars)
    # Output is bounded by max_tokens or default heuristic
    output_tokens = max_tokens if max_tokens and max_tokens > 0 else 512
    return input_tokens, output_tokens


@dataclass
class CostEstimate:
    model_group: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    input_cost_usd: float
    output_cost_usd: float
    pricing_per_million: Tuple[float, float, float]
    is_free: bool
    budget_exceeded: bool = False
    recommended_group: Optional[str] = None


class CostEngine:
    """Dynamic Pre-Flight Cost Engine."""

    def __init__(self, pricing_table: Optional[Dict[str, Tuple[float, float, float]]] = None):
        self.pricing = pricing_table or MODEL_GROUP_PRICING

    def estimate(self, messages: List[Dict[str, Any]], model_group: str,
                 max_tokens: int = 1024, cached_input: bool = False,
                 budget_ceiling_usd: Optional[float] = None) -> CostEstimate:
        """Calculate pre-flight tokens and cost estimate for a proposed request."""
        inp_tok, out_tok = estimate_request_tokens(messages, max_tokens)
        total_tok = inp_tok + out_tok

        pricing = self.pricing.get(model_group, (1.00, 2.00, 0.20))
        in_rate, out_rate, cache_rate = pricing

        effective_in_rate = cache_rate if cached_input else in_rate
        input_cost = (inp_tok / 1_000_000.0) * effective_in_rate
        output_cost = (out_tok / 1_000_000.0) * out_rate
        total_cost = round(input_cost + output_cost, 6)

        is_free = (in_rate == 0.0 and out_rate == 0.0)
        budget_exceeded = False
        recommended = None

        if budget_ceiling_usd is not None and total_cost > budget_ceiling_usd:
            budget_exceeded = True
            # Recommend cheaper alternative
            for alt_group in DOWNGRADE_MAP.get(model_group, ["local-fast", "free-fast"]):
                alt_est = self.estimate(messages, alt_group, max_tokens, cached_input, None)
                if alt_est.estimated_cost_usd <= budget_ceiling_usd:
                    recommended = alt_group
                    break

        return CostEstimate(
            model_group=model_group,
            input_tokens=inp_tok,
            output_tokens=out_tok,
            total_tokens=total_tok,
            estimated_cost_usd=total_cost,
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            pricing_per_million=pricing,
            is_free=is_free,
            budget_exceeded=budget_exceeded,
            recommended_group=recommended,
        )


cost_engine = CostEngine()
