"""
Phase 3 unit tests:
  - Cognitive / Semantic task routing
  - Dynamic pre-flight cost estimation & downgrade recommendations
  - Memory contradiction detection, automated supersession, and provenance
  - OpenRouter adapter mappings
  - Webhook escalation dispatcher
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dispatcher"))

os.environ["AUTONOMY_CONFIG"] = str(ROOT / "config/autonomy_boundary.yaml")
os.environ["TRUST_DB"] = tempfile.mktemp(suffix=".db")
os.environ["MEMORY_DB"] = tempfile.mktemp(suffix=".db")

from adapters import map_model_to_group
from autonomy_boundary import AutonomyBoundary
from cost_engine import CostEngine, estimate_request_tokens, estimate_tokens
from memory_engine import MemoryStore, SemanticFact, extract_entity_key
from semantic_router import SemanticRouter
from trust_ledger import TrustLedger


# ── Semantic Router Tests ───────────────────────────────────────────────────

def test_semantic_router_coding():
    sr = SemanticRouter()
    res = sr.classify("Write a python function to recursively traverse a JSON tree and filter null values")
    assert res.task == "coding"
    assert res.confidence > 0.4
    assert "local-fast" in res.route_candidates


def test_semantic_router_reasoning():
    sr = SemanticRouter()
    res = sr.classify("Analyze the trade-offs between monolithic database architecture and microservice event sourcing")
    assert res.task == "reasoning"
    assert res.confidence > 0.4


def test_semantic_router_math():
    sr = SemanticRouter()
    res = sr.classify("Solve the differential equation dy/dx = 3x^2 + 2x and calculate the integral")
    assert res.task == "math"


def test_semantic_router_fast_chat():
    sr = SemanticRouter()
    res = sr.classify("hi there")
    assert res.task == "fast_chat"


# ── Dynamic Pre-Flight Cost Engine Tests ────────────────────────────────────

def test_token_estimator():
    text = "Hello world, this is a test prompt for token estimation in DISPATCH."
    tok = estimate_tokens(text)
    assert 10 <= tok <= 25


def test_cost_engine_local_is_free():
    ce = CostEngine()
    msgs = [{"role": "user", "content": "Write a massive codebase with 5000 lines"}]
    est = ce.estimate(msgs, model_group="local-fast", max_tokens=2048)
    assert est.is_free is True
    assert est.estimated_cost_usd == 0.0


def test_cost_engine_premium_cost():
    ce = CostEngine()
    msgs = [{"role": "user", "content": "a" * 4000}]  # ~1000 input tokens
    est = ce.estimate(msgs, model_group="premium-balanced", max_tokens=1000)
    assert est.is_free is False
    assert est.estimated_cost_usd > 0.0
    assert est.input_tokens > 0
    assert est.output_tokens == 1000


def test_cost_engine_budget_overrun_downgrade():
    ce = CostEngine()
    msgs = [{"role": "user", "content": "a" * 40000}]
    # Force low budget ceiling
    est = ce.estimate(msgs, model_group="premium-best", max_tokens=4000, budget_ceiling_usd=0.05)
    assert est.budget_exceeded is True
    assert est.recommended_group is not None
    assert est.recommended_group != "premium-best"


# ── Memory Contradiction & Supersession Tests ───────────────────────────────

def test_memory_contradiction_detection_and_supersession():
    db_file = Path(tempfile.mktemp(suffix=".db"))
    store = MemoryStore(db_file)

    # Initial preference fact
    fact1 = SemanticFact(
        id="fact_1",
        fact="User prefers dark mode theme for all IDEs",
        category="preference",
        confidence=0.9,
        source_episodes=["ep_1"],
        created_at=time.time() - 100,
        updated_at=time.time() - 100,
        valid=True,
    )
    store.save_fact(fact1)

    active_facts = store.list_facts(category="preference")
    assert len(active_facts) == 1
    assert active_facts[0].id == "fact_1"

    # New conflicting preference fact
    fact2 = SemanticFact(
        id="fact_2",
        fact="User switched color scheme to light mode",
        category="preference",
        confidence=0.95,
        source_episodes=["ep_2"],
        created_at=time.time(),
        updated_at=time.time(),
        valid=True,
    )
    superseded = store.save_fact(fact2, check_contradictions=True)
    assert "fact_1" in superseded

    # Verify active facts query only returns valid fact2
    active_now = store.list_facts(category="preference", include_superseded=False)
    assert len(active_now) == 1
    assert active_now[0].id == "fact_2"

    # Verify provenance of fact1
    prov1 = store.get_fact_provenance("fact_1")
    assert prov1["is_active"] is False
    assert prov1["superseded_by"].id == "fact_2"
    assert len(prov1["contradiction_events"]) == 1

    # Verify contradictions history
    contradictions = store.get_contradictions()
    assert len(contradictions) == 1
    assert contradictions[0]["old_fact_id"] == "fact_1"
    assert contradictions[0]["new_fact_id"] == "fact_2"


# ── OpenRouter & Model Mapping Tests ────────────────────────────────────────

def test_openrouter_model_mapping():
    assert map_model_to_group("openrouter/anthropic/claude-3.7-sonnet") == "premium-balanced"
    assert map_model_to_group("openrouter/deepseek/deepseek-r1") == "budget-reasoning"
    assert map_model_to_group("openrouter/deepseek/deepseek-chat") == "budget-general"
    assert map_model_to_group("openrouter/qwen/qwen-2.5-72b-instruct") == "budget-multilingual"
    assert map_model_to_group("openrouter/meta-llama/llama-3.3-70b-instruct") == "free-fast"


# ── Webhook Notification Trigger Tests ──────────────────────────────────────

def test_webhook_dispatch_on_escalation():
    ledger = TrustLedger(Path(tempfile.mktemp(suffix=".db")))
    b = AutonomyBoundary(ROOT / "config/autonomy_boundary.yaml", trust_ledger=ledger)

    with patch.object(b, "_dispatch_webhook_notification") as mock_notify:
        # Create an action requiring escalation (e.g. regulated classification on tier 3)
        dec = b.evaluate("inference.tier3", classification="regulated", model_group="premium-openai", agent_id="agent:webhook-test")
        assert not dec.allowed
        assert dec.escalation is not None
        assert mock_notify.called
        assert mock_notify.call_args[0][0] == "escalation_created"

        # Resolve the escalation
        esc_id = dec.escalation["escalation_id"]
        res = b.resolve_escalation(esc_id, verdict="approved", reviewer="admin", reason="verified")
        assert res["status"] == "approved"
        assert mock_notify.call_args[0][0] == "escalation_resolved"


if __name__ == "__main__":
    test_semantic_router_coding()
    test_semantic_router_reasoning()
    test_semantic_router_math()
    test_semantic_router_fast_chat()
    test_token_estimator()
    test_cost_engine_local_is_free()
    test_cost_engine_premium_cost()
    test_cost_engine_budget_overrun_downgrade()
    test_memory_contradiction_detection_and_supersession()
    test_openrouter_model_mapping()
    test_webhook_dispatch_on_escalation()
    print("All Phase 3 tests passed successfully (11/11).")
