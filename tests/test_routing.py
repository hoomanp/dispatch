import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "dispatcher"))
from main import classify_simple, select_model, TASK_ROUTES

OK   = {"monthly": 0,  "daily": 0, "tier2_ok": True,  "tier3_ok": True}
SOFT = {"monthly": 42, "daily": 6, "tier2_ok": False, "tier3_ok": True}
HARD = {"monthly": 50, "daily": 0, "tier2_ok": False, "tier3_ok": False}


def test_classify_coding():
    assert classify_simple("write a python function to sort") == "coding"

def test_classify_long_context_by_tokens():
    assert classify_simple("anything", context_tokens=30000) == "long_context"

def test_classify_short_defaults_fast_chat():
    assert classify_simple("hi there") == "fast_chat"

def test_classify_multilingual():
    assert classify_simple("translate this paragraph in spanish please") == "multilingual"


def test_select_prefers_local_when_healthy():
    assert select_model("coding", OK).startswith(("local", "free"))

def test_budget_exhausted_forces_local():
    assert select_model("quality", HARD) == "local-always-on"

def test_soft_limit_skips_budget_tier():
    mg = select_model("multilingual", SOFT)
    assert not mg.startswith("budget"), mg

def test_force_tier_local():
    assert select_model("quality", OK, strategy="local_only").startswith("local")

def test_quality_first_reverses():
    assert select_model("coding", OK, strategy="quality_first").startswith("premium")

def test_unknown_task_falls_back_to_general():
    assert select_model("nonexistent", OK) in TASK_ROUTES["general"]
