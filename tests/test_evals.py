"""Eval harness: golden set schema + grading logic."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("harness",
                                              ROOT / "evals/harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def test_golden_parses_and_valid():
    cases = harness.load_golden()
    assert len(cases) >= 6
    for c in cases:
        assert c["check"] in ("route", "response_contains"), c["id"]
        if c["check"] == "route":
            assert "expect_task" in c or "expect_group_prefix" in c, c["id"]
        else:
            assert c.get("contains"), c["id"]

def test_golden_ids_unique():
    ids = [c["id"] for c in harness.load_golden()]
    assert len(ids) == len(set(ids))

def test_grade_route_pass_and_fail():
    case = {"expect_task": "coding", "expect_group_prefix": "local"}
    ok, _ = harness.grade_route(case, {"task": "coding",
                                       "model_group": "local-fast"})
    assert ok
    ok, why = harness.grade_route(case, {"task": "coding",
                                         "model_group": "premium-best"})
    assert not ok and "prefix" in why

def test_grade_contains_case_insensitive():
    ok, _ = harness.grade_contains({"contains": ["DEF add", "Return"]},
                                   "def add(a,b):\n    return a+b")
    assert ok
    ok, why = harness.grade_contains({"contains": ["missing_token"]}, "text")
    assert not ok and "missing" in why

def test_route_cases_agree_with_dispatcher():
    """Golden routing expectations must match the actual classifier —
    catches drift between golden set and TASK_ROUTES/classify changes."""
    sys.path.insert(0, str(ROOT / "dispatcher"))
    from main import classify_simple, select_model
    OK = {"monthly": 0, "daily": 0, "tier2_ok": True, "tier3_ok": True}
    for c in harness.load_golden():
        if c["check"] != "route":
            continue
        task = classify_simple(c["prompt"])
        if "expect_task" in c:
            assert task == c["expect_task"], \
                f"{c['id']}: classifier says {task}"
        if "expect_group_prefix" in c:
            group = select_model(task, OK)
            assert group.startswith(c["expect_group_prefix"]), \
                f"{c['id']}: selector says {group}"
