"""
Unit tests for the DISPATCH evaluation & benchmarking harness.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evals"))

import harness


def test_load_golden_cases():
    cases = harness.load_golden()
    assert len(cases) >= 10
    checks = {c["check"] for c in cases}
    assert "route" in checks
    assert "sovereignty" in checks
    assert "cost_limit" in checks


def test_grade_route():
    case = {"id": "test-route", "expect_task": "coding", "expect_group_prefix": "local"}
    
    # Matching
    ok, detail = harness.grade_route(case, {"task": "coding", "model_group": "local-fast", "confidence": 0.85})
    assert ok is True
    assert "coding" in detail

    # Mismatched task
    ok, detail = harness.grade_route(case, {"task": "reasoning", "model_group": "local-fast"})
    assert ok is False

    # Mismatched group prefix
    ok, detail = harness.grade_route(case, {"task": "coding", "model_group": "budget-general"})
    assert ok is False


def test_grade_sovereignty():
    case_allow = {"expect_allowed": True}
    ok, _ = harness.grade_sovereignty(case_allow, 200, {})
    assert ok is True

    ok, _ = harness.grade_sovereignty(case_allow, 403, {"reason": "blocked"})
    assert ok is False

    case_deny = {"expect_allowed": False}
    ok, _ = harness.grade_sovereignty(case_deny, 403, {})
    assert ok is True

    ok, _ = harness.grade_sovereignty(case_deny, 200, {})
    assert ok is False


def test_grade_cost_limit():
    case = {"expect_max_cost": 0.50}
    ok, _ = harness.grade_cost_limit(case, {"estimated_cost_usd": 0.25})
    assert ok is True

    ok, _ = harness.grade_cost_limit(case, {"estimated_cost_usd": 0.75})
    assert ok is False


def test_grade_contains():
    case = {"contains": ["def add_two", "return"]}
    text_pass = "def add_two(a, b):\n    return a + b"
    ok, _ = harness.grade_contains(case, text_pass)
    assert ok is True

    text_fail = "const addTwo = (a, b) => a + b;"
    ok, detail = harness.grade_contains(case, text_fail)
    assert ok is False
    assert "missing" in detail


def test_generate_markdown_report():
    out_file = Path(tempfile.mktemp(suffix=".md"))
    summary = {
        "timestamp": 1740000000.0,
        "base_url": "http://localhost:8080",
        "routing_only": True,
        "graded": 2,
        "passed": 2,
        "pass_rate": 1.0,
        "results": [
            {"id": "case-1", "check": "route", "status": "pass", "latency_ms": 12, "detail": "task=coding"},
            {"id": "case-2", "check": "sovereignty", "status": "pass", "latency_ms": 5, "detail": "denied by sovereignty"},
        ],
    }
    harness.generate_markdown_report(summary, out_file)
    content = out_file.read_text()
    assert "# DISPATCH Evaluation & Benchmark Report" in content
    assert "✅ PASS" in content
    assert "case-1" in content


if __name__ == "__main__":
    test_load_golden_cases()
    test_grade_route()
    test_grade_sovereignty()
    test_grade_cost_limit()
    test_grade_contains()
    test_generate_markdown_report()
    print("All eval harness unit tests passed successfully (6/6).")
