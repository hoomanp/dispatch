"""Phase 2 unit tests: Rate limiting, persistent escalations, resolution workflows, and multi-tenancy."""
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dispatcher"))

os.environ["AUTONOMY_CONFIG"] = str(ROOT / "config/autonomy_boundary.yaml")
os.environ["TRUST_DB"] = tempfile.mktemp(suffix=".db")

from autonomy_boundary import AutonomyBoundary, group_tier
from rate_limiter import SlidingWindowRateLimiter
from trust_ledger import TrustLedger

L = TrustLedger(Path(os.environ["TRUST_DB"]))
B = AutonomyBoundary(ROOT / "config/autonomy_boundary.yaml", trust_ledger=L)


# ── Rate Limiter Mechanics ───────────────────────────────────────────────────

def test_rate_limiter_allows_under_limit():
    rl = SlidingWindowRateLimiter(window_seconds=10.0)
    for _ in range(5):
        allowed, remaining, retry_after = rl.check("agent:test-under", max_rpm=10)
        assert allowed
        assert retry_after == 0
    assert remaining == 5


def test_rate_limiter_throttles_over_limit():
    rl = SlidingWindowRateLimiter(window_seconds=10.0)
    for _ in range(3):
        allowed, _, _ = rl.check("agent:test-over", max_rpm=3)
        assert allowed
    # 4th request should be throttled
    allowed, remaining, retry_after = rl.check("agent:test-over", max_rpm=3)
    assert not allowed
    assert remaining == 0
    assert retry_after > 0


def test_rate_limiter_window_slide():
    rl = SlidingWindowRateLimiter(window_seconds=0.2)
    for _ in range(2):
        assert rl.check("agent:slide", max_rpm=2)[0]
    assert not rl.check("agent:slide", max_rpm=2)[0]
    time.sleep(0.25)
    # After window slides, requests should be permitted again
    assert rl.check("agent:slide", max_rpm=2)[0]


# ── Multi-Tenant Tagging ─────────────────────────────────────────────────────

def test_multitenant_audit_stamp():
    d = B.evaluate(
        "inference.tier0",
        model_group="local-fast",
        agent_id="corp-agent",
        organization_id="org_cyberdyne",
        project_id="proj_skynet"
    )
    assert d.allowed
    assert d.audit["organization_id"] == "org_cyberdyne"
    assert d.audit["project_id"] == "proj_skynet"


def test_audit_stamp_omits_empty_tenant_tags():
    d = B.evaluate("inference.tier0", model_group="local-fast", agent_id="solo-agent")
    assert d.allowed
    assert "organization_id" not in d.audit
    assert "project_id" not in d.audit


# ── Persistent Escalation Lifecycle & Workflows ──────────────────────────────

def test_escalation_created_with_pending_status():
    d = B.evaluate(
        "inference.tier3",
        classification="regulated",
        model_group="premium-best",
        agent_id="agent-esc-pending"
    )
    assert not d.allowed
    assert d.escalation is not None
    esc_id = d.escalation["escalation_id"]
    assert d.escalation["status"] == "pending"

    # Verify queryable via escalations()
    pending = B.escalations(status="pending")
    matching = [e for e in pending if e["escalation_id"] == esc_id]
    assert len(matching) == 1
    assert matching[0]["agent_id"] == "agent-esc-pending"


def test_escalation_approval_workflow_raises_trust():
    agent_id = "agent-to-approve"
    d = B.evaluate(
        "inference.tier3",
        classification="regulated",
        model_group="premium-best",
        agent_id=agent_id
    )
    esc_id = d.escalation["escalation_id"]
    initial_score = L.snapshot(agent_id).score

    # Approve the escalation
    resolved = B.resolve_escalation(
        escalation_id=esc_id,
        verdict="approved",
        reviewer="sec-lead",
        reason="authorized for security audit"
    )
    assert resolved["status"] == "approved"
    assert resolved["resolved_by"] == "sec-lead"
    assert resolved["new_trust_score"] > initial_score

    # Status filter test: should not appear in pending
    pending_ids = {e["escalation_id"] for e in B.escalations(status="pending")}
    assert esc_id not in pending_ids
    approved_ids = {e["escalation_id"] for e in B.escalations(status="approved")}
    assert esc_id in approved_ids


def test_escalation_rejection_workflow_lowers_trust():
    agent_id = "agent-to-reject"
    # Seed agent with an allow so it's above minimum
    L.record(agent_id, "boundary_allow")
    d = B.evaluate(
        "inference.tier3",
        classification="regulated",
        model_group="premium-best",
        agent_id=agent_id
    )
    esc_id = d.escalation["escalation_id"]
    initial_score = L.snapshot(agent_id).score

    # Reject the escalation
    resolved = B.resolve_escalation(
        escalation_id=esc_id,
        verdict="rejected",
        reviewer="sec-lead",
        reason="prohibited access to regulated data"
    )
    assert resolved["status"] == "rejected"
    assert resolved["new_trust_score"] < initial_score


def test_resolve_nonexistent_escalation_raises():
    try:
        B.resolve_escalation("esc_does_not_exist", "approved")
        assert False, "should have raised KeyError"
    except KeyError:
        pass


def test_resolve_invalid_verdict_raises():
    d = B.evaluate("inference.tier3", classification="regulated", model_group="premium-best", agent_id="a1")
    esc_id = d.escalation["escalation_id"]
    try:
        B.resolve_escalation(esc_id, "maybe")
        assert False, "should have raised ValueError"
    except ValueError:
        pass
