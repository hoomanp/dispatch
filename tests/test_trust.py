import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dispatcher"))

os.environ["AUTONOMY_CONFIG"] = str(ROOT / "config/autonomy_boundary.yaml")
os.environ["TRUST_DB"] = tempfile.mktemp(suffix=".db")

import importlib

import trust_ledger
importlib.reload(trust_ledger)
from trust_ledger import TrustLedger, tier_for_score

import autonomy_boundary
importlib.reload(autonomy_boundary)
from autonomy_boundary import AutonomyBoundary

L = TrustLedger(Path(os.environ["TRUST_DB"]))
# Inject L explicitly rather than relying on the module-singleton `trust`
# picked up via reload order — this is what makes B's trust reads
# immune to any OTHER test file reloading trust_ledger.py later in the
# same pytest session (reload mutates the shared module namespace in
# place; explicit injection sidesteps that entirely).
B = AutonomyBoundary(ROOT / "config/autonomy_boundary.yaml", trust_ledger=L)


# ── Ledger mechanics ──────────────────────────────────────────────────────

def test_unknown_agent_starts_neutral():
    snap = L.snapshot("brand-new-agent")
    assert snap.score == 0.5
    assert snap.events == 0
    assert snap.tier_name == "standard"

def test_allow_events_raise_score():
    for _ in range(10):
        L.record("t-good-agent", "boundary_allow")
    snap = L.snapshot("t-good-agent")
    assert snap.score > 0.5
    assert snap.events == 10

def test_deny_events_lower_score_faster_than_allow_raises_it():
    # bad events are weighted heavier (4.0) than good events (1.0)
    for _ in range(3):
        L.record("t-mixed-agent", "boundary_allow")
    for _ in range(3):
        L.record("t-mixed-agent", "boundary_deny")
    snap = L.snapshot("t-mixed-agent")
    assert snap.score < 0.5

def test_reject_event_hits_hardest():
    L.record("t-rejected-agent", "boundary_allow")
    before = L.snapshot("t-rejected-agent").score
    L.record("t-rejected-agent", "human_reject")
    after = L.snapshot("t-rejected-agent").score
    assert after < before

def test_confidence_grows_with_evidence():
    for _ in range(50):
        L.record("t-confident-agent", "boundary_allow")
    c_many = L.snapshot("t-confident-agent").confidence
    L.record("t-sparse-agent", "boundary_allow")
    c_few = L.snapshot("t-sparse-agent").confidence
    assert c_many > c_few

def test_unknown_event_type_rejected():
    try:
        L.record("t-bad-event", "not_a_real_event")
        assert False, "should have raised"
    except ValueError:
        pass

def test_recent_events_ordered_newest_first():
    L.record("t-history", "boundary_allow", detail="first")
    L.record("t-history", "boundary_deny", detail="second")
    events = L.recent_events("t-history", limit=5)
    assert events[0]["detail"] == "second"
    assert events[1]["detail"] == "first"

def test_leaderboard_returns_snapshots():
    L.record("t-leaderboard-1", "boundary_allow")
    L.record("t-leaderboard-2", "boundary_allow")
    ids = {s.agent_id for s in L.leaderboard(limit=100)}
    assert "t-leaderboard-1" in ids and "t-leaderboard-2" in ids

def test_decay_pulls_score_toward_neutral():
    for _ in range(20):
        L.record("t-decay-agent", "boundary_allow")
    fresh_score = L.snapshot("t-decay-agent").score
    # simulate two full half-lives passing
    from trust_ledger import HALF_LIFE_SECONDS
    row = L.db.execute(
        "SELECT alpha, beta FROM trust_agents WHERE agent_id=?",
        ("t-decay-agent",)).fetchone()
    old_time = time.time() - 2 * HALF_LIFE_SECONDS
    L.db.execute(
        "UPDATE trust_agents SET last_event_at=? WHERE agent_id=?",
        (old_time, "t-decay-agent"))
    L.db.commit()
    decayed_score = L.snapshot("t-decay-agent").score
    assert abs(decayed_score - 0.5) < abs(fresh_score - 0.5)


# ── Trust tiers ──────────────────────────────────────────────────────────

def test_tier_for_score_boundaries():
    assert tier_for_score(0.0)[0] == "probation"
    assert tier_for_score(0.39)[0] == "probation"
    assert tier_for_score(0.4)[0] == "standard"
    assert tier_for_score(0.79)[0] == "standard"
    assert tier_for_score(0.8)[0] == "trusted"
    assert tier_for_score(1.0)[0] == "trusted"

def test_trusted_agent_gets_wider_cost_cap():
    for _ in range(30):
        L.record("t-earned-trust", "boundary_allow")
    assert L.snapshot("t-earned-trust").tier_name == "trusted"
    d = B.evaluate("inference.tier0", model_group="local-fast",
                   estimated_cost_usd=0.75, agent_id="t-earned-trust")
    assert d.allowed  # base cap 0.50 * 2.0 multiplier = 1.00, within it

def test_trusted_agent_effective_cap_is_base_times_multiplier():
    # base 0.50 * trusted multiplier 2.0 = 1.00 effective cap — not the
    # hard ceiling (2.00), just wider than the base/standard-tier cap.
    for _ in range(30):
        L.record("t-cap-math", "boundary_allow")
    assert L.snapshot("t-cap-math").tier_name == "trusted"
    allowed = B.evaluate("inference.tier0", model_group="local-fast",
                        estimated_cost_usd=0.99, agent_id="t-cap-math")
    denied = B.evaluate("inference.tier0", model_group="local-fast",
                       estimated_cost_usd=1.01, agent_id="t-cap-math")
    assert allowed.allowed and not denied.allowed

def test_hard_ceiling_clamps_even_if_base_cap_is_raised():
    # Trust widens the base cap, but never past hard_max_cost_per_request_usd
    # — verified by pushing the base cap itself far above the hard ceiling
    # and confirming the clamp still holds for a maximally trusted agent.
    from autonomy_boundary import AutonomyBoundary
    generous = AutonomyBoundary(ROOT / "config/autonomy_boundary.yaml",
                               trust_ledger=L)
    generous.policy["max_cost_per_request_usd"] = 5.0
    generous.policy["hard_max_cost_per_request_usd"] = 2.0
    for _ in range(50):
        L.record("t-hard-ceiling", "boundary_allow")
    assert L.snapshot("t-hard-ceiling").tier_name == "trusted"
    within = generous.evaluate("inference.tier0", model_group="local-fast",
                              estimated_cost_usd=1.99,
                              agent_id="t-hard-ceiling")
    over = generous.evaluate("inference.tier0", model_group="local-fast",
                            estimated_cost_usd=2.01,
                            agent_id="t-hard-ceiling")
    assert within.allowed
    assert not over.allowed  # 5.0 * 2.0 = 10.0 would allow this; hard cap doesn't

def test_probation_agent_needs_approval_above_tier1():
    for _ in range(3):
        L.record("t-probation-agent", "boundary_deny")
    assert L.snapshot("t-probation-agent").tier_name == "probation"
    d = B.evaluate("inference.tier2", classification="internal",
                   model_group="budget-general", agent_id="t-probation-agent")
    assert not d.allowed
    assert "trust tier" in d.reason

def test_probation_agent_still_allowed_at_tier0():
    for _ in range(3):
        L.record("t-probation-local", "boundary_deny")
    assert L.snapshot("t-probation-local").tier_name == "probation"
    d = B.evaluate("inference.tier0", classification="internal",
                   model_group="local-fast", agent_id="t-probation-local")
    assert d.allowed  # trust tightens autonomy, never below what tier0 needs

def test_trust_cannot_widen_past_classification_ceiling():
    # even a maximally trusted agent cannot escape sovereignty: regulated
    # data still cannot reach a cloud tier, regardless of trust score.
    for _ in range(50):
        L.record("t-trusted-but-regulated", "boundary_allow")
    assert L.snapshot("t-trusted-but-regulated").tier_name == "trusted"
    d = B.evaluate("inference.tier3", classification="regulated",
                   model_group="premium-best",
                   agent_id="t-trusted-but-regulated")
    assert not d.allowed

def test_new_agent_defaults_to_standard_not_probation():
    # a never-seen agent should not start locked out — score 0.5 is
    # "standard", so first-contact agents aren't penalized for being new.
    d = B.evaluate("inference.tier3", classification="internal",
                   model_group="premium-best", agent_id="t-brand-new")
    assert d.allowed
