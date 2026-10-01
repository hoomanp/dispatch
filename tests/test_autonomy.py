import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "dispatcher"))

os.environ["AUTONOMY_CONFIG"] = str(ROOT / "config/autonomy_boundary.yaml")
os.environ["TRUST_DB"] = tempfile.mktemp(suffix=".db")

import importlib

import trust_ledger
importlib.reload(trust_ledger)
import autonomy_boundary
importlib.reload(autonomy_boundary)
from autonomy_boundary import AutonomyBoundary, group_tier
from trust_ledger import TrustLedger

# Own isolated trust ledger, injected explicitly — not the module
# singleton picked up via reload order (see test_trust.py for why).
_L = TrustLedger(Path(os.environ["TRUST_DB"]))
B = AutonomyBoundary(ROOT / "config/autonomy_boundary.yaml", trust_ledger=_L)

# Every test uses its own agent_id so trust events from one test never
# influence another's starting trust score (fresh agent = score 0.5,
# tier "standard", no forced approval beyond base policy).


def test_group_tier():
    assert group_tier("local-fast") == 0
    assert group_tier("free-fast") == 1
    assert group_tier("budget-general") == 2
    assert group_tier("premium-best") == 3
    assert group_tier("weird-thing") == 3   # unknown = most restricted

def test_regulated_denied_on_cloud():
    d = B.evaluate("inference.tier3", classification="regulated",
                   model_group="premium-balanced", agent_id="t-regulated-deny")
    assert not d.allowed and d.escalation is not None

def test_regulated_allowed_local():
    d = B.evaluate("inference.tier0", classification="regulated",
                   model_group="local-fast", agent_id="t-regulated-allow")
    assert d.allowed

def test_internal_allowed_premium():
    d = B.evaluate("inference.tier3", classification="internal",
                   model_group="premium-best", agent_id="t-internal-premium")
    assert d.allowed

def test_audit_stamp_present():
    d = B.evaluate("inference.tier0", model_group="local-fast",
                   agent_id="t-audit-stamp")
    assert d.audit["audit_id"].startswith("aud_")
    assert d.audit["residency"] == "local-cluster"
    assert "classification" in d.audit
    assert d.audit["agent_id"] == "t-audit-stamp"

def test_document_blast_radius():
    d = B.evaluate("documents.compress", documents_chars=10**9,
                   agent_id="t-doc-radius")
    assert not d.allowed and "exceed" in d.reason

def test_clamp_degrades_not_denies():
    route = ["premium-gemini", "budget-fast", "free-longcontext",
             "local-always-on"]
    g = B.clamp_group_to_ceiling("premium-gemini", "regulated", route)
    assert g == "local-always-on"
    g2 = B.clamp_group_to_ceiling("premium-gemini", "confidential", route)
    assert group_tier(g2) <= 2

def test_unknown_action_requires_approval():
    d = B.evaluate("tools.shell_exec", agent_id="t-unknown-action")
    assert not d.allowed and d.escalation is not None

def test_cost_within_base_cap_allowed():
    d = B.evaluate("inference.tier0", model_group="local-fast",
                   estimated_cost_usd=0.10, agent_id="t-cost-ok")
    assert d.allowed

def test_cost_over_base_cap_denied_for_new_agent():
    # A fresh agent (standard tier, multiplier 1.0x) is capped at the
    # base policy value (0.50), not the hard ceiling (2.00).
    d = B.evaluate("inference.tier0", model_group="local-fast",
                   estimated_cost_usd=0.75, agent_id="t-cost-over-base")
    assert not d.allowed
    assert "0.50" in d.reason

def test_decision_carries_trust_info():
    d = B.evaluate("inference.tier0", model_group="local-fast",
                   agent_id="t-trust-info")
    assert d.trust is not None
    assert d.trust["agent_id"] == "t-trust-info"
    assert 0.0 <= d.trust["score"] <= 1.0
