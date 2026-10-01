"""Trust API endpoints: snapshot, leaderboard, external feedback recording.
Only exercises endpoints that don't require a live LiteLLM backend
(consistent with test_auth.py) — chat()'s agent_id wiring is covered by
unit tests on boundary.evaluate() in test_trust.py instead.

Note on shared state: reloading "main" re-executes its module namespace
in place, and auth_middleware's closure reads the DISPATCH_API_KEY global
from that shared namespace at call time — not at the time each app
instance was built. So this file must NOT unset the key test_auth.py set;
doing so would silently disable auth on test_auth.py's already-built
client too, regardless of test collection order. Converging on the same
key value and authenticating every request here keeps both files correct
independent of which one runs first."""
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "dispatcher"))
os.environ["MEMORY_DB"] = tempfile.mktemp(suffix=".db")
os.environ["TRUST_DB"] = tempfile.mktemp(suffix=".db")
os.environ["DISPATCH_API_KEY"] = "test-key-123"

import main as main_mod
importlib.reload(main_mod)
from fastapi.testclient import TestClient

client = TestClient(main_mod.app)
AUTH = {"Authorization": "Bearer test-key-123"}


def test_new_agent_snapshot_is_neutral():
    r = client.get("/trust/never-seen-before", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["score"] == 0.5
    assert body["tier"] == "standard"
    assert body["events"] == 0
    assert body["recent_events"] == []

def test_event_endpoint_records_and_returns_new_score():
    r = client.post("/trust/api-test-agent/event",
                    json={"event_type": "eval_fail", "detail": "unit test"},
                    headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["agent_id"] == "api-test-agent"
    assert body["recorded"] == "eval_fail"
    assert body["new_score"] < 0.5  # a failure should lower it from neutral

def test_event_endpoint_rejects_boundary_events():
    # boundary_allow/boundary_deny are recorded automatically by
    # evaluate() only — the public API can't inject them directly.
    r = client.post("/trust/api-test-agent/event",
                    json={"event_type": "boundary_allow"}, headers=AUTH)
    assert r.status_code == 400

def test_event_endpoint_rejects_unknown_type():
    r = client.post("/trust/api-test-agent/event",
                    json={"event_type": "made_up_event"}, headers=AUTH)
    assert r.status_code == 400

def test_leaderboard_includes_recorded_agent():
    client.post("/trust/leaderboard-test-agent/event",
               json={"event_type": "human_approve"}, headers=AUTH)
    r = client.get("/trust", headers=AUTH)
    assert r.status_code == 200
    ids = {row["agent_id"] for row in r.json()}
    assert "leaderboard-test-agent" in ids

def test_recent_events_visible_after_recording():
    client.post("/trust/history-test-agent/event",
               json={"event_type": "eval_pass", "detail": "case-42"},
               headers=AUTH)
    r = client.get("/trust/history-test-agent", headers=AUTH)
    events = r.json()["recent_events"]
    assert any(e["detail"] == "case-42" for e in events)

def test_trust_endpoints_also_require_auth():
    r = client.get("/trust/some-agent")
    assert r.status_code == 401
