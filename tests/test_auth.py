"""Auth middleware: enforced only when DISPATCH_API_KEY is set."""
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "dispatcher"))
os.environ["MEMORY_DB"] = tempfile.mktemp(suffix=".db")
os.environ["DISPATCH_API_KEY"] = "test-key-123"

import main as main_mod
importlib.reload(main_mod)
from fastapi.testclient import TestClient

client = TestClient(main_mod.app)


def test_health_exempt():
    assert client.get("/health").status_code == 200

def test_metrics_exempt():
    assert client.get("/metrics").status_code == 200

def test_protected_endpoint_rejects_no_key():
    assert client.get("/dispatch/tiers").status_code == 401

def test_protected_endpoint_rejects_wrong_key():
    r = client.get("/dispatch/tiers",
                   headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401

def test_bearer_key_accepted():
    r = client.get("/dispatch/tiers",
                   headers={"Authorization": "Bearer test-key-123"})
    assert r.status_code == 200

def test_x_api_key_accepted():
    """Anthropic SDK sends x-api-key, not Bearer."""
    r = client.get("/boundary/policy",
                   headers={"x-api-key": "test-key-123"})
    assert r.status_code == 200

def test_root_is_protected():
    assert client.get("/").status_code == 401
