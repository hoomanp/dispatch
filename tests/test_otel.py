"""OTel module: strict no-op when endpoint unset."""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "dispatcher"))
os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
import otel
importlib.reload(otel)


def test_setup_noop_when_unset():
    class FakeApp: pass
    assert otel.setup(FakeApp()) is False

def test_attributes_noop_when_unset():
    otel.set_routing_attributes("coding", "local-fast", 0, "aud_x")  # no raise
