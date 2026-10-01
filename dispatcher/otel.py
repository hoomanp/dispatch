"""
OpenTelemetry tracing — observability stack #1 (distributed tracing).

Opt-in: enabled only when OTEL_EXPORTER_OTLP_ENDPOINT is set
(e.g. http://jaeger:4318/v1/traces). When unset, everything here is a
no-op and DISPATCH runs with zero tracing overhead.

What you get when enabled:
  - A span per inbound request (FastAPI auto-instrumentation)
  - A child span per outbound httpx call, with W3C traceparent headers
    propagated to LiteLLM — which also speaks OTel, so traces continue
    dispatcher -> LiteLLM -> provider
  - Routing attributes (task, model group, tier, audit id) on the
    request span, set from main.py via set_routing_attributes()
"""

import logging
import os

logger = logging.getLogger("dispatch.otel")

_ENABLED = bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))


def setup(app) -> bool:
    """Instrument the app + httpx. Returns True if tracing is active."""
    if not _ENABLED:
        logger.info("otel: OTEL_EXPORTER_OTLP_ENDPOINT unset — tracing off")
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter)
        from opentelemetry.instrumentation.fastapi import (
            FastAPIInstrumentor)
        from opentelemetry.instrumentation.httpx import (
            HTTPXClientInstrumentor)
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create(
            {"service.name": os.getenv("OTEL_SERVICE_NAME", "dispatch")}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app,
                                           excluded_urls="health,metrics")
        HTTPXClientInstrumentor().instrument()
        logger.info("otel: tracing active -> %s",
                    os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
        return True
    except ImportError as e:
        logger.warning("otel: packages missing (%s); tracing off", e)
        return False


def set_routing_attributes(task: str, model_group: str, tier: int,
                           audit_id: str = "") -> None:
    """Attach routing decision to the current request span. No-op when
    tracing is off."""
    if not _ENABLED:
        return
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        span.set_attribute("dispatch.task", task)
        span.set_attribute("dispatch.model_group", model_group)
        span.set_attribute("dispatch.tier", tier)
        if audit_id:
            span.set_attribute("dispatch.audit_id", audit_id)
    except Exception:
        pass
