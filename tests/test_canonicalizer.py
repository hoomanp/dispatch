from dispatch.canonicalizer import canonicalize_payload


def test_prefix_determinism_across_timestamps_and_key_order():
    header_a = "System policy active at 2026-10-21T08:15:30Z req=123e4567-e89b-12d3-a456-426614174000"
    header_b = "System policy active at 2026-10-21T19:45:00Z req=987e6543-e21b-12d3-a456-426614174999"

    tools_a = [{"name": "fetch_logs", "schema": {"service": "auth", "limit": 50}}]
    tools_b = [{"schema": {"limit": 50, "service": "auth"}, "name": "fetch_logs"}]

    session_a = {"region": "us-west-2", "tier": "prod"}
    session_b = {"tier": "prod", "region": "us-west-2"}

    res_a = canonicalize_payload(header_a, tools_a, session_a, [{"role": "user", "content": "ping"}])
    res_b = canonicalize_payload(header_b, tools_b, session_b, [{"role": "user", "content": "pong"}])

    assert res_a["sticky_routing_key"] == res_b["sticky_routing_key"]
