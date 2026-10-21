"""
Unit tests for DISPATCH Module 1: 3-Block Prompt Canonicalizer.
"""

import hashlib
import json
import time
import pytest

from dispatch import (
    CanonicalPrompt,
    ConsistentHashRing,
    PromptCanonicalizer,
    canonicalize_tools,
    strip_ephemeral_tokens,
)


def test_strip_ephemeral_tokens():
    text_with_uuid = "Session ID: a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d. Execute plan."
    cleaned = strip_ephemeral_tokens(text_with_uuid)
    assert "a1b2c3d4" not in cleaned
    assert "[UUID_STRIPPED]" in cleaned

    text_with_iso_timestamp = "Generated at: 2026-10-21T12:00:00Z by system."
    cleaned_iso = strip_ephemeral_tokens(text_with_iso_timestamp)
    assert "2026-10-21" not in cleaned_iso

    text_with_header = "Current Time: Wednesday, 21 Oct 2026 12:00:00. Policy active."
    cleaned_header = strip_ephemeral_tokens(text_with_header)
    assert "Wednesday" not in cleaned_header


def test_mcp_tool_schema_deterministic_sorting():
    tools_unsorted = [
        {
            "name": "restart_service",
            "parameters": {
                "timeout": 30,
                "service_name": "auth",
                "force": True,
            },
        },
        {
            "name": "check_health",
            "parameters": {
                "verbose": False,
                "endpoint": "/healthz",
            },
        },
    ]

    tools_reordered = [
        {
            "parameters": {
                "endpoint": "/healthz",
                "verbose": False,
            },
            "name": "check_health",
        },
        {
            "parameters": {
                "force": True,
                "service_name": "auth",
                "timeout": 30,
            },
            "name": "restart_service",
        },
    ]

    c1 = canonicalize_tools(tools_unsorted)
    c2 = canonicalize_tools(tools_reordered)
    # Must be byte-for-byte identical despite different key/element insertion order
    assert c1 == c2
    parsed = json.loads(c1)
    assert parsed[0]["name"] == "check_health"
    assert parsed[1]["name"] == "restart_service"


def test_3_block_prompt_stratification():
    canon = PromptCanonicalizer()
    system_text = "Standard corporate policy. UUID: 12345678-1234-1234-1234-123456789abc"
    tools = [{"name": "tool_b"}, {"name": "tool_a"}]
    messages_turn_1 = [{"role": "user", "content": "Query 1"}]

    res_1 = canon.canonicalize(
        system_prefix=system_text,
        tools=tools,
        tenant_id="tenant-alpha",
        workspace_id="ws-main",
        messages=messages_turn_1,
    )

    # Turn 2: dynamic history changes, but static prefix (Block 1 + Block 2) remains identical
    messages_turn_2 = [
        {"role": "user", "content": "Query 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Query 2"},
    ]

    res_2 = canon.canonicalize(
        system_prefix=system_text,
        tools=tools,
        tenant_id="tenant-alpha",
        workspace_id="ws-main",
        messages=messages_turn_2,
    )

    # Block 1 & 2 must be identical
    assert res_1.block_1_static == res_2.block_1_static
    assert res_1.block_2_session == res_2.block_2_session
    # Prefix hashes MUST match across multi-turn agent execution
    assert res_1.prefix_hash == res_2.prefix_hash

    # Block 3 must capture dynamic turns
    assert "Query 1" in res_1.block_3_dynamic
    assert "Query 2" in res_2.block_3_dynamic


def test_consistent_hashing_affinity():
    nodes = ["node-us-east-1", "node-us-east-2", "node-us-west-1"]
    ring = ConsistentHashRing(nodes=nodes, replicas=64)

    test_prefix_hash = hashlib.sha256(b"static_system_prefix_block").hexdigest()
    chosen_node_1 = ring.get_node(test_prefix_hash)
    chosen_node_2 = ring.get_node(test_prefix_hash)

    # Identical prefix must route to exact same node
    assert chosen_node_1 in nodes
    assert chosen_node_1 == chosen_node_2


def test_cross_tenant_namespace_isolation():
    """Different tenants with byte-identical prompt must produce different HMAC cache keys."""
    canon = PromptCanonicalizer(tenant_secret="super-secret-salt-123")
    shared_system = "You are a cloud operations assistant."

    res_bu_a = canon.canonicalize(
        system_prefix=shared_system,
        tenant_id="business-unit-A",
        messages=[{"role": "user", "content": "Hello"}],
    )

    res_bu_b = canon.canonicalize(
        system_prefix=shared_system,
        tenant_id="business-unit-B",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert res_bu_a.tenant_cache_key != res_bu_b.tenant_cache_key
