import hashlib
import json
import re
from typing import Any, Dict, List

TIMESTAMP_REGEX = re.compile(
    r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)
UUID_REGEX = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def canonicalize_payload(
    system_prefix: str,
    tools: List[Dict[str, Any]],
    session_context: Dict[str, Any],
    dynamic_turns: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Enforces a strict 3-block prompt contract so that sequential turns in an agent
    workflow preserve byte-identical prefixes for upstream KV cache hits.
    """
    # Block 1: Immutable System Prefix + Canonicalized MCP Tool Schemas
    clean_prefix = TIMESTAMP_REGEX.sub("[STATIC_TS]", system_prefix)
    clean_prefix = UUID_REGEX.sub("[STATIC_UUID]", clean_prefix)

    canonical_tools_str = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    block_1_immutable = f"SYS:{clean_prefix}|TOOLS:{canonical_tools_str}"

    # Block 2: Session-Scoped Context (pinned inside 300s cache TTL)
    canonical_session_str = json.dumps(session_context, sort_keys=True, separators=(",", ":"))
    block_2_session = f"CTX:{canonical_session_str}"

    # Compute deterministic sticky routing hash for worker affinity
    routing_material = f"{block_1_immutable}||{block_2_session}".encode("utf-8")
    sticky_routing_key = hashlib.sha256(routing_material).hexdigest()

    # Block 3: Dynamic Conversation & Tool Execution Tail
    block_3_dynamic = [
        {"role": str(turn.get("role", "user")), "content": str(turn.get("content", ""))}
        for turn in dynamic_turns
    ]

    return {
        "block_1_immutable": block_1_immutable,
        "block_2_session": block_2_session,
        "block_3_dynamic": block_3_dynamic,
        "sticky_routing_key": sticky_routing_key,
    }
