"""
DISPATCH 3-Block Prompt Canonicalizer (Module 1).

Enforces deterministic prompt stratification to prevent transformer KV cache invalidation
and route requests via consistent hashing so multi-step agent workflows hit warm provider
prefix caches.

Block 1 (Static):
  - System instructions and corporate policy headers.
  - MCP tool schemas with recursively sorted JSON keys.
  - Regex sanitization stripping non-deterministic timestamps and UUIDs.

Block 2 (Session):
  - Tenant ID and workspace context pinned for the session TTL (300s).
  - Cryptographic tenant HMAC salting to prevent cross-tenant cache contamination.

Block 3 (Dynamic):
  - Variable tool outputs and latest user turns appended strictly at the tail.
"""

import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Patterns that inject non-determinism into static system prompts
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
    r"|\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d{2}:\d{2}:\d{2}\b"
)
_TIME_OF_DAY_RE = re.compile(
    r"\b(?:Current Time|Timestamp|Generated at|Date):\s*[^,\n\r]+\b",
    re.IGNORECASE,
)


def strip_ephemeral_tokens(text: str) -> str:
    """Strip dynamic timestamps, generated-at markers, and UUIDs from static prompt headers."""
    if not text:
        return ""
    cleaned = _TIME_OF_DAY_RE.sub("[SYSTEM_TIMESTAMP_STRIPPED]", text)
    cleaned = _TIMESTAMP_RE.sub("[TIMESTAMP_STRIPPED]", cleaned)
    cleaned = _UUID_RE.sub("[UUID_STRIPPED]", cleaned)
    return cleaned.strip()


def sort_json_structures(obj: Any) -> Any:
    """Recursively sort dictionary keys and standardize JSON data structures."""
    if isinstance(obj, dict):
        return {k: sort_json_structures(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [sort_json_structures(item) for item in obj]
    return obj


def canonicalize_tools(tools: Optional[Sequence[Dict[str, Any]]]) -> str:
    """Format and sort tool schemas deterministically."""
    if not tools:
        return ""
    sorted_tools = [sort_json_structures(t) for t in tools]
    # Sort tools by name if available, else by json representation
    sorted_tools.sort(key=lambda t: t.get("name") or json.dumps(t, sort_keys=True))
    return json.dumps(sorted_tools, sort_keys=True, separators=(",", ":"))


@dataclass
class CanonicalPrompt:
    block_1_static: str
    block_2_session: str
    block_3_dynamic: str
    prefix_hash: str
    tenant_cache_key: str
    full_prompt: str


class ConsistentHashRing:
    """Consistent hashing ring for mapping prefix hashes to provider nodes."""

    def __init__(self, nodes: Optional[List[str]] = None, replicas: int = 64):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node: str) -> None:
        for i in range(self.replicas):
            v_key = f"{node}#replica-{i}"
            h = int(hashlib.md5(v_key.encode("utf-8")).hexdigest(), 16)
            self.ring[h] = node
        self.sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node: str) -> None:
        for i in range(self.replicas):
            v_key = f"{node}#replica-{i}"
            h = int(hashlib.md5(v_key.encode("utf-8")).hexdigest(), 16)
            self.ring.pop(h, None)
        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        if not self.ring:
            return None
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        # Binary search for the first point >= h
        import bisect
        idx = bisect.bisect_right(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]


class PromptCanonicalizer:
    """3-Block Prompt Canonicalizer ensuring deterministic prefix caching."""

    def __init__(
        self,
        tenant_secret: str = "dispatch-secret-salt-default",
        session_ttl_seconds: float = 300.0,
        provider_nodes: Optional[List[str]] = None,
    ):
        self.tenant_secret = tenant_secret.encode("utf-8")
        self.session_ttl_seconds = session_ttl_seconds
        self.hash_ring = ConsistentHashRing(
            provider_nodes or ["provider-node-1", "provider-node-2", "provider-node-3"]
        )
        self._session_cache: Dict[str, Tuple[str, float]] = {}

    def get_session_context(
        self, tenant_id: str, workspace_id: Optional[str] = None
    ) -> str:
        """Return pinned session context for tenant and workspace within TTL."""
        key = f"{tenant_id}:{workspace_id or 'default'}"
        now = time.time()
        if key in self._session_cache:
            ctx, expiry = self._session_cache[key]
            if now < expiry:
                return ctx

        # Pin new session context for 300s
        context_data = {
            "tenant_id": tenant_id,
            "workspace_id": workspace_id or "default",
            "session_ttl_seconds": int(self.session_ttl_seconds),
        }
        ctx_str = json.dumps(context_data, sort_keys=True, separators=(",", ":"))
        self._session_cache[key] = (ctx_str, now + self.session_ttl_seconds)
        return ctx_str

    def canonicalize(
        self,
        system_prefix: str,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tenant_id: str = "default-tenant",
        workspace_id: Optional[str] = None,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        model_id: str = "frontier-tier",
    ) -> CanonicalPrompt:
        """Stratify prompt into 3 deterministic blocks and compute prefix hash."""
        # 1. Block 1: Static (System policy + MCP Tool Schemas with sorted JSON keys, stripped of ephemeral tokens)
        clean_system = strip_ephemeral_tokens(system_prefix or "")
        tools_block = canonicalize_tools(tools)
        if tools_block:
            block_1 = f"{clean_system}\n[TOOL_SCHEMAS]\n{tools_block}".strip()
        else:
            block_1 = clean_system

        # 2. Block 2: Session (Tenant + Workspace pinned context)
        block_2 = self.get_session_context(tenant_id=tenant_id, workspace_id=workspace_id)

        # 3. Block 3: Dynamic (Variable tool outputs and latest conversation turns)
        dynamic_parts: List[str] = []
        if messages:
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    # Multi-part content
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            parts.append(part["text"])
                        else:
                            parts.append(json.dumps(part, sort_keys=True))
                    content_str = "\n".join(parts)
                else:
                    content_str = str(content)
                dynamic_parts.append(f"[{role.upper()}]: {content_str}")
        block_3 = "\n\n".join(dynamic_parts)

        # Compute deterministic prefix hash over Block 1 + Block 2
        prefix_payload = f"{block_1}\n---\n{block_2}".encode("utf-8")
        prefix_hash = hashlib.sha256(prefix_payload).hexdigest()

        # Compute cryptographic HMAC tenant-isolated cache key:
        # CacheKey = HMAC-SHA256(TenantSecret, TenantID || ModelID || CanonicalPrompt)
        canonical_prompt_repr = f"{block_1}::{block_2}::{block_3}"
        hmac_input = f"{tenant_id}:{model_id}:{canonical_prompt_repr}".encode("utf-8")
        tenant_cache_key = hmac.new(self.tenant_secret, hmac_input, hashlib.sha256).hexdigest()

        full_prompt = f"{block_1}\n\n[SESSION_CONTEXT]\n{block_2}\n\n[DYNAMIC_HISTORY]\n{block_3}"

        return CanonicalPrompt(
            block_1_static=block_1,
            block_2_session=block_2,
            block_3_dynamic=block_3,
            prefix_hash=prefix_hash,
            tenant_cache_key=tenant_cache_key,
            full_prompt=full_prompt,
        )

    def route_prefix(self, prefix_hash: str) -> str:
        """Route prefix_hash to a sticky provider node via consistent hashing."""
        node = self.hash_ring.get_node(prefix_hash)
        return node or "provider-node-1"
