"""
DISPATCH MCP server — exposes routing, memory, budget, and boundary
controls as MCP tools so Claude Code / Claude Desktop / any MCP client
can use the cluster natively.

Install:
    pip install "mcp[cli]" httpx

Register in Claude Code:
    claude mcp add dispatch -- python /path/to/dispatcher/mcp_server.py

Register in Claude Desktop (claude_desktop_config.json):
    {"mcpServers": {"dispatch": {"command": "python",
      "args": ["/path/to/dispatcher/mcp_server.py"],
      "env": {"DISPATCH_URL": "http://localhost:8080"}}}}
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

DISPATCH_URL = os.getenv("DISPATCH_URL", "http://localhost:8080")
DISPATCH_API_KEY = os.getenv("DISPATCH_API_KEY", "")
_HEADERS = ({"Authorization": f"Bearer {DISPATCH_API_KEY}"}
            if DISPATCH_API_KEY else {})

mcp = FastMCP("dispatch")


def _post(path: str, body: dict) -> dict:
    r = httpx.post(f"{DISPATCH_URL}{path}", json=body, timeout=300.0,
                   headers=_HEADERS)
    r.raise_for_status()
    return r.json()


def _get(path: str) -> dict:
    r = httpx.get(f"{DISPATCH_URL}{path}", timeout=30.0, headers=_HEADERS)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def dispatch_infer(prompt: str, task: str = "", strategy: str = "cost_first",
                   classification: str = "internal",
                   session_id: str = "", max_tokens: int = 2048) -> dict:
    """Route an inference request through the DISPATCH cluster.

    Picks the cheapest capable backend (local hardware first, then free
    quotas, then budget/premium cloud) under the autonomy-boundary policy.
    Use classification='regulated' to force local-only routing for
    sensitive data. Returns the answer plus routing metadata (which
    backend, latency, audit stamp).
    """
    body = {"model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "classification": classification,
            "strategy": strategy}
    if task:
        body["task"] = task
    if session_id:
        body["session_id"] = session_id
    data = _post("/v1/chat/completions", body)
    return {"answer": data["choices"][0]["message"]["content"],
            "routing": data.get("_dispatch", {})}


@mcp.tool()
def dispatch_route_preview(prompt: str) -> dict:
    """Preview how DISPATCH would route a prompt without executing it:
    task classification, selected model group, and fallback chain."""
    return _post("/dispatch/classify", {"prompt": prompt})


@mcp.tool()
def dispatch_status() -> dict:
    """Cluster status: tier availability, budget spend vs. cap, and
    memory engine statistics."""
    return {"tiers": _get("/dispatch/tiers"),
            "memory": _get("/memory/stats")}


@mcp.tool()
def dispatch_memory_search(query: str) -> dict:
    """Search DISPATCH's semantic memory for durable facts (preferences,
    entities, constraints, goals) learned from previous sessions."""
    return _post("/memory/search", {"query": query})


@mcp.tool()
def dispatch_boundary_policy() -> dict:
    """View the autonomy-boundary policy: classification tier ceilings,
    per-request cost caps, action autonomy levels, pending escalations."""
    return _get("/boundary/policy")


if __name__ == "__main__":
    mcp.run()
