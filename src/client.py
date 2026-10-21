"""
DISPATCH Developer Ergonomics & Client SDK Wrapper.

Enables seamless paved-road adoption for product engineering teams:
```python
from dispatch import GatewayClient, PriorityTier

client = GatewayClient(tenant_id="sec-ops-prod", priority=PriorityTier.TIER_0_CRITICAL)
response = client.execute_step(
    workflow_id="inc-8841",
    system_prefix=SYSTEM_POLICY,
    tools=MCP_TOOL_SCHEMAS,
    messages=turn_history,
    max_workflow_budget_usd=1.00
)
```
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.gateway import GatewayEngine, GatewayResponse
from src.quota import PriorityTier

logger = logging.getLogger("dispatch.client")


class GatewayClient:
    """Enterprise Gateway Client SDK wrapper."""

    def __init__(
        self,
        tenant_id: str = "default-tenant",
        priority: PriorityTier = PriorityTier.TIER_1_INTERNAL,
        workspace_id: Optional[str] = None,
        gateway_engine: Optional[GatewayEngine] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.tenant_id = tenant_id
        self.priority = priority
        self.workspace_id = workspace_id
        self.engine = gateway_engine or GatewayEngine()
        self.default_headers = default_headers or {
            "X-Tenant-BU": tenant_id,
            "X-Workflow-Type": "agent-step",
            "X-Environment": "production",
        }

    def execute_step(
        self,
        workflow_id: str,
        system_prefix: str = "",
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        max_workflow_budget_usd: Optional[float] = 1.50,
        policy: str = "dispatch_full",
        step_index: int = 1,
        total_steps: int = 10,
        headers: Optional[Dict[str, str]] = None,
    ) -> GatewayResponse:
        """Execute a single agent step with automated prefix caching, quota gating, and telemetry."""
        merged_headers = dict(self.default_headers)
        if headers:
            merged_headers.update(headers)

        return self.engine.execute_request(
            tenant_id=self.tenant_id,
            workflow_id=workflow_id,
            priority=self.priority,
            system_prefix=system_prefix,
            tools=tools,
            messages=messages,
            workspace_id=self.workspace_id,
            policy=policy,
            step_index=step_index,
            total_steps=total_steps,
            max_workflow_budget_usd=max_workflow_budget_usd,
            headers=merged_headers,
        )
