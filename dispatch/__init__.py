"""
DISPATCH Public Client Interface.

Allows:
```python
from dispatch import GatewayClient, PriorityTier
```
"""

import sys
from pathlib import Path

# Ensure src is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.canonicalizer import (
    CanonicalPrompt,
    ConsistentHashRing,
    PromptCanonicalizer,
    canonicalize_tools,
    strip_ephemeral_tokens,
)
from src.circuit_breaker import (
    BudgetExceededError,
    CircuitBreaker,
    CircuitBreakerTrippedError,
    CircuitState,
    ExecutionOutcome,
    ProviderHealthTracker,
)
from src.client import GatewayClient
from src.gateway import GatewayEngine, GatewayResponse
from src.quota import (
    HierarchicalQuotaManager,
    PriorityTier,
    QuotaExceededError,
    QuotaPreemptedError,
    TenantQuotaConfig,
)
from src.telemetry import (
    FinOpsRecord,
    TelemetryCollector,
    finops_telemetry,
)

__all__ = [
    "GatewayClient",
    "PriorityTier",
    "PromptCanonicalizer",
    "ConsistentHashRing",
    "CanonicalPrompt",
    "canonicalize_tools",
    "strip_ephemeral_tokens",
    "HierarchicalQuotaManager",
    "QuotaExceededError",
    "QuotaPreemptedError",
    "TenantQuotaConfig",
    "CircuitBreaker",
    "CircuitBreakerTrippedError",
    "BudgetExceededError",
    "CircuitState",
    "ExecutionOutcome",
    "ProviderHealthTracker",
    "TelemetryCollector",
    "FinOpsRecord",
    "finops_telemetry",
    "GatewayEngine",
    "GatewayResponse",
]
