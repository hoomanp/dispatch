"""
Autonomy boundary — policy gate at the architectural boundary.

Synthesized from the enterprise agent architecture patterns:
  - Sovereignty as a primary axis: every action stamps a residency tag and
    an audit ID at the boundary; data classification drives tier selection
    automatically; cross-border/cloud flows are explicit and policy-gated.
  - Control plane / data plane separation: this module IS control plane —
    policies, approvals, escalations — separate from inference execution.
  - Blast-radius limits: per-request cost ceilings; premium tiers require
    explicit approval above the autonomous boundary.
  - Observability: every decision (allow, deny, escalate) is auditable.

Config lives in config/autonomy_boundary.yaml. The evaluate() call is the
single choke point; the dispatcher may not reach a provider without a
Decision from it.
"""

import os
import time
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from trust_ledger import tier_for_score, trust

logger = logging.getLogger("dispatch.autonomy")

CONFIG_PATH = Path(os.getenv(
    "AUTONOMY_CONFIG", "/app/config/autonomy_boundary.yaml"))

DEFAULT_POLICY = {
    "residency": "local-cluster",
    "default_classification": "internal",
    # classification -> highest tier the system may use AUTONOMOUSLY.
    # tiers: 0 local, 1 free-quota cloud, 2 budget cloud, 3 premium cloud
    "classification_tier_ceiling": {
        "regulated": 0,     # never leaves local hardware
        "confidential": 2,
        "internal": 3,
        "public": 3,
    },
    # blast-radius caps
    "max_cost_per_request_usd": 0.50,
    # absolute ceiling — trust score can widen max_cost_per_request_usd
    # for trusted agents, but NEVER past this. Hard limits are not
    # earnable; see trust_ledger.py.
    "hard_max_cost_per_request_usd": 2.00,
    "max_documents_chars": 50_000_000,
    # action -> autonomy level:
    #   autonomous       allowed, audited
    #   approval         denied with escalation record (HTTP 403)
    #   forbidden        denied, no escalation path
    "actions": {
        "inference.tier0": "autonomous",
        "inference.tier1": "autonomous",
        "inference.tier2": "autonomous",
        "inference.tier3": "autonomous",
        "memory.write": "autonomous",
        "memory.fact_extraction": "autonomous",
        "documents.compress": "autonomous",
    },
}

TIER_OF_GROUP = {
    "local": 0, "free": 1, "budget": 2, "premium": 3,
}


def group_tier(model_group: str) -> int:
    for prefix, tier in TIER_OF_GROUP.items():
        if model_group.startswith(prefix):
            return tier
    return 3  # unknown groups treated as most-restricted


@dataclass
class Decision:
    allowed: bool
    action: str
    reason: str
    audit: dict = field(default_factory=dict)
    escalation: dict | None = None
    trust: dict | None = None


class AutonomyBoundary:
    def __init__(self, config_path: Path = CONFIG_PATH, trust_ledger=None):
        # Stored as instance state (self.trust), not resolved as a bare
        # module global at call time — that would let a LATER reload of
        # trust_ledger.py (e.g. by an unrelated test module) silently
        # redirect an already-constructed instance to a different trust
        # store. Explicit injection makes each instance's dependency
        # fixed at construction, which is what test isolation needs and
        # what production correctness needs too.
        self.trust = trust_ledger if trust_ledger is not None else trust
        self.policy = dict(DEFAULT_POLICY)
        if config_path.exists():
            try:
                loaded = yaml.safe_load(config_path.read_text()) or {}
                # shallow-merge top level; nested dicts replaced wholesale
                for k, v in loaded.items():
                    self.policy[k] = v
                logger.info(f"autonomy policy loaded from {config_path}")
            except Exception as e:
                logger.error(f"autonomy config invalid ({e}); using defaults")
        self._escalations: list[dict] = []

    def _stamp(self, action: str, classification: str, extra: dict) -> dict:
        """Residency tag + audit ID at the boundary (Pattern 2)."""
        return {
            "audit_id": f"aud_{uuid.uuid4().hex[:16]}",
            "timestamp": time.time(),
            "residency": self.policy["residency"],
            "classification": classification,
            "action": action,
            **extra,
        }

    def tier_ceiling(self, classification: str) -> int:
        ceilings = self.policy["classification_tier_ceiling"]
        return int(ceilings.get(
            classification,
            ceilings.get(self.policy["default_classification"], 3)))

    def evaluate(self, action: str, classification: str | None = None,
                 model_group: str = "", estimated_cost_usd: float = 0.0,
                 documents_chars: int = 0,
                 agent_id: str = "unattributed") -> Decision:
        classification = (classification
                          or self.policy["default_classification"])
        snap = self.trust.snapshot(agent_id)
        _, cost_multiplier, force_approval_above = tier_for_score(snap.score)
        audit = self._stamp(action, classification,
                            {"model_group": model_group,
                             "agent_id": agent_id,
                             "trust_score": snap.score,
                             "trust_tier": snap.tier_name})
        trust_info = {"agent_id": agent_id, "score": snap.score,
                      "confidence": snap.confidence,
                      "tier": snap.tier_name, "events": snap.events}

        def deny(reason: str, escalation: dict | None = None) -> Decision:
            self.trust.record(agent_id, "boundary_deny", detail=reason[:200])
            return Decision(False, action, reason, audit, escalation,
                            trust_info)

        # 1. Action-level autonomy
        level = self.policy["actions"].get(action, "approval")
        if level == "forbidden":
            return deny(f"action '{action}' is forbidden by policy")
        if level == "approval":
            esc = {"escalation_id": f"esc_{uuid.uuid4().hex[:12]}",
                   "action": action, "audit": audit,
                   "how_to_approve": "raise the action's autonomy level in "
                                     "config/autonomy_boundary.yaml"}
            self._escalations.append(esc)
            return deny(f"action '{action}' requires approval", esc)

        # 2. Sovereignty: classification drives tier ceiling
        if model_group:
            tier = group_tier(model_group)
            ceiling = self.tier_ceiling(classification)
            if tier > ceiling:
                esc = {"escalation_id": f"esc_{uuid.uuid4().hex[:12]}",
                       "action": action, "audit": audit,
                       "reason": f"classification '{classification}' caps "
                                 f"autonomous routing at tier {ceiling}; "
                                 f"'{model_group}' is tier {tier}"}
                self._escalations.append(esc)
                return deny(esc["reason"], esc)

            # 2b. Trust: low-trust agents need approval above a tightened
            # ceiling, even when the classification ceiling would allow
            # it. This can only NARROW autonomy, never widen it past what
            # step 2 already granted — a probation agent never gets more
            # room than its data classification allows.
            if force_approval_above is not None and tier > force_approval_above:
                esc = {"escalation_id": f"esc_{uuid.uuid4().hex[:12]}",
                       "action": action, "audit": audit,
                       "reason": f"agent '{agent_id}' is on trust tier "
                                 f"'{snap.tier_name}' (score {snap.score}); "
                                 f"tier {tier} requires approval until "
                                 f"trust improves"}
                self._escalations.append(esc)
                return deny(esc["reason"], esc)

        # 3. Blast radius: cost + document size. Trust can widen the cost
        # cap for well-established agents, but never past the absolute
        # hard ceiling — trust earns room within policy, it never
        # overrides policy.
        effective_cap = min(
            float(self.policy["max_cost_per_request_usd"]) * cost_multiplier,
            float(self.policy["hard_max_cost_per_request_usd"]))
        if estimated_cost_usd > effective_cap:
            return deny(
                f"estimated cost ${estimated_cost_usd:.2f} exceeds "
                f"per-request cap ${effective_cap:.2f} "
                f"(trust tier: {snap.tier_name})")
        if documents_chars > int(self.policy["max_documents_chars"]):
            return deny(
                f"documents ({documents_chars:,} chars) exceed cap "
                f"({int(self.policy['max_documents_chars']):,})")

        self.trust.record(agent_id, "boundary_allow")
        return Decision(True, action, "within autonomy boundary", audit,
                        trust=trust_info)

    def clamp_group_to_ceiling(self, model_group: str,
                               classification: str | None,
                               task_route: list[str]) -> str:
        """Instead of denying, degrade: pick the first group in the task's
        route that fits under the classification's tier ceiling. Falls back
        to local. This is the 'sovereignty drives tier selection
        automatically' behavior."""
        ceiling = self.tier_ceiling(
            classification or self.policy["default_classification"])
        if group_tier(model_group) <= ceiling:
            return model_group
        for g in task_route:
            if group_tier(g) <= ceiling:
                logger.info(
                    f"autonomy: '{model_group}' over tier ceiling "
                    f"{ceiling}; degraded to '{g}'")
                return g
        return "local-always-on"

    def escalations(self, limit: int = 50) -> list[dict]:
        return self._escalations[-limit:]

    def policy_view(self) -> dict:
        return {**self.policy,
                "pending_escalations": len(self._escalations)}


boundary = AutonomyBoundary()
