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

import json
import logging
import os
import time
import uuid
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
    # Phase 2: Dynamic rate limiting per trust tier
    "rate_limits": {
        "default_rpm": 60,
        "probation_rpm": 10,
        "trusted_rpm": 180,
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
        self._init_escalations_db()

    def _init_escalations_db(self):
        """Phase 2: Persistent escalation store on trust ledger SQLite db."""
        if hasattr(self.trust, "db") and self.trust.db is not None:
            try:
                with self.trust.db:
                    self.trust.db.executescript("""
                        CREATE TABLE IF NOT EXISTS boundary_escalations (
                            escalation_id TEXT PRIMARY KEY,
                            action TEXT NOT NULL,
                            agent_id TEXT NOT NULL,
                            status TEXT NOT NULL DEFAULT 'pending',
                            reason TEXT NOT NULL,
                            audit TEXT NOT NULL,
                            how_to_approve TEXT DEFAULT '',
                            created_at REAL NOT NULL,
                            resolved_at REAL,
                            resolved_by TEXT,
                            resolution_reason TEXT
                        );
                        CREATE INDEX IF NOT EXISTS idx_esc_status
                            ON boundary_escalations(status);
                    """)
            except Exception as e:
                logger.warning(f"failed to initialize escalations db ({e})")

    def _stamp(self, action: str, classification: str, extra: dict) -> dict:
        """Residency tag + audit ID at the boundary (Pattern 2)."""
        base = {
            "audit_id": f"aud_{uuid.uuid4().hex[:16]}",
            "timestamp": time.time(),
            "residency": self.policy["residency"],
            "classification": classification,
            "action": action,
        }
        for k, v in extra.items():
            if v is not None and v != "":
                base[k] = v
        return base

    def tier_ceiling(self, classification: str) -> int:
        ceilings = self.policy["classification_tier_ceiling"]
        return int(ceilings.get(
            classification,
            ceilings.get(self.policy["default_classification"], 3)))

    def evaluate(self, action: str, classification: str | None = None,
                 model_group: str = "", estimated_cost_usd: float = 0.0,
                 documents_chars: int = 0,
                 agent_id: str = "unattributed",
                 organization_id: str = "",
                 project_id: str = "") -> Decision:
        classification = (classification
                          or self.policy["default_classification"])
        snap = self.trust.snapshot(agent_id)
        _, cost_multiplier, force_approval_above = tier_for_score(snap.score)
        extra = {
            "model_group": model_group,
            "agent_id": agent_id,
            "trust_score": snap.score,
            "trust_tier": snap.tier_name,
        }
        if organization_id:
            extra["organization_id"] = organization_id
        if project_id:
            extra["project_id"] = project_id

        audit = self._stamp(action, classification, extra)
        trust_info = {"agent_id": agent_id, "score": snap.score,
                      "confidence": snap.confidence,
                      "tier": snap.tier_name, "events": snap.events}

        def record_esc(esc_action: str, reason: str, how_to_approve: str = "") -> dict:
            esc_id = f"esc_{uuid.uuid4().hex[:12]}"
            esc = {
                "escalation_id": esc_id,
                "action": esc_action,
                "agent_id": agent_id,
                "status": "pending",
                "reason": reason,
                "audit": audit,
                "how_to_approve": how_to_approve or "approve via POST /boundary/escalations/{id}/verdict",
                "created_at": time.time(),
            }
            self._escalations.append(esc)
            if hasattr(self.trust, "db") and self.trust.db is not None:
                try:
                    with self.trust.db:
                        self.trust.db.execute(
                            "INSERT OR REPLACE INTO boundary_escalations "
                            "(escalation_id, action, agent_id, status, reason, audit, "
                            "how_to_approve, created_at) VALUES (?,?,?,?,?,?,?,?)",
                            (esc_id, esc_action, agent_id, "pending", reason,
                             json.dumps(audit), esc["how_to_approve"], esc["created_at"]))
                except Exception as e:
                    logger.warning(f"failed to persist escalation ({e})")
            return esc

        def deny(reason: str, escalation: dict | None = None) -> Decision:
            self.trust.record(agent_id, "boundary_deny", detail=reason[:200])
            return Decision(False, action, reason, audit, escalation,
                            trust_info)

        # 1. Action-level autonomy
        level = self.policy["actions"].get(action, "approval")
        if level == "forbidden":
            return deny(f"action '{action}' is forbidden by policy")
        if level == "approval":
            esc = record_esc(action, f"action '{action}' requires approval",
                             "raise the action's autonomy level in config/autonomy_boundary.yaml "
                             "or approve via POST /boundary/escalations/{id}/verdict")
            return deny(f"action '{action}' requires approval", esc)

        # 2. Sovereignty: classification drives tier ceiling
        if model_group:
            tier = group_tier(model_group)
            ceiling = self.tier_ceiling(classification)
            if tier > ceiling:
                reason = (f"classification '{classification}' caps "
                          f"autonomous routing at tier {ceiling}; "
                          f"'{model_group}' is tier {tier}")
                esc = record_esc(action, reason)
                return deny(reason, esc)

            # 2b. Trust: low-trust agents need approval above a tightened
            # ceiling, even when the classification ceiling would allow
            # it. This can only NARROW autonomy, never widen it past what
            # step 2 already granted.
            if force_approval_above is not None and tier > force_approval_above:
                reason = (f"agent '{agent_id}' is on trust tier "
                          f"'{snap.tier_name}' (score {snap.score}); "
                          f"tier {tier} requires approval until "
                          f"trust improves")
                esc = record_esc(action, reason)
                return deny(reason, esc)

        # 3. Blast radius: cost + document size.
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

    def resolve_escalation(self, escalation_id: str, verdict: str,
                           reviewer: str = "security-admin",
                           reason: str = "") -> dict:
        """Resolve a pending escalation (approve or reject).
        
        Approve: records human_approve in trust ledger, raising agent trust.
        Reject: records human_reject in trust ledger, penalizing agent trust.
        """
        verdict = verdict.lower().strip()
        if verdict not in ("approved", "rejected"):
            raise ValueError("verdict must be 'approved' or 'rejected'")

        now = time.time()
        agent_id = "unattributed"
        found = None

        # Check in-memory list
        for esc in reversed(self._escalations):
            if esc.get("escalation_id") == escalation_id:
                esc["status"] = verdict
                esc["resolved_at"] = now
                esc["resolved_by"] = reviewer
                esc["resolution_reason"] = reason
                agent_id = esc.get("agent_id") or esc.get("audit", {}).get("agent_id", "unattributed")
                found = dict(esc)
                break

        # Check database
        if hasattr(self.trust, "db") and self.trust.db is not None:
            try:
                row = self.trust.db.execute(
                    "SELECT * FROM boundary_escalations WHERE escalation_id=?",
                    (escalation_id,)).fetchone()
                if row:
                    agent_id = row["agent_id"]
                    with self.trust.db:
                        self.trust.db.execute(
                            "UPDATE boundary_escalations SET status=?, resolved_at=?, "
                            "resolved_by=?, resolution_reason=? WHERE escalation_id=?",
                            (verdict, now, reviewer, reason, escalation_id))
                    if found is None:
                        found = {
                            "escalation_id": row["escalation_id"],
                            "action": row["action"],
                            "agent_id": row["agent_id"],
                            "status": verdict,
                            "reason": row["reason"],
                            "audit": json.loads(row["audit"] or "{}"),
                            "resolved_at": now,
                            "resolved_by": reviewer,
                            "resolution_reason": reason,
                        }
            except Exception as e:
                logger.warning(f"failed to update escalation in db: {e}")

        if found is None:
            raise KeyError(f"escalation '{escalation_id}' not found")

        # Record human verdict in trust ledger
        event_type = "human_approve" if verdict == "approved" else "human_reject"
        self.trust.record(agent_id, event_type,
                          detail=f"escalation {escalation_id}: {reason or verdict}")

        new_snap = self.trust.snapshot(agent_id)
        found["new_trust_score"] = new_snap.score
        found["new_trust_tier"] = new_snap.tier_name
        return found

    def clamp_group_to_ceiling(self, model_group: str,
                               classification: str | None,
                               task_route: list[str]) -> str:
        """Instead of denying, degrade: pick the first group in the task's
        route that fits under the classification's tier ceiling. Falls back
        to local."""
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

    def escalations(self, status: str | None = None, limit: int = 50) -> list[dict]:
        """Return escalations, optionally filtered by status (pending, approved, rejected)."""
        if hasattr(self.trust, "db") and self.trust.db is not None:
            try:
                if status:
                    rows = self.trust.db.execute(
                        "SELECT * FROM boundary_escalations WHERE status=? "
                        "ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
                else:
                    rows = self.trust.db.execute(
                        "SELECT * FROM boundary_escalations ORDER BY created_at DESC LIMIT ?",
                        (limit,)).fetchall()
                results = []
                for r in rows:
                    results.append({
                        "escalation_id": r["escalation_id"],
                        "action": r["action"],
                        "agent_id": r["agent_id"],
                        "status": r["status"],
                        "reason": r["reason"],
                        "audit": json.loads(r["audit"] or "{}"),
                        "how_to_approve": r["how_to_approve"],
                        "created_at": r["created_at"],
                        "resolved_at": r["resolved_at"],
                        "resolved_by": r["resolved_by"],
                        "resolution_reason": r["resolution_reason"],
                    })
                return results
            except Exception:
                pass

        res = self._escalations
        if status:
            res = [e for e in res if e.get("status") == status]
        return res[-limit:]

    def policy_view(self) -> dict:
        pending = len(self.escalations(status="pending"))
        return {**self.policy,
                "pending_escalations": pending}


boundary = AutonomyBoundary()
