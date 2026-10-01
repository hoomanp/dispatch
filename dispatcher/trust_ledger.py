"""
Agent trust ledger — per-agent trust scoring for the autonomy boundary.

Design rules:
  - Trust is computed from evidence recorded OUTSIDE the agent (boundary
    decisions, eval outcomes, human approve/reject verdicts). An agent
    never reports its own trustworthiness.
  - Beta distribution per agent: alpha accumulates good evidence, beta
    accumulates bad evidence. Mean = trust score, width = confidence.
    A new agent with 5 clean actions scores lower-confidence than one
    with 5,000 — the ledger reflects that, not just a raw percentage.
  - Bad evidence weighs more than good evidence, and old evidence decays
    (half-life), so trust is slow to earn and fast to lose.
  - Trust NEVER unlocks a hard limit (payments, regulated-data egress,
    destructive actions). It only widens or narrows autonomy WITHIN what
    the base policy already allows. See tier_for_score() and its use in
    AutonomyBoundary.evaluate().

Storage: SQLite, same durability class as the memory engine.
"""

import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

def _resolve_db_path(env_var: str, default_filename: str) -> Path:
    env_val = os.getenv(env_var)
    if env_val:
        p = Path(env_val)
    elif os.path.exists("/data") and os.access("/data", os.W_OK):
        p = Path("/data") / default_filename
    else:
        p = Path(__file__).resolve().parent.parent / "data" / default_filename
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        p = Path.cwd() / "data" / default_filename
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


DB_PATH = _resolve_db_path("TRUST_DB", "trust.db")

# Evidence weights: bad events cost more trust than good events earn.
WEIGHTS = {
    "boundary_allow": 1.0,
    "boundary_deny": 4.0,
    "eval_pass": 1.0,
    "eval_fail": 3.0,
    "human_approve": 2.0,
    "human_reject": 6.0,
}

HALF_LIFE_SECONDS = 14 * 24 * 3600  # evidence weight halves every 14 days

# Trust tiers -> extra autonomy tightening/loosening applied on top of the
# base classification ceiling. Multiplier scales max_cost_per_request;
# approval_extra_tiers forces approval on tiers otherwise autonomous.
TRUST_TIERS = [
    # (min_score, name, cost_multiplier, force_approval_above_tier)
    (0.0, "probation", 0.2, 1),   # tiers 2-3 need approval regardless of policy
    (0.4, "standard", 1.0, None), # base policy applies unchanged
    (0.8, "trusted", 2.0, None),  # wider cost cap; base ceilings still apply
]


def tier_for_score(score: float) -> tuple[str, float, int | None]:
    """Return (name, cost_multiplier, force_approval_above_tier) for a score."""
    chosen = TRUST_TIERS[0]
    for row in TRUST_TIERS:
        if score >= row[0]:
            chosen = row
    return chosen[1], chosen[2], chosen[3]


@dataclass
class TrustSnapshot:
    agent_id: str
    score: float          # Beta mean, 0..1
    confidence: float      # 0..1, higher = more evidence
    alpha: float
    beta: float
    events: int
    tier_name: str


class TrustLedger:
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = DB_PATH
        self.db = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;
            CREATE TABLE IF NOT EXISTS trust_agents (
                agent_id TEXT PRIMARY KEY,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta REAL NOT NULL DEFAULT 1.0,
                events INTEGER NOT NULL DEFAULT 0,
                last_event_at REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trust_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                weight REAL NOT NULL,
                detail TEXT DEFAULT '',
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trust_events_agent
                ON trust_events(agent_id);
        """)
        self.db.commit()

    def _decay(self, alpha: float, beta: float, last_event_at: float,
              now: float) -> tuple[float, float]:
        """Exponential decay toward the uninformative prior (1,1) — old
        evidence loses influence so a stale agent isn't judged on ancient
        history, but a truly inactive agent also doesn't accumulate trust
        for doing nothing."""
        if last_event_at <= 0:
            return alpha, beta
        elapsed = max(0.0, now - last_event_at)
        decay = math.pow(0.5, elapsed / HALF_LIFE_SECONDS)
        alpha = 1.0 + (alpha - 1.0) * decay
        beta = 1.0 + (beta - 1.0) * decay
        return alpha, beta

    def record(self, agent_id: str, event_type: str, detail: str = "") -> None:
        if event_type not in WEIGHTS:
            raise ValueError(f"unknown trust event type: {event_type!r}")
        weight = WEIGHTS[event_type]
        now = time.time()
        row = self.db.execute(
            "SELECT alpha, beta, events, last_event_at FROM trust_agents "
            "WHERE agent_id=?", (agent_id,)).fetchone()
        if row is None:
            alpha, beta, events = 1.0, 1.0, 0
        else:
            alpha, beta = self._decay(row["alpha"], row["beta"],
                                      row["last_event_at"], now)
            events = row["events"]
        is_good = event_type.endswith(("allow", "pass", "approve"))
        if is_good:
            alpha += weight
        else:
            beta += weight
        with self.db:
            self.db.execute(
                "INSERT INTO trust_agents (agent_id, alpha, beta, events, "
                "last_event_at, created_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(agent_id) DO UPDATE SET "
                "alpha=excluded.alpha, beta=excluded.beta, "
                "events=excluded.events, last_event_at=excluded.last_event_at",
                (agent_id, alpha, beta, events + 1, now, now))
            self.db.execute(
                "INSERT INTO trust_events (agent_id, event_type, weight, "
                "detail, timestamp) VALUES (?,?,?,?,?)",
                (agent_id, event_type, weight, detail, now))

    def snapshot(self, agent_id: str) -> TrustSnapshot:
        row = self.db.execute(
            "SELECT alpha, beta, events, last_event_at FROM trust_agents "
            "WHERE agent_id=?", (agent_id,)).fetchone()
        if row is None:
            alpha, beta, events = 1.0, 1.0, 0
        else:
            alpha, beta = self._decay(row["alpha"], row["beta"],
                                      row["last_event_at"], time.time())
            events = row["events"]
        score = alpha / (alpha + beta)
        # Confidence: how peaked the Beta is relative to the uninformative
        # prior — grows toward 1 as total evidence (alpha+beta) grows.
        confidence = 1.0 - (2.0 / (alpha + beta))
        confidence = max(0.0, min(1.0, confidence))
        tier_name, _, _ = tier_for_score(score)
        return TrustSnapshot(agent_id=agent_id, score=round(score, 4),
                            confidence=round(confidence, 4),
                            alpha=round(alpha, 3), beta=round(beta, 3),
                            events=events, tier_name=tier_name)

    def recent_events(self, agent_id: str, limit: int = 20) -> list[dict]:
        rows = self.db.execute(
            "SELECT event_type, weight, detail, timestamp FROM trust_events "
            "WHERE agent_id=? ORDER BY timestamp DESC LIMIT ?",
            (agent_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def leaderboard(self, limit: int = 50) -> list[TrustSnapshot]:
        rows = self.db.execute(
            "SELECT agent_id FROM trust_agents ORDER BY last_event_at DESC "
            "LIMIT ?", (limit,)).fetchall()
        return [self.snapshot(r["agent_id"]) for r in rows]


trust = TrustLedger()
