"""
DISPATCH Memory Engine

Four memory types mapped to three temperature tiers:

  HOT   (0ms)    Working memory    - active session, in-process
  WARM  (~50ms)  Episodic memory   - past sessions, SQLite + Qdrant
  COLD  (~200ms) Semantic memory   - distilled facts, SQLite FTS5 + Qdrant
  COLD           Procedural memory - agent prompts/config, SQLite

Context beyond any window: hierarchical map-reduce compression using a
small local model (default phi4-mini via Ollama). Free and private.

All memory operations are local-first:
  Embeddings:  nomic-embed-text via Ollama
  Compression: phi4-mini via Ollama
  Structured:  SQLite (WAL mode, FTS5)
  Vectors:     Qdrant
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger("dispatch.memory")

QDRANT_URL     = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "nomic-embed-text")
COMPRESS_MODEL = os.getenv("COMPRESS_MODEL", "phi4-mini")


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


DB_PATH = _resolve_db_path("MEMORY_DB", "memory.db")

COLLECTION_EPISODIC = "dispatch_episodic"
COLLECTION_SEMANTIC = "dispatch_semantic"
VECTOR_DIM = 768  # nomic-embed-text


@dataclass
class Episode:
    id: str
    session_id: str
    timestamp: float
    role: str
    content: str
    task_type: str = ""
    provider_id: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class SemanticFact:
    id: str
    fact: str
    category: str          # preference | entity | skill | constraint | goal
    confidence: float
    source_episodes: list[str]
    created_at: float
    updated_at: float
    valid: bool = True


@dataclass
class WorkingContext:
    session_id: str
    system_prompt: str
    memory_injection: str
    conversation_history: list[dict]
    total_tokens_estimated: int
    sources: dict


# ── SQLite store ──────────────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = DB_PATH
        self.db = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                task_type TEXT,
                provider_id TEXT,
                tokens_used INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_ep_session ON episodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_ep_time ON episodes(timestamp);

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                ended_at REAL,
                title TEXT,
                summary TEXT,
                total_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0,
                episode_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS semantic_facts (
                id TEXT PRIMARY KEY,
                fact TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                source_episodes TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                valid INTEGER DEFAULT 1
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
                USING fts5(fact_id UNINDEXED, fact, category);

            CREATE TABLE IF NOT EXISTS procedural_prompts (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                tools TEXT DEFAULT '[]',
                model_preference TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self.db.commit()

    # Episodes
    def save_episode(self, ep: Episode):
        self.db.execute(
            "INSERT OR REPLACE INTO episodes "
            "(id,session_id,timestamp,role,content,task_type,provider_id,"
            "tokens_used,cost_usd,metadata) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ep.id, ep.session_id, ep.timestamp, ep.role, ep.content,
             ep.task_type, ep.provider_id, ep.tokens_used, ep.cost_usd,
             json.dumps(ep.metadata)))
        self.db.commit()

    def get_session_episodes(self, session_id: str, limit: int = 50) -> list[Episode]:
        rows = self.db.execute(
            "SELECT * FROM episodes WHERE session_id=? "
            "ORDER BY timestamp DESC LIMIT ?", (session_id, limit)).fetchall()
        return [self._row_to_episode(r) for r in reversed(rows)]

    def _row_to_episode(self, r) -> Episode:
        return Episode(
            id=r["id"], session_id=r["session_id"], timestamp=r["timestamp"],
            role=r["role"], content=r["content"],
            task_type=r["task_type"] or "", provider_id=r["provider_id"] or "",
            tokens_used=r["tokens_used"], cost_usd=r["cost_usd"],
            metadata=json.loads(r["metadata"] or "{}"))

    # Facts
    def save_fact(self, fact: SemanticFact):
        self.db.execute(
            "INSERT OR REPLACE INTO semantic_facts "
            "(id,fact,category,confidence,source_episodes,created_at,"
            "updated_at,valid) VALUES (?,?,?,?,?,?,?,?)",
            (fact.id, fact.fact, fact.category, fact.confidence,
             json.dumps(fact.source_episodes), fact.created_at,
             fact.updated_at, int(fact.valid)))
        self.db.execute("DELETE FROM facts_fts WHERE fact_id=?", (fact.id,))
        self.db.execute(
            "INSERT INTO facts_fts (fact_id,fact,category) VALUES (?,?,?)",
            (fact.id, fact.fact, fact.category))
        self.db.commit()

    def search_facts_fts(self, query: str, limit: int = 10) -> list[SemanticFact]:
        query = query.strip()
        if not query:
            return self.list_facts(limit=limit)
        # Sanitize for FTS5: keep alphanumeric words, OR them
        words = [w for w in "".join(
            c if c.isalnum() or c.isspace() else " " for c in query
        ).split() if len(w) > 2]
        if not words:
            return self.list_facts(limit=limit)
        fts_query = " OR ".join(words[:8])
        try:
            rows = self.db.execute(
                "SELECT sf.* FROM semantic_facts sf "
                "JOIN facts_fts f ON sf.id = f.fact_id "
                "WHERE facts_fts MATCH ? AND sf.valid=1 "
                "ORDER BY rank LIMIT ?", (fts_query, limit)).fetchall()
            return [self._row_to_fact(r) for r in rows]
        except sqlite3.OperationalError:
            return self.list_facts(limit=limit)

    def list_facts(self, category: Optional[str] = None,
                   limit: int = 50) -> list[SemanticFact]:
        if category:
            rows = self.db.execute(
                "SELECT * FROM semantic_facts WHERE category=? AND valid=1 "
                "ORDER BY confidence DESC LIMIT ?", (category, limit)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM semantic_facts WHERE valid=1 "
                "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def _row_to_fact(self, r) -> SemanticFact:
        return SemanticFact(
            id=r["id"], fact=r["fact"], category=r["category"],
            confidence=r["confidence"],
            source_episodes=json.loads(r["source_episodes"] or "[]"),
            created_at=r["created_at"], updated_at=r["updated_at"],
            valid=bool(r["valid"]))

    # Sessions
    def start_session(self, session_id: str, title: str = ""):
        self.db.execute(
            "INSERT OR IGNORE INTO sessions (id,started_at,title) VALUES (?,?,?)",
            (session_id, time.time(), title))
        self.db.commit()

    def end_session(self, session_id: str, summary: str = "",
                    total_tokens: int = 0, total_cost: float = 0):
        self.db.execute(
            "UPDATE sessions SET ended_at=?, summary=?, total_tokens=?, "
            "total_cost=?, episode_count=(SELECT COUNT(*) FROM episodes "
            "WHERE session_id=?) WHERE id=?",
            (time.time(), summary, total_tokens, total_cost,
             session_id, session_id))
        self.db.commit()

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]

    def memory_stats(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {
            "total_episodes": q("SELECT COUNT(*) FROM episodes"),
            "total_sessions": q("SELECT COUNT(*) FROM sessions"),
            "total_facts": q("SELECT COUNT(*) FROM semantic_facts WHERE valid=1"),
            "total_cost_usd": q("SELECT COALESCE(SUM(cost_usd),0) FROM episodes"),
            "total_tokens": q("SELECT COALESCE(SUM(tokens_used),0) FROM episodes"),
        }


# ── Embeddings (local) ────────────────────────────────────────────────────────

async def embed(text: str) -> Optional[list[float]]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text[:8192]})
            r.raise_for_status()
            return r.json()["embedding"]
    except Exception as e:
        logger.warning(f"embed failed ({e}); vector features degraded")
        return None


# ── Qdrant helpers ────────────────────────────────────────────────────────────

async def ensure_collections():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for collection in (COLLECTION_EPISODIC, COLLECTION_SEMANTIC):
                await client.put(
                    f"{QDRANT_URL}/collections/{collection}",
                    json={"vectors": {"size": VECTOR_DIM, "distance": "Cosine"}})
    except Exception as e:
        logger.warning(f"Qdrant unavailable ({e}); semantic search degraded")


async def upsert_vector(collection: str, point_id: str,
                        vector: list[float], payload: dict):
    try:
        int_id = int(hashlib.sha256(point_id.encode()).hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.put(
                f"{QDRANT_URL}/collections/{collection}/points",
                json={"points": [{"id": int_id, "vector": vector,
                                  "payload": {**payload, "_id": point_id}}]})
    except Exception as e:
        logger.warning(f"Qdrant upsert failed: {e}")


async def search_vectors(collection: str, vector: list[float], limit: int = 5,
                         score_threshold: float = 0.72) -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{QDRANT_URL}/collections/{collection}/points/search",
                json={"vector": vector, "limit": limit,
                      "score_threshold": score_threshold,
                      "with_payload": True})
            if r.status_code == 200:
                return r.json().get("result", [])
    except Exception:
        pass
    return []


# ── Local compression ─────────────────────────────────────────────────────────

async def compress_with_local(text: str, instruction: str,
                              max_tokens: int = 512) -> str:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": COMPRESS_MODEL,
                      "prompt": f"{instruction}\n\n---\n{text}\n---\n\nOutput:",
                      "stream": False,
                      "options": {"temperature": 0.1,
                                  "num_predict": max_tokens}})
            r.raise_for_status()
            return r.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"local compression failed: {e}")
        return text[:2000]  # graceful degradation: truncate


# ── Context pipeline: >1M tokens via hierarchical map-reduce ──────────────────

CHUNK_CHARS = 24000  # ~6k tokens

class ContextPipeline:
    def __init__(self, target_model_limit_tokens: int = 100000):
        self.limit = target_model_limit_tokens
        self.limit_chars = self.limit * 4

    async def prepare(self, documents: list[str],
                      query: str = "") -> tuple[str, dict]:
        combined = "\n\n---\n\n".join(documents)
        total_chars = len(combined)
        if total_chars // 4 <= self.limit:
            return combined, {"strategy": "direct",
                              "original_chars": total_chars,
                              "compression_ratio": 1.0}
        return await self._map_reduce(combined, query, depth=0)

    async def _map_reduce(self, text: str, query: str,
                          depth: int) -> tuple[str, dict]:
        if depth > 3:  # safety cap
            return text[:self.limit_chars], {"strategy": "truncated"}
        chunks = self._chunk(text)
        logger.info(f"map-reduce depth={depth}: {len(chunks)} chunks")
        instruction = (
            "Summarize the following passage, preserving key facts, "
            "decisions, code identifiers, and anything relevant to: "
            f"'{query or 'the overall content'}'. Dense but complete.")
        maps = await asyncio.gather(
            *[compress_with_local(c, instruction) for c in chunks])
        combined = "\n\n---\n\n".join(maps)
        if len(combined) <= self.limit_chars:
            return combined, {"strategy": f"map_reduce_depth_{depth + 1}",
                              "original_chars": len(text),
                              "chunks": len(chunks),
                              "output_chars": len(combined),
                              "compression_ratio":
                                  round(len(combined) / max(len(text), 1), 3)}
        return await self._map_reduce(combined, query, depth + 1)

    def _chunk(self, text: str) -> list[str]:
        paragraphs = text.split("\n\n")
        chunks, current, size = [], [], 0
        for p in paragraphs:
            if size + len(p) > CHUNK_CHARS and current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            current.append(p)
            size += len(p)
        if current:
            chunks.append("\n\n".join(current))
        return chunks


# ── Memory manager ────────────────────────────────────────────────────────────

class MemoryManager:
    def __init__(self):
        self.store = MemoryStore()
        self.pipeline = ContextPipeline()
        self._sessions: dict[str, list[Episode]] = {}

    async def startup(self):
        await ensure_collections()
        logger.info("memory engine ready")

    def start_session(self, session_id: str, title: str = ""):
        self.store.start_session(session_id, title)
        self._sessions.setdefault(session_id, [])

    def record(self, episode: Episode):
        self._sessions.setdefault(episode.session_id, []).append(episode)
        self.store.save_episode(episode)

    async def end_session(self, session_id: str):
        episodes = self._sessions.pop(session_id, None)
        if episodes is None:
            episodes = self.store.get_session_episodes(session_id)
        if not episodes:
            self.store.end_session(session_id)
            return

        session_text = "\n".join(
            f"[{ep.role.upper()}]: {ep.content[:1500]}"
            for ep in episodes[-30:])

        summary = await compress_with_local(
            session_text,
            "Summarize this conversation in 3-5 sentences: main topics, "
            "decisions made, outcomes.")

        facts_json = await compress_with_local(
            session_text,
            'Extract persistent, reusable facts about the user or project. '
            'Return ONLY a JSON array like '
            '[{"fact":"...","category":"preference|skill|entity|constraint|goal"}]. '
            'Skip conversation-specific details.')

        try:
            start = facts_json.find("[")
            end = facts_json.rfind("]") + 1
            if start != -1 and end > start:
                for f in json.loads(facts_json[start:end])[:20]:
                    if not isinstance(f, dict) or "fact" not in f:
                        continue
                    fact = SemanticFact(
                        id=hashlib.sha256(f["fact"].encode()).hexdigest()[:12],
                        fact=f["fact"],
                        category=f.get("category", "entity"),
                        confidence=0.8,
                        source_episodes=[ep.id for ep in episodes[-5:]],
                        created_at=time.time(), updated_at=time.time())
                    self.store.save_fact(fact)
                    vec = await embed(fact.fact)
                    if vec:
                        await upsert_vector(COLLECTION_SEMANTIC, fact.id, vec,
                                            {"fact": fact.fact,
                                             "category": fact.category})
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.info("fact extraction returned unparseable output; skipped")

        total_tokens = sum(ep.tokens_used for ep in episodes)
        total_cost = sum(ep.cost_usd for ep in episodes)
        self.store.end_session(session_id, summary, total_tokens, total_cost)

        if summary:
            vec = await embed(summary)
            if vec:
                await upsert_vector(COLLECTION_EPISODIC, session_id, vec,
                                    {"session_id": session_id,
                                     "summary": summary})

    async def build_context(self, session_id: str, query: str,
                            documents: Optional[list[str]] = None,
                            system_prompt: str = "",
                            max_history_turns: int = 20) -> WorkingContext:
        sources: dict = {}
        query_vec = await embed(query) if query else None

        # HOT: conversation history
        hot = self._sessions.get(session_id, [])[-max_history_turns:]
        conversation = [{"role": ep.role, "content": ep.content}
                        for ep in hot if ep.role in ("user", "assistant")]
        sources["hot"] = {"turns": len(conversation)}

        # WARM: similar past sessions
        similar = []
        if query_vec:
            for r in await search_vectors(COLLECTION_EPISODIC, query_vec,
                                          limit=3, score_threshold=0.75):
                p = r.get("payload", {})
                if p.get("session_id") != session_id and p.get("summary"):
                    similar.append(p["summary"])
        sources["warm"] = {"similar_sessions": len(similar)}

        # COLD: semantic facts (FTS + vector, deduplicated)
        fact_texts: set[str] = {f.fact for f in
                                self.store.search_facts_fts(query, limit=5)}
        if query_vec:
            for r in await search_vectors(COLLECTION_SEMANTIC, query_vec,
                                          limit=5, score_threshold=0.78):
                fact = r.get("payload", {}).get("fact")
                if fact:
                    fact_texts.add(fact)
        sources["cold"] = {"facts": len(fact_texts)}

        # DOCUMENTS: compress if needed
        doc_text = ""
        if documents:
            doc_text, meta = await self.pipeline.prepare(documents, query)
            sources["documents"] = meta

        parts = []
        if similar:
            parts.append("Relevant context from previous sessions:\n" +
                         "\n".join(f"- {s}" for s in similar))
        if fact_texts:
            parts.append("Known facts about this user/project:\n" +
                         "\n".join(f"- {f}" for f in sorted(fact_texts)[:10]))
        if doc_text:
            parts.append(f"Documents:\n{doc_text}")
        memory_injection = "\n\n".join(parts)

        total_chars = (len(system_prompt) + len(memory_injection) +
                       sum(len(m["content"]) for m in conversation))
        return WorkingContext(
            session_id=session_id, system_prompt=system_prompt,
            memory_injection=memory_injection,
            conversation_history=conversation,
            total_tokens_estimated=total_chars // 4, sources=sources)

    def stats(self) -> dict:
        s = self.store.memory_stats()
        return {
            "hot": {"active_sessions": len(self._sessions),
                    "turns_in_memory":
                        sum(len(v) for v in self._sessions.values())},
            "warm": {"total_episodes": s["total_episodes"],
                     "total_sessions": s["total_sessions"]},
            "cold": {"semantic_facts": s["total_facts"]},
            "total_tokens_processed": s["total_tokens"],
            "total_cost_usd": round(s["total_cost_usd"], 4),
        }


memory = MemoryManager()
