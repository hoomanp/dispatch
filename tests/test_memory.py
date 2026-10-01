import sys, os, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "dispatcher"))
os.environ["MEMORY_DB"] = tempfile.mktemp(suffix=".db")
import importlib
import memory_engine
importlib.reload(memory_engine)
from memory_engine import MemoryStore, Episode, SemanticFact, ContextPipeline

import sqlite3
S = MemoryStore.__new__(MemoryStore)
S.db = sqlite3.connect(os.environ["MEMORY_DB"], check_same_thread=False)
S.db.row_factory = sqlite3.Row
S._init_schema()


def _fact(fid, text, cat="preference"):
    return SemanticFact(id=fid, fact=text, category=cat, confidence=0.9,
                        source_episodes=[], created_at=time.time(),
                        updated_at=time.time())

def test_episode_roundtrip():
    S.start_session("s1")
    S.save_episode(Episode(id="e1", session_id="s1",
                           timestamp=time.time(), role="user",
                           content="hello"))
    assert len(S.get_session_episodes("s1")) == 1

def test_fts_search():
    S.save_fact(_fact("f1", "User prefers Rust for systems programming"))
    assert len(S.search_facts_fts("rust systems")) == 1

def test_fts_empty_and_garbage_queries_do_not_crash():
    assert isinstance(S.search_facts_fts(""), list)
    assert isinstance(S.search_facts_fts("!!*(("), list)

def test_fact_upsert_dedupes():
    S.save_fact(_fact("f1", "User prefers Rust for systems programming"))
    assert S.memory_stats()["total_facts"] == 1

def test_session_end_updates_counts():
    S.end_session("s1", "summary", 10, 0.01)
    sess = S.get_recent_sessions(1)[0]
    assert sess["episode_count"] == 1 and sess["summary"] == "summary"

def test_chunker_respects_size():
    cp = ContextPipeline()
    chunks = cp._chunk("aaa\n\n" + "b" * 30000 + "\n\nccc")
    assert len(chunks) >= 2
    assert all(len(c) <= 30100 for c in chunks)

def test_pipeline_direct_when_small():
    import asyncio
    cp = ContextPipeline(target_model_limit_tokens=100000)
    out, meta = asyncio.get_event_loop().run_until_complete(
        cp.prepare(["short doc"]))
    assert meta["strategy"] == "direct" and out == "short doc"
