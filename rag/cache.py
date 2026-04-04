"""
cache.py — Two-layer persistent cache for the RAG pipeline (SQLite-backed).

Layer 1 (response cache): query → full pipeline result (answer + sources + contexts)
Layer 2 (retrieval cache): query → reranked context chunks (skips embed + search + rerank)

Cache persists across process restarts via SQLite. Works for both CLI and web UI.

Usage:
    from rag.cache import RAGCache
    cache = RAGCache(ttl_seconds=3600)
    cache.put_response("選課上限", result_dict)
    cached = cache.get_response("選課上限")  # instant hit (even after restart)
"""

from __future__ import annotations

import json
import sqlite3
import time
import threading
import unicodedata
from pathlib import Path
from typing import Any


class _CacheEncoder(json.JSONEncoder):
    """Handle numpy float32/int types from reranker scores."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
        except ImportError:
            pass
        return super().default(obj)

# Default cache location: rag/cache.db (next to this file)
DEFAULT_DB_PATH = Path(__file__).parent / "cache.db"


def _normalize_query(query: str) -> str:
    """Normalize query for cache key: strip, normalize unicode."""
    return unicodedata.normalize("NFKC", query.strip())


class _SQLiteCache:
    """Thread-safe SQLite-backed cache with TTL and max size."""

    def __init__(self, db_path: Path, table: str,
                 maxsize: int = 200, ttl_seconds: int = 3600):
        self._db_path = str(db_path)
        self._table = table
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def get(self, key: str) -> Any | None:
        with self._lock:
            conn = self._conn()
            try:
                row = conn.execute(
                    f"SELECT value, created_at FROM {self._table} WHERE key = ?",
                    (key,)
                ).fetchone()

                if row is None:
                    self.misses += 1
                    return None

                value_json, created_at = row

                # Check TTL
                if self._ttl > 0 and (time.time() - created_at) > self._ttl:
                    conn.execute(f"DELETE FROM {self._table} WHERE key = ?", (key,))
                    conn.commit()
                    self.misses += 1
                    return None

                self.hits += 1
                return json.loads(value_json)
            finally:
                conn.close()

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            conn = self._conn()
            try:
                value_json = json.dumps(value, ensure_ascii=False, cls=_CacheEncoder)
                conn.execute(
                    f"INSERT OR REPLACE INTO {self._table} (key, value, created_at) "
                    f"VALUES (?, ?, ?)",
                    (key, value_json, time.time())
                )

                # Evict oldest if over maxsize
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {self._table}"
                ).fetchone()[0]
                if count > self._maxsize:
                    excess = count - self._maxsize
                    conn.execute(
                        f"DELETE FROM {self._table} WHERE key IN "
                        f"(SELECT key FROM {self._table} ORDER BY created_at ASC LIMIT ?)",
                        (excess,)
                    )

                conn.commit()
            finally:
                conn.close()

    def clear(self) -> None:
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(f"DELETE FROM {self._table}")
                conn.commit()
            finally:
                conn.close()
            self.hits = 0
            self.misses = 0

    def size(self) -> int:
        with self._lock:
            conn = self._conn()
            try:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {self._table}"
                ).fetchone()[0]

                # Exclude expired entries from count
                if self._ttl > 0:
                    cutoff = time.time() - self._ttl
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {self._table} WHERE created_at > ?",
                        (cutoff,)
                    ).fetchone()[0]

                return count
            finally:
                conn.close()


class RAGCache:
    """Two-layer persistent cache for the RAG pipeline.

    Layer 1 — Response cache: stores full pipeline results (answer + sources).
              Hit = skip everything, return instantly.
    Layer 2 — Retrieval cache: stores reranked context chunks.
              Hit = skip embed + search + rerank, only run LLM generation.

    Backed by SQLite — cache persists across process restarts.
    """

    def __init__(self,
                 response_maxsize: int = 200,
                 retrieval_maxsize: int = 200,
                 ttl_seconds: int = 3600,
                 db_path: Path | str = DEFAULT_DB_PATH):
        db_path = Path(db_path)
        self._response_cache = _SQLiteCache(
            db_path=db_path, table="response_cache",
            maxsize=response_maxsize, ttl_seconds=ttl_seconds)
        self._retrieval_cache = _SQLiteCache(
            db_path=db_path, table="retrieval_cache",
            maxsize=retrieval_maxsize, ttl_seconds=ttl_seconds)

    def get_response(self, query: str) -> dict | None:
        """Check response cache. Returns full pipeline result or None."""
        key = _normalize_query(query)
        return self._response_cache.get(key)

    def get_retrieval(self, query: str) -> list[dict] | None:
        """Check retrieval cache. Returns cached contexts or None."""
        key = _normalize_query(query)
        return self._retrieval_cache.get(key)

    def put_response(self, query: str, result: dict) -> None:
        """Store full pipeline result."""
        key = _normalize_query(query)
        self._response_cache.put(key, result)

    def put_retrieval(self, query: str, contexts: list[dict]) -> None:
        """Store retrieval results (reranked chunks)."""
        key = _normalize_query(query)
        self._retrieval_cache.put(key, contexts)

    def clear(self) -> None:
        """Clear all caches."""
        self._response_cache.clear()
        self._retrieval_cache.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "response": {
                "size": self._response_cache.size(),
                "hits": self._response_cache.hits,
                "misses": self._response_cache.misses,
            },
            "retrieval": {
                "size": self._retrieval_cache.size(),
                "hits": self._retrieval_cache.hits,
                "misses": self._retrieval_cache.misses,
            },
        }


# ── Quick test ─────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import tempfile
    import os

    print("=== RAGCache (SQLite) quick test ===\n")

    # Use temp file so tests don't pollute the real cache
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        cache = RAGCache(response_maxsize=3, retrieval_maxsize=3,
                         ttl_seconds=2, db_path=tmp_path)

        # Test response cache
        cache.put_response("選課上限", {"answer": "25學分", "sources": []})
        result = cache.get_response("選課上限")
        assert result is not None and result["answer"] == "25學分"
        print("[PASS] Response cache hit")

        result = cache.get_response("畢業學分")
        assert result is None
        print("[PASS] Response cache miss")

        # Test retrieval cache
        cache.put_retrieval("選課上限", [{"text": "chunk1"}, {"text": "chunk2"}])
        contexts = cache.get_retrieval("選課上限")
        assert contexts is not None and len(contexts) == 2
        print("[PASS] Retrieval cache hit")

        # Test query normalization
        cache.put_response("  選課上限  ", {"answer": "normalized"})
        result = cache.get_response("選課上限")
        assert result is not None and result["answer"] == "normalized"
        print("[PASS] Query normalization")

        # Test max size eviction
        cache.put_response("q1", {"answer": "a1"})
        cache.put_response("q2", {"answer": "a2"})
        cache.put_response("q3", {"answer": "a3"})
        cache.put_response("q4", {"answer": "a4"})  # should evict oldest
        assert cache.get_response("選課上限") is None  # evicted
        assert cache.get_response("q4") is not None
        print("[PASS] Max size eviction")

        # Test TTL expiration
        cache2 = RAGCache(ttl_seconds=1, db_path=tmp_path + ".ttl")
        cache2.put_response("ttl_test", {"answer": "expires"})
        assert cache2.get_response("ttl_test") is not None
        time.sleep(1.5)
        assert cache2.get_response("ttl_test") is None
        print("[PASS] TTL expiration")

        # Test persistence across instances
        cache3 = RAGCache(response_maxsize=10, ttl_seconds=3600, db_path=tmp_path)
        result = cache3.get_response("q4")
        assert result is not None and result["answer"] == "a4"
        print("[PASS] Persistence across restarts")

        # Test stats
        stats = cache.stats()
        print(f"\nStats: {stats}")
        print(f"  Response: {stats['response']['hits']} hits, {stats['response']['misses']} misses")
        print(f"  Retrieval: {stats['retrieval']['hits']} hits, {stats['retrieval']['misses']} misses")

        # Test clear
        cache.clear()
        assert cache.stats()["response"]["size"] == 0
        print("[PASS] Cache clear")

        print("\n=== All tests passed ===")

    finally:
        os.unlink(tmp_path)
        ttl_path = tmp_path + ".ttl"
        if os.path.exists(ttl_path):
            os.unlink(ttl_path)
