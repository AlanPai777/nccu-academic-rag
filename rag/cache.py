"""
cache.py — Two-layer LRU cache for the RAG pipeline.

Layer 1 (response cache): query → full pipeline result (answer + sources + contexts)
Layer 2 (retrieval cache): query → reranked context chunks (skips embed + search + rerank)

Usage:
    from rag.cache import RAGCache
    cache = RAGCache(ttl_seconds=3600)
    cache.put_response("選課上限", result_dict)
    cached = cache.get_response("選課上限")  # instant hit
"""

from __future__ import annotations

import copy
import time
import threading
import unicodedata
from collections import OrderedDict
from typing import Any


def _normalize_query(query: str) -> str:
    """Normalize query for cache key: strip, lowercase, normalize unicode."""
    return unicodedata.normalize("NFKC", query.strip())


class _LRUCache:
    """Thread-safe LRU cache with TTL expiration."""

    def __init__(self, maxsize: int = 200, ttl_seconds: int = 3600):
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            ts, value = self._cache[key]

            # Check TTL
            if self._ttl > 0 and (time.time() - ts) > self._ttl:
                del self._cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(value)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (time.time(), copy.deepcopy(value))
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)  # evict oldest
                self._cache[key] = (time.time(), copy.deepcopy(value))

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class RAGCache:
    """Two-layer cache for the RAG pipeline.

    Layer 1 — Response cache: stores full pipeline results (answer + sources).
              Hit = skip everything, return instantly.
    Layer 2 — Retrieval cache: stores reranked context chunks.
              Hit = skip embed + search + rerank, only run LLM generation.
    """

    def __init__(self,
                 response_maxsize: int = 200,
                 retrieval_maxsize: int = 200,
                 ttl_seconds: int = 3600):
        self._response_cache = _LRUCache(maxsize=response_maxsize,
                                          ttl_seconds=ttl_seconds)
        self._retrieval_cache = _LRUCache(maxsize=retrieval_maxsize,
                                           ttl_seconds=ttl_seconds)

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
    print("=== RAGCache quick test ===\n")

    cache = RAGCache(response_maxsize=3, retrieval_maxsize=3, ttl_seconds=2)

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

    # Test LRU eviction (maxsize=3)
    cache.put_response("q1", {"answer": "a1"})
    cache.put_response("q2", {"answer": "a2"})
    cache.put_response("q3", {"answer": "a3"})
    cache.put_response("q4", {"answer": "a4"})  # should evict oldest
    assert cache.get_response("選課上限") is None  # evicted
    assert cache.get_response("q4") is not None
    print("[PASS] LRU eviction")

    # Test TTL expiration
    import time as _time
    cache2 = RAGCache(ttl_seconds=1)
    cache2.put_response("ttl_test", {"answer": "expires"})
    assert cache2.get_response("ttl_test") is not None
    _time.sleep(1.5)
    assert cache2.get_response("ttl_test") is None
    print("[PASS] TTL expiration")

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
