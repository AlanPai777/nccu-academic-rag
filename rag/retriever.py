"""
retriever.py — Hybrid retrieval: dense search + FTS5 keyword search + RRF fusion.

Pipeline:
    query → Qdrant dense search (top-25)
          + FTS5 keyword search (top-15)
          → Reciprocal Rank Fusion (RRF, k=60) → top-5 chunks

RRF replaces the Jina CrossEncoder reranker (was ~9-10 min/query CPU).
Expected latency: <100ms total retrieval. See docs/rerank-replacement.md.

Usage:
    from rag.retriever import Retriever
    ret = Retriever()                          # hybrid (default)
    ret = Retriever(enable_keyword=False)      # dense only
    results = ret.retrieve("選課辦法是什麼？")
    # results → list of {text, text_clean, url, title, rerank_score,
    #                     category, source_type, retrieval_source}

CLI:
    python rag/retriever.py --query "選課辦法"
    python rag/retriever.py --query "選課辦法" --no-keyword
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient

from rag.embedder import Embedder
from rag.keyword_store import KeywordStore

# ── Constants ──────────────────────────────────────────────────────────────── #
COLLECTION    = "nccu_aca_v2_embeddinggemma"
QDRANT_URL    = "http://localhost:6333"
DENSE_TOP_K   = 25   # candidates from Qdrant
KEYWORD_TOP_K = 15   # candidates from FTS5 keyword search
RERANK_TOP_N  = 5    # final results after RRF
RRF_K         = 60   # RRF constant (standard; higher = flatter ranking)


# ── Retriever ──────────────────────────────────────────────────────────────── #
class Retriever:
    """
    Full retrieval pipeline:
        embed → dense search (+ keyword search) → RRF fusion → top-N results
    """

    def __init__(self,
                 qdrant_url: str = QDRANT_URL,
                 dense_top_k: int = DENSE_TOP_K,
                 keyword_top_k: int = KEYWORD_TOP_K,
                 rerank_top_n: int = RERANK_TOP_N,
                 enable_keyword: bool = True):
        self.dense_top_k   = dense_top_k
        self.keyword_top_k = keyword_top_k
        self.rerank_top_n  = rerank_top_n
        self.enable_keyword = enable_keyword
        self.embedder      = Embedder()
        self.client        = QdrantClient(url=qdrant_url)
        self.keyword_store = KeywordStore() if enable_keyword else None

    # ---------------------------------------------------------------------- #

    def _dense_search(self, query_vec: list[float]) -> list[dict]:
        """Return top-K hits from Qdrant dense index."""
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            using="dense",
            limit=self.dense_top_k,
            with_payload=True,
        )
        return [
            {
                "text":        h.payload.get("text", ""),
                "text_clean":  h.payload.get("text_clean", h.payload.get("text", "")),
                "url":         h.payload.get("url", ""),
                "title":       h.payload.get("title", ""),
                "category":    h.payload.get("category", ""),
                "source_type": h.payload.get("source_type", ""),
                "chunk_index": h.payload.get("chunk_index", 0),
                "chunk_type":  h.payload.get("chunk_type", "content"),
                "qdrant_score": h.score,
            }
            for h in results.points
        ]

    def _keyword_search(self, query: str) -> list[dict]:
        """Return keyword search candidates from FTS5."""
        if not self.keyword_store or not self.keyword_store.exists():
            return []
        results = self.keyword_store.search(query, top_k=self.keyword_top_k)
        for r in results:
            r.setdefault("qdrant_score", 0.0)
        return results

    def _rrf_fuse(self, dense: list[dict], keyword: list[dict],
                  top_n: int) -> list[dict]:
        """Reciprocal Rank Fusion across dense and keyword ranked lists.

        score(d) = Σ_i  1 / (RRF_K + rank_i(d))
        Docs absent from a retriever contribute 0 from that retriever.
        """
        scores: dict[tuple, float] = {}
        lookup: dict[tuple, dict] = {}
        dense_keys: set[tuple] = set()
        keyword_keys: set[tuple] = set()

        for rank, doc in enumerate(dense, start=1):
            key = (doc["url"], doc["chunk_index"])
            dense_keys.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            lookup[key] = doc

        for rank, doc in enumerate(keyword, start=1):
            key = (doc["url"], doc["chunk_index"])
            keyword_keys.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            lookup.setdefault(key, doc)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for key, score in ranked:
            item = dict(lookup[key])
            item["rerank_score"] = round(score, 6)
            in_dense   = key in dense_keys
            in_keyword = key in keyword_keys
            item["retrieval_source"] = (
                "hybrid"  if (in_dense and in_keyword) else
                "dense"   if in_dense else
                "keyword"
            )
            results.append(item)

        return results

    def retrieve(self, query: str,
                 top_n: int | None = None) -> list[dict]:
        """
        Full pipeline: embed → dense search (+ keyword search) → RRF fusion.

        Returns list of dicts sorted by RRF score:
            {text, text_clean, url, title, category, source_type,
             chunk_index, chunk_type, qdrant_score, rerank_score, retrieval_source}
        """
        top_n = top_n or self.rerank_top_n

        # 1. Embed query
        emb = self.embedder.embed_query(query)
        query_vec = emb["dense"]

        # 2a. Dense search — filter navigation chunks (near-empty, no answer content)
        dense_candidates = [
            c for c in self._dense_search(query_vec)
            if c.get("chunk_type") != "navigation"
        ]

        # 2b. Keyword search (FTS5)
        keyword_candidates = self._keyword_search(query) if self.enable_keyword else []

        n_dense   = len(dense_candidates)
        n_keyword = len(keyword_candidates)

        if not dense_candidates and not keyword_candidates:
            return []

        # 3. RRF fusion
        results = self._rrf_fuse(dense_candidates, keyword_candidates, top_n=top_n)

        n_hybrid  = sum(1 for r in results if r["retrieval_source"] == "hybrid")
        n_kw_only = sum(1 for r in results if r["retrieval_source"] == "keyword")
        if n_keyword > 0:
            print(f"  Hybrid retrieval: {n_dense} dense + {n_keyword} keyword → "
                  f"top-{len(results)} via RRF "
                  f"({n_hybrid} hybrid, {n_kw_only} keyword-only)")

        return results


# ── CLI ────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retriever test.")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N,
                        help=f"Number of results (default {RERANK_TOP_N})")
    parser.add_argument("--no-keyword", action="store_true",
                        help="Disable keyword search (dense only)")
    args = parser.parse_args()

    ret = Retriever(enable_keyword=not args.no_keyword)
    results = ret.retrieve(args.query, top_n=args.top_n)

    print(f"\n=== Query: {args.query!r} ===")
    print(f"Top {len(results)} results via RRF:\n")

    for i, r in enumerate(results):
        src = r.get("retrieval_source", "dense")
        print(f"[{i+1}] rrf={r['rerank_score']:.6f}  qdrant={r['qdrant_score']:.4f}  src={src}")
        print(f"     URL  : {r['url']}")
        print(f"     type : {r['source_type']}  chunk: {r['chunk_index']}")
        print(f"     text : {r['text'][:200].replace(chr(10), ' ')}…")
        print()


if __name__ == "__main__":
    main()
