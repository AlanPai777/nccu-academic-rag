"""
retriever.py — Hybrid retrieval: dense search + FTS5 keyword search + reranking.

Pipeline:
    query → Qdrant dense search (top-25)
          + FTS5 keyword search (top-15)
          → merge & deduplicate
          → Jina Reranker v2 cross-encoder rerank → top-5 chunks

The hybrid approach improves recall for exact-match terms (course codes,
regulation names) that dense search alone might miss.

Usage:
    from rag.retriever import Retriever
    ret = Retriever()                          # hybrid (default)
    ret = Retriever(enable_keyword=False)      # dense only
    results = ret.retrieve("選課辦法是什麼？")
    # results → list of {text, text_clean, url, title, score, category, source_type}

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
COLLECTION     = "nccu_aca_v2_embeddinggemma"
QDRANT_URL     = "http://localhost:6333"
DENSE_TOP_K    = 25    # candidates from Qdrant (was 30, reduced for hybrid budget)
KEYWORD_TOP_K  = 15    # candidates from FTS5 keyword search
MAX_CANDIDATES = 30    # cap total candidates before reranking
RERANK_TOP_N   = 5     # final results after reranking
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"

# device 選項說明：
#   "auto" → 自動偵測（有 CUDA 用 CUDA，否則 CPU）
#   "cpu"  → 強制 CPU（最穩定）
#   "xpu"  → Intel GPU via ipex-llm
#   "cuda" → NVIDIA GPU
RERANKER_DEVICE = "auto"


# ── Reranker ───────────────────────────────────────────────────────────────── #
class Reranker:
    """Lazy-load Jina Reranker v2 cross-encoder via sentence-transformers.

    Args:
        device: 運算裝置，"auto" / "cpu" / "xpu" / "cuda"
    """

    def __init__(self, device: str = RERANKER_DEVICE):
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Install with:\n"
                "  pip install sentence-transformers"
            ) from e

        device_str = f" (device={self.device})" if self.device != "auto" else ""
        print(f"Loading {RERANKER_MODEL}{device_str}…")

        kwargs: dict = {}
        if self.device != "auto":
            kwargs["device"] = self.device

        self._model = CrossEncoder(RERANKER_MODEL, trust_remote_code=True, **kwargs)
        print(f"Reranker loaded on {self.device}.")

    def rerank(self, query: str, passages: list[str],
               top_n: int = RERANK_TOP_N) -> list[tuple[int, float]]:
        """
        Batch-score [query, passage] pairs using cross-encoder.
        Returns list of (original_index, score) sorted by score desc, top_n.
        """
        self._load()
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        if not hasattr(scores, '__len__'):
            scores = [scores]
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])
        return indexed[:top_n]


# ── Retriever ──────────────────────────────────────────────────────────────── #
class Retriever:
    """
    Full retrieval pipeline:
        embed → dense search (+ keyword search) → merge → rerank → top-N results
    """

    def __init__(self,
                 qdrant_url: str = QDRANT_URL,
                 dense_top_k: int = DENSE_TOP_K,
                 keyword_top_k: int = KEYWORD_TOP_K,
                 max_candidates: int = MAX_CANDIDATES,
                 rerank_top_n: int = RERANK_TOP_N,
                 reranker_device: str = RERANKER_DEVICE,
                 enable_keyword: bool = True):
        self.dense_top_k    = dense_top_k
        self.keyword_top_k  = keyword_top_k
        self.max_candidates = max_candidates
        self.rerank_top_n   = rerank_top_n
        self.enable_keyword = enable_keyword
        self.embedder  = Embedder()
        self.reranker  = Reranker(device=reranker_device)
        self.client    = QdrantClient(url=qdrant_url)
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

    # ---------------------------------------------------------------------- #

    def _keyword_search(self, query: str) -> list[dict]:
        """Return keyword search candidates from FTS5."""
        if not self.keyword_store or not self.keyword_store.exists():
            return []
        results = self.keyword_store.search(query, top_k=self.keyword_top_k)
        # Add qdrant_score=0 so downstream code has a uniform schema
        for r in results:
            r["qdrant_score"] = 0.0
        return results

    @staticmethod
    def _merge_candidates(dense: list[dict], keyword: list[dict],
                          max_total: int) -> list[dict]:
        """Merge dense + keyword candidates, deduplicate by (url, chunk_index).

        Dense candidates are kept first (higher quality), then keyword-only
        candidates fill remaining slots up to max_total.
        """
        seen = set()
        merged = []

        # Dense candidates first (they have qdrant_score)
        for c in dense:
            key = (c["url"], c["chunk_index"])
            if key not in seen:
                seen.add(key)
                c["retrieval_source"] = "dense"
                merged.append(c)

        # Add keyword-only candidates (not already in dense results)
        for c in keyword:
            key = (c["url"], c["chunk_index"])
            if key not in seen:
                seen.add(key)
                c["retrieval_source"] = "keyword"
                merged.append(c)

        return merged[:max_total]

    def retrieve(self, query: str,
                 top_n: int | None = None) -> list[dict]:
        """
        Full pipeline: embed → dense search (+ keyword search) → merge → rerank.

        Returns list of dicts (sorted by rerank score):
            {text, url, title, category, source_type, chunk_index,
             qdrant_score, rerank_score, retrieval_source}
        """
        top_n = top_n or self.rerank_top_n

        # 1. Embed query
        emb = self.embedder.embed_query(query)
        query_vec = emb["dense"]

        # 2a. Dense search
        dense_candidates = self._dense_search(query_vec)

        # 2b. Keyword search (FTS5)
        keyword_candidates = self._keyword_search(query) if self.enable_keyword else []

        n_dense = len(dense_candidates)
        n_keyword = len(keyword_candidates)

        # 3. Merge & deduplicate
        candidates = self._merge_candidates(
            dense_candidates, keyword_candidates, self.max_candidates)

        if not candidates:
            return []

        n_merged = len(candidates)
        n_keyword_only = sum(1 for c in candidates if c.get("retrieval_source") == "keyword")
        if n_keyword > 0:
            print(f"  Hybrid retrieval: {n_dense} dense + {n_keyword} keyword → {n_merged} unique ({n_keyword_only} keyword-only)")

        # 4. Rerank candidates (use text_clean to avoid URL noise)
        passages = [c.get("text_clean", c["text"]) for c in candidates]
        ranked   = self.reranker.rerank(query, passages, top_n=top_n)

        # 5. Build final results
        results = []
        for orig_idx, score in ranked:
            item = dict(candidates[orig_idx])
            item["rerank_score"] = round(score, 4)
            results.append(item)

        return results


# ── CLI ────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retriever test.")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N,
                        help=f"Number of results (default {RERANK_TOP_N})")
    parser.add_argument("--reranker-device", default=RERANKER_DEVICE,
                        choices=["auto", "cpu", "xpu", "cuda"],
                        help="Reranker device: auto / cpu / xpu(Intel) / cuda(NVIDIA)")
    parser.add_argument("--no-keyword", action="store_true",
                        help="Disable keyword search (dense only)")
    args = parser.parse_args()

    ret = Retriever(reranker_device=args.reranker_device,
                    enable_keyword=not args.no_keyword)
    results = ret.retrieve(args.query, top_n=args.top_n)

    print(f"\n=== Query: {args.query!r} ===")
    print(f"Top {len(results)} results after reranking:\n")

    for i, r in enumerate(results):
        src = r.get("retrieval_source", "dense")
        print(f"[{i+1}] rerank={r['rerank_score']:.4f}  qdrant={r['qdrant_score']:.4f}  src={src}")
        print(f"     URL  : {r['url']}")
        print(f"     type : {r['source_type']}  chunk: {r['chunk_index']}")
        print(f"     text : {r['text'][:200].replace(chr(10), ' ')}…")
        print()


if __name__ == "__main__":
    main()
