"""
retriever.py — Dense search + Jina Reranker v2 reranking.

Pipeline:
    query → Ollama qwen3-embedding → Qdrant dense search top-30
          → Jina Reranker v2 cross-encoder rerank → top-5 chunks

Usage:
    from rag.retriever import Retriever
    ret = Retriever()                          # auto-detect device
    ret = Retriever(reranker_device="cpu")     # force CPU
    ret = Retriever(reranker_device="xpu")     # Intel GPU via ipex-llm
    ret = Retriever(reranker_device="cuda")    # NVIDIA GPU
    results = ret.retrieve("選課辦法是什麼？")
    # results → list of {text, text_clean, url, title, score, category, source_type}

CLI:
    python rag/retriever.py --query "選課辦法"
    python rag/retriever.py --query "選課辦法" --reranker-device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient

from rag.embedder import Embedder

# ── Constants ──────────────────────────────────────────────────────────────── #
COLLECTION   = "nccu_aca_v2_qwen3embedding"
QDRANT_URL   = "http://localhost:6333"
DENSE_TOP_K  = 30    # candidates from Qdrant (reduced from 50 for speed)
RERANK_TOP_N = 5     # final results after reranking
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
        embed → dense search → rerank → top-N results
    """

    def __init__(self,
                 qdrant_url: str = QDRANT_URL,
                 dense_top_k: int = DENSE_TOP_K,
                 rerank_top_n: int = RERANK_TOP_N,
                 reranker_device: str = RERANKER_DEVICE):
        self.dense_top_k  = dense_top_k
        self.rerank_top_n = rerank_top_n
        self.embedder  = Embedder()
        self.reranker  = Reranker(device=reranker_device)
        self.client    = QdrantClient(url=qdrant_url)

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

    def retrieve(self, query: str,
                 top_n: int | None = None) -> list[dict]:
        """
        Full pipeline: embed → search → rerank.

        Returns list of dicts (sorted by rerank score):
            {text, url, title, category, source_type, chunk_index,
             qdrant_score, rerank_score}
        """
        top_n = top_n or self.rerank_top_n

        # 1. Embed query
        emb = self.embedder.embed_query(query)
        query_vec = emb["dense"]

        # 2. Dense search → top-50 candidates
        candidates = self._dense_search(query_vec)
        if not candidates:
            return []

        # 3. Rerank candidates (use text_clean to avoid URL noise)
        passages = [c.get("text_clean", c["text"]) for c in candidates]
        ranked   = self.reranker.rerank(query, passages, top_n=top_n)

        # 4. Build final results
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
    args = parser.parse_args()

    ret = Retriever(reranker_device=args.reranker_device)
    results = ret.retrieve(args.query, top_n=args.top_n)

    print(f"\n=== Query: {args.query!r} ===")
    print(f"Top {len(results)} results after reranking:\n")

    for i, r in enumerate(results):
        print(f"[{i+1}] rerank={r['rerank_score']:.4f}  qdrant={r['qdrant_score']:.4f}")
        print(f"     URL  : {r['url']}")
        print(f"     type : {r['source_type']}  chunk: {r['chunk_index']}")
        print(f"     text : {r['text'][:200].replace(chr(10), ' ')}…")
        print()


if __name__ == "__main__":
    main()
