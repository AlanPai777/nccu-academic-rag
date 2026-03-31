"""
retriever.py — Dense search + bge-reranker-v2-m3 reranking.

Pipeline:
    query → Ollama bge-m3 embed → Qdrant dense search top-50
          → bge-reranker-v2-m3 rerank → top-5 chunks

Usage:
    from rag.retriever import Retriever
    ret = Retriever()                          # auto-detect device
    ret = Retriever(reranker_device="cpu")     # force CPU
    ret = Retriever(reranker_device="xpu")     # Intel GPU (需安裝驅動)
    ret = Retriever(reranker_device="cuda")    # NVIDIA GPU
    results = ret.retrieve("選課辦法是什麼？")
    # results → list of {text, url, title, score, category, source_type}

CLI:
    python rag/retriever.py --query "選課辦法"
    python rag/retriever.py --query "選課辦法" --reranker-device cpu
    python rag/retriever.py --query "選課辦法" --reranker-device xpu
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
COLLECTION   = "nccu_aca"
QDRANT_URL   = "http://localhost:6333"
DENSE_TOP_K  = 50    # candidates from Qdrant
RERANK_TOP_N = 5     # final results after reranking
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# device 選項說明：
#   "auto" → FlagEmbedding 自動偵測（有 CUDA 用 CUDA，否則 CPU）
#   "cpu"  → 強制 CPU（最穩定，但最慢）
#   "xpu"  → Intel GPU（需安裝 /dev/dri/ 驅動，WSL2 預設不可用）
#   "cuda" → NVIDIA GPU
RERANKER_DEVICE = "auto"


# ── Reranker ───────────────────────────────────────────────────────────────── #
class Reranker:
    """Lazy-load bge-reranker-v2-m3 cross-encoder.

    Args:
        use_fp16: 使用 FP16 半精度加速（GPU 時效果顯著，CPU 略有加速）
        device:   運算裝置，"auto" / "cpu" / "xpu" / "cuda"
    """

    def __init__(self, use_fp16: bool = True, device: str = RERANKER_DEVICE):
        self.use_fp16 = use_fp16
        self.device   = device
        self._model   = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding is required. Install with:\n"
                "  pip install FlagEmbedding"
            ) from e

        device_str = f" (device={self.device})" if self.device != "auto" else ""
        print(f"Loading {RERANKER_MODEL}{device_str}…")

        kwargs: dict = {"use_fp16": self.use_fp16}
        if self.device != "auto":
            kwargs["device"] = self.device

        self._model = FlagReranker(RERANKER_MODEL, **kwargs)
        print(f"Reranker loaded on {self.device}.")

    def rerank(self, query: str, passages: list[str],
               top_n: int = RERANK_TOP_N) -> list[tuple[int, float]]:
        """
        Score [query, passage] pairs.
        Returns list of (original_index, score) sorted by score desc, top_n.
        """
        self._load()
        pairs = [[query, p] for p in passages]
        scores = self._model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
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
        self.reranker  = Reranker(use_fp16=True, device=reranker_device)
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
                "url":         h.payload.get("url", ""),
                "title":       h.payload.get("title", ""),
                "category":    h.payload.get("category", ""),
                "source_type": h.payload.get("source_type", ""),
                "chunk_index": h.payload.get("chunk_index", 0),
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

        # 3. Rerank candidates
        passages = [c["text"] for c in candidates]
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
