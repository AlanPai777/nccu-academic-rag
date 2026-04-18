"""
retriever.py — Hybrid retrieval via LlamaIndex: dense + keyword + RRF fusion.

Pipeline:
    query → OllamaEmbedding (port 11435)
          → VectorIndexRetriever (Qdrant, top-25) + navigation filter
          + KeywordRetriever (FTS5 + jieba, top-15)
          → QueryFusionRetriever (mode=reciprocal_rerank, k=60)
          → top-5 chunks

Phase 6 migration: embedder + vector store + dense retriever + fusion are
LlamaIndex components. KeywordStore (FTS5+jieba) is wrapped as a custom
BaseRetriever. Navigation filter is a BaseNodePostprocessor applied to
dense results before fusion (same strategy as Phase 5.5).

Usage:
    from rag.retriever import Retriever
    ret = Retriever()                          # hybrid (default)
    ret = Retriever(enable_keyword=False)      # dense only
    results = ret.retrieve("選課辦法是什麼？")
    # results → list of {text, text_clean, url, title, rerank_score,
    #                     category, source_type, chunk_index, chunk_type}

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

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag.keyword_store import KeywordStore

# ── Constants ──────────────────────────────────────────────────────────────── #
COLLECTION    = "nccu_aca_v2_embeddinggemma"
QDRANT_URL    = "http://localhost:6333"
EMBED_MODEL   = "embeddinggemma"
EMBED_URL     = "http://localhost:11435"  # standalone Ollama for embedding
DENSE_TOP_K   = 25   # candidates from Qdrant
KEYWORD_TOP_K = 15   # candidates from FTS5 keyword search
RERANK_TOP_N  = 5    # final results after RRF
RRF_K         = 60   # LlamaIndex QueryFusionRetriever uses k=60 internally

_PAYLOAD_FIELDS = (
    "text", "text_clean", "url", "title", "category", "source_type",
    "chunk_index", "chunk_type",
)


# ── Postprocessor: filter navigation chunks ──────────────────────────────── #
class _NavigationFilter(BaseNodePostprocessor):
    """Drop chunks tagged chunk_type='navigation' (near-empty, no answer content).

    Navigation chunks score high in dense similarity but contain no real
    content. Filtering them *before* RRF fusion prevents them from
    occupying dense candidate slots. See docs/phase5.5-report.md.
    """

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        return [
            n for n in nodes
            if (n.node.metadata or {}).get("chunk_type") != "navigation"
        ]


# ── Custom retriever: dense + navigation filter (for fusion input) ──────── #
class _FilteredDenseRetriever(BaseRetriever):
    """Wraps a VectorIndexRetriever and strips navigation chunks.

    QueryFusionRetriever runs sub-retrievers and fuses their ranked lists.
    Navigation chunks must be filtered *before* fusion so they don't burn
    dense candidate slots. Applying the postprocessor here (on dense only)
    achieves that.
    """

    def __init__(self, inner: BaseRetriever,
                 postprocessor: BaseNodePostprocessor):
        self._inner = inner
        self._post = postprocessor
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes = self._inner._retrieve(query_bundle)
        return self._post._postprocess_nodes(nodes, query_bundle)


# ── Custom retriever: FTS5 + jieba keyword search ───────────────────────── #
class _KeywordRetriever(BaseRetriever):
    """LlamaIndex BaseRetriever wrapping the existing KeywordStore (FTS5+jieba).

    Produces NodeWithScore outputs compatible with QueryFusionRetriever.
    The BM25 score from FTS5 is used as the node score (rank is what RRF
    actually consumes).
    """

    def __init__(self, keyword_store: KeywordStore, top_k: int):
        self._ks = keyword_store
        self._top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if not self._ks.exists():
            return []
        rows = self._ks.search(query_bundle.query_str, top_k=self._top_k)
        out: list[NodeWithScore] = []
        for r in rows:
            metadata = {k: r.get(k, "") for k in _PAYLOAD_FIELDS}
            # FTS5 hits don't always carry chunk_type; default to content
            metadata["chunk_type"] = r.get("chunk_type") or "content"
            metadata["chunk_index"] = r.get("chunk_index", 0)
            node = TextNode(
                text=r.get("text", r.get("text_clean", "")),
                metadata=metadata,
            )
            out.append(NodeWithScore(node=node, score=float(r.get("bm25_score", 0.0))))
        return out


# ── Public retriever ──────────────────────────────────────────────────────── #
class Retriever:
    """Full retrieval pipeline (LlamaIndex-based):
        embed → dense search (+ keyword search) → RRF fusion → top-N results.

    Public interface kept identical to Phase 5.5 so pipeline.py, cache.py,
    and generator.py do not change.
    """

    def __init__(self,
                 qdrant_url: str = QDRANT_URL,
                 dense_top_k: int = DENSE_TOP_K,
                 keyword_top_k: int = KEYWORD_TOP_K,
                 rerank_top_n: int = RERANK_TOP_N,
                 enable_keyword: bool = True):
        self.dense_top_k    = dense_top_k
        self.keyword_top_k  = keyword_top_k
        self.rerank_top_n   = rerank_top_n
        self.enable_keyword = enable_keyword

        # Embedding model (standalone Ollama on CPU, keep_alive=0 to free RAM)
        self.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=EMBED_URL,
            ollama_additional_kwargs={"keep_alive": 0},
        )
        Settings.embed_model = self.embed_model

        # Vector store over the existing Qdrant collection.
        # text_key='text' keeps the raw markdown-linked text in node.text,
        # and leaves text_clean / url / chunk_type / etc. in metadata.
        self.client = QdrantClient(url=qdrant_url)
        self.vector_store = QdrantVectorStore(
            collection_name=COLLECTION,
            client=self.client,
            dense_vector_name="dense",
            text_key="text",
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

        # Dense retriever + navigation filter (pre-fusion)
        dense_inner = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=dense_top_k,
        )
        self.dense_retriever = _FilteredDenseRetriever(
            inner=dense_inner,
            postprocessor=_NavigationFilter(),
        )

        # Keyword retriever (optional)
        self.keyword_store = KeywordStore() if enable_keyword else None
        self.keyword_retriever = (
            _KeywordRetriever(self.keyword_store, top_k=keyword_top_k)
            if enable_keyword else None
        )

        # Fusion via RRF. num_queries=1 skips LLM sub-query generation,
        # but QueryFusionRetriever still requires a non-None LLM at init —
        # MockLLM satisfies the type without hitting any network.
        if enable_keyword:
            self.fusion_retriever = QueryFusionRetriever(
                retrievers=[self.dense_retriever, self.keyword_retriever],
                similarity_top_k=rerank_top_n,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                llm=MockLLM(),
                verbose=False,
            )
        else:
            self.fusion_retriever = None

    # ---------------------------------------------------------------------- #
    def _node_to_dict(self, nws: NodeWithScore) -> dict:
        """Convert a LlamaIndex NodeWithScore back to the dict format used
        by pipeline.py / generator.py / cache.py."""
        meta = nws.node.metadata or {}
        raw_text = nws.node.get_content() or meta.get("text", "")
        return {
            "text":        raw_text,
            "text_clean":  meta.get("text_clean", raw_text),
            "url":         meta.get("url", ""),
            "title":       meta.get("title", ""),
            "category":    meta.get("category", ""),
            "source_type": meta.get("source_type", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_type":  meta.get("chunk_type", "content"),
            "rerank_score": round(float(nws.score or 0.0), 6),
        }

    def retrieve(self, query: str,
                 top_n: int | None = None) -> list[dict]:
        """Full pipeline: embed → dense (+keyword) → RRF fusion.

        Returns list of dicts sorted by RRF score.
        """
        top_n = top_n or self.rerank_top_n

        if self.enable_keyword and self.fusion_retriever is not None:
            # QueryFusionRetriever.similarity_top_k is set at init; honor top_n override
            self.fusion_retriever.similarity_top_k = top_n
            nodes = self.fusion_retriever.retrieve(query)
        else:
            # Dense-only path: use filtered dense retriever directly
            nodes = self.dense_retriever.retrieve(query)[:top_n]

        if not nodes:
            return []

        return [self._node_to_dict(n) for n in nodes]


# ── CLI ────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retriever test (LlamaIndex).")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N,
                        help=f"Number of results (default {RERANK_TOP_N})")
    parser.add_argument("--no-keyword", action="store_true",
                        help="Disable keyword search (dense only)")
    args = parser.parse_args()

    ret = Retriever(enable_keyword=not args.no_keyword)
    results = ret.retrieve(args.query, top_n=args.top_n)

    mode = "dense-only" if args.no_keyword else "hybrid (RRF fusion)"
    print(f"\n=== Query: {args.query!r} ({mode}) ===")
    print(f"Top {len(results)} results:\n")

    for i, r in enumerate(results):
        print(f"[{i+1}] rrf={r['rerank_score']:.6f}")
        print(f"     URL  : {r['url']}")
        print(f"     type : {r['source_type']}  chunk: {r['chunk_index']}  "
              f"({r['chunk_type']})")
        text = r.get("text_clean") or r.get("text") or ""
        print(f"     text : {text[:200].replace(chr(10), ' ')}…")
        print()


if __name__ == "__main__":
    main()
