"""
pipeline.py — End-to-end RAG pipeline: retrieve → generate.

Usage:
    from rag.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.ask("逕修博士申請條件是什麼？")
    print(result["answer"])
    for s in result["sources"]:
        print(s["url"])

CLI:
    python rag/pipeline.py --query "選課辦法是什麼？"
    python rag/pipeline.py --query "How do I apply for leave of absence?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.retriever import Retriever
from rag.generator import Generator
from rag.cache import RAGCache


class Pipeline:
    """
    Full RAG pipeline:
        query → Cache check → Retriever (embed + search + rerank) → Generator (LLM)
    """

    def __init__(self,
                 rerank_top_n: int = 5,
                 llm_model: str | None = None,
                 provider: str = "local",
                 enable_cache: bool = True,
                 cache_ttl: int = 3600,
                 enable_keyword: bool = True):
        self.retriever = Retriever(rerank_top_n=rerank_top_n,
                                   enable_keyword=enable_keyword)
        self.generator = Generator(provider=provider, model=llm_model)
        self.cache = RAGCache(ttl_seconds=cache_ttl) if enable_cache else None

    def ask(self, query: str) -> dict:
        """
        Ask a question and return answer + sources.

        Returns:
            {
                "query":     str,
                "answer":    str,
                "sources":   list[{index, url, title, source_type}],
                "contexts":  list[dict],   # raw retrieved chunks
                "cache_hit": str | None     # "response", "retrieval", or None
            }
        """
        # Layer 1: Check response cache (skip everything)
        if self.cache:
            cached = self.cache.get_response(query)
            if cached is not None:
                cached["cache_hit"] = "response"
                return cached

        # Layer 2: Check retrieval cache (skip embed + search + rerank)
        contexts = None
        cache_hit = None
        if self.cache:
            contexts = self.cache.get_retrieval(query)
            if contexts is not None:
                cache_hit = "retrieval"

        # Full retrieval on cache miss
        if contexts is None:
            contexts = self.retriever.retrieve(query)

        if not contexts:
            return {
                "query":     query,
                "answer":    "抱歉，資料庫中找不到與此問題相關的內容。",
                "sources":   [],
                "contexts":  [],
                "cache_hit": None,
            }

        # Generate answer
        result = self.generator.generate(query, contexts)

        full_result = {
            "query":     query,
            "answer":    result["answer"],
            "sources":   result["sources"],
            "contexts":  contexts,
            "cache_hit": cache_hit,
        }

        # Store in caches
        if self.cache:
            self.cache.put_retrieval(query, contexts)
            self.cache.put_response(query, full_result)

        return full_result

    def cache_stats(self) -> dict | None:
        """Return cache statistics, or None if caching is disabled."""
        return self.cache.stats() if self.cache else None

    def cache_clear(self) -> None:
        """Clear all caches."""
        if self.cache:
            self.cache.clear()


# ── CLI ────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="NCCU RAG pipeline.")
    parser.add_argument("--query",   required=True, help="Question to ask")
    parser.add_argument("--model",   default=None,
                        help="LLM model name (default: qwen2.5:7b for local, gpt-oss-20b for openrouter)")
    parser.add_argument("--provider", default="local",
                        choices=["local", "openrouter"],
                        help="LLM provider: local (Ollama) or openrouter (API)")
    parser.add_argument("--top-n",   type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Also print retrieved chunks")
    args = parser.parse_args()

    pipe = Pipeline(rerank_top_n=args.top_n, llm_model=args.model,
                    provider=args.provider)
    result = pipe.ask(args.query)

    print(f"\n{'='*60}")
    print(f"問題：{result['query']}")
    print(f"{'='*60}\n")
    print(result["answer"])

    print(f"\n{'─'*60}")
    print("參考來源：")
    for s in result["sources"]:
        print(f"  [{s['index']}] ({s['source_type']}) {s['url']}")

    if args.verbose and result["contexts"]:
        print(f"\n{'─'*60}")
        print("擷取到的 chunks：")
        for i, c in enumerate(result["contexts"]):
            print(f"\n  [{i+1}] rerank={c.get('rerank_score','?')}  {c['url']}")
            print(f"       {c['text'][:200].replace(chr(10),' ')}…")


if __name__ == "__main__":
    main()
