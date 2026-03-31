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


class Pipeline:
    """
    Full RAG pipeline:
        query → Retriever (embed + search + rerank) → Generator (LLM)
    """

    def __init__(self,
                 rerank_top_n: int = 5,
                 llm_model: str = "granite4:3b",
                 reranker_device: str = "auto"):
        self.retriever = Retriever(rerank_top_n=rerank_top_n,
                                   reranker_device=reranker_device)
        self.generator = Generator(model=llm_model)

    def ask(self, query: str) -> dict:
        """
        Ask a question and return answer + sources.

        Returns:
            {
                "query":   str,
                "answer":  str,
                "sources": list[{index, url, title, source_type}],
                "contexts": list[dict]   # raw retrieved chunks
            }
        """
        # Step 1: Retrieve relevant chunks
        contexts = self.retriever.retrieve(query)

        if not contexts:
            return {
                "query":    query,
                "answer":   "抱歉，資料庫中找不到與此問題相關的內容。",
                "sources":  [],
                "contexts": [],
            }

        # Step 2: Generate answer
        result = self.generator.generate(query, contexts)

        return {
            "query":    query,
            "answer":   result["answer"],
            "sources":  result["sources"],
            "contexts": contexts,
        }


# ── CLI ────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="NCCU RAG pipeline.")
    parser.add_argument("--query",   required=True, help="Question to ask")
    parser.add_argument("--model",   default="granite4:3b",
                        help="Ollama model name (default: granite4:3b)")
    parser.add_argument("--top-n",   type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Also print retrieved chunks")
    args = parser.parse_args()

    pipe = Pipeline(rerank_top_n=args.top_n, llm_model=args.model)
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
