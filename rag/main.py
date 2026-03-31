"""
main.py — Unified CLI entry point for the NCCU RAG system.

Commands:
    python rag/main.py --build-index         # embed + index all chunks
    python rag/main.py --build-index --reset # drop collection and rebuild
    python rag/main.py --query "..."         # single Q&A in terminal
    python rag/main.py --app                 # launch Gradio web UI
    python rag/main.py --app --port 7861     # custom port
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def cmd_build_index(reset: bool) -> None:
    from rag.indexer import main as indexer_main
    sys.argv = ["indexer"]
    if reset:
        sys.argv.append("--reset")
    indexer_main()


def cmd_query(query: str, model: str, top_n: int, verbose: bool,
              reranker_device: str = "auto") -> None:
    from rag.pipeline import Pipeline
    pipe = Pipeline(rerank_top_n=top_n, llm_model=model,
                    reranker_device=reranker_device)
    result = pipe.ask(query)

    print(f"\n{'='*60}")
    print(f"問題：{result['query']}")
    print(f"{'='*60}\n")
    print(result["answer"])
    print(f"\n{'─'*60}")
    print("參考來源：")
    for s in result["sources"]:
        print(f"  [{s['index']}] ({s['source_type']}) {s['url']}")

    if verbose:
        print(f"\n{'─'*60}")
        print("擷取的 chunks：")
        for i, c in enumerate(result["contexts"]):
            print(f"\n  [{i+1}] rerank={c.get('rerank_score','?')}  {c['url']}")
            print(f"       {c['text'][:200].replace(chr(10), ' ')}…")


def cmd_app(port: int, share: bool) -> None:
    from rag.app import build_ui
    demo = build_ui()
    import gradio as gr
    demo.launch(server_port=port, share=share, inbrowser=True, theme=gr.themes.Soft())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCCU Academic Affairs RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag/main.py --build-index
  python rag/main.py --build-index --reset
  python rag/main.py --query "選課辦法是什麼？"
  python rag/main.py --query "graduation requirements" --verbose
  python rag/main.py --app
  python rag/main.py --app --port 7861 --share
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-index", action="store_true",
                      help="Build Qdrant index from chunks.jsonl")
    mode.add_argument("--query", type=str, metavar="QUESTION",
                      help="Ask a single question")
    mode.add_argument("--app", action="store_true",
                      help="Launch Gradio web UI")

    parser.add_argument("--reset",   action="store_true",
                        help="(with --build-index) Drop and recreate collection")
    parser.add_argument("--model",   default="granite4:3b",
                        help="Ollama model (default: granite4:3b)")
    parser.add_argument("--top-n",   type=int, default=5,
                        help="Chunks to retrieve (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="(with --query) Print retrieved chunks")
    parser.add_argument("--reranker-device", default="auto",
                        choices=["auto", "cpu", "xpu", "cuda"],
                        help="Reranker device: auto / cpu / xpu(Intel GPU) / cuda(NVIDIA)")
    parser.add_argument("--port",    type=int, default=7860,
                        help="(with --app) Port (default: 7860)")
    parser.add_argument("--share",   action="store_true",
                        help="(with --app) Create public Gradio link")

    args = parser.parse_args()

    if args.build_index:
        cmd_build_index(reset=args.reset)
    elif args.query:
        cmd_query(args.query, args.model, args.top_n, args.verbose,
                  reranker_device=args.reranker_device)
    elif args.app:
        cmd_app(args.port, args.share)


if __name__ == "__main__":
    main()
