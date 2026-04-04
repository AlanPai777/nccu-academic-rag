"""
app.py — Gradio Web UI for NCCU Academic Affairs RAG assistant.

Usage:
    python rag/app.py
    # Opens http://localhost:7860 in browser
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from rag.pipeline import Pipeline

# ── Global pipeline instance (loaded once) ─────────────────────────────────── #
_state = {"pipeline": None}

def get_pipeline() -> Pipeline:
    if _state["pipeline"] is None:
        _state["pipeline"] = Pipeline()
    return _state["pipeline"]


# ── Core answer function ───────────────────────────────────────────────────── #

def answer(query: str, history: list[list[str]]) -> tuple[str, list[list[str]], str]:
    """
    Called on every user submission.

    Args:
        query   : user question
        history : chat history [[user_msg, bot_msg], ...]

    Returns:
        (answer_text, updated_history, sources_text)
    """
    if not query.strip():
        return "", history, ""

    pipe = get_pipeline()
    result = pipe.ask(query.strip())

    ans = result["answer"]
    cache_hit = result.get("cache_hit")
    if cache_hit:
        ans = f"⚡ *Cache hit: {cache_hit}*\n\n{ans}"

    # Format sources
    if result["sources"]:
        src_lines = []
        seen_urls = set()
        for s in result["sources"]:
            url = s["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            icon = "📄" if s["source_type"] == "pdf" else "🌐"
            src_lines.append(f"{icon} {url}")
        sources_md = "**參考來源：**\n" + "\n".join(src_lines)
    else:
        sources_md = "*（未找到相關來源）*"

    history = history + [{"role": "user", "content": query},
                         {"role": "assistant", "content": ans}]
    return "", history, sources_md


def clear_all():
    return [], "", ""


def clear_cache():
    pipe = get_pipeline()
    pipe.cache_clear()
    stats = pipe.cache_stats()
    return f"Cache cleared. Stats reset: {stats}"


# ── Gradio UI ──────────────────────────────────────────────────────────────── #

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="政大教務處智慧助理") as demo:

        gr.Markdown("""
        # 🎓 政大教務處智慧助理
        **資料來源**：國立政治大學教務處（aca.nccu.edu.tw）
        支援中英文問答 · 由 qwen2.5:7b + embeddinggemma 提供
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="對話",
                    height=460,
                )
                with gr.Row():
                    query_box = gr.Textbox(
                        placeholder="請輸入問題，例如：選課最多可以修幾學分？",
                        label="",
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    submit_btn = gr.Button("送出", variant="primary", scale=1)
                with gr.Row():
                    clear_btn = gr.Button("清除對話", variant="secondary")
                    cache_btn = gr.Button("清除快取", variant="secondary")

            with gr.Column(scale=1):
                sources_box = gr.Markdown(
                    value="*送出問題後，參考來源會顯示在這裡*",
                    label="參考來源",
                )
                cache_status = gr.Markdown(
                    value="",
                    label="快取狀態",
                )
                gr.Markdown("""
                ---
                **範例問題**
                - 選課最多可以修幾學分？
                - 逕修博士學位的申請條件？
                - 畢業離校需要辦理哪些手續？
                - How do I apply for a leave of absence?
                - What are the graduation requirements?
                """)

        # ── Event handlers ── #
        submit_btn.click(
            fn=answer,
            inputs=[query_box, chatbot],
            outputs=[query_box, chatbot, sources_box],
        )
        query_box.submit(
            fn=answer,
            inputs=[query_box, chatbot],
            outputs=[query_box, chatbot, sources_box],
        )
        clear_btn.click(
            fn=clear_all,
            outputs=[chatbot, query_box, sources_box],
        )
        cache_btn.click(
            fn=clear_cache,
            outputs=[cache_status],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true",
                        help="Create public Gradio link")
    parser.add_argument("--provider", default="local",
                        choices=["local", "openrouter"],
                        help="LLM provider: local (Ollama) or openrouter (API)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    args = parser.parse_args()

    # Pre-create pipeline with chosen provider
    _state["pipeline"] = Pipeline(provider=args.provider)
    print(f"Pipeline loaded (provider={args.provider}, first query will load reranker)…")
    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        theme=gr.themes.Soft(),
    )
