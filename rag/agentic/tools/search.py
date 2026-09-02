"""
rag/agentic/tools/search.py
search_texts: FTS5 + LLM-judge semantic search tool (Migration Step 3).
Ported directly from rag/agentic_main.py.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from rag.keyword_store import KeywordStore
from rag.agentic.logic.rewrite import _judge_candidates

FTS_DB = "rag/fts_proto3.db"


@tool
def search_texts(state: Annotated[dict, InjectedState]) -> str:
    """語意搜尋——用這一輪系統已經幫你重寫好的搜尋詞，內部會查FTS5全文索引並判斷候選是否相關。不需要自己給關鍵字，優先使用這個工具。"""
    rewritten = state["rewritten"]
    ks = KeywordStore(db_path=FTS_DB)
    candidates = ks.search(rewritten, top_k=5, subdomain=state.get("subdomain_hint"))
    if not candidates:
        candidates = ks.search(rewritten, top_k=5, subdomain=None)
    if not candidates:
        return f"search_texts('{rewritten}') 無候選結果"
    judge = _judge_candidates(state["query"], candidates)
    if not judge.get("good_enough"):
        return f"search_texts('{rewritten}') 候選皆不相關，judge理由: {judge.get('reason','')}"
    idx = max(0, min((judge.get("selected_index") or 1) - 1, len(candidates) - 1))
    sel = candidates[idx]
    return (f"search_texts('{rewritten}') 找到候選 [{sel['subdomain']}] {sel['title']}\n"
            f"URL: {sel['url']}\n(judge通過: {judge.get('reason','')})")
