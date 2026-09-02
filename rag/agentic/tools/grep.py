"""
rag/agentic/tools/grep.py
grep_texts_tool: literal-match search tool with scoping + fallback +
already-read dedup (Migration Step 3). Ported directly from
rag/agentic_main.py.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from rag.agent_tools import grep_texts as _grep_texts_raw


def _fetched_urls(messages: list) -> set[str]:
    """URLs the agent has already called get_page_tool on this run -- used
    to avoid re-surfacing pages the agent has already fully read as if
    they were new candidates."""
    urls = set()
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc["name"] == "get_page_tool":
                    u = tc["args"].get("url")
                    if u:
                        urls.add(u)
    return urls


@tool
def grep_texts_tool(pattern: str, state: Annotated[dict, InjectedState], subdomain: str = "") -> str:
    """純字面比對搜尋，沒有相關性排序，自己指定關鍵字。當search_texts找不到好結果、或你想用一個更精確/不同角度的詞直接試時使用。預設只搜這題的subdomain範圍（找不到會自動退回全域搜尋）；若你判斷答案在別的subdomain，可自己指定subdomain參數覆蓋。已經get_page過的頁面不會重複出現在候選裡。"""
    scope = subdomain or state.get("subdomain_hint")
    r = _grep_texts_raw(pattern, subdomain=scope, max_results=20)
    if not r and scope:
        r = _grep_texts_raw(pattern, subdomain=None, max_results=20)
    if not r:
        return f"grep_texts('{pattern}') 0筆結果"
    fetched = _fetched_urls(state.get("messages", []))
    unseen = [x for x in r if x["url"] not in fetched]
    shown = unseen if unseen else r
    note = "（已濾掉你讀過的頁面）" if unseen and len(unseen) < len(r) else ""
    top3 = "\n".join(f"{i+1}. [{x['subdomain']}] {x['title'][:40]} | URL: {x['url']}" for i, x in enumerate(shown[:3]))
    return f"grep_texts('{pattern}') 找到{len(r)}筆{note}，前3筆:\n{top3}"
