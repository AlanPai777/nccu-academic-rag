"""
rag/agentic/tools/form.py
get_form_tool: fetch a known form ID's full content (Migration Step 3).
Ported directly from rag/agentic_main.py.
"""

from __future__ import annotations

from langchain_core.tools import tool

from rag.agent_tools import get_form as _get_form_raw


@tool
def get_form_tool(form_id: str) -> str:
    """取得已知表單編號的完整內容。當頁面提到表單編號、但你不確定頁面本身的敘述是否已涵蓋完整細節時使用——表單常包含頁面沒寫清楚的細節。"""
    f = _get_form_raw(form_id)
    if "error" in f:
        return f"get_form 失敗: {f['error']}"
    return f"get_form 取得表單 {f.get('form_title','')}\n\n{f.get('text','')}"
