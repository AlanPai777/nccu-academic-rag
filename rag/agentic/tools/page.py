"""
rag/agentic/tools/page.py
get_page_tool / extract_links_tool: fetch a known URL's full content, or
the links it references (Migration Step 3). Ported directly from
rag/agentic_main.py.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from rag.agent_tools import (
    get_page as _get_page_raw,
    extract_links as _extract_links_raw,
    extract_form_ids,
)
from rag.agentic.logic.office_detection import _detect_offices


@tool
def get_page_tool(url: str, state: Annotated[dict, InjectedState]) -> str:
    """取得已知URL的完整頁面內容。只在已有明確URL時呼叫。"""
    p = _get_page_raw(url, subdomain=state.get("subdomain_hint"))
    if "error" in p:
        return f"get_page 失敗: {p['error']}"
    text = p.get("text", "")
    header = f"get_page 取得 {len(text)} 字，標題: {p.get('title','')}，subdomain: {p.get('subdomain','')}"
    form_ids = sorted(extract_form_ids(text))
    if form_ids:
        header += f"\n[偵測到表單編號: {', '.join(form_ids)}]"
    else:
        # Only check here when there's no form -- when a form IS present,
        # resource_node scans the form's own (freshly-fetched) text per §M
        # D4's established sequencing, not this page's narrative text.
        # Pure CONTACT-type queries never surface a form marker at all
        # (e.g. "出納組電話幾號"), so without this branch contact_node was
        # unreachable except as a side effect of resource_node -- there was
        # no path from a plain office-name page straight into contact_node.
        offices = _detect_offices(text)
        if offices:
            header += f"\n[偵測到辦公室: {', '.join(offices)}]"
    return f"{header}\n\n全文：\n{text}"


@tool
def extract_links_tool(url: str, state: Annotated[dict, InjectedState]) -> str:
    """取得頁面正文明確提到的其他連結。當目前頁面內容不足以回答問題、但內文提到其他相關文件或頁面時使用。"""
    links = _extract_links_raw(url, subdomain=state.get("subdomain_hint"))
    if not links:
        return "extract_links 找到 0 個連結"
    return "extract_links 找到:\n" + "\n".join(f"- {l['label']}: {l['url']}" for l in links[:10])
