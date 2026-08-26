"""
rag/nodes/synthesis.py
synthesis_node: generates the final answer from retrieved context + office
contact info + extraction checklist.
"""

from __future__ import annotations

from rag.router import QueryType
from rag.llm_client import simple_chat
from rag.agent_runtime import _parametric_fallback
from rag.nodes.extraction import _format_checklist
from rag.nodes.state import AgentState

_SYNTHESIS_PROMPT = """\
{office_section}【搜尋到的頁面內容】
{context}

{checklist_section}---

學生問題：{query}

⚠️ 上方【各辦公室聯絡資訊】是該辦公室查到的完整名單，**不是要求全部列出**——請針對每一步驟，從對應辦公室的名單中挑選與這個流程步驟最直接相關的承辦人，格式：姓名（職責）分機 XXXXX。不需要、也不要把整個辦公室的人員名冊都列進答案。若上方【搜尋到的頁面內容】顯示該步驟需要多層審核（例如先由承辦人受理，再經組長、單位主管等逐層簽核），請把實際涉及的每一層審核人員都列出來，不要只挑一人；若只是單純的一般承辦窗口，列 1-2 位最相關的即可。

【回答格式規則】
1. 申辦流程問題：以**步驟清單**格式回答，每站必須列出：辦理單位、地點（行政大樓 X 樓）、電話分機、**承辦人姓名（必填，從上方聯絡資訊引用）**及職責
2. 條件限定步驟請明確標注（如：住宿生需辦、國際學生需辦）
3. 若有退費標準，列出退費比例表（全額 / 2/3 / 1/3 / 不退）
4. 說明核准後的取件方式與時效（如：三個工作天後可領取 / 郵寄 / iNCCU 系統查詢）——若資料中沒有提及，不要猜測
5. 回答最後以「【引用來源】」列出所有使用到的 URL
6. 資料不足時說「無法確認」，不猜測

⚠️ 審核層級、蓋章單位數量、是否需要補充表單等流程細節，**一律只依據上方【搜尋到的頁面內容】與 checklist 實際列出的內容回答**，不要套用你對其他申辦流程（例如休學）的既有印象去補全或類推——不同流程的審核層級可能不同，checklist 沒列出的層級就是沒有，不要自行加上。
"""


def synthesis_node(state: AgentState) -> AgentState:
    """
    Generate the final answer from retrieved context + office contact info.

    KNOWLEDGE: answer already produced by agent loop → pass through.
    RESOURCE:  answer already produced by resource_node → pass through, NO
               parametric fallback (unlike KNOWLEDGE) — a hallucinated form
               URL/ID is a much worse failure than resource_node's own
               honest "查無此表單" decline, so this branch never calls
               _parametric_fallback regardless of context_pages/answer state.
    PROCEDURE / CONTACT: call LLM with context_pages + office_context +
    extraction_checklist (condition 6).
    On E4 retry: prepends correction_hint so the LLM knows what to fix.
    """
    if state["query_type"] == QueryType.RESOURCE:
        return state

    if state["query_type"] == QueryType.KNOWLEDGE:
        answer = state.get("answer", "")
        # E7: agent loop found nothing → parametric fallback
        if not state.get("context_pages") and (not answer or "無法確認" in answer):
            return {**state, "answer": _parametric_fallback(state["query"])}
        return state

    # Dedup by URL before rendering. context_pages can carry the same page
    # multiple times — confirmed 2026-08-26 that neither retrieval_anchor_node
    # nor any single retrieval_expand_node branch is called more than once
    # (traced with a per-call print), so the duplication happens somewhere in
    # how LangGraph's operator.add reducer applies Send-branch writes, not in
    # our own node logic (root cause not fully diagnosed — deep LangGraph
    # internals, out of scope to chase further right now). Observed
    # multiplying an 18-page anchor+expand result into 68 entries (14 unique)
    # even before any self_eval retry, and self_eval_node's retry loop
    # doubles it again each retry, which is what actually blew past the
    # model's 262K-token context limit today. Deduping here is a pragmatic,
    # correctness-preserving fix at the consumption point — safe no-op if
    # duplication is ever fixed upstream, and doesn't depend on diagnosing
    # the exact LangGraph mechanism.
    seen_page_urls: set[str] = set()
    deduped_pages = []
    for page in state.get("context_pages", []):
        url = page.get("url", "")
        if url in seen_page_urls:
            continue
        seen_page_urls.add(url)
        deduped_pages.append(page)

    context_parts = []
    for i, page in enumerate(deduped_pages, 1):
        header = f"[文件 {i}] {page.get('title', '')}  來源：{page.get('url', '')}"
        body   = page.get("text", "")[:3000]
        context_parts.append(f"{header}\n{body}")
    context_str = "\n\n---\n\n".join(context_parts) or "（無搜尋結果）"

    office_section = ""
    if state.get("office_context"):
        office_section = state["office_context"] + "\n\n"

    # E7: PROCEDURE/CONTACT with no context at all → parametric fallback
    if context_str == "（無搜尋結果）" and not state.get("office_context"):
        return {**state, "answer": _parametric_fallback(state["query"]), "correction_hint": ""}

    checklist_section = _format_checklist(state.get("extraction_checklist", {}))

    base_prompt = _SYNTHESIS_PROMPT.format(
        office_section=office_section,
        context=context_str,
        checklist_section=checklist_section,
        query=state["query"],
    )

    correction = state.get("correction_hint", "")
    prompt = f"{correction}\n\n{base_prompt}" if correction else base_prompt

    answer = simple_chat(messages=[{"role": "user", "content": prompt}])
    return {**state, "answer": answer, "correction_hint": ""}
