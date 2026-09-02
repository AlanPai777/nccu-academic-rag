"""
rag/agentic/nodes/self_eval.py
self_eval_node / _after_self_eval -- two-stage design (Migration Step 7,
docs/phase_h_agentic_rag_migration_plan.md §6.2/§6.3/Part 5 Step 7).

Stage 1 (logic/self_eval_checks.py's stage1_checklist): deterministic,
free checks. On failure, returns a concrete correction hint immediately
and skips Stage 2 entirely -- the checklist already knows what's missing,
no LLM call needed to describe it.

Stage 2 (only runs if Stage 1 passes): agentic_main.py's own
self_eval_node/_SELF_EVAL_PROMPT, ported directly -- a single LLM
judgment against the FULL original query, catching what Stage 1
structurally cannot (format-correct-but-off-topic answers, compound
queries silently answered half). Same three-way verdict (通過/情況A/情況B).

Per §6.3 (explicit decision, reversing an earlier draft proposal): runs
uniformly on ALL paths, including ones plan_node routed directly (never
skipped just because a path "looks simple/direct") -- a smooth routing
path doesn't guarantee a correct answer (see this session's own
_detect_offices() history: 14/15 failures happened on exactly the
"routed directly, no problem visible" path).

Per §0.1 (explicit decision): no _MAX_SELF_EVAL_RETRIES-style hard cap,
unlike production's `max 2`. _SELF_EVAL_MAX_TURN is a generous risk-only
ceiling, not a correctness cap -- reuses state["turn"] (incremented by
every rewrite_node pass) instead of inventing a dedicated counter.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from rag.llm_client import simple_chat
from rag.agentic.state import AgentState
from rag.agentic.logic.self_eval_checks import stage1_checklist
from rag.agentic.tools.grep import _fetched_urls

_SELF_EVAL_MAX_TURN = 20  # generous risk-only ceiling (not a correctness cap,
# same philosophy as _MAX_STUCK/removing _MAX_TURNS elsewhere in this package) --
# self_eval's retry loops (both 情況A and 情況B) route back through
# rewrite_node, which increments state["turn"] every pass, so this reuses
# that existing counter instead of inventing a new one just for self_eval.

_SELF_EVAL_PROMPT = """你是品質複核者，任務是判斷這次的回答有沒有問題——不是重新回答問題，只需要判斷。

原始完整問題（可能包含多個子問題，務必對照全文，不要只看其中一部分）：{query}

Plan_node當初的分類：{query_type}

這輪執行過程中，有沒有觸發過表單抓取或辦公室聯絡資訊查詢：{resource_contact_fired}

已讀過的頁面/表單URL：{urls}

最終答案：
{answer}

請依序判斷：
1. 如果原始問題其實包含好幾個不同主題的子問題，但答案只回答了其中一部分（例如問題同時問了兩件不同的事，答案只處理了一件）——這代表當初的分類沒有正確處理複合問題，回傳「情況B」。
2. 如果答案已經完整回答了原始問題的每一部分，回傳「通過」。
3. 如果分類本身看起來合理，只是內容不夠完整（例如procedure類問題找到了辦理流程，但完全沒有提到表單或聯絡辦公室，而這類問題通常需要）——回傳「情況A」，並具體寫出提醒文字：點名已讀過的URL、還沒呼叫過哪些工具（grep_texts_tool/get_page_tool/get_form_tool），建議下一步怎麼做。提醒文字要具體，不能只寫「請確認答案完整」這種空話。

只回傳一個JSON，不要有其他文字：
{{"verdict": "通過" 或 "情況A" 或 "情況B", "reminder": "情況A時的具體提醒文字，其他情況留空字串"}}"""


def self_eval_node(state: AgentState) -> dict:
    """Stage 1 first (free, deterministic) -- on failure, return its
    concrete hint directly and skip Stage 2 (the LLM call) entirely.
    Stage 2 only runs when Stage 1 passes, matching agentic_main.py's own
    self_eval_node otherwise unchanged: one LLM call (_SELF_EVAL_PROMPT)
    judges against the FULL original state["query"] (not a rewritten
    sub-query) -- this is what lets it catch compound queries a
    keyword-only detector structurally cannot.

    Honest caveat carried over from the reference implementation: Stage 2
    is still "message hint -> LLM decides whether to act", the same
    mechanism shape already shown unreliable elsewhere in this project
    (_AGENT_SYSTEM's timing clauses). Stage 1 existing specifically to
    catch the cheap, unambiguous failures before Stage 2's softer
    judgment is asked to do any work is the direct mitigation for that,
    not a claim that Stage 2 itself became more reliable."""
    stage1_failures = stage1_checklist(state)
    if stage1_failures:
        hint = "\n".join(f"- {f}" for f in stage1_failures)
        return {"self_eval_note": hint, "messages": [HumanMessage(content=f"[self_eval提醒] {hint}")]}

    fired = any(
        isinstance(m, HumanMessage) and ("[表單全文" in str(m.content) or "[辦公室聯絡資訊" in str(m.content))
        for m in state["messages"]
    )
    urls = sorted(_fetched_urls(state["messages"]))
    prompt = _SELF_EVAL_PROMPT.format(
        query=state["query"],
        query_type=state.get("query_type") or "未知",
        resource_contact_fired="有" if fired else "沒有",
        urls="、".join(urls) if urls else "（尚無）",
        answer=state.get("answer") or "（無答案）",
    )
    raw = simple_chat(messages=[{"role": "user", "content": prompt}], max_tokens=400)
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        result = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"self_eval_note": None}  # parse failure -> pass through, don't loop on a broken judge call

    verdict = result.get("verdict", "通過")
    if verdict == "情況A":
        reminder = (result.get("reminder") or "").strip() or "請確認答案是否完整回應原始問題各部分。"
        return {"self_eval_note": reminder, "messages": [HumanMessage(content=f"[self_eval提醒] {reminder}")]}
    if verdict == "情況B":
        return {
            "self_eval_note": "self_eval判斷原始問題可能未被正確分類或存在未處理的子問題，重新分類中。",
            "query_type": None,
        }
    return {"self_eval_note": None}


def _after_self_eval(state: AgentState) -> str:
    if not state.get("self_eval_note"):
        return "end"
    if state.get("turn", 0) >= _SELF_EVAL_MAX_TURN:
        return "end"  # risk-only ceiling, not a correctness judgment
    if state.get("query_type") is None:
        return "plan"  # 情況B (or a Stage 1 failure, which never clears query_type -- see below)
    return "rewrite"  # 情況A, or a Stage 1 failure
