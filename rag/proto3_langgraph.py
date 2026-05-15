"""
rag/proto3_langgraph.py
Prototype 3: LangGraph-based agentic RAG pipeline.

Graph:
  START → router_node ──(PROCEDURE/KNOWLEDGE)──▶ retrieval_node ──▶ office_lookup_node
                      ╰──(CONTACT)─────────────────────────────────▶ office_lookup_node
                                                                             │
                                                                     synthesis_node ◀─╮
                                                                             │         │ correction_hint set
                                                                     self_eval_node ──╯ (max 2 retries)
                                                                             │
                                                                            END

Run:
    python -m rag.proto3_langgraph "如何辦理休學"
    python -m rag.proto3_langgraph "出納組電話"
    python -m rag.proto3_langgraph "選課上限幾學分"
    python -m rag.proto3_langgraph "如何辦理休學" --no-eval
"""

from __future__ import annotations

import json
import re
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from rag.router import route, QueryType
from rag.llm_client import chat_with_tools, simple_chat, get_active_model, PROVIDER
from rag.agent_tools import grep_texts, get_page, extract_links, get_children, get_form
from rag.skills.procedure_skill import ProcedureSkill
from rag.eval import print_score_report


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:           str
    query_type:      str        # "procedure" / "contact" / "knowledge"
    route_method:    str        # "keyword" / "llm" / "default"
    context_pages:   list[dict] # retrieved pages (PROCEDURE / KNOWLEDGE)
    office_context:  str        # formatted contact info — E3: OfficeLookupSkill
    answer:          str        # final answer
    sources:         list[str]  # cited URLs
    correction_hint: str        # E4: self-eval feedback for synthesis retry
    iteration:       int        # E4: retry counter (max _MAX_SELF_EVAL_RETRIES)


# ── Tool schema ───────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name":        "grep_texts",
            "description": "在 NCCU 文件庫全文搜尋，回傳相關頁面列表（含預覽文字）",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern":     {"type": "string"},
                    "subdomain":   {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_page",
            "description": "取得指定 URL 的完整頁面內文（整頁，不分段）",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "extract_links",
            "description": "提取某頁面中所有指向 NCCU 子網域的 markdown 連結",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_children",
            "description": "取得某 URL 在 BFS 爬蟲中的所有直屬子頁面",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":       {"type": "string"},
                    "subdomain": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_form",
            "description": "取得官方 moltke 表單的完整內容（申辦流程、各單位地點）",
            "parameters": {
                "type": "object",
                "properties": {"form_id": {"type": "string"}},
                "required": ["form_id"],
            },
        },
    },
]

_AGENT_SYSTEM = """你是國立政治大學（政大）的學務助理，使用工具搜尋政大官方資料回答學生問題。

【回答格式規則】
1. 申辦流程問題：必須以**步驟清單**格式回答，列出每站辦理單位、地點（行政大樓 X 樓）、電話分機
2. 條件限定步驟（只有住宿生 or 只有國際學生需要辦）請在步驟旁明確標注
3. 若有退費標準，請列出退費比例表（全額 / 2/3 / 1/3 / 不退）
4. 保留所有官方表單連結（如 QP-T01-03-02）
5. 只根據工具搜尋到的資料回答；資料不足時說「無法確認」而非猜測
6. 回答最後以「【引用來源】」列出所有使用到的 URL

【工具使用策略】
步驟一：grep_texts("關鍵字", subdomain="aca") 找主頁
步驟二：get_page(主頁 URL) 取完整內文
步驟三：extract_links(主頁 URL) 取所有 NCCU 跨處室連結
步驟四：get_page(退費頁 URL) 取退費標準
步驟五：get_form("QP-T01-03-02") 取蓋章流程與樓層

⚠️ 絕對不可自行猜測 URL。跨處室連結必須透過 extract_links 取得後再 get_page。
"""

_SYNTHESIS_PROMPT = """\
{office_section}【搜尋到的頁面內容】
{context}

---

學生問題：{query}

⚠️ 上方【各辦公室聯絡資訊】中列出的承辦人姓名**必須逐一出現在對應步驟中**，格式：姓名（職責）分機 XXXXX，不可省略。

【回答格式規則】
1. 申辦流程問題：以**步驟清單**格式回答，每站必須列出：辦理單位、地點（行政大樓 X 樓）、電話分機、**承辦人姓名（必填，從上方聯絡資訊引用）**及職責
2. 條件限定步驟請明確標注（如：住宿生需辦、國際學生需辦）
3. 教務處步驟需明確列出三層審核：承辦人 → 組長批示 → 教務長核准
4. 若有退費標準，列出退費比例表（全額 / 2/3 / 1/3 / 不退）
5. 若有補充表單（如委託書 QP-T01-02-05、提早復學 QP-T01-03-04），請一併列出
6. 說明核准後的取件方式與時效（如：三個工作天後可領取 / 郵寄 / iNCCU 系統查詢）
7. 若資料中提及身心健康中心，請附上電話
8. 回答最後以「【引用來源】」列出所有使用到的 URL
9. 資料不足時說「無法確認」，不猜測
"""


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _dispatch(name: str, args: dict):
    if name == "grep_texts":    return grep_texts(**args)
    if name == "get_page":      return get_page(**args)
    if name == "extract_links": return extract_links(**args)
    if name == "get_children":  return get_children(**args)
    if name == "get_form":      return get_form(**args)
    return {"error": f"Unknown tool: {name}"}


# ── Nodes ─────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """Classify the query into PROCEDURE / CONTACT / KNOWLEDGE."""
    result = route(state["query"])
    return {
        **state,
        "query_type":   result.query_type.value,
        "route_method": result.method,
    }


def retrieval_node(state: AgentState) -> AgentState:
    """
    Fetch relevant pages.

    PROCEDURE → ProcedureSkill (deterministic 3-step: grep → links → form)
    KNOWLEDGE → LLM-driven agent loop (grep + get_page)
    CONTACT   → not reached (conditional edge routes directly to office_lookup_node)
    """
    query = state["query"]

    if state["query_type"] == QueryType.PROCEDURE:
        skill  = ProcedureSkill()
        result = skill.run(query)
        return {
            **state,
            "context_pages": result.get("context", []),
            "sources":       result.get("source_urls", []),
        }

    # KNOWLEDGE: LLM-driven agent loop
    # E5 Doom Loop Detector constants
    _MAX_TURNS      = 10   # hard cap on LLM turns
    _MAX_STUCK      = 3    # exit if all tool calls are duplicates for this many consecutive turns

    messages: list[dict] = [{"role": "user", "content": query}]
    seen_calls: list[str] = []
    last_answer = ""
    context_pages: list[dict] = []
    sources: list[str] = []
    stuck_turns = 0        # E5: consecutive turns without new context pages

    for turn in range(_MAX_TURNS):
        content, tool_calls = chat_with_tools(
            messages=messages,
            tools=_TOOLS,
            system_prompt=_AGENT_SYSTEM,
        )
        if not tool_calls:
            last_answer = content or last_answer
            break

        if content:
            last_answer = content

        messages.append({
            "role":       "assistant",
            "content":    content,
            "tool_calls": tool_calls,
        })

        new_calls_this_turn = 0   # E5: count non-duplicate calls this turn

        for tc in tool_calls:
            call_sig = f"{tc['function']['name']}:{tc['function']['arguments']}"
            if call_sig in seen_calls:
                result_str = json.dumps({"warning": "已搜尋過相同條件，請換不同關鍵字"}, ensure_ascii=False)
            else:
                seen_calls.append(call_sig)
                new_calls_this_turn += 1
                # E5: tool call format validation — catch malformed JSON from LLM
                try:
                    args   = json.loads(tc["function"]["arguments"])
                    result = _dispatch(tc["function"]["name"], args)
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    result = {"error": f"工具呼叫格式錯誤：{e}"}
                result_str = json.dumps(result, ensure_ascii=False)

                if tc["function"]["name"] == "get_page" and isinstance(result, dict) and "text" in result:
                    context_pages.append(result)
                    if result.get("url"):
                        sources.append(result["url"])

            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result_str,
            })

        # E5: stuck detection — all calls this turn were duplicates (no new ideas)
        if new_calls_this_turn == 0:
            stuck_turns += 1
            if stuck_turns >= _MAX_STUCK:
                print(
                    f"[doom-loop] {stuck_turns} consecutive turns with only duplicate calls "
                    f"(turn {turn+1}/{_MAX_TURNS}) — exiting agent loop",
                    file=sys.stderr,
                )
                break
        else:
            stuck_turns = 0  # reset counter when agent makes new calls

    return {
        **state,
        "context_pages": context_pages,
        "sources":       list(dict.fromkeys(sources)),
        "answer":        last_answer,
    }


# Canonical office list for procedure queries (aliases excluded)
_PROCEDURE_OFFICES = ["生僑組", "住宿組", "出納組", "圖書館", "國際合作事務處", "教務處"]

# All office names to match against query text
_ALL_OFFICE_NAMES = [
    "生僑組", "住宿組", "出納組", "圖書館",
    "國際合作事務處", "國合處", "教務處", "教務長", "註冊組",
]


def _offices_from_query(query: str) -> list[str]:
    """Extract office names mentioned in the query."""
    return [name for name in _ALL_OFFICE_NAMES if name in query]


def office_lookup_node(state: AgentState) -> AgentState:
    """
    Inject office contact info into synthesis context.

    PROCEDURE: always inject all key offices (KNOWN_CONTACTS always-on,
               not dependent on dynamic grep success).
    CONTACT:   inject offices mentioned in the query; fallback to all if none found.
    KNOWLEDGE: skip — office info not needed for factual queries.

    OfficeLookupSkill first reads KNOWN_CONTACTS (static, reliable),
    then supplements with dynamic grep from extracted_texts.jsonl.
    """
    from rag.skills.office_lookup_skill import OfficeLookupSkill

    qtype = state["query_type"]

    if qtype == QueryType.KNOWLEDGE:
        return {**state, "office_context": ""}

    if qtype == QueryType.PROCEDURE:
        offices = _PROCEDURE_OFFICES
    else:  # CONTACT
        offices = _offices_from_query(state["query"]) or _PROCEDURE_OFFICES

    skill  = OfficeLookupSkill()
    result = skill.run(offices)
    return {**state, "office_context": skill.format_context(result)}


def synthesis_node(state: AgentState) -> AgentState:
    """
    Generate the final answer from retrieved context + office contact info.

    KNOWLEDGE: answer already produced by agent loop → pass through.
    PROCEDURE / CONTACT: call LLM with context_pages + office_context.
    On E4 retry: prepends correction_hint so the LLM knows what to fix.
    """
    if state["query_type"] == QueryType.KNOWLEDGE:
        answer = state.get("answer", "")
        # E7: agent loop found nothing → parametric fallback
        if not state.get("context_pages") and (not answer or "無法確認" in answer):
            return {**state, "answer": _parametric_fallback(state["query"])}
        return state

    context_parts = []
    for i, page in enumerate(state.get("context_pages", []), 1):
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

    base_prompt = _SYNTHESIS_PROMPT.format(
        office_section=office_section,
        context=context_str,
        query=state["query"],
    )

    correction = state.get("correction_hint", "")
    prompt = f"{correction}\n\n{base_prompt}" if correction else base_prompt

    answer = simple_chat(messages=[{"role": "user", "content": prompt}])
    return {**state, "answer": answer, "correction_hint": ""}


# ── E4: Self-evaluation checklist ────────────────────────────────────────────

_MAX_SELF_EVAL_RETRIES = 2

_SELF_EVAL_CRITERIA = [
    {
        "name":     "person_names",
        "keywords": ["盧誼甄", "王揚忠", "黃婉綝", "陳哲良", "林啓屏", "蕭毓璇"],
        "min_hits": 2,
        "hint":     "答案缺少承辦人姓名（需 ≥2 位）。請在每個步驟列出對應承辦人姓名（從【各辦公室聯絡資訊】引用）。",
    },
    {
        "name":     "sources",
        "keywords": ["引用來源", "https://"],
        "min_hits": 1,
        "hint":     "答案缺少引用來源。請在最後加上【引用來源】並列出所有 URL。",
    },
    {
        "name":     "procedure_format",
        "keywords": ["步驟", "第一步", "①", "第1步", "1.", "1、"],
        "min_hits": 1,
        "hint":     "答案未使用步驟清單格式。請改為以編號步驟列出辦理流程。",
    },
]


def self_eval_node(state: AgentState) -> AgentState:
    """
    E4: Self-evaluation checklist for PROCEDURE answers.

    Checks answer against critical criteria.  If any check fails and the
    retry budget has not been exhausted, sets correction_hint and increments
    iteration so that _after_self_eval routes back to synthesis_node.
    """
    # Only evaluate PROCEDURE answers; KNOWLEDGE/CONTACT pass through.
    if state["query_type"] != QueryType.PROCEDURE:
        return state

    iteration = state.get("iteration", 0)
    if iteration >= _MAX_SELF_EVAL_RETRIES:
        return state  # budget exhausted — accept current answer

    answer  = state.get("answer", "")
    failures = []

    for check in _SELF_EVAL_CRITERIA:
        hits = sum(1 for kw in check["keywords"] if kw in answer)
        if hits < check["min_hits"]:
            failures.append(check["hint"])
    
    if not failures:
        return state  # all checks passed
    '''
    if not failures:
        print("[self_eval] PASS — no retry needed", file=sys.stderr)
        return state
    print(f"[self_eval] FAIL — retry {state.get('iteration',0)+1}: {failures}", file=sys.stderr)
    '''
    correction = (
        "【自評回饋：以下項目不符合要求，請重新生成更完整的答案】\n"
        + "\n".join(f"- {f}" for f in failures)
    )
    return {**state, "correction_hint": correction, "iteration": iteration + 1}


# ── Conditional routing ───────────────────────────────────────────────────────

def _after_router(state: AgentState) -> str:
    if state["query_type"] == QueryType.CONTACT:
        return "office_lookup_node"
    return "retrieval_node"


def _after_self_eval(state: AgentState) -> str:
    """Route back to synthesis_node if correction_hint is set; else end."""
    if state.get("correction_hint"):
        return "synthesis_node"
    return END


# ── E6: Staleness warning ────────────────────────────────────────────────────

_CURRENT_ACA_YEAR = 114  # 114 學年度 = 2025-2026

def _staleness_warning(context_pages: list[dict]) -> str:
    """
    Scan context text for ROC academic-year mentions (e.g. 113學年度).
    Return a specific warning if data is older than current year, else a generic disclaimer.
    """
    all_text = " ".join(
        p.get("text", "") + " " + p.get("title", "")
        for p in context_pages
    )
    years_found = {int(m) for m in re.findall(r'(\d{3})學年度', all_text)}
    if years_found and max(years_found) < _CURRENT_ACA_YEAR:
        oldest = min(years_found)
        return (
            f"\n\n【資料時效說明】本回答含 {oldest} 學年度文件，"
            "申辦日期、費用或規定可能已異動，請至各處室官網確認最新版本。"
        )
    return (
        "\n\n【資料時效說明】本回答依據政大官網資料，"
        "如涉及重要申辦日期或規定，請以教務處／學務處當學期公告為準。"
    )


# ── E7: Parametric knowledge fallback ────────────────────────────────────────

_PARAMETRIC_SYSTEM = (
    "你是國立政治大學（政大）的學務助理。"
    "官方文件搜尋未找到相關資料，請根據你的訓練知識回答，"
    "並在答案最前方加上【以下來自模型訓練知識，非政大官方文件，請至官網確認】。"
)


def _parametric_fallback(query: str) -> str:
    """Call LLM without retrieval context; mark answer as parametric knowledge."""
    content, _ = chat_with_tools(
        messages=[{"role": "user", "content": query}],
        tools=[],
        system_prompt=_PARAMETRIC_SYSTEM,
    )
    return content or "（無法回答此問題）"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router_node",        router_node)
    g.add_node("retrieval_node",     retrieval_node)
    g.add_node("office_lookup_node", office_lookup_node)
    g.add_node("synthesis_node",     synthesis_node)
    g.add_node("self_eval_node",     self_eval_node)

    g.add_edge(START, "router_node")
    g.add_conditional_edges("router_node",    _after_router)
    g.add_edge("retrieval_node",     "office_lookup_node")
    g.add_edge("office_lookup_node", "synthesis_node")
    g.add_edge("synthesis_node",     "self_eval_node")
    g.add_conditional_edges("self_eval_node", _after_self_eval)

    return g.compile()


_graph = None


def run(query: str) -> str:
    global _graph
    if _graph is None:
        _graph = build_graph()

    final = _graph.invoke({
        "query":           query,
        "query_type":      "",
        "route_method":    "",
        "context_pages":   [],
        "office_context":  "",
        "answer":          "",
        "sources":         [],
        "correction_hint": "",
        "iteration":       0,
    })
    # E6: append staleness warning (post-processing, not a graph node)
    return final["answer"] + _staleness_warning(final.get("context_pages", []))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query",     help="學生問題")
    parser.add_argument("--no-eval", action="store_true", help="跳過 eval 評分")
    args = parser.parse_args()

    print(f"[proto3] model={get_active_model()}  provider={PROVIDER}", file=sys.stderr)

    answer = run(args.query)
    print(answer)

    if not args.no_eval:
        print_score_report(answer)
