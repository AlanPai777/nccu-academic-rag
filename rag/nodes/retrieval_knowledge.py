"""
rag/nodes/retrieval_knowledge.py
retrieval_node: the KNOWLEDGE-path LLM-driven agent loop (grep + get_page,
via _TOOLS/ChatOllama-style tool calling), plus the PROCEDURE branch kept
only for merge_node's composite-sub-query path (see retrieval_node's own
docstring for why the main single-query PROCEDURE path no longer reaches
this branch).
"""

from __future__ import annotations

import json
import sys

from rag.router import QueryType
from rag.llm_client import chat_with_tools
from rag.agent_tools import grep_texts, get_page, extract_links, get_children, get_form
from rag.domain_router import route_domain
from rag.skills.procedure_skill import ProcedureSkill
from rag.nodes.state import AgentState

# ── Tool schema ───────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name":        "grep_texts",
            "description": "還不知道確切 URL、只有關鍵字時的第一步：全文搜尋回傳相關頁面列表（含預覽文字）。已經有明確 URL 時不要再呼叫這個工具，改用 get_page 直接取頁面。",
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
            "description": "取得指定 URL 的完整頁面內文（整頁，不分段）。只在已有明確 URL（來自 grep_texts 或 extract_links 的結果）時呼叫，不要對猜測、拼湊出來的 URL 呼叫。",
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
            "description": "抓出目前頁面正文明確提到的、指向其他 NCCU 子網域的 markdown 連結——只在目前頁面沒有直接答案、但內容提到其他相關頁面（如跨處室蓋章流程）時才用。內容已經足以回答問題就不用呼叫。跟 get_children 不同：這裡只看正文明確提到的連結，範圍窄但精準。",
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
            "description": "取得某 URL 在爬蟲階層下的所有直屬子頁面，範圍是整個爬蟲結構、不限於正文提到的連結。跟 extract_links 的差異：extract_links 只抓正文明確提到的連結（精準但窄），get_children 抓爬蟲階層下全部子頁（廣但較模糊）——先試 extract_links，找不到答案再考慮這個。",
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
            "description": "取得官方 moltke 表單的完整內容（申辦流程、各單位地點）。只在問題明確需要官方表單本身（如表單編號、下載/填寫細節）時使用；一般性流程說明用 get_page/grep_texts 通常就夠，不需要額外呼叫這個工具。",
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
步驟一：grep_texts("關鍵字", subdomain="{subdomain_hint}") 找主頁——{subdomain_hint} 是系統依問題內容預先判斷的最可能子網域，找不到結果時再改用不指定 subdomain 的全域搜尋
步驟二：get_page(主頁 URL) 取完整內文
步驟三：extract_links(主頁 URL) 取所有 NCCU 跨處室連結
步驟四：get_page(退費頁 URL) 取退費標準
步驟五：若內文提到官方表單編號（格式如 QP-xxx-xx-xx），用 get_form(表單編號) 取蓋章流程與樓層

⚠️ 絕對不可自行猜測 URL。跨處室連結必須透過 extract_links 取得後再 get_page。
"""


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _dispatch(name: str, args: dict):
    if name == "grep_texts":    return grep_texts(**args)
    if name == "get_page":      return get_page(**args)
    if name == "extract_links": return extract_links(**args)
    if name == "get_children":  return get_children(**args)
    if name == "get_form":      return get_form(**args)
    return {"error": f"Unknown tool: {name}"}


def retrieval_node(state: AgentState) -> AgentState:
    """
    Fetch relevant pages.

    KNOWLEDGE → LLM-driven agent loop (grep + get_page) — the main-path use.
    PROCEDURE → ProcedureSkill (deterministic 3-step: grep → links → form) —
                kept ONLY for merge_node's composite-sub-query loop (Step 2.5),
                which calls this function directly and isn't wired through
                Send. The main single-query PROCEDURE path no longer reaches
                this branch: router_node routes PROCEDURE to
                retrieval_anchor_node instead (Step 4.5, condition 2) — same
                grep→links→form idea, but anchor is sequential/deterministic
                and links+forms expand in parallel via Send instead of
                ProcedureSkill's fixed sequential loop. Known follow-up, not
                yet done: give merge_node's composite path the same
                anchor+expand adaptivity — it still uses this fixed 3-step
                version for now.
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
    _TOOL_SAMPLE_N  = 3    # condition 8-A: turn-1 majority-vote resample count

    # Step 5 (condition 5): Domain Router picks the subdomain hint the LLM's
    # step-1 grep_texts call is guided towards, replacing the hardcoded
    # "aca" every KNOWLEDGE query used to get regardless of topic. The LLM
    # is still free to ignore the hint or search unscoped — this only
    # changes the example in the prompt, not a hard constraint.
    subdomain_hint = route_domain(query) or "aca"
    agent_system = _AGENT_SYSTEM.format(subdomain_hint=subdomain_hint)

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
            system_prompt=agent_system,
        )

        # Condition 8-A (Step 6): turn-1 "no tool call" is the observed E7
        # parametric-fallback failure mode (CLAUDE.md "KNOWLEDGE path
        # instability" — model answers from training knowledge instead of
        # retrieving). Resample up to _TOOL_SAMPLE_N-1 more times and
        # majority-vote "should a tool have been called" — this only pays
        # the extra LLM-call cost when the default single shot already
        # looks like a fallback, not on every turn or every query.
        if turn == 0 and not tool_calls:
            tool_votes = 0
            fallback_votes = 1  # the original sample voted "no tool"
            candidate = None
            for _ in range(_TOOL_SAMPLE_N - 1):
                c2, tc2 = chat_with_tools(
                    messages=messages,
                    tools=_TOOLS,
                    system_prompt=agent_system,
                )
                if tc2:
                    tool_votes += 1
                    candidate = candidate or (c2, tc2)
                else:
                    fallback_votes += 1
            if tool_votes > fallback_votes and candidate is not None:
                print(
                    f"[tool-sample] turn 1: {tool_votes}/{_TOOL_SAMPLE_N} samples chose to "
                    f"call a tool — overriding default no-tool answer",
                    file=sys.stderr,
                )
                content, tool_calls = candidate

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
