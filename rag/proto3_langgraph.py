"""
rag/proto3_langgraph.py
Prototype 3: LangGraph-based agentic RAG pipeline.

Graph:
  START → query_decomposition_node ──(not composite)──▶ router_node ──(PROCEDURE)──▶ retrieval_anchor_node ─[Send ×N]─▶ retrieval_expand_node ──▶ office_lookup_node
                                  │                                 ├──(KNOWLEDGE)──────────────────────────────────────────────────────────────▶ retrieval_node ──▶ office_lookup_node
                                  │                                 ╰──(CONTACT)───────────────────────────────────────────────────────────────────────────────────▶ office_lookup_node
                                  │                                                                                                                                          │
                                  │                                                                                                                                  extraction_node
                                  ╰──(composite)──▶ merge_node (v1: sequential loop over sub_queries, still ProcedureSkill-based — TODO: Send + anchor/expand) ─────────────╯
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
import operator
import re
import sys
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from rag.router import route, QueryType, _keyword_route
from rag.llm_client import chat_with_tools, simple_chat, get_active_model, PROVIDER
from rag.agent_tools import grep_texts, get_page, extract_links, get_children, get_form, extract_form_ids
from rag.domain_router import route_domain, is_ambiguous
from rag.skills.procedure_skill import ProcedureSkill, _extract_keyword
from rag.eval import print_score_report


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:                str
    query_type:           str        # "procedure" / "contact" / "knowledge"
    route_method:         str        # "keyword" / "llm" / "default"
    sub_queries:          list[str]  # Step 2.5: non-empty only for composite queries
    # Annotated with operator.add so N parallel retrieval_expand_node branches
    # (Step 4.5, Send-based fan-out) each contribute their own page/URL without
    # clobbering each other or the anchor node's own contribution — LangGraph
    # sums all writes to this field within one superstep. Safe because exactly
    # one path (anchor+expand XOR retrieval_node XOR merge_node) ever writes to
    # it per graph run, always starting from the [] set in run()'s initial state.
    context_pages:        Annotated[list[dict], operator.add]
    sources:               Annotated[list[str], operator.add]
    detected_offices:     list[str]  # Step 4.5: set by retrieval_anchor_node so
                                      # office_lookup_node doesn't re-detect (condition 3)
    _anchor_links:        list[str]  # Step 4.5: cross-domain links found in anchor content
    _anchor_form_ids:     list[str]  # Step 4.5: form IDs found in anchor content
    expand_target:        dict       # Step 4.5: per-Send-branch payload {"kind","value"}
    office_context:       str        # formatted contact info — E3: OfficeLookupSkill
    office_lookup_result: dict       # raw {office: {contacts: [...], ...}} from
                                      # OfficeLookupSkill.run() — kept structured
                                      # (not just the flattened office_context
                                      # string) so extraction_node can build a
                                      # per-office checklist without regex
                                      # re-parsing already-formatted text
    extraction_checklist: dict       # condition 6: pre-synthesis dynamic checklist
    answer:               str        # final answer
    correction_hint:      str        # E4: self-eval feedback for synthesis retry
    iteration:            int        # E4: retry counter (max _MAX_SELF_EVAL_RETRIES)


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


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _dispatch(name: str, args: dict):
    if name == "grep_texts":    return grep_texts(**args)
    if name == "get_page":      return get_page(**args)
    if name == "extract_links": return extract_links(**args)
    if name == "get_children":  return get_children(**args)
    if name == "get_form":      return get_form(**args)
    return {"error": f"Unknown tool: {name}"}


# ── Nodes ─────────────────────────────────────────────────────────────────────

# Split on 逗號/句號 (full-width and half-width) — clause boundaries are the
# unit query_decomposition_node reasons about.
_CLAUSE_SPLIT_RE = re.compile(r'[，。,.]')


def query_decomposition_node(state: AgentState) -> AgentState:
    """
    Step 2.5: detect composite queries (multiple distinct topics in one
    question) BEFORE routing, so they don't get silently collapsed to a
    single QueryType by router_node's single-label classification (confirmed
    in Step 1 Q6: "如何辦理休學，圖書館的電話是多少" degraded the 休學 half
    when both topics were forced through one router_node/synthesis_node pass).

    v1 detection is pure Layer-1 keyword matching, reusing router.py's
    existing _keyword_route() per clause — no LLM call. This mirrors
    router.py's own "keyword first, LLM only when ambiguous" principle at
    the decomposition level: a clause with no clear keyword signal doesn't
    force an LLM call just to decide whether the query is composite.

    A query is composite only if 2+ clauses resolve to DIFFERENT QueryTypes
    via _keyword_route(). A single-topic query that merely has a pause
    (e.g. "休學需要注意哪些事，包含要去哪些辦公室" — both clauses are
    QueryType.PROCEDURE or ambiguous) is NOT composite; sub_queries stays
    empty and the query proceeds through the normal single-query path.
    """
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(state["query"]) if c.strip()]
    if len(clauses) < 2:
        return {**state, "sub_queries": []}

    types_found = {t for t in (_keyword_route(c) for c in clauses) if t is not None}
    if len(types_found) < 2:
        return {**state, "sub_queries": []}  # not composite — 0 or 1 distinct type

    return {**state, "sub_queries": clauses}


def merge_node(state: AgentState) -> AgentState:
    """
    Step 2.5: process each sub_query and merge results at the DATA layer
    only (context_pages / office_context / extraction_checklist) — never
    merges generated text, to avoid fragmented answers and multiplying LLM
    calls. One synthesis_node call handles the merged data for the whole
    composite query, same as a single query would.

    v1 is a plain sequential loop (established TODO, do not let this quietly
    become permanent): upgrade to Send-based parallel dispatch by replacing
    just this for-loop — the merge logic below already operates on
    independent per-sub-query results, so it is unaffected by whether the
    loop runs sequentially or via Send.

    Reuses retrieval_node / office_lookup_node / extraction_node as plain
    function calls against an isolated per-sub-query state (not through
    graph routing) — same node functions the single-query path uses,
    not reimplemented. retrieval_node is skipped for CONTACT sub-queries,
    matching _after_router's routing behaviour (retrieval_node's KNOWLEDGE
    branch would otherwise run for CONTACT, which it was never designed for).
    """
    merged_context_pages: list[dict] = []
    merged_sources: list[str] = []
    office_sections: list[str] = []
    merged_person_names: list[dict] = []
    merged_forms: list[dict] = []
    merged_notes: list[str] = []
    seen_form_ids: set[str] = set()
    seen_notes: set[str] = set()

    for sub_query in state["sub_queries"]:
        result = route(sub_query)
        sub_state: AgentState = {
            **state,
            "query":                sub_query,
            "query_type":           result.query_type.value,
            "route_method":         result.method,
            "context_pages":        [],
            "office_context":       "",
            "extraction_checklist": {},
            "sources":              [],
        }

        if result.query_type in (QueryType.PROCEDURE, QueryType.KNOWLEDGE):
            sub_state = retrieval_node(sub_state)

        sub_state = office_lookup_node(sub_state)
        sub_state = extraction_node(sub_state)

        merged_context_pages.extend(sub_state.get("context_pages", []))
        merged_sources.extend(sub_state.get("sources", []))
        if sub_state.get("office_context"):
            office_sections.append(sub_state["office_context"])

        checklist = sub_state.get("extraction_checklist", {})
        merged_person_names.extend(checklist.get("person_names", []))
        for f in checklist.get("forms", []):
            if f["id"] not in seen_form_ids:
                seen_form_ids.add(f["id"])
                merged_forms.append(f)
        for n in checklist.get("notes", []):
            if n["text"] not in seen_notes:
                seen_notes.add(n["text"])
                merged_notes.append(n)

    return {
        **state,
        "context_pages":        merged_context_pages,
        "sources":              list(dict.fromkeys(merged_sources)),
        "office_context":       "\n\n".join(office_sections),
        "extraction_checklist": {
            "person_names": merged_person_names,
            "forms":        merged_forms,
            "notes":        merged_notes,
        },
        # Composite answers always go through the full prompt-based synthesis
        # (office_section + checklist), never the KNOWLEDGE pass-through —
        # needed to weave multiple sub-topics into one coherent answer even
        # when every sub_query happened to be KNOWLEDGE-type. Also keeps
        # self_eval_node's retry loop active (PROCEDURE-only gate), valuable
        # here since composite answers are the highest-complexity case.
        "query_type":            QueryType.PROCEDURE.value,
    }


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


def _offices_from_context(context_pages: list[dict]) -> list[str]:
    """
    Condition 3: detect which offices are actually mentioned in retrieved
    content, replacing the hardcoded _PROCEDURE_OFFICES injection every
    PROCEDURE answer used to get regardless of relevance. Reuses condition
    4's dynamic contact-lookup infrastructure (checks against the same
    _PROCEDURE_OFFICES keys OfficeLookupSkill/_OFFICE_NAME_MAP already
    understand) — this function only decides WHICH offices to look up, not
    how to look them up.

    Deliberately narrow scope (Phase F Step 4): only this detection function
    is new. retrieval_node's own fetch logic is untouched — Step 4.5's
    anchor+expand redesign re-wires retrieval around this exact function
    rather than rewriting the detection itself.
    """
    combined = " ".join(p.get("text", "") for p in context_pages)
    return [office for office in _PROCEDURE_OFFICES if office in combined]


# ── Step 4.5 (condition 2): anchor + expand ──────────────────────────────────
# Replaces ProcedureSkill's fixed 3-step (grep → links → form) for the main
# single-query PROCEDURE path. anchor is sequential/deterministic (low risk,
# no LLM judgment); expand fans out via LangGraph's native Send API — one
# branch per cross-domain link or form ID actually found in anchor content,
# not a hardcoded or LLM-guessed count. No ToolNode/Ollama-Cloud-compatibility
# dependency here — Send is pure LangGraph graph mechanics.

def retrieval_anchor_node(state: AgentState) -> AgentState:
    """
    Sequential step: grep_texts + get_page to find the main page(s) — same
    idea as ProcedureSkill's old step 1, reusing its _extract_keyword()
    stripping helper rather than reimplementing it.

    Also runs office detection (condition 3, _offices_from_context) on JUST
    the anchor content, immediately — earlier than before, so
    office_lookup_node doesn't need to re-detect after expand completes
    (its own docstring covers why). And collects the cross-domain links +
    form IDs mentioned in anchor content into detected_offices/expand
    candidates for _dispatch_expand to fan out over.
    """
    query = state["query"]
    keyword = _extract_keyword(query)

    # Step 5 (condition 5): Domain Router replaces the hardcoded aca-first
    # bias — falls back to "aca" only if Domain Router itself finds nothing
    # (Layer 1 + Layer 2 both empty), not as a silent default otherwise.
    subdomain = route_domain(query) or "aca"
    main_results = grep_texts(keyword, subdomain=subdomain, max_results=5)
    if not main_results:
        main_results = grep_texts(keyword, max_results=5)

    anchor_pages: list[dict] = []
    seen_urls: set[str] = set()
    for r in main_results:
        if r["url"] not in seen_urls:
            full = get_page(r["url"])
            if "error" not in full:
                anchor_pages.append(full)
                seen_urls.add(r["url"])

    links: list[str] = []
    for page in anchor_pages:
        for link in extract_links(page["url"]):
            if link["url"] not in seen_urls:
                links.append(link["url"])
                seen_urls.add(link["url"])

    all_text = " ".join(p.get("text", "") for p in anchor_pages[:3])
    form_ids = extract_form_ids(all_text)

    return {
        **state,
        "context_pages":    anchor_pages,
        "sources":          [p["url"] for p in anchor_pages],
        "detected_offices": _offices_from_context(anchor_pages),
        "_anchor_links":    links,
        "_anchor_form_ids": form_ids,
    }


def _dispatch_expand(state: AgentState) -> list[Send]:
    """
    Conditional-edge routing function: builds one Send per expand target
    (link or form ID found by retrieval_anchor_node) — N determined by what
    anchor actually found, not fixed or LLM-guessed. If anchor found nothing
    to expand, Send straight to office_lookup_node so the graph still
    proceeds (an empty Send list would stall the graph, not skip forward).
    """
    targets = [
        Send("retrieval_expand_node", {**state, "expand_target": {"kind": "link", "value": link}})
        for link in state.get("_anchor_links", [])
    ] + [
        Send("retrieval_expand_node", {**state, "expand_target": {"kind": "form", "value": fid}})
        for fid in state.get("_anchor_form_ids", [])
    ]
    if not targets:
        return [Send("office_lookup_node", state)]
    return targets


def retrieval_expand_node(state: AgentState) -> AgentState:
    """
    One Send-dispatched branch: fetches exactly ONE expand target (a single
    get_page(link) or get_form(form_id) call) and contributes it to
    context_pages/sources via the operator.add reducer — LangGraph merges
    all N branches' contributions (plus retrieval_anchor_node's own) once
    every branch completes, before office_lookup_node runs.

    ⚠️ Must return ONLY the fields being updated (context_pages/sources), NOT
    `**state` — N branches run concurrently in the same superstep, and
    spreading the full state means every branch also "writes" every
    unchanged field (query, query_type, ...); those are plain last-value
    channels, so N parallel writes to the same non-reducer field raises
    InvalidUpdateError ("Can receive only one value per step"). Confirmed by
    hitting exactly this error before fixing it — not a hypothetical concern.
    """
    target = state.get("expand_target", {})
    kind, value = target.get("kind"), target.get("value")

    if kind == "link":
        page = get_page(value)
        if "error" in page:
            return {"context_pages": [], "sources": []}
        return {"context_pages": [page], "sources": [page["url"]]}

    if kind == "form":
        form = get_form(value)
        if "error" in form:
            return {"context_pages": [], "sources": []}
        page = {"url": form["url"], "title": form.get("form_title", ""), "text": form.get("text", "")}
        return {"context_pages": [page], "sources": [page["url"]]}

    return {"context_pages": [], "sources": []}


def office_lookup_node(state: AgentState) -> AgentState:
    """
    Inject office contact info into synthesis context.

    PROCEDURE: runs _offices_from_context() fresh, on the FULL context_pages
               (Step 4.5 fixed this 2026-08-26: originally prioritized
               detected_offices, which retrieval_anchor_node snapshots from
               ONLY its own anchor pages before the Send-based expand fan-out
               has run — office names that only appear in expand-fetched
               content, e.g. 休學's moltke form mentioning 住宿組/國際合作事務處,
               were silently missed. context_pages is the merged anchor+all-
               expand-branches set by the time THIS node runs (fan-in already
               complete via the operator.add reducer), so re-running the same
               detection function against it is provably a superset of the
               anchor-only snapshot — negligible cost (pure substring scan,
               no I/O/LLM), so there's no real reason to prefer the narrower
               snapshot. detected_offices is kept as a field (still useful as
               an early signal if expand's own targeting gets smarter later)
               but is no longer this node's primary source.
    CONTACT:   inject offices mentioned in the query; fallback to all if none found.
    KNOWLEDGE: skip — office info not needed for factual queries.

    OfficeLookupSkill's contact lookup is itself dynamic (condition 4,
    office_contacts_index.jsonl primary, KNOWN_CONTACTS fallback) — this
    node only decides WHICH offices to ask it about.
    """
    from rag.skills.office_lookup_skill import OfficeLookupSkill

    qtype = state["query_type"]

    if qtype == QueryType.KNOWLEDGE:
        return {**state, "office_context": "", "office_lookup_result": {}}

    if qtype == QueryType.PROCEDURE:
        offices = _offices_from_context(state.get("context_pages", [])) or _PROCEDURE_OFFICES
    else:  # CONTACT
        offices = _offices_from_query(state["query"]) or _PROCEDURE_OFFICES

    skill  = OfficeLookupSkill()
    result = skill.run(offices)
    return {
        **state,
        "office_context":       skill.format_context(result),
        "office_lookup_result": result,
    }


# Matches the numbering conventions Chinese-language forms use for
# "注意事項"/"說明"-style enumerated notes — Arabic-顿號 ("1、2、3、...",
# 休學's QP-T01-03-02), Chinese-numeral-顿號 ("一、二、三、...", 復學's
# QP-T01-03-04, which also nests an Arabic sub-list under one item), and
# Arabic-句點 ("1. 2. 3. ...", QP-T01-02-05's own "委託代辦說明" section —
# found 2026-08-26 while checking whether this regex generalized: it didn't,
# 0 notes extracted from that form until this separator was added).
# Deliberately a closed, enumerable set (unlike open-ended vocabularies such
# as office job titles) — Chinese formal documents only use a small number
# of numbering conventions, so widening this regex to cover them is not the
# same class of blind spot as _ADMIN_TITLES (office_lookup_skill.py); still,
# no claim this exhausts every convention (e.g. ①②③ or （一）（二） are not
# covered) — just the ones confirmed present in this corpus so far.
_NOTE_MARKER_RE = re.compile(r'(?:^|(?<=[。\s]))([一二三四五六七八九十]{1,3}|[0-9]{1,2})[、.]')
_TRAILING_TABLE_PIPES_RE = re.compile(r'(\s*\|)+\s*$')


def _extract_candidate_notes(text: str) -> list[str]:
    """
    Find "注意事項"-style enumerated notes and return each item as its own
    string. Scoped to ONE LINE at a time (`\\n`-split) rather than scanning
    the whole page/form text — 2026-08-26 lesson from a reverted earlier
    attempt: these notes live inside a single markdown table ROW (one
    source-line per row), sandwiched between unrelated rows before and
    after (e.g. QP-T01-03-02's 說明 row sits between a 住宿組/國際學生 row
    and a 申請人簽章/領取方式 row). Cutting from the first marker to the
    end of the whole text crossed those row boundaries and mangled both
    the notes and the unrelated rows around them. A line needs 2+ markers
    to count (avoids firing on a single stray "N、" inside ordinary prose,
    e.g. a line mentioning "7日內" without being a real enumerated list).
    """
    notes: list[str] = []
    for line in text.split("\n"):
        markers = list(_NOTE_MARKER_RE.finditer(line))
        if len(markers) < 2:
            continue
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(line)
            item = _TRAILING_TABLE_PIPES_RE.sub("", line[start:end]).strip()
            if item:
                notes.append(item)
    return notes


def extraction_node(state: AgentState) -> AgentState:
    """
    Condition 6: build a fact checklist from what retrieval/office_lookup
    actually found, BEFORE synthesis — v1 is pure regex, zero extra LLM calls.

    person_names: built directly from office_lookup_result's structured
                   {office: {contacts: [{name, duty, ext, email}, ...]}} —
                   reading OfficeLookupSkill's own per-office contact lists,
                   not regex re-parsing the flattened office_context string
                   (was `_PERSON_LINE_RE` matching "• 姓名（職責）分機 XXXXX"
                   lines — worked but was a fragile round-trip through
                   already-formatted text; office_lookup_result carries the
                   same data structured, so this reads it directly instead).
    forms:        form IDs parsed from context_pages text via agent_tools.extract_form_ids()
                   (reused, not reimplemented), each paired with its title when
                   the form's own page is in context_pages — lets the LLM judge
                   relevance instead of guessing from a bare ID.

    Replaces static content rules in _SYNTHESIS_PROMPT (old rules 3/5/7: fixed
    3-layer 教務處 review, fixed supplementary-form list, fixed 身心健康中心
    mention) with "here is what you found" — the LLM is told what's actually
    in the data instead of being told what to say regardless of the query.
    Directly targets the hallucination confirmed in Step 1 Q2 (復學 answer
    fabricated a 教務長 approval layer that doesn't exist in QP-T01-03-04).
    """
    person_names: list[dict] = []
    for office, info in state.get("office_lookup_result", {}).items():
        for c in info.get("contacts", []):
            name, duty, ext = c.get("name", ""), c.get("duty", ""), c.get("ext", "")
            if not name or not ext:
                continue
            person_names.append({"name": name, "duty": duty, "ext": ext})

    context_pages = state.get("context_pages", [])

    form_ids: set[str] = set()
    for page in context_pages:
        form_ids.update(extract_form_ids(page.get("text", "")))

    # Attach a title to each form ID when its own page is in context_pages
    # (true whenever ProcedureSkill actually called get_form() on it) — a bare
    # ID like "QP-T01-03-05" gives the LLM nothing to judge relevance by; the
    # title ("學生退學離校申請書") makes it obvious that one's a different
    # procedure (退學, not 休學) and should be left out, while a form whose
    # title clearly complements the question should be kept.
    forms: list[dict] = []
    for fid in sorted(form_ids):
        title = next(
            (p.get("title", "") for p in context_pages if fid in p.get("url", "")),
            "",
        )
        forms.append({"id": fid, "title": title})

    # Tagged with source_title (not flattened into one pool) so the prompt
    # can present notes grouped by which page/form they came from — 2026-08-26
    # finding: a straight flat list mixed 休學's own 6-8 notes with 7 more
    # from 復學's QP-T01-03-04 (cross-referenced on the same aca overview
    # page, fetched by expand alongside 休學's own form), diluting the
    # candidate pool with a different procedure's content. Reuses the SAME
    # title-based relevance judgment _SYNTHESIS_PROMPT already asks the LLM
    # to apply to `forms` below, instead of inventing a new filter.
    notes: list[dict] = []
    seen_notes: set[str] = set()
    for page in context_pages:
        source_title = page.get("title", "")
        for note in _extract_candidate_notes(page.get("text", "")):
            if note not in seen_notes:
                seen_notes.add(note)
                notes.append({"text": note, "source_title": source_title})

    checklist = {
        "person_names": person_names,
        "forms":        forms,
        "notes":        notes,
    }
    print(
        f"[extraction] {len(person_names)} person(s), "
        f"forms={[(f['id'], f['title']) for f in forms]}, "
        f"notes={len(notes)}",
        file=sys.stderr,
    )
    return {**state, "extraction_checklist": checklist}


def _format_checklist(checklist: dict) -> str:
    """Format extraction_checklist into a prompt-injectable block, or "" if empty."""
    if not checklist:
        return ""

    lines = ["【本次搜尋結果中實際找到的事實 checklist——以下項目若跟問題相關就必須出現在答案中，"
              "不要遺漏；checklist 沒列出的人名/表單/審核層級也不要自行補上】"]

    if checklist.get("person_names"):
        lines.append("承辦人：")
        for p in checklist["person_names"]:
            lines.append(f"  • {p['name']}（{p['duty']}）分機 {p['ext']}")

    if checklist.get("forms"):
        lines.append("相關表單（逐一判斷是否與學生問題相關；標題明顯是同一流程或直接後續步驟的，"
                     "必須在答案中列出，不可省略；標題明顯是不同流程的可以略過）：")
        for f in checklist["forms"]:
            title = f"（{f['title']}）" if f["title"] else ""
            lines.append(f"  • {f['id']}{title}")

    if checklist.get("notes"):
        lines.append("表單/頁面列出的補充說明事項（按來源分組——判斷相關性的方式跟上面「相關表單」規則一樣："
                     "來源標題明顯是同一流程的，其補充說明若與學生情況相關就適當納入答案；"
                     "來源標題明顯是不同流程的，整組可以略過，不需要每一則都寫進答案）：")
        last_source = None
        for n in checklist["notes"]:
            if n["source_title"] != last_source:
                lines.append(f"  [來源：{n['source_title'] or '（無標題）'}]")
                last_source = n["source_title"]
            lines.append(f"    • {n['text']}")

    if len(lines) == 1:
        return ""  # nothing extracted — omit the section entirely
    return "\n".join(lines) + "\n\n"


def synthesis_node(state: AgentState) -> AgentState:
    """
    Generate the final answer from retrieved context + office contact info.

    KNOWLEDGE: answer already produced by agent loop → pass through.
    PROCEDURE / CONTACT: call LLM with context_pages + office_context +
    extraction_checklist (condition 6).
    On E4 retry: prepends correction_hint so the LLM knows what to fix.
    """
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


# ── E4: Self-evaluation checklist ────────────────────────────────────────────

_MAX_SELF_EVAL_RETRIES = 2

# person_names dropped from here — condition 6 made it dynamic (see self_eval_node
# below): it now checks against extraction_checklist's actual person_names for
# THIS query instead of a fixed list of 6 休學-specific names hardcoded regardless
# of query type. sources/procedure_format stay static — they're format checks,
# not content specific to any one procedure.
def _char_bigrams(s: str) -> set[str]:
    s = re.sub(r'\s+', '', s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}


def _note_referenced(note: str, answer: str, threshold: float = 0.4) -> bool:
    """
    Posthoc similarity check (FullCite-inspired, 2026-08-26) for whether a
    checklist note's content shows up somewhere in the answer — used
    instead of requiring the LLM to emit a rigid citation ID (which
    "Models Can Model, But Can't Bind" and this session's own experience
    both suggest is an unreliable ask of an untrained model) or requiring
    literal substring match (too strict — the LLM is expected to
    paraphrase/summarize a note, not quote the source verbatim). Character
    bigram overlap coefficient (bigrams(note) ∩ bigrams(answer) /
    bigrams(note)) rather than full Jaccard, since a short note compared
    against the whole (much longer) answer via symmetric Jaccard would
    always score near-zero regardless of match quality. Threshold 0.4
    empirically separates a real paraphrase (~0.76 on a real 休學 note) from
    unrelated answer text (~0.06) with margin to spare.
    """
    note_bg = _char_bigrams(note)
    if not note_bg:
        return False
    answer_bg = _char_bigrams(answer)
    return len(note_bg & answer_bg) / len(note_bg) >= threshold


# ── Condition 8-C (Step 6): Router-as-judge semantic check ──────────────────
# Standalone helper, NOT yet wired into self_eval_node's retry loop — keyword
# checks (_SELF_EVAL_CRITERIA below) can only verify certain strings appear,
# they structurally cannot catch "format right, content off-topic" answers
# (e.g. Domain Router's osa-dilution finding: a 復學 answer that's actually
# mostly about dorm eligibility, but still contains "復學" and step-list
# formatting so every keyword check passes). Known weakness from literature
# (Zheng et al. 2023, MT-Bench "LLM-as-judge"): the judge itself can be
# wrong or format-sensitive — validated here against curated known-good/
# known-bad examples BEFORE deciding whether to wire it into the live retry
# loop, not assumed reliable by default.
_ROUTER_JUDGE_SYSTEM = "你是 NCCU 學術事務 Q&A 系統的品質判官，只根據語意判斷答案內容是否真的回答了問題，不看格式對不對。"

_ROUTER_JUDGE_USER = """學生問題：{query}

系統答案：
{answer}

請判斷：這個答案的內容是否真的針對「{query}」這個問題本身作答？特別留意「格式正確、關鍵字也對，但內容其實主要在講別的主題」這種情況——例如答案主要在介紹某個不相關的規定或活動，只是剛好提到問題裡的關鍵字或表面相關。

只回傳一個 JSON，不要有其他文字：{{"verdict": "PASS 或 FAIL", "reason": "一句話原因"}}"""


def _router_judge(query: str, answer: str) -> dict:
    """Send (query, answer) to the LLM for a semantic PASS/FAIL judgment."""
    raw = simple_chat(
        messages=[
            {"role": "system", "content": _ROUTER_JUDGE_SYSTEM},
            {"role": "user",   "content": _ROUTER_JUDGE_USER.format(query=query, answer=answer)},
        ],
        max_tokens=200,
    )
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"verdict": "ERROR", "reason": raw[:200]}


_SELF_EVAL_CRITERIA = [
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

    # Condition 6: dynamic person_names check against this query's own
    # extraction_checklist instead of a hardcoded 休學-specific name list.
    checklist_names = [p["name"] for p in state.get("extraction_checklist", {}).get("person_names", [])]
    if checklist_names:
        required = min(2, len(checklist_names))
        hits = sum(1 for name in checklist_names if name in answer)
        if hits < required:
            failures.append(
                f"答案缺少承辦人姓名（需 ≥{required} 位，checklist 中找到：{'、'.join(checklist_names)}）。"
                "請在每個步驟列出對應承辦人姓名（從【各辦公室聯絡資訊】引用）。"
            )

    # 2026-08-26: tried a dynamic check here — "does the answer show
    # multiple review layers for offices whose real contact roster has 2+
    # distinct job titles" — as a generalization of eval.py's static
    # nine_stamps criterion. Reverted: distinct-duty-count in
    # OfficeLookupSkill's contact roster measures staffing diversity (how
    # many different job titles an office happens to employ — 住宿組 has 13,
    # purely from having lots of staff with different day-to-day jobs), not
    # documented approval-chain depth. That's the wrong data source — it
    # would force 2+ named contacts onto offices that don't need them,
    # recreating the original roster-bloat bug for offices other than the
    # one it was meant to fix. The right signal (whether the PROCEDURE
    # ITSELF documents multiple approval sub-stages for a step — e.g.
    # QP-T01-03-02's table literally splits 教務處's row into 組長 and
    # 教務長 sub-cells) lives in context_pages' table structure, not in the
    # office contact roster; extracting that is a separate, harder task,
    # not done yet. See phase_f_planning_report.md Gotchas.

    # mental_health_note (condition 6 extension, 2026-08-26): dynamic check
    # against extraction_checklist's own "notes" — enumerated 注意事項 items
    # regex-extracted from the actual form/page text (see
    # _extract_candidate_notes), not any procedure-specific hardcoded
    # phrase. Posthoc bigram-overlap matching (_note_referenced) instead of
    # requiring exact substring or a citation ID, since the LLM is expected
    # to paraphrase a note, not quote it verbatim. Same min(2, len()) floor
    # as person_names — not "all notes must appear" (some may genuinely be
    # irrelevant to this student's situation; forcing all of them in would
    # risk the same over-inclusion problem office_role_diversity just hit).
    checklist_notes = state.get("extraction_checklist", {}).get("notes", [])
    if checklist_notes:
        required = min(2, len(checklist_notes))
        hits = sum(1 for n in checklist_notes if _note_referenced(n["text"], answer))
        if hits < required:
            unreferenced = [n["text"] for n in checklist_notes if not _note_referenced(n["text"], answer)]
            preview = "；".join(n[:40] + ("…" if len(n) > 40 else "") for n in unreferenced[:3])
            failures.append(
                f"答案未充分反映表單的補充說明事項（需至少 {required} 則有對應到答案內容，"
                f"目前只有 {hits} 則；例如：{preview}）。"
                "請檢視這些補充說明是否與學生情況相關，相關的請適當納入答案（可用自己的話摘要，不需逐字照抄）。"
            )

    # Condition 8-C (Step 6): Router-as-judge, gated on domain_router's
    # is_ambiguous() so the extra LLM call only happens on queries where
    # routing was shaky to begin with (e.g. "如何辦理復學" — the real
    # osa-dilution case this was built for), not on every PROCEDURE answer.
    # On FAIL, the correction explicitly asks for an honest decline rather
    # than "try again" — re-synthesizing from the SAME context_pages can't
    # fix a wrong-subdomain retrieval, so asking for a rewrite would just
    # produce another confidently-off-topic answer. This is a stopgap, not
    # a fix for the underlying routing gap — see phase_f_planning_report.md
    # (Step 5's risk 2 / future Layer 3 CRAG-lite retry is the real fix).
    if is_ambiguous(state["query"]):
        judge = _router_judge(state["query"], answer)
        print(f"[router-judge] {judge.get('verdict')}: {judge.get('reason', '')}", file=sys.stderr)
        if judge.get("verdict") == "FAIL":
            failures.append(
                f"Router-as-judge 判定答案可能離題（{judge.get('reason', '')}）。"
                "若上方【搜尋到的頁面內容】明顯與問題主題不符，請不要勉強套用不相關的資料組出答案，"
                "改為誠實回答「資料不足以確認」並說明需要進一步查證。"
            )

    if not failures:
        return state  # all checks passed

    correction = (
        "【自評回饋：以下項目不符合要求，請重新生成更完整的答案】\n"
        + "\n".join(f"- {f}" for f in failures)
    )
    return {**state, "correction_hint": correction, "iteration": iteration + 1}


# ── Conditional routing ───────────────────────────────────────────────────────

def _after_decomposition(state: AgentState) -> str:
    if state.get("sub_queries"):
        return "merge_node"
    return "router_node"


def _after_router(state: AgentState) -> str:
    if state["query_type"] == QueryType.CONTACT:
        return "office_lookup_node"
    if state["query_type"] == QueryType.PROCEDURE:
        return "retrieval_anchor_node"  # Step 4.5: anchor+expand, not retrieval_node
    return "retrieval_node"  # KNOWLEDGE


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

    g.add_node("query_decomposition_node", query_decomposition_node)
    g.add_node("merge_node",           merge_node)
    g.add_node("router_node",          router_node)
    g.add_node("retrieval_anchor_node", retrieval_anchor_node)
    g.add_node("retrieval_expand_node", retrieval_expand_node)
    g.add_node("retrieval_node",       retrieval_node)
    g.add_node("office_lookup_node",   office_lookup_node)
    g.add_node("extraction_node",      extraction_node)
    g.add_node("synthesis_node",       synthesis_node)
    g.add_node("self_eval_node",       self_eval_node)

    g.add_edge(START, "query_decomposition_node")
    g.add_conditional_edges("query_decomposition_node", _after_decomposition)
    g.add_edge("merge_node",            "synthesis_node")
    g.add_conditional_edges("router_node", _after_router)
    g.add_conditional_edges("retrieval_anchor_node", _dispatch_expand)
    g.add_edge("retrieval_expand_node", "office_lookup_node")
    g.add_edge("retrieval_node",        "office_lookup_node")
    g.add_edge("office_lookup_node",    "extraction_node")
    g.add_edge("extraction_node",       "synthesis_node")
    g.add_edge("synthesis_node",        "self_eval_node")
    g.add_conditional_edges("self_eval_node", _after_self_eval)

    return g.compile()


_graph = None


def run(query: str) -> str:
    global _graph
    if _graph is None:
        _graph = build_graph()

    final = _graph.invoke({
        "query":                query,
        "query_type":           "",
        "route_method":         "",
        "sub_queries":          [],
        "context_pages":        [],
        "sources":              [],
        "detected_offices":     [],
        "_anchor_links":        [],
        "_anchor_form_ids":     [],
        "expand_target":        {},
        "office_context":       "",
        "office_lookup_result": {},
        "extraction_checklist": {},
        "answer":               "",
        "correction_hint":      "",
        "iteration":            0,
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
