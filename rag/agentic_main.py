"""
rag/agentic_main.py
Step 13 (Phase F/G) — standalone, experimental LangGraph pipeline.

Collapses the old PROCEDURE/RESOURCE/CONTACT/KNOWLEDGE query_type routing into
one unified KNOWLEDGE-style agent loop: rewrite_node -> agent_node -> tools_node,
looping until the agent judges it has enough to answer. grep_texts (old,
literal) is kept as an agent-selectable fallback alongside search_texts (new,
FTS5 + LLM judge) -- neither replaces the other.

NOT wired into proto3_langgraph.py's production graph. This module is
deliberately standalone so it carries zero risk to the existing, validated
26/26 PROCEDURE path. Converts the plain-Python-loop validation done in this
session's scratchpad testing into real LangGraph nodes/edges, specifically so
`.stream(stream_mode="updates")` gives per-node observability for free --
this was the whole motivation for moving off the scratchpad script (see
docs/phase_f_planning_report.md and the Step 13 section of the project plan
for the full design history and the many known-open issues this build does
NOT yet address: judge has no answer-type awareness, agent has no explicit
"give up searching, answer with the general rule" stopping heuristic,
resource_node/contact_node/domain_router-repositioning/final_rewrite_node/
synthesis_node are all still out of scope for this first real-graph pass).

Usage:
    python -m rag.agentic_main "如何辦理休學"
    python -m rag.agentic_main "如何辦理休學" --subdomain aca --stream
"""

from __future__ import annotations

import json
import operator
import sys
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from rag.llm_client import chat_with_tools, simple_chat
from rag.keyword_store import KeywordStore
from rag.agent_tools import grep_texts, get_page, extract_links, get_form

FTS_DB = "rag/fts_proto3.db"
_MAX_TURNS = 8
_MAX_STUCK = 3


# ── State ─────────────────────────────────────────────────────────────────

class Step13State(TypedDict):
    query: str
    subdomain_hint: str | None
    turn: int
    rewritten: str
    turn_log: Annotated[list[dict], operator.add]
    context_pages: Annotated[list[dict], operator.add]
    seen_calls: Annotated[list[str], operator.add]
    stuck_turns: int
    pending_tool_calls: list[dict]
    answer: str | None


# ── Prompts (unchanged from the validated scratchpad version) ──────────────

_REWRITE_PROMPT = """你是一個搜尋詞改寫助手，任務是把使用者的問題改寫成最適合查詢政大官網全文索引的搜尋詞。

請依以下步驟判斷（只需輸出最後結果，不要輸出思考過程）：
1. 找出問題的核心主題——通常是2-6字的名詞或動詞片語，代表使用者真正想辦理/查詢的事項
2. 忽略以下不影響核心主題的內容：
   - 身分/情境描述（例如「我是大學部學生」「我是碩士生」「在職生」）
   - 疑問句型（例如「怎麼辦理」「如何申請」「要辦」「想問」）
   - 禮貌用語或語助詞（例如「請問」「謝謝」）
3. 保留：專有名詞、規定/制度名稱、表單相關字眼（如果使用者明確提到表單/文件）

輸出格式：只輸出改寫後的搜尋詞，不要有其他文字、不要加引號、不要解釋。

範例：
輸入：如何辦理休學
輸出：休學

輸入：我是大學部學生，想問要怎麼辦理休學
輸出：休學

輸入：休學申請表在哪裡下載
輸出：休學表單

輸入：我是碩士生，復學要怎麼辦
輸出：復學

現在請改寫：
輸入：{query}
輸出："""

_JUDGE_PROMPT = """你是搜尋結果的「起點頁面」判官，不是最終答案判官。這個系統的完整流程是：先選一個相關的起點頁面(anchor)，再抓取該頁面全文，並視情況追蹤裡面提到的表單、連結，最後才彙整成完整答案——你的工作只負責「選對起點」，不需要這一頁本身就把問題完整回答完。

原始問題：{query}

候選頁面（依BM25分數排序）：
{candidates}

請判斷：這份候選清單裡，有沒有一個頁面主題上屬於使用者原始問題所屬的同一個程序/規定？不限於分數最高的那個，只要清單裡任何一個候選頁面談的是同一件事（同一個程序名稱、同一類規定），就算找到了，不需要這頁本身就列出完整步驟或涵蓋所有細節。只有在清單裡全部候選頁面都明顯是討論完全不同的主題時，才判定沒找到。

只回傳一個JSON，不要有其他文字：
{{"good_enough": true 或 false, "selected_index": 找到的候選頁面編號(從1開始；若good_enough為false則填null), "reason": "一句話說明"}}
"""

_AGENT_SYSTEM = """你是政大學務問答系統的搜尋agent，任務是找到能回答使用者問題的頁面內容。

你有search_texts/grep_texts/get_page/extract_links/get_form這幾個工具可以用（見工具說明）。

**重要規則**：search_texts或grep_texts回傳的候選只有標題跟預覽，不是完整內容。如果操作記錄裡剛找到一個看起來有希望的候選（標題主題相符），下一步幾乎都該是對那個候選的URL呼叫get_page取得全文，不要對同一個主題再搜一次或改搜候選的標題文字——重複搜尋不會得到比全文更多的資訊，只有get_page才能真正確認候選內容對不對。只有在get_page讀完全文、確認這個候選不是答案（且內文沒有指向其他文件的線索）時，才考慮換關鍵字重新搜尋。

已經執行過的操作記錄：
{turn_log_rendered}

這一輪系統建議的搜尋方向：{rewritten}

原始問題：{query}

請判斷下一步該做什麼。如果你已經有足夠資訊能回答原始問題了，不要呼叫任何工具，但你的文字回覆本身就必須「是」完整答案——直接寫出具體的申請流程/條件/期限等實際內容並附來源URL，不要寫「我會列出...」「我將說明...」這種描述你打算做什麼的句子，那不是答案，是你還沒寫答案。"""

_TOOLS = [
    {"type": "function", "function": {
        "name": "search_texts",
        "description": "語意搜尋——用這一輪系統已經幫你重寫好的搜尋詞，內部會查FTS5全文索引並判斷候選是否相關。不需要自己給關鍵字，優先使用這個工具。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "grep_texts",
        "description": "純字面比對搜尋，沒有相關性排序，自己指定關鍵字。當search_texts找不到好結果、或你想用一個更精確/不同角度的詞直接試時使用。",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "subdomain": {"type": "string"}}, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "get_page",
        "description": "取得已知URL的完整頁面內容。只在已有明確URL時呼叫。",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "extract_links",
        "description": "取得頁面正文明確提到的其他連結。",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "get_form",
        "description": "取得已知表單編號的完整內容。",
        "parameters": {"type": "object", "properties": {"form_id": {"type": "string"}}, "required": ["form_id"]},
    }},
]


# ── Helpers ──────────────────────────────────────────────────────────────

def _render_turn_log(turn_log: list[dict]) -> str:
    if not turn_log:
        return "（尚無記錄）"
    return "\n".join(
        f"[第{e['turn']}輪] {e['actor']}" + (f"({e['tool']})" if e.get("tool") else "")
        + f" -- 目的:{e['purpose']} / 結果:{e['outcome']}"
        for e in turn_log
    )


def _rewrite_query(text: str) -> str:
    prompt = _REWRITE_PROMPT.format(query=text)
    return simple_chat(messages=[{"role": "user", "content": prompt}], max_tokens=50).strip()


def _judge_candidates(query: str, candidates: list[dict]) -> dict:
    candidates_str = "\n".join(
        f"{i+1}. [{c['subdomain']}] {c['title']} (score={c['bm25_score']})\n   {c['text_clean'][:200]}"
        for i, c in enumerate(candidates)
    )
    prompt = _JUDGE_PROMPT.format(query=query, candidates=candidates_str)
    raw = simple_chat(messages=[{"role": "user", "content": prompt}], max_tokens=200)
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"good_enough": None, "selected_index": None, "reason": f"JSON解析失敗: {raw[:200]}"}


def _do_search_texts(query: str, rewritten: str, subdomain: str | None) -> tuple[dict, str]:
    ks = KeywordStore(db_path=FTS_DB)
    candidates = ks.search(rewritten, top_k=5, subdomain=subdomain)
    if not candidates:
        candidates = ks.search(rewritten, top_k=5, subdomain=None)
    if not candidates:
        return {"error": "no candidates"}, f"search_texts('{rewritten}') 無候選結果"
    judge = _judge_candidates(query, candidates)
    if not judge.get("good_enough"):
        return {"error": "judge rejected", "judge": judge}, \
            f"search_texts('{rewritten}') 候選皆不相關，judge理由: {judge.get('reason','')}"
    idx = max(0, min((judge.get("selected_index") or 1) - 1, len(candidates) - 1))
    sel = candidates[idx]
    return {"url": sel["url"], "title": sel["title"]}, \
        f"search_texts('{rewritten}') 找到候選 [{sel['subdomain']}] {sel['title']}\n      URL: {sel['url']}\n      (judge通過: {judge.get('reason','')})"


def _dispatch_tool(name: str, args: dict, query: str, rewritten: str, subdomain_hint: str | None):
    if name == "search_texts":
        return _do_search_texts(query, rewritten, subdomain_hint)
    if name == "grep_texts":
        r = grep_texts(args.get("pattern", ""), subdomain=args.get("subdomain") or None, max_results=5)
        if not r:
            return {"results": []}, f"grep_texts('{args.get('pattern')}') 0筆結果"
        top3 = "\n      ".join(f"{i+1}. {x['title'][:40]} | URL: {x['url']}" for i, x in enumerate(r[:3]))
        return {"results": r}, f"grep_texts('{args.get('pattern')}') 找到{len(r)}筆，前3筆:\n      {top3}"
    if name == "get_page":
        p = get_page(args.get("url", ""))
        if "error" in p:
            return p, f"get_page 失敗: {p['error']}"
        return p, f"get_page 取得 {len(p.get('text',''))} 字，標題: {p.get('title','')[:30]}"
    if name == "extract_links":
        links = extract_links(args.get("url", ""))
        return {"links": links}, f"extract_links 找到 {len(links)} 個連結"
    if name == "get_form":
        f = get_form(args.get("form_id", ""))
        if "error" in f:
            return f, f"get_form 失敗: {f['error']}"
        return f, f"get_form 取得表單 {f.get('form_title','')[:30]}"
    return {"error": f"unknown tool {name}"}, f"未知工具 {name}"


# ── Nodes ────────────────────────────────────────────────────────────────

def rewrite_node(state: Step13State) -> dict:
    turn = state.get("turn", 0) + 1
    turn_log = state.get("turn_log", [])
    basis = state["query"] if not turn_log else f"{state['query']}\n\n已知進度：\n{_render_turn_log(turn_log)}"
    rewritten = _rewrite_query(basis)
    return {
        "turn": turn,
        "rewritten": rewritten,
        "turn_log": [{"turn": turn, "actor": "rewrite_node", "tool": None,
                       "purpose": "重寫本輪搜尋方向", "outcome": rewritten}],
    }


def agent_node(state: Step13State) -> dict:
    system = _AGENT_SYSTEM.format(
        turn_log_rendered=_render_turn_log(state.get("turn_log", [])),
        rewritten=state["rewritten"], query=state["query"])
    content, tool_calls = chat_with_tools(
        messages=[{"role": "user", "content": "請判斷下一步。"}],
        tools=_TOOLS, system_prompt=system)

    extra_log: list[dict] = []
    if not tool_calls:
        votes, candidate = 0, None
        for _ in range(2):
            c2, tc2 = chat_with_tools(
                messages=[{"role": "user", "content": "請判斷下一步。"}],
                tools=_TOOLS, system_prompt=system)
            if tc2:
                votes += 1
                candidate = candidate or (c2, tc2)
        if votes >= 1 and candidate is not None:
            extra_log.append({"turn": state["turn"], "actor": "system", "tool": None,
                               "purpose": "8-A重採樣",
                               "outcome": f"初次未呼叫工具，重採樣後{votes}/2票選擇呼叫，改用重採樣結果"})
            content, tool_calls = candidate

    return {
        "pending_tool_calls": tool_calls or [],
        "answer": content if not tool_calls else None,
        "turn_log": extra_log,
    }


def _after_agent(state: Step13State) -> str:
    if not state.get("pending_tool_calls"):
        return "end"
    if state["turn"] >= _MAX_TURNS:
        return "end"
    return "tools"


def tools_node(state: Step13State) -> dict:
    turn = state["turn"]
    seen = set(state.get("seen_calls", []))
    new_context_pages, new_turn_log, new_seen = [], [], []
    new_calls_this_turn = 0

    for tc in state["pending_tool_calls"]:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            args = {}
        sig = f"{name}:{state['rewritten']}" if name == "search_texts" else f"{name}:{args}"
        if sig in seen:
            new_turn_log.append({"turn": turn, "actor": "tools_node", "tool": name,
                                  "purpose": "(重複呼叫，略過)", "outcome": "已執行過相同呼叫"})
            continue
        new_seen.append(sig)
        new_calls_this_turn += 1
        result, outcome_str = _dispatch_tool(name, args, state["query"], state["rewritten"], state.get("subdomain_hint"))
        new_turn_log.append({"turn": turn, "actor": "tools_node", "tool": name,
                              "purpose": f"agent呼叫{name}({args})", "outcome": outcome_str})
        if name == "get_page" and "error" not in result:
            new_context_pages.append(result)

    stuck_turns = state.get("stuck_turns", 0)
    stuck_turns = stuck_turns + 1 if new_calls_this_turn == 0 else 0
    if stuck_turns >= _MAX_STUCK:
        new_turn_log.append({"turn": turn, "actor": "system", "tool": None,
                              "purpose": "doom-loop偵測", "outcome": f"連續{stuck_turns}輪無新呼叫，中止"})

    return {
        "context_pages": new_context_pages,
        "turn_log": new_turn_log,
        "seen_calls": new_seen,
        "stuck_turns": stuck_turns,
    }


def _after_tools(state: Step13State) -> str:
    if state.get("stuck_turns", 0) >= _MAX_STUCK:
        return "end"
    if state["turn"] >= _MAX_TURNS:
        return "end"
    return "rewrite"


# ── Graph assembly ───────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(Step13State)
    g.add_node("rewrite_node", rewrite_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tools_node", tools_node)

    g.add_edge(START, "rewrite_node")
    g.add_edge("rewrite_node", "agent_node")
    g.add_conditional_edges("agent_node", _after_agent, {"tools": "tools_node", "end": END})
    g.add_conditional_edges("tools_node", _after_tools, {"rewrite": "rewrite_node", "end": END})

    return g.compile()


def run(query: str, subdomain_hint: str | None = None, stream: bool = False) -> dict:
    graph = build_graph()
    initial: Step13State = {
        "query": query, "subdomain_hint": subdomain_hint, "turn": 0, "rewritten": "",
        "turn_log": [], "context_pages": [], "seen_calls": [], "stuck_turns": 0,
        "pending_tool_calls": [], "answer": None,
    }

    if stream:
        final_state = initial
        for update in graph.stream(initial, stream_mode="updates"):
            for node_name, delta in update.items():
                print(f"[{node_name}] {list(delta.keys())}", file=sys.stderr)
                final_state = {**final_state, **{k: (final_state.get(k, []) + v if isinstance(v, list) else v)
                                                  for k, v in delta.items()}}
        return final_state

    return graph.invoke(initial)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--subdomain", default=None)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    result = run(args.query, subdomain_hint=args.subdomain, stream=args.stream)

    print(f"\n{'='*70}")
    print(f"Query: {args.query!r}")
    print(f"{'='*70}")
    for e in result.get("turn_log", []):
        tag = f"[{e['tool']}]" if e.get("tool") else ""
        print(f"  [{e['turn']}] {e['actor']}{tag}: {e['purpose']} -> {e['outcome'][:100]}")
    print(f"\n最終答案 ({result.get('turn', 0)}輪):")
    print(result.get("answer"))
