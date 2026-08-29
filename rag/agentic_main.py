"""
rag/agentic_main.py
Step 13 (Phase F/G) — standalone, experimental LangGraph pipeline.
Transitional name (not `main.py` yet — see project plan for why).

v2: real ToolNode + ChatOllama.bind_tools(), replacing the v1 custom
dispatcher + native `ollama` client (rag/llm_client.py's chat_with_tools()).
2026-08-28 spike confirmed ChatOllama+bind_tools() against Ollama Cloud
(gemma4:31b-cloud) reliably returns structured tool_calls (3/3 stable) when
the system prompt explicitly instructs tool use — the ChatOllama/Ollama
Cloud compatibility question left open since Step 6 is now verified, not
theoretical. Real ToolNode buys: automatic parallel tool_call execution,
built-in per-tool error handling, standard ToolMessage/AIMessage plumbing —
replacing this session's hand-rolled dispatcher + turn_log bookkeeping.

Collapses the old PROCEDURE/RESOURCE/CONTACT/KNOWLEDGE query_type routing
into one unified KNOWLEDGE-style agent loop: rewrite_node -> agent_node ->
tools (real ToolNode), looping until the agent judges it has enough to
answer. grep_texts (old, literal) is kept as an agent-selectable fallback
alongside search_texts (new, FTS5 + LLM judge) -- neither replaces the other.

NOT wired into proto3_langgraph.py's production graph.

Usage:
    python -m rag.agentic_main "如何辦理休學"
    python -m rag.agentic_main "如何辦理休學" --subdomain aca --stream
"""

from __future__ import annotations

import json
import operator
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from rag.llm_client import simple_chat
from rag.keyword_store import KeywordStore
from rag.agent_tools import (
    grep_texts as _grep_texts_raw,
    get_page as _get_page_raw,
    extract_links as _extract_links_raw,
    get_form as _get_form_raw,
    extract_form_ids,
)
from rag.domain_router import _layer1_match, layer2_candidates, is_ambiguous

FTS_DB = "rag/fts_proto3.db"
_MAX_STUCK = 3  # doom-loop detection only -- no turn-count cap this round, see _after_agent

# Known office mandates -- deliberately incomplete (only core offices seen in
# testing so far). Other subdomains show "尚無職掌描述" in domain_router_node's
# candidate message; see docs/phase_g_clean_pipeline_design.md §B.4.
_SUBDOMAIN_DESC = {
    "aca": "教務處：負責註冊/學籍/畢業/休復學等",
    "osa": "學務處：負責宿舍/獎學金/社團等",
    "cashier": "出納組：負責繳費/退費",
}

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")


# ── State ─────────────────────────────────────────────────────────────────

class AgenticState(TypedDict):
    query: str
    subdomain_hint: str | None
    turn: int
    rewritten: str
    stuck_turns: int
    messages: Annotated[list, add_messages]
    answer: str | None


# ── Rewrite / judge (unchanged prompts, reused logic) ───────────────────────

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

_AGENT_SYSTEM = """你是政大學務問答系統的搜尋agent，任務是找到能回答使用者問題的頁面內容，並用找到的內容回答問題。

你必須使用search_texts/grep_texts/get_page/extract_links/get_form等工具搜尋政大官方資料來回答問題，絕對不可以直接用自己的知識回答，因為政大的規定可能隨時變動，你的訓練知識可能過時或不準確。

**重要規則**：search_texts或grep_texts回傳的候選只有標題跟預覽，不是完整內容。如果剛找到一個看起來有希望的候選（標題主題相符），下一步幾乎都該是對那個候選的URL呼叫get_page取得全文，不要對同一個主題再搜一次或改搜候選的標題文字——只有get_page才能真正確認候選內容對不對。只有在get_page讀完全文、確認這個候選不是答案（且內文沒有指向其他文件的線索）時，才考慮換關鍵字重新搜尋。

如果你已經有足夠資訊能回答原始問題了，不要呼叫任何工具，但你的文字回覆本身就必須「是」完整答案——直接寫出具體的申請流程/條件/期限等實際內容並附來源URL，不要寫「我會列出...」「我將說明...」這種描述你打算做什麼的句子，那不是答案，是你還沒寫答案。

**表單編號規則**：如果get_page的回傳裡出現「[偵測到表單編號: ...]」這個標記，代表頁面提到官方表單。procedure頁面的文字敘述常常只是概略帶過，實際的細節（可能是蓋章站點、可能是費用/退費標準、可能是資格條件或期限——類型不固定，視表單本身而定）常常只寫在表單裡，procedure頁面不會重複。如果你不確定procedure頁面的敘述是否已經涵蓋了回答這題所需的全部實務細節，該呼叫get_form(form_id=...)確認，比只憑procedure頁面的文字敘述回答更保險。若問題只是單純問「在哪裡下載」，看到的markdown連結本身就已經是答案，不需要額外呼叫get_form。

**範例1：get_page讀完全文後，內容確實回答了問題，該直接作答**
情境：問題是「如何辦理休學」，你已經呼叫get_page讀到「休學規定」頁面全文，內容包含申請方式、時間、費用等。
正確做法：不呼叫任何工具，直接輸出：「辦理休學的流程如下：1. 填寫休學申請書...2. ...（實際列出全文裡的具體條文內容）...來源：[URL]」
錯誤做法：輸出「我已經取得完整內容，將列出休學的申請條件、辦理流程...」——這只是描述你打算做什麼，沒有把內容本身寫出來，等於還沒回答。

**範例2：問題裡有身分/情境修飾詞，搜尋2-3次仍找不到該修飾詞的專屬規定時，該用一般規則作答並誠實註明**
情境：問題是「在職生怎麼辦理復學」，你已經get_page讀到一般性的「復學」規定頁面（沒有特別提到「在職生」），也嘗試搜尋「在職生 復學」「在職生」等詞2-3次都找不到專屬於在職生的特殊規定。
正確做法：不要再繼續搜尋，直接用已有的一般復學規定作答，並加一句「未查到在職生專屬的特殊規定，以下為一般復學流程」。
錯誤做法：持續換不同關鍵字搜尋超過2-3次仍找不到，既不作答也不停止——這是在浪費輪次，一般規則沒有明確排除某身分時，適用一般規則是合理的假設，比無限期搜尋更有用。"""


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


def _render_messages(messages: list) -> str:
    lines = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"[agent決定呼叫] {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            lines.append(f"[工具結果] {str(m.content)}")
    return "\n".join(lines) if lines else "（尚無記錄）"


def _fetched_urls(messages: list) -> set[str]:
    """URLs the agent has already called get_page_tool on this run -- used by
    grep_texts_tool to avoid re-surfacing pages the agent has already fully
    read as if they were new candidates."""
    urls = set()
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc["name"] == "get_page_tool":
                    u = tc["args"].get("url")
                    if u:
                        urls.add(u)
    return urls


# ── Tools — real LangChain @tool functions, dispatched by real ToolNode ────

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
    return f"{header}\n\n全文：\n{text}"


@tool
def extract_links_tool(url: str, state: Annotated[dict, InjectedState]) -> str:
    """取得頁面正文明確提到的其他連結。當目前頁面內容不足以回答問題、但內文提到其他相關文件或頁面時使用。"""
    links = _extract_links_raw(url, subdomain=state.get("subdomain_hint"))
    if not links:
        return "extract_links 找到 0 個連結"
    return "extract_links 找到:\n" + "\n".join(f"- {l['label']}: {l['url']}" for l in links[:10])


@tool
def get_form_tool(form_id: str) -> str:
    """取得已知表單編號的完整內容。當頁面提到表單編號、但你不確定頁面本身的敘述是否已涵蓋完整細節時使用——表單常包含頁面沒寫清楚的細節。"""
    f = _get_form_raw(form_id)
    if "error" in f:
        return f"get_form 失敗: {f['error']}"
    return f"get_form 取得表單 {f.get('form_title','')}\n\n{f.get('text','')}"


TOOLS = [search_texts, grep_texts_tool, get_page_tool, extract_links_tool, get_form_tool]

_llm = ChatOllama(
    model=OLLAMA_MODEL, base_url=OLLAMA_HOST,
    client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
)
_llm_with_tools = _llm.bind_tools(TOOLS)


# ── Nodes ────────────────────────────────────────────────────────────────

def rewrite_node(state: AgenticState) -> dict:
    turn = state.get("turn", 0) + 1
    messages = state.get("messages", [])
    basis = state["query"] if not messages else f"{state['query']}\n\n已知進度：\n{_render_messages(messages)}"
    rewritten = _rewrite_query(basis)
    prompt = (f"這一輪系統建議的搜尋方向：{rewritten}\n\n原始問題：{state['query']}\n\n"
              f"請判斷下一步該做什麼。")
    return {"turn": turn, "rewritten": rewritten, "messages": [HumanMessage(content=prompt)]}


def domain_router_node(state: AgenticState) -> dict:
    """Runs once, right after turn 1's rewrite_node (consumes its cleaned
    `rewritten` output, not the raw query -- see phase_g_clean_pipeline_design.md
    §A). No-ops whenever subdomain_hint is already truthy, which both
    preserves an explicit --subdomain CLI override and makes this node
    effectively execute only once despite sitting in the every-turn loop."""
    if state.get("subdomain_hint"):
        return {}

    q = state["rewritten"]
    hit = _layer1_match(q)
    if hit:
        msg = (f"候選subdomain（依「{q}」查詢得出）：\n"
               f"1. {hit}（{_SUBDOMAIN_DESC.get(hit, '尚無職掌描述')}）— 關鍵字精確比對命中，非FTS5票數")
        return {"subdomain_hint": hit, "messages": [HumanMessage(content=msg)]}

    candidates = layer2_candidates(q)
    if not candidates:
        msg = f"候選subdomain（依「{q}」查詢得出）：無（Layer1/Layer2皆未找到相關subdomain，將使用全域搜尋）"
        return {"subdomain_hint": None, "messages": [HumanMessage(content=msg)]}

    lines = [f"候選subdomain（依票數排序，依「{q}」查詢得出）："]
    for i, (sub, count) in enumerate(candidates[:5], start=1):
        lines.append(f"{i}. {sub}（{_SUBDOMAIN_DESC.get(sub, '尚無職掌描述')}）— {count}筆命中")
    if is_ambiguous(q):
        top, second = candidates[0][1], candidates[1][1]
        lines.append(f"判斷：候選接近（{second}/{top}≈{second/top:.2f}），建議優先查{candidates[0][0]}，"
                     f"若結果不理想可考慮{candidates[1][0]}")
    else:
        lines.append(f"判斷：{candidates[0][0]}明確領先，非模糊案例")
    return {"subdomain_hint": candidates[0][0], "messages": [HumanMessage(content="\n".join(lines))]}


def agent_node(state: AgenticState) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=_AGENT_SYSTEM)] + messages
    resp = _llm_with_tools.invoke(messages)
    result: dict = {"messages": [resp]}
    if not resp.tool_calls:
        result["answer"] = resp.content
    return result


def _after_agent(state: AgenticState) -> str:
    # 2026-08-28: _MAX_TURNS hard cap deliberately removed for this experiment
    # -- let the agent decide for itself when it has enough, per the session's
    # core research question (don't cap turns deterministically to avoid
    # trusting the LLM's own stopping judgment). doom-loop detection (below,
    # in _after_tools) is the ONLY remaining safety net, and it only catches
    # exact-repeat tool_calls -- it will NOT catch an agent that keeps trying
    # genuinely different search terms without ever concluding (the "在職生
    # 復學" failure mode from v1 testing was exactly this shape, and would
    # NOT have been caught by doom-loop alone). If this experiment shows real
    # runaway cost, a generous risk-only ceiling (not a correctness cap) is
    # the fallback -- not reintroducing _MAX_TURNS=8 as-is.
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "end"
    return "tools"


def _after_tools(state: AgenticState) -> str:
    # doom-loop: count consecutive AIMessages with identical tool_calls signature.
    # This is now the ONLY loop-termination safety net -- see _after_agent.
    ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    if len(ai_msgs) >= _MAX_STUCK:
        sigs = [tuple(sorted((tc["name"], json.dumps(tc["args"], sort_keys=True)) for tc in m.tool_calls))
                for m in ai_msgs[-_MAX_STUCK:]]
        if len(set(sigs)) == 1:
            return "end"
    return "rewrite"


# ── Graph assembly ───────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgenticState)
    g.add_node("rewrite_node", rewrite_node)
    g.add_node("domain_router_node", domain_router_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tools", ToolNode(TOOLS))

    g.add_edge(START, "rewrite_node")
    g.add_edge("rewrite_node", "domain_router_node")
    g.add_edge("domain_router_node", "agent_node")
    g.add_conditional_edges("agent_node", _after_agent, {"tools": "tools", "end": END})
    g.add_conditional_edges("tools", _after_tools, {"rewrite": "rewrite_node", "end": END})

    return g.compile()


def run(query: str, subdomain_hint: str | None = None, stream: bool = False) -> dict:
    graph = build_graph()
    initial: AgenticState = {
        "query": query, "subdomain_hint": subdomain_hint, "turn": 0, "rewritten": "",
        "stuck_turns": 0, "messages": [], "answer": None,
    }

    if stream:
        final_state = dict(initial)
        for update in graph.stream(initial, stream_mode="updates"):
            for node_name, delta in update.items():
                delta = delta or {}  # a node returning {} (e.g. domain_router_node's no-op turns) surfaces as None here
                print(f"[{node_name}] {list(delta.keys())}", file=sys.stderr)
                for k, v in delta.items():
                    if k == "messages":
                        final_state["messages"] = final_state.get("messages", []) + v
                    else:
                        final_state[k] = v
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
    print(_render_messages(result.get("messages", [])))
    print(f"\n最終答案 ({result.get('turn', 0)}輪):")
    print(result.get("answer"))
