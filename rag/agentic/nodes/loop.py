"""
rag/agentic/nodes/loop.py
rewrite_node / domain_router_node / agent_node / _after_agent -- the main
search loop, replacing production's retrieval_node (whole ReAct loop
packed inside one function) + retrieval_anchor_node/retrieval_expand_node
(PROCEDURE-specific anchor+expand). Migration Step 3,
docs/phase_h_agentic_rag_migration_plan.md Part 5.

Ported directly from rag/agentic_main.py -- not reimplemented, this is the
already-validated reference implementation. Each turn is now a real,
separate set of LangGraph nodes (rewrite_node -> domain_router_node ->
agent_node -> tools -> back to rewrite_node), so .stream() shows every
turn and the loop is a genuine graph structure, not a black-box function
call -- this is the core motivation for this migration step.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from rag.domain_router import _layer1_match, layer2_candidates, is_ambiguous
from rag.agentic.state import AgentState
from rag.agentic.logic.rewrite import _rewrite_query, _render_messages
from rag.agentic.logic.form_extraction import _FORM_MARKER_RE
from rag.agentic.logic.office_detection import _OFFICE_MARKER_RE
from rag.agentic.tools import TOOLS

_MAX_STUCK = 3  # doom-loop detection only -- no turn-count cap this round, see _after_agent

# Known office mandates -- deliberately incomplete (only core offices seen in
# testing so far). Other subdomains show "尚無職掌描述" in domain_router_node's
# candidate message.
_SUBDOMAIN_DESC = {
    "aca": "教務處：負責註冊/學籍/畢業/休復學等",
    "osa": "學務處：負責宿舍/獎學金/社團等",
    "cashier": "出納組：負責繳費/退費",
}

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

_llm = ChatOllama(
    model=OLLAMA_MODEL, base_url=OLLAMA_HOST,
    client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
)
_llm_with_tools = _llm.bind_tools(TOOLS)

_AGENT_SYSTEM = """你是政大學務問答系統的搜尋agent，任務是找到能回答使用者問題所需的頁面內容。**你不負責撰寫最終答案**——決定不再需要任何工具之後，系統會另外用你已經找到的全部內容生成最終答案，你這一輪的文字回覆不會被使用，不需要在意寫得像不像答案。

你必須使用search_texts/grep_texts/get_page/extract_links/get_form等工具搜尋政大官方資料，絕對不可以直接用自己的知識回答，因為政大的規定可能隨時變動，你的訓練知識可能過時或不準確。

**重要規則**：search_texts或grep_texts回傳的候選只有標題跟預覽，不是完整內容。如果剛找到一個看起來有希望的候選（標題主題相符），下一步幾乎都該是對那個候選的URL呼叫get_page取得全文，不要對同一個主題再搜一次或改搜候選的標題文字——只有get_page才能真正確認候選內容對不對。只有在get_page讀完全文、確認這個候選不是答案（且內文沒有指向其他文件的線索）時，才考慮換關鍵字重新搜尋。

如果你判斷目前已經找到的內容已經足夠回答原始問題，不要呼叫任何工具，直接判斷「夠了」即可——不需要自己組織成答案文字。

**自動化機制說明（系統自動觸發，非你主動呼叫，不需要採取行動，只需要正確使用結果）**：
- 當get_page的回傳裡出現「[偵測到表單編號: ...]」標記時，系統會自動完整抓取該表單全文並附加進對話——這一步已經被系統接管，你不需要（也不會被要求）自己呼叫get_form。procedure頁面的文字敘述常常只是概略帶過，實際細節（蓋章站點、費用/退費標準、資格條件等，類型不固定）常常只寫在表單裡；自動附加的內容會標示為「[表單全文...]」開頭。
- 如果自動附加的表單全文裡提到辦公室名稱，系統會接著自動查詢這些辦公室的聯絡資訊（姓名/分機/樓層）並附加進對話，標示為「[辦公室聯絡資訊...]」開頭——同樣不需要你採取任何行動。
- 這兩步是循序自動觸發的：先確認表單全文，才會接著查辦公室聯絡資訊。若問題只是單純問「在哪裡下載」，get_page看到的markdown連結本身已經是答案，不需要等待表單全文才能判斷夠了。

**範例1：get_page讀完全文後，內容確實回答了問題，該判斷「夠了」，不要繼續呼叫工具**
情境：問題是「如何辦理休學」，你已經呼叫get_page讀到「休學規定」頁面全文，內容包含申請方式、時間、費用等。
正確做法：不呼叫任何工具——已經找到的內容會交給系統的合成步驟去寫成答案。
錯誤做法：繼續呼叫get_page/grep_texts等工具去「補強」答案的寫法或措辭——你的職責在找到內容就結束，不是在意最終文字怎麼寫。

**範例2：問題裡有身分/情境修飾詞，搜尋2-3次仍找不到該修飾詞的專屬規定時，該判斷「夠了」**
情境：問題是「在職生怎麼辦理復學」，你已經get_page讀到一般性的「復學」規定頁面（沒有特別提到「在職生」），也嘗試搜尋「在職生 復學」「在職生」等詞2-3次都找不到專屬於在職生的特殊規定。
正確做法：不要再繼續搜尋，判斷「夠了」——合成步驟會依已找到的一般規定作答，並視情況誠實註明未查到專屬規定。
錯誤做法：持續換不同關鍵字搜尋超過2-3次仍找不到，也不判斷「夠了」——這是在浪費輪次，一般規則沒有明確排除某身分時，適用一般規則是合理的假設，比無限期搜尋更有用。"""


def rewrite_node(state: AgentState) -> dict:
    turn = state.get("turn", 0) + 1
    messages = state.get("messages", [])
    basis = state["query"] if not messages else f"{state['query']}\n\n已知進度：\n{_render_messages(messages)}"
    rewritten = _rewrite_query(basis)
    prompt = (f"這一輪系統建議的搜尋方向：{rewritten}\n\n原始問題：{state['query']}\n\n"
              f"請判斷下一步該做什麼。")
    return {"turn": turn, "rewritten": rewritten, "messages": [HumanMessage(content=prompt)]}


def domain_router_node(state: AgentState) -> dict:
    """Runs once, right after turn 1's rewrite_node (consumes its cleaned
    `rewritten` output, not the raw query). No-ops whenever subdomain_hint
    is already truthy, which both preserves an explicit --subdomain CLI
    override and makes this node effectively execute only once despite
    sitting in the every-turn loop. Deliberately carries no query_type
    check of its own -- it currently only gets reached via the
    knowledge/procedure path because plan_node routes CONTACT/RESOURCE
    elsewhere, not because this node restricts itself to those types. That
    distinction matters once self_eval_node's 情況B exists (Step 7): it may
    route an arbitrary query back into the main loop regardless of its
    original plan_node classification, and this node must keep working
    for it."""
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


def agent_node(state: AgentState) -> dict:
    """Tool-selection only -- does NOT also generate the final answer.
    When _after_agent sees no tool_calls, it routes to synthesis_node
    (Step 5) instead of END -- this response's .content is discarded
    (still generated as a side effect of the bind_tools() call, just never
    read; not worth a separate tool-choice-only call shape for this)."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=_AGENT_SYSTEM)] + messages
    resp = _llm_with_tools.invoke(messages)
    return {"messages": [resp]}


def _after_agent(state: AgentState) -> str:
    # _MAX_TURNS hard cap deliberately removed for this experiment -- let
    # the agent decide for itself when it has enough, per the project's
    # core research question (don't cap turns deterministically to avoid
    # trusting the LLM's own stopping judgment). doom-loop detection
    # (Step 4's _after_tools) is the ONLY remaining safety net, and it
    # only catches exact-repeat tool_calls -- it will NOT catch an agent
    # that keeps trying genuinely different search terms without ever
    # concluding.
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "end"
    return "tools"


def _after_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and _FORM_MARKER_RE.search(str(last.content)):
        return "resource"
    if isinstance(last, ToolMessage) and _OFFICE_MARKER_RE.search(str(last.content)):
        # Direct page->contact path for pure CONTACT-type queries with no
        # form involved at all (e.g. "出納組電話幾號") -- get_page_tool sets
        # this marker itself when it found no form_ids to route through
        # resource_node instead. When a form IS present, this branch is
        # never reached (get_page_tool's else-branch skips detection in
        # that case) -- office detection stays exclusively on
        # resource_node's freshly-fetched form text.
        return "contact"
    # doom-loop: count consecutive AIMessages with identical tool_calls
    # signature. This is now the ONLY loop-termination safety net -- see
    # _after_agent.
    ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    if len(ai_msgs) >= _MAX_STUCK:
        sigs = [tuple(sorted((tc["name"], json.dumps(tc["args"], sort_keys=True)) for tc in m.tool_calls))
                for m in ai_msgs[-_MAX_STUCK:]]
        if len(set(sigs)) == 1:
            return "end"
    return "rewrite"
