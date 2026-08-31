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

import glob
import json
import operator
import os
import re
import sys
from pathlib import Path
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
    CRAWLER_OUTPUT,
)
from rag.domain_router import _layer1_match, layer2_candidates, is_ambiguous, _keyword_table
from rag.router import route as _classify_query, _keyword_route, QueryType

FTS_DB = "rag/fts_proto3.db"
_MAX_STUCK = 3  # doom-loop detection only -- no turn-count cap this round, see _after_agent

# Split on 逗號/句號 (full-width and half-width) -- clause boundaries plan_node's
# compound-query detection and multi_sub_query_node's splitting both reason
# about. Same pattern as production's rag/nodes/decomposition.py.
_CLAUSE_SPLIT_RE = re.compile(r'[，。,.]')

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


# ── State ─────────────────────────────────────────────────────────────────

class AgenticState(TypedDict):
    query: str
    subdomain_hint: str | None
    query_type: str | None
    turn: int
    rewritten: str
    stuck_turns: int
    messages: Annotated[list, add_messages]
    answer: str | None
    self_eval_note: str | None


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

_FORM_JUDGE_PROMPT = """你要判斷回答這個問題需不需要抓取特定表單的全文，以及該抓哪一份。

使用者問題：{query}

目前已有的內容（可能是已抓取的頁面/表單全文，也可能還沒有任何內容）：
{context}

已知的表單清單（含表單編號、標題、用途說明、承辦單位——這份清單本來就存在，不是搜尋結果）：
{forms}

請判斷：這份表單清單裡，有沒有表單是回答這題需要抓取全文查看細節的？頁面敘述常常只是概略帶過，實際細節（蓋章站點、費用標準、資格條件等）常常只寫在表單裡；但也可能已有內容已經足夠，不需要任何表單。

只回傳需要抓取的表單編號，用逗號分隔；如果都不需要，回傳「無」。不要有其他文字。"""

_OFFICE_JUDGE_PROMPT = """以下文字裡可能提到了一些政大的辦公室/單位，但提到的名稱常常是簡稱或口語說法，不一定是正式全名（例如文字裡寫「住宿組」，正式全名可能是「住宿輔導組」；文字裡寫「圖書館」，正式全名可能是「圖書館(各組聯絡資訊)」）。

文字內容：
{text}

已知的政大辦公室/單位清單（正式名稱，格式為 名稱（所屬subdomain codebase）——這份清單本來就存在，不是搜尋結果）：
{catalog}

請判斷：上面清單裡，哪些辦公室有在文字內容中被提到（用簡稱、口語說法、或全名都算，只要語意上是指同一個單位就算）？只回傳清單裡列出的正式名稱，用逗號分隔；如果都沒提到，回傳「無」。不要自己發明清單以外的名稱，不要有其他文字。"""

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

_SYNTHESIS_PROMPT = """你是政大學務問答系統的最終答案撰寫者，讀取以下對話歷史（已找到的頁面內容、表單全文、辦公室聯絡資訊），撰寫一則完整、有根據的答案。

原始問題：{query}

對話歷史（含已抓取的頁面/表單全文、辦公室聯絡資訊全文）：
{history}

撰寫規則：
1. 直接寫出答案本身，不要描述你打算做什麼；具體引用找到的內容（流程步驟、費用、期限、表單連結等）。
2. 如果對話歷史裡有【辦公室聯絡資訊】區塊：這是查到的完整名單（每個辦公室可能各有十幾位承辦人），「不是要求全部列出」指的是**同一個辦公室內部**不用把整份名冊都塞進答案，不是指可以跳過整個辦公室。**這個區塊裡列出的每一個辦公室，只要有回傳承辦人資料，都必須在答案對應的段落/站點附上聯絡人——這是完整性要求，不是「挑幾個看起來最相關的」篩選**（跟蓋章站點清單「全部都要列出」是同一個道理，缺一個辦公室的聯絡人跟漏掉一個蓋章站點是同等級的錯誤）。若某個辦公室在名單裡但完全沒有回傳任何人員資料（例如只有辦公室名稱、無姓名），該站可以只標註樓層/分機、註明查無承辦人資訊，不要跳過整站不提。

   格式：對每個有資料的辦公室，從其名單裡挑選最相關的1-2位承辦人（同一辦公室內部才需要篩選，不是辦公室之間篩選）：
   `姓名（職責）分機XXXXX——選擇原因：[一句話，例如「負責最終核准」「第一線受理窗口」「這是唯一列出分機的聯絡人」]`
   選擇原因必須具體對應這個人的職責欄位或這題的辦理流程，不能只寫「相關人員」這種空話。如果內容顯示某個步驟需要多層審核（例如先由承辦人受理，再經組長、單位主管逐層簽核），把實際涉及的每一層都列出來、每層各自附上選擇原因，不要只挑一位；如果只是單純的一般承辦窗口，列1-2位最相關的即可。**承辦人姓名是必填項目，不能只寫辦公室名稱、樓層、分機。**
3. 如果對話歷史顯示已經搜尋多次仍找不到某個細節，誠實說明未查到，不要杜撰。
4. 回答最後附上來源URL。"""

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


def _render_full_messages(messages: list) -> str:
    """Full-text renderer for synthesis_node -- unlike _render_messages()
    below (deliberately compact one-liners, used by rewrite_node to decide
    what to search next), synthesis needs the actual page/form/contact
    content verbatim, not a summary; truncating here would silently drop
    the exact names/numbers synthesis is supposed to cite. Includes
    ToolMessage (get_page/get_form/grep results) and HumanMessage
    (resource_node/contact_node's injected content, domain_router_node's
    candidate list, rewrite_node's own prompts) -- everything except
    SystemMessage/AIMessage, since the latter is just the agent's own
    tool-call decisions, not retrieved content."""
    parts = [str(m.content) for m in messages if isinstance(m, (ToolMessage, HumanMessage))]
    return "\n\n---\n\n".join(parts) if parts else "（尚無內容）"


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

def plan_node(state: AgenticState) -> dict:
    """D1 (docs/phase_g_clean_pipeline_design.md §M): classification layer
    reusing router.py's existing 2-layer keyword->LLM classifier -- not a
    new classifier, the exact one already validated in production. Per D2,
    PROCEDURE and KNOWLEDGE are NOT different graph paths; the label is
    recorded for logging/self_eval only and both fall through to the same
    default branch (_after_plan). CONTACT/RESOURCE get a genuine direct
    route straight to contact_node/resource_node, skipping rewrite_node/
    domain_router_node/search_texts/agent_node entirely. This works without
    those two nodes needing any special-cased "how was I entered" logic:
    resource_node/contact_node each read whatever context is currently in
    state["messages"] (empty here, since the main loop never ran) and reason
    over that -- an empty context is just an input to their existing LLM
    judgment (resource_node) / deterministic scan (contact_node), not a
    separate code path (2026-08-30 redesign).

    D14 (2026-08-30 addition): checks for a compound query FIRST, before
    the single-label classifier runs at all -- reusing router.py's own
    _keyword_route() per clause (split on 逗號/句號), same "keyword first,
    zero LLM cost" layer production's query_decomposition_node already
    validated. A query is compound only if 2+ clauses resolve to DIFFERENT
    QueryTypes; a single-topic query with a mere pause is not (mirrors
    production's exact reasoning, see rag/nodes/decomposition.py). This
    only catches CROSS-type compound queries ("休學，圖書館電話多少") --
    same-TYPE compound queries ("休學和退學的差別") are a structural blind
    spot here, caught instead by self_eval_node's post-hoc full-query
    check (D7-D9's two-layer-defense table)."""
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(state["query"]) if c.strip()]
    if len(clauses) >= 2:
        types_found = {t for t in (_keyword_route(c) for c in clauses) if t is not None}
        if len(types_found) >= 2:
            return {"query_type": "compound"}
    result = _classify_query(state["query"], use_llm_fallback=True)
    return {"query_type": result.query_type.value}


def _after_plan(state: AgenticState) -> str:
    qt = state.get("query_type")
    if qt == "compound":
        return "compound"
    if qt == "contact":
        return "contact"
    if qt == "resource":
        return "resource"
    return "knowledge"  # procedure + knowledge share one path, per D2


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
    `rewritten` output, not the raw query). No-ops whenever subdomain_hint
    is already truthy, which both preserves an explicit --subdomain CLI
    override and makes this node effectively execute only once despite
    sitting in the every-turn loop. Deliberately carries no query_type
    check of its own (docs/phase_g_clean_pipeline_design.md §M D1,
    2026-08-30 clarification) -- it currently only gets reached via the
    knowledge/procedure path because Plan_node routes CONTACT/RESOURCE
    elsewhere, not because this node restricts itself to those types. That
    distinction matters once self_eval_node's 情況B exists: it may route an
    arbitrary query back into the main loop regardless of its original
    Plan_node classification, and this node must keep working for it."""
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
    """Tool-selection only (2026-08-30 split) -- this used to also double as
    the final-answer generator (via resp.content) whenever it decided no
    more tools were needed, sharing _AGENT_SYSTEM (a tool-selection-oriented
    prompt) for both concerns. That meant synthesis-specific rules (mandatory
    contact names, how to pick from a large roster with a stated reason)
    had nowhere to live, since _AGENT_SYSTEM was never the right prompt for
    them. Now: when _after_agent sees no tool_calls, it routes to
    synthesis_node instead of END -- this response's .content is discarded
    (still generated as a side effect of the bind_tools() call, just never
    read; not worth a separate tool-choice-only call shape for this)."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=_AGENT_SYSTEM)] + messages
    resp = _llm_with_tools.invoke(messages)
    return {"messages": [resp]}


def synthesis_node(state: AgenticState) -> dict:
    """Dedicated final-answer generation (2026-08-30 addition, adapted from
    production's rag/nodes/synthesis.py's _SYNTHESIS_PROMPT) -- makes ONE
    call with a synthesis-specific prompt over the full message history
    (_render_full_messages(), not the compact _render_messages() rewrite_node
    uses), so contact-name/roster-selection rules have a dedicated home
    instead of being crammed into _AGENT_SYSTEM alongside tool-selection
    instructions."""
    history = _render_full_messages(state["messages"])
    prompt = _SYNTHESIS_PROMPT.format(query=state["query"], history=history)
    answer = simple_chat(messages=[{"role": "user", "content": prompt}])
    return {"answer": answer}


_SELF_EVAL_MAX_TURN = 20  # generous risk-only ceiling (not a correctness cap,
# same philosophy as _MAX_STUCK/removing _MAX_TURNS elsewhere in this file) --
# self_eval's retry loops (both 情況A and 情況B) route back through
# rewrite_node, which increments state["turn"] every pass, so this reuses
# that existing counter instead of inventing a new one just for self_eval.


def self_eval_node(state: AgenticState) -> dict:
    """D7-D9 (docs/phase_g_clean_pipeline_design.md §M, 2026-08-30 design,
    first implementation): runs once after every synthesis_node call, not
    gated behind a narrow pre-check -- the original D7 draft only fired when
    "procedure classified AND resource/contact never triggered," but that
    signal structurally can't catch D8's own worked example (a cross-type
    compound query where the procedure half DID trigger resource/contact
    normally, silently dropping the other half). One LLM call
    (_SELF_EVAL_PROMPT) judges against the FULL original state["query"]
    (not a rewritten sub-query) -- this is what lets it catch same-type
    compound queries that #8 (Plan_node's keyword-type compound detection,
    still unimplemented) structurally cannot (see design doc's two-layer-
    defense table). Returns one of: pass (self_eval_note=None), 情況A
    (content gap -- injects a concrete reminder message and loops back via
    rewrite_node), 情況B (classification may be wrong -- clears query_type
    so _after_self_eval routes back to plan_node for reclassification).

    Honest caveat carried over from the design doc: this is still
    "message hint -> LLM decides whether to act," the same mechanism shape
    already shown unreliable for _AGENT_SYSTEM's timing clauses -- moving it
    to a dedicated post-hoc step doesn't guarantee it works better, only
    that it's now testable in isolation. Not validated beyond compiling and
    a first smoke-test run at implementation time."""
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


def _after_self_eval(state: AgenticState) -> str:
    if not state.get("self_eval_note"):
        return "end"
    if state.get("turn", 0) >= _SELF_EVAL_MAX_TURN:
        return "end"  # risk-only ceiling, not a correctness judgment
    if state.get("query_type") is None:
        return "plan"  # 情況B
    return "rewrite"  # 情況A


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


_FORM_MARKER_RE = re.compile(r"\[偵測到表單編號: ([^\]]+)\]")
_OFFICE_MARKER_RE = re.compile(r"\[偵測到辦公室: ([^\]]+)\]")


def _detect_offices(text: str) -> list[str]:
    """LLM-judged semantic match against the full ~433-entry office catalog
    (domain_router.py's _keyword_table()) -- replaces a pure substring scan
    (2026-08-30 redesign). A form/page refers to offices colloquially
    ("住宿組") while the catalog carries official full names ("住宿輔導
    組") -- these share no contiguous substring, so no fixed-string match
    would ever catch it. A first fix attempt hardcoded a small alias table
    (~9 entries) -- rejected (2026-08-30, user correction): that's the same
    whack-a-mole pattern this project has rejected before (_STRIP_RE, 範例
    3/4), covering only offices someone happened to notice were broken,
    needing endless future patching for every other short/full-name
    mismatch. "Does this text refer to the same office as this catalog
    entry" is a semantic judgment, not an objective check -- LLM territory,
    same reasoning as _judge_forms(). Catalog is NOT subdomain-scoped: a
    single form/page can reference offices across multiple subdomains (an
    aca form's station list can include an osa office), so narrowing to one
    subdomain here would silently drop cross-subdomain office mentions."""
    catalog = _keyword_table()
    if not catalog:
        return []
    catalog_str = "\n".join(f"- {name}（{sub}）" for name, sub in catalog)
    prompt = _OFFICE_JUDGE_PROMPT.format(text=text, catalog=catalog_str)
    raw = simple_chat(messages=[{"role": "user", "content": prompt}], max_tokens=300).strip()
    valid_names = {name for name, _sub in catalog}
    candidates = [n.strip() for n in re.split(r"[,，]", raw)]
    return [n for n in dict.fromkeys(candidates) if n in valid_names]


def _list_forms_metadata(subdomain_hint: str | None) -> list[dict]:
    """Reads supplementary_map.json's own records directly -- form_id/
    form_title/form_description/form_unit already exist there for every
    form (confirmed non-empty, real data: e.g. aca's QP-K13-00-03 carries
    "包含中正圖書館...之場地借用申請" as form_description). This is a
    bounded lookup over already-indexed metadata, NOT a search capability --
    resource_node must not have one (2026-08-30 correction: it "早就有能力
    判斷休學表單是什麼，因為他都是讀取extracted_supplementary.jsonl設計
    的" -- the data needed to judge relevance was already sitting there,
    no FTS5/page search required). Scoped to subdomain_hint's own file when
    given (~20-40 forms per subdomain); globs all subdomains only when no
    hint exists yet (still just reading existing files, not searching)."""
    pattern = (f"{CRAWLER_OUTPUT}/{subdomain_hint}/supplementary_map.json"
               if subdomain_hint else f"{CRAWLER_OUTPUT}/*/supplementary_map.json")
    forms: list[dict] = []
    for path in glob.glob(pattern):
        try:
            for r in json.loads(Path(path).read_text()):
                fid = r.get("form_id")
                if fid:
                    forms.append({
                        "form_id": fid,
                        "title": r.get("form_title", ""),
                        "description": r.get("form_description", ""),
                        "unit": r.get("form_unit", ""),
                    })
        except (json.JSONDecodeError, OSError):
            pass
    return forms


def _judge_forms(query: str, context_text: str, subdomain_hint: str | None) -> list[str]:
    """Single unified judgment step, used identically no matter how
    resource_node was entered (2026-08-30 redesign, replacing what used to
    be two separate code paths -- a regex marker-scan branch and a full
    anchor-page-search branch -- with one LLM call that always sees the same
    two inputs: whatever context_text is currently available (may be "" --
    the judge handles that naturally in its own prompt, no branching needed
    in the caller) and the full known-forms metadata list
    (_list_forms_metadata() -- a bounded lookup, never a search). A raw
    RESOURCE query like "休學申請表在哪裡下載" never contains a form_id as
    a literal substring the way a CONTACT query contains an office name, so
    unlike _detect_offices() this can't be a plain substring scan -- but it
    also doesn't need search, because the candidate list (every form's
    title/description/unit) is already sitting in supplementary_map.json."""
    scope = subdomain_hint or _layer1_match(query)
    forms = _list_forms_metadata(scope)
    if not forms:
        return []
    forms_str = "\n".join(f"- {f['form_id']}：{f['title']}（{f['unit']}）{f['description']}" for f in forms)
    prompt = _FORM_JUDGE_PROMPT.format(
        query=query,
        context=context_text if context_text else "（尚無內容）",
        forms=forms_str,
    )
    raw = simple_chat(messages=[{"role": "user", "content": prompt}], max_tokens=150).strip()
    valid_ids = {f["form_id"] for f in forms}
    return [fid for fid in valid_ids if fid in raw]


def resource_node(state: AgenticState) -> dict:
    """Deterministically routed here by either _after_tools (marker found on
    a ToolMessage) or _after_plan (Plan_node classified the query RESOURCE
    directly) -- both are the objective gate (§M D3/D6); this node itself no
    longer re-derives "how did I get here" via marker regex vs empty-
    messages branching (2026-08-30 redesign -- that branching was doing the
    same underlying judgment three different ways in code instead of once
    via a prompt). One code path: read whatever context_text is currently
    available (the last message's content, or "" if this is the direct
    Plan_node route), hand it to _judge_forms() along with the query --
    that single LLM call decides which form(s), if any, are worth fetching,
    using the full known-forms metadata list rather than a page search (see
    _list_forms_metadata()/_judge_forms() docstrings). Fetches each judged-
    relevant form's full content via the existing get_form_tool logic (no
    duplication), then scans the fetched content for office names (§M D4)
    so _after_resource can decide whether to chain into contact_node. Hands
    back a HumanMessage -- same pattern as domain_router_node, since this
    isn't a response to any AIMessage tool_call and so can't legitimately be
    a ToolMessage."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    context_text = str(last.content) if last is not None else ""
    relevant_ids = _judge_forms(state["query"], context_text, state.get("subdomain_hint"))
    if not relevant_ids:
        return {}
    fetched_ids = set(relevant_ids)
    results = [get_form_tool.invoke({"form_id": fid}) for fid in relevant_ids]
    combined = "\n\n".join(results)

    # Cross-reference pass: a fetched form's own text can reference another
    # form_id never mentioned before -- re-run the SAME judge on the now-
    # larger context_text (original context + newly fetched form text),
    # excluding ids already fetched. Bounded to a single extra pass (not
    # recursive) to keep this deterministic node's cost bounded; no case
    # observed yet needing a deeper chain.
    more_ids = [fid for fid in _judge_forms(state["query"], combined, state.get("subdomain_hint"))
                if fid not in fetched_ids]
    if more_ids:
        combined += "\n\n" + "\n\n".join(get_form_tool.invoke({"form_id": fid}) for fid in more_ids)

    combined = "[表單全文，系統偵測到表單編號後自動抓取，請直接引用其中的流程/站點/費用等細節]\n\n" + combined
    offices = _detect_offices(combined)
    if offices:
        combined += f"\n\n[偵測到辦公室: {', '.join(offices)}]"
    return {"messages": [HumanMessage(content=combined)]}


def contact_node(state: AgenticState) -> dict:
    """Deterministically routed here by _after_tools, _after_resource, or
    _after_plan (§M D3/D4/D6 -- all three are objective gates: a marker on a
    ToolMessage/HumanMessage, or Plan_node classifying the query CONTACT
    directly). One code path regardless of which gate routed here
    (2026-08-30 redesign, same reasoning as resource_node): read whatever
    context_text is available (last message's content, or the raw query
    itself if messages is still empty), run the existing deterministic
    _detect_offices() substring scan against it -- this works identically
    whether context_text is a fetched form's full text or the bare query
    "出納組電話幾號", since both are just text to scan. No separate "which
    mode am I in" branching needed.

    Wraps the existing OfficeLookupSkill batch lookup -- a genuinely new
    capability (no prior tool did batch contact lookup), unlike
    resource_node which just re-routes get_form_tool's existing logic.

    Deliberately does NOT run offices through an LLM relevance filter the
    way resource_node's _judge_forms() filters form_ids -- offices detected
    from a form's own station list are a completeness requirement (Step 9,
    phase_g_clean_pipeline_design.md: ALL stations must appear, not a
    filterable relevance list), and a pre-fetch filter risks permanently
    losing a required station's contact info on a bad judgment call, with
    no fallback (unlike a filtered-out form, whose narrative text usually
    still covers the basics). A direct CONTACT query naming N offices has
    no such tension either -- naming them means the user wants info on all
    N. Fetching all detected offices is cheap (single batch lookup); any
    filtering for what actually surfaces in the answer belongs in
    synthesis, after the data exists (validated by §M D10: the agent
    correctly omitted a full staff roster from a procedure answer while
    keeping station+location, i.e. filter-after-fetch, not filter-before)."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    context_text = str(last.content) if last is not None else state["query"]
    offices = _detect_offices(context_text)
    if not offices:
        return {}
    from rag.skills.office_lookup_skill import OfficeLookupSkill
    skill = OfficeLookupSkill()
    result = skill.run(offices)
    header = (
        f"[辦公室聯絡資訊，以下是內容中提及的全部 {len(offices)} 個辦公室（{'、'.join(offices)}），"
        f"未經相關性篩選——這是完整清單，不代表每一個都跟這題直接相關。"
        f"是否每個都要寫進最終答案由你根據問題判斷，不要假設清單已經先篩過。]"
    )
    context = header + "\n\n" + skill.format_context(result)
    return {"messages": [HumanMessage(content=context)]}


def _after_tools(state: AgenticState) -> str:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and _FORM_MARKER_RE.search(str(last.content)):
        return "resource"
    if isinstance(last, ToolMessage) and _OFFICE_MARKER_RE.search(str(last.content)):
        # Direct page->contact path for pure CONTACT-type queries with no
        # form involved at all (e.g. "出納組電話幾號") -- get_page_tool sets
        # this marker itself when it found no form_ids to route through
        # resource_node instead. When a form IS present, this branch is
        # never reached (get_page_tool's else-branch skips detection in
        # that case) -- office detection stays exclusively on resource_node's
        # freshly-fetched form text, per §M D4's established sequencing.
        return "contact"
    # doom-loop: count consecutive AIMessages with identical tool_calls signature.
    # This is now the ONLY loop-termination safety net -- see _after_agent.
    ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    if len(ai_msgs) >= _MAX_STUCK:
        sigs = [tuple(sorted((tc["name"], json.dumps(tc["args"], sort_keys=True)) for tc in m.tool_calls))
                for m in ai_msgs[-_MAX_STUCK:]]
        if len(set(sigs)) == 1:
            return "end"
    return "rewrite"


def _after_resource(state: AgenticState) -> str:
    """§M D4: resource -> contact is sequential, not parallel -- contact's
    trigger signal only exists once resource_node has actually fetched the
    form content to scan. Guards against empty messages: reachable via
    Plan_node's direct RESOURCE route (D1), where resource_node may judge
    that no form is needed and return {}, leaving state["messages"] exactly
    as empty as when this node started."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if last is not None and isinstance(last, HumanMessage) and _OFFICE_MARKER_RE.search(str(last.content)):
        return "contact"
    return "rewrite"


# ── Compound query handling (§M D14, v1 sequential) ─────────────────────

_loop_graph_cache = None


def _build_loop_graph():
    """Compiles rewrite_node<->agent_node<->tools (plus resource_node/
    contact_node's D3/D4 marker chain) as an INDEPENDENT StateGraph, separate
    from build_graph()'s outer compiled graph -- reused by
    multi_sub_query_node's nested .invoke() calls (D14 Phase 1b) so each
    sub-query gets its own fully isolated turn/rewritten/subdomain_hint/
    query_type, with zero risk of the concurrent-write InvalidUpdateError
    Phase 1 proved happens when multiple executions share ONE graph run
    (scratchpad/spike_send.py). Reuses the exact same node functions the
    outer graph uses -- no logic duplicated, just a second, smaller
    assembly of them. Entry routing reuses _after_plan's own contact/
    resource/knowledge logic (a sub-query's query_type is already set by
    multi_sub_query_node before invoking this, so no plan_node/compound-
    detection step is needed here -- compound-within-compound is out of
    scope). Compiled once (module-level cache): compiling is cheap, but
    there's no reason to repeat it every sub-query call."""
    global _loop_graph_cache
    if _loop_graph_cache is not None:
        return _loop_graph_cache
    g = StateGraph(AgenticState)
    g.add_node("rewrite_node", rewrite_node)
    g.add_node("domain_router_node", domain_router_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("resource_node", resource_node)
    g.add_node("contact_node", contact_node)
    g.add_conditional_edges(START, _after_plan,
                             {"knowledge": "rewrite_node", "resource": "resource_node", "contact": "contact_node"})
    g.add_edge("rewrite_node", "domain_router_node")
    g.add_edge("domain_router_node", "agent_node")
    g.add_conditional_edges("agent_node", _after_agent, {"tools": "tools", "end": END})
    g.add_conditional_edges("tools", _after_tools, {"resource": "resource_node", "contact": "contact_node", "rewrite": "rewrite_node", "end": END})
    g.add_conditional_edges("resource_node", _after_resource, {"contact": "contact_node", "rewrite": "rewrite_node"})
    g.add_edge("contact_node", "rewrite_node")
    _loop_graph_cache = g.compile()
    return _loop_graph_cache


def multi_sub_query_node(state: AgenticState) -> dict:
    """v1 sequential implementation of #8 (§M D14) -- runs each detected
    sub-query through its own fully independent invocation of
    _build_loop_graph(), one after another. Deliberately NOT Send/parallel
    (2026-08-30 decision): sequential sidesteps the concurrent-write problem
    entirely (nothing shares a graph run, so there's nothing to conflict),
    matches production's own v1-then-Send-upgrade precedent
    (rag/nodes/decomposition.py), and Send's actual wall-clock parallelism
    benefit here is unverified -- upgrading to Send is an explicit follow-up
    (§M D14), not abandoned.

    Each sub-query's turn/rewritten/subdomain_hint/query_type live ONLY
    inside that one nested .invoke() call and are discarded when it returns
    -- only `messages` (which has add_messages as its reducer) crosses back,
    via plain list concatenation here since this is one node's single
    return, not a multi-branch merge.

    Failure isolation: each sub-query's .invoke() is wrapped in its own
    try/except (§M D14 Phase 4) -- one sub-query failing appends an honest
    failure note to messages instead of raising, which would otherwise abort
    the whole compound-query answer per Phase 1 Test 2's confirmed behavior
    (an uncaught exception propagates to the caller and kills the entire
    run, not just one branch).

    Known limitation (v1, accepted -- see D14): if self_eval_node retries
    this compound query, the WHOLE for-loop re-runs (all N sub-queries),
    not just the one that was actually deficient -- self_eval currently has
    no way to know which sub-query, if any, was the problem. Not solved
    this pass."""
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(state["query"]) if c.strip()]
    loop_graph = _build_loop_graph()
    all_messages: list = []
    for clause in clauses:
        qt = _keyword_route(clause) or QueryType.KNOWLEDGE
        try:
            result = loop_graph.invoke({
                "query": clause, "subdomain_hint": None, "query_type": qt.value,
                "turn": 0, "rewritten": "", "stuck_turns": 0, "messages": [],
                "answer": None, "self_eval_note": None,
            })
            all_messages.extend(result.get("messages", []))
        except Exception as e:
            all_messages.append(HumanMessage(content=f"[子問題「{clause}」查詢時發生錯誤，未能取得資訊：{e}]"))
    return {"messages": all_messages}


# ── Graph assembly ───────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgenticState)
    g.add_node("plan_node", plan_node)
    g.add_node("rewrite_node", rewrite_node)
    g.add_node("domain_router_node", domain_router_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("resource_node", resource_node)
    g.add_node("contact_node", contact_node)
    g.add_node("synthesis_node", synthesis_node)
    g.add_node("self_eval_node", self_eval_node)
    g.add_node("multi_sub_query_node", multi_sub_query_node)

    g.add_edge(START, "plan_node")
    g.add_conditional_edges("plan_node", _after_plan,
                             {"knowledge": "rewrite_node", "resource": "resource_node",
                              "contact": "contact_node", "compound": "multi_sub_query_node"})
    g.add_edge("multi_sub_query_node", "synthesis_node")
    g.add_edge("rewrite_node", "domain_router_node")
    g.add_edge("domain_router_node", "agent_node")
    g.add_conditional_edges("agent_node", _after_agent, {"tools": "tools", "end": "synthesis_node"})
    g.add_conditional_edges("tools", _after_tools, {"resource": "resource_node", "contact": "contact_node", "rewrite": "rewrite_node", "end": "synthesis_node"})
    g.add_conditional_edges("resource_node", _after_resource, {"contact": "contact_node", "rewrite": "rewrite_node"})
    g.add_edge("contact_node", "rewrite_node")
    g.add_edge("synthesis_node", "self_eval_node")
    g.add_conditional_edges("self_eval_node", _after_self_eval, {"end": END, "rewrite": "rewrite_node", "plan": "plan_node"})

    return g.compile()


def run(query: str, subdomain_hint: str | None = None, stream: bool = False) -> dict:
    graph = build_graph()
    initial: AgenticState = {
        "query": query, "subdomain_hint": subdomain_hint, "query_type": None, "turn": 0, "rewritten": "",
        "stuck_turns": 0, "messages": [], "answer": None, "self_eval_note": None,
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
