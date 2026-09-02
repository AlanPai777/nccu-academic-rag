"""
rag/agentic/nodes/synthesis.py
synthesis_node -- dedicated final-answer generation, replacing production's
rag/nodes/synthesis.py (which branches by query_type: RESOURCE/KNOWLEDGE
pass-through, only PROCEDURE/CONTACT actually call an LLM) and the answer-
writing previously scattered across retrieval_node/resource_node.
Migration Step 5, docs/phase_h_agentic_rag_migration_plan.md Part 5.

Ported directly from rag/agentic_main.py. All paths now converge on this
ONE node -- agent_node's own .content (from the tool-selection call) is
always discarded, never used as an answer.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag.agentic.nodes.loop import _llm
from rag.agentic.state import AgentState

_SYNTHESIS_PROMPT = """你是政大學務問答系統的最終答案撰寫者，讀取以下對話歷史（已找到的頁面內容、表單全文、辦公室聯絡資訊），撰寫一則完整、有根據的答案。

原始問題：{query}

對話歷史（含已抓取的頁面/表單全文、辦公室聯絡資訊全文）：
{history}

撰寫規則：
1. 直接寫出答案本身，不要描述你打算做什麼；具體引用找到的內容（流程步驟、費用、期限、表單連結等）。
2. 如果問題是「如何辦理OO」這類流程類問題，答案要寫成可以照著操作的流程手冊，不是概略描述。用「第一步、第二步、...」明確分步驟，每一步都要具體回答：
   (a) 這一步要準備/填寫哪一份表單（附下載連結）；
   (b) 這份表單本身有沒有需要特別注意的填寫規則或但書——例如需要委託書及受託人證件、需要先繳費單據正本、依申請時間點金額或流程不同、需要先由某處室核章才能到下一個處室辦理等（【表單其他決策/資訊項目，答案應涵蓋】區塊如果有出現，裡面常常就是這類細節，尤其是「先至OO核章、再至OO辦理/繳費」這種跨處室的順序，這是具體操作步驟，必須寫進對應的那一步，不能省略成一句籠統帶過）；
   (c) 辦完這一步要拿著東西去哪個處室、地點（樓層/門牌，若有查到）；
   (d) 這個處室的負責人姓名與分機（若【辦公室聯絡資訊】有查到，依規則3列出）。
   不要只寫「至OO處室辦理」這種空泛描述——要具體到「填寫XX表 → 持表至OO（地點）辦理/由某某人核章 → 再至XX處室繳費」這種可執行的操作順序。單純查資訊/查電話這類非流程類問題不受此規則限制，維持簡潔回答即可。
3. 如果對話歷史裡有【辦公室聯絡資訊】區塊：這是查到的完整名單（每個辦公室可能各有十幾位承辦人），「不是要求全部列出」指的是**同一個辦公室內部**不用把整份名冊都塞進答案，不是指可以跳過整個辦公室。**這個區塊裡列出的每一個辦公室，只要有回傳承辦人資料，都必須在答案對應的段落/站點附上聯絡人——這是完整性要求，不是「挑幾個看起來最相關的」篩選**（跟蓋章站點清單「全部都要列出」是同一個道理，缺一個辦公室的聯絡人跟漏掉一個蓋章站點是同等級的錯誤）。若某個辦公室在名單裡但完全沒有回傳任何人員資料（例如只有辦公室名稱、無姓名），該站可以只標註樓層/分機、註明查無承辦人資訊，不要跳過整站不提。

   格式：對每個有資料的辦公室，從其名單裡挑選最相關的1-2位承辦人（同一辦公室內部才需要篩選，不是辦公室之間篩選）：
   `姓名（職責）分機XXXXX——選擇原因：[一句話，例如「負責最終核准」「第一線受理窗口」「這是唯一列出分機的聯絡人」]`
   選擇原因必須具體對應這個人的職責欄位或這題的辦理流程，不能只寫「相關人員」這種空話。如果內容顯示某個步驟需要多層審核（例如先由承辦人受理，再經組長、單位主管逐層簽核），把實際涉及的每一層都列出來、每層各自附上選擇原因，不要只挑一位；如果只是單純的一般承辦窗口，列1-2位最相關的即可。**承辦人姓名是必填項目，不能只寫辦公室名稱、樓層、分機。**
4. 如果對話歷史裡有【表單全文】區塊，且裡面包含**兩份以上**表單：這些是`resource_node`偵測到的全部表單，全部抓取、未經相關性篩選（跟D11/contact_node同一個原則：fetch階段不篩，篩選留給你這一步）。**只挑跟原始問題最直接相關的1-2份**引用細節、列出下載連結，其餘不相關情境的表單（例如問題只問休學，卻同時抓到「提早復學申請書」這種不同情境用的表單）不要列出、不要引用其內容——但只有一份表單時，直接使用即可，不需要這條規則。
5. 如果對話歷史顯示已經搜尋多次仍找不到某個細節，誠實說明未查到，不要杜撰。
6. 回答最後附上來源URL。"""


def _render_full_messages(messages: list) -> str:
    """Full-text renderer for synthesis_node -- unlike loop.py's compact
    _render_messages() (used by rewrite_node to decide what to search
    next), synthesis needs the actual page/form/contact content verbatim,
    not a summary; truncating here would silently drop the exact names/
    numbers synthesis is supposed to cite. Includes ToolMessage (get_page/
    get_form/grep results) and HumanMessage (resource_node/contact_node's
    injected content, domain_router_node's candidate list, rewrite_node's
    own prompts) -- everything except SystemMessage/AIMessage, since the
    latter is just the agent's own tool-call decisions, not retrieved
    content."""
    parts = [str(m.content) for m in messages if isinstance(m, (ToolMessage, HumanMessage))]
    return "\n\n---\n\n".join(parts) if parts else "（尚無內容）"


def synthesis_node(state: AgentState) -> dict:
    """Makes ONE call with a synthesis-specific prompt over the full
    message history (_render_full_messages(), not rewrite_node's compact
    _render_messages()), so contact-name/roster-selection rules have a
    dedicated home instead of being crammed into _AGENT_SYSTEM alongside
    tool-selection instructions."""
    history = _render_full_messages(state["messages"])
    prompt = _SYNTHESIS_PROMPT.format(query=state["query"], history=history)
    # Uses loop.py's already-proven ChatOllama instance (not llm_client.py's
    # simple_chat(), which calls the raw ollama/openai SDKs directly and is
    # therefore invisible to LangGraph's stream_mode="messages" token
    # streaming) so server.py's SSE endpoint can stream this node's answer
    # token-by-token, filtered by langgraph_node=="synthesis_node". agent_node
    # already only supports Ollama cloud regardless of LLM_PROVIDER (see
    # loop.py) -- reusing the same _llm here doesn't introduce a new
    # inconsistency, it follows the one already established.
    # .bind() adds an explicit num_predict, same rationale as llm_client.py's
    # own comment: synthesis answers run long (1000-2000+ tokens), and the
    # Ollama cloud API needs a positive num_predict, not -1/unlimited.
    answer = _llm.bind(options={"num_predict": 8192}).invoke([HumanMessage(content=prompt)]).content
    # Also append to messages (not just state["answer"]) so the conclusion
    # survives into the next turn via the add_messages reducer -- state["answer"]
    # itself gets reset to None by every new turn's initial_state() and no node
    # reads it directly, so without this the answer would otherwise be
    # unreachable from messages, which is what every node's context-building
    # actually consumes. Safe against both structural consumers: _after_tools'
    # marker regex only matches ToolMessage, and _render_full_messages() above
    # only renders ToolMessage/HumanMessage, so a retry within the same turn
    # won't cite this AIMessage back to itself as if it were new source content.
    return {"answer": answer, "messages": [AIMessage(content=answer)]}
