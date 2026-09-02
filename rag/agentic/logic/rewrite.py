"""
rag/agentic/logic/rewrite.py
Rewrite/judge helpers for the main search loop (Migration Step 3,
docs/phase_h_agentic_rag_migration_plan.md Part 5). Ported directly from
rag/agentic_main.py -- _rewrite_query/_judge_candidates/_render_messages,
unchanged.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from rag.llm_client import simple_chat

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
    """Compact one-liner rendering of tool-call history, used by
    rewrite_node to decide what to search next (contrast with
    synthesis_node's future full-text renderer, which needs verbatim
    content, not a summary)."""
    lines = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"[agent決定呼叫] {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            lines.append(f"[工具結果] {str(m.content)}")
    return "\n".join(lines) if lines else "（尚無記錄）"
