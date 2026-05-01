"""
rag/proto1_direct.py
Prototype 1: Agentic RAG with 5 tools.
Uses llm_client.chat_with_tools() — works with both Ollama cloud and OpenRouter.

Run:
    # Ollama cloud (default)
    python rag/proto1_direct.py "如何辦理休學"

    # OpenRouter
    LLM_PROVIDER=openrouter python rag/proto1_direct.py "如何辦理休學"

    # Verbose (shows tool calls)
    python rag/proto1_direct.py "如何辦理休學" --verbose
"""

import json
import sys

from rag.agent_tools import grep_texts, get_page, extract_links, get_children, get_form
from rag.llm_client import chat_with_tools, get_active_model, PROVIDER
from rag.eval import print_score_report

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name":        "grep_texts",
            "description": "在 NCCU 文件庫全文搜尋，回傳相關頁面列表（含預覽文字）",
            "parameters": {
                "type":       "object",
                "properties": {
                    "pattern":     {"type": "string",  "description": "搜尋字串（中文或英文）"},
                    "subdomain":   {"type": "string",  "description": "限縮範圍：aca / osa / www.lib / learning / nccuga / oic"},
                    "max_results": {"type": "integer", "description": "最多回傳筆數（預設 5）"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_page",
            "description": "取得指定 URL 的完整頁面內文（整頁，不分段）。適合取得完整的規定說明、退費標準、電話分機等。",
            "parameters": {
                "type":       "object",
                "properties": {"url": {"type": "string", "description": "完整 URL"}},
                "required":   ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "extract_links",
            "description": "提取某頁面中所有指向 NCCU 子網域（*.nccu.edu.tw）的 markdown 連結。用於追蹤跨處室資訊，例如從 aca 頁面找到 osa 退費頁的連結。",
            "parameters": {
                "type":       "object",
                "properties": {"url": {"type": "string", "description": "要提取連結的頁面 URL"}},
                "required":   ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_children",
            "description": "取得某 URL 在 BFS 爬蟲中的所有直屬子頁面。用於從分類頁往下找特定子頁。",
            "parameters": {
                "type":       "object",
                "properties": {
                    "url":       {"type": "string"},
                    "subdomain": {"type": "string", "description": "限縮 subdomain（可選）"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_form",
            "description": "取得官方 moltke 表單的完整內容，包含申辦流程、各單位辦公地點（行政大樓幾樓）。表單 ID 格式：QP-T01-03-02",
            "parameters": {
                "type":       "object",
                "properties": {"form_id": {"type": "string", "description": "表單 ID，格式如 QP-T01-03-02"}},
                "required":   ["form_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """你是國立政治大學（政大）的學務助理，使用工具搜尋政大官方資料回答學生問題。

【回答格式規則】
1. 申辦流程問題：必須以**步驟清單**格式回答，列出每站辦理單位、地點（行政大樓 X 樓）、電話分機
2. 條件限定步驟（只有住宿生 or 只有國際學生需要辦）請在步驟旁明確標注
3. 若有退費標準，請列出退費比例表（全額 / 2/3 / 1/3 / 不退）
4. 保留所有官方表單連結（如 QP-T01-03-02）
5. 只根據工具搜尋到的資料回答；資料不足時說「無法確認」而非猜測
6. 回答最後以「【引用來源】」列出所有使用到的 URL

【工具使用策略】
步驟一：grep_texts("休學", subdomain="aca") 找主頁
步驟二：get_page(主頁 URL) 取完整內文
步驟三：extract_links(主頁 URL) 取所有 NCCU 跨處室連結 → 找退費標準、表單等連結
步驟四：get_page(退費頁 URL)  ← 必須用 extract_links 取得的 URL，不可自行猜測 URL
步驟五：get_form("QP-T01-03-02") 取蓋章流程與樓層

⚠️ 重要：絕對不可自行猜測或推算 URL。跨處室連結（osa、moltke 等）必須透過 extract_links 取得後再 get_page，URL 猜錯會導致找不到資料。
"""


def _dispatch(name: str, args: dict):
    if name == "grep_texts":    return grep_texts(**args)
    if name == "get_page":      return get_page(**args)
    if name == "extract_links": return extract_links(**args)
    if name == "get_children":  return get_children(**args)
    if name == "get_form":      return get_form(**args)
    return {"error": f"Unknown tool: {name}"}


def run_agent(query: str, max_turns: int = 10, verbose: bool = False) -> str:
    messages:    list[dict] = [{"role": "user", "content": query}]
    seen_calls:  list[str]  = []
    last_content: str | None = None

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Turn {turn+1}]", file=sys.stderr)

        content, tool_calls = chat_with_tools(
            messages=messages,
            tools=TOOLS_SCHEMA,
            system_prompt=SYSTEM_PROMPT,
        )

        # Final answer
        if not tool_calls:
            return content or last_content or "（無回應）"

        if content:
            last_content = content
            if verbose and content:
                print(f"  [LLM reasoning] {content[:200]}", file=sys.stderr)

        # Build assistant message for history
        messages.append({
            "role":       "assistant",
            "content":    content,
            "tool_calls": tool_calls,
        })

        # Execute tools and append results
        for tc in tool_calls:
            call_sig = f"{tc['function']['name']}:{tc['function']['arguments']}"

            if call_sig in seen_calls:
                result_str = json.dumps(
                    {"warning": "已搜尋過相同條件，請換不同關鍵字或工具"},
                    ensure_ascii=False,
                )
                if verbose:
                    print(f"  [LOOP] {tc['function']['name']} repeated", file=sys.stderr)
            else:
                seen_calls.append(call_sig)
                args   = json.loads(tc["function"]["arguments"])
                result = _dispatch(tc["function"]["name"], args)
                result_str = json.dumps(result, ensure_ascii=False)
                if verbose:
                    fn = tc["function"]["name"]
                    print(f"  [{fn}] {args}", file=sys.stderr)
                    if isinstance(result, list):
                        print(f"    → {len(result)} items", file=sys.stderr)
                        for item in result[:2]:
                            print(f"      · {item.get('title','?')[:40]}  {item.get('url','')[:60]}", file=sys.stderr)
                    elif isinstance(result, dict) and "text" in result:
                        print(f"    → {len(result['text'])} chars: {result['text'][:120]!r}", file=sys.stderr)
                    elif isinstance(result, dict) and "error" in result:
                        print(f"    → ERROR: {result['error']}", file=sys.stderr)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result_str,
            })

    return last_content or "達到最大輪數。"


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    args    = [a for a in sys.argv[1:] if not a.startswith("--")]
    query   = args[0] if args else "如何辦理休學"

    print(f"Provider : {PROVIDER}")
    print(f"Model    : {get_active_model()}")
    print(f"Query    : {query}")
    print("─" * 60)

    answer = run_agent(query, verbose=verbose)
    print(answer)
    print_score_report(answer)
