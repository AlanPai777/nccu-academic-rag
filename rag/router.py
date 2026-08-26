"""
rag/router.py
Two-layer query router for the Phase E LangGraph pipeline.

Layer 1: Keyword matching — deterministic, no LLM call
Layer 2: LLM fallback — for queries with no clear keyword signal

QueryType:
  PROCEDURE — multi-step procedure (休學, 復學, 離校...) → ProcedureSkill + moltke priority
  CONTACT   — office contact info (電話, 分機, email, 幾樓) → OfficeLookupSkill
  KNOWLEDGE — factual / regulation query (學分上限, 退費比例) → grep + get_page agent loop
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    PROCEDURE = "procedure"
    CONTACT   = "contact"
    KNOWLEDGE = "knowledge"


@dataclass
class RouteResult:
    query_type: QueryType
    method: str  # "keyword" or "llm"


# ── Layer 1 keyword lists ────────────────────────────────────────────────────

_PROCEDURE_KEYWORDS = [
    # How-to phrasing
    "如何辦理", "如何申請", "怎麼辦理", "怎麼申請",
    "辦理流程", "申請流程", "辦理手續", "申請步驟",
    "辦理程序", "申請程序",
    # Academic status changes (always multi-step)
    "休學", "復學", "退學", "轉學",
    # Other procedure topics
    "請假", "離校", "畢業申請", "延畢", "補辦",
]

_CONTACT_KEYWORDS = [
    # Phone / extension
    "電話", "分機", "聯絡電話",
    # How to contact
    "怎麼聯絡", "如何聯絡", "聯絡方式",
    # Email
    "email", "信箱", "e-mail",
    # Location
    "幾樓", "在哪裡", "在哪", "地址", "辦公室在",
    # Office hours
    "辦公時間", "上班時間",
]

# KNOWLEDGE is the default — no keyword list needed


# ── Layer 1: deterministic keyword matching ──────────────────────────────────

def _dedupe_substring_matches(matched: list[str]) -> list[str]:
    """
    Drop a matched keyword if it's a substring of another matched keyword
    in the same list (e.g. "在哪" matched inside "在哪裡") — otherwise the
    same textual signal gets counted twice and skews proc_hits/cont_hits.
    """
    return [kw for kw in matched if not any(kw != other and kw in other for other in matched)]


def _keyword_route(query: str) -> QueryType | None:
    """
    Return QueryType if keyword signal is clear; None if ambiguous.
    Ties (both procedure + contact hit) → None, let LLM decide.
    """
    q = query  # keep original case for Chinese matching

    proc_matched = _dedupe_substring_matches([kw for kw in _PROCEDURE_KEYWORDS if kw in q])
    cont_matched = _dedupe_substring_matches([kw for kw in _CONTACT_KEYWORDS   if kw in q])
    proc_hits = len(proc_matched)
    cont_hits = len(cont_matched)

    if proc_hits > 0 and proc_hits > cont_hits:
        return QueryType.PROCEDURE
    if cont_hits > 0 and cont_hits > proc_hits:
        return QueryType.CONTACT
    return None  # ambiguous


# ── Layer 2: LLM fallback ────────────────────────────────────────────────────

_LLM_SYSTEM = "你是 NCCU 學術事務 Q&A 系統的 query router，只回傳一個詞。"

_LLM_USER = """將以下問題分類為三種類型之一：
- procedure：需要辦理流程（如休學、復學、退學、申請表格、離校手續）
- contact：詢問聯絡資訊（電話、分機、email、地址、樓層、辦公時間）
- knowledge：詢問法規、政策、事實（學分上限、退費比例、畢業條件等）

只回傳一個詞：procedure / contact / knowledge

問題：{query}"""


def _llm_route(query: str) -> QueryType:
    from rag.llm_client import chat_with_tools

    content, _ = chat_with_tools(
        messages=[{"role": "user", "content": _LLM_USER.format(query=query)}],
        tools=[],
        system_prompt=_LLM_SYSTEM,
        max_tokens=10,
    )
    text = (content or "").strip().lower()
    if "procedure" in text:
        return QueryType.PROCEDURE
    if "contact" in text:
        return QueryType.CONTACT
    return QueryType.KNOWLEDGE


# ── Public API ───────────────────────────────────────────────────────────────

def route(query: str, use_llm_fallback: bool = True) -> RouteResult:
    """
    Classify a query into PROCEDURE / CONTACT / KNOWLEDGE.

    Args:
        query:            The user's natural-language question.
        use_llm_fallback: If False, ambiguous queries default to KNOWLEDGE
                          (useful for smoke tests that should not make API calls).
    """
    layer1 = _keyword_route(query)
    if layer1 is not None:
        return RouteResult(query_type=layer1, method="keyword")

    if use_llm_fallback:
        return RouteResult(query_type=_llm_route(query), method="llm")

    return RouteResult(query_type=QueryType.KNOWLEDGE, method="default")


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _TESTS = [
        # Expected: PROCEDURE
        ("如何辦理休學",         QueryType.PROCEDURE),
        ("如何申請復學",         QueryType.PROCEDURE),
        ("退學的流程是什麼",     QueryType.PROCEDURE),
        ("離校手續怎麼辦",       QueryType.PROCEDURE),
        # Expected: CONTACT
        ("出納組的電話是幾號",   QueryType.CONTACT),
        ("住宿組在幾樓",         QueryType.CONTACT),
        ("生僑組 email",         QueryType.CONTACT),
        ("國合處聯絡方式",       QueryType.CONTACT),
        # Expected: KNOWLEDGE
        ("選課最多可以修幾學分", QueryType.KNOWLEDGE),
        ("退費比例是多少",       QueryType.KNOWLEDGE),
        ("畢業條件有哪些",       QueryType.KNOWLEDGE),
    ]

    print(f"{'Query':<24} {'Expected':<12} {'Got':<12} {'Method':<8} {'OK'}")
    print("-" * 70)
    passed = 0
    for query, expected in _TESTS:
        result = route(query, use_llm_fallback=False)
        ok = "✓" if result.query_type == expected else "✗"
        if result.query_type == expected:
            passed += 1
        print(f"{query:<24} {expected.value:<12} {result.query_type.value:<12} {result.method:<8} {ok}")

    print(f"\n{passed}/{len(_TESTS)} passed (keyword layer only, no LLM)")
