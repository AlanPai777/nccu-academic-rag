"""
rag/agent_runtime.py
Shared helpers used by 2+ node modules — office-name detection (used by both
retrieval_procedure and office_lookup nodes), the E6 staleness warning
(post-processing in proto3_langgraph.run()), and the E7 parametric-knowledge
fallback (used by both retrieval_knowledge and synthesis nodes). Phase F
Step 7 (nodes/ package split): this is the "共用skill層" the plan's original
Codebase架構 section named agent_runtime.py for — currently holds these four
helpers; Doom Loop / CRAG-lite retry logic stays inline in
nodes/retrieval_knowledge.py for now (not extracted here), since pulling
those out is a bigger refactor than this split's scope.
"""

from __future__ import annotations

import re

from rag.llm_client import chat_with_tools

# ── Office-name detection (condition 3/4 shared infra) ──────────────────────

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
    """
    combined = " ".join(p.get("text", "") for p in context_pages)
    return [office for office in _PROCEDURE_OFFICES if office in combined]


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
