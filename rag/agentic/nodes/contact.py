"""
rag/agentic/nodes/contact.py
contact_node -- deterministic office-contact lookup, replacing
production's office_lookup.py's hardcoded _PROCEDURE_OFFICES fallback
(Migration Step 4, docs/phase_h_agentic_rag_migration_plan.md Part 5).
Ported directly from rag/agentic_main.py.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from rag.agentic.state import AgentState
from rag.agentic.logic.office_detection import _detect_offices


def contact_node(state: AgentState) -> dict:
    """Deterministically routed here by _after_tools, _after_resource, or
    _after_plan -- all three are objective gates: a marker on a
    ToolMessage/HumanMessage, or plan_node classifying the query CONTACT
    directly. One code path regardless of which gate routed here: read
    whatever context_text is available (last message's content, or the
    raw query itself if messages is still empty), run the existing
    deterministic _detect_offices() scan against it -- this works
    identically whether context_text is a fetched form's full text or the
    bare query "出納組電話幾號", since both are just text to scan. No
    separate "which mode am I in" branching needed.

    Wraps OfficeLookupSkill's batch lookup. Deliberately does NOT run
    offices through an LLM relevance filter the way resource_node's
    _judge_forms() filters form_ids -- offices detected from a form's own
    station list are a completeness requirement (ALL stations must
    appear, not a filterable relevance list), and a pre-fetch filter risks
    permanently losing a required station's contact info on a bad
    judgment call, with no fallback. Fetching all detected offices is
    cheap (single batch lookup); any filtering for what actually surfaces
    in the answer belongs in synthesis, after the data exists."""
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
