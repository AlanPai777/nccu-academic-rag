"""
rag/agentic/nodes/resource.py
resource_node / _after_resource -- deterministic fetch-all form retrieval,
replacing production's retrieval_resource.py (Migration Step 4,
docs/phase_h_agentic_rag_migration_plan.md Part 5). Ported directly from
rag/agentic_main.py.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from rag.agent_tools import extract_form_ids
from rag.agentic.state import AgentState
from rag.agentic.tools.form import get_form_tool
from rag.agentic.logic.form_extraction import (
    _judge_forms, _extract_station_roles, _offices_from_role_keywords, _extract_checklist_blocks,
)
from rag.agentic.logic.office_detection import _detect_offices, _OFFICE_MARKER_RE


def resource_node(state: AgentState) -> dict:
    """Deterministically routed here by either _after_tools (marker found
    on a ToolMessage) or _after_plan (plan_node classified the query
    RESOURCE directly) -- both are the objective gate.

    Fetches ALL forms deterministically detected via extract_form_ids() --
    no pre-fetch relevance judge. A prior design (_judge_forms() deciding
    which detected forms were "worth" fetching before fetching them) had
    the same structural risk contact_node's docstring warns against: a bad
    pre-fetch judgment call permanently loses content with no fallback.
    This also matches production's own retrieval_resource.py: it fetches
    every detected form_id unconditionally and defers ALL relevance
    filtering to the synthesis prompt. _judge_forms() (metadata-search) is
    kept ONLY for the direct plan_node RESOURCE route, where there's no
    page text to extract form_ids from at all -- a raw query never
    contains a literal form_id."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    context_text = str(last.content) if last is not None else ""
    detected_ids = sorted(extract_form_ids(context_text)) if context_text else []
    if detected_ids:
        relevant_ids = detected_ids
    else:
        relevant_ids = _judge_forms(state["query"], context_text, state.get("subdomain_hint"))
        if not relevant_ids:
            return {}
    fetched_ids = set(relevant_ids)
    results = [get_form_tool.invoke({"form_id": fid}) for fid in relevant_ids]
    combined = "\n\n".join(results)

    # Cross-reference pass: a fetched form's own text can reference another
    # form_id never mentioned before -- fetch those too, same fetch-all
    # principle (no judge here either). Bounded to a single extra pass.
    new_ids = [fid for fid in extract_form_ids(combined) if fid not in fetched_ids]
    if new_ids:
        combined += "\n\n" + "\n\n".join(get_form_tool.invoke({"form_id": fid}) for fid in new_ids)

    # Structural extraction runs BEFORE office detection so its findings
    # can feed the offices list (not after resource->contact has already
    # run and it's too late to fetch what extraction discovers).
    station_roles = _extract_station_roles(combined)
    checklist_blocks = _extract_checklist_blocks(combined)

    offices = _detect_offices(combined)
    role_offices = _offices_from_role_keywords(
        [cell for cells in station_roles.values() for cell in cells])
    for o in role_offices:
        if o not in offices:
            offices.append(o)

    combined = "[表單全文，系統偵測到表單編號後自動抓取，請直接引用其中的流程/站點/費用等細節]\n\n" + combined
    if station_roles:
        lines = [f"- 站點{n}：{'、'.join(roles)}（多層審核，每一層都要在答案中列出對應聯絡人，不要只挑一層）"
                 for n, roles in station_roles.items()]
        combined += "\n\n[表單站點審核層級偵測]\n" + "\n".join(lines)
    if checklist_blocks:
        lines = [f"- {b['label']}：{b['options']}" for b in checklist_blocks]
        combined += "\n\n[表單其他決策/資訊項目，答案應涵蓋]\n" + "\n".join(lines)
    if offices:
        combined += f"\n\n[偵測到辦公室: {', '.join(offices)}]"
    return {"messages": [HumanMessage(content=combined)]}


def _after_resource(state: AgentState) -> str:
    """resource -> contact is sequential, not parallel -- contact's
    trigger signal only exists once resource_node has actually fetched the
    form content to scan. Guards against empty messages: reachable via
    plan_node's direct RESOURCE route, where resource_node may judge that
    no form is needed and return {}, leaving state["messages"] exactly as
    empty as when this node started."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if last is not None and isinstance(last, HumanMessage) and _OFFICE_MARKER_RE.search(str(last.content)):
        return "contact"
    return "rewrite"
