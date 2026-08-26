"""
rag/nodes/office_lookup.py
office_lookup_node: injects office contact info into synthesis context.
"""

from __future__ import annotations

from rag.router import QueryType
from rag.agent_runtime import _offices_from_context, _offices_from_query, _PROCEDURE_OFFICES
from rag.nodes.state import AgentState


def office_lookup_node(state: AgentState) -> AgentState:
    """
    Inject office contact info into synthesis context.

    PROCEDURE: runs _offices_from_context() fresh, on the FULL context_pages
               (Step 4.5 fixed this 2026-08-26: originally prioritized
               detected_offices, which retrieval_anchor_node snapshots from
               ONLY its own anchor pages before the Send-based expand fan-out
               has run — office names that only appear in expand-fetched
               content, e.g. 休學's moltke form mentioning 住宿組/國際合作事務處,
               were silently missed. context_pages is the merged anchor+all-
               expand-branches set by the time THIS node runs (fan-in already
               complete via the operator.add reducer), so re-running the same
               detection function against it is provably a superset of the
               anchor-only snapshot — negligible cost (pure substring scan,
               no I/O/LLM), so there's no real reason to prefer the narrower
               snapshot. detected_offices is kept as a field (still useful as
               an early signal if expand's own targeting gets smarter later)
               but is no longer this node's primary source.
    CONTACT:   inject offices mentioned in the query; fallback to all if none found.
    KNOWLEDGE: skip — office info not needed for factual queries.

    OfficeLookupSkill's contact lookup is itself dynamic (condition 4,
    office_contacts_index.jsonl primary, KNOWN_CONTACTS fallback) — this
    node only decides WHICH offices to ask it about.
    """
    from rag.skills.office_lookup_skill import OfficeLookupSkill

    qtype = state["query_type"]

    if qtype == QueryType.KNOWLEDGE:
        return {**state, "office_context": "", "office_lookup_result": {}}

    if qtype == QueryType.PROCEDURE:
        offices = _offices_from_context(state.get("context_pages", [])) or _PROCEDURE_OFFICES
    else:  # CONTACT
        offices = _offices_from_query(state["query"]) or _PROCEDURE_OFFICES

    skill  = OfficeLookupSkill()
    result = skill.run(offices)
    return {
        **state,
        "office_context":       skill.format_context(result),
        "office_lookup_result": result,
    }
