"""
rag/nodes/decomposition.py
Step 2.5: composite-query handling — detects multiple distinct topics in one
question, processes each sub_query via a Send-dispatched branch
(sub_query_node), and merges the per-sub_query results at the data layer
(merge_node).
"""

from __future__ import annotations

import re

from langgraph.types import Send

from rag.router import route, QueryType, _keyword_route
from rag.nodes.state import AgentState
from rag.nodes.retrieval_knowledge import retrieval_node
from rag.nodes.office_lookup import office_lookup_node
from rag.nodes.extraction import extraction_node

# Split on 逗號/句號 (full-width and half-width) — clause boundaries are the
# unit query_decomposition_node reasons about.
_CLAUSE_SPLIT_RE = re.compile(r'[，。,.]')


def query_decomposition_node(state: AgentState) -> AgentState:
    """
    Step 2.5: detect composite queries (multiple distinct topics in one
    question) BEFORE routing, so they don't get silently collapsed to a
    single QueryType by router_node's single-label classification (confirmed
    in Step 1 Q6: "如何辦理休學，圖書館的電話是多少" degraded the 休學 half
    when both topics were forced through one router_node/synthesis_node pass).

    v1 detection is pure Layer-1 keyword matching, reusing router.py's
    existing _keyword_route() per clause — no LLM call. This mirrors
    router.py's own "keyword first, LLM only when ambiguous" principle at
    the decomposition level: a clause with no clear keyword signal doesn't
    force an LLM call just to decide whether the query is composite.

    A query is composite only if 2+ clauses resolve to DIFFERENT QueryTypes
    via _keyword_route(). A single-topic query that merely has a pause
    (e.g. "休學需要注意哪些事，包含要去哪些辦公室" — both clauses are
    QueryType.PROCEDURE or ambiguous) is NOT composite; sub_queries stays
    empty and the query proceeds through the normal single-query path.
    """
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(state["query"]) if c.strip()]
    if len(clauses) < 2:
        return {**state, "sub_queries": []}

    types_found = {t for t in (_keyword_route(c) for c in clauses) if t is not None}
    if len(types_found) < 2:
        return {**state, "sub_queries": []}  # not composite — 0 or 1 distinct type

    return {**state, "sub_queries": clauses}


def sub_query_node(state: AgentState) -> AgentState:
    """
    Step 2.5 Send upgrade: processes ONE sub_query from a composite query —
    a Send-branch target dispatched by _dispatch_sub_queries, one branch per
    sub_query, running in parallel. Must return ONLY the fields it updates
    (context_pages / sources / sub_query_results), never a full {**state,
    ...} merge — same rule retrieval_expand_node already follows, since N
    branches write concurrently within one LangGraph superstep.

    Body is unchanged from merge_node's old per-sub_query loop iteration:
    route() → retrieval_node/office_lookup_node/extraction_node as plain
    function calls against an isolated sub_state (not through graph
    routing) — same node functions the single-query path uses, not
    reimplemented. retrieval_node is skipped for CONTACT sub-queries,
    matching _after_router's routing behaviour (retrieval_node's KNOWLEDGE
    branch would otherwise run for CONTACT, which it was never designed for).
    """
    sub_query = state["_sub_query"]
    result = route(sub_query)
    sub_state: AgentState = {
        **state,
        "query":                sub_query,
        "query_type":           result.query_type.value,
        "route_method":         result.method,
        "context_pages":        [],
        "office_context":       "",
        "extraction_checklist": {},
        "sources":              [],
    }

    if result.query_type in (QueryType.PROCEDURE, QueryType.KNOWLEDGE):
        sub_state = retrieval_node(sub_state)

    sub_state = office_lookup_node(sub_state)
    sub_state = extraction_node(sub_state)

    return {
        "context_pages": sub_state.get("context_pages", []),
        "sources":       sub_state.get("sources", []),
        "sub_query_results": [{
            "office_context":       sub_state.get("office_context", ""),
            "extraction_checklist": sub_state.get("extraction_checklist", {}),
        }],
    }


def _dispatch_sub_queries(state: AgentState) -> list[Send]:
    """
    Conditional-edge routing function: one Send per sub_query, mirroring
    _dispatch_expand's pattern (Step 4.5) — N determined by how many
    sub_queries query_decomposition_node actually found, not fixed.
    _after_decomposition only calls this when sub_queries is non-empty, so
    no empty-list fallback is needed here (unlike _dispatch_expand).
    """
    return [
        Send("sub_query_node", {**state, "_sub_query": sub_query})
        for sub_query in state["sub_queries"]
    ]


def merge_node(state: AgentState) -> AgentState:
    """
    Step 2.5: convergence point after all sub_query_node Send-branches
    complete. context_pages/sources are already fully merged by their own
    operator.add reducers by the time this runs (same mechanism
    retrieval_expand_node's branches rely on) — this function only flattens
    and dedupes the per-sub_query office_context/extraction_checklist
    fragments collected in sub_query_results, since those two fields aren't
    reducer-safe (see AgentState's sub_query_results docstring). Never
    merges generated text — one synthesis_node call handles the merged data
    for the whole composite query, same as a single query would.
    """
    office_sections: list[str] = []
    merged_person_names: list[dict] = []
    merged_forms: list[dict] = []
    merged_notes: list[str] = []
    seen_form_ids: set[str] = set()
    seen_notes: set[str] = set()

    for r in state.get("sub_query_results", []):
        if r.get("office_context"):
            office_sections.append(r["office_context"])

        checklist = r.get("extraction_checklist", {})
        merged_person_names.extend(checklist.get("person_names", []))
        for f in checklist.get("forms", []):
            if f["id"] not in seen_form_ids:
                seen_form_ids.add(f["id"])
                merged_forms.append(f)
        for n in checklist.get("notes", []):
            if n["text"] not in seen_notes:
                seen_notes.add(n["text"])
                merged_notes.append(n)

    return {
        **state,
        "sources":               list(dict.fromkeys(state.get("sources", []))),
        "office_context":       "\n\n".join(office_sections),
        "extraction_checklist": {
            "person_names": merged_person_names,
            "forms":        merged_forms,
            "notes":        merged_notes,
        },
        # Composite answers always go through the full prompt-based synthesis
        # (office_section + checklist), never the KNOWLEDGE pass-through —
        # needed to weave multiple sub-topics into one coherent answer even
        # when every sub_query happened to be KNOWLEDGE-type. Also keeps
        # self_eval_node's retry loop active (PROCEDURE-only gate), valuable
        # here since composite answers are the highest-complexity case.
        "query_type":            QueryType.PROCEDURE.value,
    }
