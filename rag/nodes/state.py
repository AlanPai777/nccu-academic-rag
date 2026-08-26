"""
rag/nodes/state.py
Shared LangGraph state schema for proto3_langgraph.py's pipeline. Split into
its own module (Phase F Step 7, nodes/ package split) so every node module
can import it without importing proto3_langgraph.py itself (which would be
circular, since proto3_langgraph.py imports the node functions).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    query:                str
    query_type:           str        # "procedure" / "contact" / "knowledge"
    route_method:         str        # "keyword" / "llm" / "default"
    sub_queries:          list[str]  # Step 2.5: non-empty only for composite queries
    # Annotated with operator.add so N parallel retrieval_expand_node branches
    # (Step 4.5, Send-based fan-out) each contribute their own page/URL without
    # clobbering each other or the anchor node's own contribution — LangGraph
    # sums all writes to this field within one superstep. Safe because exactly
    # one path (anchor+expand XOR retrieval_node XOR merge_node) ever writes to
    # it per graph run, always starting from the [] set in run()'s initial state.
    context_pages:        Annotated[list[dict], operator.add]
    sources:               Annotated[list[str], operator.add]
    # Step 2.5 Send upgrade: each sub_query_node Send-branch appends ONE dict
    # ({"office_context", "extraction_checklist"}) for its own sub_query —
    # office_context/extraction_checklist themselves have no sane
    # operator.add semantics (str concatenation / dict merging aren't safe
    # across N concurrent branches), so the raw per-branch results are
    # collected here instead and flattened/deduped by merge_node after all
    # branches converge, same division of labor context_pages/sources
    # already have via their own reducer.
    sub_query_results:    Annotated[list[dict], operator.add]
    _sub_query:           str  # Step 2.5 Send upgrade: this branch's one sub_query text
    detected_offices:     list[str]  # Step 4.5: set by retrieval_anchor_node so
                                      # office_lookup_node doesn't re-detect (condition 3)
    _anchor_links:        list[str]  # Step 4.5: cross-domain links found in anchor content
    _anchor_form_ids:     list[str]  # Step 4.5: form IDs found in anchor content
    expand_target:        dict       # Step 4.5: per-Send-branch payload {"kind","value"}
    office_context:       str        # formatted contact info — E3: OfficeLookupSkill
    office_lookup_result: dict       # raw {office: {contacts: [...], ...}} from
                                      # OfficeLookupSkill.run() — kept structured
                                      # (not just the flattened office_context
                                      # string) so extraction_node can build a
                                      # per-office checklist without regex
                                      # re-parsing already-formatted text
    extraction_checklist: dict       # condition 6: pre-synthesis dynamic checklist
    answer:               str        # final answer
    correction_hint:      str        # E4: self-eval feedback for synthesis retry
    iteration:            int        # E4: retry counter (max _MAX_SELF_EVAL_RETRIES)
