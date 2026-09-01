"""
rag/agentic/state.py
Shared LangGraph state schema for rag/agentic_rag.py's pipeline (Migration
Step 1, docs/phase_h_agentic_rag_migration_plan.md Part 5).

Starts from production's rag/nodes/state.py (all fields preserved --
composite-query handling in particular still needs sub_queries/
sub_query_results/_sub_query exactly as production defined them, see
Part 5 Step 6) and adds four fields agentic_main.py's live multi-turn loop
needs that production's state never had, because production's KNOWLEDGE
path packs its whole ReAct loop inside one function/node (retrieval_node)
instead of exposing turn-by-turn state to the outer graph:

  - turn / stuck_turns: incremented by rewrite_node/agent_node across
    MULTIPLE separate graph nodes within one query -- this is exactly the
    shape that made Send incompatible with directly dispatching into this
    loop (see migration plan §0.2: Send branches writing different values
    to a non-reducer field like `turn` raise InvalidUpdateError). None of
    production's existing Send usage (retrieval_procedure.py's expand
    fan-out, decomposition.py's sub_query_node) touches a field like this,
    which is precisely why production's Send patterns never hit the
    problem this session's spike_send.py surfaced.
  - rewritten: this turn's rewrite_node output (the search term actually
    used), read by domain_router_node/agent_node.
  - subdomain_hint: set once by domain_router_node on turn 1, reused for
    the rest of the loop -- production has no direct equivalent field
    (route_domain()/is_ambiguous() are called fresh, per-node, inside each
    of retrieval_procedure.py/retrieval_knowledge.py/retrieval_resource.py
    individually against that node's own query text).

Not carried over from agentic_main.py's AgenticState: `query_type` here
keeps production's str-based convention (not agentic_main.py's), and
`answer`/`self_eval_note` naming follows production's existing fields
(`answer`/`correction_hint`) rather than introducing a second parallel
naming scheme -- self_eval's actual merged Stage 1/2 design (migration
plan §6.2) is Step 7's work, not Step 1's; this file only adds the state
fields the loop nodes (Step 2-3) will need to read/write.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    query:                str
    query_type:           str        # "procedure" / "contact" / "knowledge" / "resource" / "compound"
    route_method:         str        # "keyword" / "llm" / "default"
    sub_queries:          list[str]  # non-empty only for composite queries

    # Annotated with operator.add so N parallel Send-branches (production's
    # retrieval_procedure.py expand fan-out, decomposition.py's
    # sub_query_node) each contribute their own page/URL without
    # clobbering each other -- LangGraph sums all writes to this field
    # within one superstep. Preserved as-is from production; still valid
    # under the migrated architecture since nothing about these fields'
    # accumulator semantics changes.
    context_pages:        Annotated[list[dict], operator.add]
    sources:               Annotated[list[str], operator.add]

    # Composite-query Send-branch results (production's decomposition.py) --
    # see migration plan §0.2/Part 5 Step 6 for why this specific pattern
    # (a Send branch writing a per-branch dict into an accumulator list)
    # stays safe while a direct Send into the live loop would not: each
    # sub_query_node branch here is a single-shot pass, never a multi-turn
    # loop with its own live counter field.
    sub_query_results:    Annotated[list[dict], operator.add]
    _sub_query:           str  # this branch's one sub_query text (Send-branch payload)

    detected_offices:     list[str]
    _anchor_links:        list[str]
    _anchor_form_ids:     list[str]
    expand_target:        dict
    office_context:       str
    office_lookup_result: dict
    extraction_checklist: dict

    answer:               str
    correction_hint:      str
    iteration:            int

    # ── New in Migration Step 1 (agentic_main.py's live loop needs these;
    #    production's KNOWLEDGE path never exposed turn-by-turn state to
    #    the outer graph, so it never needed them) ──────────────────────
    turn:                 int         # incremented once per rewrite_node/agent_node
                                       # cycle; doom-loop/risk-ceiling checks read this
    rewritten:            str         # this turn's rewrite_node output
    stuck_turns:          int         # consecutive-identical-tool-call counter
                                       # (doom-loop detection, _MAX_STUCK)
    subdomain_hint:       str | None  # set once by domain_router_node (turn 1),
                                       # reused for the rest of the loop
