"""
rag/agentic/state.py
Shared LangGraph state schema for rag/agentic_rag.py's pipeline (Migration
Step 1, docs/phase_h_agentic_rag_migration_plan.md Part 5).

⚠ Revised during Step 3 (2026-09-01), replacing the Step 1 version. The
original Step 1 draft started from production's rag/nodes/state.py
(context_pages/sources/office_context/extraction_checklist/sub_queries/
etc., all Annotated[..., operator.add] accumulators) and added 4 fields on
top. That was wrong: every node this migration actually ports from
agentic_main.py (rewrite_node/domain_router_node/agent_node/tools,
resource_node/contact_node, synthesis_node, self_eval_node) is built
entirely around state["messages"] (LangChain AIMessage/ToolMessage/
HumanMessage objects) -- ToolNode, InjectedState, and the deterministic
marker-regex routing (_FORM_MARKER_RE/_OFFICE_MARKER_RE checking the last
ToolMessage's content) all depend on that shape. None of production's
structured fields are read or written by any node this migration carries
forward -- they belonged exclusively to retrieval_procedure.py/
retrieval_knowledge.py/extraction.py/office_lookup.py/decomposition.py's
Send-based sub_query_node, all of which retire (Part 4.1). Keeping them
in state.py would have meant either rewriting every ported node against
the wrong schema (defeating "port the validated implementation") or
silently carrying dead fields. Confirmed via direct grep of
agentic_main.py that composite-query handling (multi_sub_query_node)
never reads a stored sub_queries field either -- it re-splits state["query"]
fresh each time, so that field was never actually needed under this
design.

This file now mirrors agentic_main.py's own AgenticState field-for-field
(9 fields, not 21) -- not a redesign, a direct port of the schema the
reference implementation actually uses.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query:          str
    subdomain_hint: str | None  # set once by domain_router_node (turn 1), reused
                                 # for the rest of the loop
    query_type:     str | None  # "procedure" / "contact" / "resource" / "knowledge" /
                                 # "compound" -- set by plan_node (Step 2); PROCEDURE is
                                 # a label only, not a distinct graph path (Migration
                                 # Step 1 decision)
    turn:           int         # incremented once per rewrite_node/agent_node cycle;
                                 # doom-loop/risk-ceiling checks read this
    rewritten:      str         # this turn's rewrite_node output
    stuck_turns:    int         # consecutive-identical-tool-call counter (doom-loop
                                 # detection, _MAX_STUCK)
    messages:       Annotated[list, add_messages]  # AIMessage/ToolMessage/HumanMessage
                                                     # history; add_messages is the
                                                     # reducer LangChain's ToolNode and
                                                     # ChatOllama.bind_tools() expect
    answer:         str | None  # final answer, written by synthesis_node
    self_eval_note: str | None  # Part 6.2's Stage 1/2 self_eval output
