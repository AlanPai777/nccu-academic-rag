"""
rag/agentic/nodes/compound.py
Compound-query handling (v1 sequential) -- multi_sub_query_node runs each
detected sub-query through its own fully independent nested .invoke() of
an inner loop graph, replacing production's Send-based sub_query_node/
merge_node (decomposition.py). Migration Step 6,
docs/phase_h_agentic_rag_migration_plan.md Part 5 / §0.2.

Ported directly from rag/agentic_main.py. Deliberately NOT Send/parallel:
sequential sidesteps the concurrent-write problem entirely (per §0.2,
Send cannot directly target this package's live multi-turn loop nodes --
spike_send.py proved concurrent branches writing different values to a
non-reducer field like `turn` raise InvalidUpdateError). Send upgrade
(via a wrapper node that internally does its own nested .invoke(), the
pattern spike_nested_invoke.py validated) is an explicit follow-up, not
solved this step -- matches production's own v1-then-Send-upgrade
precedent for the same feature.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from rag.router import _keyword_route, QueryType
from rag.agentic.state import AgentState
from rag.agentic.nodes.plan import _CLAUSE_SPLIT_RE, _after_plan
from rag.agentic.nodes.loop import (
    rewrite_node, domain_router_node, agent_node, _after_agent, _after_tools,
)
from rag.agentic.nodes.resource import resource_node, _after_resource
from rag.agentic.nodes.contact import contact_node
from rag.agentic.tools import TOOLS

_loop_graph_cache = None


def _build_loop_graph():
    """Compiles rewrite_node<->agent_node<->tools (plus resource_node/
    contact_node's marker chain) as an INDEPENDENT StateGraph, separate
    from the outer graph (agentic_rag.py at repo root, Step 9) -- reused by
    multi_sub_query_node's nested .invoke() calls so each sub-query gets
    its own fully isolated turn/rewritten/subdomain_hint/query_type, with
    zero risk of the concurrent-write InvalidUpdateError a shared graph
    run would hit. Reuses the exact same node functions the outer graph
    uses -- no logic duplicated, just a second, smaller assembly of them.
    Entry routing reuses _after_plan's own contact/resource/knowledge
    logic (a sub-query's query_type is already set by multi_sub_query_node
    before invoking this, so no plan_node/compound-detection step is
    needed here -- compound-within-compound is out of scope). Compiled
    once (module-level cache): compiling is cheap, but there's no reason
    to repeat it every sub-query call."""
    global _loop_graph_cache
    if _loop_graph_cache is not None:
        return _loop_graph_cache
    g = StateGraph(AgentState)
    g.add_node("rewrite_node", rewrite_node)
    g.add_node("domain_router_node", domain_router_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("resource_node", resource_node)
    g.add_node("contact_node", contact_node)
    g.add_conditional_edges(START, _after_plan,
                             {"knowledge": "rewrite_node", "resource": "resource_node", "contact": "contact_node"})
    g.add_edge("rewrite_node", "domain_router_node")
    g.add_edge("domain_router_node", "agent_node")
    g.add_conditional_edges("agent_node", _after_agent, {"tools": "tools", "end": END})
    g.add_conditional_edges("tools", _after_tools, {"resource": "resource_node", "contact": "contact_node", "rewrite": "rewrite_node", "end": END})
    g.add_conditional_edges("resource_node", _after_resource, {"contact": "contact_node", "rewrite": "rewrite_node"})
    g.add_edge("contact_node", "rewrite_node")
    _loop_graph_cache = g.compile()
    return _loop_graph_cache


def multi_sub_query_node(state: AgentState) -> dict:
    """v1 sequential: runs each detected sub-query through its own fully
    independent invocation of _build_loop_graph(), one after another.

    Each sub-query's turn/rewritten/subdomain_hint/query_type live ONLY
    inside that one nested .invoke() call and are discarded when it
    returns -- only `messages` (which has add_messages as its reducer)
    crosses back, via plain list concatenation here since this is one
    node's single return, not a multi-branch merge.

    Failure isolation: each sub-query's .invoke() is wrapped in its own
    try/except -- one sub-query failing appends an honest failure note to
    messages instead of raising, which would otherwise abort the whole
    compound-query answer (an uncaught exception propagates to the caller
    and kills the entire run, not just one branch).

    Known limitation (v1, accepted): if self_eval_node retries this
    compound query, the WHOLE for-loop re-runs (all N sub-queries), not
    just the one that was actually deficient -- self_eval currently has no
    way to know which sub-query, if any, was the problem. Not solved this
    step."""
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(state["query"]) if c.strip()]
    loop_graph = _build_loop_graph()
    all_messages: list = []
    for clause in clauses:
        qt = _keyword_route(clause) or QueryType.KNOWLEDGE
        try:
            result = loop_graph.invoke({
                "query": clause, "subdomain_hint": None, "query_type": qt.value,
                "turn": 0, "rewritten": "", "stuck_turns": 0, "messages": [],
                "answer": None, "self_eval_note": None,
            })
            all_messages.extend(result.get("messages", []))
        except Exception as e:
            all_messages.append(HumanMessage(content=f"[子問題「{clause}」查詢時發生錯誤，未能取得資訊：{e}]"))
    return {"messages": all_messages}
