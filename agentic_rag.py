"""
agentic_rag.py
Graph assembly + CLI for the migrated agentic RAG pipeline (Migration
Step 9, docs/phase_h_agentic_rag_migration_plan.md Part 5). Replaces
rag/proto3_langgraph.py as the production entry point -- same role
(build_graph()/run()/CLI only, no node logic of its own), assembled
entirely from rag/agentic/*'s already-ported and verified node functions
(Steps 1-8).

Moved to the repo root (post-Step-10 reorg) so the production entry point
is visible at a glance, sitting alongside rag/classic_rag.py (classic RAG,
renamed from rag/main.py in the same reorg). Both direct execution and
module invocation work without any sys.path patch, since the repo root is
this file's own directory:
    python agentic_rag.py "如何辦理休學"
    python -m agentic_rag "出納組的電話" --subdomain aca
    python agentic_rag.py "選課上限幾學分" --stream
    python agentic_rag.py "如何辦理休學" --no-eval
"""

from __future__ import annotations

import sys

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from rag.llm_client import get_active_model, PROVIDER
from rag.eval import print_score_report

from rag.agentic.state import AgentState
from rag.agentic.nodes.plan import plan_node, _after_plan
from rag.agentic.nodes.loop import rewrite_node, domain_router_node, agent_node, _after_agent, _after_tools
from rag.agentic.nodes.resource import resource_node, _after_resource
from rag.agentic.nodes.contact import contact_node
from rag.agentic.nodes.synthesis import synthesis_node
from rag.agentic.nodes.compound import multi_sub_query_node
from rag.agentic.nodes.self_eval import self_eval_node, _after_self_eval
from rag.agentic.logic.rewrite import _render_messages
from rag.agentic.tools import TOOLS


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("plan_node", plan_node)
    g.add_node("rewrite_node", rewrite_node)
    g.add_node("domain_router_node", domain_router_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("resource_node", resource_node)
    g.add_node("contact_node", contact_node)
    g.add_node("synthesis_node", synthesis_node)
    g.add_node("self_eval_node", self_eval_node)
    g.add_node("multi_sub_query_node", multi_sub_query_node)

    g.add_edge(START, "plan_node")
    g.add_conditional_edges("plan_node", _after_plan,
                             {"knowledge": "rewrite_node", "resource": "resource_node",
                              "contact": "contact_node", "compound": "multi_sub_query_node"})
    g.add_edge("multi_sub_query_node", "synthesis_node")
    g.add_edge("rewrite_node", "domain_router_node")
    g.add_edge("domain_router_node", "agent_node")
    g.add_conditional_edges("agent_node", _after_agent, {"tools": "tools", "end": "synthesis_node"})
    g.add_conditional_edges("tools", _after_tools, {"resource": "resource_node", "contact": "contact_node", "rewrite": "rewrite_node", "end": "synthesis_node"})
    g.add_conditional_edges("resource_node", _after_resource, {"contact": "contact_node", "rewrite": "rewrite_node"})
    g.add_edge("contact_node", "rewrite_node")
    g.add_edge("synthesis_node", "self_eval_node")
    g.add_conditional_edges("self_eval_node", _after_self_eval, {"end": END, "rewrite": "rewrite_node", "plan": "plan_node"})

    return g.compile(checkpointer=checkpointer)


def initial_state(query: str, subdomain_hint: str | None = None) -> AgentState:
    """Shared per-turn starting state -- used by both the CLI's run() (no
    checkpointer, always a fresh conversation) and server.py (checkpointer-
    backed, one call per turn on a possibly-existing thread_id). Kept in one
    place so a future field never gets added to one caller and silently
    missed in the other."""
    return {
        "query": query, "subdomain_hint": subdomain_hint, "query_type": None, "turn": 0, "rewritten": "",
        "stuck_turns": 0, "messages": [], "answer": None, "self_eval_note": None,
    }


def run(query: str, subdomain_hint: str | None = None, stream: bool = False) -> dict:
    graph = build_graph()
    initial = initial_state(query, subdomain_hint)

    if stream:
        final_state = dict(initial)
        for update in graph.stream(initial, stream_mode="updates"):
            for node_name, delta in update.items():
                delta = delta or {}  # a node returning {} (e.g. domain_router_node's no-op turns) surfaces as None here
                print(f"[{node_name}] {list(delta.keys())}", file=sys.stderr)
                for k, v in delta.items():
                    if k == "messages":
                        final_state["messages"] = final_state.get("messages", []) + v
                    else:
                        final_state[k] = v
        return final_state

    return graph.invoke(initial)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--subdomain", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="跳過 eval 評分")
    args = parser.parse_args()

    print(f"[agentic_rag] model={get_active_model()}  provider={PROVIDER}", file=sys.stderr)

    result = run(args.query, subdomain_hint=args.subdomain, stream=args.stream)

    print(f"\n{'='*70}")
    print(f"Query: {args.query!r}")
    print(f"{'='*70}")
    print(_render_messages(result.get("messages", [])))
    print(f"\n最終答案 ({result.get('turn', 0)}輪):")
    answer = result.get("answer")
    print(answer)

    if not args.no_eval:
        print_score_report(answer)
