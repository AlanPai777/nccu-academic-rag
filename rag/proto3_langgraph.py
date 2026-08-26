"""
rag/proto3_langgraph.py
Prototype 3: LangGraph-based agentic RAG pipeline.

Phase F Step 7 (2026-08-26): node implementations live in rag/nodes/ — this
file is graph assembly only (AgentState import, conditional-edge routing,
build_graph(), run(), CLI). Crossed the plan's own 1000-line maintainability
threshold (1262 lines) before this split.

Graph:
  START → query_decomposition_node ──(not composite)──▶ router_node ──(PROCEDURE)──▶ retrieval_anchor_node ─[Send ×N]─▶ retrieval_expand_node ──▶ office_lookup_node
                                  │                                 ├──(KNOWLEDGE)──────────────────────────────────────────────────────────────▶ retrieval_node ──▶ office_lookup_node
                                  │                                 ├──(RESOURCE)───────────────────────────────────────────────────────────────▶ resource_node ──▶ office_lookup_node
                                  │                                 ╰──(CONTACT)───────────────────────────────────────────────────────────────────────────────────▶ office_lookup_node
                                  │                                                                                                                                          │
                                  │                                                                                                                                  extraction_node
                                  ╰──(composite)──▶ sub_query_node [Send ×N, one per sub_query] ──▶ merge_node (flatten+dedupe; each branch still uses retrieval_node/ProcedureSkill internally, not anchor+expand — known follow-up) ─╯
                                                                                                                                                                              │
                                                                                                                                                                      synthesis_node ◀─╮
                                                                                                                                                                              │         │ correction_hint set
                                                                                                                                                                      self_eval_node ──╯ (max 2 retries)
                                                                                                                                                                              │
                                                                                                                                                                             END

Run:
    python -m rag.proto3_langgraph "如何辦理休學"
    python -m rag.proto3_langgraph "出納組電話"
    python -m rag.proto3_langgraph "選課上限幾學分"
    python -m rag.proto3_langgraph "如何辦理休學" --no-eval
"""

from __future__ import annotations

import sys

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from rag.router import QueryType
from rag.llm_client import get_active_model, PROVIDER
from rag.eval import print_score_report
from rag.agent_runtime import _staleness_warning

from rag.nodes.state import AgentState
from rag.nodes.decomposition import (
    query_decomposition_node, sub_query_node, _dispatch_sub_queries, merge_node,
)
from rag.nodes.routing import router_node
from rag.nodes.retrieval_procedure import retrieval_anchor_node, _dispatch_expand, retrieval_expand_node
from rag.nodes.retrieval_knowledge import retrieval_node
from rag.nodes.retrieval_resource import resource_node
from rag.nodes.office_lookup import office_lookup_node
from rag.nodes.extraction import extraction_node
from rag.nodes.synthesis import synthesis_node
from rag.nodes.self_eval import self_eval_node


# ── Conditional routing ───────────────────────────────────────────────────────

def _after_decomposition(state: AgentState) -> str | list[Send]:
    if state.get("sub_queries"):
        return _dispatch_sub_queries(state)
    return "router_node"


def _after_router(state: AgentState) -> str:
    if state["query_type"] == QueryType.CONTACT:
        return "office_lookup_node"
    if state["query_type"] == QueryType.PROCEDURE:
        return "retrieval_anchor_node"  # Step 4.5: anchor+expand, not retrieval_node
    if state["query_type"] == QueryType.RESOURCE:
        return "resource_node"
    return "retrieval_node"  # KNOWLEDGE


def _after_self_eval(state: AgentState) -> str:
    """Route back to synthesis_node if correction_hint is set; else end."""
    if state.get("correction_hint"):
        return "synthesis_node"
    return END


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("query_decomposition_node", query_decomposition_node)
    g.add_node("sub_query_node",       sub_query_node)
    g.add_node("merge_node",           merge_node)
    g.add_node("router_node",          router_node)
    g.add_node("retrieval_anchor_node", retrieval_anchor_node)
    g.add_node("retrieval_expand_node", retrieval_expand_node)
    g.add_node("retrieval_node",       retrieval_node)
    g.add_node("resource_node",        resource_node)
    g.add_node("office_lookup_node",   office_lookup_node)
    g.add_node("extraction_node",      extraction_node)
    g.add_node("synthesis_node",       synthesis_node)
    g.add_node("self_eval_node",       self_eval_node)

    g.add_edge(START, "query_decomposition_node")
    g.add_conditional_edges("query_decomposition_node", _after_decomposition)
    g.add_edge("sub_query_node",        "merge_node")
    g.add_edge("merge_node",            "synthesis_node")
    g.add_conditional_edges("router_node", _after_router)
    g.add_conditional_edges("retrieval_anchor_node", _dispatch_expand)
    g.add_edge("retrieval_expand_node", "office_lookup_node")
    g.add_edge("retrieval_node",        "office_lookup_node")
    g.add_edge("resource_node",         "office_lookup_node")
    g.add_edge("office_lookup_node",    "extraction_node")
    g.add_edge("extraction_node",       "synthesis_node")
    g.add_edge("synthesis_node",        "self_eval_node")
    g.add_conditional_edges("self_eval_node", _after_self_eval)

    return g.compile()


_graph = None


def run(query: str) -> str:
    global _graph
    if _graph is None:
        _graph = build_graph()

    final = _graph.invoke({
        "query":                query,
        "query_type":           "",
        "route_method":         "",
        "sub_queries":          [],
        "sub_query_results":    [],
        "_sub_query":           "",
        "context_pages":        [],
        "sources":              [],
        "detected_offices":     [],
        "_anchor_links":        [],
        "_anchor_form_ids":     [],
        "expand_target":        {},
        "office_context":       "",
        "office_lookup_result": {},
        "extraction_checklist": {},
        "answer":               "",
        "correction_hint":      "",
        "iteration":            0,
    })
    # E6: append staleness warning (post-processing, not a graph node)
    return final["answer"] + _staleness_warning(final.get("context_pages", []))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query",     help="學生問題")
    parser.add_argument("--no-eval", action="store_true", help="跳過 eval 評分")
    args = parser.parse_args()

    print(f"[proto3] model={get_active_model()}  provider={PROVIDER}", file=sys.stderr)

    answer = run(args.query)
    print(answer)

    if not args.no_eval:
        print_score_report(answer)
