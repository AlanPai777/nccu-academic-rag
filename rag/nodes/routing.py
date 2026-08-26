"""
rag/nodes/routing.py
router_node: classifies a single (non-composite) query into PROCEDURE /
CONTACT / KNOWLEDGE via rag/router.py's 2-layer classifier. Named routing.py
(not router.py) to avoid a same-basename clash with rag/router.py itself.
"""

from __future__ import annotations

from rag.router import route
from rag.nodes.state import AgentState


def router_node(state: AgentState) -> AgentState:
    """Classify the query into PROCEDURE / CONTACT / KNOWLEDGE."""
    result = route(state["query"])
    return {
        **state,
        "query_type":   result.query_type.value,
        "route_method": result.method,
    }
