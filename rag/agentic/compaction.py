"""
rag/agentic/compaction.py
compact_previous_turn() -- collapses a closed conversation turn's raw
scratch work down to a clean [question, answer] pair before a new turn
starts on the same thread_id. Design record: docs/phase_i_assistant_api_design.md
(Decision C).

Why this exists: synthesis_node's _render_full_messages() renders the
*entire* messages list with no per-turn scoping, so an uncompacted
multi-turn conversation would leak a previous turn's fetched page/form/
office text into an unrelated new topic's synthesis prompt -- a correctness
risk, not just a growing-cost one. Trimming by message count (à la
CourseLangChain/agents/brain_agent.py's pre_model_hook) doesn't remove this
risk, only delays it, since stale cross-topic content can still be inside
whatever window survives.

Safe to reduce a turn down to just its answer because _SYNTHESIS_PROMPT
(rag/agentic/nodes/synthesis.py) rule 6 already requires the answer to end
with a 來源：URL citation -- verified producing real 來源：lines in the
2026-09-02 regression run, not just a prompt promise. The compacted answer
is therefore already "conclusion + source" in one string.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage


def compact_previous_turn(graph, config: dict) -> None:
    """Call once, right before starting a new turn on an existing
    thread_id (config["configurable"]["thread_id"]) -- never mid-turn, since
    this assumes the previous turn already reached synthesis_node.

    No-ops on a brand-new thread_id (nothing to compact yet) and on a
    previous turn that never produced an answer (e.g. it errored before
    reaching synthesis_node) -- compacting to a [query, None] pair would
    just replace real scratch with a useless placeholder.
    """
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    prev_messages = values.get("messages") or []
    prev_query = values.get("query")
    prev_answer = values.get("answer")
    if not prev_messages or not prev_query or not prev_answer:
        return

    removals = [RemoveMessage(id=m.id) for m in prev_messages if getattr(m, "id", None)]
    compacted = [HumanMessage(content=prev_query), AIMessage(content=prev_answer)]
    graph.update_state(config, {"messages": removals + compacted})
