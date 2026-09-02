"""
server.py
FastAPI service exposing agentic_rag.py's graph over SSE to
CourseLangChain-frontend's AssistantChat.vue page (VITE_ASSISTANT_API_BASE).

docs/phase_i_assistant_api_design.md is the design record for why this
shape -- filtering stream_mode="messages" by langgraph_node instead of
CourseLangChain/app.py's "no tool_calls = final answer" heuristic -- was
needed: agentic_rag.py's graph has several nodes that call an LLM, and only
synthesis_node's output is meant to reach the user.

    python server.py
    uvicorn server:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag import build_graph, initial_state
from rag.agentic.compaction import compact_previous_turn

load_dotenv()  # rag/agentic/nodes/loop.py already does this on import; kept for direct runs

app = FastAPI()

# The assistant page is a deliberately different host from the course-
# scheduling frontend (config/assistant.ts: "不同 host，所以不走 /api 的
# dev proxy") -- CORS is required here even though CourseLangChain's app.py
# doesn't need it (that one stays same-origin behind an nginx /api proxy).
_frontend_origins = [
    o.strip()
    for o in os.environ.get("ASSISTANT_FRONTEND_ORIGIN", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# InMemorySaver must be a single instance shared across every request --
# it's the in-process store LangGraph's checkpointer reads/writes, so a
# per-request instance would never see a previous turn's state.
_graph = build_graph(checkpointer=InMemorySaver())

# SSE heartbeat interval, same rationale as CourseLangChain/app.py: the
# reverse proxy's read-timeout tracks silence between writes, not total
# request length, and inference can stay quiet for minutes at a time.
_HEARTBEAT_SEC = 15
_STREAM_DONE = object()  # sentinel distinct from a legitimate None chunk

# node_name -> static status phrase, for nodes with no natural-language
# output of their own to reuse (see _status_text below for the nodes that do).
_STATIC_STATUS = {
    "plan_node": "正在判斷問題類型…",
    "agent_node": "正在決定下一步…",
    "tools": "正在查詢政大官方資料…",
    "resource_node": "正在讀取表單…",
    "contact_node": "正在查詢辦公室聯絡資訊…",
    "multi_sub_query_node": "正在拆解問題…",
}


def _status_text(node_name: str, delta: dict) -> str | None:
    """One node update -> one human-readable status line.

    Prefers text the node already generated for its own purposes over a
    static label (phase_g_brainstorming.txt's "one LLM output, two
    consumers" idea) -- rewrite_node's `rewritten` and domain_router_node/
    self_eval_node's injected messages are natural-language LLM output
    already, no separate summarizer call needed.
    """
    if node_name == "rewrite_node" and delta.get("rewritten"):
        return f"正在搜尋：{delta['rewritten']}"
    if node_name in ("domain_router_node", "self_eval_node"):
        msgs = delta.get("messages") or []
        if msgs:
            return str(msgs[-1].content).split("\n")[0]
    return _STATIC_STATUS.get(node_name)


def _thread_config(session_id: str | None) -> dict:
    # No session_id -> one-off id, same "ephemeral = no memory" fallback
    # CourseLangChain/main.py's _base_config uses (a checkpointer-backed
    # graph raises without SOME thread_id).
    return {"configurable": {"thread_id": session_id or f"ephemeral-{os.urandom(8).hex()}"}}


async def _pump(
    question: str, subdomain_hint: str | None, session_id: str | None, config: dict, queue: asyncio.Queue
) -> None:
    """Run one turn, push every SSE-worthy event into the queue.

    Exceptions are queued rather than handled here -- the SSE consumer
    (_generate) is the one place that knows how to format them for the
    frontend's `error` field, matching CourseLangChain/app.py's split.
    """
    try:
        if session_id:
            # Collapse the PREVIOUS turn's raw scratch before this new one
            # starts -- an ephemeral (session_id-less) thread is always
            # fresh, so skip the wasted checkpoint read for those.
            compact_previous_turn(_graph, config)
        state = initial_state(question, subdomain_hint)
        async for stream_mode, payload in _graph.astream(
            state, config=config, stream_mode=["updates", "messages"]
        ):
            if stream_mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") == "synthesis_node" and chunk.content:
                    await queue.put({"data": chunk.content})
            elif stream_mode == "updates":
                for node_name, delta in payload.items():
                    text = _status_text(node_name, delta or {})
                    if text:
                        await queue.put({"type": "status", "text": text})
    except Exception as e:  # noqa: BLE001 -- formatted for the user by _generate
        await queue.put(e)
    finally:
        await queue.put(_STREAM_DONE)


async def _generate(question: str, subdomain_hint: str | None, session_id: str | None):
    config = _thread_config(session_id)
    queue: asyncio.Queue = asyncio.Queue()
    pump = asyncio.create_task(_pump(question, subdomain_hint, session_id, config, queue))

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SEC)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if item is _STREAM_DONE:
                break
            if isinstance(item, Exception):
                msg = f"抱歉，系統發生錯誤，暫時無法處理您的要求。（{type(item).__name__}）"
                yield f"data: {json.dumps({'data': msg, 'error': str(item)})}\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"
    finally:
        # A closed SSE connection (tab closed mid-turn) should stop the
        # graph too, not keep burning turns nobody is listening to.
        pump.cancel()

    yield f"data: {json.dumps({'data': 'SPECIAL_END_TOKEN'})}\n\n"


@app.get("/ask")
async def ask(question: str, session_id: str | None = None, subdomain: str | None = None):
    """SSE chat endpoint. `session_id` ties multi-turn context to a
    LangGraph thread_id; omit it for a one-off, memory-less question."""
    return StreamingResponse(
        _generate(question, subdomain, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
