# Phase I: Assistant API Server — Design Record

2026-09-02. This document records the design discussion that led to
`server.py` (the FastAPI service exposing `agentic_rag.py` to the
`CourseLangChain-frontend`'s "政大 AI 助手" page), and the specific risks
flagged during that discussion for future implementers to watch for.

## Why this exists

`CourseLangChain-frontend` already ships a page (`AssistantApp.vue` /
`AssistantChat.vue`) whose SSE call contract is hard-coded in
`src/composables/useChat.ts` and `src/config/assistant.ts`
(`VITE_ASSISTANT_API_BASE` / `ASSISTANT_ASK_URL` / `ASSISTANT_UPLOAD_URL`), but
`VITE_ASSISTANT_API_BASE` has never been filled in — there was no backend to
point it at. This repo's `agentic_rag.py` is that backend; this phase adds the
HTTP service layer.

## Reference implementation, and why it can't be copied wholesale

The sibling repo `CourseLangChain` (the course-scheduling backend) already
implements the *same* SSE contract (`app.py`, `main.py`,
`agents/brain_agent.py`) — same payload shapes
(`{"data":...}` / `{"type":"status",...}` / `error` / `SPECIAL_END_TOKEN`),
same `session_id` → `thread_id` mapping, same heartbeat-under-SSE pattern.
Worth reading before touching `server.py`.

The one thing that does **not** transfer directly: `CourseLangChain`'s agent
is a single `create_react_agent` ReAct loop, so "the last LLM message with no
`tool_calls`" unambiguously means "the final answer." `agentic_rag.py` is a
multi-node `StateGraph` (`plan_node → rewrite_node → domain_router_node →
agent_node ⇄ tools → resource_node/contact_node → synthesis_node →
self_eval_node`) — several nodes call an LLM, and only `synthesis_node`'s
output is ever meant to reach the user. Reusing CourseLangChain's
"no tool_calls = done" filter here would leak `plan_node`'s classification or
`self_eval_node`'s internal judgment straight into the chat bubble. See
Decision B below for how this repo's `server.py` actually identifies the real
answer.

## Decisions

### B — Streaming: `stream_mode="messages"`, filtered by `langgraph_node`

`server.py` streams token-by-token (matching `Data_Cool/UI-10-Answer-
generation-in-progress.png`'s reference design — the answer visibly types in,
not delivered as one block) via
`graph.astream(input, config, stream_mode=["updates", "messages"])`.

- The `"messages"` stream yields `(chunk, metadata)`; only chunks where
  `metadata["langgraph_node"] == "synthesis_node"` become
  `{"data": chunk.content}` SSE events. Every other node's LLM output is
  discarded from the visible-answer channel entirely — this is the concrete
  mechanism that replaces CourseLangChain's "no tool_calls" heuristic.
- The `"updates"` stream (the same mode `agentic_rag.py --stream` already
  used for its debug printout) drives the `{"type":"status",...}` side
  channel — see Decision D.

### C — Cross-turn message growth: compact by turn boundary, not by count

**The risk isn't just cost, it's answer quality.** `synthesis_node`'s
`_render_full_messages()` renders the *entire* `messages` list with no
scoping — so if turn 1 was "如何辦理休學" and turn 2 is "圖書館幾點開",
turn 2's synthesis call would still see turn 1's fetched page/form/office
text verbatim. A blind "keep last N messages" trim (CourseLangChain's
`pre_model_hook` approach) only delays this, it doesn't remove the risk,
because the wrong-topic content is still inside whatever window survives.

The fix implemented: before starting a new turn on an existing `thread_id`,
`rag/agentic/compaction.py`'s `compact_previous_turn()` reads the previous
turn's checkpoint and replaces its entire `messages` list with just
`[HumanMessage(prev_query), AIMessage(prev_answer)]`, using LangGraph's
`RemoveMessage` to actually delete the old scratch (search candidates, page/
form full text, office rosters) from the checkpointer — not just hide it from
one LLM call.

This is safe to do *because* `_SYNTHESIS_PROMPT` rule 6 already requires the
answer to end with a `來源：` URL citation (verified producing real `來源：`
lines in the 2026-09-02 regression run — not just a prompt promise). The
compacted answer is therefore already "conclusion + source" in one string; no
separate sources field was designed, on purpose — it would just be another
field that needs "don't let the caller blindly overwrite it" handling (see
the `subdomain_hint` gotcha below).

**Soft dependency to watch**: this design's reliability rides on
`_SYNTHESIS_PROMPT` rule 6's citation format staying reasonably consistent
(`來源：` prefix). If a future prompt change lets citations drift into loose
prose, a later turn trying to re-fetch "the page we found last time" from the
compacted answer text has a harder time locating the URL. If that starts
happening, fix rule 6's wording — don't rebuild the compaction mechanism.

### D — Thinking-log UI, inspired by `Data_Cool`'s reference screenshots

`Data_Cool/UI-2` through `UI-11` (claude.ai's own extended-thinking panel)
show: a collapsed header with *the current* human-readable phrase + elapsed
time; expanded, a **full accumulating list** of every past phrase (not
overwritten); collapses to "Thought for Xm Ys" once done.

`CourseLangChain-frontend`'s `useChat.ts` currently overwrites `statusText`
on every `type:"status"` event — there is no accumulation. Getting the
`Data_Cool` effect needs an additive frontend change: a `statusLog` array
that every status event pushes into, alongside (not replacing) the existing
`statusText`.

The narration text itself doesn't need a dedicated summarizer LLM call —
`phase_g_brainstorming.txt` point 6 anticipated this exact need ("可以存
plan 的輸出，兩種輸出，一個給UI，一個給程式碼的"): `rewrite_node`'s
`rewritten`, `domain_router_node`'s candidate message, and
`self_eval_node`'s `reminder` are already natural-language LLM output that
existing nodes produce for their own internal purposes — `server.py`'s status
side-channel just reuses them as-is.

**Scope boundary** (explicit instruction): backend work stays inside
`nccu-academic-rag` only. Frontend work, if any, touches only
`AssistantChat.vue` / `useChat.ts` — the latter is shared with
`Chat.vue` (course scheduling), so the change must be strictly additive
(new field, existing `statusText` untouched) so the scheduling side is
unaffected. `CourseLangChain`'s backend is not touched at all.

### E — `synthesis_node` also appends its answer to `messages`

Before this fix, `synthesis_node` wrote only to `state["answer"]` — a
non-accumulating field the caller resets to `None` every turn. No node reads
`state["answer"]` directly (confirmed by reading `rag/agentic/nodes/loop.py`),
so even leaving it unreset wouldn't have helped; the *actual* conclusion was
never reachable through `messages`, which is what every node's context-
building actually reads.

Fix: `synthesis_node` now returns
`{"answer": answer, "messages": [AIMessage(content=answer)]}`. Verified safe
against the two things that scan `messages` structurally:
`_after_tools`' marker regex only fires on `isinstance(last, ToolMessage)`
(an `AIMessage` never matches), and `synthesis_node`'s own
`_render_full_messages()` only renders `ToolMessage`/`HumanMessage` (so a
retry within the same turn won't cite its own prior draft answer as if it
were newly-found source content).

### F — Voice-to-text: confirmed there is no reference implementation anywhere

Checked `CourseLangChain/pyproject.toml` + `uv.lock` (no STT dependency),
`CourseLangChain/app.py` (no `/api/upload` route), and
`CourseLangChain-frontend/src/App.vue` (the course-scheduling chat page — no
mic/upload/audio code at all). The mic button only exists on
`AssistantApp.vue`, calling an endpoint that has never been implemented on
either backend. This phase does not include it; it's unscoped future work,
not a gap in this design.

## Verified code facts (read directly, not assumed)

- `agentic_rag.py`: `build_graph()`/`g.compile()` had no checkpointer;
  `run()` rebuilt a full `initial` state dict on every call.
- `rag/agentic/state.py`: only `messages` has a reducer (`add_messages`,
  accumulates); the other 8 fields are plain overwrite-on-write — whatever
  the caller passes replaces the checkpointed value, full stop.
- `rag/agentic/nodes/plan.py`: `plan_node` classifies off `state["query"]`
  **only** — it never reads `messages`. This rules out "old messages
  confuse the classifier," but flips the risk the other way: a short
  follow-up question ("那在職生呢?") with no context of its own may
  classify worse than a self-contained question would. Worth testing with
  real follow-ups once multi-turn is live.
- `rag/agentic/nodes/loop.py`: `rewrite_node`/`domain_router_node` inject
  synthetic `HumanMessage`s that are system narration, not literal user
  input. `agent_node`'s `SystemMessage` is prepended locally per call and
  never persisted to the checkpoint.
- `rag/agentic/nodes/resource.py` / `contact.py`: both append plain
  `HumanMessage`s — no other message types to special-case.
- `requirements.txt` had only `langgraph` — `fastapi`/`uvicorn` were added
  for this phase.

## Implementation finding: `synthesis_node` needed a second change to actually stream

Decision B's `stream_mode="messages"` + `langgraph_node` filter was implemented in
`server.py` exactly as designed, and manual SSE testing confirmed the status
side-channel worked correctly -- but the final answer still arrived as **one
whole `data` event**, not token-by-token, on the first end-to-end test.

Root cause: `synthesis_node` called `rag/llm_client.py`'s `simple_chat()`,
which drives the raw `ollama.Client`/`openai.OpenAI` SDKs directly, never
touching LangChain's `BaseChatModel` interface. LangGraph's
`stream_mode="messages"` token streaming works by intercepting a LangChain
chat model's own streaming callback -- a node whose LLM call never goes
through that interface still shows up in the `"messages"` stream (tagged
with the right `langgraph_node`), but as a single complete message, not
incremental chunks, because there were never any chunks to intercept.

Fix: `synthesis_node` now calls `rag/agentic/nodes/loop.py`'s `_llm` (a
`ChatOllama` instance, already proven in production via `agent_node`) instead
of `simple_chat()`, with `.bind(options={"num_predict": 8192})` to preserve
`simple_chat()`'s original truncation-avoidance margin (long synthesis
answers run 1000-2000+ tokens; Ollama's cloud API rejects `num_predict=-1`,
so an explicit positive cap is required, same reasoning `llm_client.py`'s own
comment gives). Verified via curl against a running `server.py`: the answer
arrived as 147 separate small `data` events instead of one.

**Scope note**: only `synthesis_node` was switched. `self_eval_node` and
`rewrite_node`'s `_rewrite_query()` still use `simple_chat()` — their LLM
output is only ever consumed as a single status-line string (never streamed
character-by-character to the user), so there was no reason to touch them.
`agent_node` already only supports Ollama cloud regardless of `LLM_PROVIDER`
(see `loop.py`'s `_llm` construction) — reusing that same instance in
`synthesis_node` follows an existing pattern in this codebase, not a new
inconsistency introduced by this phase.

## Known gotcha carried over from earlier discussion (applies beyond this phase)

Any caller that reconstructs the full `initial` state dict on every turn (as
`run()` does) must be careful not to pass a default value for a field it
doesn't actually intend to reset — `subdomain_hint` is the concrete example:
`domain_router_node` no-ops once it's set, but if the caller keeps passing
`subdomain_hint: None` every turn (the CLI's default when `--subdomain` is
omitted), that silently overwrites the checkpointed value back to `None` and
defeats the no-op guard. `server.py`'s per-turn input construction needs to
omit fields it isn't intentionally resetting, not mirror `run()`'s literal
dict.
