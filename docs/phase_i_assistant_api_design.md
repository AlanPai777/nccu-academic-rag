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

## Implementation finding: self_eval retries can duplicate the answer in the frontend, and the fix for it needed a second pass

Once streaming worked, a remaining design gap surfaced during discussion (before it was ever seen live): `self_eval_node` can reject `synthesis_node`'s draft and route back through `rewrite_node`/`plan_node` for another pass -- meaning `synthesis_node` can run more than once within one `/ask` turn. `server.py`'s SSE filter only checked `langgraph_node == "synthesis_node"`, with no way to tell "this turn's rejected draft" apart from "the eventual accepted answer" -- both would stream into the same frontend `output` buffer back to back, with no separator.

**Design choice**: rather than buffering server-side until `self_eval_node` judges the draft (which would kill live typing for *every* turn, not just retried ones -- you don't know a run is "the accepted one" until after it fully finishes generating and gets judged), the frontend keeps streaming live throughout, and a `{"type":"reset"}` event tells it to file the just-typed draft into the thinking-log (`statusLog`) and clear the answer area before the replacement starts typing. The draft isn't lost, just relocated -- and this needs zero server-side buffering, since the frontend already has the full draft text on hand from the live stream by the time reset fires.

**Bug found while implementing the trigger condition**: the natural-seeming detection ("did `self_eval_node`'s delta include a `messages` entry?") misses one of self_eval's three failure branches. `self_eval.py`'s three failure paths:
- Stage 1 checklist failure → sets both `self_eval_note` and `messages`
- 情況A (content incomplete) → sets both `self_eval_note` and `messages`
- 情況B (compound query possibly misclassified, routes back to `plan_node`) → sets **only** `self_eval_note`, no `messages` at all -- because `plan_node` never reads `messages` (confirmed earlier, see Verified code facts above), so there was never a reason for that branch to write one

A `messages`-keyed detector silently misses 情況B: the retry happens, but no reset fires, and the exact same duplication bug reappears through a different branch. Fix: `server.py`'s `_is_retry_signal()` keys off `self_eval_note` being truthy instead -- the one field all three branches set, since it's what the branches actually mean ("a retry is happening"), not `messages`'s presence, which was only ever a side effect of what each branch's *downstream reader* needed (`rewrite_node` reads `messages`; `plan_node` doesn't). Verified with a standalone test feeding all three branches' real `self_eval.py` return shapes into `_is_retry_signal`/`_status_text` directly (no live LLM call needed) -- 情況B correctly triggers `retry=True` after the fix.

## Implementation finding: a second, unrelated duplication bug — `stream_mode="messages"` surfaces state-added messages, not just live tokens

Discovered live-testing "如何辦理休學" after the self_eval-retry fix: the answer still arrived duplicated (same content twice, concatenated with no separator), but with **0 reset events** and only one pass through every node (confirmed via the `updates` stream's status-event count) -- ruling out self_eval retries as the cause entirely. This is a distinct bug from the one above.

**First (wrong) hypothesis**: `synthesis_node`'s `_llm` is the same object `agent_node` uses (via `loop.py`), just with a different `.bind()` view (`.bind(options=...)` vs `.bind_tools(...)`) -- suspected LangGraph's per-node stream tagging was getting confused by two different bound views of one shared instance. Tried giving `synthesis_node` an isolated `ChatOllama` instance; **did not fix it** (confirmed via the same live repro, still duplicated) -- so this was reverted back to sharing `loop.py`'s `_llm` (confirmed via testing, not assumed, that the isolation was never actually the fix).

**Actual root cause**, found by inspecting the raw SSE chunk sizes: 1282 small chunks (1-3 chars each, genuine token deltas) followed by **one final 1767-char chunk containing the entire answer again**. `stream_mode="messages"` doesn't only stream live LLM token deltas -- it also surfaces any *complete* message added to `state["messages"]` via a node's return value, since from LangGraph's perspective that's indistinguishable from "a chat model produced this." `synthesis_node`'s Item E fix (`return {"answer": answer, "messages": [AIMessage(content=answer)]}`, added earlier so the next turn can read the prior conclusion) means every synthesis run's answer gets added to `messages` as a complete `AIMessage` -- and that addition itself gets surfaced as one more `"messages"`-mode event, layered on top of the genuine incremental deltas that already streamed the same text.

Confirmed via a direct (non-graph) `ChatOllama.stream()` test that the model's own streaming terminates cleanly (final metadata-carrying chunk has empty `.content`) -- the duplication is specific to LangGraph's graph-level `"messages"` tap picking up the `AIMessage` add, not a `langchain_ollama` bug.

Fix: `server.py`'s filter now also checks `isinstance(chunk, AIMessageChunk)` (from `langchain_core.messages`) before forwarding -- genuine incremental deltas are always `AIMessageChunk`; a complete message added via a node's return value is a plain `AIMessage`, which now gets filtered out of the answer channel entirely. Verified via the same live repro (0 duplication after the fix, `第一步`/`第二步`/`第三步`/`來源` each now appear exactly once) and a full eval-scored CLI regression (26/26, unaffected).

**This does not affect short-term memory.** The `isinstance` filter only changes what `server.py` forwards to the SSE queue (i.e. what the frontend hears) -- it has no effect on `synthesis_node`'s actual return value or on what gets written to the checkpointer. The `AIMessage(content=answer)` still gets added to `state["messages"]` exactly as Item E designed, still gets read by `compact_previous_turn()` when building next turn's compacted `[HumanMessage(prev_query), AIMessage(prev_answer)]` pair. The two concerns -- "what gets persisted for cross-turn memory" and "what gets streamed live to the current viewer" -- turned out to be genuinely separate, and this bug was purely in the second one.

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
