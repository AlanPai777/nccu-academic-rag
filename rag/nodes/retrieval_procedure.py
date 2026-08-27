"""
rag/nodes/retrieval_procedure.py
Step 4.5 (condition 2): anchor + expand — replaces ProcedureSkill's fixed
3-step (grep → links → form) for the main single-query PROCEDURE path.
anchor is sequential/deterministic (low risk, no LLM judgment); expand fans
out via LangGraph's native Send API — one branch per cross-domain link or
form ID actually found in anchor content, not a hardcoded or LLM-guessed
count. No ToolNode/Ollama-Cloud-compatibility dependency here — Send is pure
LangGraph graph mechanics.
"""

from __future__ import annotations

from langgraph.types import Send

from rag.agent_tools import grep_texts, get_page, extract_links, get_form, extract_form_ids
from rag.domain_router import route_domain, is_ambiguous
from rag.skills.procedure_skill import _extract_keyword
from rag.agent_runtime import _offices_from_context
from rag.nodes.state import AgentState


def retrieval_anchor_node(state: AgentState) -> AgentState:
    """
    Sequential step: grep_texts + get_page to find the main page(s) — same
    idea as ProcedureSkill's old step 1, reusing its _extract_keyword()
    stripping helper rather than reimplementing it.

    Also runs office detection (condition 3, _offices_from_context) on JUST
    the anchor content, immediately — earlier than before, so
    office_lookup_node doesn't need to re-detect after expand completes
    (its own docstring covers why). And collects the cross-domain links +
    form IDs mentioned in anchor content into detected_offices/expand
    candidates for _dispatch_expand to fan out over.
    """
    query = state["query"]
    keyword = _extract_keyword(query)

    # Step 5 (condition 5): Domain Router replaces the hardcoded aca-first
    # bias — falls back to "aca" only if Domain Router itself finds nothing
    # (Layer 1 + Layer 2 both empty), not as a silent default otherwise.
    subdomain = route_domain(query) or "aca"

    # Layer 3 (CRAG-lite, condition 5): when Domain Router itself was
    # unsure which subdomain to pick (is_ambiguous — the same signal
    # Router-as-judge's gate uses), skip domain-scoping and search globally
    # instead of trusting the possibly-wrong scoped subdomain. grep_texts()
    # has no relevance ranking (plain `grep -i`, first-N-matches by glob
    # order), so this isn't a "pick the better-scored result" comparison —
    # it's a plain substitution, matching the plan's literal design
    # ("重試改成subdomain=None，查全域FTS5索引"). Confirmed empirically
    # (2026-08-27, before wiring this in) that global search still tends to
    # surface the right page for the osa-dilution case ("復學" → aca's
    # genuine page ranks #2 globally, ahead of osa's dorm-eligibility
    # pages) — a side effect of "aca" sorting alphabetically early in
    # grep_texts()'s glob, not a designed ranking mechanism, so this is a
    # v1 that could regress differently on a case where the correct
    # subdomain sorts LATER than a diluting one; not assumed to generalize
    # beyond what's been tested.
    if is_ambiguous(query):
        main_results = grep_texts(keyword, max_results=5)
    else:
        main_results = grep_texts(keyword, subdomain=subdomain, max_results=5)
        if not main_results:
            main_results = grep_texts(keyword, max_results=5)

    anchor_pages: list[dict] = []
    seen_urls: set[str] = set()
    for r in main_results:
        if r["url"] not in seen_urls:
            full = get_page(r["url"])
            if "error" not in full:
                anchor_pages.append(full)
                seen_urls.add(r["url"])

    links: list[str] = []
    for page in anchor_pages:
        for link in extract_links(page["url"]):
            if link["url"] not in seen_urls:
                links.append(link["url"])
                seen_urls.add(link["url"])

    all_text = " ".join(p.get("text", "") for p in anchor_pages[:3])
    form_ids = extract_form_ids(all_text)

    return {
        **state,
        "context_pages":    anchor_pages,
        "sources":          [p["url"] for p in anchor_pages],
        "detected_offices": _offices_from_context(anchor_pages),
        "_anchor_links":    links,
        "_anchor_form_ids": form_ids,
    }


def _dispatch_expand(state: AgentState) -> list[Send]:
    """
    Conditional-edge routing function: builds one Send per expand target
    (link or form ID found by retrieval_anchor_node) — N determined by what
    anchor actually found, not fixed or LLM-guessed. If anchor found nothing
    to expand, Send straight to office_lookup_node so the graph still
    proceeds (an empty Send list would stall the graph, not skip forward).
    """
    targets = [
        Send("retrieval_expand_node", {**state, "expand_target": {"kind": "link", "value": link}})
        for link in state.get("_anchor_links", [])
    ] + [
        Send("retrieval_expand_node", {**state, "expand_target": {"kind": "form", "value": fid}})
        for fid in state.get("_anchor_form_ids", [])
    ]
    if not targets:
        return [Send("office_lookup_node", state)]
    return targets


def retrieval_expand_node(state: AgentState) -> AgentState:
    """
    One Send-dispatched branch: fetches exactly ONE expand target (a single
    get_page(link) or get_form(form_id) call) and contributes it to
    context_pages/sources via the operator.add reducer — LangGraph merges
    all N branches' contributions (plus retrieval_anchor_node's own) once
    every branch completes, before office_lookup_node runs.

    ⚠️ Must return ONLY the fields being updated (context_pages/sources), NOT
    `**state` — N branches run concurrently in the same superstep, and
    spreading the full state means every branch also "writes" every
    unchanged field (query, query_type, ...); those are plain last-value
    channels, so N parallel writes to the same non-reducer field raises
    InvalidUpdateError ("Can receive only one value per step"). Confirmed by
    hitting exactly this error before fixing it — not a hypothetical concern.
    """
    target = state.get("expand_target", {})
    kind, value = target.get("kind"), target.get("value")

    if kind == "link":
        page = get_page(value)
        if "error" in page:
            return {"context_pages": [], "sources": []}
        return {"context_pages": [page], "sources": [page["url"]]}

    if kind == "form":
        form = get_form(value)
        if "error" in form:
            return {"context_pages": [], "sources": []}
        page = {"url": form["url"], "title": form.get("form_title", ""), "text": form.get("text", "")}
        return {"context_pages": [page], "sources": [page["url"]]}

    return {"context_pages": [], "sources": []}


def run_anchor_expand_sequential(sub_state: AgentState) -> AgentState:
    """
    Sequential-in-branch version of anchor+expand, for sub_query_node's
    PROCEDURE branch (decomposition.py) — the composite-query path's known
    follow-up (see retrieval_node's old docstring note). sub_query_node is
    ALREADY a Send-branch itself (dispatched by _dispatch_sub_queries), so
    giving it a SECOND, nested layer of Send-based parallelism for expand
    would need Send targets reachable from within an already-Send-dispatched
    node — a bigger structural change than this pass is scoped for. Mirrors
    Step 2.5's own precedent (query_decomposition shipped sequential first,
    Send-parallelized later as an explicit follow-up, not silently left
    sequential forever) — same treatment here, not a permanent design
    decision.

    Reuses _dispatch_expand()'s own target-building logic (same list of
    Send objects the real single-query graph path would fan out on) instead
    of reimplementing which links/form IDs become targets — just calls each
    target's node function directly instead of routing it through the
    graph, accumulating results with a plain list instead of LangGraph's
    operator.add reducer.
    """
    anchor_result = retrieval_anchor_node(sub_state)
    targets = _dispatch_expand(anchor_result)

    context_pages = list(anchor_result.get("context_pages", []))
    sources = list(anchor_result.get("sources", []))
    for send in targets:
        if send.node != "retrieval_expand_node":
            continue  # the "no targets" fallback Send (→ office_lookup_node) — nothing to run here
        delta = retrieval_expand_node(send.arg)
        context_pages.extend(delta.get("context_pages", []))
        sources.extend(delta.get("sources", []))

    return {
        **anchor_result,
        "context_pages": context_pages,
        "sources":       sources,
    }

    return {"context_pages": [], "sources": []}
