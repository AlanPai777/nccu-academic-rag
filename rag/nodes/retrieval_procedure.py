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
from rag.domain_router import route_domain
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
