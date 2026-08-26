"""
rag/nodes/retrieval_resource.py
resource_node: RESOURCE query type (Phase F, adds the 4th branch the
original architecture always intended — see router.py docstring). Wants a
specific form/document itself ("休學申請表在哪裡下載"), not a full
辦理流程 explanation — deliberately short-circuits to a direct download
link instead of routing through anchor+expand's multi-step procedure logic.

Same anchor idea as retrieval_anchor_node (grep → get_page → extract form
IDs), but stops there and calls get_form() directly for the actual URL,
then produces a short, link-focused answer itself — no office contact
info, no multi-step synthesis prompt. Never falls back to parametric
knowledge (see synthesis_node's RESOURCE branch): a hallucinated form URL
is a much worse failure than an honest "查無此表單資訊" — this query type
is specifically about getting a real link right.
"""

from __future__ import annotations

import re

from rag.agent_tools import grep_texts, get_page, extract_form_ids, get_form
from rag.domain_router import route_domain
from rag.llm_client import simple_chat
from rag.nodes.state import AgentState

# Strip resource-phrasing (download/fetch intent) and the generic
# "表/表格/表單" noun to get the core topic for grep — e.g.
# "休學申請表在哪裡下載" → "休學申請" → close enough for grep_texts to hit
# the same overview page retrieval_anchor_node would find for "休學".
# Longest phrases listed first so re.sub prefers them over their own
# substrings (e.g. "在哪裡下載" over bare "下載") at the same match position.
_RESOURCE_STRIP_RE = re.compile(
    r'在哪裡下載|去哪裡下載|哪裡下載|在哪下載|去哪下載|'
    r'在哪裡拿|去哪裡拿|哪裡拿|在哪裡領|去哪裡領|哪裡領|'
    r'在哪裡索取|哪裡索取|下載連結|下載點|下載|'
    r'申請書|申請表|表格|表單'
)


def _extract_resource_keyword(query: str) -> str:
    """'休學申請表在哪裡下載' → '休學申請'"""
    stripped = _RESOURCE_STRIP_RE.sub('', query).strip()
    return stripped if stripped else query


_RESOURCE_SYNTHESIS_PROMPT = """\
學生問題：{query}

【搜尋到的表單資訊】
{forms_text}

請直接針對學生想要的表單回答，格式：
- 表單名稱：...
- 表單編號：...
- 下載連結：...
（若上方有多個表單，只挑跟學生問題最相關的 1-2 個列出，不要全部列出；
資料不足或找不到對應表單時，誠實說「查無此表單，請洽相關處室確認」，
絕對不可自行編造下載連結或表單編號）
"""


def resource_node(state: AgentState) -> AgentState:
    """
    grep the overview page → extract form IDs mentioned there → get_form()
    each one directly for its real URL → short link-focused synthesis.
    Produces state["answer"] directly (same pattern as retrieval_node's
    KNOWLEDGE branch) — synthesis_node's RESOURCE branch is a pass-through.
    """
    query = state["query"]
    keyword = _extract_resource_keyword(query)
    # Route on the STRIPPED keyword, not the raw query — unlike
    # retrieval_anchor_node (PROCEDURE), whose raw query text is already
    # close to the topic itself, a RESOURCE query's raw text carries
    # download/fetch-intent phrasing ("表格去哪裡拿") that isn't well
    # represented in the corpus and can pull Domain Router's Layer 2
    # aggregation toward an unrelated subdomain (confirmed 2026-08-26:
    # "復學表格去哪裡拿" routed to outgoing-iep on the raw query, aca on
    # the stripped "復學" keyword).
    subdomain = route_domain(keyword) or "aca"

    results = grep_texts(keyword, subdomain=subdomain, max_results=5)
    if not results:
        results = grep_texts(keyword, max_results=5)

    pages: list[dict] = []
    seen_urls: set[str] = set()
    for r in results:
        if r["url"] not in seen_urls:
            full = get_page(r["url"])
            if "error" not in full:
                pages.append(full)
                seen_urls.add(r["url"])

    # Scan ALL grep-matched pages, not just the first few — unlike
    # retrieval_anchor_node's anchor_pages[:3] (safe there because
    # ProcedureSkill's keyword stripping leaves specific compound terms
    # like "休學" that reliably rank the right overview page early),
    # resource_node's stripped keyword is often a single generic term
    # (e.g. "退學") whose grep results can put the actual overview page
    # further down the top-5 (confirmed 2026-08-26: ranked 5th for "退學",
    # would have been missed at [:3]). Cheap to scan all — max_results=5.
    all_text = " ".join(p.get("text", "") for p in pages)
    form_ids = extract_form_ids(all_text)

    forms: list[dict] = []
    for fid in form_ids:
        form = get_form(fid)
        if "error" not in form:
            forms.append(form)

    if not forms:
        return {
            **state,
            "context_pages": pages,
            "sources":       [p["url"] for p in pages],
            "answer":        "查無對應的官方表單，請洽相關處室確認是否需要臨櫃辦理或另有申請方式。",
        }

    forms_text = "\n\n".join(
        f"表單編號：{f['form_id']}\n標題：{f.get('form_title', '')}\n下載連結：{f['url']}"
        for f in forms
    )
    prompt = _RESOURCE_SYNTHESIS_PROMPT.format(query=query, forms_text=forms_text)
    answer = simple_chat(messages=[{"role": "user", "content": prompt}])

    return {
        **state,
        "context_pages": pages,
        "sources":       [f["url"] for f in forms],
        "answer":        answer,
    }
