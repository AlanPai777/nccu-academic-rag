"""
rag/agentic/logic/office_detection.py
_detect_offices(): office-name detection against the full ~433-entry
catalog (Migration Step 3, docs/phase_h_agentic_rag_migration_plan.md
Part 5). Ported directly from rag/agentic_main.py's own _detect_offices()
(D12's LLM-judged redesign + N.7's Layer 1 substring pre-check) -- pulled
forward from its originally-planned Step 4 home because
rag/agentic/tools/page.py's get_page_tool needs it directly (a page with
no form IDs runs office detection on its own text, per §M D4's
established sequencing). Step 4's contact_node/resource_node will import
this same function, not a second copy.

Pure logic, no LangGraph state dependency -- takes text, returns a list
of office names. Independently testable (this session repeatedly used
throwaway scripts to sample-test this exact function's reliability; being
a plain function makes that natural rather than requiring a fake
AgentState).
"""

from __future__ import annotations

import re

from rag.llm_client import simple_chat
from rag.domain_router import _keyword_table

_OFFICE_JUDGE_PROMPT = """以下文字裡可能提到了一些政大的辦公室/單位，但提到的名稱常常是簡稱或口語說法，不一定是正式全名（例如文字裡寫「住宿組」，正式全名可能是「住宿輔導組」；文字裡寫「圖書館」，正式全名可能是「圖書館(各組聯絡資訊)」）。

文字內容：
{text}

已知的政大辦公室/單位清單（正式名稱，格式為 名稱（所屬subdomain codebase）——這份清單本來就存在，不是搜尋結果）：
{catalog}

請判斷：上面清單裡，哪些辦公室有在文字內容中被提到（用簡稱、口語說法、或全名都算，只要語意上是指同一個單位就算）？只回傳清單裡列出的正式名稱，用逗號分隔；如果都沒提到，回傳「無」。不要自己發明清單以外的名稱，不要有其他文字。"""


def _detect_offices(text: str) -> list[str]:
    """LLM-judged semantic match against the full ~433-entry office catalog
    (domain_router.py's _keyword_table()). A form/page refers to offices
    colloquially ("住宿組") while the catalog carries official full names
    ("住宿輔導組") -- these share no contiguous substring, so no fixed-
    string match alone would ever catch it; "does this text refer to the
    same office as this catalog entry" is a semantic judgment, LLM
    territory. Catalog is NOT subdomain-scoped: a single form/page can
    reference offices across multiple subdomains, so narrowing to one
    subdomain here would silently drop cross-subdomain office mentions.

    Layer 1 (deterministic substring pre-check) runs BEFORE the LLM judge
    and its hits are UNIONED into the result, not used to skip the LLM
    call. Added after finding _detect_offices() fails ~93% of the time
    (14/15 sampled calls) on a SHORT bare query (e.g. contact_node's
    direct Plan_node-CONTACT-route entry, "出納組電話幾號") even though the
    office name is an EXACT literal substring of that query and of exactly
    one catalog entry ("出納組") -- confirmed the same LLM judge reliably
    finds "出納組" when given a full page of text instead, so this is a
    short-input-specific failure mode of the LLM step, not a general
    catalog-matching failure. Layer 1 sidesteps it for free (pure string
    containment, ~433 checks, negligible cost) without touching the LLM
    judge at all -- it does NOT replace it, since Layer 1 still can't
    resolve colloquial/short-vs-full-name mismatches ("住宿組" ->
    "住宿輔導組"), which remains the LLM judge's job below."""
    catalog = _keyword_table()
    if not catalog:
        return []
    valid_names = {name for name, _sub in catalog}
    layer1_hits = [name for name in valid_names if name in text]

    catalog_str = "\n".join(f"- {name}（{sub}）" for name, sub in catalog)
    prompt = _OFFICE_JUDGE_PROMPT.format(text=text, catalog=catalog_str)
    raw = simple_chat(messages=[{"role": "user", "content": prompt}], max_tokens=300).strip()
    candidates = [n.strip() for n in re.split(r"[,，]", raw)]
    llm_hits = [n for n in candidates if n in valid_names]

    return list(dict.fromkeys(layer1_hits + llm_hits))
