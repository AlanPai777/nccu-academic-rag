"""
rag/skills/procedure_skill.py
Deterministic multi-step retrieval for NCCU procedure-type questions.

Design principles:
- Deterministic (LLM doesn't decide each step) → high stability
- LLM only synthesizes the final answer
- Domain knowledge encoded: procedure questions always need
  main page + cross-domain linked pages + moltke form
"""

import re

from rag.agent_tools import (
    grep_texts, get_page, extract_links,
    get_children, get_form, extract_form_ids
)
from rag.domain_router import route_domain

# Strip question-phrasing words to get the core topic noun for grep
_STRIP_RE = re.compile(
    r'如何辦理|如何申請|怎麼辦理|怎麼申請|如何|怎麼|辦理程序|申請流程|辦理手續|申請步驟|步驟|流程|程序|辦理'
)


def _extract_keyword(query: str) -> str:
    """'如何辦理休學' → '休學'"""
    stripped = _STRIP_RE.sub('', query).strip()
    return stripped if stripped else query


class ProcedureSkill:
    """
    Usage:
        skill = ProcedureSkill()
        context = skill.run("如何辦理休學")
        # context contains all relevant page texts, pass to LLM for synthesis
    """

    def run(self, query: str) -> dict:
        context_pages = []
        seen_urls: set[str] = set()

        # Step 1: Find main relevant pages (top-3)
        # Use stripped keyword so "如何辦理休學" → grep "休學"
        # Step 5 (condition 5): Domain Router picks the subdomain instead of
        # a hardcoded aca-first guess; falls back to aca, then to an
        # unscoped global search, if Domain Router finds nothing.
        keyword = _extract_keyword(query)
        subdomain = route_domain(query) or "aca"
        main_results = grep_texts(keyword, subdomain=subdomain, max_results=5)
        if not main_results:
            main_results = grep_texts(keyword, max_results=5)
        if not main_results:
            return {"error": "No relevant pages found", "context": []}

        for r in main_results:
            if r["url"] not in seen_urls:
                full = get_page(r["url"])
                if "error" not in full:
                    context_pages.append(full)
                    seen_urls.add(r["url"])

        # Step 2: Follow all cross-subdomain links from main pages
        # Example: aca main page → [退費標準](osa_url)
        for main_page in context_pages[:]:
            links = extract_links(main_page["url"])
            for link in links:
                if link["url"] not in seen_urls:
                    linked = get_page(link["url"])
                    if "error" not in linked:
                        context_pages.append(linked)
                        seen_urls.add(link["url"])

        # Step 3: Extract form IDs from main text and fetch full forms
        # Example: main page mentions QP-T01-03-02 → fetch moltke form (7-stop chain)
        all_main_text = " ".join(p.get("text", "") for p in context_pages[:3])
        form_ids = extract_form_ids(all_main_text)
        forms = []
        for fid in form_ids:
            form = get_form(fid)
            if "error" not in form:
                forms.append(form)
                if form["url"] not in seen_urls:
                    context_pages.append({
                        "url":   form["url"],
                        "title": form["form_title"],
                        "text":  form["text"],
                    })
                    seen_urls.add(form["url"])

        return {
            "context":     context_pages,
            "forms":       forms,
            "source_urls": sorted(seen_urls),
        }


if __name__ == "__main__":
    skill = ProcedureSkill()
    result = skill.run("如何辦理休學")

    print(f"Pages found: {len(result['context'])}")
    print(f"Forms found: {len(result['forms'])}")
    print(f"Source URLs:")
    for url in result["source_urls"]:
        print(f"  {url}")

    print("\n--- Context preview ---")
    for page in result["context"]:
        print(f"\n[{page.get('title','?')}] {page.get('url','?')[:60]}")
        print(f"  {len(page.get('text',''))} chars")
        print(f"  {page.get('text','')[:150]}...")
