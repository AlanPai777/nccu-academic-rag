"""
rag/agent_tools.py
Five core tools that read directly from NCCU-Crawler output.
No Qdrant, no embedding — follows ELITE + LongRAG spirit:
  - whole-page retrieval (no chunking)
  - grep full-text search (no embedding)
"""

import glob
import json
import re
import subprocess
import urllib.parse
from pathlib import Path

CRAWLER_OUTPUT = str(Path(__file__).parent.parent / "output")
NCCU_LINK_RE   = re.compile(r'\[([^\]]+)\]\((https?://[^)]*\.nccu\.edu\.tw[^)]*)\)')
FORM_ID_RE     = re.compile(r'QP-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+')

# Group names for multi-subdomain search (grep_texts group= parameter).
SUBDOMAIN_GROUPS: dict[str, list[str]] = {
    "nccuga": ["nccuga", "cashier", "dean", "docu", "aff",
               "police", "wealth", "mend", "environ", "cpds"],
    "osa":    ["osa"],
    "aca":    ["aca"],
    "oic":    ["oic"],
    "lib":    ["www.lib"],
    "www":    ["www"],
    "learning": ["learning"],
}


# ── Tool 1: grep_texts ────────────────────────────────────────────────────────

def grep_texts(
    pattern: str,
    subdomain: str | None = None,
    group: str | None = None,
    max_results: int = 5
) -> list[dict]:
    """
    Full-text search across extracted_texts.jsonl files.
    subdomain: restrict to one subdomain (e.g. "aca", "cashier")
    group: restrict to a named group defined in SUBDOMAIN_GROUPS
           (e.g. "nccuga" searches all 10 general-affairs subdomains)
    Returns [{url, title, preview, subdomain}, ...]
    """
    if subdomain:
        paths = [f"{CRAWLER_OUTPUT}/{subdomain}/extracted_texts.jsonl"]
    elif group:
        subdomains = SUBDOMAIN_GROUPS.get(group, [])
        paths = [f"{CRAWLER_OUTPUT}/{s}/extracted_texts.jsonl" for s in subdomains]
    else:
        paths = glob.glob(f"{CRAWLER_OUTPUT}/*/extracted_texts.jsonl")

    results = []
    for path in paths:
        if not Path(path).exists():
            continue
        try:
            proc = subprocess.run(
                ["grep", "-i", pattern, path],
                capture_output=True, text=True, timeout=15
            )
            sub = path.split("/")[-2]
            for line in proc.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    results.append({
                        "url":       d["url"],
                        "title":     d.get("title", ""),
                        "preview":   d.get("text", "")[:400],
                        "subdomain": sub,
                    })
                except json.JSONDecodeError:
                    pass
        except subprocess.TimeoutExpired:
            pass

    return results[:max_results]


# ── Tool 2: get_page ──────────────────────────────────────────────────────────

def get_page(url: str) -> dict:
    """
    Fetch full page text for a specific URL (whole page, not chunked).
    Falls back to prefix matching when the stored URL is a child of the input URL.
    Example: input '/休退學' matches stored '/休退學暨研究生畢業退費/休-退-學退費標準'.
    Returns {url, title, text, subdomain}
    """
    decoded = urllib.parse.unquote(url)

    for path in glob.glob(f"{CRAWLER_OUTPUT}/*/extracted_texts.jsonl"):
        if not Path(path).exists():
            continue
        try:
            proc = subprocess.run(
                ["grep", "-F", decoded, path],
                capture_output=True, text=True, timeout=10
            )
            sub = path.split("/")[-2]
            prefix_hit = None
            for line in proc.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    stored = d.get("url", "")
                    # Exact match wins immediately
                    if stored == url or stored == decoded:
                        return {
                            "url":       stored,
                            "title":     d.get("title", ""),
                            "text":      d.get("text", ""),
                            "subdomain": sub,
                        }
                    # Prefix match: stored URL is a child of the input URL
                    if stored.startswith(decoded) and prefix_hit is None:
                        prefix_hit = (d, sub)
                except json.JSONDecodeError:
                    pass
            if prefix_hit:
                d, sub = prefix_hit
                return {
                    "url":       d["url"],
                    "title":     d.get("title", ""),
                    "text":      d.get("text", ""),
                    "subdomain": sub,
                }
        except subprocess.TimeoutExpired:
            pass
    return {"error": f"URL not found: {url}"}


# ── Tool 3: extract_links ─────────────────────────────────────────────────────

def extract_links(url: str) -> list[dict]:
    """
    Extract all markdown links to *.nccu.edu.tw from a page's text.
    Critical for following aca→osa cross-domain references (e.g. refund page).
    Returns [{label, url}, ...]
    """
    page = get_page(url)
    if "error" in page:
        return []

    text = page.get("text", "")
    matches = NCCU_LINK_RE.findall(text)

    seen: set[str] = set()
    result = []
    for label, link_url in matches:
        if link_url not in seen:
            seen.add(link_url)
            result.append({"label": label, "url": link_url})
    return result


# ── Tool 4: get_children ──────────────────────────────────────────────────────

def get_children(
    url: str,
    subdomain: str | None = None
) -> list[dict]:
    """
    BFS tree navigation — find pages discovered FROM a given URL.
    Use case: drill down from aca root → 服務項目/學籍 category.
    Returns [{url, type, category, child_count, depth}, ...]
    """
    if subdomain:
        map_paths = [f"{CRAWLER_OUTPUT}/{subdomain}/map.json"]
    else:
        map_paths = glob.glob(f"{CRAWLER_OUTPUT}/*/map.json")

    results = []
    for path in map_paths:
        if not Path(path).exists():
            continue
        try:
            data = json.loads(Path(path).read_text())
            for r in data:
                if r.get("parent") == url:
                    results.append({
                        "url":         r["url"],
                        "type":        r.get("type", ""),
                        "category":    r.get("category", ""),
                        "child_count": r.get("child_count", 0),
                        "depth":       r.get("depth", 0),
                    })
        except (json.JSONDecodeError, OSError):
            pass
    return results


# ── Tool 5: get_form ──────────────────────────────────────────────────────────

def get_form(form_id: str) -> dict:
    """
    Look up a moltke form by ID, return metadata + extracted text.
    form_id format: QP-T01-03-02
    Returns {form_id, form_title, form_unit, url, text}
    """
    for path in glob.glob(f"{CRAWLER_OUTPUT}/*/supplementary_map.json"):
        if not Path(path).exists():
            continue
        try:
            data = json.loads(Path(path).read_text())
            for r in data:
                if r.get("form_id") == form_id:
                    form_text = get_page(r["url"])
                    return {
                        "form_id":          r["form_id"],
                        "form_title":       r.get("form_title", ""),
                        "form_description": r.get("form_description", ""),
                        "form_unit":        r.get("form_unit", ""),
                        "url":              r["url"],
                        "text":             form_text.get("text", ""),
                    }
        except (json.JSONDecodeError, OSError):
            pass
    return {"error": f"Form not found: {form_id}"}


# ── Helper ────────────────────────────────────────────────────────────────────

def extract_form_ids(text: str) -> list[str]:
    """Extract all moltke form IDs from page text."""
    return list(set(FORM_ID_RE.findall(text)))


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== agent_tools self-test ===\n")

    t0 = time.time()
    results = grep_texts("休學", subdomain="aca", max_results=3)
    print(f"grep_texts('休學', aca): {time.time()-t0:.2f}s, {len(results)} results")
    for r in results:
        print(f"  [{r['subdomain']}] {r['title'][:50]}  {r['url'][:60]}")

    if results:
        main_url = results[0]["url"]
        t0 = time.time()
        page = get_page(main_url)
        print(f"\nget_page: {time.time()-t0:.2f}s, {len(page.get('text',''))} chars")
        print(f"  title: {page.get('title','')}")

        t0 = time.time()
        links = extract_links(main_url)
        print(f"\nextract_links: {time.time()-t0:.2f}s, {len(links)} NCCU links")
        for lk in links[:5]:
            print(f"  [{lk['label'][:30]}] → {lk['url'][:60]}")

        form_ids = extract_form_ids(page.get("text", ""))
        print(f"\nextract_form_ids: {form_ids}")

    t0 = time.time()
    form = get_form("QP-T01-03-02")
    print(f"\nget_form('QP-T01-03-02'): {time.time()-t0:.2f}s")
    if "error" not in form:
        print(f"  title: {form['form_title']}")
        print(f"  unit:  {form['form_unit']}")
        print(f"  text length: {len(form.get('text',''))} chars")
        print(f"  text preview: {form.get('text','')[:200]}")
    else:
        print(f"  ERROR: {form['error']}")
