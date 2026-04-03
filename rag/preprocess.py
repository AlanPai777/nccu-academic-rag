"""
preprocess.py — Extract clean text from HTML and PDF files.

HTML: locates div.item-page (Joomla main content area), strips nav/footer/scripts.
PDF:  uses pdfplumber; returns empty string if text < 100 chars (likely scanned).
"""

import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag


def _node_to_text(node, base_url: str) -> str:
    """
    Recursively convert a BeautifulSoup node to plain text,
    rendering <a href=...> tags as markdown [text](url) links.
    Relative URLs are resolved against base_url.
    """
    if isinstance(node, NavigableString):
        return str(node)

    if not isinstance(node, Tag):
        return ""

    # Block-level tags: add newlines around their content
    block_tags = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
                  "br", "td", "th", "dt", "dd", "blockquote", "pre"}

    if node.name == "a":
        href = node.get("href", "").strip()
        text = node.get_text(strip=True)
        if href and text:
            # Resolve relative URLs
            if href.startswith(("http://", "https://")):
                full_url = href
            elif href.startswith("/") and base_url:
                parsed = urlparse(base_url)
                full_url = f"{parsed.scheme}://{parsed.netloc}{href}"
            else:
                full_url = urljoin(base_url, href) if base_url else href
            return f"[{text}]({full_url})"
        return text

    parts = []
    for child in node.children:
        parts.append(_node_to_text(child, base_url))

    content = "".join(parts)

    if node.name in block_tags:
        return f"\n{content}\n"
    return content


def extract_html(path: str | Path, url: str = "") -> str:
    """
    Extract main content text from a downloaded HTML file.

    Targets div.item-page (Joomla CMS content area).
    Falls back to <main>, then <body> if div.item-page is absent.
    Removes nav, footer, header, script, style, aside tags before extraction.
    Preserves hyperlinks as markdown [text](url).

    Returns plain text with normalised whitespace.
    """
    path = Path(path)
    if not path.exists():
        return ""

    try:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        return ""

    # Extract page title as fallback for empty/placeholder pages
    # If no <title> tag, derive from URL filename
    if soup.title and soup.title.string and soup.title.string.strip():
        page_title = soup.title.string.strip()
    else:
        page_title = Path(url.rstrip("/").split("/")[-1]).stem if url else ""

    # Remove noise elements
    for tag in soup.find_all(["nav", "footer", "header", "script", "style", "aside"]):
        tag.decompose()

    # Also remove Joomla-specific navigation/breadcrumb wrappers
    for tag in soup.find_all(class_=re.compile(r"nav|menu|breadcrumb|sidebar|footer|header", re.I)):
        tag.decompose()

    # Try Joomla content area first
    content = soup.find("div", class_="item-page")
    if not content:
        content = soup.find("main")
    if not content:
        content = soup.find("article")
    if not content:
        content = soup.find("body")
    if not content:
        return page_title

    text = _node_to_text(content, url)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if "Blank Component" in text or not text:
        # Joomla placeholder or empty page — fall back to page title
        return page_title

    return text


def extract_pdf(path: str | Path) -> str:
    """
    Extract text from a text-based PDF file using pdfplumber.

    Returns empty string if:
    - File doesn't exist
    - pdfplumber not installed
    - Extracted text < 100 chars (likely scanned/image PDF)
    - Any extraction error occurs
    """
    path = Path(path)
    if not path.exists():
        return ""

    try:
        import pdfplumber
    except ImportError:
        return ""

    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t.strip())
            text = "\n\n".join(pages)
    except Exception:
        return ""

    if len(text) < 100:
        return ""  # Likely scanned PDF — skip for now

    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    base = Path(__file__).parent.parent

    # Quick test: show extraction results for a few files
    map_path = base / "output" / "map.json"
    if not map_path.exists():
        print("map.json not found")
        sys.exit(1)

    records = json.loads(map_path.read_text(encoding="utf-8"))

    html_shown = 0
    pdf_shown = 0

    for rec in records:
        if rec.get("status") != "ok" or not rec.get("saved_path"):
            continue

        fpath = base / rec["saved_path"]
        ftype = rec.get("type", "")

        if ftype == "html" and html_shown < 3:
            text = extract_html(fpath, rec.get("url", ""))
            if text:
                print(f"\n{'='*60}")
                print(f"[HTML] {rec['url']}")
                print(f"Chars: {len(text)}")
                print(text[:400])
                html_shown += 1

        elif ftype == "pdf" and pdf_shown < 3:
            text = extract_pdf(fpath)
            if text:
                print(f"\n{'='*60}")
                print(f"[PDF]  {rec['url']}")
                print(f"Chars: {len(text)}")
                print(text[:400])
                pdf_shown += 1

        if html_shown >= 3 and pdf_shown >= 3:
            break

    print(f"\nDone. HTML samples: {html_shown}, PDF samples: {pdf_shown}")
