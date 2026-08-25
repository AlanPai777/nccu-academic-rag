"""
preprocess.py — Extract clean text from HTML and PDF files.

HTML: locates div.item-page (Joomla) or div.page-inner (PageOne CMS), strips
      nav/footer/scripts, appends footer contact/address info separately.
PDF:  uses pdfplumber; falls back to OCR if extracted text < 10 chars (likely scanned).
"""

import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

# pdfminer emits "Could not get FontBBox from font descriptor because None cannot be
# parsed as 4 floats" for Type3 fonts that lack a FontBBox entry. These fonts are used
# for invisible layout glyphs (spacing, empty boxes) — they carry no readable text.
# All actual Chinese content is in Type0 fonts (MingLiU, JhengHei) which extract cleanly.
# The warning is a pdfminer bug (github.com/pdfminer/pdfminer.six/issues/1162) and has
# zero effect on extraction quality — suppressing it here to keep preprocessing output readable.
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def _node_to_text(node, base_url: str) -> str:
    """
    Recursively convert a BeautifulSoup node to plain text,
    rendering <a href=...> tags as markdown [text](url) links.
    Relative URLs are resolved against base_url.
    """
    if isinstance(node, Comment):
        # HTML comments are a NavigableString subclass — must be excluded before
        # the NavigableString check below, or their raw text leaks into output.
        return ""

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


def _decode_cf_email(encoded: str) -> str:
    """
    Decode a Cloudflare email protection token (data-cfemail attribute).
    The encoding is a simple XOR cipher: first byte is the key,
    remaining bytes are XOR'd with it to recover the original email.
    """
    try:
        r = int(encoded[:2], 16)
        return "".join(
            chr(int(encoded[i:i + 2], 16) ^ r)
            for i in range(2, len(encoded), 2)
        )
    except Exception:
        return ""


def _decode_cf_emails_in_soup(soup: BeautifulSoup) -> None:
    """
    In-place: replace all Cloudflare-obfuscated email spans with plain text.
    Joomla uses <span class="__cf_email__" data-cfemail="..."> to hide emails
    from crawlers. The obfuscation is client-side JS; we decode it server-side.
    """
    for span in soup.find_all(attrs={"data-cfemail": True}):
        email = _decode_cf_email(span["data-cfemail"])
        if email:
            span.replace_with(email)


def is_cloudflare_page(html_bytes: bytes) -> bool:
    """
    Return True if the bytes are a Cloudflare challenge/interstitial page
    rather than real content (e.g. under a temporary bot-check).
    Checked against just the first 4KB — the challenge markers always appear
    near the top of the document, and a full-document decode is wasted work
    on the (common) non-challenge case.
    """
    try:
        snippet = html_bytes[:4096].decode("utf-8", errors="ignore")
        return "_cf_chl_opt" in snippet or "cdn-cgi/challenge-platform" in snippet
    except Exception:
        return False


def _parse_soup(soup: BeautifulSoup, url: str = "") -> str:
    """
    Core HTML extraction logic shared by extract_html (file-based) and
    extract_html_bytes (Playwright-rendered bytes, no file on disk).

    Targets div.item-page (Joomla CMS) or div.page-inner (PageOne CMS).
    Falls back to <main>, <article>, then <body> if neither is present.
    Removes nav, footer, header, script, style, aside tags before extraction,
    but re-appends footer-inner/footer-info text separately — that block
    carries office address/floor/phone which is otherwise lost.
    Preserves hyperlinks as markdown [text](url).
    Decodes Cloudflare email protection (data-cfemail) before extraction.
    """
    # Decode Cloudflare-obfuscated emails before any other processing
    _decode_cf_emails_in_soup(soup)

    # Extract page title as fallback for empty/placeholder pages
    # If no <title> tag, derive from URL filename
    if soup.title and soup.title.string and soup.title.string.strip():
        page_title = soup.title.string.strip()
    else:
        page_title = Path(url.rstrip("/").split("/")[-1]).stem if url else ""

    # Footer carries office address/floor/phone — pull it out before the
    # generic footer removal below discards the tag entirely.
    # Class name varies by CMS/theme across the 231 subdomains (empirically
    # confirmed on the 6 Phase F core offices, not exhaustively surveyed):
    #   footer-inner / footer-info-*  — dean (PageOne CMS custom footer)
    #   jlfooterinfo-<id>             — aca (Joomla, numeric suffix per install)
    #   footer_text / customfooter_*  — osa (underscore-style class names)
    # Deliberately NOT "fat-footer" (cashier) — verified that class holds the
    # site nav menu (訊息公告/關於本組/業務簡介...), not contact info; including
    # it would inject nav noise into every cashier page's text.
    footer_info = ""
    footer_inner = soup.find(class_=re.compile(
        r"^footer-inner$|^footer-info|jlfooterinfo|footer_text|customfooter", re.I
    ))
    if footer_inner:
        footer_info = footer_inner.get_text(separator=" ", strip=True)
        # dean.nccu.edu.tw's footer-inner is confirmed compromised (Black Hat SEO
        # link injection: raw IPs + gambling-site keywords mixed into the real
        # contact block). Bare IP literals never appear in legitimate NCCU footer
        # text, so their presence is a reliable signal to drop the whole block
        # rather than risk surfacing spam links in a student-facing answer.
        if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", footer_info):
            footer_info = ""

    # Remove noise elements
    for tag in soup.find_all(["nav", "footer", "header", "script", "style", "aside"]):
        tag.decompose()

    # Also remove Joomla-specific navigation/breadcrumb wrappers, but never
    # decompose <body>/<html> themselves even if a stray class happens to match.
    # Delimiter-anchored, not a bare substring search: page builders (Avia/Enfold,
    # confirmed on mepa.nccu.edu.tw) emit auto-generated element IDs like
    # "av-4naveu-857d169d..." where "nav" appears mid-word by coincidence of the
    # hash — a bare substring match decomposed the entire bio content on that
    # page. Requiring a start/end/hyphen/underscore boundary around the keyword
    # still matches genuine tokens ("nav-wrapper", "site-header", "footer_text")
    # but not accidental mid-word hits.
    _NOISE_CLASS_RE = re.compile(
        r"(?:^|[-_])(?:nav|menu|breadcrumb|sidebar|footer|header)(?:$|[-_])", re.I
    )
    for tag in soup.find_all(class_=_NOISE_CLASS_RE):
        if tag.name in ("body", "html"):
            continue
        tag.decompose()

    # Try Joomla content area first, then PageOne CMS, then WordPress
    # Avia/Enfold theme (confirmed on mepa.nccu.edu.tw and foreign.nccu.edu.tw
    # via Step 0c contact.csv crawl — <main>/<article> on these sites only
    # wrap a near-empty page-transition shell, real content lives in this class).
    content = soup.find("div", class_="item-page")
    if not content:
        content = soup.find("div", class_="page-inner")
    if not content:
        content = soup.find("div", class_="all_colors")
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

    if footer_info:
        text = text + "\n\n" + footer_info

    return text


def extract_html(path: str | Path, url: str = "") -> str:
    """
    Extract main content text from a downloaded HTML file on disk.
    See _parse_soup() for the extraction logic. Returns "" if the file is
    missing or unparseable.
    """
    path = Path(path)
    if not path.exists():
        return ""

    try:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        return ""

    return _parse_soup(soup, url)


def extract_html_bytes(html_bytes: bytes, url: str = "") -> str:
    """
    Extract main content text from Playwright-rendered HTML bytes (no file
    on disk — used for JS-rendered pages fetched via headless browser).
    See _parse_soup() for the extraction logic.
    """
    try:
        soup = BeautifulSoup(html_bytes, "lxml")
    except Exception:
        return ""

    return _parse_soup(soup, url)


def _extract_pdf_fitz(path: Path) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz) instead of pdfplumber.

    Used for large PDFs (>50 MB) where pdfminer loads the entire file into Python
    objects and causes memory spikes. PyMuPDF uses MuPDF's C library which handles
    large files with much lower memory overhead.

    Returns extracted text, or "" if fitz is not installed or extraction fails.
    Subject to the same cid:XX and image-only checks as the pdfplumber path.
    """
    try:
        import fitz
    except ImportError:
        return ""

    try:
        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            t = page.get_text()
            if t and t.strip():
                pages.append(t.strip())
        doc.close()
        text = "\n\n".join(pages)
    except Exception:
        return ""

    if len(text) < 100:
        return ""

    cid_chars = sum(len(m) for m in re.findall(r"\(cid:\d+\)", text))
    if cid_chars / len(text) > 0.2:
        return ""

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_pdf(path: Path) -> str:
    """
    Render each PDF page as an image with PyMuPDF and OCR it with pytesseract.

    Used as fallback when pdfplumber produces image-only or cid:XX undecodable text.
    Requires: fitz (PyMuPDF), pytesseract, and tesseract-ocr with chi_tra language pack.
    Install: sudo apt-get install tesseract-ocr tesseract-ocr-chi-tra && pip install pytesseract

    Alternative: PaddleOCR (pip install paddleocr paddlepaddle, no sudo) gives better
    Traditional Chinese accuracy but loads ~1GB model. Prefer it if OCR quality is poor.

    Returns empty string if dependencies are missing or OCR fails.
    """
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        return ""

    try:
        doc = fitz.open(str(path))
    except Exception:
        return ""

    pages = []
    try:
        for page in doc:
            # Render at 300 DPI (scale factor 300/72 ≈ 4.17) — enough for Chinese characters
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            # chi_tra: Traditional Chinese; eng: Latin fallback
            ocr_text = pytesseract.image_to_string(img, lang="chi_tra+eng")
            if ocr_text.strip():
                pages.append(ocr_text.strip())
    finally:
        doc.close()

    text = "\n\n".join(pages)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_markdown(table: list[list]) -> str:
    """
    Convert a pdfplumber table (list of rows) to a Markdown table string.

    Why Markdown instead of flattened text?
    pdfplumber's extract_text() linearises multi-column tables into a single
    stream of characters ordered by x/y position.  For forms like moltke
    QP-T01-03-02 the three side-by-side approval columns
    (教務處承辦人 | 組長批示 | 教務長) get interleaved into an unreadable
    sequence — the LLM cannot tell that these are three independent fields.
    Markdown pipes make column boundaries explicit so the model reliably
    identifies parallel slots.

    Empty-column pruning: PDF merged cells leave ghost columns whose every
    cell is empty after whitespace stripping.  Keeping them adds noise and
    widens the table without adding information, so they are dropped.
    """
    if not table:
        return ""

    # Normalise cells: None → "", strip whitespace and internal newlines
    rows = [
        [str(c).replace("\n", " ").strip() if c else "" for c in row]
        for row in table
    ]

    # Drop columns that are empty in every row (merged-cell artefacts)
    col_count = max(len(r) for r in rows)
    non_empty_cols = [
        i for i in range(col_count)
        if any(i < len(r) and r[i] for r in rows)
    ]
    rows = [[r[i] for i in non_empty_cols if i < len(r)] for r in rows]

    if not rows or not rows[0]:
        return ""

    md_rows = ["| " + " | ".join(row) + " |" for row in rows]
    separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    md_rows.insert(1, separator)
    return "\n".join(md_rows)


def _extract_pdf_pdfplumber(path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber with smart table handling.

    Per-page strategy
    -----------------
    Pages WITHOUT tables:
        extract_text() — plain linearised text, same as before.

    Pages WITH tables (detected via PDF line/rectangle primitives):
        Two-step approach to avoid duplication:

        Step 1 — outside text:
            page.filter(_outside_tables).extract_text()
            Keeps only characters whose bounding box falls OUTSIDE every
            table's bbox.  This captures headers, titles, footnotes and
            any prose that sits beside the table — content that
            extract_tables() would silently drop.

        Step 2 — Markdown tables:
            find_tables() → each Table.extract() → _table_to_markdown()
            Preserves column structure so the LLM sees parallel cells on
            the same line rather than an interleaved character stream.

        Result: outside_text + Markdown tables, zero duplication.

    Why not just extract_text() + extract_tables()?
        extract_text() already includes linearised table cell content.
        Appending Markdown tables on top duplicates every cell — once as
        garbled linear text, once as clean Markdown.  For QP-T01-03-02
        this inflated output from 1,998 chars to 3,382 chars with no
        quality gain.

    Applicability to other PDFs:
        find_tables() relies on PDF line/rectangle primitives (explicit
        borders).  Borderless tables (whitespace-aligned columns) are not
        detected and fall through to extract_text() — safe, no data loss.
        Image-only or CID-encoded PDFs produce empty output here and are
        handled by the _ocr_pdf() fallback in extract_pdf().
    """
    try:
        import pdfplumber
    except ImportError:
        return ""

    def _outside_tables(char, bboxes):
        # pdfplumber char dicts carry x0/top/x1/bottom in PDF points.
        # ±1 pt tolerance prevents floating-point edge cases where a
        # character sitting exactly on a table border gets misclassified.
        for bbox in bboxes:
            if (char["x0"] >= bbox[0] - 1 and char["top"] >= bbox[1] - 1 and
                    char["x1"] <= bbox[2] + 1 and char["bottom"] <= bbox[3] + 1):
                return False  # inside this table — exclude
        return True           # outside all tables — keep

    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages:
                table_objects = page.find_tables()
                if table_objects:
                    bboxes = [t.bbox for t in table_objects]

                    # Step 1: text outside all tables (headers, footnotes, etc.)
                    outside = (
                        page.filter(lambda c: _outside_tables(c, bboxes))
                        .extract_text() or ""
                    ).strip()

                    # Step 2: each table as a Markdown block
                    parts = [_table_to_markdown(t.extract()) for t in table_objects]
                    md = "\n\n".join(p for p in parts if p)

                    if md:
                        combined = (outside + "\n\n" + md).strip() if outside else md
                        pages.append(combined)
                    elif outside:
                        # table borders detected but all cells empty — keep prose only
                        pages.append(outside)
                else:
                    # No tables: plain text (regulations, announcements, etc.)
                    plain = (page.extract_text() or "").strip()
                    if plain:
                        pages.append(plain)
    except Exception:
        return ""

    return "\n\n".join(pages)


def extract_pdf(path: str | Path) -> str:
    """
    Extract text from a PDF file.

    Cascade:
    1. pdfplumber mixed mode (_extract_pdf_pdfplumber) — text-based PDFs under 50 MB.
       Pages with tables → Markdown (preserves column structure for LLMs).
       Pages without tables → plain text (original behaviour).
       PyMuPDF (_extract_pdf_fitz) — large PDFs (≥50 MB); lower RAM overhead.
    2. OCR fallback (_ocr_pdf) — if result is image-only (<10 chars) or
       cid:XX undecodable (>20% of chars are cid codes). Requires pytesseract
       + tesseract-ocr chi_tra. Returns empty string if not installed.

    Returns empty string if file doesn't exist or all extraction attempts fail.
    """
    path = Path(path)
    if not path.exists():
        return ""

    # Large PDFs: pdfminer loads the entire file into Python objects causing RAM spikes.
    # PyMuPDF (fitz) uses MuPDF's C library and handles large files with much lower overhead.
    if path.stat().st_size >= 50 * 1024 * 1024:
        text = _extract_pdf_fitz(path)
        if not text:
            return _ocr_pdf(path)
        return text

    text = _extract_pdf_pdfplumber(path)

    if not text:
        return _ocr_pdf(path)  # PDF open error or empty — try OCR directly

    if len(text) < 10:
        # Image-only PDF (no text layer) — fall back to OCR.
        # Threshold lowered from 100→10 (commit c617b38 upstream): 100 was
        # misclassifying short-but-real PDF content (~50 chars) as scanned.
        return _ocr_pdf(path)

    cid_chars = sum(len(m) for m in re.findall(r"\(cid:\d+\)", text))
    if cid_chars / len(text) > 0.2:
        # CID-encoded fonts without ToUnicode mapping — undecodable without OCR
        return _ocr_pdf(path)

    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    import sys
    import json
    import tempfile
    from pathlib import Path

    base = Path(__file__).parent.parent

    # ── CF email decode unit test (no output/ needed) ──────────────────────────
    print("=== CF email decode test ===")

    def _cf_encode(email: str, key: int = 0x55) -> str:
        """Helper: encode an email with CF XOR cipher for test HTML generation."""
        result = format(key, "02x")
        for c in email:
            result += format(ord(c) ^ key, "02x")
        return result

    test_cases = [
        "karena@nccu.edu.tw",
        "nccudorm@nccu.edu.tw",
        "oic@nccu.edu.tw",
    ]

    # Build a test HTML with Joomla-style CF-obfuscated emails
    rows = ""
    for email in test_cases:
        encoded = _cf_encode(email)
        rows += (
            f'<tr><td>e-mail</td>'
            f'<td><a href="/cdn-cgi/l/email-protection#ignored">'
            f'<span class="__cf_email__" data-cfemail="{encoded}">'
            f'[email&#160;protected]</span></a></td></tr>\n'
        )

    test_html = f"""<!DOCTYPE html>
<html><head><title>CF Email Test</title></head>
<body><div class="item-page">
<table>{rows}</table>
</div></body></html>"""

    with tempfile.NamedTemporaryFile(suffix=".html", mode="w",
                                     encoding="utf-8", delete=False) as f:
        f.write(test_html)
        tmp_path = f.name

    result = extract_html(tmp_path, "https://test.nccu.edu.tw/")
    Path(tmp_path).unlink()

    passed = all(email in result for email in test_cases)
    for email in test_cases:
        status = "✓" if email in result else "✗"
        print(f"  {status} {email}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}\n")

    if not passed:
        print("Extracted text:\n", result)
        sys.exit(1)

    # ── Quick test: show extraction results from per-subdomain output/ ─────────
    # Prefer output/<subdomain>/map.json (current structure); fall back to the
    # old flat output/map.json for compatibility with a pre-231-subdomain checkout.
    output_dir = base / "output"
    map_path = None
    if output_dir.exists():
        for sub_map in sorted(output_dir.glob("*/map.json")):
            map_path = sub_map
            break
    if map_path is None:
        flat = base / "output" / "map.json"
        if flat.exists():
            map_path = flat

    if not map_path:
        print("No map.json found — run the crawler first.")
        sys.exit(0)

    print(f"Reading: {map_path}")
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

        elif ftype == "document" and pdf_shown < 3:
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
