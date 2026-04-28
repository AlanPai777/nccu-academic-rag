"""
preprocess.py — Extract clean text from HTML and PDF files.

HTML: locates div.item-page (Joomla main content area), strips nav/footer/scripts.
PDF:  uses pdfplumber; returns empty string if text < 100 chars (likely scanned).
"""

import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

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


def extract_pdf(path: str | Path) -> str:
    """
    Extract text from a PDF file.

    Cascade:
    1. pdfplumber — fast, text-based PDFs (files under 50 MB)
       PyMuPDF (_extract_pdf_fitz) — large PDFs (≥50 MB); same accuracy, far lower RAM
    2. OCR fallback (_ocr_pdf) — if result is image-only (<100 chars) or
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
        return _ocr_pdf(path)  # PDF open error — try OCR directly

    if len(text) < 100:
        # Image-only PDF (no text layer) — fall back to OCR
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
