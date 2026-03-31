"""
chunker.py — Split extracted text into chunks suitable for embedding.

HTML pages : split by heading markers, max 800 chars per chunk.
PDF docs   : fixed-size windows, 512 tokens (~400 chars) + 128-token overlap.
"""

import re
from typing import Any


# --------------------------------------------------------------------------- #
# HTML chunking — header-aware                                                #
# --------------------------------------------------------------------------- #

# Matches lines that look like headings (short, possibly all-caps, or followed
# by a blank line in the extracted plain text).
_HEADING_RE = re.compile(
    r"^(.{1,80})\n(?=\n|[A-Z\u4e00-\u9fff])",  # short line before blank/CJK
    re.MULTILINE,
)

MAX_HTML_CHUNK = 800    # characters
OVERLAP_HTML   = 100    # characters of overlap between chunks


def chunk_html(text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Split HTML-extracted text into chunks.

    Strategy:
    1. Split on blank lines (paragraph boundaries).
    2. Merge paragraphs into chunks up to MAX_HTML_CHUNK chars.
    3. Each chunk inherits metadata + chunk_index.
    """
    if not text.strip():
        return []

    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[dict[str, Any]] = []
    current = ""
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph stays within limit, accumulate
        if len(current) + len(para) + 2 <= MAX_HTML_CHUNK:
            current = (current + "\n\n" + para).strip()
        else:
            # Flush current chunk
            if current:
                chunks.append(_make_chunk(current, metadata, idx))
                idx += 1
                # Keep last OVERLAP_HTML chars as overlap seed
                current = current[-OVERLAP_HTML:].strip() + "\n\n" + para
                current = current.strip()
            else:
                # Para itself is too long — hard split
                for sub in _hard_split(para, MAX_HTML_CHUNK):
                    chunks.append(_make_chunk(sub, metadata, idx))
                    idx += 1
                current = ""

    if current:
        chunks.append(_make_chunk(current, metadata, idx))

    return chunks


# --------------------------------------------------------------------------- #
# PDF chunking — fixed-size token window                                      #
# --------------------------------------------------------------------------- #

# Approximate: 1 token ≈ 4 chars (works for mixed CJK/Latin)
CHARS_PER_TOKEN = 4
MAX_PDF_TOKENS  = 512
OVERLAP_TOKENS  = 128

MAX_PDF_CHARS   = MAX_PDF_TOKENS  * CHARS_PER_TOKEN   # 2048
OVERLAP_PDF     = OVERLAP_TOKENS  * CHARS_PER_TOKEN   # 512


def chunk_pdf(text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Split PDF-extracted text with fixed-size windows and overlap.

    Uses character count as a proxy for token count (4 chars ≈ 1 token).
    """
    if not text.strip():
        return []

    text = text.strip()
    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + MAX_PDF_CHARS
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(_make_chunk(chunk_text, metadata, idx))
            idx += 1
        start += MAX_PDF_CHARS - OVERLAP_PDF
        if start >= len(text):
            break

    return chunks


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_chunk(text: str, metadata: dict[str, Any], index: int) -> dict[str, Any]:
    chunk = dict(metadata)   # shallow copy of parent metadata
    chunk["text"] = text
    chunk["chunk_index"] = index
    chunk["chunk_len"] = len(text)
    return chunk


def _hard_split(text: str, max_len: int) -> list[str]:
    """Split a single long string into pieces of at most max_len chars."""
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


# --------------------------------------------------------------------------- #
# Quick test                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sample_html = """首頁介紹

本系提供多樣化的課程，包括必修與選修科目。
學生需完成 128 學分方可畢業。

選課辦法

每學期開學前兩週為加退選期間。
學生可於系統中進行選課操作。
如有衝突請洽各系辦公室協助。

注意事項

請注意選課截止日期。
逾期恕不受理。"""

    meta = {"url": "https://example.com/page", "title": "測試頁", "depth": 2,
            "source_type": "html", "category": "admin_academic"}

    chunks = chunk_html(sample_html, meta)
    print(f"HTML chunks: {len(chunks)}")
    for c in chunks:
        print(f"  [{c['chunk_index']}] len={c['chunk_len']}  {c['text'][:60]!r}")

    sample_pdf = "政大教務處規定 " * 300  # ~2100 chars
    meta_pdf = {**meta, "source_type": "pdf"}
    pdf_chunks = chunk_pdf(sample_pdf, meta_pdf)
    print(f"\nPDF chunks: {len(pdf_chunks)}")
    for c in pdf_chunks:
        print(f"  [{c['chunk_index']}] len={c['chunk_len']}")
