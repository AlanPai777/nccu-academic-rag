"""
fetch_moltke_forms.py — Download form PDFs from moltke.nccu.edu.tw/formservice_SSO.

For each known form ID:
  1. Fetch viewdetail page → parse metadata + goFile() tokens
  2. Download Chinese PDF (falls back to English PDF if no Chinese version)
  3. Save to output/docs/admin_academic/moltke_<ID>.pdf
  4. Upsert records into output/supplementary_map.json via supplementary_map.update()

Usage:
    python rag/fetch_moltke_forms.py           # fetch all forms
    python rag/fetch_moltke_forms.py --dry-run # parse pages, skip downloads
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import ssl
import sys

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import rag.supplementary_map as smap

OUTPUT_DIR = ROOT / "output" / "docs" / "admin_academic"
BASE_URL   = "https://moltke.nccu.edu.tw/formservice_SSO"
DELAY      = 0.5   # seconds between requests

# All form IDs found in the corpus (from extracted_texts.jsonl URL scan)
FORM_IDS = [
    "QP-K13-00-03",
    "QP-T01-02-01", "QP-T01-02-02", "QP-T01-02-03", "QP-T01-02-05",
    "QP-T01-03-01", "QP-T01-03-02", "QP-T01-03-03", "QP-T01-03-04",
    "QP-T01-03-05", "QP-T01-03-08",
    "QP-T01-04-01", "QP-T01-04-02", "QP-T01-04-03", "QP-T01-04-05",
    "QP-T01-04-13", "QP-T01-04-17", "QP-T01-04-19", "QP-T01-04-21",
    "QP-T01-04-22", "QP-T01-04-23",
    "QP-T01-05-02", "QP-T01-05-03", "QP-T01-05-05", "QP-T01-05-06",
    "QP-T01-05-08", "QP-T01-05-09", "QP-T01-05-11", "QP-T01-05-13",
    "QP-T01-05-14",
    "QP-T01-06-03",
    "QP-T01-21-01", "QP-T01-21-02",
    "QP-T04-07-02", "QP-T04-07-13",
]



def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def fetch_form_page(client: httpx.Client, form_id: str) -> dict | None:
    """Fetch the form detail page. Returns parsed metadata dict, or None if
    the form is disabled / unreachable."""
    url = f"{BASE_URL}/link.jsp?viewdetail={form_id}"
    try:
        resp = client.get(url, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching {form_id}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    if "查無此表格" in soup.get_text() or "此表格停用" in soup.get_text():
        print(f"  SKIP {form_id}: disabled or not found")
        return None

    meta: dict = {"form_id": form_id, "url": url}

    for row in soup.select("table tr"):
        cells = row.select("td")
        # Rows are either 2-column or 4-column (two label-value pairs side by side)
        pairs = []
        if len(cells) == 2:
            pairs = [(cells[0], cells[1])]
        elif len(cells) >= 4:
            pairs = [(cells[0], cells[1]), (cells[2], cells[3])]
        for key_cell, val_cell in pairs:
            key = key_cell.get_text(separator=" ", strip=True)
            val = val_cell.get_text(separator=" ", strip=True)
            if "表單名稱" in key:
                meta["form_title"] = val
            elif "表單描述" in key:
                meta["form_description"] = val
            elif "表單關鍵字" in key:
                meta["form_keywords"] = val
            elif "承辦單位" in key:
                meta["form_unit"] = val

    # Collect all download links: {token, link_text, file_type}
    links = []
    for a in soup.select('a[href="#"]'):
        m = re.search(r"goFile\('(.+?)'\)", a.get("onclick", ""))
        if not m:
            continue
        prev = a.find_previous(string=True)
        links.append({
            "token":     m.group(1),
            "text":      a.get_text(strip=True),
            "file_type": prev.strip() if prev else "",
        })
    meta["links"] = links
    return meta


def pick_pdf_token(links: list[dict]) -> tuple[str, str] | None:
    """Return (token, link_text) for the best PDF: Chinese first, else any PDF."""
    pdfs = [l for l in links if "*(pdf)" in l.get("file_type", "")]
    if not pdfs:
        return None
    chinese = [l for l in pdfs if _has_chinese(l["text"])]
    chosen = (chinese or pdfs)[0]
    return chosen["token"], chosen["text"]


def download_pdf(client: httpx.Client, token: str, dest: Path) -> int | None:
    """Download via df?<token>. Returns file size in bytes, or None on failure."""
    try:
        resp = client.get(f"{BASE_URL}/df?{token}", follow_redirects=True)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "pdf" not in ctype and len(resp.content) < 500:
            print(f"  WARNING: unexpected content-type '{ctype}', skipping")
            return None
        dest.write_bytes(resp.content)
        return len(resp.content)
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return None



def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch moltke form PDFs into output/.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse pages but skip file downloads")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped = 0
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NCCU-RAG-fetcher/1.0)"}

    # moltke uses legacy TLS renegotiation blocked by OpenSSL 3 by default
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT

    with httpx.Client(headers=headers, timeout=30, verify=ssl_ctx) as client:
        for i, form_id in enumerate(FORM_IDS):
            print(f"[{i+1}/{len(FORM_IDS)}] {form_id}", flush=True)

            meta = fetch_form_page(client, form_id)
            time.sleep(DELAY)
            if meta is None:
                skipped += 1
                continue

            print(f"  Title : {meta.get('form_title', '(no title)')}")
            print(f"  Unit  : {meta.get('form_unit', '(unknown)')}")

            result = pick_pdf_token(meta.get("links", []))
            if result is None:
                print(f"  SKIP  : no PDF link found")
                skipped += 1
                continue

            token, link_text = result
            print(f"  PDF   : {link_text}")

            dest = OUTPUT_DIR / f"moltke_{form_id.replace('/', '_')}.pdf"
            canonical = f"{BASE_URL}/link.jsp?viewdetail={form_id}"

            if args.dry_run:
                print(f"  →     : {dest.relative_to(ROOT)} [dry-run]")
                continue

            file_size = download_pdf(client, token, dest)
            time.sleep(DELAY)
            if file_size is None:
                skipped += 1
                continue

            print(f"  Saved : {dest.name} ({file_size:,} bytes)")

            records.append({
                "url":              canonical,
                "depth":            1,
                "parent":           f"{BASE_URL}/",
                "status":           "ok",
                "type":             "document",
                "fetched_at":       datetime.now().isoformat(),
                "category":         "admin_academic",
                "saved_path":       str(dest.relative_to(ROOT)),
                "child_count":      0,
                "file_size":        file_size,
                # Provenance fields — ignored by build_chunks.py, useful for debugging
                "form_id":          form_id,
                "form_title":       meta.get("form_title", ""),
                "form_description": meta.get("form_description", ""),
                "form_unit":        meta.get("form_unit", ""),
                "form_keywords":    meta.get("form_keywords", ""),
            })

    if not args.dry_run:
        total = smap.update(records)
        print(f"\n{'='*50}")
        print(f"Downloaded : {len(records)}")
        print(f"Skipped    : {skipped}")
        print(f"Map total  : {total} records in {smap.MAP_OUT}")
    else:
        print(f"\n{'='*50}")
        print(f"Downloaded : {len(records)}")
        print(f"Skipped    : {skipped}")


if __name__ == "__main__":
    main()
