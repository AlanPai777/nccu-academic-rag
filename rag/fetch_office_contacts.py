"""
fetch_office_contacts.py — Playwright re-crawl of docs/FindURLs/contact.csv.

Reads the user-curated office/person -> URL list, fetches each unique URL
with a headless browser (JS-rendered content, low error tolerance — contact
info is not something we want to risk missing via static fetch), extracts
clean text with preprocess.extract_html_bytes(), and writes one JSON record
per CSV row to output/office_contacts_index.jsonl.

Does NOT touch any existing output/<subdomain>/ folder — independent output.

Usage:
    python rag/fetch_office_contacts.py                # full run (434 rows)
    python rag/fetch_office_contacts.py --test          # first 10 unique URLs only
    python rag/fetch_office_contacts.py --resume        # skip URLs already in output file
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.preprocess import extract_html_bytes, is_cloudflare_page

CSV_PATH = ROOT / "docs" / "FindURLs" / "contact.csv"
OUT_PATH = ROOT / "output" / "office_contacts_index.jsonl"

NAV_TIMEOUT_MS = 20_000
SETTLE_MS = 2000  # wait after load for late JS-injected content (staff directories
                   # are often client-rendered — "load" fires before that finishes;
                   # this is the whole reason we're on Playwright instead of static
                   # fetch, so err generous rather than risk silent partial content)


def parse_contact_csv(path: Path) -> list[dict]:
    """
    Parse contact.csv into row dicts.

    The file is NOT a clean rectangular CSV — it has blank separator lines
    between category blocks and at least one stray annotation row
    ("(以上完成)", a checklist note the curator left in). Rows come in two
    shapes: [category, name, url] (most common) or [name, url] (a few rows
    with no explicit category, e.g. individual 副校長 entries). Any row that
    doesn't end in an http(s) URL is skipped — it's noise, not data.
    """
    rows = []
    with path.open(encoding="utf-8") as f:
        for raw in csv.reader(f):
            cells = [c.strip() for c in raw]
            if len(cells) == 3 and cells[2].startswith("http"):
                category, name, url = cells
            elif len(cells) == 2 and cells[1].startswith("http"):
                category, name, url = "", cells[0], cells[1]
            else:
                continue  # blank line, "(以上完成)" marker, or other malformed row
            rows.append({"category": category, "name": name, "url": url})
    return rows


def fetch_all(rows: list[dict], test: bool, resume: bool) -> None:
    from playwright.sync_api import sync_playwright

    unique_urls = list(dict.fromkeys(r["url"] for r in rows))
    if test:
        unique_urls = unique_urls[:10]

    already_done: set[str] = set()
    if resume and OUT_PATH.exists():
        with OUT_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["url"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"--resume: {len(already_done)} URLs already in {OUT_PATH.name}, will skip")

    # Rows sharing one URL are written together right after that URL's fetch
    # completes — nothing is batched to the end of the run. A run interrupted
    # partway through (tool timeout, ctrl-C, crash) still leaves every URL
    # processed so far correctly recorded, and --resume picks up from there.
    url_to_rows: dict[str, list[dict]] = {}
    for r in rows:
        url_to_rows.setdefault(r["url"], []).append(r)

    fail_log: list[dict] = []
    written = 0

    mode = "a" if resume and OUT_PATH.exists() else "w"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p, OUT_PATH.open(mode, encoding="utf-8") as out_f:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, url in enumerate(unique_urls):
            if url in already_done:
                continue

            text = None
            try:
                # "networkidle" hangs indefinitely on sites with persistent background
                # connections (analytics beacons, chat widgets, etc.) — confirmed on
                # tsc.nccu.edu.tw and cids.nccu.edu.tw, which timed out under
                # networkidle but load instantly under "load". "load" + a fixed
                # settle delay is more robust across the 144-subdomain variety here.
                page.goto(url, timeout=NAV_TIMEOUT_MS, wait_until="load")
                page.wait_for_timeout(SETTLE_MS)
                html_bytes = page.content().encode("utf-8")

                if is_cloudflare_page(html_bytes):
                    fail_log.append({"url": url, "reason": "cloudflare_interstitial"})
                else:
                    text = extract_html_bytes(html_bytes, url)
                    if not text:
                        fail_log.append({"url": url, "reason": "empty_extraction"})
            except Exception as e:
                fail_log.append({"url": url, "reason": f"{type(e).__name__}: {e}"})

            if text:
                subdomain = urlparse(url).netloc
                fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                for r in url_to_rows[url]:
                    entry = {
                        "category":   r["category"],
                        "name":       r["name"],
                        "url":        r["url"],
                        "subdomain":  subdomain,
                        "text":       text,
                        "fetched_at": fetched_at,
                    }
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    written += 1
                out_f.flush()

            if (i + 1) % 20 == 0:
                print(f"  fetched {i+1}/{len(unique_urls)} unique URLs, {written} rows written so far...")

        browser.close()

    print(f"\n{'='*50}")
    print(f"Unique URLs attempted : {len(unique_urls) - len(already_done)}")
    print(f"Unique URLs failed    : {len(fail_log)}")
    print(f"Rows written          : {written}")
    print(f"Output                : {OUT_PATH}")

    if fail_log:
        fail_path = OUT_PATH.with_suffix(".failures.json")
        fail_path.write_text(json.dumps(fail_log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Failures logged       : {fail_path} ({len(fail_log)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Playwright re-crawl of contact.csv office URLs.")
    parser.add_argument("--test", action="store_true", help="First 10 unique URLs only")
    parser.add_argument("--resume", action="store_true", help="Skip URLs already in the output file")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    rows = parse_contact_csv(CSV_PATH)
    print(f"Parsed {len(rows)} valid rows from {CSV_PATH.name} "
          f"({len(set(r['url'] for r in rows))} unique URLs)")

    fetch_all(rows, test=args.test, resume=args.resume)


if __name__ == "__main__":
    main()
