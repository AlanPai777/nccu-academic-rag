"""
analyze_links.py — Discover supplementary link sources in crawled subdomain data.

Reads output/extracted_texts.jsonl, filters by subdomain, extracts all URLs
found in text content, and groups them by domain. Run this after crawling a new
subdomain to identify what supplementary data sources exist before deciding
what to add to fetch_supplementary.py or writing a new fetch script.

Usage:
    python rag/analyze_links.py --subdomain osa    # analyze osa pages only
    python rag/analyze_links.py                    # analyze all subdomains
    python rag/analyze_links.py --subdomain osa --show-urls  # show individual URLs
    python rag/analyze_links.py --subdomain osa --min-count 1  # show all domains
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT  = Path(__file__).parent.parent
INPUT = ROOT / "output" / "extracted_texts.jsonl"

# Known supplementary systems — flagged so you know how to handle them
KNOWN_SYSTEMS: dict[str, str] = {
    "moltke.nccu.edu.tw":  "moltke formservice_SSO → fetch_moltke_forms.py",
    "newdoc.nccu.edu.tw":  "newdoc direct files → fetch_supplementary.py SOURCES",
    "forms.gle":           "Google Forms (not extractable)",
    "pse.is":              "URL shortener (unknown target)",
    "reurl.cc":            "URL shortener (unknown target)",
    "bit.ly":              "URL shortener (unknown target)",
    "ch.com.tw":           "URL shortener (unknown target)",
}

_URL_RE = re.compile(r'https?://[^\s"\'<>，。、）\]\\]+')


def _domain(url: str) -> str:
    try:
        return urlparse(url.rstrip(".,;)\"'")).netloc.lower()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze supplementary link sources in crawled data."
    )
    parser.add_argument("--subdomain", default=None,
                        help="Filter to records from this subdomain (e.g. osa). "
                             "Omit to analyze all subdomains.")
    parser.add_argument("--show-urls", action="store_true",
                        help="Show individual URLs under each domain (up to 10)")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Only show domains with at least this many unique URLs "
                             "(default: 2)")
    args = parser.parse_args()

    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found. Run preprocess_all.py first.")
        sys.exit(1)

    # Load records, optionally filtered by subdomain
    records: list[dict] = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.subdomain and args.subdomain not in r.get("url", ""):
                continue
            records.append(r)

    label = f"subdomain={args.subdomain}" if args.subdomain else "all subdomains"
    print(f"Records analyzed : {len(records)} ({label})")

    # Extract and group URLs from text content
    by_domain: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        self_domain = _domain(rec.get("url", ""))
        for raw_url in _URL_RE.findall(rec.get("text", "")):
            url = raw_url.rstrip(".,;)\"'")
            d = _domain(url)
            if not d or d == self_domain:
                continue  # skip self-references
            by_domain[d].append(url)

    # Deduplicate per domain
    by_domain = {d: list(dict.fromkeys(urls)) for d, urls in by_domain.items()}

    # Sort by count descending
    rows = sorted(by_domain.items(), key=lambda x: -len(x[1]))

    print(f"\n{'Domain':<48} {'URLs':>6}  Note")
    print("─" * 85)
    for domain, urls in rows:
        if len(urls) < args.min_count:
            continue
        note = KNOWN_SYSTEMS.get(domain, "")
        flag = "⚠️  " if note else "   "
        print(f"{flag}{domain:<45} {len(urls):>6}  {note}")
        if args.show_urls:
            for u in sorted(urls)[:10]:
                print(f"     {u}")
            if len(urls) > 10:
                print(f"     ... and {len(urls) - 10} more")

    # Highlight moltke form IDs specifically and check against FORM_IDS
    moltke_urls = by_domain.get("moltke.nccu.edu.tw", [])
    moltke_ids = set()
    for url in moltke_urls:
        m = re.search(r"viewdetail=(QP-[\w-]+)", url)
        if m:
            moltke_ids.add(m.group(1))

    if moltke_ids:
        print(f"\nMoltke form IDs found: {len(moltke_ids)}")
        try:
            from rag.fetch_moltke_forms import FORM_IDS
            known  = moltke_ids & set(FORM_IDS)
            new    = moltke_ids - set(FORM_IDS)
            print(f"  Already in FORM_IDS : {len(known)}")
            if new:
                print(f"  NEW — add to fetch_moltke_forms.py FORM_IDS:")
                for fid in sorted(new):
                    print(f"    \"{fid}\",")
            else:
                print(f"  No new form IDs.")
        except ImportError:
            for fid in sorted(moltke_ids):
                print(f"  {fid}")


if __name__ == "__main__":
    main()
