"""
fetch_supplementary.py — Download direct-URL files from supplementary sources.

Handles sources where files are directly accessible (no token/auth mechanism).
Each source is a named URL list. Records are upserted into
output/<subdomain>/supplementary_map.json via supplementary_map.update().

To add a new office source: add an entry to SOURCES below.

Usage:
    python rag/fetch_supplementary.py                      # download all (default: aca)
    python rag/fetch_supplementary.py --subdomain osa      # for osa-discovered links
    python rag/fetch_supplementary.py --dry-run            # plan without downloading
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import rag.supplementary_map as smap

DELAY = 0.5

# Each source: (label, category, list of direct URLs)
# DOCX/XLSX are downloaded for future use; build_chunks.py currently indexes PDF only.
SOURCES: list[tuple[str, str, list[str]]] = [
    (
        "newdoc.nccu.edu.tw",
        "admin_academic",
        [
            "https://newdoc.nccu.edu.tw/formservice/973/973_5246.pdf",
            "https://newdoc.nccu.edu.tw/formservice/1178/1178_2197.pdf",
            "https://newdoc.nccu.edu.tw/formservice/1875/1875_5884.pdf",
            # 404 as of 2026-04-21: 42_3392.pdf, 44_3372.pdf, 1380_3053.docx, 1381_3056.xlsx
        ],
    ),
    # Future sources — add here when subdomains are crawled:
    # ("osa.nccu.edu.tw forms", "admin_student", [...]),
    # ("nccuga.nccu.edu.tw forms", "admin_general", [...]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download supplementary direct-URL files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan downloads without fetching")
    parser.add_argument("--subdomain", default="aca",
                        help="Subdomain directory to write into (default: aca)")
    args = parser.parse_args()

    output_dir = ROOT / "output" / args.subdomain / "docs" / "admin_academic"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped = 0

    with httpx.Client(
        headers={"User-Agent": "Mozilla/5.0 (compatible; NCCU-RAG-fetcher/1.0)"},
        timeout=30,
        follow_redirects=True,
    ) as client:
        for label, category, urls in SOURCES:
            print(f"\n--- {label} ({len(urls)} files) ---")

            for url in urls:
                filename = url.split("/")[-1]
                dest = output_dir / f"newdoc_{filename}"
                print(f"  {filename}", flush=True)

                if args.dry_run:
                    print(f"  →     : {dest.relative_to(ROOT)} [dry-run]")
                    continue

                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    file_size = len(resp.content)
                    print(f"  Saved : {dest.name} ({file_size:,} bytes)")
                except Exception as e:
                    print(f"  ERROR : {e}")
                    skipped += 1
                    continue

                time.sleep(DELAY)

                records.append({
                    "url":         url,
                    "depth":       1,
                    "parent":      f"https://{url.split('/')[2]}/",
                    "status":      "ok",
                    "type":        "document",
                    "fetched_at":  datetime.now().isoformat(),
                    "category":    category,
                    "saved_path":  str(dest.relative_to(ROOT)),
                    "child_count": 0,
                    "file_size":   file_size,
                })

    if not args.dry_run:
        total = smap.update(records, subdomain=args.subdomain)
        print(f"\n{'='*50}")
        print(f"Downloaded : {len(records)}")
        print(f"Skipped    : {skipped}")
        print(f"Map total  : {total} records in {smap.get_path(args.subdomain)}")
    else:
        print(f"\n{'='*50}")
        print(f"Would download : {sum(len(u) for _, _, u in SOURCES)}")


if __name__ == "__main__":
    main()
