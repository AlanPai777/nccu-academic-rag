"""
preprocess_all.py — Extract clean text from all crawled HTML and PDF files
and write to output/extracted_texts.jsonl.

Each line is a JSON object with:
  url, title, source_type, category, text

Usage:
    python rag/preprocess_all.py           # process all files
    python rag/preprocess_all.py --test    # process first 20 files only
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.preprocess import extract_html, extract_pdf


def main():
    parser = argparse.ArgumentParser(description="Extract text from crawled HTML/PDF files.")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first 20 records (quick test)")
    args = parser.parse_args()

    map_path = ROOT / "output" / "map.json"
    if not map_path.exists():
        print(f"ERROR: {map_path} not found. Run the crawler first.")
        sys.exit(1)

    out_path = ROOT / "output" / "extracted_texts.jsonl"

    records = json.loads(map_path.read_text(encoding="utf-8"))
    if args.test:
        records = records[:20]
        print(f"[TEST MODE] Processing first {len(records)} records only.\n")

    total = 0
    skipped = 0
    html_count = 0
    pdf_count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            if rec.get("status") != "ok" or not rec.get("saved_path"):
                skipped += 1
                continue

            fpath = ROOT / rec["saved_path"]
            raw_type = rec.get("type", "")

            if raw_type == "html":
                text = extract_html(fpath, rec.get("url", ""))
                source_type = "html"
            elif raw_type == "document" and fpath.suffix.lower() == ".pdf":
                text = extract_pdf(fpath)
                source_type = "pdf"
            else:
                skipped += 1
                continue

            if not text:
                skipped += 1
                continue

            entry = {
                "url":         rec.get("url", ""),
                "title":       rec.get("url", "").split("/")[-1] or rec.get("url", ""),
                "source_type": source_type,
                "category":    rec.get("category", ""),
                "text":        text,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1

            if source_type == "html":
                html_count += 1
            else:
                pdf_count += 1

            if (i + 1) % 100 == 0 and not args.test:
                print(f"  Processed {i+1}/{len(records)} records, {total} extracted so far...")

    print(f"\n{'='*50}")
    print(f"Extracted  : {total}")
    print(f"  HTML     : {html_count}")
    print(f"  PDF      : {pdf_count}")
    print(f"Skipped    : {skipped}")
    print(f"Output     : {out_path}")


if __name__ == "__main__":
    main()
