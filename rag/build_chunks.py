"""
build_chunks.py — Batch-process all HTML and PDF files from the crawl output
and write chunks to rag/chunks.jsonl.

Usage:
    python rag/build_chunks.py           # process all files
    python rag/build_chunks.py --test    # process first 20 files only
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.preprocess import extract_html, extract_pdf
from rag.chunker import chunk_html, chunk_pdf


def build_metadata(rec: dict) -> dict:
    """Extract metadata fields from a map.json record."""
    return {
        "url":         rec.get("url", ""),
        "title":       rec.get("url", "").split("/")[-1] or rec.get("url", ""),
        "depth":       rec.get("depth", 0),
        "source_type": rec.get("type", "html"),
        "category":    rec.get("category", ""),
        "fetched_at":  rec.get("fetched_at", ""),
    }


def process_record(rec: dict, base: Path) -> list[dict]:
    """Return a list of chunks for one crawl record, or [] if skipped."""
    if rec.get("status") != "ok":
        return []
    saved = rec.get("saved_path")
    if not saved:
        return []

    fpath = base / saved
    if not fpath.exists():
        return []

    # Resolve actual file type: map.json uses "document" for all non-HTML files,
    # so fall back to the file extension to distinguish PDF from Office formats.
    raw_type = rec.get("type", "")
    if raw_type == "document":
        ext = fpath.suffix.lower()
        if ext == ".pdf":
            actual_type = "pdf"
        else:
            # .doc / .docx / .odt / .pptx — skip (requires additional libraries)
            return []
    else:
        actual_type = raw_type  # "html" or anything else

    meta = build_metadata(rec)
    meta["source_type"] = actual_type  # override "document" with resolved type

    if actual_type == "html":
        text = extract_html(fpath, meta["url"])
        if not text:
            return []
        return chunk_html(text, meta)

    elif actual_type == "pdf":
        text = extract_pdf(fpath)
        if not text:
            return []
        return chunk_pdf(text, meta)

    return []


def main():
    parser = argparse.ArgumentParser(description="Build RAG chunks from crawl output.")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first 20 records (quick test)")
    args = parser.parse_args()

    map_path = ROOT / "output" / "map.json"
    if not map_path.exists():
        print(f"ERROR: {map_path} not found. Run the crawler first.")
        sys.exit(1)

    out_path = ROOT / "rag" / "chunks.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    records = json.loads(map_path.read_text(encoding="utf-8"))
    if args.test:
        records = records[:20]
        print(f"[TEST MODE] Processing first {len(records)} records only.\n")

    total_chunks = 0
    skipped = 0
    skip_failed = 0
    skip_office = 0
    skip_empty  = 0
    html_count = 0
    pdf_count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            # Track skip reason before calling process_record
            if rec.get("status") != "ok" or not rec.get("saved_path"):
                skip_failed += 1
            else:
                raw_type = rec.get("type", "")
                if raw_type == "document":
                    ext = (ROOT / rec["saved_path"]).suffix.lower()
                    if ext != ".pdf":
                        skip_office += 1

            chunks = process_record(rec, ROOT)
            if not chunks:
                skipped += 1
                continue

            ftype = rec.get("type", "")
            if ftype == "html":
                html_count += 1
            elif ftype == "document":
                pdf_count += 1

            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

            if (i + 1) % 100 == 0 and not args.test:
                print(f"  Processed {i+1}/{len(records)} records, {total_chunks} chunks so far...")

    skip_empty = skipped - skip_failed - skip_office
    print(f"\n{'='*50}")
    print(f"Records processed : {len(records) - skipped}")
    print(f"  HTML files       : {html_count}")
    print(f"  PDF  files       : {pdf_count}")
    print(f"Records skipped   : {skipped}")
    print(f"  fetch failed     : {skip_failed}")
    print(f"  office/other fmt : {skip_office}  (.doc/.odt/.pptx etc.)")
    print(f"  empty content    : {skip_empty}")
    print(f"Total chunks      : {total_chunks}")
    print(f"Output            : {out_path}")

    if args.test and total_chunks > 0:
        print(f"\n--- First 3 chunks ---")
        with out_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                chunk = json.loads(line)
                print(f"\n[Chunk {i}]")
                print(f"  url    : {chunk['url']}")
                print(f"  type   : {chunk['source_type']}")
                print(f"  depth  : {chunk['depth']}")
                print(f"  chars  : {chunk['chunk_len']}")
                print(f"  text   : {chunk['text'][:200]!r}")


if __name__ == "__main__":
    main()
