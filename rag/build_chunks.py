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
        "title":       rec.get("form_title") or rec.get("url", "").split("/")[-1] or rec.get("url", ""),
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
        chunks = chunk_html(text, meta)
        # Tag short title-only chunks as navigation (still indexed, not skipped)
        for chunk in chunks:
            if len(chunk.get("text_clean", chunk["text"])) < 30:
                chunk["chunk_type"] = "navigation"
            else:
                chunk["chunk_type"] = "content"
        return chunks

    elif actual_type == "pdf":
        text = extract_pdf(fpath)
        if not text:
            # Stub chunk: preserve URL so the document is discoverable via keyword/dense search.
            # chunk_type="navigation" keeps it out of dense candidates (_NavigationFilter)
            # but it remains in FTS5. On re-run after OCR, real chunks replace this stub.
            stub_text = meta.get("title") or fpath.name
            stub = {**meta,
                    "text":       stub_text,
                    "text_clean": stub_text,
                    "chunk_len":  len(stub_text),
                    "chunk_type": "navigation"}
            return [stub]
        chunks = chunk_pdf(text, meta)
        for chunk in chunks:
            chunk["chunk_type"] = "content"
        return chunks

    return []


def main():
    parser = argparse.ArgumentParser(description="Build RAG chunks from crawl output.")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first 20 records (quick test)")
    args = parser.parse_args()

    output_dir = ROOT / "output"
    out_path = ROOT / "rag" / "chunks.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    # Collect records from all output/<subdomain>/map.json and supplementary_map.json
    raw_records: list[dict] = []
    subdirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
    if not subdirs:
        print(f"ERROR: No subdomain directories found in {output_dir}. Run the crawler first.")
        sys.exit(1)

    for subdir in subdirs:
        sub_map = subdir / "map.json"
        sub_supp = subdir / "supplementary_map.json"
        if sub_map.exists():
            extra = json.loads(sub_map.read_text(encoding="utf-8"))
            raw_records += extra
            print(f"  + {subdir.name}/map.json: {len(extra)} records")
        if sub_supp.exists():
            extra = json.loads(sub_supp.read_text(encoding="utf-8"))
            raw_records += extra
            print(f"  + {subdir.name}/supplementary_map.json: {len(extra)} records")

    # Deduplicate by URL — keep first occurrence, preserve cross-subdomain provenance
    # in raw map files but avoid duplicate Qdrant vectors
    seen_urls: set[str] = set()
    records: list[dict] = []
    for r in raw_records:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            records.append(r)
    if len(raw_records) != len(records):
        print(f"  Deduplicated: {len(raw_records)} → {len(records)} records ({len(raw_records)-len(records)} duplicates removed)")
    if args.test:
        records = records[:20]
        print(f"[TEST MODE] Processing first {len(records)} records only.\n")

    total_chunks = 0
    skipped = 0
    skip_failed = 0
    skip_office = 0
    skip_empty  = 0
    stub_count  = 0
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
            chunk_types = {c.get("chunk_type") for c in chunks}
            if ftype == "html":
                html_count += 1
            elif ftype == "document":
                if chunk_types == {"navigation"} and len(chunks) == 1 and len(chunks[0]["text"]) < 200:
                    stub_count += 1
                else:
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
    print(f"  PDF stubs        : {stub_count}  (URL-only, pending OCR)")
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
                print(f"  url        : {chunk['url']}")
                print(f"  type       : {chunk['source_type']}")
                print(f"  chunk_type : {chunk.get('chunk_type', 'N/A')}")
                print(f"  depth      : {chunk['depth']}")
                print(f"  chars      : {chunk['chunk_len']}")
                print(f"  text       : {chunk['text'][:200]!r}")
                print(f"  text_clean : {chunk.get('text_clean', '')[:200]!r}")


if __name__ == "__main__":
    main()
