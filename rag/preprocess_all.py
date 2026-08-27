"""
preprocess_all.py — Extract clean text from crawled HTML and PDF files.

Each subdomain writes to its own output/<subdomain>/extracted_texts.jsonl
(--mode all, the default, or --mode main) or extracted_supplementary.jsonl
(--mode supplementary). export_for_llm.py merges all subdomain files into
llm_context.txt.

Each line is a JSON object with:
  url, title, source_type, category, depth, text

Usage:
    python rag/preprocess_all.py --subdomain aca   # process aca only
    python rag/preprocess_all.py --subdomain osa   # process osa only
    python rag/preprocess_all.py                   # process all subdomain dirs
    python rag/preprocess_all.py --test            # first 20 records (quick test)
    python rag/preprocess_all.py --subdomain aca --mode main
        # map.json only → extracted_texts.jsonl. Use when the subdomain
        # already has its own separately-sourced extracted_supplementary.jsonl
        # (e.g. from an NCCU-Crawler bulk sync) and re-merging
        # supplementary_map.json would just duplicate it — see
        # phase_f_planning_report.md, 2026-08-27 aca/osa duplication finding.
    python rag/preprocess_all.py --subdomain aca --mode supplementary
        # supplementary_map.json only → extracted_supplementary.jsonl (a new
        # output file this script didn't produce before this mode existed).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.preprocess import extract_html, extract_pdf


def process_subdir(subdir: Path, test: bool = False, mode: str = "all") -> None:
    """
    Extract text from one subdomain directory.

    mode:
      "all"           (default) — map.json + supplementary_map.json merged
                       into extracted_texts.jsonl. Unchanged legacy
                       behavior — the subdomains with no separately-sourced
                       extracted_supplementary.jsonl of their own rely on
                       this to get supplementary content at all.
      "main"          — map.json only, into extracted_texts.jsonl. Use when
                       the subdomain already has an extracted_supplementary
                       .jsonl from elsewhere (e.g. an NCCU-Crawler bulk
                       sync) and re-merging supplementary_map.json would
                       just duplicate it (aca/osa, 2026-08-27 finding —
                       see phase_f_planning_report.md).
      "supplementary" — supplementary_map.json only, into a NEW output file
                       extracted_supplementary.jsonl (this script never
                       wrote that file before this mode existed) — aligns
                       with NCCU-Crawler's own two-file convention.
                       Deliberately never writes extracted_texts.jsonl in
                       this mode: if "main" and "supplementary" both wrote
                       the same file, running them in sequence would have
                       each clobber the other's output (plain "w" mode) —
                       writing to disjoint files is what makes running both
                       modes back-to-back equivalent to "all", instead of a
                       data-loss trap.
    """
    sub_map  = subdir / "map.json"
    sub_supp = subdir / "supplementary_map.json"
    out_path = subdir / ("extracted_supplementary.jsonl" if mode == "supplementary" else "extracted_texts.jsonl")

    records: list[dict] = []
    if mode in ("all", "main") and sub_map.exists():
        extra = json.loads(sub_map.read_text(encoding="utf-8"))
        records += extra
        print(f"  + {subdir.name}/map.json: {len(extra)} records")
    if mode in ("all", "supplementary") and sub_supp.exists():
        extra = json.loads(sub_supp.read_text(encoding="utf-8"))
        records += extra
        print(f"  + {subdir.name}/supplementary_map.json: {len(extra)} records")

    if not records:
        print(f"  No records found in {subdir.name}/ for mode={mode!r}")
        return

    if test:
        records = records[:20]
        print(f"[TEST MODE] Processing first {len(records)} records only.\n")

    total = skipped = html_count = pdf_count = 0

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
                "title":       rec.get("form_title") or rec.get("url", "").split("/")[-1] or rec.get("url", ""),
                "source_type": source_type,
                "category":    rec.get("category", ""),
                "depth":       rec.get("depth"),
                "text":        text,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1
            if source_type == "html":
                html_count += 1
            else:
                pdf_count += 1

            if (i + 1) % 100 == 0 and not test:
                print(f"  Processed {i+1}/{len(records)} records, {total} extracted so far...")

    print(f"\n{'='*50}")
    print(f"Subdomain  : {subdir.name}")
    print(f"Extracted  : {total}  (HTML: {html_count}, PDF: {pdf_count})")
    print(f"Skipped    : {skipped}")
    print(f"Output     : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract text from crawled HTML/PDF files.")
    parser.add_argument("--subdomain", default=None,
                        help="Process only this subdomain (e.g. aca). Omit to process all.")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first 20 records (quick test)")
    parser.add_argument("--mode", choices=["all", "main", "supplementary"], default="all",
                        help="all (default) = map.json+supplementary_map.json -> extracted_texts.jsonl "
                             "(legacy behavior); main = map.json only -> extracted_texts.jsonl; "
                             "supplementary = supplementary_map.json only -> extracted_supplementary.jsonl")
    args = parser.parse_args()

    output_dir = ROOT / "output"

    if args.subdomain:
        subdir = output_dir / args.subdomain
        if not subdir.is_dir():
            print(f"ERROR: {subdir} not found.")
            sys.exit(1)
        process_subdir(subdir, test=args.test, mode=args.mode)
    else:
        subdirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
        if not subdirs:
            print(f"ERROR: No subdomain directories found in {output_dir}.")
            sys.exit(1)
        for subdir in subdirs:
            print(f"\n--- Processing {subdir.name}/ ---")
            process_subdir(subdir, test=args.test, mode=args.mode)


if __name__ == "__main__":
    main()
