"""
export_for_llm.py — Export extracted texts as a single file for web LLM testing.

Reads all output/<subdomain>/extracted_texts.jsonl files and merges them into
a single clean text file for upload to Claude.ai, Gemini, or ChatGPT.

Usage:
    python rag/export_for_llm.py                    # export all subdomains
    python rag/export_for_llm.py --subdomain aca    # export one subdomain only
    python rag/export_for_llm.py --category 選課    # filter by category
    python rag/export_for_llm.py --max-chars 800000 # truncate to fit model limit
    python rag/export_for_llm.py --stats            # show stats only

Approximate limits:
    Claude.ai (claude-sonnet-4-6) : ~800,000 chars
    Gemini 1.5 Pro (web)          : ~4,000,000 chars
    ChatGPT-4o                    : ~400,000 chars
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT        = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "output" / "llm_context.txt"


def find_input_files(subdomain: str | None) -> list[Path]:
    """Return all extracted_texts.jsonl paths to read from."""
    output_dir = ROOT / "output"
    if subdomain:
        p = output_dir / subdomain / "extracted_texts.jsonl"
        if not p.exists():
            print(f"ERROR: {p} not found. Run: python rag/preprocess_all.py --subdomain {subdomain}")
            sys.exit(1)
        return [p]
    paths = sorted(output_dir.glob("*/extracted_texts.jsonl"))
    if not paths:
        print(f"ERROR: No extracted_texts.jsonl found in {output_dir}/*/")
        print("Run: python rag/preprocess_all.py")
        sys.exit(1)
    return paths


def load_pages(paths: list[Path], category: str | None = None) -> list[dict]:
    pages = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if category and category not in r.get("category", ""):
                    continue
                if not r.get("text", "").strip():
                    continue
                pages.append(r)
    return pages


def format_page(page: dict, index: int) -> str:
    title       = page.get("title", "").strip()
    url         = page.get("url", "")
    category    = page.get("category", "")
    source_type = page.get("source_type", "")
    text        = page.get("text", "").strip()
    header = f"=== [{index}] {title or url} ==="
    meta   = f"URL: {url} | 類型: {source_type} | 類別: {category}"
    return f"{header}\n{meta}\n{text}"


def main():
    parser = argparse.ArgumentParser(description="Export extracted texts for web LLM testing.")
    parser.add_argument("--subdomain", type=str, default=None,
                        help="Export one subdomain only (e.g. aca). Omit for all.")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category (e.g. 選課, 畢業, 學籍)")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Truncate output to this many chars (e.g. 800000 for Claude.ai)")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats only, don't write file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help=f"Output file path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    input_files = find_input_files(args.subdomain)
    label = args.subdomain or "all subdomains"
    print(f"Loading pages from {len(input_files)} file(s) ({label})...")

    pages = load_pages(input_files, category=args.category)
    print(f"Loaded {len(pages)} pages" + (f" (category: {args.category})" if args.category else ""))

    cats = Counter(p.get("category", "") for p in pages)
    print("\nCategory breakdown:")
    for cat, count in cats.most_common(10):
        print(f"  {cat or '(none)'}: {count}")

    total_chars = sum(len(p.get("text", "")) for p in pages)
    print(f"\nTotal chars: {total_chars:,}")
    print(f"Approx tokens: {total_chars // 4:,}")
    print(f"\nModel fit estimate:")
    print(f"  Claude.ai (~800K chars):  {min(100, total_chars/800000*100):.0f}% — fits {min(len(pages), int(len(pages)*800000/max(total_chars,1)))} pages")
    print(f"  Gemini web (~4M chars):   {min(100, total_chars/4000000*100):.0f}% — fits {min(len(pages), int(len(pages)*4000000/max(total_chars,1)))} pages")
    print(f"  ChatGPT-4o (~400K chars): {min(100, total_chars/400000*100):.0f}% — fits {min(len(pages), int(len(pages)*400000/max(total_chars,1)))} pages")

    if args.stats:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    intro = (
        "以下是國立政治大學教務處官方網站的完整內容，包含課程選修、畢業規定、學籍管理等資訊。\n"
        "請根據以下資料回答問題。如果資料中找不到答案，請說明無法從現有資料回答。\n"
        f"資料來源：{len(pages)} 頁面，包含 HTML 網頁與 PDF 文件。\n"
        + "=" * 60 + "\n\n"
    )

    lines = [intro]
    total_written = len(intro)
    pages_written = 0
    truncated = False

    for i, page in enumerate(pages):
        block = format_page(page, i + 1) + "\n\n"
        if args.max_chars and total_written + len(block) > args.max_chars:
            truncated = True
            break
        lines.append(block)
        total_written += len(block)
        pages_written += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nOutput written: {output_path}")
    print(f"Pages included: {pages_written}/{len(pages)}")
    print(f"Total chars: {total_written:,}")
    if truncated:
        print(f"[Truncated at --max-chars {args.max_chars:,}]")
    print(f"\nNow upload '{output_path.name}' to Claude.ai / Gemini / ChatGPT and ask your questions.")


if __name__ == "__main__":
    main()
