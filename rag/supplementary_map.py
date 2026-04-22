"""
supplementary_map.py — Shared helper for output/<subdomain>/supplementary_map.json.

Each subdomain's supplementary fetch scripts write here via update(subdomain=...).
build_chunks.py and preprocess_all.py glob all output/*/supplementary_map.json.
Adding a new data source never requires changing build_chunks.py.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def get_path(subdomain: str) -> Path:
    return ROOT / "output" / subdomain / "supplementary_map.json"


def update(new_records: list[dict], subdomain: str) -> int:
    """Upsert new_records into output/<subdomain>/supplementary_map.json keyed by URL.
    Re-running any fetch script only updates its own records.
    Returns total record count after merge."""
    map_out = get_path(subdomain)
    map_out.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if map_out.exists():
        for r in json.loads(map_out.read_text(encoding="utf-8")):
            existing[r["url"]] = r
    for r in new_records:
        existing[r["url"]] = r
    map_out.write_text(
        json.dumps(list(existing.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(existing)
