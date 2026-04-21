"""
supplementary_map.py — Shared helper for output/supplementary_map.json.

All supplementary fetch scripts write here via update().
build_chunks.py and preprocess_all.py read this alongside map.json.
Adding a new data source never requires changing build_chunks.py.
"""

import json
from pathlib import Path

ROOT    = Path(__file__).parent.parent
MAP_OUT = ROOT / "output" / "supplementary_map.json"


def update(new_records: list[dict]) -> int:
    """Upsert new_records into supplementary_map.json keyed by URL.
    Re-running any fetch script only updates its own records.
    Returns total record count after merge."""
    existing: dict[str, dict] = {}
    if MAP_OUT.exists():
        for r in json.loads(MAP_OUT.read_text(encoding="utf-8")):
            existing[r["url"]] = r
    for r in new_records:
        existing[r["url"]] = r
    MAP_OUT.write_text(
        json.dumps(list(existing.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(existing)
