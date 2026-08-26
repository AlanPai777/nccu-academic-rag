"""
rag/domain_router.py
Phase F Step 5 (condition 5): decides which subdomain grep_texts()/
retrieval_anchor_node should search, replacing the subdomain="aca" hardcode
that Phase E's PROCEDURE/KNOWLEDGE paths inherited from having only the
single 休學 test case to develop against.

Two layers implemented here (see phase_f_planning_report.md 6.12 for the
full four-layer design and its unresolved risks):

  Layer 1 (keyword table, near-zero cost): office/unit names harvested from
           output/office_contacts_index.jsonl (144 subdomains, Step 0c) —
           substring match against the query, longest name wins first.
  Layer 2 (FTS5 subdomain aggregation, Step 4.7's rag/fts_proto3.db):
           only reached when Layer 1 finds nothing. Aggregates per-row BM25
           hits by subdomain (count of matching pages) and returns the top
           subdomain.

Deliberately NOT implemented yet: candidate hedging (Send-based dispatch
when top-2 Layer 2 scores are close) and Layer 3 (CRAG-lite retry with
subdomain=None on low-quality results). The planning report's Step 5 risk
list explicitly says the CRAG-lite quality threshold and hedging trigger
frequency are uncalibrated until Step 1's 6-question upper-bound data is
run through this router — building those two now would mean tuning
thresholds against zero real examples. route_domain() returns a single
best-effort subdomain (or None); callers already have their own
fallback-to-global behavior (grep_texts's own
"if not main_results: grep_texts(keyword)" pattern), so a wrong or missing
Domain Router guess degrades to Phase E's old behavior, not a hard failure.

Layer 2 aggregation method: count (number of matching pages per
subdomain), not max or sum — chosen empirically (2026-08-26) after
head-to-head testing against real queries confirmed max picks the wrong
subdomain when a single page's BM25 score is inflated by term-density
artifacts (e.g. a table-heavy PDF), and count is the only one of the three
that is structurally immune to that exact failure mode (an inflated score
still contributes its full magnitude to sum, but only ever "+1" to count).
sum and count were empirically indistinguishable on every test query run —
count was preferred as the more principled fix (removes the failure
mechanism rather than diluting it). The "large/diffuse subdomain
out-scores a small/precise one" risk this could introduce is real (page
counts across the 152 subdomains range from 1 to 26,289) but UNTESTED —
none of the queries that could have exposed it actually reach Layer 2 in
practice, because Layer 1's office-name keyword table already intercepts
them first. See phase_f_planning_report.md Gotchas for the full comparison
data and this open risk.
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

_CONTACTS_INDEX_PATH = Path(__file__).parent.parent / "output" / "office_contacts_index.jsonl"
_FTS_DB_PATH = Path(__file__).parent / "fts_proto3.db"


@lru_cache(maxsize=1)
def _keyword_table() -> list[tuple[str, str]]:
    """
    Layer 1 data: (office_name, subdomain) pairs from office_contacts_index.jsonl,
    covering all 144 subdomains that file spans — not just the 6
    _PROCEDURE_OFFICES core offices office_lookup_skill.py's _OFFICE_NAME_MAP
    covers, since a query can be about any of the 144 (e.g. "校長室" → www).

    subdomain there is the full hostname ("aca.nccu.edu.tw"); normalized to
    the short form grep_texts()/output/ directory names use ("aca") by
    stripping the ".nccu.edu.tw" suffix.

    Sorted longest-name-first so a more specific name wins when a query
    contains multiple overlapping substrings (same convention as
    office_lookup_skill.py's _ADMIN_TITLES sort order).
    """
    entries: list[tuple[str, str]] = []
    if not _CONTACTS_INDEX_PATH.exists():
        return entries
    seen: set[tuple[str, str]] = set()
    with _CONTACTS_INDEX_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = d.get("name", "").strip()
            sub = d.get("subdomain", "").replace(".nccu.edu.tw", "")
            if not name or not sub or len(name) < 3:
                continue
            key = (name, sub)
            if key in seen:
                continue
            seen.add(key)
            entries.append((name, sub))
    entries.sort(key=lambda kv: -len(kv[0]))
    return entries


def _layer1_match(query: str) -> str | None:
    for name, sub in _keyword_table():
        if name in query:
            return sub
    return None


def layer2_candidates(query: str, top_k: int = 200) -> list[tuple[str, int]]:
    """
    Query Step 4.7's FTS5 index directly (not via KeywordStore.search(),
    which doesn't expose subdomain in its return dict) and aggregate hits
    by subdomain using COUNT — see module docstring for why count was
    chosen over max/sum. Returns [(subdomain, hit_count), ...] sorted
    best-first.

    top_k=200 (raised from an initial 30 during 2026-08-26 testing): with a
    narrow window, a query whose jieba segmentation produces a noisy token
    (e.g. "轉學考資訊" → "轉學" OR "考資訊", the latter garbage) could let a
    handful of irrelevant subdomains matching only the noise token fill the
    whole top-30 and win the count vote before the correct subdomain's many
    genuine "轉學"-token matches had a chance to be counted at all — 200
    confirmed empirically wide enough for the correct subdomain to
    dominate by volume in every query tested, at negligible extra latency
    (SQLite FTS5's ORDER BY+LIMIT is index-driven; observed cost stayed
    under ~2s regardless of top_k in this range).
    """
    if not _FTS_DB_PATH.exists():
        return []

    from rag.keyword_store import KeywordStore
    fts_query = KeywordStore._fts_query(query)
    if not fts_query:
        return []

    conn = sqlite3.connect(f"file:{_FTS_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT m.subdomain, bm25(chunks_fts) AS score
            FROM chunks_fts f JOIN chunks_meta m ON f.rowid = m.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (fts_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    agg: dict[str, int] = {}
    for sub, _score in rows:
        agg[sub] = agg.get(sub, 0) + 1
    return sorted(agg.items(), key=lambda kv: -kv[1])


def is_ambiguous(query: str, ratio_threshold: float = 0.6) -> bool:
    """
    Step 6 condition 8-C: cheap signal for "is this query's routing shaky
    enough to be worth an extra Router-as-judge semantic check". Layer 1
    hits are exact office-name matches — never ambiguous by this
    definition. Layer 2 is ambiguous when the #2 subdomain's hit count is
    within ratio_threshold of #1's — the same "large/diffuse subdomain
    might out-vote a small/precise one" risk this module's docstring
    already flags as real-but-untested (confirmed real for "如何辦理復學":
    aca vs osa) lives here. Deliberately loose/inclusive (0.6, not tuned):
    a false positive only costs one extra LLM call in self_eval_node, a
    false negative silently ships an unguarded off-topic answer — the
    asymmetry favors erring toward "ambiguous".
    """
    if _layer1_match(query):
        return False
    candidates = layer2_candidates(query)
    if len(candidates) < 2 or candidates[0][1] == 0:
        return False
    return (candidates[1][1] / candidates[0][1]) >= ratio_threshold


def route_domain(query: str) -> str | None:
    """
    Layer 1 -> Layer 2 -> None. None means neither layer found anything —
    callers should fall back to their pre-Step-5 behavior (hardcoded "aca"
    or an unscoped global search), not treat it as an error.
    """
    hit = _layer1_match(query)
    if hit:
        return hit
    candidates = layer2_candidates(query)
    if not candidates:
        return None
    return candidates[0][0]


if __name__ == "__main__":
    print("=== domain_router self-test ===\n")
    print(f"Layer 1 keyword table: {len(_keyword_table())} entries\n")

    test_queries = [
        ("如何辦理休學", "aca"),
        ("出納組的電話是幾號", "cashier"),
        ("宿舍申請", "osa"),
        ("圖書館借書規則", None),  # www.lib in office_contacts_index — check normalization
        ("校長室在哪裡", "www"),
        ("選課上限幾學分", "aca"),   # Layer-2-only; max used to misroute this to flc
        ("退宿規定", "osa"),         # Layer-2-only
        ("轉學考資訊", "aca"),       # Layer-2-only; jieba mis-segments this — needs wide top_k to recover
    ]
    for q, expected in test_queries:
        hit1 = _layer1_match(q)
        result = route_domain(q)
        layer_used = "L1" if hit1 else ("L2" if result else "none")
        marker = "OK" if (expected is None or result == expected) else "MISMATCH"
        print(f"[{layer_used:4}] {q!r:30} -> {result!r:12} (expected {expected!r}) {marker}")
