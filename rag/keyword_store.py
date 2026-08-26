"""
keyword_store.py — SQLite-FTS5 keyword index for hybrid retrieval.

Builds a full-text search index from chunks.jsonl and exposes BM25 keyword
search. Used alongside Qdrant dense search to improve recall for exact-match
terms (course codes, regulation names, specific Chinese phrases).

Chinese text is segmented using jieba before indexing and querying, so FTS5
can match Chinese words properly (e.g. "選課", "畢業學分").

Usage:
    from rag.keyword_store import KeywordStore
    ks = KeywordStore()
    ks.build_index()                        # build from chunks.jsonl
    results = ks.search("選課上限", top_k=20)  # BM25 keyword search
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import jieba

ROOT = Path(__file__).parent.parent
CHUNKS_PATH = ROOT / "rag" / "chunks.jsonl"
DEFAULT_DB_PATH = Path(__file__).parent / "fts.db"

# NCCU domain-vocabulary compounds jieba's default dictionary mis-segments
# (Phase F, 2026-08-26 — found via Domain Router's "如何辦理復學" misrouting
# to osa instead of aca: jieba split "復學" into "復"+"學", and "學" alone
# is common enough corpus-wide that it diluted Layer 2's count aggregation).
# Loaded once at import time so index-build and query-time segmentation
# (both call _segment() below) stay consistent — an index built before this
# was loaded would still have the old "復"+"學" tokens, so any FTS5 index
# built before this file existed must be rebuilt for the fix to take effect.
# Every entry here was individually confirmed mis-segmented before adding
# (see git history / phase_f_planning_report.md) — this is not a guessed
# list, and is not claimed to be exhaustive; other NCCU-specific compounds
# not tested yet may still mis-segment.
_USERDICT_PATH = Path(__file__).parent / "jieba_userdict.txt"
if _USERDICT_PATH.exists():
    jieba.load_userdict(str(_USERDICT_PATH))


def _segment(text: str) -> str:
    """Segment text using jieba for Chinese word matching in FTS5."""
    return " ".join(jieba.cut(text, cut_all=False))


class KeywordStore:
    """SQLite-FTS5 backed keyword search over RAG chunks."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self._db_path = str(db_path)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def exists(self) -> bool:
        """Check if the FTS index has been built."""
        if not Path(self._db_path).exists():
            return False
        conn = self._conn()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
            ).fetchone()[0]
            return count > 0
        finally:
            conn.close()

    def count(self) -> int:
        """Return number of indexed chunks."""
        if not self.exists():
            return 0
        conn = self._conn()
        try:
            return conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
        finally:
            conn.close()

    def build_index(self, chunks_path: Path | str | list[Path | str] = CHUNKS_PATH,
                    force: bool = False) -> int:
        """Build FTS5 index from one or more JSONL files.

        Args:
            chunks_path: Path to chunks.jsonl (classic RAG, chunked records),
                         OR a list of paths — e.g. glob.glob("output/*/extracted_texts.jsonl")
                         for proto3's whole-page records (Phase F Step 4.7).
                         Field mapping degrades gracefully for whichever
                         schema is present (chunk_index/chunk_len/chunk_type/
                         fetched_at all default when absent, as extracted_texts.jsonl's do).
                         Subdomain is inferred per-file from its parent directory
                         name (output/<subdomain>/extracted_texts.jsonl) when the
                         record itself has no "subdomain" field — needed for
                         Domain Router's Layer 2 per-subdomain aggregation.
            force: If True, drop and rebuild even if index exists

        Returns:
            Number of records indexed.
        """
        paths = [chunks_path] if isinstance(chunks_path, (str, Path)) else list(chunks_path)
        paths = [Path(p) for p in paths]
        existing_paths = [p for p in paths if p.exists()]
        if not existing_paths:
            print(f"ERROR: none of {len(paths)} input path(s) exist. Run build_chunks.py "
                  f"(or preprocess_all.py for extracted_texts.jsonl) first.")
            sys.exit(1)
        if len(existing_paths) < len(paths):
            print(f"WARNING: {len(paths) - len(existing_paths)} of {len(paths)} "
                  f"input path(s) not found — indexing the remaining {len(existing_paths)}.")

        conn = self._conn()
        try:
            if force:
                conn.execute("DROP TABLE IF EXISTS chunks_fts")
                conn.execute("DROP TABLE IF EXISTS chunks_meta")

            # Metadata table: stores full chunk info for retrieval
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks_meta (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    title TEXT,
                    subdomain TEXT,
                    depth INTEGER DEFAULT 0,
                    source_type TEXT,
                    category TEXT,
                    fetched_at TEXT,
                    text TEXT,
                    text_clean TEXT,
                    chunk_index INTEGER DEFAULT 0,
                    chunk_len INTEGER DEFAULT 0,
                    chunk_type TEXT DEFAULT 'content'
                )
            """)

            # FTS5 virtual table: indexes jieba-segmented text for keyword search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(text_segmented)
            """)

            # Check if already populated
            existing = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
            if existing > 0 and not force:
                print(f"FTS index already has {existing} chunks. Use force=True to rebuild.")
                return existing

            # Clear and rebuild
            conn.execute("DELETE FROM chunks_meta")
            conn.execute("DELETE FROM chunks_fts")

            total = 0
            for path in existing_paths:
                # output/<subdomain>/extracted_texts.jsonl -> "<subdomain>"; for a
                # flat file like rag/chunks.jsonl this just yields its parent dir
                # name and is ignored below since those records carry no real
                # per-record subdomain concept.
                path_subdomain = path.parent.name

                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text_clean = chunk.get("text_clean", chunk.get("text", ""))
                        subdomain = chunk.get("subdomain") or path_subdomain

                        # Insert metadata
                        conn.execute("""
                            INSERT INTO chunks_meta
                                (url, title, subdomain, depth, source_type, category, fetched_at,
                                 text, text_clean, chunk_index, chunk_len, chunk_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            chunk.get("url", ""),
                            chunk.get("title", ""),
                            subdomain,
                            chunk.get("depth", 0),
                            chunk.get("source_type", ""),
                            chunk.get("category", ""),
                            chunk.get("fetched_at", ""),
                            chunk.get("text", ""),
                            text_clean,
                            chunk.get("chunk_index", 0),
                            chunk.get("chunk_len", 0),
                            chunk.get("chunk_type", "content"),
                        ))

                        # Insert jieba-segmented text into FTS5
                        # rowid matches chunks_meta.rowid (both auto-increment from 1)
                        conn.execute(
                            "INSERT INTO chunks_fts(rowid, text_segmented) VALUES (?, ?)",
                            (total + 1, _segment(text_clean))
                        )
                        total += 1

                        if total % 2000 == 0:
                            print(f"  [{total}] records indexed...")
                            conn.commit()

            conn.commit()

            print(f"FTS5 index built: {total} records indexed from {len(existing_paths)} "
                  f"file(s) -> {self._db_path}")
            return total

        finally:
            conn.close()

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 keyword search.

        Args:
            query: Search query (Chinese or English)
            top_k: Maximum results to return

        Returns:
            List of dicts matching retriever format:
            {text, text_clean, url, title, category, source_type,
             chunk_index, chunk_type, bm25_score}
        """
        if not self.exists():
            return []

        conn = self._conn()
        try:
            fts_query = self._fts_query(query)
            if not fts_query:
                return []

            # FTS5 MATCH with BM25 ranking
            # bm25() returns negative scores (more negative = better match)
            rows = conn.execute("""
                SELECT
                    m.url, m.title, m.source_type, m.category,
                    m.text, m.text_clean, m.chunk_index, m.chunk_type,
                    bm25(chunks_fts) as score
                FROM chunks_fts f
                JOIN chunks_meta m ON f.rowid = m.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
            """, (fts_query, top_k)).fetchall()

            results = []
            for row in rows:
                results.append({
                    "url":         row[0],
                    "title":       row[1],
                    "source_type": row[2],
                    "category":    row[3],
                    "text":        row[4],
                    "text_clean":  row[5],
                    "chunk_index": row[6],
                    "chunk_type":  row[7],
                    "bm25_score":  round(-row[8], 4),  # flip sign: higher = better
                })
            return results

        except Exception:
            # If query syntax is invalid for FTS5, return empty
            return []
        finally:
            conn.close()

    @staticmethod
    def _fts_query(query: str) -> str:
        """Convert user query to FTS5 MATCH syntax.

        Segments query with jieba, wraps each token in quotes,
        joins with OR for broader recall (reranker handles precision).
        """
        segmented = _segment(query)
        tokens = re.findall(r'[\w]+', segmented, re.UNICODE)
        if not tokens:
            return ""
        quoted = [f'"{t}"' for t in tokens]
        return " OR ".join(quoted)


# ── CLI / Quick test ──────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FTS5 keyword store.")
    parser.add_argument("--build", action="store_true", help="Build FTS index from chunks.jsonl")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild FTS index")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--stats", action="store_true", help="Show index stats")
    args = parser.parse_args()

    ks = KeywordStore()

    if args.build or args.rebuild:
        count = ks.build_index(force=args.rebuild)
        print(f"Indexed {count} chunks.")

    elif args.stats:
        print(f"FTS index exists: {ks.exists()}")
        print(f"Chunks indexed: {ks.count()}")

    elif args.query:
        if not ks.exists():
            print("FTS index not built. Run with --build first.")
            sys.exit(1)

        results = ks.search(args.query, top_k=args.top_k)
        print(f"\n=== Keyword search: {args.query!r} ===")
        print(f"Found {len(results)} results:\n")
        for i, r in enumerate(results):
            print(f"[{i+1}] bm25={r['bm25_score']:.4f}")
            print(f"     URL  : {r['url']}")
            print(f"     type : {r['source_type']}  chunk: {r['chunk_index']}")
            print(f"     text : {r['text_clean'][:200].replace(chr(10), ' ')}...")
            print()

    else:
        # Quick self-test
        import tempfile
        import os

        print("=== KeywordStore quick test ===\n")
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_path = tmp.name
        tmp.close()

        # Create test chunks
        tmp_chunks = tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False, encoding="utf-8")
        test_data = [
            {"url": "http://test/1", "title": "選課規定", "text": "每學期選課上限為25學分",
             "text_clean": "每學期選課上限為25學分", "chunk_index": 0, "chunk_len": 12,
             "source_type": "html", "category": "reg", "chunk_type": "content"},
            {"url": "http://test/2", "title": "畢業學分", "text": "畢業最低學分數為128學分",
             "text_clean": "畢業最低學分數為128學分", "chunk_index": 0, "chunk_len": 13,
             "source_type": "html", "category": "reg", "chunk_type": "content"},
            {"url": "http://test/3", "title": "Course Info", "text": "Maximum 25 credits per semester",
             "text_clean": "Maximum 25 credits per semester", "chunk_index": 0, "chunk_len": 30,
             "source_type": "html", "category": "reg", "chunk_type": "content"},
        ]
        for d in test_data:
            tmp_chunks.write(json.dumps(d, ensure_ascii=False) + "\n")
        tmp_chunks.close()

        try:
            ks_test = KeywordStore(db_path=tmp_path)

            # Build
            count = ks_test.build_index(chunks_path=tmp_chunks.name, force=True)
            assert count == 3, f"Expected 3 chunks, got {count}"
            print(f"[PASS] Build index: {count} chunks")

            # Exists
            assert ks_test.exists()
            print("[PASS] Index exists")

            # Count
            assert ks_test.count() == 3
            print("[PASS] Count = 3")

            # Search Chinese
            results = ks_test.search("選課學分", top_k=5)
            assert len(results) > 0, f"Chinese search returned 0 results"
            assert any("選課" in r["text_clean"] for r in results)
            print(f"[PASS] Chinese search: {len(results)} results")

            # Search English
            results = ks_test.search("credits semester", top_k=5)
            assert len(results) > 0, f"English search returned 0 results"
            assert any("credits" in r["text_clean"].lower() for r in results)
            print(f"[PASS] English search: {len(results)} results")

            # Search mixed
            results = ks_test.search("25學分", top_k=5)
            assert len(results) > 0
            print(f"[PASS] Mixed search: {len(results)} results")

            # Rebuild
            count = ks_test.build_index(chunks_path=tmp_chunks.name, force=True)
            assert count == 3
            print("[PASS] Rebuild index")

            print("\n=== All tests passed ===")

        finally:
            os.unlink(tmp_path)
            os.unlink(tmp_chunks.name)
