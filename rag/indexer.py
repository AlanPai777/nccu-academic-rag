"""
indexer.py — Batch-index rag/chunks.jsonl into a local Qdrant collection.

Prerequisites:
    - Qdrant running on localhost:6333
      docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
    - rag/chunks.jsonl built by build_chunks.py

Usage:
    python rag/indexer.py            # full index build
    python rag/indexer.py --verify   # just print collection stats
    python rag/indexer.py --reset    # drop + recreate collection, then re-index
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Qdrant ────────────────────────────────────────────────────────────────── #
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    PointStruct,
    SparseVector,
    models as qmodels,
)

from rag.embedder import Embedder

# ── Constants ──────────────────────────────────────────────────────────────── #
COLLECTION   = "nccu_aca"
DENSE_DIM    = 1024          # bge-m3 dense output dimension
BATCH_SIZE   = 32            # chunks per embed+upsert batch
QDRANT_URL   = "http://localhost:6333"
CHUNKS_PATH  = ROOT / "rag" / "chunks.jsonl"


# ── Qdrant helpers ─────────────────────────────────────────────────────────── #

def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def collection_exists(client: QdrantClient) -> bool:
    return any(c.name == COLLECTION for c in client.get_collections().collections)


def create_collection(client: QdrantClient) -> None:
    """Create the nccu_aca collection with dense vector support (Ollama mode)."""
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
    )
    print(f"Collection '{COLLECTION}' created (dense only, Ollama bge-m3).")


def drop_collection(client: QdrantClient) -> None:
    client.delete_collection(COLLECTION)
    print(f"Collection '{COLLECTION}' dropped.")


# ── Indexing ───────────────────────────────────────────────────────────────── #

def build_payload(chunk: dict) -> dict:
    """Select fields to store as Qdrant payload."""
    return {
        "text":        chunk.get("text", ""),
        "url":         chunk.get("url", ""),
        "title":       chunk.get("title", ""),
        "depth":       chunk.get("depth", 0),
        "source_type": chunk.get("source_type", ""),
        "category":    chunk.get("category", ""),
        "fetched_at":  chunk.get("fetched_at", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "chunk_len":   chunk.get("chunk_len", 0),
    }


def index_chunks(client: QdrantClient, embedder: Embedder) -> int:
    """Read chunks.jsonl, embed in batches, upsert to Qdrant. Returns total points."""
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found. Run build_chunks.py first.")
        sys.exit(1)

    all_chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))

    total = len(all_chunks)
    print(f"Indexing {total} chunks into '{COLLECTION}'…\n")

    upserted = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Embed
        emb_result = embedder.embed_batch(texts)
        dense_vecs  = emb_result["dense"]
        sparse_vecs = emb_result["sparse"]

        # Build Qdrant points (dense only, Ollama mode)
        points = []
        for i, chunk in enumerate(batch):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": dense_vecs[i]},
                    payload=build_payload(chunk),
                )
            )

        client.upsert(collection_name=COLLECTION, points=points)
        upserted += len(batch)

        pct = upserted / total * 100
        print(f"  [{upserted:>5}/{total}] {pct:.1f}%  (batch size {len(batch)})")

    return upserted


# ── CLI ────────────────────────────────────────────────────────────────────── #

def print_stats(client: QdrantClient) -> None:
    info = client.get_collection(COLLECTION)
    count = client.count(COLLECTION).count
    print(f"\n=== Collection '{COLLECTION}' stats ===")
    print(f"  Points (vectors) : {count}")
    print(f"  Status           : {info.status}")
    print(f"  Dense dim        : {DENSE_DIM}")
    vectors_count = getattr(info, "vectors_count", None)
    if vectors_count:
        print(f"  Vectors count    : {vectors_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index chunks into Qdrant.")
    parser.add_argument("--verify", action="store_true",
                        help="Only print collection stats, skip indexing")
    parser.add_argument("--reset",  action="store_true",
                        help="Drop and recreate collection before indexing")
    args = parser.parse_args()

    client = get_client()

    # Verify only
    if args.verify:
        if not collection_exists(client):
            print(f"Collection '{COLLECTION}' does not exist yet.")
        else:
            print_stats(client)
        return

    # Optionally drop
    if args.reset and collection_exists(client):
        drop_collection(client)

    # Create if needed
    if not collection_exists(client):
        create_collection(client)

    embedder = Embedder(batch_size=BATCH_SIZE)
    upserted = index_chunks(client, embedder)

    print(f"\n{'='*50}")
    print(f"Indexing complete. Total points upserted: {upserted}")
    print_stats(client)


if __name__ == "__main__":
    main()
