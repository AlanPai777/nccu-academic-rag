"""
embedder.py — Embedding via Ollama bge-m3 (Intel GPU accelerated, dense only).

Usage:
    from rag.embedder import Embedder
    emb = Embedder()
    result = emb.embed_batch(["選課辦法", "graduation requirements"])
    # result["dense"]  → list of 1024-dim float lists
    # result["sparse"] → None (Ollama mode, dense only)
"""

from __future__ import annotations

import httpx
from typing import Any


OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "bge-m3"


class Embedder:
    """
    Calls Ollama bge-m3 embedding API (Intel GPU accelerated).
    Dense vectors only (1024-dim).
    """

    def __init__(self, batch_size: int = 32,
                 ollama_url: str = OLLAMA_URL,
                 model: str = OLLAMA_MODEL):
        self.batch_size = batch_size
        self.ollama_url = ollama_url
        self.model = model
        self._verified = False

    # ---------------------------------------------------------------------- #

    def _verify(self) -> None:
        """Check Ollama is reachable and model is available."""
        if self._verified:
            return
        try:
            r = httpx.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            # match "bge-m3" or "bge-m3:latest"
            if not any(self.model in m for m in models):
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Available: {models}\n"
                    f"Pull it with: ./ollama pull {self.model}"
                )
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_url}.\n"
                "Make sure ipex-llm Ollama is running (./start-ollama.sh)."
            )
        self._verified = True

    def _embed_one(self, text: str) -> list[float]:
        r = httpx.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        """Ollama /api/embed supports batch input."""
        r = httpx.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    # ---------------------------------------------------------------------- #

    def embed_batch(self, texts: list[str]) -> dict[str, Any]:
        """
        Embed a list of texts via Ollama bge-m3 (Intel GPU).

        Returns:
            {
                "dense":  list[list[float]],  # 1024-dim per text
                "sparse": None                # not available in Ollama mode
            }
        """
        self._verify()

        dense_vecs: list[list[float]] = []
        # Process in sub-batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self._embed_many(batch)
            dense_vecs.extend(vecs)

        return {"dense": dense_vecs, "sparse": None}

    def embed_query(self, query: str) -> dict[str, Any]:
        """Single-query embedding."""
        result = self.embed_batch([query])
        return {
            "dense":  result["dense"][0],
            "sparse": None,
        }


# --------------------------------------------------------------------------- #
# Quick test                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    queries = [
        "選課辦法",
        "graduation requirements",
        "逕修博士學位申請條件",
        "How to apply for leave of absence?",
    ]

    print("=== Embedder (Ollama bge-m3) quick test ===\n")
    emb = Embedder()
    result = emb.embed_batch(queries)

    for i, q in enumerate(queries):
        dvec = result["dense"][i]
        norm = sum(x * x for x in dvec) ** 0.5
        print(f"[{i}] {q!r}")
        print(f"     dense: dim={len(dvec)}, norm≈{norm:.4f}")
        print()
