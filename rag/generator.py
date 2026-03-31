"""
generator.py — LLM answer generation via Ollama (ipex-llm Intel GPU).

Usage:
    from rag.generator import Generator
    gen = Generator()
    result = gen.generate("選課辦法是什麼？", contexts)
    # result → {answer: str, sources: list[{url, title, source_type}]}
"""

from __future__ import annotations

import httpx
from typing import Any

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "granite4:3b"
MAX_CTX_CHARS = 3000   # max total chars from context chunks fed to LLM

SYSTEM_PROMPT = """你是國立政治大學教務處的智慧助理。
請根據以下提供的參考資料回答問題。
規則：
1. 只根據參考資料中的內容回答，不要自行添加資料中沒有的資訊
2. 回答語言請與問題語言一致（問中文答中文，問英文答英文）
3. 若參考資料中找不到答案，請直接說「根據現有資料無法回答此問題」
4. 回答後請列出資料來源的編號"""


def _build_context(contexts: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts = []
    total = 0
    for i, c in enumerate(contexts):
        text = c.get("text", "").strip()
        if not text:
            continue
        entry = f"[{i+1}] {text}"
        if total + len(entry) > MAX_CTX_CHARS:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


def _build_sources(contexts: list[dict]) -> list[dict]:
    """Deduplicate and format source metadata."""
    seen = set()
    sources = []
    for i, c in enumerate(contexts):
        url = c.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({
                "index":       i + 1,
                "url":         url,
                "title":       c.get("title", url.split("/")[-1]),
                "source_type": c.get("source_type", ""),
            })
    return sources


class Generator:
    """Calls Ollama LLM to generate answers from retrieved context."""

    def __init__(self,
                 model: str = OLLAMA_MODEL,
                 ollama_url: str = OLLAMA_URL,
                 temperature: float = 0.1,
                 num_ctx: int = 4096):
        self.model       = model
        self.ollama_url  = ollama_url
        self.temperature = temperature
        self.num_ctx     = num_ctx

    def generate(self, query: str,
                 contexts: list[dict]) -> dict[str, Any]:
        """
        Generate an answer given query and retrieved context chunks.

        Args:
            query    : user question
            contexts : list of chunk dicts from Retriever.retrieve()

        Returns:
            {
                "answer":  str,
                "sources": list[{index, url, title, source_type}]
            }
        """
        context_str = _build_context(contexts)
        sources     = _build_sources(contexts)

        user_prompt = (
            f"參考資料：\n{context_str}\n\n"
            f"問題：{query}"
        )

        payload = {
            "model":  self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx":     self.num_ctx,
            },
            "messages": [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_prompt},
            ],
        }

        try:
            r = httpx.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            answer = r.json()["message"]["content"].strip()
        except httpx.ConnectError:
            answer = f"[錯誤] 無法連接 Ollama ({self.ollama_url})，請確認服務已啟動。"
        except Exception as e:
            answer = f"[錯誤] LLM 呼叫失敗：{e}"

        return {"answer": answer, "sources": sources}


# ── Quick test ─────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    # Minimal smoke test with fake context
    gen = Generator()
    fake_ctx = [
        {
            "text": "學生選課應於每學期公告之選課期間內完成，逾期不予受理。每學期最多選修學分數為25學分。",
            "url": "https://aca.nccu.edu.tw/zh/選課辦法",
            "title": "選課辦法",
            "source_type": "html",
        }
    ]
    result = gen.generate("選課最多可以修幾學分？", fake_ctx)
    print("=== Generator smoke test ===")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources: {result['sources']}")
