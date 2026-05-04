"""
rag/llm_client.py
Unified LLM client supporting two providers:
  - ollama (default): gemma4:31b-cloud via Ollama cloud native API
  - openrouter: any model via OpenRouter OpenAI-compatible API

Switch provider:  export LLM_PROVIDER=openrouter
Default:          LLM_PROVIDER=ollama
"""

import json
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

# ── Provider selection ────────────────────────────────────────────────────────
PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# ── Ollama cloud ──────────────────────────────────────────────────────────────
# Auth: ollama signin  OR  export OLLAMA_API_KEY=...
# Docs: https://ollama.com → Settings → API Keys
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST",  "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")

# ── OpenRouter ────────────────────────────────────────────────────────────────
# Docs: https://openrouter.ai
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")


def get_active_model() -> str:
    """Return the currently configured model name."""
    return OPENROUTER_MODEL if PROVIDER == "openrouter" else OLLAMA_MODEL


# ── Unified interface ─────────────────────────────────────────────────────────

def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str | None, list | None]:
    """
    Chat with tool definitions. Provider-agnostic.

    max_tokens:
        None（預設）= 不限制輸出長度，讓模型遇到 EOS token 自行停止。
            - Ollama：num_predict=-1
            - OpenRouter：省略 max_tokens 欄位（傳 -1 會 API error）
        int = 硬性截斷上限；適合快速測試或避免計費超支。
        注意：max_tokens 只限制「輸出」token 數，不影響模型能看到的 context 長度。

    Returns:
        (content, tool_calls)
        content:    string final answer, or None if LLM only called tools
        tool_calls: list of dicts in unified format (see below), or None

    tool_calls format (same for both providers):
        [{"id": "...", "type": "function",
          "function": {"name": "...", "arguments": '{"key": "value"}'}}]
        Note: arguments is always a JSON string.
    """
    full_messages = (
        [{"role": "system", "content": system_prompt}] + messages
        if system_prompt else messages
    )
    if PROVIDER == "openrouter":
        return _openrouter_chat(full_messages, tools, max_tokens)
    return _ollama_chat(full_messages, tools, max_tokens)


def simple_chat(messages: list[dict], max_tokens: int | None = None) -> str:
    """
    Chat without tools; returns answer string.

    max_tokens=None（預設）= 不限制輸出長度，適合完整答案評估。
    傳入 int 可限制輸出，適合快速冒煙測試（如 __main__ 的自我介紹測試）。
    """
    content, _ = chat_with_tools(messages, tools=[], max_tokens=max_tokens)
    return content or ""


# ── Ollama native client ──────────────────────────────────────────────────────

def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    """
    Convert unified message history to Ollama format.
    Key difference: tool_calls[].function.arguments must be a dict (not JSON string).
    Also strips unsupported fields (id, type) from tool_calls.
    """
    import copy
    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg = copy.copy(msg)
            fixed_tcs = []
            for tc in msg["tool_calls"]:
                args = tc["function"]["arguments"]
                fixed_tcs.append({
                    "function": {
                        "name":      tc["function"]["name"],
                        "arguments": json.loads(args) if isinstance(args, str) else args,
                    }
                })
            msg = {**msg, "tool_calls": fixed_tcs}
        result.append(msg)
    return result


def _ollama_chat(
    messages: list[dict],
    tools: list[dict],
    max_tokens: int | None,
) -> tuple[str | None, list | None]:
    from ollama import Client

    client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )
    # Ollama 用 num_predict 控制輸出長度。
    # 本地 Ollama：-1 = 無限制（模型遇到 EOS 自行停止）
    # Ollama cloud API：不接受 -1，必須傳正整數（回傳 400 error）
    # → None 時改用 8192，實際上對所有查詢已足夠（休學答案約 1000–2000 tokens）
    num_predict = 8192 if max_tokens is None else max_tokens
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=_to_ollama_messages(messages),
        tools=tools if tools else None,
        options={"num_predict": num_predict},
    )
    msg = response.message
    content = msg.content or None

    if not msg.tool_calls:
        return content, None

    # Normalize Ollama tool_calls → unified dict format
    normalized = []
    for tc in msg.tool_calls:
        args = tc.function.arguments
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        normalized.append({
            "id":       f"call_{uuid.uuid4().hex[:8]}",
            "type":     "function",
            "function": {"name": tc.function.name, "arguments": args},
        })
    return content, normalized


# ── OpenRouter client ─────────────────────────────────────────────────────────

def _openrouter_chat(
    messages: list[dict],
    tools: list[dict],
    max_tokens: int | None,
) -> tuple[str | None, list | None]:
    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)
    kwargs: dict = {
        "model":       OPENROUTER_MODEL,
        "messages":    messages,
        "temperature": 0,
    }
    # OpenRouter 不接受 max_tokens=-1（會回傳 API error）。
    # 省略這個欄位才是「不限制」；有值時才加入 kwargs。
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    content = msg.content or None

    if not msg.tool_calls:
        return content, None

    normalized = [
        {
            "id":       tc.id,
            "type":     "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in msg.tool_calls
    ]
    return content, normalized


# ── Framework helpers (Prototype 2 & 3) ──────────────────────────────────────

def get_langchain_llm():
    """LangChain ChatOpenAI — OpenRouter only (Prototype 3)."""
    from langchain_openai import ChatOpenAI
    if PROVIDER != "openrouter":
        raise ValueError("LangChain integration requires LLM_PROVIDER=openrouter")
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_base=OPENROUTER_BASE,
        openai_api_key=OPENROUTER_API_KEY,
        temperature=0,
    )


def get_llamaindex_llm():
    """LlamaIndex LLM — OpenRouter only (Prototype 2)."""
    if PROVIDER != "openrouter":
        raise ValueError("LlamaIndex integration requires LLM_PROVIDER=openrouter")
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    return LlamaOpenAI(
        model=OPENROUTER_MODEL,
        api_base=OPENROUTER_BASE,
        api_key=OPENROUTER_API_KEY,
    )


if __name__ == "__main__":
    print(f"Provider : {PROVIDER}")
    print(f"Model    : {get_active_model()}")
    print("Sending test message...")
    answer = simple_chat(
        [{"role": "user", "content": "你好，請用一句話介紹你自己"}],
        max_tokens=100,
    )
    print(f"Response : {answer}")
