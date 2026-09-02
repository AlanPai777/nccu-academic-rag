# Setup

This is the operational runbook: environment install, and the two separate
paths — "just run the assistant" (no data prep needed) vs. "I want to
refresh or expand the corpus." For what the system is and how it's
architected, see the main [README](../README.md) and
[architecture.md](architecture.md).

## Path 1 — Just run it (no data prep needed)

The repo ships with preprocessed data already committed — 154 subdomains'
worth of `extracted_texts.jsonl`/`supplementary_map.json` under `output/`
are tracked in git (see `.gitignore`'s explicit un-ignore rules). You do
**not** need to crawl or preprocess anything to run the production system.

```bash
git clone <repo-url>
cd nccu-academic-rag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Configure an LLM provider:

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and fill in the corresponding key
```

**Option A — Ollama cloud (`gemma4:31b-cloud`, recommended):**
```bash
# Sign in: ollama signin   OR create a key at https://ollama.com/settings/keys
# In .env:  LLM_PROVIDER=ollama  /  OLLAMA_API_KEY=<your key>
python agentic_rag.py "如何辦理休學"
```

**Option B — OpenRouter API:**
```bash
# In .env:  LLM_PROVIDER=openrouter  /  OPENROUTER_API_KEY=<your key>
python agentic_rag.py "如何辦理休學"
```

That's it — no Qdrant, no embedding model, no index build. This is the
path for the production agentic RAG pipeline (`agentic_rag.py`).

## Path 2 — Refresh or expand the corpus

Only needed if you want to update stale data, add a new subdomain, or
reproduce the dataset from scratch. Three steps, each independently
re-runnable:

```bash
# Step 1 — crawl (separate sibling project, not part of this repo)
# See NCCU-Crawler: https://github.com/alanpai/NCCU-Crawler
# Produces raw HTML/PDF under output/<subdomain>/ (not tracked in this repo's git)

# Step 2 — extract text from the crawled raw files
python rag/preprocess_all.py --subdomain aca   # single subdomain
python rag/preprocess_all.py                   # all subdomains
# Writes output/<subdomain>/extracted_texts.jsonl (this is the file agentic_rag.py reads)

# Step 3 — only if you also run Classic RAG (see below), rebuild its chunk/index files
python rag/build_chunks.py                     # -> rag/chunks.jsonl
python rag/classic_rag.py --build-index        # embed + index into Qdrant + FTS5 (~40-60 min)
python rag/classic_rag.py --build-fts          # FTS5 only, fast, no embedding
```

`agentic_rag.py` never needs Step 3 — it reads `extracted_texts.jsonl`
directly via FTS5 (`rag/fts_proto3.db`-style keyword search) and grep, no
vector index involved.

## Classic RAG's services (only if running `rag/classic_rag.py`)

The agentic pipeline (`agentic_rag.py`, production) needs none of this —
skip this section unless you're specifically running the classic
(embedding-based) system, which has been dormant for a while and is kept
around as a stable reference implementation, not active development.

Classic RAG needs 3 services running concurrently, in separate terminals:

| Service | How to start | Port |
|---|---|---|
| Qdrant (vector DB) | `cd ~/qdrant && ./qdrant` | 6333 |
| ipex-llm Ollama (LLM inference) | `cd ~/ollama-ipex-llm-*/ && ./start-ollama.sh` | 11434 |
| Standalone Ollama (embedding) | `OLLAMA_HOST=0.0.0.0:11435 /usr/local/bin/ollama serve` | 11435 |

```bash
./ollama pull qwen2.5:7b
OLLAMA_HOST=localhost:11435 ollama pull embeddinggemma
```

Two separate Ollama instances are required because `embeddinggemma`
(Gemma3 architecture) is incompatible with the ipex-llm SYCL backend, and
embedding + LLM generation can't coexist in 16GB RAM — `keep_alive=0` on
embedding requests unloads `embeddinggemma` before the LLM is needed.

Once running:

```bash
python rag/classic_rag.py --app                 # web UI, http://localhost:7860
python rag/classic_rag.py --query "選課最多可以修幾學分？"
```

## Environment notes

- `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  before `pip install -r requirements.txt` if you don't already have a
  CPU-only torch build — the default PyPI wheel pulls CUDA dependencies
  this project doesn't use.
- `.env` is gitignored — never commit real keys. `.env.example` documents
  every variable, including the currently-unused Langfuse tracing block.
