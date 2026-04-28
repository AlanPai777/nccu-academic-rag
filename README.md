# NCCU Academic Affairs RAG Assistant

A bilingual (Chinese/English) Retrieval-Augmented Generation (RAG) Q&A system for **National Chengchi University (NCCU)** academic affairs. Students and staff can ask natural-language questions about course registration, graduation requirements, academic regulations, and more — and receive grounded answers with cited sources.

---

## Features

- **Hybrid retrieval**: Qdrant dense search (top-25) + SQLite-FTS5 keyword search with jieba (top-15) → RRF fusion → top-5 chunks
- **Two-layer caching**: SQLite-backed response cache (skip everything) and retrieval cache (skip search, run LLM only). Cache persists across restarts.
- **Multilingual**: embeddinggemma embedding model handles Traditional Chinese and English queries
- **Grounded answers**: qwen2.5:7b LLM generates answers strictly from retrieved context; refuses to hallucinate
- **Dual LLM mode**: local `qwen2.5:7b` via Ollama (Intel GPU) or `gpt-oss-20b` via OpenRouter API (free tier)
- **Intel GPU acceleration**: LLM inference accelerated via [ipex-llm Ollama](https://github.com/intel/ipex-llm) on Intel Arc / Core Ultra iGPU
- **Web interface**: Gradio-based chat UI at `localhost:7860`

---

## Architecture

```
Query
  │
  ▼
[RAGCache] Layer 1: response cache ──HIT──▶ instant return
  │ MISS
  ▼
[RAGCache] Layer 2: retrieval cache ─HIT──▶ skip to LLM
  │ MISS
  ├──[Dense]   embeddinggemma (CPU, port 11435)
  │            → Qdrant top-25 candidates
  │            → navigation chunk filter
  │
  ├──[Keyword] FTS5 + jieba, top-15 candidates
  │
  ▼
[RRF Fusion] Reciprocal Rank Fusion (k=60) → top-5
  │
  ▼
[Generator] qwen2.5:7b (Intel GPU, port 11434)
            OR gpt-oss-20b via OpenRouter API
  │
  ▼
Answer + Source URLs
```

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Embedding | `embeddinggemma` (Gemma3-based) | 768-dim dense, 300M params, CPU via standalone Ollama |
| Vector DB | Qdrant (binary) | ~14,430 points, cosine similarity |
| Keyword search | SQLite-FTS5 + jieba | BM25 ranking, Chinese word segmentation |
| Fusion | Reciprocal Rank Fusion (RRF) | k=60, replaces cross-encoder reranker, <1ms |
| LLM (local) | `qwen2.5:7b` via Ollama | Runs on Intel GPU (ipex-llm) |
| LLM (API) | `gpt-oss-20b` via OpenRouter | Free tier, 200 req/day |
| GPU Acceleration | ipex-llm Ollama | Intel Arc / Core Ultra iGPU |
| RAG Framework | LlamaIndex (retriever layer) | OllamaEmbedding, QdrantVectorStore, QueryFusionRetriever |
| Web UI | Gradio 6.x | Bilingual chat interface |
| HTML parsing | BeautifulSoup + lxml | Removes nav/footer noise; preserves markdown links |
| PDF parsing | pdfplumber | Traditional Chinese support |
| Cache | SQLite (two-layer) | Response + retrieval cache, persists across restarts |

---

## Dataset

The knowledge base is built from 6 NCCU subdomains crawled with a custom BFS web crawler, plus supplementary PDFs fetched from moltke.nccu.edu.tw and newdoc.nccu.edu.tw.

### Coverage

| Subdomain | Office | Extracted records | Supplementary PDFs |
|---|---|---|---|
| `aca.nccu.edu.tw` | Academic Affairs (教務處) | 6,243 | 40 (moltke + newdoc) |
| `osa.nccu.edu.tw` | Student Affairs (學務處) | 9,374 | 46 (moltke + newdoc) |
| `www.lib.nccu.edu.tw` | Library (圖書館) | 3,161 | 32 (moltke + newdoc + ref.lib) |
| `learning.nccu.edu.tw` | Teaching & Learning Center | 4,669 | — |
| `nccuga.nccu.edu.tw` | General Affairs (總務處) | 156 | — |
| `oic.nccu.edu.tw` | International Cooperation (國際合作事務處) | 340 | — |
| **Total** | | **23,943** | **118** |

### Topics Covered

- **Course Registration** — credit limits, add/drop deadlines, cross-school enrollment, summer courses
- **Academic Records** — grade policies, transcripts, honor rolls, academic warnings
- **Graduation** — requirements for bachelor's / master's / doctoral degrees, thesis submission, degree application
- **Leave of Absence** — suspension, reinstatement, withdrawal procedures
- **Double Major / Minor / Programs** — application requirements and procedures
- **Tuition & Fees** — fee schedules, payment deadlines, refund policies
- **Student Affairs** — scholarship applications, dormitory forms, student clubs, counseling services
- **Library Services** — borrowing rules, database access, room booking, interlibrary loan forms
- **Research Databases** — guides for Web of Science, Scopus, TEJ, LSEG, Turnitin, Bloomberg, SciVal, LawBank
- **International Students** — exchange programs, visiting student applications, OIC services
- **Teaching & Learning** — course development resources, TA guidelines, teaching evaluation
- **Academic Regulations** — full text of university academic bylaws (PDF) with official forms
- **Forms & Downloads** — 118 official application forms (PDF) with direct download links preserved

Both **Traditional Chinese** and **English** versions of pages are included where available.

### Data Availability

Crawled HTML/PDF files and `rag/chunks.jsonl` are **not included** in this repository due to size. To reproduce:

1. Crawl each subdomain using the [NCCU-Crawler](https://github.com/alanpai/NCCU-Crawler)
2. Run `python rag/preprocess_all.py --subdomain <name>` for each subdomain
3. Run `python rag/build_chunks.py` to generate `rag/chunks.jsonl`
4. Run `python rag/main.py --build-index` to index into Qdrant and build the FTS5 keyword index

---

## Setup

### Prerequisites — 3 services required

| Service | How to start | Port |
|---|---|---|
| Qdrant | `cd ~/qdrant && ./qdrant` | 6333 |
| ipex-llm Ollama (LLM) | `cd ~/ollama-ipex-llm-*/ && ./start-ollama.sh` | 11434 |
| Standalone Ollama (embedding) | `OLLAMA_HOST=0.0.0.0:11435 /usr/local/bin/ollama serve` | 11435 |

Pull the required models:
```bash
# From the ipex-llm Ollama directory (port 11434)
./ollama pull qwen2.5:7b

# From standalone Ollama (port 11435)
OLLAMA_HOST=localhost:11435 ollama pull embeddinggemma
```

Two separate Ollama instances are required because embeddinggemma (Gemma3 architecture) is incompatible with the ipex-llm SYCL backend, and embedding + LLM cannot coexist in memory (each ~5GB on 16GB RAM). `keep_alive=0` on embedding requests ensures the embedding model is unloaded before the LLM is needed.

### Installation

```bash
git clone <repo-url>
cd nccu-academic-rag

python3 -m venv venv
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Building the index

**Step 1 — Build chunks from crawled data** (skip if you already have `rag/chunks.jsonl`):
```bash
python rag/build_chunks.py
# Output: rag/chunks.jsonl (~14,430 chunks with text + text_clean dual fields)
```

**Step 2 — Index into Qdrant and build FTS5:**
```bash
python rag/main.py --build-index
# Embeds all chunks (embeddinggemma, CPU) and builds SQLite FTS5 index.
# Takes ~40-60 min. Qdrant collection: nccu_aca_v2_embeddinggemma (768-dim)
```

### Running

**Web UI:**
```bash
python rag/main.py --app
# Open http://localhost:7860
```

**CLI:**
```bash
python rag/main.py --query "選課最多可以修幾學分？"
python rag/main.py --query "What are the graduation requirements?"
python rag/main.py --query "選課辦法" --provider openrouter   # use API instead
python rag/main.py --query "選課辦法" --no-cache              # bypass cache
```

---

## Project Structure

```
rag/
├── preprocess.py       # HTML/PDF content extraction; preserves markdown links [text](url)
├── preprocess_all.py   # Batch extraction → output/<subdomain>/extracted_texts.jsonl
├── chunker.py          # HTML chunking (800 chars, link-aware); PDF chunking (512 tokens)
├── build_chunks.py     # Batch pipeline → chunks.jsonl (with text + text_clean fields)
├── embedder.py         # embeddinggemma via Ollama REST API (port 11435)
├── indexer.py          # Qdrant collection creation + FTS5 index build
├── keyword_store.py    # SQLite-FTS5 BM25 keyword search with jieba segmentation
├── retriever.py        # LlamaIndex hybrid retrieval: dense + keyword → RRF fusion
├── cache.py            # Two-layer SQLite cache: response + retrieval (persistent)
├── generator.py        # LLM generation: local qwen2.5:7b or OpenRouter API
├── pipeline.py         # End-to-end pipeline: cache → retrieve → generate
├── main.py             # CLI entry point
└── app.py              # Gradio web UI
```

---

## Performance

| Stage | Device | Latency |
|---|---|---|
| Query embedding (embeddinggemma) | CPU (port 11435) | 1–3s |
| Qdrant dense search | In-memory | <1s |
| FTS5 keyword search | CPU | <50ms |
| RRF fusion | CPU | <1ms |
| LLM generation (qwen2.5:7b) | Intel GPU (port 11434) | 40–90s |
| **Cache hit (response)** | — | **<50ms** |
| **Cache hit (retrieval)** | — | **~40–90s (LLM only)** |
| **First query (cold, no cache)** | | **~1.5–3 min** |
| **Repeated queries** | | **<50ms (cached)** |

RRF replaced the previous bge-reranker-v2-m3 cross-encoder (5–10 min on CPU), reducing uncached query time from 7–10 min to ~1.5–3 min.

---

## License

MIT
