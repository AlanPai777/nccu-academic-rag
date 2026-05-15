# NCCU Academic Affairs RAG Assistant

A bilingual (Chinese/English) Q&A system for **National Chengchi University (NCCU)** academic affairs. Students and staff can ask natural-language questions about course registration, graduation requirements, leave of absence procedures, and more — and receive grounded answers with cited sources.

Two implementations are provided:

| | Agentic RAG (Proto 3) | Agentic RAG (Proto 1) | Classic RAG |
|---|---|---|---|
| Entry point | `rag/proto3_langgraph.py` | `rag/proto1_direct.py` | `rag/main.py` |
| Status | **Active development** | Baseline / reference | Stable |
| Retrieval | LangGraph pipeline: router → skill → synthesis | LLM-driven tool loop | Qdrant dense + FTS5 → RRF |
| Setup | Ollama or OpenRouter API key | Ollama or OpenRouter API key | 3 services + 40–60 min index build |
| Data needed | `extracted_texts.jsonl` (included) | `extracted_texts.jsonl` (included) | Qdrant collection + FTS5 index |

---

## Agentic RAG (Active Development)

### Architecture — Proto 3 (LangGraph)

```
Query
  │
  ▼
router_node — 2-layer: keyword match → LLM fallback
  │
  ├─(PROCEDURE)──▶ retrieval_node (ProcedureSkill: grep → links → form)
  ├─(KNOWLEDGE)──▶ retrieval_node (LLM agent loop, up to 10 turns)
  │                  ├── grep_texts(pattern, subdomain)
  │                  ├── get_page(url)
  │                  ├── extract_links(url)
  │                  ├── get_children(url)
  │                  └── get_form(form_id)
  └─(CONTACT)────▶ office_lookup_node (skip retrieval)
                        │
                   office_lookup_node — OfficeLookupSkill (KNOWN_CONTACTS, always-on)
                        │
                   synthesis_node ◀──────────────────────╮
                        │                                │ correction_hint (max 2 retries)
                   self_eval_node ──(retry if FAIL)──────╯
                        │
                       END
```

**Phase E features** (2026-05-14):
- **E1** Query router: keyword layer → LLM fallback [Adaptive-RAG]
- **E3** KNOWN_CONTACTS always-on injection (not dependent on grep success)
- **E4** Self-eval node: correction_hint + retry loop [Self-RAG]
- **E5** Doom Loop Detector: max turns + duplicate call detection + format validation
- **E6** Staleness warning appended to every answer [Astute RAG]
- **E7** Parametric knowledge fallback when retrieval finds nothing [Astute RAG]

Inspired by [ELITE](https://github.com/tjzvbokbnft/ELITE-Embedding-Less-retrieval-with-Iterative-Text-Exploration) (iterative grep) and [LongRAG](https://arxiv.org/abs/2406.15319) (whole-page retrieval without chunking).

### Quickstart

```bash
git clone <repo-url>
cd nccu-academic-rag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Configure API keys:**
```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and fill in the corresponding API key
```

**Option A — Ollama cloud (gemma4:31b, recommended):**
```bash
# 1. Sign in:  ollama signin   OR create a key at https://ollama.com/settings/keys
# 2. In .env:  LLM_PROVIDER=ollama  /  OLLAMA_API_KEY=<your key>
python -m rag.proto3_langgraph "如何辦理休學"
```

**Option B — OpenRouter API:**
```bash
# In .env:  LLM_PROVIDER=openrouter  /  OPENROUTER_API_KEY=<your key>
python -m rag.proto3_langgraph "如何辦理休學"
```

No Qdrant, no embedding model, no index building required. `extracted_texts.jsonl` for all 15 subdomains is included in this repo (~92MB).

### Running

```bash
# Proto 3 (LangGraph, recommended)
python -m rag.proto3_langgraph "如何辦理休學"
python -m rag.proto3_langgraph "出納組的電話"
python -m rag.proto3_langgraph "選課上限幾學分"
python -m rag.proto3_langgraph "如何辦理休學" --no-eval

# Proto 1 (simple agent loop, baseline)
python -m rag.proto1_direct "如何辦理休學"           # Mode 3: full agent (default)
python -m rag.proto1_direct "如何辦理休學" --mode 1  # Mode 1: pure parametric
python -m rag.proto1_direct "如何辦理休學" --mode 2  # Mode 2: single doc (synthesis test)
python -m rag.proto1_direct "如何辦理休學" --debug   # full tool outputs per turn
```

Eval runs automatically after each query (26 pts, 13 criteria).

### Eval Scores — "如何辦理休學"

| Pipeline | Model | Score | Notes |
|---|---|---|---|
| Proto 3 (LangGraph) | Gemma4:31b | **26/26** | Phase E complete (2026-05-14) |
| Proto 1 Mode 3 (full agent) | Sonnet 4.6 | 26/26 | Gold standard |
| Proto 1 Mode 3 (full agent) | Gemma4:31b | 13–15/26 | 4 tool calls; no router/self-eval |
| Proto 1 Mode 2 (single doc) | Gemma4:31b | 13/26 | QP-T01-03-02 only; tests synthesis |
| Proto 1 Mode 1 (parametric) | Gemma4:31b | 7/26 | No retrieval |

### Key Modules

| Module | Role |
|---|---|
| `rag/proto3_langgraph.py` | LangGraph pipeline: router → retrieval → office_lookup → synthesis → self_eval |
| `rag/router.py` | 2-layer query router: keyword match → LLM fallback (PROCEDURE / CONTACT / KNOWLEDGE) |
| `rag/agent_tools.py` | 5 core tools: `grep_texts`, `get_page`, `extract_links`, `get_children`, `get_form` |
| `rag/proto1_direct.py` | Baseline agent loop + Mode 1/2/3 + `--verbose`/`--debug` |
| `rag/llm_client.py` | Unified LLM client: Ollama + OpenRouter; `max_tokens` defaults to 8192 |
| `rag/eval.py` | 26 pts, 13 criteria (v2, 2026-05-02) |
| `rag/skills/office_lookup_skill.py` | KNOWN_CONTACTS (~18 staff, Playwright-extracted) + KNOWN_FLOORS + extracted_texts lookup |
| `rag/skills/procedure_skill.py` | ProcedureSkill: deterministic 3-step retrieval (grep → links → form) |

---

## Classic RAG (Stable)

### Architecture

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

### Prerequisites — 3 services required

| Service | How to start | Port |
|---|---|---|
| Qdrant | `cd ~/qdrant && ./qdrant` | 6333 |
| ipex-llm Ollama (LLM) | `cd ~/ollama-ipex-llm-*/ && ./start-ollama.sh` | 11434 |
| Standalone Ollama (embedding) | `OLLAMA_HOST=0.0.0.0:11435 /usr/local/bin/ollama serve` | 11435 |

```bash
./ollama pull qwen2.5:7b
OLLAMA_HOST=localhost:11435 ollama pull embeddinggemma
```

Two separate Ollama instances are required because embeddinggemma (Gemma3 architecture) is incompatible with the ipex-llm SYCL backend, and embedding + LLM cannot coexist in 16GB RAM (`keep_alive=0` ensures the embedding model is unloaded before the LLM is needed).

### Building the Index

```bash
# Step 1 — build chunks (skip if rag/chunks.jsonl already exists)
python rag/build_chunks.py
# Output: rag/chunks.jsonl (~14,430 chunks, text + text_clean dual fields)

# Step 2 — index into Qdrant + FTS5 (~40-60 min)
python rag/main.py --build-index
```

### Running

```bash
python rag/main.py --app                                        # web UI (localhost:7860)
python rag/main.py --query "選課最多可以修幾學分？"
python rag/main.py --query "選課辦法" --provider openrouter    # OpenRouter API
python rag/main.py --query "選課辦法" --no-cache               # bypass cache
```

### Performance

| Stage | Device | Latency |
|---|---|---|
| Query embedding | CPU (port 11435) | 1–3s |
| Qdrant dense search | In-memory | <1s |
| FTS5 keyword search | CPU | <50ms |
| RRF fusion | CPU | <1ms |
| LLM generation (qwen2.5:7b) | Intel GPU (port 11434) | 40–90s |
| **Cache hit (response)** | — | **<50ms** |
| **First query (cold)** | — | **~1.5–3 min** |

---

## Dataset

### Coverage

| Subdomain | Office | Records |
|---|---|---|
| `aca.nccu.edu.tw` | Academic Affairs / 教務處 | 6,221 |
| `osa.nccu.edu.tw` | Student Affairs / 學務處 | 9,374 |
| `www.lib.nccu.edu.tw` | Library / 圖書館 | 3,161 |
| `learning.nccu.edu.tw` | Teaching & Learning Center | 4,669 |
| `oic.nccu.edu.tw` | International Cooperation / 國合處 | 340 |
| `nccuga.nccu.edu.tw` | General Affairs / 總務處 (main) | 156 |
| `cashier.nccu.edu.tw` | Cashier / 出納組 | 103 |
| `dean.nccu.edu.tw` | Dean's Office / 總務長室 | 46 |
| `docu.nccu.edu.tw` | Document Management / 文書組 | 157 |
| `aff.nccu.edu.tw` | General Affairs / 事務組 | 135 |
| `environ.nccu.edu.tw` | Environment / 環安組 | 335 |
| `wealth.nccu.edu.tw` | Asset Management / 財產組 | 281 |
| `mend.nccu.edu.tw` | Maintenance / 修繕組 | 94 |
| `police.nccu.edu.tw` | Campus Police / 警衛隊 | 93 |
| `cpds.nccu.edu.tw` | Procurement / 採購組 | 66 |
| **Total** | | **25,231** |

Supplementary PDFs (official forms from moltke.nccu.edu.tw and newdoc.nccu.edu.tw): ~118 forms across aca, osa, and www.lib.

> **Note:** HTML pages under nccuga sub-units (cashier, dean, docu, etc.) are JavaScript-rendered; only PDF documents are extractable. Staff contact information for these offices is hardcoded in `OfficeLookupSkill` via Playwright extraction.

### Topics Covered

- **Course Registration** — credit limits, add/drop deadlines, cross-school enrollment
- **Academic Records** — grade policies, transcripts, academic warnings
- **Graduation** — bachelor's / master's / doctoral requirements, thesis submission
- **Leave of Absence** — suspension, reinstatement, withdrawal procedures and forms
- **Tuition & Fees** — schedules, refund policies (full / 2/3 / 1/3 by week)
- **Student Affairs** — scholarships, dormitory, clubs, counseling
- **Library Services** — borrowing rules, database access, interlibrary loan
- **International Students** — exchange programs, OIC services
- **General Affairs** — facilities, campus safety, procurement
- **Academic Regulations** — full bylaws (PDF) with official application forms

### Data Availability

`extracted_texts.jsonl` (preprocessed text, ~92MB total) and `supplementary_map.json` (moltke/newdoc form index) for all 15 subdomains are **included in this repository** — no crawling required to run the agentic RAG.

Raw HTML/PDF files (~8.5GB) are excluded. To reproduce from scratch:
1. Crawl each subdomain using [NCCU-Crawler](https://github.com/alanpai/NCCU-Crawler)
2. `python rag/preprocess_all.py` — regenerate `extracted_texts.jsonl`
3. `python rag/build_chunks.py` — regenerate `rag/chunks.jsonl` (classic RAG only)
4. `python rag/main.py --build-index` — rebuild Qdrant + FTS5 (classic RAG only)

---

## Project Structure

```
rag/
├── proto1_direct.py        # Agentic RAG: agent loop, Mode 1/2/3, --verbose/--debug
├── agent_tools.py          # 5 tools: grep_texts, get_page, extract_links, get_children, get_form
├── llm_client.py           # Unified LLM client: Ollama + OpenRouter
├── eval.py                 # Evaluation: 26 pts, 13 criteria (v2)
├── skills/
│   ├── office_lookup_skill.py  # KNOWN_CONTACTS + KNOWN_FLOORS + extracted_texts lookup
│   └── procedure_skill.py      # Keyword detection for procedure-type questions
├── preprocess.py           # HTML/PDF extraction; smart table → Markdown; OCR fallback
├── preprocess_all.py       # Batch extraction → output/<subdomain>/extracted_texts.jsonl
├── chunker.py              # HTML (800 chars) + PDF (512 tokens) chunking
├── build_chunks.py         # → rag/chunks.jsonl (classic RAG)
├── embedder.py             # embeddinggemma via Ollama (port 11435)
├── indexer.py              # Qdrant collection + FTS5 index build
├── keyword_store.py        # SQLite-FTS5 + jieba BM25
├── retriever.py            # LlamaIndex hybrid: dense + keyword → RRF fusion
├── cache.py                # Two-layer SQLite cache (response + retrieval)
├── generator.py            # LLM generation: qwen2.5:7b or OpenRouter
├── pipeline.py             # Classic RAG end-to-end
├── main.py                 # Classic RAG CLI entry point
└── app.py                  # Gradio web UI

output/
└── <subdomain>/
    ├── extracted_texts.jsonl   # Preprocessed text (tracked in git, ~92MB total)
    └── supplementary_map.json  # Moltke/newdoc form index (tracked in git)
```

---

## License

MIT
