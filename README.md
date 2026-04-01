# NCCU Academic Affairs RAG Assistant

A bilingual (Chinese/English) Retrieval-Augmented Generation (RAG) Q&A system for **National Chengchi University (NCCU)** academic affairs. Students and staff can ask natural-language questions about course registration, graduation requirements, academic regulations, and more — and receive grounded answers with cited sources.

---

## Features

- **Two-stage retrieval**: Qdrant dense search (top-50) → bge-reranker-v2-m3 cross-encoder reranking (top-5), achieving reranker scores of 0.96+ on relevant documents
- **Multilingual**: bge-m3 embedding model handles Traditional Chinese and English queries with a single model
- **Grounded answers**: granite4:3b LLM generates answers strictly from retrieved context; refuses to hallucinate
- **Intel GPU acceleration**: Embedding and LLM inference accelerated via [ipex-llm Ollama](https://github.com/intel/ipex-llm) on Intel Arc / Core Ultra iGPU
- **Web interface**: Gradio-based chat UI at `localhost:7860`

---

## Architecture

```
User Query
    │
    ▼
[Embedder] bge-m3 (Intel GPU via Ollama)
    │  1024-dim dense vector
    ▼
[Qdrant] Dense retrieval — top-50 candidates
    │  nccu_aca collection, 9,724 indexed chunks
    ▼
[Reranker] bge-reranker-v2-m3 cross-encoder — top-5
    │  score each (query, chunk) pair
    ▼
[Generator] granite4:3b (Intel GPU via Ollama)
    │  system prompt: answer only from context, cite sources
    ▼
Answer + Source URLs
```

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Embedding | `BAAI/bge-m3` | Multilingual, 1024-dim dense vectors |
| Vector DB | Qdrant (binary) | 9,724 points, cosine similarity |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder, runs on CPU |
| LLM | `granite4:3b` via Ollama | Runs on Intel GPU |
| GPU Acceleration | ipex-llm Ollama | Intel Arc / Core Ultra iGPU |
| Web UI | Gradio 6.x | Bilingual chat interface |
| HTML parsing | BeautifulSoup + lxml | Removes nav/footer noise |
| PDF parsing | pdfplumber | Traditional Chinese support |

---

## Dataset

The knowledge base was built by crawling `aca.nccu.edu.tw` (NCCU Academic Affairs Office) using a custom BFS web crawler.

### Coverage

| Item | Count |
|---|---|
| HTML pages crawled | 2,188 |
| PDF / document files | 4,110 |
| Total data size | ~1.25 GB |
| Text chunks indexed | **9,724** |

### Topics Covered

The dataset covers the full content of NCCU's Academic Affairs Office website, including:

- **Course Registration** — credit limits, add/drop deadlines, cross-school enrollment, summer courses
- **Academic Records** — grade policies, transcripts, honor rolls, academic warnings
- **Graduation** — requirements for bachelor's / master's / doctoral degrees, thesis submission, degree application
- **Leave of Absence** — suspension, reinstatement, withdrawal procedures
- **Double Major / Minor / Programs** — application requirements and procedures
- **Tuition & Fees** — fee schedules, payment deadlines, refund policies
- **International Students** — exchange programs, visiting student applications
- **Academic Regulations** — full text of university academic bylaws (PDF)
- **Forms & Downloads** — official application forms (PDF)
- **Announcements** — news from each administrative unit (Chinese and English)

Both **Chinese** (`/zh/`) and **English** (`/en/`) versions of the website are included.

### Data Availability

The crawled HTML/PDF files and `rag/chunks.jsonl` are **not included** in this repository due to size (~1.25 GB). To reproduce:

1. Crawl the target site with a BFS web crawler (saving HTML pages and PDF documents)
2. Place HTML files under `output/html/` and documents under `output/docs/`
3. Run `python rag/build_chunks.py` to generate `rag/chunks.jsonl`
4. Run `python rag/main.py --build-index` to index into Qdrant

---

## Setup

### Prerequisites

| Service | How to start | Port |
|---|---|---|
| Qdrant | `cd ~/qdrant && ./qdrant` (download binary from [qdrant.tech](https://qdrant.tech)) | 6333 |
| Ollama (ipex-llm) | `cd ~/ollama-ipex-llm-*/ && ./start-ollama.sh` | 11434 |

Pull the required models:
```bash
# From the Ollama directory
./ollama pull bge-m3
./ollama pull granite4:3b
```

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd nccu-academic-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU example; see requirements.txt for GPU options)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt
```

### Running

**Step 1 — Build chunks from crawled data** (skip if you already have `rag/chunks.jsonl`):
```bash
python rag/build_chunks.py
# Output: rag/chunks.jsonl (~9,724 chunks)
```

**Step 2 — Index into Qdrant:**
```bash
python rag/main.py --build-index
# First run only. Takes ~40-60 min depending on hardware.
```

**Step 3 — Launch the web UI:**
```bash
python rag/app.py
# Open http://localhost:7860
```

**Or use the CLI:**
```bash
python rag/main.py --query "選課最多可以修幾學分？"
python rag/main.py --query "What are the graduation requirements?"
```

---

## Project Structure

```
rag/
├── preprocess.py     # HTML/PDF content extraction, noise removal
├── chunker.py        # Fixed-size chunking (512 chars, 128 overlap)
├── build_chunks.py   # Batch preprocessing pipeline → chunks.jsonl
├── embedder.py       # bge-m3 embedding via Ollama API
├── indexer.py        # Qdrant collection creation and bulk upsert
├── retriever.py      # Dense retrieval + cross-encoder reranking
├── generator.py      # LLM answer generation with source citation
├── pipeline.py       # End-to-end RAG pipeline (query → answer)
├── main.py           # CLI entry point (--query, --build-index)
└── app.py            # Gradio web UI
```

---

## Performance Notes

| Stage | Device | Latency |
|---|---|---|
| Query embedding (bge-m3) | Intel GPU | < 1s |
| Qdrant dense search | In-memory | < 1s |
| Cross-encoder reranking (50 pairs) | CPU | 5–10min ⚠️ |
| LLM generation (granite4:3b) | Intel GPU | 20-40s |
| **First query (incl. model load)** | | **~7–10 min** |
| **Subsequent queries** | | **~5-10 min** |

The reranker is the current bottleneck as it runs on CPU. If Intel GPU drivers are available in your environment (`/dev/dri/` present in WSL2), you can use `--reranker-device xpu` to accelerate it.

---

## License

MIT
