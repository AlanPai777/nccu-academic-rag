# NCCU Academic Affairs RAG Assistant

A bilingual (Chinese/English) Q&A system for **National Chengchi University (NCCU)** academic affairs. Students and staff ask natural-language questions about course registration, graduation requirements, leave-of-absence procedures, tuition, dormitories, and more — the system returns grounded answers with cited sources, pulled from 150+ crawled NCCU subdomains.

**Topics covered**: course registration (credit limits, add/drop deadlines, cross-school enrollment) · academic records & graduation requirements · leave of absence, reinstatement, and withdrawal procedures · tuition & fee refund policies · student affairs (scholarships, dormitories, clubs, counseling) · library services · international/exchange student services · general affairs & campus regulations.

Two implementations live in this repo:

| | Agentic RAG | Classic RAG |
|---|---|---|
| Entry point | `agentic_rag.py` | `rag/classic_rag.py` |
| Status | **Production, active development** | Stable, not actively developed |
| Retrieval | LLM-driven tool loop (search/grep/fetch pages, no chunking) | Qdrant dense + FTS5 keyword → RRF fusion |
| Setup | LLM API key only | 3 local services + a 40–60 min index build |
| Data needed | Included in this repo | Included, plus a built Qdrant/FTS5 index |

## Quickstart

The preprocessed corpus ships with this repo — no crawling, no index build, just an LLM API key:

```bash
git clone <repo-url>
cd nccu-academic-rag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key (Ollama cloud or OpenRouter)

python agentic_rag.py "如何辦理休學"
```

Full setup detail (both paths, plus Classic RAG's service requirements) is in [docs/setup.md](docs/setup.md).

## Documentation

| I want to... | See |
|---|---|
| Get it running | [docs/setup.md](docs/setup.md) |
| Understand how the pipeline works | [docs/architecture.md](docs/architecture.md) |
| Run the SSE API server (connect a frontend) | [docs/setup.md](docs/setup.md#path-1b--run-the-assistant-api-server-for-a-frontend-integration) |
| Update or expand the crawled corpus | [docs/setup.md](docs/setup.md#path-2--refresh-or-expand-the-corpus) |
| See why a specific design decision was made | `docs/phase_h_agentic_rag_migration_plan.md` and the other `docs/phase_*` research documents |
| Run the classic (embedding-based) system | [docs/setup.md](docs/setup.md#classic-rags-services-only-if-running-ragclassic_ragpy), README table above |

## License

MIT
