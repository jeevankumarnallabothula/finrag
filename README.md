# FinRAG — Financial Document Q&A with RAG

> **Ask natural language questions against financial PDFs — annual reports, 10-K filings, earnings call transcripts, and more.**

Built with **FastAPI · LangChain · OpenAI · FAISS** — production-ready RAG pipeline.

---

## Demo

```
POST /api/v1/query
{
  "doc_id": "abc123",
  "question": "What was the net revenue growth in Q4 2023?"
}

→ {
  "answer": "Net revenue grew 8% year-over-year to $24.5 billion in Q4 2023,
             driven by strong performance in the asset management segment.",
  "sources": [{ "page": 12, "snippet": "Net revenue increased 8% to $24.5B..." }]
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FinRAG Pipeline                          │
├──────────────────────────────┬──────────────────────────────────┤
│         INGESTION            │           QUERY                  │
│                              │                                  │
│  PDF Upload                  │  User Question                   │
│      │                       │       │                          │
│      ▼                       │       ▼                          │
│  PyPDF Loader                │  OpenAI Embeddings               │
│      │                       │       │                          │
│      ▼                       │       ▼                          │
│  RecursiveCharacter          │  FAISS MMR Search                │
│  TextSplitter                │  (Top-K chunks)                  │
│  (800 tokens, 100 overlap)   │       │                          │
│      │                       │       ▼                          │
│      ▼                       │  GPT-4o-mini + Prompt            │
│  OpenAI Embeddings           │       │                          │
│      │                       │       ▼                          │
│      ▼                       │  Answer + Source Citations       │
│  FAISS Vector Store          │                                  │
│  (persisted to disk)         │                                  │
└──────────────────────────────┴──────────────────────────────────┘
```

**Key design decisions:**
- **MMR retrieval** (Maximal Marginal Relevance) — reduces redundant chunks, improves answer diversity
- **Conversation memory** — 5-turn sliding window for multi-turn Q&A
- **Persisted vector stores** — FAISS indexes survive API restarts
- **Finance-tuned prompt** — instructs the LLM to cite specific numbers and stay grounded in the document

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Pydantic v2 |
| LLM Orchestration | LangChain |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (local) |
| PDF Parsing | PyPDF |
| Containerization | Docker + Docker Compose |
| Testing | pytest |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/jeevankumar-nallabothula/finrag.git
cd finrag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 3. Run

```bash
uvicorn app.main:app --reload
```

API is live at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`

### 4. Or run with Docker

```bash
docker-compose up --build
```

---

## API Reference

### `POST /api/v1/ingest`
Upload a PDF and build its vector index.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@apple_10k_2023.pdf"

# Response
{
  "doc_id": "f47ac10b-...",
  "file": "apple_10k_2023.pdf",
  "pages": 88,
  "chunks": 412,
  "message": "Document ingested successfully."
}
```

### `POST /api/v1/query`
Ask a question. Optionally pass `chat_history` for multi-turn conversation.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "f47ac10b-...",
    "question": "What are the main risk factors mentioned?",
    "chat_history": []
  }'
```

### `GET /api/v1/documents`
List all ingested document IDs.

### `DELETE /api/v1/documents/{doc_id}`
Remove a document and its vector index.

---

## Project Structure

```
finrag/
├── app/
│   ├── main.py              # FastAPI app, CORS, lifespan
│   ├── api/
│   │   ├── routes.py        # Endpoints: /ingest /query /documents
│   │   └── schemas.py       # Pydantic request/response models
│   ├── core/
│   │   └── config.py        # Settings via pydantic-settings + .env
│   └── services/
│       └── rag_service.py   # Core RAG logic: ingest, embed, retrieve, generate
├── tests/
│   └── test_api.py          # Unit tests (pytest + mocks)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Extending This Project

Ideas to take it further:
- **Swap FAISS → Pinecone** for cloud-native vector storage
- **Add Azure OpenAI support** — swap `ChatOpenAI` for `AzureChatOpenAI`
- **Multi-document querying** — merge vector stores for cross-document Q&A
- **Streaming responses** — use `StreamingResponse` for real-time LLM output
- **Authentication** — add API key middleware for production use

---

## Author

**Jeevan Kumar Nallabothula**
[LinkedIn](https://www.linkedin.com/in/jeevankumar-nallabothula) · Gen AI / LLM Engineer

---

## License

MIT
