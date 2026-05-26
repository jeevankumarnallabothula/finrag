"""
API Routes — /ingest, /query, /documents, /documents/{doc_id}
"""

import os
import uuid
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status

from app.api.schemas import (
    QueryRequest, QueryResponse, IngestResponse,
    DocumentListResponse, DeleteResponse, SourceChunk,
)
from app.services.rag_service import RAGService, get_rag_service
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ── POST /ingest ───────────────────────────────────────────────────────────
@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a financial PDF",
    tags=["Documents"],
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file (annual report, 10-K, earnings call, etc.)"),
    rag: RAGService = Depends(get_rag_service),
):
    """
    Upload a PDF financial document. The API will:
    1. Extract text from all pages
    2. Split into overlapping chunks
    3. Generate embeddings via OpenAI
    4. Store in a FAISS vector index

    Returns a `doc_id` — use this in `/query` requests.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    # Save uploaded file temporarily
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid.uuid4())
    file_path = upload_dir / f"{doc_id}_{file.filename}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved upload: {file_path}")

        result = rag.ingest_document(str(file_path), doc_id=doc_id)
        return IngestResponse(
            **result,
            message=f"Document ingested successfully. Use doc_id '{doc_id}' to query."
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


# ── POST /query ────────────────────────────────────────────────────────────
@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about an ingested document",
    tags=["Q&A"],
)
async def query_document(
    request: QueryRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """
    Ask a natural language question against a previously ingested document.

    Supports multi-turn conversation — pass previous `(question, answer)` pairs
    in `chat_history` to maintain context across turns.

    Example questions:
    - "What was the total revenue in fiscal year 2023?"
    - "Summarize the key risk factors."
    - "What is the company's debt-to-equity ratio?"
    - "How did operating income change year-over-year?"
    """
    try:
        result = rag.query(
            doc_id=request.doc_id,
            question=request.question,
            chat_history=request.chat_history or [],
        )
        return QueryResponse(
            answer=result["answer"],
            sources=[SourceChunk(**s) for s in result["sources"]],
            doc_id=result["doc_id"],
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


# ── GET /documents ─────────────────────────────────────────────────────────
@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
    tags=["Documents"],
)
async def list_documents(rag: RAGService = Depends(get_rag_service)):
    docs = rag.list_documents()
    return DocumentListResponse(documents=docs, count=len(docs))


# ── DELETE /documents/{doc_id} ─────────────────────────────────────────────
@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete an ingested document and its vector index",
    tags=["Documents"],
)
async def delete_document(
    doc_id: str,
    rag: RAGService = Depends(get_rag_service),
):
    deleted = rag.delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found."
        )
    return DeleteResponse(
        doc_id=doc_id,
        deleted=True,
        message="Document and vector index deleted successfully."
    )
