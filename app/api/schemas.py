"""
Request / Response schemas — Pydantic models for API validation and docs.
"""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    doc_id: str = Field(..., description="ID of the ingested document to query")
    question: str = Field(..., min_length=5, description="Natural language question")
    chat_history: Optional[list[tuple[str, str]]] = Field(
        default=None,
        description="Previous (question, answer) pairs for multi-turn conversation"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "doc_id": "abc123",
                "question": "What was the net revenue in Q4 2023?",
                "chat_history": []
            }
        }
    }


class SourceChunk(BaseModel):
    page: int | str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    doc_id: str


class IngestResponse(BaseModel):
    doc_id: str
    file: str
    pages: int
    chunks: int
    message: str


class DocumentListResponse(BaseModel):
    documents: list[str]
    count: int


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str


class ErrorResponse(BaseModel):
    detail: str
