"""
Unit tests for FinRAG API.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.rag_service import RAGService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_rag():
    rag = MagicMock(spec=RAGService)
    rag.list_documents.return_value = ["doc-001", "doc-002"]
    rag.query.return_value = {
        "answer": "Net revenue was $24.5 billion in Q4 2023.",
        "sources": [{"page": 12, "snippet": "Net revenue increased 8% to $24.5 billion..."}],
        "doc_id": "doc-001",
    }
    rag.delete_document.return_value = True
    return rag


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "FinRAG" in response.json()["app"]


def test_list_documents(client, mock_rag):
    with patch("app.api.routes.get_rag_service", return_value=mock_rag):
        response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert "doc-001" in data["documents"]


def test_query_success(client, mock_rag):
    with patch("app.api.routes.get_rag_service", return_value=mock_rag):
        response = client.post("/api/v1/query", json={
            "doc_id": "doc-001",
            "question": "What was the net revenue in Q4 2023?",
        })
    assert response.status_code == 200
    data = response.json()
    assert "24.5 billion" in data["answer"]
    assert len(data["sources"]) == 1


def test_query_document_not_found(client, mock_rag):
    mock_rag.query.side_effect = ValueError("Document 'bad-id' not found.")
    with patch("app.api.routes.get_rag_service", return_value=mock_rag):
        response = client.post("/api/v1/query", json={
            "doc_id": "bad-id",
            "question": "What is the revenue?",
        })
    assert response.status_code == 404


def test_delete_document(client, mock_rag):
    with patch("app.api.routes.get_rag_service", return_value=mock_rag):
        response = client.delete("/api/v1/documents/doc-001")
    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_delete_document_not_found(client, mock_rag):
    mock_rag.delete_document.return_value = False
    with patch("app.api.routes.get_rag_service", return_value=mock_rag):
        response = client.delete("/api/v1/documents/nonexistent")
    assert response.status_code == 404


def test_ingest_non_pdf(client, mock_rag):
    with patch("app.api.routes.get_rag_service", return_value=mock_rag):
        response = client.post(
            "/api/v1/ingest",
            files={"file": ("report.txt", b"some text", "text/plain")},
        )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]
