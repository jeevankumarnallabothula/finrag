"""
FinRAG - Financial Document Q&A API
Powered by LangChain, FastAPI, and OpenAI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print(f"🚀 FinRAG API starting up — {settings.APP_NAME} v{settings.VERSION}")
    yield
    print("🛑 FinRAG API shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "A Retrieval-Augmented Generation (RAG) API for querying financial documents. "
        "Upload PDFs (annual reports, earnings calls, SEC filings) and ask questions in natural language."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
