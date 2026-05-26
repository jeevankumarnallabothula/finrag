"""
RAG Service — handles document ingestion, embedding, and retrieval.

Pipeline:
  PDF upload → text extraction → chunking → embedding → FAISS vector store
  Query       → embed query    → similarity search → rerank → LLM answer
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Prompt template ────────────────────────────────────────────────────────
FINANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a financial analyst assistant. Use ONLY the context below to answer
the question. If the answer is not in the context, say "I don't have enough information
in the provided documents to answer that."

Be precise, cite specific numbers/dates when available, and keep answers concise.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGService:
    """
    Manages the full RAG lifecycle:
    - Ingest: load PDF → chunk → embed → store in FAISS
    - Query:  embed question → retrieve top-K chunks → generate answer
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        self.vector_stores: dict[str, FAISS] = {}
        self._load_persisted_stores()

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_document(self, file_path: str, doc_id: Optional[str] = None) -> dict:
        """
        Load a PDF, chunk it, embed chunks, store in FAISS.
        Returns metadata about the ingested document.
        """
        doc_id = doc_id or str(uuid.uuid4())
        logger.info(f"Ingesting document: {file_path} → id={doc_id}")

        # 1. Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        logger.info(f"  Loaded {len(pages)} pages")

        # 2. Chunk
        chunks = self.splitter.split_documents(pages)
        logger.info(f"  Split into {len(chunks)} chunks")

        # 3. Embed + store
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_stores[doc_id] = vector_store

        # 4. Persist to disk
        store_path = Path(settings.VECTOR_STORE_PATH) / doc_id
        store_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(store_path))
        logger.info(f"  Saved vector store to {store_path}")

        return {
            "doc_id": doc_id,
            "pages": len(pages),
            "chunks": len(chunks),
            "file": Path(file_path).name,
        }

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, doc_id: str, question: str, chat_history: list = None) -> dict:
        """
        Retrieve relevant chunks and generate an answer using the LLM.
        Supports multi-turn conversation via chat_history.
        """
        if doc_id not in self.vector_stores:
            raise ValueError(f"Document '{doc_id}' not found. Please ingest it first.")

        retriever = self.vector_stores[doc_id].as_retriever(
            search_type="mmr",                       # Max Marginal Relevance — reduces redundancy
            search_kwargs={"k": settings.TOP_K_RESULTS, "fetch_k": 20},
        )

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5,  # keep last 5 turns
        )
        if chat_history:
            for human, ai in chat_history:
                memory.chat_memory.add_user_message(human)
                memory.chat_memory.add_ai_message(ai)

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": FINANCE_PROMPT},
            return_source_documents=True,
            verbose=False,
        )

        result = chain.invoke({"question": question})

        # Extract source page numbers for transparency
        sources = [
            {
                "page": doc.metadata.get("page", "N/A"),
                "snippet": doc.page_content[:200].strip(),
            }
            for doc in result.get("source_documents", [])
        ]

        return {
            "answer": result["answer"],
            "sources": sources,
            "doc_id": doc_id,
        }

    # ── Utilities ──────────────────────────────────────────────────────────

    def list_documents(self) -> list[str]:
        return list(self.vector_stores.keys())

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.vector_stores:
            return False
        del self.vector_stores[doc_id]
        import shutil
        store_path = Path(settings.VECTOR_STORE_PATH) / doc_id
        if store_path.exists():
            shutil.rmtree(store_path)
        return True

    def _load_persisted_stores(self):
        """Reload any previously saved vector stores on startup."""
        base = Path(settings.VECTOR_STORE_PATH)
        if not base.exists():
            return
        for folder in base.iterdir():
            if folder.is_dir():
                try:
                    self.vector_stores[folder.name] = FAISS.load_local(
                        str(folder),
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info(f"Reloaded vector store: {folder.name}")
                except Exception as e:
                    logger.warning(f"Could not reload {folder.name}: {e}")


# Singleton
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
