from __future__ import annotations

from typing import Any

from config_loader import SETTINGS
from rag.ingest import ingest_knowledge_base
from rag.vectorstore import get_vectorstore

_initialized = False


def ensure_kb_ingested() -> None:
    global _initialized
    if not _initialized:
        ingest_knowledge_base()
        _initialized = True


def retrieve_documents(query: str, k: int | None = None) -> list[dict[str, Any]]:
    ensure_kb_ingested()
    store = get_vectorstore()

    docs = store.similarity_search(
        query,
        k=int(k or SETTINGS["rag"]["top_k"]),
    )

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id"),
        }
        for doc in docs
    ]


def documents_to_context(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "(no KB evidence retrieved)"

    return "\n\n---\n\n".join(
        f"DOCUMENT_ID={i}\n"
        f"SOURCE={doc['source']}\n"
        f"CHUNK_ID={doc['chunk_id']}\n\n"
        f"{doc['content']}"
        for i, doc in enumerate(docs)
    )
