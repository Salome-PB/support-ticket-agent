from __future__ import annotations

import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config_loader import BASE_DIR, SETTINGS
from rag.vectorstore import get_vectorstore
from pathlib import Path


#KB_DIR = BASE_DIR / "knowledge_base"
KB_DIR = Path("../../data/knowledge_base")



def load_kb_documents() -> list[Document]:
    docs: list[Document] = []

    for path in sorted(KB_DIR.glob("*.md")):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )

    if not docs:
        raise RuntimeError(f"No Markdown KB files found in {KB_DIR}")

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    cfg = SETTINGS["rag"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(cfg["chunk_size"]),
        chunk_overlap=int(cfg["chunk_overlap"]),
    )
    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def chunk_id(doc: Document) -> str:
    raw = (
        f"{doc.metadata.get('source')}::"
        f"{doc.metadata.get('chunk_id')}::"
        f"{doc.page_content}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ingest_knowledge_base() -> dict[str, int]:
    """
    Idempotently add new chunks. Existing deterministic IDs are not duplicated.
    """
    store = get_vectorstore()
    chunks = split_documents(load_kb_documents())

    existing_ids = set(store.get().get("ids", []))
    new_docs: list[Document] = []
    new_ids: list[str] = []

    for chunk in chunks:
        doc_id = chunk_id(chunk)
        if doc_id not in existing_ids:
            new_docs.append(chunk)
            new_ids.append(doc_id)

    if new_docs:
        store.add_documents(new_docs, ids=new_ids)

    return {
        "total_chunks": len(chunks),
        "new_chunks_added": len(new_docs),
    }


if __name__ == "__main__":
    print(ingest_knowledge_base())
