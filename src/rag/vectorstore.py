from __future__ import annotations

from pathlib import Path
from langchain_chroma import Chroma

from config_loader import BASE_DIR, SETTINGS
from rag.embeddings import LocalHashingEmbeddings

RAG_CFG = SETTINGS["rag"]
CHROMA_DIR = BASE_DIR / RAG_CFG["persist_directory"]

embeddings = LocalHashingEmbeddings(
    n_features=int(RAG_CFG["embedding_dimensions"])
)


def get_vectorstore() -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=RAG_CFG["collection_name"],
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
