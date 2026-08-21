from __future__ import annotations

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import HashingVectorizer


class LocalHashingEmbeddings(Embeddings):
    """
    Lightweight local embeddings for the small demo KB.

    HashingVectorizer is stateless, deterministic, fixed-dimensional and
    requires no model download or API token.
    """

    def __init__(self, n_features: int = 384):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        matrix = self.vectorizer.transform([text])
        return matrix.toarray()[0].astype(float).tolist()
