"""
Local validation that does not call the OpenAI API.

It verifies:
- all important files exist
- VADER tool runs
- KB ingestion into Chroma runs
- vector retrieval returns documents

Run after installing requirements:
    python validate_local.py
"""
from tools.sentiment_tool import sentiment_classification_tool
from rag.ingest import ingest_knowledge_base
from rag.retriever import retrieve_documents


def main():
    sentiment = sentiment_classification_tool.invoke(
        {"text": "I am very frustrated because my account is locked."}
    )
    assert sentiment["label"] == "negative", sentiment

    ingestion = ingest_knowledge_base()
    assert ingestion["total_chunks"] > 0, ingestion

    docs = retrieve_documents("expired password reset link", k=3)
    assert docs, "Retriever returned no documents."
    assert all("source" in d and "content" in d for d in docs)

    print("VADER:", sentiment)
    print("Ingestion:", ingestion)
    print("Retrieved sources:", [d["source"] for d in docs])
    print("LOCAL VALIDATION PASSED")


if __name__ == "__main__":
    main()
