from __future__ import annotations

import json
from langchain_core.prompts import ChatPromptTemplate

from config_loader import PROMPTS, get_llm
from rag.retriever import documents_to_context, retrieve_documents
from state import SupportState
from utils.helpers import add_trace, ticket_text


def rag_agent(state: SupportState) -> dict:
    ticket = state["ticket"]
    query = state.get("retrieval_query") or ticket_text(ticket)
    docs = retrieve_documents(query)

    recent = [
        {
            "role": getattr(msg, "type", "unknown"),
            "content": str(msg.content),
        }
        for msg in state.get("messages", [])[-3:]
    ]

    cfg = PROMPTS["rag_agent"]
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", cfg["system"]),
                ("human", cfg["human"]),
            ]
        )
        | get_llm()
    )

    result = chain.invoke(
        {
            "ticket": ticket_text(ticket),
            "conversation": json.dumps(recent, ensure_ascii=False),
            "context": documents_to_context(docs),
        }
    )

    return {
        "retrieval_query": query,
        "retrieved_documents": docs,
        "rag_draft": result.content,
        "trace": add_trace(
            state,
            "rag_agent",
            "retrieve_and_draft",
            f"retrieved={len(docs)}",
        ),
    }
