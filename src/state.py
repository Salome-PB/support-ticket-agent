from __future__ import annotations

from typing import Annotated, Any, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from schemas import Route


class SupportState(TypedDict, total=False):
    ticket: dict[str, Any]

    # Thread-scoped customer conversation memory.
    messages: Annotated[list[AnyMessage], add_messages]

    sentiment: dict[str, Any]
    classification: dict[str, Any]
    policy_result: dict[str, Any]
    routing: dict[str, Any]

    retrieval_query: str
    retrieved_documents: list[dict[str, Any]]
    rag_draft: str
    retrieval_attempts: int

    route: Route
    route_reason: str
    route_confidence: float
    final_draft: str

    groundedness_score: float
    answer_confidence: float
    unsupported_claims: list[str]
    needs_more_retrieval: bool

    human_decision: str | None
    human_feedback: str | None
    human_revision_cycles: int

    final_status: str
    audit_event_id: str
    trace: list[dict[str, Any]]
