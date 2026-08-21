from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.classifier_agent import sentiment_and_classification_agent
from agents.policy_agent import policy_agent
from agents.rag_agent import rag_agent
from agents.response_agent import (
    ask_more_info_response,
    auto_resolve_response,
    escalate_response,
    refuse_response,
)
from agents.router_agent import department_router_agent
from agents.triage_agent import triage_agent
from audit import audit_node, init_audit_db
from config_loader import PROMPTS, SETTINGS, get_llm
from evaluator import groundedness_evaluator
from rag.retriever import documents_to_context
from schemas import QueryRefinement, RevisedResponse
from state import SupportState
from utils.constants import TicketStatus
from utils.helpers import add_trace, enrich_ticket, ticket_text


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------

def ticket_in_node(state: SupportState) -> dict:
    ticket = enrich_ticket(state["ticket"])
    ticket_id = str(ticket["ticket_id"])

    return {
        "ticket": ticket,
        "messages": [
            HumanMessage(
                id=f"{ticket_id}:incoming",
                content=ticket_text(ticket),
                additional_kwargs={
                    "ticket_id": ticket_id,
                    "customer_id": str(ticket.get("customer_id", "")),
                },
            )
        ],
        "retrieval_query": ticket_text(ticket),
        "retrieval_attempts": 0,
        "human_revision_cycles": 0,
        "human_decision": None,
        "human_feedback": None,
        "trace": [],
        "final_status": TicketStatus.PROCESSING,
    }


# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def route_decision(state: SupportState) -> str:
    return state["route"]


def confidence_router(
    state: SupportState,
) -> Literal["hitl_approval", "refine_retrieval", "force_escalation"]:
    if state["route"] in ("refuse", "escalate"):
        return "hitl_approval"

    grounded = (
        state.get("groundedness_score", 0.0)
        >= float(SETTINGS["rag"]["groundedness_threshold"])
    )
    confident = (
        state.get("answer_confidence", 0.0)
        >= float(SETTINGS["rag"]["confidence_threshold"])
    )
    no_unsupported = not state.get("unsupported_claims")

    if grounded and confident and no_unsupported:
        return "hitl_approval"

    if (
        state.get("needs_more_retrieval", False)
        and state.get("retrieval_attempts", 0)
        < int(SETTINGS["rag"]["max_refinement_attempts"])
    ):
        return "refine_retrieval"

    return "force_escalation"


# ---------------------------------------------------------------------------
# Retrieval refinement loop
# ---------------------------------------------------------------------------

def refine_retrieval_node(state: SupportState) -> dict:
    cfg = PROMPTS["retrieval_refiner"]
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", cfg["system"]),
                ("human", cfg["human"]),
            ]
        )
        | get_llm().with_structured_output(QueryRefinement)
    )

    result: QueryRefinement = chain.invoke(
        {
            "ticket": ticket_text(state["ticket"]),
            "previous_query": state.get("retrieval_query", ""),
            "unsupported_claims": json.dumps(
                state.get("unsupported_claims", []),
                ensure_ascii=False,
            ),
        }
    )

    attempt = state.get("retrieval_attempts", 0) + 1

    return {
        "retrieval_query": result.refined_query,
        "retrieval_attempts": attempt,
        "trace": add_trace(
            state,
            "retrieval_refiner",
            "rewrite_query",
            f"attempt={attempt}; query={result.refined_query}",
        ),
    }


def force_escalation_node(state: SupportState) -> dict:
    return {
        "route": "escalate",
        "route_confidence": 1.0,
        "route_reason": (
            "The draft did not meet groundedness/confidence requirements after "
            "the allowed retrieval refinement attempts."
        ),
        "final_draft": (
            "Thanks for contacting support. I could not find sufficiently "
            "reliable information in the available knowledge base to resolve "
            "this request safely. I’m preparing it for human review.\n\n"
            "[Draft — pending human approval]"
        ),
        "groundedness_score": 1.0,
        "answer_confidence": 1.0,
        "unsupported_claims": [],
        "needs_more_retrieval": False,
        "trace": add_trace(
            state,
            "graph",
            "force_escalation",
            "grounding/confidence threshold not met",
        ),
    }


# ---------------------------------------------------------------------------
# Human approval/revision loop
# ---------------------------------------------------------------------------

def hitl_approval_node(
    state: SupportState,
) -> Command[Literal["audit_log", "revise_response", "groundedness_evaluator"]]:
    # Do not perform non-idempotent external side effects before interrupt().
    review_payload = {
        "ticket_id": state["ticket"]["ticket_id"],
        "customer_id": state["ticket"].get("customer_id"),
        "route": state["route"],
        "route_reason": state["route_reason"],
        "route_confidence": state["route_confidence"],
        "groundedness_score": state.get("groundedness_score"),
        "answer_confidence": state.get("answer_confidence"),
        "unsupported_claims": state.get("unsupported_claims", []),
        "policy_evidence": state.get("policy_result", {}).get(
            "exact_policy_evidence", []
        ),
        "draft": state["final_draft"],
        "allowed_decisions": ["approve", "edit", "reject"],
        "note": "This workflow records drafts only; it never sends them.",
    }

    human = interrupt(review_payload)

    if not isinstance(human, dict):
        raise ValueError("HITL resume payload must be a dictionary.")

    decision = str(human.get("decision", "")).strip().lower()

    if decision == "approve":
        status_map = {
            "auto_resolve": TicketStatus.APPROVED_NOT_SENT,
            "escalate": TicketStatus.ESCALATED_NOT_SENT,
            "refuse": TicketStatus.REFUSED_NOT_SENT,
            "ask_more_info": TicketStatus.ASK_MORE_INFO_NOT_SENT,
        }
        return Command(
            update={
                "human_decision": "approve",
                "human_feedback": None,
                "final_status": status_map[state["route"]],
                "trace": add_trace(
                    state, "human", "approve", "draft approved; not sent"
                ),
            },
            goto="audit_log",
        )

    revision_cycles = state.get("human_revision_cycles", 0) + 1
    max_cycles = int(SETTINGS["hitl"]["max_revision_cycles"])

    if revision_cycles > max_cycles:
        return Command(
            update={
                "route": "escalate",
                "human_decision": decision,
                "human_feedback": human.get("feedback", ""),
                "human_revision_cycles": revision_cycles,
                "final_status": TicketStatus.ESCALATED_NOT_SENT,
                "trace": add_trace(
                    state, "human", "review_limit", "maximum revision cycles reached"
                ),
            },
            goto="audit_log",
        )

    if decision == "edit":
        edited = str(human.get("edited_response", "")).strip()
        if not edited:
            raise ValueError("edited_response is required for decision='edit'.")

        return Command(
            update={
                "final_draft": edited,
                "human_decision": "edit",
                "human_feedback": human.get("feedback", ""),
                "human_revision_cycles": revision_cycles,
                "trace": add_trace(
                    state, "human", "edit", "human edit sent to grounding re-check"
                ),
            },
            goto="groundedness_evaluator",
        )

    if decision == "reject":
        return Command(
            update={
                "human_decision": "reject",
                "human_feedback": str(human.get("feedback", "")).strip(),
                "human_revision_cycles": revision_cycles,
                "trace": add_trace(
                    state,
                    "human",
                    "reject",
                    f"revision_cycle={revision_cycles}",
                ),
            },
            goto="revise_response",
        )

    raise ValueError("decision must be one of: approve, edit, reject")


def revise_response_node(state: SupportState) -> dict:
    cfg = PROMPTS["revision_agent"]
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", cfg["system"]),
                ("human", cfg["human"]),
            ]
        )
        | get_llm().with_structured_output(RevisedResponse)
    )

    result: RevisedResponse = chain.invoke(
        {
            "ticket": ticket_text(state["ticket"]),
            "draft": state["final_draft"],
            "feedback": state.get("human_feedback", ""),
            "context": documents_to_context(
                state.get("retrieved_documents", [])
            ),
        }
    )

    draft = result.response.strip()
    if "[draft" not in draft.lower():
        draft += "\n\n[Draft — pending human approval]"

    return {
        "final_draft": draft,
        "trace": add_trace(
            state, "revision_agent", "revise", "draft revised from human feedback"
        ),
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    init_audit_db()

    builder = StateGraph(SupportState)

    builder.add_node("ticket_in", ticket_in_node)
    builder.add_node("sentiment_and_classification", sentiment_and_classification_agent)
    builder.add_node("policy_check", policy_agent)
    builder.add_node("department_router", department_router_agent)
    builder.add_node("rag_answer_draft", rag_agent)
    builder.add_node("route_decision", triage_agent)

    builder.add_node("auto_resolve", auto_resolve_response)
    builder.add_node("escalate", escalate_response)
    builder.add_node("refuse", refuse_response)
    builder.add_node("ask_more_info", ask_more_info_response)

    builder.add_node("groundedness_evaluator", groundedness_evaluator)
    builder.add_node("refine_retrieval", refine_retrieval_node)
    builder.add_node("force_escalation", force_escalation_node)
    builder.add_node("hitl_approval", hitl_approval_node)
    builder.add_node("revise_response", revise_response_node)
    builder.add_node("audit_log", audit_node)

    builder.add_edge(START, "ticket_in")
    builder.add_edge("ticket_in", "sentiment_and_classification")
    builder.add_edge("sentiment_and_classification", "policy_check")
    builder.add_edge("policy_check", "department_router")
    builder.add_edge("department_router", "rag_answer_draft")
    builder.add_edge("rag_answer_draft", "route_decision")

    builder.add_conditional_edges(
        "route_decision",
        route_decision,
        {
            "auto_resolve": "auto_resolve",
            "escalate": "escalate",
            "refuse": "refuse",
            "ask_more_info": "ask_more_info",
        },
    )

    for response_node in (
        "auto_resolve",
        "escalate",
        "refuse",
        "ask_more_info",
    ):
        builder.add_edge(response_node, "groundedness_evaluator")

    builder.add_conditional_edges(
        "groundedness_evaluator",
        confidence_router,
        {
            "hitl_approval": "hitl_approval",
            "refine_retrieval": "refine_retrieval",
            "force_escalation": "force_escalation",
        },
    )

    # Retrieval refinement loop:
    # evaluator -> refine -> RAG -> route -> response -> evaluator
    builder.add_edge("refine_retrieval", "rag_answer_draft")

    builder.add_edge("force_escalation", "hitl_approval")

    # Human reject/edit loops back through grounding.
    builder.add_edge("revise_response", "groundedness_evaluator")

    builder.add_edge("audit_log", END)

    return builder.compile(
        checkpointer=checkpointer or InMemorySaver()
    )


graph = build_graph()
