from __future__ import annotations

import json
from langchain_core.prompts import ChatPromptTemplate

from config_loader import PROMPTS, get_llm
from schemas import TriageOutput
from state import SupportState
from utils.helpers import add_trace, ticket_text


def triage_agent(state: SupportState) -> dict:
    policy = state["policy_result"]

    # Required hard safety mapping.
    if policy.get("refund_abuse"):
        return {
            "route": "refuse",
            "route_confidence": 1.0,
            "route_reason": "Retrieved refund policy supports refusal for refund abuse.",
            "trace": add_trace(
                state, "triage_agent", "hard_route", "refuse: refund abuse"
            ),
        }

    if policy.get("abusive_content_request"):
        return {
            "route": "refuse",
            "route_confidence": 1.0,
            "route_reason": "Retrieved abusive-content policy supports refusal.",
            "trace": add_trace(
                state, "triage_agent", "hard_route", "refuse: abusive-content request"
            ),
        }

    # Required fail-closed behavior.
    if not policy.get("policy_found"):
        return {
            "route": "escalate",
            "route_confidence": 1.0,
            "route_reason": (
                "No relevant policy/FAQ evidence was established from the KB; "
                "escalating rather than fabricating."
            ),
            "trace": add_trace(
                state, "triage_agent", "hard_route", "escalate: no policy evidence"
            ),
        }

    cfg = PROMPTS["triage_agent"]
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", cfg["system"]),
                ("human", cfg["human"]),
            ]
        )
        | get_llm().with_structured_output(TriageOutput)
    )

    result: TriageOutput = chain.invoke(
        {
            "ticket": ticket_text(state["ticket"]),
            "classification": json.dumps(state["classification"], ensure_ascii=False),
            "sentiment": json.dumps(state["sentiment"], ensure_ascii=False),
            "policy": json.dumps(state["policy_result"], ensure_ascii=False),
            "rag_draft": state["rag_draft"],
        }
    )

    return {
        "route": result.action,
        "route_confidence": float(result.confidence),
        "route_reason": result.reason,
        "trace": add_trace(
            state,
            "triage_agent",
            "route",
            f"{result.action}: {result.confidence:.2f}",
        ),
    }
