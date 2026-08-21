from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from config_loader import PROMPTS, get_llm
from rag.retriever import documents_to_context
from schemas import GroundingOutput
from state import SupportState
from utils.helpers import add_trace


def groundedness_evaluator(state: SupportState) -> dict:
    route = state["route"]

    # Generic scripted refusal/escalation contains no specific fabricated policy claim.
    if route in ("refuse", "escalate"):
        return {
            "groundedness_score": 1.0,
            "answer_confidence": 1.0,
            "unsupported_claims": [],
            "needs_more_retrieval": False,
            "trace": add_trace(
                state, "groundedness_evaluator", "evaluate",
                f"safe scripted {route} response"
            ),
        }

    docs = state.get("retrieved_documents", [])
    if not docs:
        return {
            "groundedness_score": 0.0,
            "answer_confidence": 0.0,
            "unsupported_claims": ["No KB evidence was retrieved."],
            "needs_more_retrieval": True,
            "trace": add_trace(
                state, "groundedness_evaluator", "evaluate", "no evidence"
            ),
        }

    cfg = PROMPTS["groundedness_evaluator"]
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", cfg["system"]),
                ("human", cfg["human"]),
            ]
        )
        | get_llm().with_structured_output(GroundingOutput)
    )

    result: GroundingOutput = chain.invoke(
        {
            "route": route,
            "draft": state["final_draft"],
            "context": documents_to_context(docs),
        }
    )

    score = float(result.groundedness_score)
    unsupported = list(result.unsupported_claims)

    if unsupported:
        # Defensive postcondition: unsupported claims cannot pass a 0.78 threshold.
        score = min(score, 0.77)

    return {
        "groundedness_score": score,
        "answer_confidence": float(result.answer_confidence),
        "unsupported_claims": unsupported,
        "needs_more_retrieval": bool(result.needs_more_retrieval),
        "trace": add_trace(
            state,
            "groundedness_evaluator",
            "evaluate",
            (
                f"groundedness={score:.2f}; "
                f"confidence={result.answer_confidence:.2f}; "
                f"unsupported={len(unsupported)}"
            ),
        ),
    }
