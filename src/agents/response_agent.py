from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from config_loader import PROMPTS, get_llm
from rag.retriever import documents_to_context
from state import SupportState
from utils.helpers import add_trace, ticket_text


REFUSAL_SCRIPT = (
    "Thanks for contacting support. I’m unable to assist with that request "
    "because it conflicts with the applicable support policy. If you believe "
    "the request has been misunderstood, a support specialist can review the "
    "case.\n\n[Draft — pending human approval]"
)

ESCALATION_SCRIPT = (
    "Thanks for contacting support. The available knowledge base does not "
    "provide enough verified information to safely resolve this request "
    "automatically. I’m preparing it for human review.\n\n"
    "[Draft — pending human approval]"
)


# def auto_resolve_response(state: SupportState) -> dict:
#     draft = state["rag_draft"].strip()
#     if "[draft" not in draft.lower():
#         draft += "\n\n[Draft — pending human approval]"
#
#     return {
#         "final_draft": draft,
#         "trace": add_trace(
#             state, "response_agent", "auto_resolve_draft", "draft prepared"
#         ),
#    }

from utils.helpers import (
    add_trace,
    ticket_text,
    normalize_llm_content,
)


def auto_resolve_response(
    state: SupportState,
) -> dict:

    draft = normalize_llm_content(
        state.get("rag_draft")
    )

    if not draft:
        draft = (
            "I could not generate a grounded response "
            "from the available knowledge base."
        )

    if "[draft" not in draft.lower():
        draft += (
            "\n\n"
            "[Draft — pending human approval]"
        )

    return {
        "final_draft": draft,
        "trace": add_trace(
            state,
            "response_agent",
            "auto_resolve_draft",
            "draft prepared",
        ),
    }


def escalate_response(state: SupportState) -> dict:
    return {
        "final_draft": ESCALATION_SCRIPT,
        "trace": add_trace(
            state, "response_agent", "escalation_draft", "safe escalation draft prepared"
        ),
    }


def refuse_response(state: SupportState) -> dict:
    # Intentionally generic: no LLM-generated policy quote.
    return {
        "final_draft": REFUSAL_SCRIPT,
        "trace": add_trace(
            state, "response_agent", "refusal_draft", "scripted refusal prepared"
        ),
    }


def ask_more_info_response(state: SupportState) -> dict:
    cfg = PROMPTS["ask_more_info_response"]
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", cfg["system"]),
                ("human", cfg["human"]),
            ]
        )
        | get_llm()
    )

    # result = chain.invoke(
    #     {
    #         "ticket": ticket_text(state["ticket"]),
    #         "context": documents_to_context(state.get("retrieved_documents", [])),
    #     }
    # )
    result = chain.invoke(
        {
            "ticket": ticket_text(
                state["ticket"]
            ),
            "context": documents_to_context(
                state.get(
                    "retrieved_documents",
                    [],
                )
            ),
        }
    )

    draft = normalize_llm_content(
        result.content
    )

    if "[draft" not in draft.lower():
        draft += (
            "\n\n"
            "[Draft — pending human approval]"
        )

    #draft = result.content.strip()
    draft = normalize_llm_content(result.content)
    if "[draft" not in draft.lower():
        draft += "\n\n[Draft — pending human approval]"

    return {
        "final_draft": draft,
        "trace": add_trace(
            state, "response_agent", "ask_more_info_draft", "draft prepared"
        ),
    }
