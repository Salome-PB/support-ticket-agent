from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from config_loader import PROMPTS, get_llm
from rag.retriever import documents_to_context, retrieve_documents
from schemas import PolicyOutput
from state import SupportState
from utils.helpers import add_trace, ticket_text


def policy_agent(state: SupportState) -> dict:
    """
    Retrieve first, classify policy second.

    Exact policy evidence is copied from retrieved KB chunks; the LLM never
    supplies authoritative policy quote text.
    """
    ticket = state["ticket"]
    docs = retrieve_documents(ticket_text(ticket), k=6)

    cfg = PROMPTS["policy_agent"]
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", cfg["system"]),
                ("human", cfg["human"]),
            ]
        )
        | get_llm().with_structured_output(PolicyOutput)
    )

    result: PolicyOutput = chain.invoke(
        {
            "ticket": ticket_text(ticket),
            "context": documents_to_context(docs),
        }
    )

    valid_ids = [
        i
        for i in result.relevant_document_ids
        if isinstance(i, int) and 0 <= i < len(docs)
    ]

    policy_found = bool(result.policy_found and valid_ids)

    evidence = [
        {
            "source": docs[i]["source"],
            "chunk_id": docs[i]["chunk_id"],
            "quote": docs[i]["content"],  # exact KB chunk, not LLM-generated
        }
        for i in valid_ids
    ]

    refund_evidence = any(
        e["source"] == "refund_policy.md"
        for e in evidence
    )
    abusive_evidence = any(
        e["source"] == "abusive_content_policy.md"
        for e in evidence
    )

    data = result.model_dump()
    data["policy_found"] = policy_found
    data["relevant_document_ids"] = valid_ids
    data["exact_policy_evidence"] = evidence

    # Hard safety postconditions: unsupported refusal flags are cleared.
    data["refund_abuse"] = bool(
        policy_found and result.refund_abuse and refund_evidence
    )
    data["abusive_content_request"] = bool(
        policy_found and result.abusive_content_request and abusive_evidence
    )
    data["policy_violation"] = bool(
        policy_found and result.policy_violation
    )

    return {
        "policy_result": data,
        "trace": add_trace(
            state,
            "policy_agent",
            "policy_check",
            f"policy_found={policy_found}; reason={result.reason}",
        ),
    }
