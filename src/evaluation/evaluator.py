from langchain_core.prompts import ChatPromptTemplate

from config import llm
from schemas import GroundingEvaluation
from state import TicketState
from utils import (
    add_trace,
    documents_to_context,
)


EVALUATOR_SYSTEM_PROMPT = """
You are an independent groundedness evaluator for a customer-support
Retrieval-Augmented Generation system.

You are NOT the customer-support agent.

Your only job is to assess whether the proposed draft is supported by the
retrieved knowledge-base evidence.

Evaluate:

1. groundedness_score

The proportion and strength of substantive claims supported by retrieved
evidence.

2. answer_confidence

How confident the workflow should be that the draft is sufficiently
supported and complete to proceed to human review.

This does NOT mean the answer may be automatically sent.

3. unsupported_claims

List substantive factual or policy statements that are not supported by
the supplied evidence.

Do not flag conversational language such as:
"Thanks for contacting us."

4. needs_more_retrieval

Set true only when another retrieval attempt against the same KB could
reasonably find evidence that would improve the answer.

STRICT RULES:

- Evaluate only against supplied KB evidence.
- Do not use outside knowledge.
- Do not rewrite the answer.
- Do not answer the ticket yourself.
- Do not invent policy.
- Do not expose chain-of-thought.
- Return only the structured evaluation.
"""


EVALUATOR_USER_PROMPT = """
Evaluate the following response.

ROUTE
-----
{route}

CUSTOMER RESPONSE DRAFT
-----------------------
{draft}

RETRIEVED KNOWLEDGE BASE
------------------------
{knowledge_base}

Return:

- groundedness_score
- answer_confidence
- unsupported_claims
- needs_more_retrieval
"""


evaluation_prompt = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                EVALUATOR_SYSTEM_PROMPT,
            ),
            (
                "human",
                EVALUATOR_USER_PROMPT,
            ),
        ]
    )
)


evaluation_chain = (
    evaluation_prompt
    | llm.with_structured_output(
        GroundingEvaluation
    )
)
evaluation = evaluation_chain.invoke(
    {
        "route":
            state["route"],

        "draft":
            state["final_draft"],

        "knowledge_base":
            documents_to_context(
                state["retrieved_documents"]
            ),
    }
)