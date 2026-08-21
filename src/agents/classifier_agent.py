from __future__ import annotations

import json
from langchain_core.prompts import ChatPromptTemplate

from config_loader import PROMPTS, SETTINGS, get_llm
from schemas import ClassificationOutput
from state import SupportState
from tools.sentiment_tool import sentiment_classification_tool
from utils.constants import Priority
from utils.helpers import add_trace, ticket_text


def _priority_from_sentiment_and_keywords(
    sentiment_score: float,
    keyword_count: int,
) -> str:
    thresholds = SETTINGS["priority_thresholds"]

    if (
        sentiment_score <= float(thresholds["urgent"]["sentiment_score"])
        or keyword_count >= int(thresholds["urgent"]["escalation_keyword_count"])
    ):
        return Priority.URGENT

    if (
        sentiment_score <= float(thresholds["high"]["sentiment_score"])
        or keyword_count >= int(thresholds["high"]["escalation_keyword_count"])
    ):
        return Priority.HIGH

    if sentiment_score <= float(thresholds["medium"]["sentiment_score"]):
        return Priority.MEDIUM

    return Priority.LOW


def sentiment_and_classification_agent(state: SupportState) -> dict:
    ticket = state["ticket"]
    text = ticket_text(ticket)

    # Explicit LangChain tool invocation. No LLM is used for sentiment.
    sentiment = sentiment_classification_tool.invoke({"text": text})

    cfg = PROMPTS["classifier_agent"]
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", cfg["system"]),
                ("human", cfg["human"]),
            ]
        )
        | get_llm().with_structured_output(ClassificationOutput)
    )

    keywords = ticket.get("metadata", {}).get("escalation_keywords_found", [])
    result: ClassificationOutput = chain.invoke(
        {
            "ticket_id": ticket["ticket_id"],
            "ticket": text,
            "sentiment": json.dumps(sentiment),
            "keywords": json.dumps(keywords),
        }
    )

    priority = _priority_from_sentiment_and_keywords(
        sentiment_score=float(sentiment["compound"]),
        keyword_count=len(keywords),
    )

    classification = {
        "category": result.category,
        "key_issues": result.key_issues,
        "confidence": float(result.confidence),
        "priority": priority,
    }

    return {
        "sentiment": sentiment,
        "classification": classification,
        "trace": add_trace(
            state,
            "classifier_agent",
            "sentiment_and_classification",
            f"category={result.category}, priority={priority}, sentiment={sentiment['label']}",
        ),
    }
