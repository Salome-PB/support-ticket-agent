from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from utils.constants import ESCALATION_KEYWORDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ticket_text(ticket: dict[str, Any]) -> str:
    subject = ticket.get("subject", "")
    message = ticket.get("message", ticket.get("content", ""))
    return f"Subject: {subject}\nMessage: {message}".strip()


def find_escalation_keywords(text: str) -> list[str]:
    lower = text.lower()
    return [kw for kw in ESCALATION_KEYWORDS if kw in lower]


def enrich_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with metadata expected by the base-repo-style agents."""
    result = dict(ticket)
    text = ticket_text(result)

    result.setdefault(
        "timestamp",
        datetime.now(timezone.utc).isoformat(),
    )
    result.setdefault("priority", "medium")

    metadata = dict(result.get("metadata", {}))
    metadata.setdefault("word_count", len(text.split()))
    metadata.setdefault(
        "has_email",
        bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)),
    )
    metadata.setdefault(
        "has_url",
        bool(re.search(r"https?://\S+", text)),
    )
    metadata.setdefault(
        "escalation_keywords_found",
        find_escalation_keywords(text),
    )
    result["metadata"] = metadata
    return result


def add_trace(
    state: dict[str, Any],
    component: str,
    action: str,
    result: str,
) -> list[dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "action": action,
            "result": result,
        }
    )
    return trace

def normalize_llm_content(content) -> str:
    """
    Convert LangChain model content into plain text.

    Supports:
    - OpenAI-style string content
    - Gemini/LangChain content block lists
    - dict-based text blocks
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                # Common LangChain/Gemini text block
                text = block.get("text")

                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content).strip()


def log_agent_action(agent: str, action: str, details: dict[str, Any]) -> None:
    logger.info("[%s] %s: %s", agent, action, details)
