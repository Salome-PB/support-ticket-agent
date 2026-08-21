from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from config_loader import BASE_DIR, SETTINGS
from state import SupportState
from utils.helpers import add_trace

AUDIT_DB = BASE_DIR / SETTINGS["audit"]["sqlite_path"]


def init_audit_db() -> None:
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUDIT_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                customer_id TEXT,
                route TEXT NOT NULL,
                human_decision TEXT,
                final_status TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def audit_node(state: SupportState) -> dict:
    init_audit_db()

    ticket = state["ticket"]
    revision = state.get("human_revision_cycles", 0)
    decision = state.get("human_decision") or "none"
    status = state.get("final_status", "completed")

    event_id = (
        f"{ticket['ticket_id']}:{revision}:{decision}:{status}"
    )

    payload = {
        "ticket": ticket,
        "sentiment": state.get("sentiment"),
        "classification": state.get("classification"),
        "policy_result": state.get("policy_result"),
        "routing": state.get("routing"),
        "retrieval_query": state.get("retrieval_query"),
        "retrieved_documents": state.get("retrieved_documents"),
        "rag_draft": state.get("rag_draft"),
        "route": state.get("route"),
        "route_reason": state.get("route_reason"),
        "route_confidence": state.get("route_confidence"),
        "groundedness_score": state.get("groundedness_score"),
        "answer_confidence": state.get("answer_confidence"),
        "unsupported_claims": state.get("unsupported_claims"),
        "human_decision": state.get("human_decision"),
        "human_feedback": state.get("human_feedback"),
        "final_draft": state.get("final_draft"),
        "final_status": status,
        "trace": state.get("trace", []),
    }

    with sqlite3.connect(AUDIT_DB) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO audit_events
            (event_id, ticket_id, customer_id, route, human_decision, final_status, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                str(ticket["ticket_id"]),
                str(ticket.get("customer_id", "")),
                str(state["route"]),
                state.get("human_decision"),
                status,
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()

    return {
        "audit_event_id": event_id,
        "trace": add_trace(state, "audit", "write", event_id),
    }
