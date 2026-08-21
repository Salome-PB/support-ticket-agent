from __future__ import annotations

import json
from pathlib import Path

from langgraph.types import Command

from config_loader import BASE_DIR
from graph import graph


def load_ticket_queue() -> list[dict]:
    path = BASE_DIR / "data" / "synthetic_tickets.json"
    return json.loads(path.read_text(encoding="utf-8"))


def start_ticket(ticket: dict) -> tuple[dict, dict]:
    # Use customer_id to preserve a customer's conversation thread.
    # Use ticket_id instead if each ticket must be isolated.
    config = {
        "configurable": {
            "thread_id": str(ticket["customer_id"])
        }
    }
    result = graph.invoke({"ticket": ticket}, config=config)
    return result, config


def resume_ticket(result: dict, config: dict) -> dict:
    while result.get("__interrupt__"):
        review = result["__interrupt__"][0].value

        print("\nHUMAN REVIEW REQUIRED")
        print(json.dumps(review, indent=2, default=str))

        decision = input("\napprove / edit / reject: ").strip().lower()

        if decision == "approve":
            payload = {"decision": "approve"}

        elif decision == "edit":
            payload = {
                "decision": "edit",
                "edited_response": input("Edited draft: ").strip(),
                "feedback": input("Optional feedback: ").strip(),
            }

        elif decision == "reject":
            payload = {
                "decision": "reject",
                "feedback": input("Reviewer feedback: ").strip(),
            }

        else:
            print("Invalid decision.")
            continue

        # SAME thread_id/config is required to resume the interrupted run.
        result = graph.invoke(
            Command(resume=payload),
            config=config,
        )

    return result


def main() -> None:
    tickets = load_ticket_queue()

    for ticket in tickets:
        print("\n" + "=" * 80)
        print(f"PROCESSING {ticket['ticket_id']}: {ticket['subject']}")
        print("=" * 80)

        result, config = start_ticket(ticket)
        result = resume_ticket(result, config)

        print(
            json.dumps(
                {
                    "ticket_id": ticket["ticket_id"],
                    "route": result.get("route"),
                    "route_reason": result.get("route_reason"),
                    "final_status": result.get("final_status"),
                    "final_draft": result.get("final_draft"),
                    "audit_event_id": result.get("audit_event_id"),
                },
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
