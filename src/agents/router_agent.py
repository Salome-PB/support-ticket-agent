from __future__ import annotations

from config_loader import SETTINGS
from state import SupportState
from utils.constants import Department, Priority
from utils.helpers import add_trace


def department_router_agent(state: SupportState) -> dict:
    """
    Deterministic department routing, retaining the base repository's concept.
    """
    category = state["classification"]["category"]
    priority = state["classification"]["priority"]

    possible = SETTINGS["department_mappings"].get(
        category,
        [Department.CUSTOMER_SUCCESS],
    )

    if (
        priority in (Priority.URGENT, Priority.HIGH)
        and Department.ESCALATION_TEAM in possible
    ):
        primary = Department.ESCALATION_TEAM
    else:
        primary = possible[0]

    routing = {
        "primary_department": primary,
        "backup_departments": possible[1:] if len(possible) > 1 else [],
    }

    return {
        "routing": routing,
        "trace": add_trace(
            state,
            "router_agent",
            "department_route",
            f"department={primary}",
        ),
    }
